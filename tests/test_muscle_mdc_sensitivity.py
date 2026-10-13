from __future__ import annotations

import numpy as np
import pandas as pd

from app_helpers.mdc_sensitivity import VARIANTS, public_mdc_tables
from scripts import run_muscle_m5_mdc as M
from scripts import run_muscle_m5_mdc_significance as S
from scripts import run_muscle_mdc_sensitivity as R
from app_helpers import charts, data as D


def small_network():
    rng = np.random.default_rng(9)
    frames = {t: pd.DataFrame(rng.normal(size=(2, 20)), index=[f"{t}A", f"{t}B"])
              for t in M.TISSUES}
    rows = pd.DataFrame({"Cluster ID": [7] * 6,
        "Tissue": ["QUAD", "VH", "SMA", "VH", "QUAD", "SMA"],
        "Gene ID": ["QUADA", "VHA", "SMAA", "VHB", "QUADB", "SMAB"]})
    nulls = {f"{a}-{b}": {"mu_null": .05, "sigma_null": .03} for a,b in M.BLOCKS if a!=b}
    return rows,frames,nulls


def test_packed_high_m5_matches_direct_original_with_mixed_tissue_order():
    rows,frames,nulls = small_network()
    part = R.block_parts(rows, frames, "Control", nulls)
    direct,labels,ids = M._raw_module_m5(7, rows, frames, "Control", nulls)
    i,j=np.triu_indices(6,1)
    np.testing.assert_array_equal(part["labels"],labels)
    np.testing.assert_array_equal(part["genes"],ids)
    np.testing.assert_array_equal(part["values"]["high_m5"],np.abs(direct[i,j]))
    for code,(left,right) in enumerate(M.BLOCKS):
        mask=part['codes']==code
        assert np.all((labels[i[mask]]==left)&(labels[j[mask]]==right)
                      |(labels[i[mask]]==right)&(labels[j[mask]]==left))


def test_shared_scales_ignore_ad_divisors_and_clipping_is_explicit():
    part={"codes":np.array([0,3]),"values":{f:np.array([4.,80.],np.float32) for f in R.FAMILIES}}
    control={(f,c):2. for f in R.FAMILIES for c in [0,3]}
    ad={(f,c):20. for f in R.FAMILIES for c in [0,3]}
    np.testing.assert_array_equal(R.variant_weights(part,'m5_high_shared_unclipped',ad,control),[2.,40.])
    np.testing.assert_array_equal(R.variant_weights(part,'m5_high_shared_clipped',ad,control),[2.,10.])
    np.testing.assert_allclose(R.variant_weights(part,'original',ad,control),[.2,4.],rtol=1e-7)
    np.testing.assert_array_equal(R.variant_weights(part,'pearson',ad,control),[4.,80.])


def test_scope_sums_recombine_and_common_scale_does_not_change_pair_ratio():
    rows,frames,nulls=small_network()
    cp,cq=R.group_parts({7:rows},frames,'Control',nulls)
    ap,aq=R.group_parts({7:rows},frames,'Control',nulls)
    for family in R.FAMILIES:
        ap[7]['values'][family]*=2
    aq={k:v*2 for k,v in aq.items()}
    data=R.summarize_parts(ap,cp,aq,cq)
    for variant,group in data.groupby('normalization_variant'):
        lookup=group.set_index('scope')
        for diagnosis in ['ad','control']:
            sums=lookup['mean_abs_'+diagnosis]*lookup.n_edges
            assert np.isclose(sums['total'],sums['CT']+sums['TS'],rtol=1e-6)
            assert np.isclose(sums['CT'],sums['CT_VH__SMA']+sums['CT_VH__QUAD']+sums['CT_SMA__QUAD'],rtol=1e-6)
        if not VARIANTS[variant]['clip']:
            expected=1 if VARIANTS[variant]['scale']=='separate' else 2
            np.testing.assert_allclose(lookup.mdc_raw.dropna(),expected,rtol=1e-6)
            assert lookup.loc[lookup.mdc_raw.isna(),'mean_abs_control'].le(1e-12).all()


def test_pending_public_rows_have_no_fdr_and_completed_rows_preserve_p_values():
    rows,frames,nulls=small_network()
    cp,cq=R.group_parts({7:rows},frames,'Control',nulls)
    data=R.summarize_parts(cp,cp,cq,cq)
    source=data[data.normalization_variant.eq('pearson')]
    details=pd.DataFrame({'module':[7],'module_size':[6]})
    summary,resolved=public_mdc_tables(source,details)
    assert summary.directional_fdr_ct.isna().all()
    assert resolved.directional_fdr.isna().all()
    assert not resolved.inference_available.any()
    source=source.assign(directional_fdr=.02,directional_p_sample=.005,
                         directional_p_gene=.01,inference_available=True,inference_unavailable_reason='')
    summary,resolved=public_mdc_tables(source,details)
    np.testing.assert_allclose(resolved.directional_p_sample,.005)
    np.testing.assert_allclose(summary.directional_fdr_ct,.02)


def test_array_inference_rejects_wrong_null_family_shapes():
    observed=pd.DataFrame({'module':[1]})
    try:
        S.build_significance_from_arrays(observed,np.array([1]),np.zeros((1,9,4)),np.zeros((1,9,3)),3,3)
    except ValueError as error:
        assert 'Sample-null dimensions' in str(error)
    else:
        raise AssertionError('A mismatched null family was accepted')


def test_predicate_reader_selects_exact_normalization_family_and_keeps_baseline():
    baseline=D.load_mdc_summary('muscle_control_l4')
    alternative=D.load_mdc_summary('muscle_control_l4',normalization_variant='m5_low_shared_unclipped')
    resolved=D.load_mdc_resolved('muscle_control_l4',normalization_variant='m5_low_shared_unclipped')
    assert len(baseline)==len(alternative)==155
    assert len(resolved)==930
    assert set(alternative.normalization_variant)=={'m5_low_shared_unclipped'}
    assert set(resolved.normalization_variant)=={'m5_low_shared_unclipped'}
    assert not np.allclose(baseline.mdc_ct,alternative.mdc_ct,equal_nan=True)
    assert baseline.directional_fdr_ct.notna().any()
    try:
        D.load_mdc_summary('control_derived',normalization_variant='pearson')
    except ValueError:
        pass
    else:
        raise AssertionError('Muscle sensitivity leaked into brain module routing')


def test_comparison_chart_marks_selected_module_for_every_available_variant():
    frame=D.load_mdc_normalization_summary('muscle_control_l4')
    figure=charts.mdc_normalization_comparison_figure(frame,'ct',36)
    assert len(figure.data)==len(VARIANTS)+1
    selected=figure.data[-1]
    assert len(selected.x)==len(VARIANTS)
    source=frame[frame.module.eq(36)].set_index('normalization_variant')
    np.testing.assert_allclose(selected.customdata,[source.loc[v,'mdc_ct'] for v in VARIANTS])


def test_small_positive_mdc_values_are_not_rounded_to_zero_in_raw_heatmaps():
    frame=pd.DataFrame({'module':[1],'component_label':['CT: VH - SMA'],
                        'mdc':[.000037],'log2_mdc':[np.log2(.000037)],'directional_fdr':[.03]})
    figure=charts.mdc_resolved_heatmap_figure(frame,.05,1,scale='raw')
    assert '3.7e-05' in figure.data[0].text[0][0]
    assert '0.00' not in figure.data[0].text[0][0]
