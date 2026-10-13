from __future__ import annotations

import numpy as np
import pandas as pd
from app_helpers.mdc_validation import add_contrasts, permute_within_strata, proportional_subset, summarize_validation


def fixture():
    rows=[]
    for v in ["pearson", "m5_high_no_q90"]:
        for m in [1,2]:
            for s,z in [("CT",-2.),("TS",-.5)]:
                rows.append(dict(normalization_variant=v,module=m,scope=s,statistic=z,
                    mdc_raw=2**z,mean_abs_ad=.1,mean_abs_control=.2,n_edges=50,
                    test_eligible=True,unavailable_reason=""))
    return add_contrasts(pd.DataFrame(rows))


def test_stratified_labels_preserve_roles_profiles_and_counts():
    labels=np.array(["AD","Control","AD","Control","AD","AD"])
    strata=np.array(["A|0","A|0","A|1","A|1","B|0","B|0"])
    a=permute_within_strata(labels,strata,42)
    np.testing.assert_array_equal(a,permute_within_strata(labels,strata,42))
    for s in set(strata):
        assert sorted(labels[strata==s])==sorted(a[strata==s])
    # Donor indices—not entries of expression matrices—are assigned to groups.
    expression=np.arange(24).reshape(4,6)
    np.testing.assert_array_equal(np.sort(np.r_[np.flatnonzero(a=="AD"),np.flatnonzero(a=="Control")]),np.arange(6))
    np.testing.assert_array_equal(expression[:,a=="AD"],expression[:,np.flatnonzero(a=="AD")])


def test_balanced_sampling_is_exact_without_replacement_and_stratified():
    strata=np.array(["a"]*7+["b"]*3)
    out=proportional_subset(np.arange(10),strata,6,42)
    assert len(out)==len(set(out))==6
    assert dict(zip(*np.unique(strata[out],return_counts=True)))=={"a":4,"b":2}
    np.testing.assert_array_equal(out,proportional_subset(np.arange(10),strata,6,42))


def test_donor_selection_preserves_values_and_original_float32_memory_layout():
    from scripts import run_muscle_mdc_label_validation as runner
    frame=pd.DataFrame(np.arange(40,dtype=np.float32).reshape(5,8))
    runner.STATE['pooled']={'VH':frame}
    out=runner.selected_frames(np.array([0,2,5]))['VH']
    np.testing.assert_array_equal(out.to_numpy(),frame.iloc[:,[0,2,5]].to_numpy())
    assert out.to_numpy().flags.c_contiguous
    runner.STATE.clear()


def test_contrast_is_paired_log_ratio_difference_and_missing_stays_missing():
    f=fixture()
    np.testing.assert_allclose(f.loc[f.scope=="CT_minus_TS","statistic"],-1.5)
    source=f[f.scope!="CT_minus_TS"].copy()
    source.loc[(source.module==1)&(source.scope=="CT"),"test_eligible"]=False
    out=add_contrasts(source)
    assert out.loc[(out.module==1)&(out.scope=="CT_minus_TS"),"statistic"].isna().all()


def test_two_sided_label_p_uses_empirical_null_not_mdc_one_and_separate_bh():
    observed=fixture()
    keys=list(zip(observed.normalization_variant,observed.module,observed.scope))
    reps=[{key:float(-1+(rep/10)) for key in keys} for rep in range(20)]
    out=summarize_validation(observed,reps,reps,20,20)
    # CT is below every null: plus-one two-sided p=2/21, irrespective of MDC=1.
    np.testing.assert_allclose(out.loc[out.scope=="CT","label_p"],2/21)
    np.testing.assert_allclose(out.loc[out.scope=="CT","label_fdr"],2/21)
    assert out.tested_module_count.eq(2).all()  # not 2 weightings × 2 modules × 3 scopes
    assert out.label_test_complete.all() and out.balanced_complete.all()


def test_incomplete_and_constant_nulls_do_not_create_fdr_or_zero_values():
    f=fixture();keys=list(zip(f.normalization_variant,f.module,f.scope))
    reps=[{k:float(r) for k in keys} for r in range(5)]
    incomplete=summarize_validation(f,reps,[],20,100)
    assert incomplete.label_fdr.isna().all()
    assert incomplete.balanced_median.isna().all()
    constant=summarize_validation(f,[{k:0. for k in keys}]*20,[],20,100)
    assert constant.label_p.isna().all()
    assert not constant.test_eligible.any()


def test_contrast_fdr_and_noneligible_rows_have_independent_family_sizes():
    f=fixture();f.loc[(f.module==2)&(f.scope=="CT"),"test_eligible"]=False
    keys=list(zip(f.normalization_variant,f.module,f.scope))
    reps=[{k:float(r/10) for k in keys} for r in range(20)]
    out=summarize_validation(f,reps,[],20,100)
    assert out.loc[out.scope=="CT","tested_module_count"].eq(1).all()
    assert out.loc[out.scope=="TS","tested_module_count"].eq(2).all()
    assert out.loc[(out.module==2)&(out.scope=="CT"),"label_fdr"].isna().all()


def test_public_observed_reproduction_and_privacy():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'data/muscle_control_l4'
    new=pd.read_parquet(root/'mdc_label_validation/results.parquet')
    assert len(new)==3100 and new.module.nunique()==155
    assert not {'donor','projid','gene_id','role_hash'}.intersection(new.columns)
    old=pd.read_parquet(root/'mdc_sensitivity/resolved.parquet')
    old=old[old.normalization_variant.isin(['pearson','m5_high_no_q90'])]
    matched=new.merge(old[['module','component','normalization_variant','mdc']],left_on=['module','scope','normalization_variant'],right_on=['module','component','normalization_variant'])
    np.testing.assert_allclose(matched.mdc_raw,matched.mdc,equal_nan=True,rtol=1e-7,atol=1e-12)


def test_validation_charts_keep_labels_directions_and_descriptive_intervals():
    from app_helpers.mdc_validation_ui import heatmap_figure,null_figure,balanced_figure
    f=fixture()
    keys=list(zip(f.normalization_variant,f.module,f.scope))
    reps=[{k:float(r/10) for k in keys} for r in range(20)]
    data=summarize_validation(f,reps,reps,20,20)
    # Fill other scopes absent from this fixture through the real chart reindex.
    h=heatmap_figure(data[data.normalization_variant=='pearson'],1)
    assert h.data[0].zmid==0
    n=null_figure(pd.DataFrame({'statistic':[-3.,-2.,-1.]}),-2.5,'CT_minus_TS')
    assert n.layout.shapes[0].x0==-2.5
    b=balanced_figure(data[data.normalization_variant=='pearson'],'CT')
    assert 'not confidence intervals' in b.layout.title.text
    assert len(b.data)==2
