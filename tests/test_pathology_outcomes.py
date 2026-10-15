from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from app_helpers import data as D
from app_helpers import regional_modules as R
from app_helpers import score_comparison as C
from app_helpers.pathology import (
    PATHOLOGY_OUTCOMES, component_statistics, statistics_from_scores, adjust_module_fdr,
)

APP = Path(__file__).resolve().parents[1]


def synthetic():
    rng = np.random.default_rng(42)
    parts = []
    for module in (1,2,3):
        for diagnosis in ("Control", "AD"):
            ct, ts, y = rng.normal(size=(3, 40))
            frame = pd.DataFrame(dict(module=module, metric_family="connectivity",
                diagnosis_group=diagnosis, lioness_method="control_anchored",
                CT_raw=ct, TS_raw=ts, CT_asinh=np.arcsinh(ct), TS_asinh=np.arcsinh(ts),
                CT_rint=ct, TS_rint=ts, gpath=y+.5*ct, amyloid=y, tangles=y-.3*ts))
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_robustness_matches_direct_statistics_and_missing_pair_counts():
    frame=synthetic().iloc[:40]
    y=frame.gpath.to_numpy().copy()
    y[0]=np.nan
    result=component_statistics(frame.CT_raw, frame.CT_asinh, frame.CT_rint,y)
    valid=np.isfinite(y)
    assert result['n'] == 39
    assert result['r_raw'] == pytest.approx(stats.pearsonr(frame.CT_raw[valid], y[valid]).statistic)
    direct=stats.spearmanr(frame.CT_raw[valid],y[valid])
    assert result['rho'] == pytest.approx(direct.statistic)
    assert result['p_spearman'] == pytest.approx(direct.pvalue)
    full=stats.pearsonr(frame.CT_raw[valid],y[valid]).statistic
    x=frame.CT_raw.to_numpy()[valid]; target=y[valid]
    loo=[stats.pearsonr(np.delete(x,i),np.delete(target,i)).statistic for i in range(len(x))]
    assert result['loo_rint_max_delta'] == pytest.approx(np.max(np.abs(np.array(loo)-full)))
    unavailable=component_statistics(np.zeros(40),np.zeros(40),np.zeros(40),y)
    assert np.isnan(unavailable['rho']) and unavailable['unavailable_reason']


def test_module_only_fdr_and_dependent_test_use_identical_donors():
    frame=synthetic()
    result=adjust_module_fdr(statistics_from_scores(frame,resolved=False),resolved=False)
    for _, group in result.groupby(['phenotype','diagnosis_group']):
        assert set(group.q_spearman_CT_across_modules_n) == {3}
        assert np.allclose(group.q_spearman_CT_across_modules,
            stats.false_discovery_control(group.p_spearman_CT))
    assert result.filter(like='all12_global').isna().all().all()
    group=frame.iloc[:40]
    row=result.iloc[0]
    ct,ts,y=group.CT_raw.to_numpy(),group.TS_raw.to_numpy(),group.gpath.to_numpy()
    a,b,c=stats.pearsonr(ct,y).statistic,stats.pearsonr(ts,y).statistic,stats.pearsonr(ct,ts).statistic
    determinant=1-a*a-b*b-c*c+2*a*b*c
    t=(a-b)*np.sqrt(39*(1+c))/np.sqrt(2*determinant*39/37+(a+b)**2/4*(1-c)**3)
    assert row.component_t_rint == pytest.approx(t)
    assert row.p_component_rint == pytest.approx(2*stats.t.sf(abs(t),37))
    frame.loc[frame.module.eq(1),'CT_raw']=np.nan
    absent=statistics_from_scores(frame,resolved=False)
    assert absent.loc[absent.module.eq(1),'n_CT'].eq(0).all()
    assert absent.loc[absent.module.eq(1),'n_TS'].eq(40).all()
    assert absent.loc[absent.module.eq(1),'p_component_rint'].isna().all()


def test_source_coverage_metadata_preservation_and_prediction_exclusion():
    manifest=json.loads((APP/'data/pathology/manifest.json').read_text())
    assert manifest['prediction_changed'] is False
    for module_set, counts in [('full_cohort',(450,449,448)),('muscle_control_l4',(354,357,357)),('muscle_all_donors_l4',(354,357,357))]:
        metadata=D.load_sample_metadata(module_set)
        assert tuple(metadata[list(PATHOLOGY_OUTCOMES)].notna().sum()) == counts
        assert not {'donor','projid'} & set(metadata)
    assert not set(PATHOLOGY_OUTCOMES)&set(D.PREDICTION_OUTCOME_LABELS)
    assert set(PATHOLOGY_OUTCOMES)<=set(D.NUMERIC_OUTCOMES)
    assert set(PATHOLOGY_OUTCOMES)<=set(D.COLOR_LABELS)
    assert set(PATHOLOGY_OUTCOMES)<=set(D.HOVER_LABELS)
    assert not set(PATHOLOGY_OUTCOMES)&set(D.ASSOCIATION_GROUP_LABELS)


def widget(elements,label):
    return next(v for v in elements if v.label==label)


@pytest.mark.parametrize('view',['Associations','Statistics','CT–TS screen','Module finder','Donor edge explorer','Feature distributions','Correlation heatmaps'])
def test_numeric_views_accept_pathology(view):
    app=AppTest.from_file(APP/'streamlit_app.py',default_timeout=180).run()
    app=widget(app.pills,'Analysis view').set_value(view).run()
    app=widget(app.selectbox,'Association outcome').set_value('amyloid').run()
    assert not app.exception
    assert widget(app.selectbox,'Association outcome').value == 'amyloid'


@pytest.mark.parametrize('source,tissue,cohort',[
    ('single_region_complete_case_l3','AC','common_450'),
    ('single_region_full_tissue_l3','PCG','maximum_tissue'),
    (R.MOTOR_SOURCE,'VH','common_357'),
])
def test_regional_and_comparison_pathology(source,tissue,cohort):
    module=int(R.load_regional_details().loc[lambda f:f.partition_source.eq(source)&f.tissue.eq(tissue),'module'].iloc[0])
    scores=R.load_regional_scores(source,cohort,'eigengene','not_applicable',tissue,module)
    assert set(PATHOLOGY_OUTCOMES)<=set(scores)
    spec=C.ComparisonSeriesSpec(source,feature='eigengene',method='not_applicable',component='regional',cohort_scope=cohort,tissues=(tissue,))
    result=C.calculate_series(spec,'tangles','all_donors',10,'pathology-v1')
    assert not result.empty and result.eligible.any()


def test_saved_pathology_catalog_matches_scores_and_excludes_constant_tests():
    result=D.load_aggregate_statistics('control_anchored',phenotype='gpath',metric_family='connectivity',module_set='control_derived')
    assert len(result)==186*3
    row=result.loc[result.diagnosis_group.eq('AD') & result.rho_CT.notna()].iloc[0]
    scores=D.load_aggregate('control_anchored',int(row.module),'connectivity',module_set='control_derived')
    metadata=D.load_sample_metadata()
    scores=scores.merge(metadata[['sample_id','gpath']],on='sample_id').loc[lambda f:f.diagnosis_group.eq('AD')]
    good=scores[['CT_raw','gpath']].dropna()
    direct=stats.spearmanr(good.CT_raw,good.gpath)
    assert row.rho_CT==pytest.approx(direct.statistic)
    assert row.p_spearman_CT==pytest.approx(direct.pvalue)
    ad=result.loc[result.diagnosis_group.eq('AD')]
    good=ad.p_spearman_CT.notna()
    assert np.allclose(ad.loc[good,'q_spearman_CT_across_modules'],stats.false_discovery_control(ad.loc[good,'p_spearman_CT']))


@pytest.mark.parametrize('rule',[
    'ad_control_discovery_fdr05',
    'ad_control_discovery_effect__hedges_g__fixed_0.40__ad_higher',
])
def test_filtered_pathology_statistics_have_full_module_families(rule):
    result=D.load_aggregate_statistics('control_anchored',phenotype='tangles',metric_family='connectivity',
        module_set='control_derived',differential_edge_rule=rule,differential_fdr_scope='per_module',
        differential_fdr_threshold=.10,analysis_subset='validation_ad_control')
    assert set(result.diagnosis_group)=={'AD','Control'}
    assert len(result)==186*2
    assert result.n_TS.max()<=50
    for _,group in result.groupby('diagnosis_group'):
        good=group.p_spearman_TS.notna()
        if good.any():
            np.testing.assert_allclose(group.loc[good,'q_spearman_TS_across_modules'],
                stats.false_discovery_control(group.loc[good,'p_spearman_TS']))


def test_motor_pooled_module_finder_and_grouped_pathology_associations_render():
    app=AppTest.from_file(APP/'streamlit_app.py',default_timeout=180).run()
    app=widget(app.selectbox,'Module definition').set_value('muscle_control_l4').run()
    app=widget(app.selectbox,'Association outcome').set_value('gpath').run()
    app=widget(app.selectbox,'Group correlations by').set_value('clusters').run()
    assert not app.exception
    app=widget(app.pills,'Analysis view').set_value('Module finder').run()
    assert not app.exception
    assert app.get('plotly_chart')
