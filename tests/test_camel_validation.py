"""Offline readback tests: never download private source data or contact Drive."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr,spearmanr
from app_helpers import camel_validation as V, data as D
from app_helpers.camel import VALIDATION_SCOPE
from app_helpers import streaming_associations as S
from app_helpers.correlations import add_across_module_fdr

APP=Path(__file__).resolve().parents[1]
STAGE=APP.parents[1]/"out/camel_validation/20261021_frozen164_incomplete_regions/public_bundle"


@pytest.fixture
def staged(monkeypatch):
    if not (STAGE/"modules/M121.parquet").exists(): pytest.skip("Private validation run not present; no Drive read")
    monkeypatch.setenv("CAMEL_VALIDATION_DATA_ROOT",str(STAGE))
    return STAGE


def test_metadata_is_disjoint_scored_component_cohorts_are_not_filled(staged):
    meta=D.load_sample_metadata("control_derived",VALIDATION_SCOPE)
    assert len(meta)==793 and meta.sample_id.is_unique
    assert not {"donor","projid"}&set(meta)
    old=D.load_sample_metadata()
    assert not set(old.sample_id)&set(meta.sample_id)
    expected={"TS_AC":266,"TS_DLPFC":752,"TS_PCGBA23":195,
        "CT_AC__DLPFC":230,"CT_AC__PCGBA23":14,"CT_DLPFC__PCGBA23":176}
    values=V.scores(121,"connectivity",scale="raw")
    assert values.groupby("component").size().to_dict()==expected
    assert values.diagnosis_group.isna().sum()==0
    assert not {"CT","TS","total"}&set(values.component)


def test_narrow_reader_uses_frozen_maps_and_preserves_pruned_identities(staged,monkeypatch):
    monkeypatch.setattr(V,"available",lambda rule="all":True)
    full=D.load_resolved("control_anchored",121,"connectivity",module_set="control_derived",cohort_scope=VALIDATION_SCOPE)
    narrow=D.load_resolved_scope("control_anchored",121,"connectivity",module_set="control_derived",cohort_scope=VALIDATION_SCOPE,
        component="TS_DLPFC",metric_scale="raw",include_embedded_metadata=False)
    assert len(narrow)==752 and "metric_rint" not in narrow and "diagnosis_group" not in narrow
    comparison=full.loc[full.component.eq("TS_DLPFC")].set_index("sample_id")
    np.testing.assert_array_equal(narrow.metric_raw,comparison.loc[narrow.sample_id,"metric_raw"])
    for component,group in full.groupby("component"):
        anchor=pd.read_parquet(staged/"rint/M121.parquet",filters=[("component","=",component),("differential_edge_rule","=","all"),
            ("score_normalization","=","standard_pruned"),("metric_family","=","connectivity")]).iloc[0]
        np.testing.assert_allclose(group.metric_asinh,np.arcsinh(group.metric_raw/anchor.asinh_scale),equal_nan=True)
        assert group.metric_rint.min()>=min(anchor.z) and group.metric_rint.max()<=max(anchor.z)
    summary=V.summaries(121)
    np.testing.assert_allclose(summary.absolute_weight_sum,summary.positive_weight_sum+summary.negative_weight_magnitude,equal_nan=True)
    assert (summary.n_positive_edges+summary.n_negative_edges+summary.n_zero_edges==summary.n_retained_edges).all()


def test_validation_rejects_other_definitions_fdr_and_eigengenes(staged,monkeypatch):
    monkeypatch.setattr(V,"available",lambda rule="all":True)
    for kwargs in ({"module_set":"full_cohort"},{"metric_family":"eigengene"},
                   {"differential_edge_rule":"ad_control_discovery_fdr05"}):
        args=dict(method="control_anchored",module=121,metric_family="connectivity",module_set="control_derived",cohort_scope=VALIDATION_SCOPE)
        args.update(kwargs)
        with pytest.raises(ValueError): D.load_resolved(**args)
    with pytest.raises(ValueError,match="one module at a time"): V.scores(None)


def test_validation_grouped_associations_match_direct_and_separate_fdr_families(staged,monkeypatch):
    monkeypatch.setattr(V,"available",lambda rule="all":True)
    result=S.stream_grouped_correlations((121,771),V.metadata(),module_set="control_derived",estimator="lioness",
        method="control_anchored",resolved=True,feature="connectivity",phenotype="cogn_global",scale="raw",
        components=("TS_DLPFC",),diagnoses=("AD","Control","MCI","Unclassified"),grouping_variable="study",
        grouping_levels=("ROS","MAP"),include_pooled=True,min_group_n=10,edge_rule="all",differential_edge_rule="all",
        differential_fdr_scope="per_module",differential_fdr_threshold=.05,score_normalization="standard_pruned",
        analysis_subset="all_donors",cohort_scope=VALIDATION_SCOPE)
    result=add_across_module_fdr(result,family_columns=["metric_family","component","outcome","grouping_variable","grouping_level"])
    assert result.spearman_fdr_module_family_n.le(2).all()
    assert set(result.grouping_level)=={"ROS","MAP","__pooled__"}
    data=V.scores(121,"connectivity",component="TS_DLPFC",scale="raw").dropna(subset=["metric_raw","cogn_global"])
    data=data.loc[data.study.eq("ROS")]
    row=result.loc[result.module.eq(121)&result.grouping_level.eq("ROS")].iloc[0]
    assert row.pearson_r==pytest.approx(pearsonr(data.metric_raw,data.cogn_global).statistic)
    assert row.spearman_rho==pytest.approx(spearmanr(data.metric_raw,data.cogn_global).statistic)


def test_small_ac_pcg_mdc_is_descriptive_and_structural_absence_is_missing(staged):
    f=V.mdc()
    if f.empty or "CT_AC__PCGBA23" not in set(f.component): pytest.skip("MDC stage still running")
    block=f.loc[f.component.eq("CT_AC__PCGBA23")]
    assert block.n_ad.eq(3).all() and block.n_control.eq(6).all()
    assert not block.inferential.any()
    assert block.directional_fdr.isna().all() and block.p_loss_sample.isna().all()
