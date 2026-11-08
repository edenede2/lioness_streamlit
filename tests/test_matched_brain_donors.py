from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import pearsonr,spearmanr,false_discovery_control
from streamlit.testing.v1 import AppTest
from ui_interactions import apply_driver
from app_helpers import matched_brain_donors as D
from app_helpers.camel import FrozenRINT,feature_values
from app_helpers.matched_brain_donor_ui import _subtitles


def sample(**updates):
    args=dict(cohort="Held-out validation",components=("TS_DLPFC",),feature="connectivity",rule="all",normalization="standard_pruned",scale="raw")
    args.update(updates)
    return D.load_scores(13,**args)


def test_approved_bundle_counts_hashes_and_privacy():
    m=D.manifest()
    assert D.available() and m["sample_count"]==1243 and m["module_count"]==247
    assert m["expected_rows"]==dict(scores=5614804,eigengenes=621121,transformations=106704)
    total={k:0 for k in m["expected_rows"]}
    forbidden={"donor","projid","pair_id","validation_pair_id","anchor_pair_id","anchor_ad","anchor_control","ad","control"}
    for e in m["files"]:
        p=D.ROOT/e["path"];pf=pq.ParquetFile(p)
        assert not forbidden&set(pf.schema_arrow.names)
        assert pf.metadata.num_rows==e["rows"] and p.stat().st_size==e["bytes"]<95*1024**2
        assert hashlib.sha256(p.read_bytes()).hexdigest()==e["sha256"]
        if e["kind"] in total:total[e["kind"]]+=e["rows"]
    assert total==m["expected_rows"]
    meta=D.metadata()
    assert len(meta)==meta.sample_id.nunique()==1243
    assert set(meta)==set(m["metadata_columns"])


@pytest.mark.parametrize("feature",["connectivity","positive_density","negative_density","abs_sum","positive_abs_sum","negative_abs_sum"])
@pytest.mark.parametrize("normalization",["standard_pruned","retained_edge"])
def test_feature_formulas_and_frozen_scales(feature,normalization):
    raw=sample(feature=feature,normalization=normalization,rule="either")
    np.testing.assert_allclose(raw.metric_value,feature_values(raw,feature,normalization),equal_nan=True)
    mapping=D._load("transformations",13,(("component","=","TS_DLPFC"),("metric_family","=",feature),
        ("score_normalization","=",normalization),("differential_edge_rule","=","either"))).iloc[0]
    asinh=sample(feature=feature,normalization=normalization,rule="either",scale="asinh")
    rint=sample(feature=feature,normalization=normalization,rule="either",scale="rint")
    np.testing.assert_allclose(asinh.metric_value,np.arcsinh(raw.metric_value/mapping.asinh_scale),equal_nan=True)
    np.testing.assert_allclose(rint.metric_value,FrozenRINT(np.array(mapping.rint_x),np.array(mapping.rint_z)).transform(raw.metric_value.to_numpy()),equal_nan=True)
    assert mapping.fit_cohort=="Discovery"


def test_frozen_source_values_are_copied_without_changes():
    private=Path('/media/psylab-6028/DATA/Eden/inter-tissue-CoExpression/out/brain_discovery_validation/20261028_matched80_control_l4')
    if not private.exists():pytest.skip("Private source comparison runs only on analysis host")
    for kind in ("scores","eigengenes","transformations"):
        for module in (13,1225):
            public=D.ROOT/kind/f'M{module}.parquet'
            if public.exists():assert public.read_bytes()==(private/kind/public.name).read_bytes()


def test_cohort_filters_no_imputed_components_and_eigengenes():
    with pytest.raises(ValueError,match="pooled"):sample(cohort=D.BALANCED,components=("CT",))
    with pytest.raises(ValueError,match="per tissue"):sample(feature="eigengene",components=("CT_AC__DLPFC",))
    with pytest.raises(ValueError):sample(scale="not_a_scale")
    with pytest.raises(ValueError):D.load_scores(999999,cohort="Discovery",components=("TS_AC",))
    for cohort in (D.BALANCED,D.PROFILE):
        f=sample(cohort=cohort)
        assert not f.empty and f.diagnosis_group.value_counts().nunique()==1
        assert not f.sample_id.isin(D.metadata().loc[lambda x:x.analysis_role.eq("Discovery"),"sample_id"]).any()
        assert not f.duplicated(["sample_id","component"]).any()
    f=sample(feature="eigengene",normalization="unfiltered",scale="rint")
    np.testing.assert_allclose(f.metric_value,f.metric_rint,equal_nan=True)


def test_numeric_statistics_are_direct_and_never_adjust_across_displayed_groups():
    f=sample()
    stats=D.numeric_statistics(f,"cogn_global","diagnosis_group",["AD","Control"],10)
    assert not any("fdr" in c for c in stats)
    for row in stats.itertuples():
        g=f if row.grouping_level=="__pooled__" else f.loc[f.diagnosis_group.eq(row.grouping_level)]
        g=g.dropna(subset=["metric_value","cogn_global"])
        assert row.n==len(g)
        np.testing.assert_allclose([row.pearson_r,row.pearson_p],pearsonr(g.metric_value,g.cogn_global))
        np.testing.assert_allclose([row.spearman_rho,row.spearman_p],spearmanr(g.metric_value,g.cogn_global))
    with pytest.raises(ValueError,match="Nominal"):D.numeric_statistics(f,"clusters","diagnosis_group",["AD","Control"])


def test_full_family_bh_keeps_groups_and_components_separate():
    stats=D.family_statistics(cohort="Held-out validation",components=("TS_DLPFC",),feature="connectivity",rule="all",
        normalization="standard_pruned",scale="raw",diagnoses=("AD","Control"),grouping="diagnosis_group",
        levels=("AD","Control"),outcome="cogn_global",min_n=10)
    assert stats.module.nunique()==247
    for _,g in stats.groupby(["component","grouping_level"]):
        for method in ("pearson","spearman"):
            valid=g[method+"_p"].notna()
            np.testing.assert_allclose(g.loc[valid,method+"_fdr_across_modules"],false_discovery_control(g.loc[valid,method+"_p"]))
            assert g[method+"_fdr_module_family_n"].eq(valid.sum()).all()
    assert not any("displayed_family" in c for c in stats)


def widget(app,kind,label):return next(w for w in getattr(app,kind) if w.label==label)


def run_view():
    from app_helpers.matched_brain_donor_ui import render_view
    from app_helpers.figure_export import render_exportable_plot
    render_view(render_exportable_plot,13)


def run_distributions():
    from app_helpers.matched_brain_donor_ui import render_view
    from app_helpers.figure_export import render_exportable_plot
    render_view(render_exportable_plot,13,distributions=True)


def test_scatter_group_controls_eigengenes_and_nominal_outcomes():
    app=apply_driver(AppTest.from_function(run_view,default_timeout=120)).run()
    assert not app.exception and app.get("plotly_chart") and app.get("download_button")
    app=widget(app,"checkbox","Show group-specific trend lines").set_value(False).run()
    app=widget(app,"checkbox","Solid black pooled line").set_value(True).run()
    app=widget(app,"selectbox","Group donor correlations by").set_value("study").run()
    assert not app.exception
    app=widget(app,"selectbox","Donor module feature").set_value("eigengene").run()
    assert not app.exception
    app=widget(app,"selectbox","Donor outcome").set_value("clusters").run()
    assert not app.exception and app.get("plotly_chart")
    assert not any("correlation statistic" in s.label for s in app.radio)


def test_distribution_views_and_incomplete_cohort_render():
    app=apply_driver(AppTest.from_function(run_distributions,default_timeout=120)).run()
    assert not app.exception and app.get("plotly_chart")
    for v in ["ECDF","Violin","Histogram","Raincloud"]:
        app=widget(app,"selectbox","Donor distribution view").set_value(v).run()
        assert not app.exception and app.get("plotly_chart")
    app=widget(app,"selectbox","Donor cohort").set_value(D.PROFILE).run()
    app=widget(app,"selectbox","Divide distributions by").set_value("study").run()
    assert not app.exception and app.get("download_button")


def test_scope_matched_kegg_subtitles():
    subtitles=_subtitles(13,["TS_DLPFC","CT_AC__DLPFC","CT"],"either")
    assert "regional support" in subtitles["CT_AC__DLPFC"]
    assert "tissue-expanded" in subtitles["CT"]
    assert "Frozen" in subtitles["TS_DLPFC"]


def test_donor_layout_has_explicit_nonoverlapping_title_band():
    from app_helpers.charts import distribution_figure
    from app_helpers.matched_brain_donor_ui import donor_layout
    f=sample(components=("TS","CT"))
    fig=donor_layout(distribution_figure(f,"Connectivity","Raw",["AD","Control"],13,"Raincloud"))
    assert fig.layout.title.y==.99 and fig.layout.title.yanchor=="top"
    assert fig.layout.meta["figure_export"]["fixed_container_title"]
    assert fig.layout.legend.y>1 and fig.layout.margin.t>100
