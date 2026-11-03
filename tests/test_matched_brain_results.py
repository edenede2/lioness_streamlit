from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import false_discovery_control
from streamlit.testing.v1 import AppTest
from app_helpers import matched_brain_results as D
from app_helpers.matched_brain_results_ui import SECTIONS,effect_figure,mdc_figure


def context():
    return D.camel(next(c for c in D.COHORTS if c.startswith("Discovery-demography")),"TS_DLPFC",
        test_type="covariate_adjusted_association",metric_family="connectivity",
        score_normalization="standard_pruned",differential_edge_rule="all",scale="raw",
        phenotype="cogn_global",correlation_method="spearman")


def test_complete_catalog_hashes_counts_and_aggregate_privacy():
    manifest=D.manifest()
    assert manifest["full_module_catalog"] and manifest["camel_rows"]==30794712
    assert manifest["module_count"]==247
    assert len(D.entries())==27
    forbidden={"donor","sample_id","projid","pair_id","validation_pair_id"}
    for item in manifest["files"]:
        p=D.ROOT/item["path"];pf=pq.ParquetFile(p)
        assert pf.metadata.num_rows==item["rows"]
        assert p.stat().st_size==item["bytes"]<95*1024**2
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
        assert not forbidden&set(pf.schema_arrow.names)
    assert len(D.table("mdc_labels"))==47424


def test_narrow_load_and_module_only_fdr_survive_display_selection():
    f=context();valid=f.p_permutation.notna()
    assert 100<len(f)<=247
    np.testing.assert_allclose(f.loc[valid,"q_permutation"],
        false_discovery_control(f.loc[valid,"p_permutation"]),atol=1e-13)
    assert f.q_permutation_family_n.eq(valid.sum()).all()
    top=D.selected_modules(f,k=5)
    assert len(top)==5
    for row in top.itertuples():
        assert row.q_permutation==f.set_index("module").loc[row.module,"q_permutation"]
    assert D.selected_modules(f,k=10000).shape[0]==len(f)
    assert D.camel(next(c for c in D.COHORTS if c.startswith("Discovery-demography")),"CT",
        test_type="covariate_adjusted_association",metric_family="connectivity",score_normalization="standard_pruned",
        differential_edge_rule="all",scale="raw",phenotype="cogn_global",correlation_method="spearman").empty


def test_unbounded_reads_and_path_traversal_rejected():
    with pytest.raises(ValueError,match="full narrow"): D.camel("Held-out validation","TS_DLPFC")
    with pytest.raises(ValueError): D._path("../private_metadata.parquet")
    with pytest.raises(ValueError): D._path("/tmp/private.parquet")


def test_raster_export_title_uses_final_pixel_rounded_height(monkeypatch):
    import io
    from PIL import Image
    from app_helpers.figure_export import PlotExportSpec,ExportConfig,PlotContentConfig
    from app_helpers.figure_export import adapters
    config=ExportConfig(width=180,height=140,dpi=300)
    observed={}
    def renderer(fig,fmt,width,height,scale):
        observed["y"]=fig.layout.title.y
        observed["expected"]=max(.5,1-float(fig.layout.title.font.size)/height)
        image=Image.new("RGB",config.pixel_dimensions,"white")
        stream=io.BytesIO();image.save(stream,format="PNG");return stream.getvalue()
    monkeypatch.setattr(adapters,"static_image",renderer)
    adapters.render_figure(PlotExportSpec.from_figure("matched_qa",effect_figure(context())),config,PlotContentConfig())
    assert observed["y"]==observed["expected"]


def test_monotonic_raw_asinh_spearman_values_and_permutation_fdr_agree():
    raw=context()
    other=D.camel(next(c for c in D.COHORTS if c.startswith("Discovery-demography")),"TS_DLPFC",
        test_type="covariate_adjusted_association",metric_family="connectivity",
        score_normalization="standard_pruned",differential_edge_rule="all",scale="asinh",
        phenotype="cogn_global",correlation_method="spearman")
    pair=raw.merge(other,on="module",suffixes=("_raw","_asinh"),validate="one_to_one")
    for c in ("effect","p_permutation","q_permutation","n"):
        np.testing.assert_allclose(pair[c+"_raw"],pair[c+"_asinh"],equal_nan=True,atol=1e-12)


def test_charts_keep_signed_effect_and_raw_mdc_sequential_colors():
    f=context();fig=effect_figure(f,k=8)
    assert fig is not None and len(fig.data[0].x)==8
    assert list(fig.data[0].x)==D.selected_modules(f,k=8).effect.tolist()
    assert fig.data[0].cliponaxis is False and fig.data[0].showlegend is False
    assert fig.layout.xaxis.range[1]>max(fig.data[0].x)
    assert "Global cognition" in fig.layout.title.text
    from app_helpers.figure_export import PlotExportSpec,ExportConfig,PlotContentConfig
    from app_helpers.figure_export.adapters import build_artifact
    exported=build_artifact(PlotExportSpec.from_figure("matched_qa",fig),ExportConfig(width=180,height=140,dpi=300),PlotContentConfig()).figure
    assert exported.layout.title.automargin is False
    m=D.table("mdc_labels")
    m=m.loc[m.component.eq("TS_DLPFC")&m.cohort_scope.eq(next(c for c in D.COHORTS if c.startswith("Discovery-demography")))&
        m.differential_edge_rule.eq("all")&m.test_type.eq("matched_pair_label")].copy()
    m["display_fdr"]=m.q_two_sided
    heatmap=mdc_figure(m,"Raw MDC",k=10)
    assert heatmap is not None and heatmap.data[0].zmid is None
    assert np.nanmin(heatmap.data[0].z)>=0
    assert heatmap.data[0].colorscale[-1][1]!="rgb(5,48,97)"


def run_view():
    from app_helpers.matched_brain_results_ui import render_view
    from app_helpers.figure_export import render_exportable_plot
    render_view(render_exportable_plot)


def widget(app,kind,label):
    return next(w for w in getattr(app,kind) if w.label==label)


def test_every_new_section_renders_and_has_downloads():
    app=AppTest.from_function(run_view,default_timeout=120).run()
    assert not app.exception
    assert app.get("plotly_chart")
    assert app.get("download_button")
    for section in SECTIONS:
        app=widget(app,"radio","Matched-control results section").set_value(section).run()
        assert not app.exception,section
        assert app.get("download_button"),section


def test_new_view_is_reachable_from_real_app_navigation():
    entrypoint=Path(__file__).resolve().parents[1]/"streamlit_app.py"
    app=AppTest.from_file(entrypoint,default_timeout=180)
    app.query_params["view"]="Matched-control results"
    app.run()
    assert not app.exception
    assert any(h.value=="Matched-control results" for h in app.header)
    assert widget(app,"radio","Matched-control results section").value=="CAMEL associations"


def test_cohort_comparison_eigengene_filter_and_empty_evidence_states():
    app=AppTest.from_function(run_view,default_timeout=120).run()
    cohorts=[next(c for c in D.COHORTS if c.startswith("Discovery-demography")),"Held-out validation"]
    app=widget(app,"multiselect","Comparison cohorts").set_value(cohorts).run()
    assert not app.exception
    app=widget(app,"selectbox","Module score feature").set_value("eigengene").run()
    assert not app.exception
    app=widget(app,"radio","Correlation method").set_value("pearson").run()
    assert not app.exception
    app=widget(app,"selectbox","Frozen discovery edge set").set_value("ad_higher").run()
    assert not app.exception
    assert any("Eigengenes have no edge mask" in str(x.value) for x in app.info)
    app=widget(app,"selectbox","Display FDR filter").set_value(.05).run()
    assert not app.exception


def test_raw_and_legacy_mdc_controls_render_without_changing_catalog():
    app=AppTest.from_function(run_view,default_timeout=120).run()
    app=widget(app,"radio","Matched-control results section").set_value("MDC").run()
    app=widget(app,"radio","MDC scale").set_value("Raw MDC").run()
    assert not app.exception and app.get("plotly_chart")
    app=widget(app,"selectbox","MDC significance test").set_value("legacy").run()
    assert not app.exception
    app=widget(app,"multiselect","Comparison cohorts").set_value(["Discovery","Held-out validation"]).run()
    assert not app.exception and app.get("plotly_chart")
    app=widget(app,"checkbox","Compare MDC with discovery").set_value(True).run()
    assert not app.exception
