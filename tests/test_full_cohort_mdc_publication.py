"""Source-backed release checks for both incomplete-region brain MDC methods."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]
RUN = APP.parents[1] / "out/partition_validation/20261021_frozen_references_legacy_conventions/full_cohort"
METHODS = ("standard", "control_anchored")


@pytest.mark.parametrize("method", METHODS)
def test_public_mdc_is_complete_private_free_and_indexed(method):
    root = APP / "data/validation/full_cohort" / method
    if not (root / "mdc/release_qa.json").is_file():
        pytest.skip("Validated release has not been staged; never download in tests")
    report = json.loads((root / "mdc/release_qa.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert report["status"] == "passed"
    assert report["backend_verification"]["status"] == "passed"
    assert report["validation_donors"] == 793
    assert report["excluded_complete_donors"] == 464
    assert manifest["backend_verified"] and manifest["mdc_status"] == "complete"
    assert manifest["module_set"] == "full_cohort" and manifest["network_method"] == method
    frame = pd.read_parquet(root / "mdc/resolved.parquet")
    assert len(frame) == 154 * 6 * 31
    assert not frame.duplicated(["module", "component", "differential_edge_rule"]).any()
    assert frame.module.nunique() == 154 and 1918 in set(frame.module)
    assert not {"sample_id", "donor", "projid"} & set(frame.columns)
    sparse = frame.loc[frame.component.eq("CT_AC__PCGBA23")]
    assert not sparse.inferential.any() and sparse.directional_fdr.isna().all()
    index = json.loads((APP / "drive_file_index.json").read_text())["files"]
    prefix = f"validation/full_cohort/{method}/"
    assert prefix + "manifest.json" in index
    assert prefix + "mdc/resolved.parquet" in index
    assert sum(k.startswith(prefix + "mdc/adjacency/") for k in index) == 154 * 6
    assert (root / "mdc/resolved.parquet").stat().st_size < 95 * 1024**2


@pytest.mark.parametrize("method", METHODS)
def test_full_cohort_mdc_charts_masks_and_triangle_render(method, monkeypatch):
    from app_helpers import camel_validation as V
    bundle = RUN / method / "public_bundle"
    if not (bundle / "mdc/resolved.parquet").is_file():
        pytest.skip("Private numerical results not available; never download in tests")
    monkeypatch.setattr(V, "root", lambda module_set="full_cohort", method=method: bundle)
    app = AppTest.from_string(f'''
import streamlit as st
from app_helpers.camel_validation_ui import render_mdc
from app_helpers.figure_export import render_exportable_plot
rule = st.selectbox("Test edge set", ["all", "ad_control_discovery_effect__hedges_g__fixed_0.40__ad_higher", "ad_control_discovery_effect__hedges_g__fixed_0.40__control_higher", "ad_control_discovery_effect__hedges_g__fixed_0.40__either"])
render_mdc(1918, rule, render_exportable_plot, module_set="full_cohort", network_method={method!r})
''', default_timeout=90).run()
    assert not app.exception and app.get("plotly_chart")
    for rule in next(e for e in app.selectbox if e.label == "Test edge set").options:
        next(e for e in app.selectbox if e.label == "Test edge set").set_value(rule)
        app.run()
        assert not app.exception
        displayed = next(e.value for e in app.dataframe if "directional_fdr" in e.value)
        assert len(displayed) == 154 * 6 and displayed.differential_edge_rule.eq(rule).all()
    next(e for e in app.selectbox if e.label == "Test edge set").set_value("all")
    app.run()
    next(e for e in app.selectbox if e.label == "Validation adjacency component").set_value("TS_DLPFC")
    next(e for e in app.checkbox if e.label == "Show validation AD/Control triangle heatmap").check()
    app.run()
    assert not app.exception and len(app.get("plotly_chart")) == 2
    next(e for e in app.radio if e.label == "Validation MDC view").set_value("All-module component heatmap")
    next(e for e in app.radio if e.label == "Validation MDC scale").set_value("log2 MDC")
    app.run()
    assert not app.exception and len(app.get("plotly_chart")) == 2


def test_mdc_installer_cannot_replace_donor_data(tmp_path, monkeypatch):
    from scripts import prepare_full_cohort_mdc_release as release
    monkeypatch.setattr(release, "APP", tmp_path / "app")
    root = tmp_path / "private"
    for relative in ("mdc/resolved.parquet", "review/mdc_comparison.parquet"):
        path = root / "public_bundle" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"aggregate fixture")
    target = release.APP / "data/validation/full_cohort/standard"
    target.mkdir(parents=True)
    (target / "sample_metadata.parquet").write_bytes(b"unchanged metadata")
    report = {"status": "passed", "backend_verification": {"status": "passed"}}
    release.install(root, "standard", report)
    assert (target / "sample_metadata.parquet").read_bytes() == b"unchanged metadata"
    assert json.loads((target / "mdc/release_qa.json").read_text()) == report
    release.install(root, "standard", report)
    assert len(list((root / "rollback").glob("*/mdc/resolved.parquet"))) == 1
