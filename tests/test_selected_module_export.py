"""Selection overlays are optional presentation, never removed observations."""

from dataclasses import replace
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app_helpers import charts
from app_helpers.figure_export import PlotContentConfig, PlotExportSpec, preset_config
from app_helpers.figure_export import adapters
from app_helpers.figure_export.config import render_fingerprint
from app_helpers.figure_export.config import restore_figure


@pytest.fixture
def module_frame():
    frame = pd.DataFrame({
        "module": [935, 1918, 356],
        "module_size": [120, 542, 62],
        "cluster_type": ["CT", "CT", "TS"],
        "n_tissues": [3, 3, 1],
        "tissue_entropy": [1.5, 1.3, 0.0],
        "tissue_entropy_normalized": [0.95, 0.82, 0.0],
        "mdc_ts": [0.8, 1.3, 1.1],
        "mdc_ct": [1.4, 0.6, np.nan],
        "mdc_total": [1.1, 0.9, 1.1],
        "directional_fdr_total": [0.2, 0.3, 0.4],
        "directional_fdr_ts": [0.02, 0.3, 0.4],
        "directional_fdr_ct": [0.3, 0.01, np.nan],
        "direction_ts": ["Higher in Control", "Higher in AD", "Higher in AD"],
        "direction_ct": ["Higher in AD", "Higher in Control", "Not available"],
        "normalization_variant": ["pearson"] * 3,
        "n_ts_edges": [20, 40, 25],
        "display_rank": [1, 2, 3],
        "finder_score": [0.9, 0.7, 0.1],
        "ct_ts_abs_delta_correlation": [0.5, 0.4, 0.1],
        "ad_control_max_abs_delta_correlation": [0.4, 0.3, 0.1],
        "pathway_id": ["test"] * 3,
        "pathway_label": ["Test pathway"] * 3,
        "component_label": ["AC–DLPFC"] * 3,
        "mdc": [1.4, 0.6, 1.1],
        "directional_fdr": [0.3, 0.01, 0.4],
        "direction": ["Higher in AD", "Higher in Control", "Higher in AD"],
        "enrichment_fdr": [0.01, 0.02, 0.03],
        "enrichment_scope": ["tissue-expanded"] * 3,
        "n_edges": [20, 40, 25],
    })
    for suffix in ("ts", "ct"):
        frame[f"log2_mdc_{suffix}"] = np.log2(frame[f"mdc_{suffix}"])
    frame["log2_mdc"] = np.log2(frame["mdc"])
    return frame


@pytest.mark.parametrize("chart_kind", [
    "module_entropy", "ts_entropy_raw", "ts_entropy_log2", "ct_entropy_raw",
    "ct_entropy_log2", "overview_raw", "overview_log2", "normalization",
    "module_finder", "pathway",
])
def test_all_selection_overlays_can_be_hidden_without_altering_data(module_frame, chart_kind):
    if chart_kind == "module_entropy":
        source = charts.module_entropy_figure(module_frame, selected_module=935)
    elif "entropy_" in chart_kind:
        scope, _, scale = chart_kind.split("_")
        source = charts.mdc_entropy_figure(
            module_frame, scope=scope, selected_module=935, threshold=0.05, scale=scale
        )
    elif chart_kind.startswith("overview_"):
        source = charts.mdc_overview_figure(
            module_frame, selected_module=935, threshold=0.05,
            scale=chart_kind.split("_")[1],
        )
    elif chart_kind == "normalization":
        source = charts.mdc_normalization_comparison_figure(module_frame, "ts", 935)
    elif chart_kind == "module_finder":
        source = charts.module_finder_figure(
            module_frame, selected_module=935, phenotype_label="Outcome",
            feature_label="Connectivity", correlation_method="Spearman",
            ct_ts_diagnosis="All donors", criterion_label="Both",
        )
    else:
        source = charts.pathway_mdc_detail_figure(
            module_frame, pathway_id="test", selected_module=935, threshold=0.05
        )
    original = source.to_json()
    config = preset_config("Grant figure")
    shown = adapters.build_figure(original, None, config, PlotContentConfig())
    hidden = adapters.build_figure(
        original, None, config, PlotContentConfig(show_selected_module_annotation=False)
    )
    overlays = [t for t in shown.figure.data if adapters.trace_role(t) == "selected_module"]
    assert overlays and all(t.visible is not False for t in overlays)
    for before, after in zip(shown.figure.data, hidden.figure.data, strict=True):
        if adapters.trace_role(before) == "selected_module":
            assert before.meta["export_role"] == "selected_module"
            assert after.visible is False and after.showlegend is False
        else:
            assert before.to_plotly_json() == after.to_plotly_json()
    for before, after in zip(restore_figure(original).data, hidden.figure.data, strict=True):
        np.testing.assert_array_equal(before.x, after.x)
        np.testing.assert_array_equal(before.y, after.y)
        np.testing.assert_array_equal(before.customdata, after.customdata)
    assert shown.figure.layout.title.text == hidden.figure.layout.title.text
    assert shown.figure.layout.shapes == hidden.figure.layout.shapes
    assert source.to_json() == original
    spec = PlotExportSpec.from_figure(chart_kind, source)
    assert render_fingerprint(spec, config, PlotContentConfig()) != render_fingerprint(
        spec, config, PlotContentConfig(show_selected_module_annotation=False)
    )


@pytest.mark.parametrize("name,meta", [
    ("Selected module", None), ("Selected M935", None),
    ("Custom highlight", {"export_role": "selected_module"}),
])
def test_legacy_and_explicit_selection_roles(name, meta):
    source = go.Figure([
        go.Scatter(x=[1, 2], y=[3, 4], mode="markers", name="Module M935"),
        go.Scatter(x=[1], y=[3], mode="markers+text", name=name, meta=meta, text=["M935"]),
    ])
    result = adapters.build_figure(
        source.to_json(), None, preset_config("Grant figure"),
        PlotContentConfig(show_selected_module_annotation=False),
    ).figure
    assert result.data[0].visible is not False
    assert result.data[1].visible is False


@pytest.mark.skipif(
    os.environ.get("LIONESS_EXPORT_INTEGRATION") != "1",
    reason="Real image export requires Chrome/Chromium",
)
def test_real_export_removes_selected_label_and_legend(module_frame, tmp_path):
    source = charts.module_entropy_figure(module_frame, selected_module=935)
    spec = PlotExportSpec.from_figure("module_entropy", source)
    config = replace(preset_config("Grant figure", spec.aspect_ratio), format="svg")
    for enabled in (True, False):
        content = PlotContentConfig(show_selected_module_annotation=enabled)
        fitted = adapters.fit_automatic_config(spec, config, content)
        svg, _ = adapters.render_figure(spec, fitted, content)
        (tmp_path / f"module_entropy_selected_{enabled}.svg").write_bytes(svg)
        text = " ".join(ET.fromstring(svg).itertext())
        assert ("M935" in text) == enabled
        assert ("Selected module" in text) == enabled
        assert "CT" in text and "TS" in text and "Normalized Shannon entropy" in text
