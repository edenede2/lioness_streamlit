"""Scientific integrity, export geometry, and opt-in real-browser verification."""

from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from PIL import Image
from plotly.subplots import make_subplots

from app_helpers.charts import distribution_figure, grouped_association_figure
from app_helpers.correlations import calculate_correlations
from app_helpers.figure_export import ExportConfig, PlotContentConfig, PlotExportSpec, preset_config
from app_helpers.figure_export import adapters
from app_helpers.figure_export.config import (
    PRESETS,
    convert_length,
    default_filename,
    output_summary,
    pixels,
    render_fingerprint,
    restore_figure,
    sanitize_filename,
    widget_key,
)


def correlation_figure(
    components=("CT", "TS"),
    *,
    kegg_subtitles=None,
    feature_label="Eigengene",
    phenotype_label="Outcome",
):
    rows = []
    for component in components:
        for group in ("Control", "AD"):
            for i in range(12):
                rows.append(
                    dict(
                        component=component,
                        component_label=component,
                        diagnosis_group=group,
                        metric_value=i + (group == "AD") * 0.3,
                        outcome=i * 0.4 + (i % 3) * 0.15,
                        sample_id=f"{component}-{group}-{i}",
                    )
                )
    frame = pd.DataFrame(rows)
    stats = calculate_correlations(
        frame.assign(grouping_level=frame.diagnosis_group),
        ["component", "grouping_level"],
        ["outcome"],
        min_group_n=5,
    )
    pooled = calculate_correlations(
        frame.assign(grouping_level="__pooled__"),
        ["component", "grouping_level"],
        ["outcome"],
        min_group_n=5,
    )
    stats = pd.concat([stats, pooled], ignore_index=True)
    stats["spearman_fdr_across_modules"] = stats.spearman_p * 2
    stats["pearson_fdr_across_modules"] = stats.pearson_p * 2
    figure = grouped_association_figure(
        frame,
        stats,
        phenotype="outcome",
        phenotype_label=phenotype_label,
        feature_label=feature_label,
        scale_label="Raw",
        grouping_variable="diagnosis_group",
        grouping_levels=["Control", "AD"],
        grouping_labels={"Control": "Control", "AD": "AD"},
        module=1112,
        color_by="diagnosis_group",
        color_label="Diagnosis",
        hover_fields={},
        categorical_color_fields=["diagnosis_group"],
        minimum_group_n=5,
        kegg_subtitles=kegg_subtitles,
    )
    return frame, stats, figure


def enrichment_figure():
    components = ("CT DLPFC–PCC", "CT DLPFC–AC", "CT PCC–AC", "TS DLPFC", "TS PCC", "TS AC")
    return correlation_figure(
        components,
        kegg_subtitles={
            c: "Regional edge enrichment: Organismal Systems / Nervous system / "
            "Cholinergic synapse and neuroactive ligand receptor interaction (FDR=0.00123)"
            for c in components
        },
        feature_label="Module eigengene expression (residualized and standardized)",
        phenotype_label="Global cognition score (residualized for age and sex)",
    )


@pytest.mark.parametrize("preset,width,unit,font,dpi", [(k, *v) for k, v in PRESETS.items()])
def test_presets(preset, width, unit, font, dpi):
    config = preset_config(preset, 1.5)
    assert (config.width, config.unit, config.typography.base, config.dpi) == (
        width,
        unit,
        font,
        dpi,
    )
    assert config.actual_height == pytest.approx(width / 1.5)
    if preset == "Grant figure":
        assert (
            config.typography.title,
            config.typography.subtitle,
            config.typography.axis_title,
            config.typography.tick,
            config.typography.legend,
            config.typography.annotation,
        ) == (16, 13, 13, 11, 11, 11)


@pytest.mark.parametrize("source,value", [("mm", 25.4), ("cm", 2.54), ("in", 1)])
def test_units_pixels_and_round_trip(source, value):
    assert convert_length(value, source, "in") == pytest.approx(1)
    assert pixels(value, source, 600) == 600
    for target in ("mm", "cm", "in"):
        assert convert_length(
            convert_length(value, source, target), target, source
        ) == pytest.approx(value)
    assert replace(ExportConfig(), width=165, height=110).pixel_dimensions == (3898, 2598)


def test_filenames_and_widget_keys():
    assert sanitize_filename("../../M1112 / CT: α?.png", "svg") == "M1112_CT.svg"
    assert sanitize_filename("CON", "pdf") == "figure_CON.pdf"
    assert sanitize_filename("...") == "figure"
    assert len(sanitize_filename("x" * 500)) == 170
    assert widget_key("a/b", "context", "width") != widget_key("a:b", "context", "width")
    assert widget_key("plot", "context", "width") == widget_key("plot", "context", "width")
    assert widget_key("plot", "context", "width") != widget_key("plot", "other", "width")
    spec = PlotExportSpec.from_figure(
        "pooled_correlation",
        go.Figure(),
        {"module": "M1112", "component": "CT", "outcome": "global_cognition"},
    )
    assert (
        default_filename(spec, preset_config("Grant figure"))
        == "M1112_CT_global_cognition_pooled_correlation_165mm_600dpi.png"
    )


def test_points_statistics_and_source_are_unchanged(monkeypatch):
    frame, stats, source = correlation_figure()
    original_frame, original_stats, snapshot = (
        frame.copy(deep=True),
        stats.copy(deep=True),
        source.to_json(),
    )
    spec = PlotExportSpec.from_figure("correlation", source)
    # The exporter may not rerun either a correlation or an OLS fit.
    monkeypatch.setattr(np, "polyfit", lambda *a, **k: pytest.fail("Export refit an OLS model"))
    config = preset_config("Grant figure")
    for mode in ("Pooled only", "Group-specific only", "Both", "None"):
        artifact = adapters.build_figure(
            spec.figure_snapshot,
            None,
            config,
            PlotContentConfig(correlation_mode=mode, statistic_precision=5),
        )
        assert [(list(t.x), list(t.y)) for t in artifact.figure.data] == [
            (list(t.x), list(t.y)) for t in source.data
        ]
        assert artifact.figure.layout.meta == source.layout.meta
        assert all(not t.marker.colorbar.to_plotly_json() for t in artifact.figure.data)
        annotation = " ".join(
            str(a.text)
            for a in artifact.figure.layout.annotations
            if (a.name or "").startswith("publication-statistics") and a.visible is not False
        )
        assert ("pooled" in annotation) == (mode in {"Both", "Pooled only"})
        assert ("<b>Control</b>" in annotation) == (mode in {"Both", "Group-specific only"})
        if mode != "None":
            assert "ρ=" in annotation
    assert source.to_json() == snapshot
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_frame_equal(stats, original_stats)


def test_typed_arrays_point_typography_and_deterministic_rebuild():
    source = go.Figure(
        go.Scatter(x=np.array([1.0, 2.0, np.nan]), y=np.array([2.0, 3.0, 4.0]), mode="markers")
    )
    spec = PlotExportSpec.from_figure("typed", source)
    a = adapters.build_figure(
        spec.figure_snapshot, None, preset_config("Grant figure"), PlotContentConfig()
    )
    b = adapters.build_figure(
        spec.figure_snapshot, None, preset_config("Grant figure"), PlotContentConfig()
    )
    assert a.figure.to_json() == b.figure.to_json()
    assert a.figure.layout.font.size == 12 * 96 / 72
    assert a.figure.layout.xaxis.tickfont.size == 11 * 96 / 72
    assert np.isnan(a.figure.data[0].x[-1])
    assert a.figure.layout.width == pytest.approx(165 / 25.4 * 96)
    assert (
        spec.fingerprint
        != PlotExportSpec.from_figure("typed", go.Figure(go.Scatter(x=[9], y=[9]))).fingerprint
    )
    assert render_fingerprint(spec, ExportConfig(), PlotContentConfig()) != render_fingerprint(
        spec, replace(ExportConfig(), dpi=300), PlotContentConfig()
    )


def test_panels_reorder_filter_and_statistics_alignment():
    _, _, figure = correlation_figure()
    config = preset_config("Grant figure")
    a = adapters.build_figure(
        figure.to_json(),
        None,
        config,
        PlotContentConfig(panel_order=("x2:y2", "x:y"), panel_labels=True),
    )
    assert [p.identifier for p in a.panels] == ["x2:y2", "x:y"]
    title = next(t for t in a.figure.layout.annotations if t.name == "publication-panel-title-1")
    assert title.x == pytest.approx(sum(a.figure.layout.xaxis2.domain) / 2)
    single = adapters.build_figure(
        figure.to_json(), None, config, PlotContentConfig(panel_order=("x2:y2",))
    )
    assert all(t.xaxis == "x2" for t in single.figure.data)
    assert (
        len(
            [
                a
                for a in single.figure.layout.annotations
                if (a.name or "").startswith("publication-statistics")
            ]
        )
        == 1
    )
    with pytest.raises(adapters.ExportError, match="panel"):
        adapters.build_figure(figure.to_json(), None, config, PlotContentConfig(panel_order=()))


def test_axis_sharing_handles_numpy_and_rejects_incompatible_types():
    from app_helpers.figure_export.config import figure_panels

    figure = make_subplots(rows=1, cols=2)
    for column in (1, 2):
        figure.add_trace(go.Scatter(x=np.arange(3), y=np.arange(3)), row=1, col=column)
    panels = figure_panels(figure)
    assert adapters.can_share_axes(figure, panels)
    figure.update_xaxes(type="log", row=1, col=2)
    assert not adapters.can_share_axes(figure, panels)


def test_tight_layout_wraps_titles_and_frames_3d_without_changing_data():
    figure = go.Figure(go.Scatter3d(x=[1, 2, 3], y=[3, 2, 1], z=[1, 3, 2]))
    figure.update_layout(
        title="A very long publication title " * 4, scene_camera_eye=dict(x=2, y=1, z=1)
    )
    snapshot = figure.to_json()
    artifact = adapters.build_figure(
        snapshot, None, preset_config("Grant figure"), PlotContentConfig()
    )
    assert "<br>" in artifact.figure.layout.title.text
    assert artifact.figure.layout.title.yref == "container"
    assert artifact.figure.layout.scene.camera.eye.x == pytest.approx(2.8)
    assert artifact.figure.layout.scene.xaxis.tickfont.size == pytest.approx(11 * 96 / 72)
    assert artifact.figure.layout.scene.xaxis.title.font.size == pytest.approx(13 * 96 / 72)
    assert list(artifact.figure.data[0].x) == [1, 2, 3]
    assert figure.to_json() == snapshot


def test_custom_builder_receives_isolated_snapshots():
    from app_helpers.figure_export import FigureArtifact

    data, results = {"values": [1, 2]}, {"rho": 0.25}
    captured = []

    def builder(frame, stats, config, content):
        captured.append((list(frame["values"]), stats["rho"]))
        frame["values"].append(9)
        stats["rho"] = 1
        return FigureArtifact(go.Figure(go.Scatter(x=frame["values"], y=frame["values"])), ())

    spec = PlotExportSpec(
        "callback",
        go.Figure().to_json(),
        builder=builder,
        data_snapshot=data,
        analysis_snapshot=results,
    )
    data["values"].append(3)
    results["rho"] = 0.9
    for _ in range(2):
        adapters.build_artifact(spec, ExportConfig(), PlotContentConfig())
    assert captured == [([1, 2], 0.25), ([1, 2], 0.25)]


def test_automatic_height_keeps_dense_annotations_and_manual_sizes_exact():
    _, _, figure = correlation_figure()
    spec = PlotExportSpec.from_figure("dense", figure)
    config = preset_config("Manuscript – single column", spec.aspect_ratio)
    fitted = adapters.fit_automatic_config(spec, config, PlotContentConfig())
    artifact = adapters.build_artifact(spec, fitted, PlotContentConfig())
    assert fitted.actual_height > config.actual_height
    assert (
        artifact.figure.layout.height
        - artifact.figure.layout.margin.t
        - artifact.figure.layout.margin.b
        > 90
    )
    manual = replace(config, height=65)
    assert adapters.fit_automatic_config(spec, manual, PlotContentConfig()) == manual
    assert artifact.figure.layout.height == pytest.approx(fitted.inches[1] * 96)


@pytest.mark.parametrize("columns", [2, 3])
def test_enrichment_multirow_layout_reserves_headers_and_axis_titles(columns):
    _, _, source = enrichment_figure()
    spec = PlotExportSpec.from_figure("enriched", source)
    config = preset_config("Grant figure", spec.aspect_ratio)
    config = replace(config, style=replace(config.style, panel_columns=columns, panel_spacing=0))
    content = PlotContentConfig(panel_labels=True)
    fitted = adapters.fit_automatic_config(spec, config, content)
    artifact = adapters.build_artifact(spec, fitted, content)
    figure = artifact.figure
    assert figure.layout.height + 0.5 >= artifact.minimum_height
    assert fitted.actual_height > config.actual_height
    plot_width = figure.layout.width - figure.layout.margin.l - figure.layout.margin.r
    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    pt = 96 / 72
    rows = sorted(
        {tuple(figure.layout[p.yaxis.replace("y", "yaxis", 1)].domain) for p in artifact.panels},
        reverse=True,
    )
    assert len(rows) == int(np.ceil(6 / columns))
    for p in artifact.panels:
        xaxis = figure.layout[p.xaxis.replace("x", "xaxis", 1)]
        yaxis = figure.layout[p.yaxis.replace("y", "yaxis", 1)]
        heading = next(
            a
            for a in adapters.panel_title_annotations(figure)
            if a.x == pytest.approx(sum(xaxis.domain) / 2) and a.y == pytest.approx(yaxis.domain[1])
        )
        available = (xaxis.domain[1] - xaxis.domain[0]) * plot_width
        assert all(
            adapters._text_width(line, config.typography.subtitle * pt, "DejaVu Sans") <= available
            for line in heading.text.split("<br>")
        )
        assert "<br>" in heading.text
        shared_x = next(
            a for a in figure.layout.annotations if a.name == "publication-shared-x-title"
        )
        assert "<br>" in shared_x.text
        assert not xaxis.title.text
        header_extent = (
            heading.yshift
            + (heading.text.count("<br>") + 1) * config.typography.subtitle * pt * 1.3
        )
        row_index = rows.index(tuple(yaxis.domain))
        if row_index:
            # The lower header cannot enter the preceding row's plotting area.
            gap = (rows[row_index - 1][0] - yaxis.domain[1]) * plot_height
            assert gap > header_extent + config.typography.tick * pt
        else:
            assert figure.layout.margin.t > header_extent
    assert [(list(t.x), list(t.y)) for t in figure.data] == [
        (list(t.x), list(t.y)) for t in source.data
    ]
    assert figure.layout.meta == source.layout.meta
    manual = replace(config, height=80)
    assert adapters.fit_automatic_config(spec, manual, content) == manual
    assert any(
        "manual height" in w for w in adapters.build_artifact(spec, manual, content).warnings
    )


def test_export_title_axis_and_legend_visibility_preserves_source_and_statistics():
    _, _, source = enrichment_figure()
    original = source.to_json()
    config = preset_config("Grant figure")
    content = PlotContentConfig(
        show_title=False,
        show_subtitle=False,
        show_panel_titles=False,
        show_panel_subtitles=False,
        show_x_title=False,
        show_y_title=False,
    )
    artifact = adapters.build_figure(
        original, None, replace(config, style=replace(config.style, legend_visible=False)), content
    )
    assert not artifact.figure.layout.title.text
    assert artifact.figure.layout.showlegend is False
    assert all(a.visible is False for a in adapters.panel_title_annotations(artifact.figure))
    assert all(
        not artifact.figure.layout[p.xaxis.replace("x", "xaxis", 1)].title.text
        and not artifact.figure.layout[p.yaxis.replace("y", "yaxis", 1)].title.text
        for p in artifact.panels
    )
    assert all(
        a.visible is not False
        for a in artifact.figure.layout.annotations
        if (a.name or "").startswith("publication-statistics")
    )
    assert artifact.figure.layout.meta == source.layout.meta
    assert source.to_json() == original
    headings_only = adapters.build_figure(
        original, None, config, PlotContentConfig(show_panel_subtitles=False)
    )
    assert all(
        a.visible is not False and "enrichment" not in a.text and "Cholinergic" not in a.text
        for a in adapters.panel_title_annotations(headings_only.figure)
    )
    enrichment_only = adapters.build_figure(
        original, None, config, PlotContentConfig(show_panel_titles=False)
    )
    assert all(
        "enrichment" in a.text.replace("<br>", "")
        for a in adapters.panel_title_annotations(enrichment_only.figure)
    )


def test_visibility_for_native_subplot_titles_and_3d_axis_titles():
    native = make_subplots(rows=1, cols=2, subplot_titles=["First panel", "Second panel"])
    for col in (1, 2):
        native.add_trace(go.Scatter(x=[1, 2], y=[2, 3]), row=1, col=col)
    artifact = adapters.build_figure(
        native.to_json(), None, ExportConfig(), PlotContentConfig(show_panel_titles=False)
    )
    assert all(a.visible is False for a in artifact.figure.layout.annotations)
    spatial = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3]))
    spatial.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    artifact = adapters.build_figure(
        spatial.to_json(),
        None,
        ExportConfig(),
        PlotContentConfig(show_x_title=False, show_y_title=False, show_z_title=False),
    )
    assert all(not artifact.figure.layout.scene[d + "axis"].title.text for d in ("x", "y", "z"))


def test_shared_axis_titles_can_be_disabled_without_changing_axis_ranges():
    _, _, source = correlation_figure()
    source.update_xaxes(range=[-2, 15], type="linear")
    source.update_yaxes(range=[-1, 8], type="linear")
    config = preset_config("Grant figure")
    shared = adapters.build_figure(source.to_json(), None, config, PlotContentConfig())
    repeated = adapters.build_figure(
        source.to_json(), None, config, PlotContentConfig(shared_axis_titles=False)
    )
    assert (
        len(
            [
                a
                for a in shared.figure.layout.annotations
                if (a.name or "").startswith("publication-shared-")
            ]
        )
        == 2
    )
    assert not any(
        (a.name or "").startswith("publication-shared-") for a in repeated.figure.layout.annotations
    )
    for p in shared.panels:
        for direction in ("x", "y"):
            key = direction + "axis" + getattr(p, direction + "axis")[1:]
            assert (
                shared.figure.layout[key].range
                == repeated.figure.layout[key].range
                == source.layout[key].range
            )
            assert repeated.figure.layout[key].title.text
            assert not shared.figure.layout[key].title.text


def test_narrow_automatic_layout_uses_fewer_columns_without_duplicate_labels():
    _, _, source = enrichment_figure()
    spec = PlotExportSpec.from_figure("narrow", source)
    content = PlotContentConfig(panel_labels=True)
    config = adapters.fit_automatic_config(
        spec, preset_config("Manuscript – single column", spec.aspect_ratio), content
    )
    artifact = adapters.build_artifact(spec, config, content)
    domains = {
        tuple(artifact.figure.layout[p.xaxis.replace("x", "xaxis", 1)].domain)
        for p in artifact.panels
    }
    assert len(domains) < 3
    assert (
        len([a for a in artifact.figure.layout.annotations if a.name == "publication-panel-label"])
        == 6
    )
    assert [p.identifier for p in artifact.panels] == [
        p.identifier for p in adapters.figure_panels(source)
    ]
    assert artifact.figure.layout.height + 0.5 >= artifact.minimum_height
    assert any("column(s)" in w for w in artifact.warnings)


def test_wrapping_long_emphasized_words_and_escaped_labels_preserves_text():
    wrapped = adapters._wrap_annotation("<b>LongComponentIdentifier</b> &amp; pathway", 50, 14)
    assert "<br>" in wrapped
    assert "<b>" in wrapped and "</b>" in wrapped and "&amp;" in wrapped
    assert "LongComponentIdentifier" in wrapped.replace("<br>", "")
    assert all(
        adapters._text_width(line, 14, "DejaVu Sans") <= 50 for line in wrapped.split("<br>")
    )


def test_continuous_color_legend_can_be_hidden_in_vector_exports():
    source = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], coloraxis="coloraxis"))
    source.update_layout(coloraxis=dict(colorscale="Viridis", colorbar_title="Effect"))
    artifact = adapters.build_figure(
        source.to_json(), None, ExportConfig(), PlotContentConfig(show_color_scale=False)
    )
    assert artifact.figure.layout.coloraxis.showscale is False
    assert artifact.figure.data[0].showscale is False
    vector = adapters.vectorize_figure(artifact.figure)
    assert all(
        getattr(getattr(t, "marker", None), "showscale", None) is not True for t in vector.data
    )
    assert list(artifact.figure.data[0].z[0]) == list(source.data[0].z[0])
    assert source.layout.coloraxis.showscale is None


def test_below_panel_statistics_align_across_a_row():
    _, _, source = correlation_figure()
    spec = PlotExportSpec.from_figure("below", source)
    content = PlotContentConfig(statistic_position="Below panel")
    config = adapters.fit_automatic_config(
        spec, preset_config("Grant figure", spec.aspect_ratio), content
    )
    artifact = adapters.build_artifact(spec, config, content)
    annotations = [
        a
        for a in artifact.figure.layout.annotations
        if (a.name or "").startswith("publication-statistics")
    ]
    assert len(annotations) == 2
    assert annotations[0].yshift == pytest.approx(annotations[1].yshift)


@pytest.mark.skipif(
    os.environ.get("LIONESS_EXPORT_INTEGRATION") != "1",
    reason="SVG text geometry requires a working Chrome runtime",
)
@pytest.mark.parametrize(
    "columns,preset", [(0, "Manuscript – single column"), (2, "Grant figure"), (3, "Grant figure")]
)
def test_real_multi_panel_svg_text_does_not_overlap_or_clip(columns, preset, tmp_path):
    browser = (
        os.environ.get("BROWSER_PATH") or shutil.which("google-chrome") or shutil.which("chromium")
    )
    if not browser:
        pytest.skip("Chrome/Chromium is not available on PATH")
    _, _, source = enrichment_figure()
    spec = PlotExportSpec.from_figure("text_geometry", source)
    content = PlotContentConfig(panel_labels=True)
    if columns == 0:
        content = replace(content, panel_order=("x:y", "x2:y2"))
    config = preset_config(preset, spec.aspect_ratio)
    config = replace(config, format="svg", style=replace(config.style, panel_columns=columns))
    config = adapters.fit_automatic_config(spec, config, content)
    svg, _ = adapters.render_figure(spec, config, content)
    # Measure the actual rendered glyph bounds, independently of the export
    # layout's estimates. Include shared axis titles, panel labels and legends.
    script = """
<script>
document.fonts.ready.then(() => {
 const svg = document.querySelector('svg');
 const outer = svg.getBoundingClientRect();
 const boxes = Array.from(svg.querySelectorAll('text.annotation-text, text.gtitle, text.legendtext')).map(node => {
   const b = node.getBoundingClientRect();
   return {text:node.textContent, x:b.x-outer.x, y:b.y-outer.y, width:b.width, height:b.height};
 });
 const outside = boxes.filter(b => b.x < -1 || b.y < -1 || b.x+b.width > outer.width+1 || b.y+b.height > outer.height+1);
 const overlaps = [];
 for(let i=0;i<boxes.length;i++) for(let j=i+1;j<boxes.length;j++) {
   const a=boxes[i], b=boxes[j];
   if(Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>1 && Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>1) overlaps.push([a.text,b.text]);
 }
 const output=document.createElement('pre'); output.id='report';
 output.textContent=JSON.stringify({count:boxes.length, outside, overlaps});
 document.body.append(output);
});
</script>
"""
    page = tmp_path / "figure.html"
    page.write_text(svg.decode() + script)
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--virtual-time-budget=3000",
            "--dump-dom",
            page.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    match = re.search(r'<pre id="report">(.*?)</pre>', result.stdout, re.S)
    assert match, result.stderr[-1000:]
    report = json.loads(match.group(1))
    assert report["count"] >= 10
    assert not report["outside"]
    assert not report["overlaps"]


@pytest.mark.parametrize(
    "detail,message",
    [
        (b"ChromeNotFoundError", "Chrome/Chromium"),
        (b"unexpected keyword argument headers", "incompatible"),
        (b"renderer stopped", "could not export"),
    ],
)
def test_worker_failures_are_actionable(monkeypatch, detail, message):
    class Worker:
        returncode = 1

        def communicate(self, *args, **kwargs):
            return b"", detail

    monkeypatch.setattr(adapters.subprocess, "Popen", lambda *args, **kwargs: Worker())
    with pytest.raises(adapters.ExportError, match=message):
        adapters.static_image(go.Figure(), "png", 100, 100)


def test_worker_timeout_terminates_process_group(monkeypatch):
    killed = []

    class Worker:
        pid = 12345

        def communicate(self, payload=None, timeout=None):
            if timeout is not None:
                raise adapters.subprocess.TimeoutExpired("renderer", timeout)
            return b"", b""

    monkeypatch.setattr(adapters.subprocess, "Popen", lambda *args, **kwargs: Worker())
    monkeypatch.setattr(adapters.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(adapters.ExportError, match="timed out"):
        adapters.static_image(go.Figure(), "png", 100, 100, timeout=0.1)
    assert killed == [(12345, adapters.signal.SIGKILL)]


def test_vector_heatmap_preserves_gaps_cells_labels_and_gl_points():
    original = go.Figure(
        go.Heatmap(
            z=np.array([[-1.0, np.nan], [0.5, 1.0]]),
            x=["A", "B"],
            y=["C", "D"],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=[["*", ""], ["", "*"]],
            texttemplate="%{text}",
        )
    )
    original.add_trace(go.Scattergl(x=["A"], y=["C"], mode="markers", customdata=[["kept"]]))
    artifact = adapters.build_figure(
        original.to_json(), None, preset_config("Grant figure"), PlotContentConfig()
    )
    vector = adapters.vectorize_figure(artifact.figure)
    cells = [s for s in vector.layout.shapes if s.name == "publication-heatmap-cell"]
    assert len(cells) == 3
    assert all(cell.layer == "between" for cell in cells)
    assert not any(t.type in {"heatmap", "scattergl"} for t in vector.data)
    assert sum(len(t.text) for t in vector.data if t.mode == "text") == 2
    text_trace = next(t for t in vector.data if t.mode == "text")
    assert text_trace.textfont.color == ("#fff", "#fff")
    assert vector.layout.xaxis.ticktext == ("A", "B")
    assert vector.data[-1].customdata == (["kept"],)
    assert adapters.vectorize_figure(artifact.figure).to_json() == vector.to_json()


def test_histogram_bins_and_seeded_jitter_are_retained():
    frame, _, _ = correlation_figure()
    for kind in ("Histogram", "Raincloud", "Violin", "ECDF"):
        source = distribution_figure(frame, "Eigengene", "Raw", ["Control", "AD"], 1112, kind)
        config = preset_config("Grant figure")
        artifact = adapters.build_figure(source.to_json(), None, config, PlotContentConfig())
        for old, new in zip(source.data, artifact.figure.data, strict=True):
            np.testing.assert_array_equal(old.x, new.x)
            if kind == "Histogram":
                assert old.xbins == new.xbins
            elif kind in {"Raincloud", "Violin"}:
                assert old.jitter == new.jitter
        assert (
            artifact.figure.to_json()
            == adapters.build_figure(
                source.to_json(), None, config, PlotContentConfig()
            ).figure.to_json()
        )


def fake_renderer(figure, format, width, height, scale=1, timeout=45):
    if format == "png":
        image = Image.new(
            "RGBA", (round(width * scale), round(height * scale)), (255, 255, 255, 255)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if format == "svg":
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><path d="M0 0 L1 1"/></svg>'.encode()
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=width * 0.75, height=height * 0.75)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("format", ["png", "tiff", "svg", "pdf"])
def test_output_dimensions_format_and_resolution(monkeypatch, format):
    monkeypatch.setattr(adapters, "static_image", fake_renderer)
    config = replace(preset_config("Grant figure"), height=110, format=format)
    spec = PlotExportSpec.from_figure("scatter", go.Figure(go.Scatter(x=[1, 2], y=[3, 4])))
    data, _ = adapters.render_figure(spec, config, PlotContentConfig())
    if format in {"png", "tiff"}:
        image = Image.open(io.BytesIO(data))
        assert image.size == (3898, 2598)
        assert image.info["dpi"][0] == pytest.approx(600, abs=0.1)
        if format == "tiff":
            assert image.tag_v2[259] == 5  # Lossless LZW.
    elif format == "svg":
        root = ET.fromstring(data)
        assert root.attrib["width"] == "165mm" and root.attrib["height"] == "110mm"
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
    else:
        from pypdf import PdfReader

        page = PdfReader(io.BytesIO(data)).pages[0]
        assert float(page.mediabox.width) == pytest.approx(165 / 25.4 * 72)
        assert float(page.mediabox.height) == pytest.approx(110 / 25.4 * 72)


def test_unsupported_formats_and_preflight_errors():
    spec = replace(PlotExportSpec.from_figure("example", go.Figure()), supported_formats=("png",))
    with pytest.raises(adapters.ExportError, match="unsupported"):
        adapters.render_figure(spec, replace(ExportConfig(), format="pdf"), PlotContentConfig())
    with pytest.raises(ValueError, match="Unsupported"):
        replace(ExportConfig(), format="gif").validate()
    with pytest.raises(ValueError, match="100 million"):
        replace(ExportConfig(), width=2000, height=2000).validate()
    assert "vector" in output_summary(replace(ExportConfig(), format="svg"))
    assert "DPI" not in output_summary(replace(ExportConfig(), format="svg"))


def test_all_rendering_sites_are_exportable_and_inline_builders_extracted():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for filename in [
        "streamlit_app.py",
        "app_helpers/regional_modules.py",
        "app_helpers/protocol_v11.py",
    ]:
        tree = ast.parse((root / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                assert name not in {"px.scatter", "px.bar", "px.histogram", "px.scatter_3d"}
                assert not name.endswith(".plotly_chart")
                if name == "render_plotly_chart":
                    count += 1
                    assert any(k.arg in {"key", "plot_id"} for k in node.keywords)
    assert count == 72


@pytest.mark.skipif(
    os.environ.get("LIONESS_EXPORT_INTEGRATION") != "1",
    reason="Real Kaleido tests require a working Chrome runtime",
)
@pytest.mark.parametrize(
    "kind", ["scatter", "raincloud", "heatmap", "webgl", "3d", "volcano", "enrichment"]
)
def test_real_kaleido_artifacts(kind, tmp_path):
    frame, _, figure = correlation_figure()
    if kind == "raincloud":
        figure = distribution_figure(
            frame, "Eigengene", "Raw", ["Control", "AD"], 1112, "Raincloud"
        )
    elif kind == "enrichment":
        _, _, figure = enrichment_figure()
    elif kind == "heatmap":
        figure = go.Figure(
            go.Heatmap(
                z=[[1, np.nan], [0.5, -1]],
                x=["A", "B"],
                y=["C", "D"],
                colorscale="RdBu",
                text=[["*", ""], ["", "**"]],
                texttemplate="%{text}",
            )
        )
    elif kind == "webgl":
        figure = go.Figure(go.Scattergl(x=np.arange(20), y=np.sin(np.arange(20)), mode="markers"))
    elif kind == "3d":
        figure = go.Figure(go.Scatter3d(x=[1, 2, 3], y=[2, 4, 3], z=[1, 4, 2], mode="markers"))
        figure.add_trace(
            go.Surface(
                x=[1, 2, 3],
                y=[2, 3, 4],
                z=[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                opacity=0.25,
                showscale=False,
                name="OLS plane",
            )
        )
        figure.update_layout(height=760, title="Endpoint expression triangle")
    elif kind == "volcano":
        from app_helpers.charts import edge_volcano_figure
        from app_helpers.data import load_volcano_candidates, load_volcano_bins

        figure = edge_volcano_figure(
            load_volcano_candidates("full_cohort", "lioness", "control_anchored", 935),
            load_volcano_bins("full_cohort", "lioness", "control_anchored", 935, "global"),
            module=935,
            scope="CT",
        )
    spec = PlotExportSpec.from_figure(kind, figure)
    for format in ("png", "tiff", "svg", "pdf"):
        config = replace(
            preset_config("Grant figure", spec.aspect_ratio),
            dpi=600 if kind == "scatter" else 150,
            height=None if kind in {"3d", "enrichment"} else 110,
            format=format,
        )
        config = adapters.fit_automatic_config(spec, config, PlotContentConfig())
        data, _ = adapters.render_figure(spec, config, PlotContentConfig())
        (tmp_path / f"{kind}.{format}").write_bytes(data)
        if format in {"png", "tiff"}:
            assert Image.open(io.BytesIO(data)).size == config.pixel_dimensions
        elif format == "svg":
            images = ET.fromstring(data).findall(".//{http://www.w3.org/2000/svg}image")
            assert bool(images) == (kind == "3d")
        else:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            assert float(reader.pages[0].mediabox.width) == pytest.approx(165 / 25.4 * 72)
            # Embedded fonts survive page normalization.
            fonts = reader.pages[0]["/Resources"].get("/Font", {})
            assert fonts
            for font in fonts.values():
                obj = font.get_object()
                descendant = obj.get("/DescendantFonts", [obj])[0].get_object()
                descriptor = descendant.get("/FontDescriptor")
                assert descriptor is not None
                assert any(
                    k in descriptor.get_object() for k in ("/FontFile", "/FontFile2", "/FontFile3")
                )
    if kind in {"scatter", "raincloud"}:
        config = replace(
            preset_config("Grant figure"),
            dpi=600 if kind == "scatter" else 150,
            height=110,
            format="png",
        )
        assert (
            adapters.render_figure(spec, config, PlotContentConfig())[0]
            == (tmp_path / f"{kind}.png").read_bytes()
        )
    if kind == "heatmap":
        # Inspect the exact vector-cell adapter through a PNG render as well.
        config = replace(preset_config("Grant figure"), dpi=150, height=110)
        vector = adapters.vectorize_figure(
            adapters.build_artifact(spec, config, PlotContentConfig()).figure
        )
        w, h = (v * 96 for v in config.inches)
        data = adapters.static_image(vector, "png", w, h, config.dpi / 96)
        (tmp_path / "heatmap_vector_preview.png").write_bytes(data)
