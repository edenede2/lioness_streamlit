"""Split-triangle display/export controls must never alter scientific pixels."""

import base64
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from app_helpers import mdc_adjacency as mdc
from app_helpers.figure_export import PlotContentConfig, PlotExportSpec, preset_config
from app_helpers.figure_export.adapters import build_figure, is_mixed, trace_role
from app_helpers.figure_export import ui


@pytest.fixture
def brain_image(tmp_path):
    # Source-renderer-compatible scientific fixture, not a reconstructed network.
    image = Image.new("RGB", (3200, 3200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((416, 672, 2591, 2815), fill="#FFFF00")
    draw.polygon([(416, 672), (416, 2815), (2591, 2815)], fill="#FF0000")
    draw.rectangle((362, 672, 396, 2815), fill="#4C78A8")
    draw.rectangle((416, 621, 2591, 655), fill="#4C78A8")
    path = tmp_path / "M121.png"
    image.save(path)
    pd.DataFrame([dict(
        module_set="control_derived", module=121, n_nodes=300,
        png=path.name, upper_triangle="AD", lower_triangle="Control",
        color_palette="yellow_orange_red", display_vmax=0.2574,
        mdc_total=1.2, mdc_ct=1.3, mdc_ts=1.1, width=3200, height=3200,
        png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )]).to_csv(tmp_path / "manifest.tsv", sep="\t", index=False)
    return path


def figure_for(path):
    return mdc.split_heatmap_figure(
        path, 121, "control_derived", "Control-derived L4 modules",
        region_counts={"AC":100, "DLPFC":100, "PCG":100},
    )


def test_source_bytes_numeric_scale_and_triangle_order_preserved(brain_image):
    figure = figure_for(brain_image)
    image = figure.data[0]
    assert image.type == "image" and image.z is None
    assert base64.b64decode(image.source.split(",", 1)[1]) == brain_image.read_bytes()
    assert "Upper triangle: AD" in figure.layout.title.text
    assert "Lower triangle: Control" in figure.layout.title.text
    bar = next(t for t in figure.data if trace_role(t) == "colorbar")
    assert bar.marker.cmin == 0 and bar.marker.cmax == 0.2574
    assert list(bar.marker.colorscale) == [(v, c) for v, c in mdc.COLORSCALE]
    assert is_mixed(figure)
    assert all(all(v is None for v in t.x) for t in figure.data[1:])
    assert {t.marker.color for t in figure.data if trace_role(t) == "legend"} == set(mdc.REGION_COLORS.values())


def test_region_viewport_titles_legend_colorbar_and_summary_export(brain_image):
    source = figure_for(brain_image)
    snapshot = source.to_json()
    content = PlotContentConfig(
        show_region_annotations=False, x_label="Horizontal genes", y_label="Vertical genes",
        show_color_scale=False, show_statistics=False, show_subtitle=False,
    )
    config = preset_config("Grant figure", 1)
    config = replace(config, format="svg", style=replace(config.style, legend_visible=False))
    artifact = build_figure(snapshot, None, config, content)
    exported = artifact.figure
    meta = exported.layout.meta["figure_export"]["split_mdc_heatmap"]
    left, top, right, bottom = meta["core_bounds"]
    assert exported.layout.xaxis.range == (left, right)
    assert exported.layout.yaxis.range == (bottom, top)
    assert exported.layout.xaxis.fixedrange and exported.layout.yaxis.fixedrange
    assert not meta["region_annotations_shown"]
    assert exported.data[0].source == source.data[0].source
    assert exported.layout.xaxis.title.text == "Horizontal genes"
    assert exported.layout.yaxis.title.text == "Vertical genes"
    assert not exported.layout.showlegend
    assert not next(t for t in exported.data if trace_role(t) == "colorbar").marker.showscale
    assert not exported.layout.annotations[0].visible
    assert "Upper triangle" not in exported.layout.title.text
    assert any("raster" in warning for warning in artifact.warnings)
    assert not any("Wide 3D" in warning for warning in artifact.warnings)
    assert source.to_json() == snapshot  # Rebuilding never mutates the main plot.
    for trace in exported.data:
        if trace_role(trace) == "legend":
            assert trace.marker.size == 12 and trace.marker.opacity is None


def test_custom_palette_rejected_instead_of_mismatching_raster(brain_image):
    config = preset_config("Grant figure", 1)
    config = replace(config, style=replace(config.style, palette="Viridis"))
    with pytest.raises(ValueError, match="matching original palette"):
        build_figure(figure_for(brain_image).to_json(), None, config, PlotContentConfig())


@pytest.mark.parametrize("changed,value,error", [
    ("module_set", "full_cohort", "different module definition"),
    ("upper_triangle", "Control", "diagnosis ordering"),
    ("display_vmax", 0, "numeric color limit"),
    ("png_sha256", "bad", "checksum"),
    ("png", "M935.png", "filename"),
    ("color_palette", "other", "yellow–red"),
])
def test_mismatched_release_rejected(brain_image, changed, value, error):
    path = brain_image.parent / "manifest.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.loc[0, changed] = value
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match=error):
        figure_for(brain_image)


def test_legacy_comma_delimited_motor_manifest_without_region_strips(tmp_path):
    path = tmp_path / "M7.png"
    image = Image.new("RGB", (600, 600), "white")
    ImageDraw.Draw(image).rectangle((90, 90, 499, 499), fill="#FFA500")
    image.save(path)
    pd.DataFrame([dict(module=7, filename="M7.png", n_nodes=250, display_q99=0.125,
                       upper_triangle="AD", lower_triangle="Control", color_palette="yellow_orange_red")
                  ]).to_csv(tmp_path / "manifest.tsv", index=False)
    figure = mdc.split_heatmap_figure(path, 7, "muscle_all_donors_l4", "Motor all-donor modules")
    assert not figure.layout.meta["figure_export"]["split_mdc_heatmap"]["region_annotations_available"]
    assert not figure.layout.showlegend
    assert not figure.layout.annotations
    assert figure.data[-1].marker.cmax == 0.125


def test_missing_manifest_and_unknown_image_fail_without_reconstruction(brain_image, monkeypatch):
    (brain_image.parent / "manifest.tsv").unlink()
    def missing(_):
        raise FileNotFoundError("Matching manifest unavailable")
    monkeypatch.setattr(mdc, "ensure_data_path", missing)
    with pytest.raises(FileNotFoundError):
        figure_for(brain_image)
    other = brain_image.parent / "unsupported.png"
    Image.new("RGB", (300, 300), "white").save(other)
    with pytest.raises(ValueError, match="layout"):
        mdc._image_geometry(str(other), other.stat().st_size, other.stat().st_mtime_ns)
    assert mdc._image_geometry.cache_info().maxsize == 8


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_main_configuration_and_export_controls_render(brain_image, monkeypatch):
    calls = []
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")

    def render(spec, config, content):
        calls.append((config, content))
        return buffer.getvalue(), ()

    monkeypatch.setattr(ui, "_render_cached", render)
    import streamlit as st
    monkeypatch.setattr(ui, "export_dialog", lambda spec: st.session_state.__setitem__("test_export", spec))
    app = AppTest.from_string(f'''
from pathlib import Path
import streamlit as st
from app_helpers.mdc_adjacency_ui import render_configurable_mdc_heatmap
from app_helpers.figure_export.ui import export_dialog_body
render_configurable_mdc_heatmap(Path({str(brain_image)!r}),121,"control_derived","Control-derived L4 modules","original",{{"AC":100,"DLPFC":100,"PCG":100}})
if "test_export" in st.session_state:
    export_dialog_body(st.session_state["test_export"])
''', default_timeout=20).run()
    assert not app.exception
    assert not calls  # No browser renderer until the export dialog is requested.
    widget(app.checkbox, "Show region annotation strips").uncheck().run()
    widget(app.checkbox, "Show tissue legend").uncheck().run()
    widget(app.checkbox, "Show edge-weight color bar").uncheck().run()
    widget(app.text_input, "Heatmap X-axis title").set_value("New X title").run()
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert not figure["layout"]["showlegend"]
    assert not figure["layout"]["meta"]["figure_export"]["split_mdc_heatmap"]["region_annotations_shown"]
    assert figure["layout"]["xaxis"]["title"]["text"] == "New X title"
    # Re-enable for separate exporter controls to be offered.
    widget(app.checkbox, "Show edge-weight color bar").check().run()
    widget(app.button, "Export figure").click().run()
    assert not app.exception
    assert widget(app.selectbox, "Color palette").options == ["Original"]
    assert not any(e.label == "Z-axis label" for e in app.text_input)
    assert not any(e.label == "Point size (pt)" for e in app.number_input)
    region_export = [e for e in app.checkbox if e.label == "Show region annotation strips"][-1]
    assert not region_export.value  # Export starts with the main plot's viewport.
    region_export.check().run()
    assert calls[-1][1].show_region_annotations
    assert not app.exception


def test_real_mdc_page_is_lazy_and_routes_configuration(brain_image, monkeypatch):
    from app_helpers import data
    calls = []
    def load(*args):
        calls.append(args)
        return brain_image
    monkeypatch.setattr(data, "load_mdc_split_heatmap", load)
    monkeypatch.setattr(data, "mdc_split_heatmap_available", lambda *args: True)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=120).run()
    widget(app.selectbox, "Module definition").set_value("control_derived").run()
    widget(app.selectbox, "Module").set_value(121).run()
    widget(app.pills, "Analysis view").set_value("MDC").run()
    assert not app.exception
    assert not calls  # Entering MDC must not fetch image/manifests.
    widget(app.toggle, "Load the selected module adjacency heatmap").set_value(True).run()
    assert not app.exception
    assert calls == [("control_derived", 121, "all", "original")]
    widget(app.checkbox, "Show region annotation strips").uncheck().run()
    widget(app.checkbox, "Show tissue legend").uncheck().run()
    widget(app.text_input, "Heatmap Y-axis title").set_value("").run()
    assert not app.exception
    figure = next(json.loads(c.proto.spec) for c in app.get("plotly_chart")
                  if '"split_mdc_heatmap"' in c.proto.spec)
    assert not figure["layout"]["showlegend"]
    assert figure["layout"]["yaxis"]["title"]["text"] == ""
    assert not figure["layout"]["meta"]["figure_export"]["split_mdc_heatmap"]["region_annotations_shown"]


@pytest.mark.skipif(os.environ.get("LIONESS_EXPORT_INTEGRATION") != "1", reason="Requires local Chrome")
@pytest.mark.parametrize("format", ["png", "tiff", "svg", "pdf"])
def test_real_mdc_export(brain_image, format, tmp_path):
    from app_helpers.figure_export.adapters import render_figure
    spec = PlotExportSpec.from_figure("mdc.configurable", figure_for(brain_image))
    config = replace(preset_config("Grant figure", 1), format=format, dpi=150)
    config = replace(config, style=replace(config.style, legend_visible=False))
    content = PlotContentConfig(show_region_annotations=False, show_statistics=False,
                                show_subtitle=False, show_color_scale=False,
                                x_label="Horizontal genes", y_label="Vertical genes")
    blob, artifact = render_figure(spec, config, content)
    assert blob
    assert artifact.figure.data[0].source == figure_for(brain_image).data[0].source
    (tmp_path / f"mdc.{format}").write_bytes(blob)
    if format == "svg":
        assert b"Horizontal genes" in blob and b"Vertical genes" in blob
        assert b"data:image/png;base64," in blob
        assert b"Upper triangle:" not in blob and b"AC (100)" not in blob
    if format in {"png", "tiff"}:
        with Image.open(io.BytesIO(blob)) as image:
            assert image.width == config.pixel_dimensions[0]
