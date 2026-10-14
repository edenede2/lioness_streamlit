from __future__ import annotations

import io
import json

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app_helpers.figure_export import ui


def fixture_app(monkeypatch, native=False, selected_module=False):
    calls = []
    image = Image.new("RGB", (100, 80), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    def render(spec, config, content):
        calls.append((spec.plot_id, config, content))
        return buffer.getvalue(), ()

    monkeypatch.setattr(ui, "_render_cached", render)
    # AppTest runs the full script for every widget event, whereas real dialogs
    # rerun their fragment. Persist the body in this fixture to test its callbacks.
    if not native and ui.export_dialog is not None:
        import streamlit as st

        monkeypatch.setattr(
            ui, "export_dialog", lambda spec: st.session_state.__setitem__("test_open_spec", spec)
        )
    app = AppTest.from_string(
        """
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app_helpers.figure_export import render_exportable_plot
st.session_state.setdefault("analysis_setting", "unchanged")
for identifier in ("first", "second"):
    figure = make_subplots(rows=1, cols=2, subplot_titles=["First panel<br>Enrichment pathway", "Second panel<br>Enrichment pathway"])
    for column in (1, 2):
        figure.add_trace(go.Scatter(x=[1,2,3], y=[2,4,3], mode="markers"), row=1, col=column)
    if SELECTED_MODULE_OVERLAY:
        figure.add_trace(go.Scatter(x=[2], y=[4], mode="markers+text", text=["M935"], name="Selected M935", meta={"export_role": "selected_module"}), row=1, col=1)
    figure.update_layout(title="Original plot title<br><sup>Original subtitle</sup>")
    render_exportable_plot(figure, plot_id=identifier)
if "test_open_spec" in st.session_state:
    from app_helpers.figure_export.ui import export_dialog_body
    export_dialog_body(st.session_state["test_open_spec"])
""".replace("SELECTED_MODULE_OVERLAY", repr(selected_module)),
        default_timeout=15,
    ).run()
    return app, calls


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_collapsed_export_does_no_rendering_and_keys_are_unique(monkeypatch):
    app, calls = fixture_app(monkeypatch)
    assert not app.exception
    assert not calls
    assert len(app.button) == 2
    assert app.button[0].key != app.button[1].key
    assert not app.selectbox


def test_native_dialog_opens(monkeypatch):
    app, calls = fixture_app(monkeypatch, native=True)
    app.button[0].click().run()
    assert not app.exception
    assert widget(app.selectbox, "Publication preset")


def test_dialog_presets_units_and_main_state_are_isolated(monkeypatch):
    app, calls = fixture_app(monkeypatch)
    original_specs = [e.proto.spec for e in app.get("plotly_chart")]
    app.button[0].click().run()
    assert not app.exception
    assert calls[-1][0] == "first"
    widget(app.selectbox, "Publication preset").select("Grant figure").run()
    assert widget(app.number_input, "Figure width").value == 165
    assert widget(app.number_input, "Base font (pt)").value == 12
    assert widget(app.number_input, "Title font (pt)").value == 16
    widget(app.selectbox, "Dimension units").select("in").run()
    assert widget(app.number_input, "Figure width").value == pytest.approx(165 / 25.4)
    widget(app.number_input, "Title font (pt)").set_value(20.0).run()
    assert app.session_state["analysis_setting"] == "unchanged"
    assert [e.proto.spec for e in app.get("plotly_chart")] == original_specs
    widget(app.button, "Reset to defaults").click().run()
    assert widget(app.number_input, "Title font (pt)").value == 16
    assert not app.exception


def test_prepare_download_and_errors(monkeypatch):
    app, calls = fixture_app(monkeypatch)
    app.button[0].click().run()
    widget(app.button, "Prepare download").click().run()
    assert not app.exception
    assert app.get("download_button")
    assert calls[-1][1].dpi == 600

    def failure(*args):
        raise ui.ExportError("Chrome is unavailable; check browser setup.")

    monkeypatch.setattr(ui, "_render_cached", failure)
    widget(app.number_input, "Figure width").set_value(90.0).run()
    assert not app.exception
    assert any("Chrome" in e.value for e in app.error)


def test_visibility_controls_preview_download_and_reset_are_isolated(monkeypatch):
    app, calls = fixture_app(monkeypatch)
    original_plots = [e.proto.spec for e in app.get("plotly_chart")]
    app.button[0].click().run()
    for label in (
        "Show plot title",
        "Show plot subtitle",
        "Show panel titles",
        "Show panel subtitles / enrichment",
        "Show X-axis title",
        "Show Y-axis title",
        "Show legend",
    ):
        widget(app.checkbox, label).uncheck().run()
    assert not app.exception
    _, preview_config, content = calls[-1]
    assert not any(
        (
            content.show_title,
            content.show_subtitle,
            content.show_panel_titles,
            content.show_panel_subtitles,
            content.show_x_title,
            content.show_y_title,
        )
    )
    assert preview_config.style.legend_visible is False
    widget(app.button, "Prepare download").click().run()
    assert calls[-1][2] == content
    assert calls[-1][1].style.legend_visible is False
    assert app.get("download_button")
    assert [e.proto.spec for e in app.get("plotly_chart")] == original_plots
    assert app.session_state["analysis_setting"] == "unchanged"
    widget(app.button, "Reset to defaults").click().run()
    assert calls[-1][2].show_title and calls[-1][2].show_panel_subtitles
    assert calls[-1][1].style.legend_visible
    assert not app.exception


def test_popover_fallback(monkeypatch):
    monkeypatch.setattr(ui, "export_dialog", None)
    # Multiple open popovers remain namespaced, even on a legacy runtime.
    app, calls = fixture_app(monkeypatch)
    assert not app.exception
    assert not calls
    app.checkbox[0].check().run()
    assert not app.exception
    assert {c[0] for c in calls} == {"first"}


def test_selected_module_annotation_control_preview_download_and_reset(monkeypatch):
    app, calls = fixture_app(monkeypatch, selected_module=True)
    original_plots = [e.proto.spec for e in app.get("plotly_chart")]
    app.button[0].click().run()
    control = widget(app.checkbox, "Show selected-module annotation")
    assert control.value is True
    control.uncheck().run()
    assert not app.exception
    assert calls[-1][2].show_selected_module_annotation is False
    widget(app.button, "Prepare download").click().run()
    assert calls[-1][2].show_selected_module_annotation is False
    assert app.get("download_button")
    assert [e.proto.spec for e in app.get("plotly_chart")] == original_plots
    assert app.session_state["analysis_setting"] == "unchanged"
    widget(app.button, "Reset to defaults").click().run()
    assert calls[-1][2].show_selected_module_annotation is True
    # Visibility settings are local to this plot, not the next export.
    app.button[1].click().run()
    assert widget(app.checkbox, "Show selected-module annotation").value is True


def test_selected_module_annotation_control_absent_without_overlay(monkeypatch):
    app, _ = fixture_app(monkeypatch)
    app.button[0].click().run()
    assert not app.exception
    assert not any(c.label == "Show selected-module annotation" for c in app.checkbox)


def test_cache_uses_complete_snapshot_and_settings_and_is_session_bounded(monkeypatch):
    import plotly.graph_objects as go
    from app_helpers.figure_export import (
        ExportConfig,
        FigureArtifact,
        PlotContentConfig,
        PlotExportSpec,
    )

    state, calls = {}, []
    monkeypatch.setattr(ui.st, "session_state", state)

    def renderer(spec, config, content):
        calls.append(spec.figure_snapshot)
        return spec.figure_snapshot.encode(), FigureArtifact(go.Figure(), ())

    monkeypatch.setattr(ui, "render_figure", renderer)
    config, content = ExportConfig(), PlotContentConfig()
    spec = PlotExportSpec.from_figure("first", go.Figure(go.Scatter(x=[1], y=[2])))
    first = ui._render_cached(spec, config, content)
    assert ui._render_cached(spec, config, content) == first
    assert len(calls) == 1
    for i in range(5):
        other = PlotExportSpec.from_figure(str(i), go.Figure(go.Scatter(x=[i], y=[i])))
        assert ui._render_cached(other, config, content)[0] == other.figure_snapshot.encode()
    assert len(state["figure_export::render_cache"]) == 3
    changed = PlotContentConfig(title="New title")
    ui._render_cached(spec, config, changed)
    assert len(calls) == 7

    # Custom builders cannot inherit cached output from a different callback.
    callback = PlotExportSpec("custom", spec.figure_snapshot, builder=lambda *args: None)
    ui._render_cached(callback, config, content)
    ui._render_cached(callback, config, content)
    assert len(calls) == 9
