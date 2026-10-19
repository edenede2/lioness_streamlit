from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import pyarrow.parquet as pq
from plotly.subplots import make_subplots
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers.charts import prediction_performance_figure, protocol_performance_figure
from app_helpers.figure_export import ExportConfig, PlotContentConfig, PlotExportSpec
from app_helpers.figure_export import adapters, ui
from app_helpers.figure_export.bar_selection import bar_color_defaults, bar_options
from app_helpers.figure_export.config import render_fingerprint


BLOCKS = {"CT_pooled": "CT pooled", "TS_pooled": "TS pooled", "CT_TS_pooled": "CT + TS pooled"}
MODELS = {"covariates": "Demographics/APOE", "network_only": "Connectivity"}


def performance_figure():
    frame = pd.DataFrame([
        dict(metric="roc_auc", value=.6 + .03 * i + .01 * j, n_oof=331,
             predictor_block=block, model_variant=model)
        for i, block in enumerate(BLOCKS) for j, model in enumerate(MODELS)
    ])
    return prediction_performance_figure(
        frame, metric="roc_auc", block_labels=BLOCKS, block_order=tuple(BLOCKS),
        model_labels=MODELS, title="Nested-CV OOF performance",
    )


def build(figure, content):
    return adapters.build_artifact(PlotExportSpec.from_figure("prediction", figure),
                                   ExportConfig(width=180, height=120), content)


def test_models_and_predictor_blocks_filter_without_changing_values_or_source():
    source = performance_figure()
    snapshot = source.to_json()
    choices = bar_options(source)
    assert choices.series_label == "Models" and choices.category_label == "Predictor blocks"
    assert choices.categories == tuple(BLOCKS.values())
    assert len(choices.items) == 6
    content = PlotContentConfig(bar_series=("Connectivity",), bar_categories=("CT pooled", "TS pooled"))
    result = build(source, content).figure
    assert len(result.data) == 1 and result.data[0].name == "Connectivity"
    assert tuple(result.data[0].x) == ("CT pooled", "TS pooled")
    assert np.allclose(result.data[0].y, source.data[1].y[:2])
    assert np.array_equal(np.asarray(result.data[0].customdata, dtype=object), source.data[1].customdata[:2])
    assert result.data[0].marker.color == source.data[1].marker.color
    assert tuple(result.layout.xaxis.categoryarray) == ("CT pooled", "TS pooled")
    assert source.to_json() == snapshot


def test_explicit_bar_colors_survive_filtering_reordering_and_global_palette():
    source = performance_figure()
    snapshot = source.to_json()
    defaults = bar_color_defaults(source)
    assert defaults["bar:1:0"].lower() == source.data[1].marker.color.lower()
    content = PlotContentConfig(
        bar_series=("Connectivity", "Demographics/APOE"),
        bar_categories=("TS pooled", "CT pooled"),
        bar_series_colors=(("Connectivity", "#112233"), ("Demographics/APOE", "#445566")),
        bar_item_colors=(("bar:1:1", "#FF00AA"), ("bar:1:2", "#00FF00")),
    )
    config = ExportConfig(width=180, height=120)
    config = replace(config, style=replace(config.style, palette="Color-blind safe"))
    result = adapters.build_artifact(PlotExportSpec.from_figure("prediction", source), config, content).figure
    assert result.data[0].name == "Connectivity"
    assert tuple(result.data[0].marker.color) == ("#112233", "#FF00AA")
    assert tuple(result.data[1].marker.color) == ("#445566", "#445566")
    assert np.allclose(result.data[0].y, source.data[1].y[:2])
    assert source.to_json() == snapshot


@pytest.mark.parametrize("colors", [
    {"bar_series_colors": (("Unknown", "#001122"),)},
    {"bar_item_colors": (("bar:99:0", "#001122"),)},
    {"bar_item_colors": (("bar:0:0", "red"),)},
    {"bar_series_colors": (("Connectivity", "#001122"), ("Connectivity", "#334455"))},
])
def test_invalid_bar_color_overrides_are_explicit(colors):
    with pytest.raises(ValueError, match="bar color|Bar colors"):
        build(performance_figure(), PlotContentConfig(**colors))


def test_individual_bar_selection_keeps_point_styles_errors_text_and_hover_aligned():
    trace = go.Bar(
        name="Model", x=["CT", "TS", "Combined"], y=[.7, .6, .8],
        text=["CT text", "TS text", "Joint text"], ids=["ct", "ts", "all"],
        base=[.01, .02, .03], width=[.3, .4, .5],
        marker=dict(color=["red", "blue", "green"], opacity=[.4, .6, .8],
                    line=dict(color=["black", "white", "red"], width=[1, 2, 3])),
        customdata=[[10, "a"], [20, "b"], [30, "c"]],
        hovertext=["first", "second", "third"], textfont=dict(color=["red", "blue", "green"]),
        error_y=dict(type="data", symmetric=False, array=[.1, .2, .3], arrayminus=[.01, .02, .03]),
    )
    source = go.Figure(trace).update_yaxes(range=[0, 1])
    snapshot = source.to_json()
    result = build(source, PlotContentConfig(bar_items=("bar:0:0", "bar:0:2"))).figure
    kept = result.data[0]
    assert tuple(kept.x) == ("CT", "Combined") and tuple(kept.y) == (.7, .8)
    assert tuple(kept.text) == ("CT text", "Joint text")
    assert tuple(kept.ids) == ("ct", "all")
    assert tuple(kept.base) == (.01, .03) and tuple(kept.width) == (.3, .5)
    assert tuple(kept.marker.color) == ("red", "green")
    assert tuple(kept.marker.line.color) == ("black", "red")
    assert tuple(kept.error_y.array) == (.1, .3) and tuple(kept.error_y.arrayminus) == (.01, .03)
    assert np.array_equal(kept.customdata, [[10, "a"], [30, "c"]])
    assert tuple(kept.hovertext) == ("first", "third")
    assert tuple(result.layout.yaxis.range) == (0, 1)
    assert source.to_json() == snapshot


def test_category_and_model_order_are_export_only_and_horizontal_axes_compact():
    source = performance_figure()
    result = build(source, PlotContentConfig(bar_series=("Connectivity", "Demographics/APOE"),
                                             bar_categories=("TS pooled", "CT pooled"))).figure
    assert [t.name for t in result.data] == ["Connectivity", "Demographics/APOE"]
    assert tuple(result.layout.xaxis.categoryarray) == ("TS pooled", "CT pooled")
    horizontal = go.Figure(go.Bar(x=[1, 2, 3], y=["AC", "PCG", "DLPFC"], orientation="h"))
    horizontal.update_yaxes(autorange="reversed", tickvals=["AC", "PCG", "DLPFC"], ticktext=["a", "p", "d"])
    result = build(horizontal, PlotContentConfig(bar_categories=("DLPFC", "AC"))).figure
    assert tuple(result.layout.yaxis.categoryarray) == ("DLPFC", "AC")
    assert result.layout.yaxis.autorange == "reversed"
    assert tuple(result.layout.yaxis.ticktext) == ("a", "d")
    assert tuple(result.data[0].x) == (1, 3)


@pytest.mark.parametrize("content", [
    PlotContentConfig(bar_series=()), PlotContentConfig(bar_categories=()), PlotContentConfig(bar_items=()),
    PlotContentConfig(bar_categories=("Unknown",)), PlotContentConfig(bar_series=("Unknown",)),
    PlotContentConfig(bar_categories=("CT pooled", "CT pooled")), PlotContentConfig(bar_items=("bar:999:0",)),
])
def test_empty_or_invalid_bar_selection_is_explicit(content):
    with pytest.raises(ValueError, match="Select at least one bar|Invalid bar selection"):
        build(performance_figure(), content)


def test_mixed_kegg_matrices_and_numeric_positions_are_not_silently_filtered():
    mixed = go.Figure([go.Bar(x=[1, 2], y=["A", "B"], orientation="h"), go.Heatmap(z=[[1], [2]])])
    assert bar_options(mixed) is None
    assert bar_options(go.Figure(go.Bar(x=[1, 2], y=[3, 4]))) is None
    with pytest.raises(ValueError, match="unavailable"):
        build(mixed, PlotContentConfig(bar_categories=("A",)))
    annotated = performance_figure().add_annotation(x="CT pooled", y=.7, text="Point annotation")
    assert bar_options(annotated) is None


def test_empty_facets_are_removed_without_orphan_headings_or_axes():
    source = make_subplots(rows=1, cols=2, subplot_titles=["ROC-AUC", "Log loss"])
    source.add_trace(go.Bar(name="Model", x=["CT", "TS"], y=[.7, .8]), row=1, col=1)
    source.add_trace(go.Bar(name="Model", x=["CT", "TS"], y=[.3, .2]), row=1, col=2)
    content = PlotContentConfig(bar_items=("bar:0:0",), panel_order=("x:y", "x2:y2"))
    artifact = build(source, content)
    assert [p.identifier for p in artifact.panels] == ["x:y"]
    assert len(artifact.figure.data) == 1
    assert not any("Log loss" in str(a.text) for a in artifact.figure.layout.annotations)
    assert not artifact.figure.layout.xaxis2.to_plotly_json()
    source.data[0].showlegend = True
    source.data[1].showlegend = False
    artifact = build(source, PlotContentConfig(bar_items=("bar:1:0",), panel_order=("x:y", "x2:y2")))
    assert [p.identifier for p in artifact.panels] == ["x2:y2"]
    assert artifact.figure.data[0].showlegend is True  # The remaining model still has its legend.


def test_protocol_labels_and_render_cache_identity_include_selection():
    source = protocol_performance_figure(pd.DataFrame([
        dict(representation=r, value=.6, metric="auc", model_variant="ridge") for r in ["B", "E", "C0"]
    ]))
    choices = bar_options(source)
    assert choices.category_label == "Representations"
    spec = PlotExportSpec.from_figure("prediction", source)
    original = render_fingerprint(spec, ExportConfig(), PlotContentConfig())
    for content in (PlotContentConfig(bar_series=("ridge",)), PlotContentConfig(bar_categories=("C0",)), PlotContentConfig(bar_items=("bar:0:0",))):
        assert render_fingerprint(spec, ExportConfig(), content) != original


def test_saved_nested_cv_results_support_pooled_block_exports_without_recalculation():
    from app_helpers import data as D
    path = D.TARGETED_PREDICTION_FILES["oof_performance"]
    if not path.exists():
        pytest.skip("Local completed prediction catalog unavailable; do not download")
    frame = pq.read_table(path, filters=[
        ("model_outcome", "=", "diagnosis_binary"), ("evidence_tier", "=", "primary"),
        ("module_definition", "=", "control_derived"), ("network_method", "=", "control_anchored"),
        ("score_transform", "=", "raw"), ("edge_mask", "=", "all"),
        ("panel_strategy", "=", "tissue_neutral_ad"), ("metric", "=", "roc_auc"),
    ]).to_pandas()
    frame = frame.loc[frame.eigengene_source.isin(["matched_multitissue", "not_applicable"])]
    assert not frame.empty
    assert not frame.duplicated(["predictor_block", "model_variant"]).any()
    figure = prediction_performance_figure(
        frame, metric="roc_auc", block_labels=D.PREDICTION_BLOCK_LABELS,
        block_order=D.PREDICTION_BLOCK_ORDER, model_labels=D.PREDICTION_MODEL_LABELS,
        title="Nested-CV OOF performance",
    )
    series = (D.PREDICTION_MODEL_LABELS["network_only"], D.PREDICTION_MODEL_LABELS["covariates_plus_network"])
    result = build(figure, PlotContentConfig(bar_categories=("CT pooled", "TS pooled"), bar_series=series)).figure
    assert len(result.data) == 2
    for source, exported in [(next(t for t in figure.data if t.name == name), next(t for t in result.data if t.name == name)) for name in series]:
        positions = [i for i, x in enumerate(source.x) if x in {"CT pooled", "TS pooled"}]
        assert len(exported.x) == 2
        assert np.allclose(exported.y, np.asarray(source.y)[positions])
        assert np.array_equal(np.asarray(exported.customdata, dtype=object), np.asarray(source.customdata, dtype=object)[positions])


def _prediction_export_app():
    import pandas as pd
    import streamlit as st
    from app_helpers.charts import prediction_performance_figure
    from app_helpers.figure_export import render_exportable_plot
    from app_helpers.figure_export.ui import export_dialog_body
    st.session_state.setdefault("analysis_setting", "unchanged")
    frame = pd.DataFrame([
        dict(metric="roc_auc", value=.7, n_oof=331, predictor_block=b, model_variant=m)
        for b in ("CT", "TS", "CT_TS") for m in ("network_only", "covariates")
    ])
    figure = prediction_performance_figure(
        frame, metric="roc_auc", block_labels={"CT": "CT pooled", "TS": "TS pooled", "CT_TS": "CT + TS pooled"},
        model_labels={"network_only": "Connectivity", "covariates": "Demographics/APOE"}, title="Nested-CV OOF performance",
    )
    render_exportable_plot(figure, plot_id="prediction_test")
    if "test_open_spec" in st.session_state:
        export_dialog_body(st.session_state["test_open_spec"])


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_export_dialog_bar_controls_preview_download_reset_and_main_state(monkeypatch):
    import streamlit as st
    calls = []
    image = io.BytesIO()
    Image.new("RGB", (50, 40), "white").save(image, format="PNG")
    def renderer(spec, config, content):
        calls.append((spec, config, content))
        return image.getvalue(), ()
    monkeypatch.setattr(ui, "_render_cached", renderer)
    monkeypatch.setattr(ui, "export_dialog", lambda spec: st.session_state.__setitem__("test_open_spec", spec))
    app = AppTest.from_function(_prediction_export_app, default_timeout=30).run()
    assert not app.exception and not calls
    source_plot = app.get("plotly_chart")[0].proto.spec
    widget(app.button, "Export figure").click().run()
    assert not app.exception
    widget(app.multiselect, "Models to include").set_value(["Connectivity"]).run()
    widget(app.multiselect, "Predictor blocks to include").set_value(["CT pooled", "TS pooled"]).run()
    assert not app.exception
    assert calls[-1][2].bar_series == ("Connectivity",)
    assert calls[-1][2].bar_categories == ("CT pooled", "TS pooled")
    widget(app.checkbox, "Choose individual bars").check().run()
    individual = widget(app.multiselect, "Individual bars to include")
    assert len(individual.value) == 2
    individual.set_value(individual.value[:1]).run()
    assert len(calls[-1][2].bar_items) == 1
    widget(app.button, "Prepare download").click().run()
    assert app.get("download_button")
    assert len(calls[-1][2].bar_items) == 1 and calls[-1][1].dpi == 600
    assert app.get("plotly_chart")[0].proto.spec == source_plot
    assert app.session_state["analysis_setting"] == "unchanged"
    # No old prepared download remains visible for a newly empty selection.
    widget(app.multiselect, "Models to include").set_value([]).run()
    assert not app.exception
    assert any("Select at least one bar" in error.value for error in app.error)
    assert not app.get("download_button")
    widget(app.button, "Reset to defaults").click().run()
    assert calls[-1][2].bar_series is None and calls[-1][2].bar_categories is None
    assert calls[-1][2].bar_items is None
    assert not app.exception


def test_export_dialog_individual_colors_preview_download_and_reset(monkeypatch):
    import streamlit as st
    calls = []
    image = io.BytesIO()
    Image.new("RGB", (50, 40), "white").save(image, format="PNG")
    def renderer(spec, config, content):
        calls.append((spec, config, content))
        return image.getvalue(), ()
    monkeypatch.setattr(ui, "_render_cached", renderer)
    monkeypatch.setattr(ui, "export_dialog", lambda spec: st.session_state.__setitem__("test_open_spec", spec))
    app = AppTest.from_function(_prediction_export_app, default_timeout=30).run()
    source_plot = app.get("plotly_chart")[0].proto.spec
    widget(app.button, "Export figure").click().run()
    widget(app.multiselect, "Models to include").set_value(["Connectivity"]).run()
    widget(app.multiselect, "Predictor blocks to include").set_value(["CT pooled", "TS pooled"]).run()
    widget(app.checkbox, "Customize bar colors").check().run()
    widget(app.color_picker, "Connectivity color").set_value("#224466").run()
    widget(app.checkbox, "Override individual bar colors").check().run()
    picker = widget(app.multiselect, "Bars to recolor individually")
    picker.set_value(["bar:1:0"]).run()
    individual = next(control for control in app.color_picker if control.label.startswith("Bar color:"))
    individual.set_value("#CC3355").run()
    assert not app.exception
    assert dict(calls[-1][2].bar_series_colors) == {"Connectivity": "#224466"}
    assert list(dict(calls[-1][2].bar_item_colors).values()) == ["#CC3355"]
    exported = adapters.build_artifact(*calls[-1]).figure
    assert tuple(exported.data[0].marker.color) == ("#CC3355", "#224466")
    widget(app.button, "Prepare download").click().run()
    assert app.get("download_button") and calls[-1][2].bar_item_colors
    assert app.get("plotly_chart")[0].proto.spec == source_plot
    next(control for control in app.color_picker if control.label.startswith("Bar color:")).set_value("#009977").run()
    assert not app.get("download_button")  # Old bytes cannot be downloaded for a new color.
    assert list(dict(calls[-1][2].bar_item_colors).values()) == ["#009977"]
    widget(app.button, "Reset to defaults").click().run()
    assert not widget(app.checkbox, "Customize bar colors").value
    assert calls[-1][2].bar_series_colors == calls[-1][2].bar_item_colors == ()
    assert not app.exception
