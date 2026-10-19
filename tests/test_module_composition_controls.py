from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers.charts import module_region_composition_figure
from app_helpers.figure_export import ExportConfig, PlotContentConfig, PlotExportSpec
from app_helpers.figure_export.adapters import build_artifact
from app_helpers.figure_export.config import restore_figure


TISSUE_COLUMNS = {
    "AC": "n_genes_ac", "DLPFC": "n_genes_dlpfc", "PCG": "n_genes_pcg",
    "VH": "n_genes_vh", "SMA": "n_genes_sma", "QUAD": "n_genes_quad",
}


def details(tissues):
    frame = pd.DataFrame({"module": [3, 2, 1], "module_size": [100, 60, 100],
                          "cluster_type": ["CT", "TS", "CT"]})
    for tissue, counts in zip(tissues, ([10, 60, 20], [60, 0, 30], [30, 0, 50])):
        frame[TISSUE_COLUMNS[tissue]] = counts
    return frame


@pytest.mark.parametrize("tissues", [("AC", "DLPFC", "PCG"), ("VH", "SMA", "QUAD")])
@pytest.mark.parametrize("orientation", ["h", "v"])
@pytest.mark.parametrize("scale", ["counts", "proportions"])
def test_composition_orientation_scale_colors_and_raw_hover(tissues, orientation, scale):
    frame = details(tissues)
    original = frame.copy(deep=True)
    colors = dict(zip(tissues, ("#112233", "#445566", "#778899")))
    figure = module_region_composition_figure(frame, orientation=orientation, composition_scale=scale, tissue_colors=colors)
    category_axis = figure.layout.yaxis if orientation == "h" else figure.layout.xaxis
    assert tuple(category_axis.categoryarray) == ("M1", "M3", "M2")
    if orientation == "h":
        assert category_axis.autorange == "reversed"
    ordered = frame.sort_values(["module_size", "module"], ascending=[False, True])
    values = []
    for trace, tissue in zip(figure.data, tissues):
        counts = ordered[TISSUE_COLUMNS[tissue]].to_numpy()
        heights = trace.x if orientation == "h" else trace.y
        values.append(np.asarray(heights, dtype=float))
        assert trace.orientation == orientation and trace.marker.color == colors[tissue]
        assert list(trace.customdata[:, 3]) == counts.tolist()
        assert list(trace.customdata[:, 0]) == [100, 100, 60]
        assert np.allclose(heights, counts / np.array([100, 100, 60]) if scale == "proportions" else counts)
        assert "customdata[3]" in trace.hovertemplate
    assert np.allclose(sum(values), np.ones(3) if scale == "proportions" else [100, 100, 60])
    if scale == "proportions":
        value_axis = figure.layout.xaxis if orientation == "h" else figure.layout.yaxis
        assert tuple(value_axis.range) == (0, 1) and value_axis.tickformat == ".0%"
    pd.testing.assert_frame_equal(frame, original)


def test_percentage_composition_export_preserves_denominator_and_can_recolor_tissue():
    figure = module_region_composition_figure(details(("AC", "DLPFC", "PCG")), orientation="v", composition_scale="proportions")
    snapshot = figure.to_json()
    spec = PlotExportSpec.from_figure("composition", figure)
    content = PlotContentConfig(bar_series=("AC",), bar_series_colors=(("AC", "#BB2233"),))
    result = build_artifact(spec, ExportConfig(width=180, height=120), content).figure
    assert len(result.data) == 1 and np.allclose(result.data[0].y, [.2, .1, 1.])
    assert tuple(result.layout.yaxis.range) == (0, 1)
    assert tuple(result.data[0].marker.color) == ("#BB2233",) * 3
    assert figure.to_json() == snapshot


@pytest.mark.parametrize("tissues", [("AC", "DLPFC", "PCG"), ("VH", "SMA", "QUAD")])
@pytest.mark.parametrize("orientation", ["h", "v"])
@pytest.mark.parametrize("scale", ["counts", "proportions"])
@pytest.mark.parametrize("show_x,show_y", [(True, True), (False, True), (True, False), (False, False)])
def test_composition_ticks_are_independent_without_changing_bars(tissues, orientation, scale, show_x, show_y):
    frame = details(tissues)
    original = module_region_composition_figure(frame, orientation=orientation, composition_scale=scale)
    figure = module_region_composition_figure(
        frame, orientation=orientation, composition_scale=scale,
        show_x_ticks=show_x, show_y_ticks=show_y,
    )
    assert figure.layout.xaxis.showticklabels is show_x
    assert figure.layout.yaxis.showticklabels is show_y
    if not show_x:
        assert figure.layout.xaxis.ticks == ""
    if not show_y:
        assert figure.layout.yaxis.ticks == ""
    assert figure.layout.xaxis.title.text == original.layout.xaxis.title.text
    assert figure.layout.yaxis.title.text == original.layout.yaxis.title.text
    assert figure.layout.xaxis.range == original.layout.xaxis.range
    assert figure.layout.yaxis.range == original.layout.yaxis.range
    assert figure.layout.xaxis.categoryarray == original.layout.xaxis.categoryarray
    assert figure.layout.yaxis.categoryarray == original.layout.yaxis.categoryarray
    assert json.loads(figure.to_json())["data"] == json.loads(original.to_json())["data"]


def test_composition_exports_preserve_ticks_by_default_and_can_override():
    figure = module_region_composition_figure(details(("AC", "DLPFC", "PCG")),
                                              show_x_ticks=False, show_y_ticks=False)
    snapshot = figure.to_json()
    spec = PlotExportSpec.from_figure("composition.hidden_ticks", figure)
    config = ExportConfig(width=180, height=130)
    original = build_artifact(spec, config, PlotContentConfig()).figure
    assert not original.layout.xaxis.showticklabels and not original.layout.yaxis.showticklabels
    for x, y in ((True, False), (False, True), (True, True)):
        result = build_artifact(spec, config, PlotContentConfig(show_x_ticks=x, show_y_ticks=y)).figure
        assert result.layout.xaxis.showticklabels is x
        assert result.layout.yaxis.showticklabels is y
        assert json.loads(result.to_json())["data"] == json.loads(original.to_json())["data"]
    assert figure.to_json() == snapshot


def test_percentage_composition_rejects_invalid_denominators_and_counts():
    frame = details(("AC", "DLPFC", "PCG"))
    frame.loc[0, "n_genes_ac"] += 1
    with pytest.raises(ValueError, match="sum to each positive module size"):
        module_region_composition_figure(frame, composition_scale="proportions")
    with pytest.raises(ValueError, match="horizontal/vertical"):
        module_region_composition_figure(frame, orientation="wrong")


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def composition_figure(app):
    for chart in app.get("plotly_chart"):
        figure = restore_figure(chart.proto.spec)
        if "composition by module" in str(figure.layout.title.text):
            return figure
    raise AssertionError("Composition chart is missing")


@pytest.mark.parametrize("module_set,tissues", [
    ("control_derived", ("AC", "DLPFC", "PCG")),
    ("muscle_all_donors_l4", ("VH", "SMA", "QUAD")),
])
def test_real_module_details_page_controls_and_tissue_colors(module_set, tissues):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=120).run()
    app = widget(app.selectbox, "Module definition").set_value(module_set).run()
    app = widget(app.pills, "Analysis view").set_value("Module details").run()
    assert not app.exception
    assert widget(app.selectbox, "Composition orientation").value == "h"
    assert widget(app.selectbox, "Composition values").value == "counts"
    assert widget(app.checkbox, "Show X-axis ticks").value
    assert widget(app.checkbox, "Show Y-axis ticks").value
    app = widget(app.selectbox, "Composition orientation").set_value("v").run()
    app = widget(app.selectbox, "Composition values").set_value("proportions").run()
    app = widget(app.color_picker, f"{tissues[0]} tissue color").set_value("#AB1234").run()
    assert not app.exception
    figure = composition_figure(app)
    assert all(trace.orientation == "v" for trace in figure.data)
    assert all(trace.name in tissues for trace in figure.data)
    assert np.allclose(sum(np.asarray(trace.y, dtype=float) for trace in figure.data), 1)
    assert figure.data[0].marker.color == "#AB1234"
    assert tuple(figure.layout.yaxis.range) == (0, 1)
    app = widget(app.radio, "Module types to show").set_value("CT only").run()
    assert not app.exception
    assert widget(app.selectbox, "Composition orientation").value == "v"
    assert widget(app.color_picker, f"{tissues[0]} tissue color").value == "#AB1234"
    original = composition_figure(app)
    app = widget(app.checkbox, "Show X-axis ticks").uncheck().run()
    figure = composition_figure(app)
    assert not figure.layout.xaxis.showticklabels and figure.layout.yaxis.showticklabels
    assert json.loads(figure.to_json())["data"] == json.loads(original.to_json())["data"]
    app = widget(app.checkbox, "Show Y-axis ticks").uncheck().run()
    assert not app.exception
    figure = composition_figure(app)
    assert not figure.layout.xaxis.showticklabels and not figure.layout.yaxis.showticklabels
    assert figure.layout.xaxis.title.text == original.layout.xaxis.title.text
    assert figure.layout.yaxis.title.text == original.layout.yaxis.title.text
    app = widget(app.selectbox, "Composition orientation").set_value("h").run()
    assert not composition_figure(app).layout.xaxis.showticklabels
    assert not composition_figure(app).layout.yaxis.showticklabels
    app = widget(app.checkbox, "Show X-axis ticks").check().run()
    assert composition_figure(app).layout.xaxis.showticklabels
    assert not composition_figure(app).layout.yaxis.showticklabels
    assert not app.exception


def _composition_export_controls_app():
    import streamlit as st
    import pandas as pd
    from dataclasses import asdict
    from app_helpers.charts import module_region_composition_figure
    from app_helpers.figure_export import PlotExportSpec
    from app_helpers.figure_export.ui import _controls

    frame = pd.DataFrame({"module": [1], "module_size": [60], "cluster_type": ["CT"],
                          "n_genes_ac": [10], "n_genes_dlpfc": [20], "n_genes_pcg": [30]})
    figure = module_region_composition_figure(frame, show_x_ticks=False)
    _, content, _ = _controls(PlotExportSpec.from_figure("composition.tick_controls", figure))
    st.session_state["export_content_test"] = asdict(content)


def test_composition_export_dialog_offers_independent_ticks_from_main_plot():
    app = AppTest.from_function(_composition_export_controls_app, default_timeout=30).run()
    assert not app.exception
    assert not widget(app.checkbox, "Show X-axis ticks").value
    assert widget(app.checkbox, "Show Y-axis ticks").value
    app = widget(app.checkbox, "Show X-axis ticks").check().run()
    app = widget(app.checkbox, "Show Y-axis ticks").uncheck().run()
    assert not app.exception
    content = app.session_state["export_content_test"]
    assert content["show_x_ticks"] and not content["show_y_ticks"]
    assert content["show_x_title"] and content["show_y_title"]


@pytest.mark.skipif(os.environ.get("LIONESS_EXPORT_INTEGRATION") != "1", reason="Requires local Chrome")
@pytest.mark.parametrize("show_x,show_y", [(True, True), (False, True), (True, False), (False, False)])
def test_real_composition_svg_has_only_selected_ticks(show_x, show_y):
    import xml.etree.ElementTree as ET
    from app_helpers.figure_export.adapters import render_figure

    figure = module_region_composition_figure(
        details(("AC", "DLPFC", "PCG")), orientation="v", composition_scale="proportions",
        show_x_ticks=show_x, show_y_ticks=show_y,
    )
    blob, _ = render_figure(
        PlotExportSpec.from_figure("composition.ticks", figure),
        ExportConfig(width=180, height=125, format="svg"), PlotContentConfig(),
    )
    root = ET.fromstring(blob)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert bool(root.findall('.//svg:g[@class="xtick"]', ns)) is show_x
    assert bool(root.findall('.//svg:g[@class="ytick"]', ns)) is show_y
    labels = [" ".join(text.itertext()) for text in root.findall(".//svg:text", ns)]
    assert "Module" in labels and "Proportion of module genes" in labels
