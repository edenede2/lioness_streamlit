from __future__ import annotations

from pathlib import Path

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
