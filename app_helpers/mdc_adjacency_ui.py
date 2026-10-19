"""Lazy single-image MDC presentation controls (no network recomputation)."""

from pathlib import Path

import streamlit as st

from app_helpers.figure_export import render_exportable_plot
from app_helpers.mdc_adjacency import configure_heatmap_view, split_heatmap_figure


def render_configurable_mdc_heatmap(
    path: Path, module: int, module_set: str, module_set_label: str,
    normalization_variant: str, region_counts: dict[str, int],
) -> None:
    figure = split_heatmap_figure(
        path, module, module_set, module_set_label, region_counts=region_counts,
    )
    metadata = figure.layout.meta["figure_export"]["split_mdc_heatmap"]
    prefix = f"mdc_adjacency_config_{module_set}_{module}_{normalization_variant}"
    with st.expander("Configure adjacency heatmap"):
        left, right = st.columns(2)
        with left:
            regions = st.checkbox(
                "Show region annotation strips",
                value=metadata["region_annotations_available"],
                disabled=not metadata["region_annotations_available"], key=prefix + "_regions",
            )
            legend = st.checkbox("Show tissue legend", value=figure.layout.showlegend,
                                 disabled=not metadata["region_annotations_available"],
                                 key=prefix + "_legend")
            colorbar = st.checkbox("Show edge-weight color bar", value=True, key=prefix + "_colorbar")
            summary = st.checkbox("Show MDC summary", value=bool(figure.layout.annotations),
                                  disabled=not bool(figure.layout.annotations), key=prefix + "_summary")
        with right:
            title = st.checkbox("Show heatmap title", value=True, key=prefix + "_title")
            diagnoses = st.checkbox("Show triangle labels and node count", value=True,
                                    key=prefix + "_diagnoses")
            x_label = st.text_input("Heatmap X-axis title", value=figure.layout.xaxis.title.text,
                                    key=prefix + "_x")
            y_label = st.text_input("Heatmap Y-axis title", value=figure.layout.yaxis.title.text,
                                    key=prefix + "_y")
        st.caption("Clear an axis title to hide it. Export figure offers fonts, size, margins and the same visibility controls. The heatmap remains the original raster; SVG/PDF include that raster with editable labels.")
    configure_heatmap_view(figure, regions)
    parts = figure.layout.title.text.split("<br>", 1)
    figure.layout.title.text = (parts[0] if title else "") + (
        ("<br>" if title else "") + parts[1] if diagnoses and len(parts) > 1 else ""
    )
    figure.update_layout(showlegend=legend)
    figure.update_xaxes(title=x_label)
    figure.update_yaxes(title=y_label)
    for annotation in figure.layout.annotations:
        annotation.visible = summary
    for trace in figure.data:
        if isinstance(trace.meta, dict) and trace.meta.get("export_role") == "colorbar":
            trace.marker.showscale = colorbar
    render_exportable_plot(
        figure, plot_id="main.mdc_split_triangle", metadata={
            "module_set": module_set, "module": f"M{module}",
            "normalization": normalization_variant,
        },
    )
