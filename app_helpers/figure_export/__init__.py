"""Publication figure export models and lazily loaded Streamlit integration."""

from .config import ExportConfig, FigureArtifact, PlotContentConfig, PlotExportSpec, preset_config


def render_exportable_plot(figure, **kwargs):
    from .ui import render_exportable_plot as render

    return render(figure, **kwargs)


__all__ = [
    "ExportConfig",
    "FigureArtifact",
    "PlotContentConfig",
    "PlotExportSpec",
    "preset_config",
    "render_exportable_plot",
]
