"""Immutable publication settings and figure snapshots (no Streamlit dependency)."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import unicodedata
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go

CSS_DPI = 96
POINTS_PER_INCH = 72
FORMATS = ("png", "tiff", "svg", "pdf")
UNIT_INCHES = {"mm": 1 / 25.4, "cm": 1 / 2.54, "in": 1.0}
MIME_TYPES = {
    "png": "image/png",
    "tiff": "image/tiff",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}


def convert_length(value: float, source: str, destination: str) -> float:
    return float(value) * UNIT_INCHES[source] / UNIT_INCHES[destination]


def pixels(value: float, unit: str, dpi: float) -> int:
    return max(1, int(math.floor(convert_length(value, unit, "in") * dpi + 0.5)))


@dataclass(frozen=True)
class Typography:
    family: str = "Arial"
    base: float = 9
    title: float = 12
    subtitle: float = 9.75
    axis_title: float = 9.75
    tick: float = 8.25
    legend: float = 8.25
    annotation: float = 8.25
    bold_title: bool = True
    color: str = "#202124"
    annotation_spacing: float = 2


@dataclass(frozen=True)
class Style:
    background: str = "#FFFFFF"
    transparent: bool = False
    tight: bool = True
    # Final physical points, including padding.
    margin_left: float = 40
    margin_right: float = 18
    margin_top: float = 32
    margin_bottom: float = 38
    grid: bool = True
    grid_opacity: float = 0.18
    axis_lines: bool = True
    axis_width: float = 0.6
    tick_angle: float | None = None
    number_format: str = "Original"
    decimals: int = 3
    palette: str = "Original"
    point_size: float = 4
    point_opacity: float = 0.74
    point_shape: str = "Original"
    point_encoding: str = "Original"
    single_color: str = "#0072B2"
    line_width: float = 1.5
    line_dash: str = "Original"
    line_opacity: float = 1
    regression_color: str = "Original"
    regression_width: float = 1.8
    regression_dash: str = "Original"
    ci_opacity: float = 0.2
    legend_visible: bool = True
    legend_position: str = "Original"
    legend_orientation: str = "Original"
    legend_title: str | None = None
    panel_columns: int = 0
    panel_spacing: float | None = None
    shared_axes: bool | None = None


@dataclass(frozen=True)
class ExportConfig:
    preset: str = "Manuscript – single column"
    width: float = 85
    height: float | None = None
    unit: str = "mm"
    aspect_ratio: float = 1.5
    dpi: int = 600
    format: str = "png"
    typography: Typography = field(default_factory=Typography)
    style: Style = field(default_factory=Style)

    @property
    def actual_height(self) -> float:
        return self.height if self.height is not None else self.width / self.aspect_ratio

    @property
    def inches(self) -> tuple[float, float]:
        return convert_length(self.width, self.unit, "in"), convert_length(
            self.actual_height, self.unit, "in"
        )

    @property
    def pixel_dimensions(self) -> tuple[int, int]:
        return pixels(self.width, self.unit, self.dpi), pixels(
            self.actual_height, self.unit, self.dpi
        )

    @property
    def vector(self) -> bool:
        return self.format in {"svg", "pdf"}

    def validate(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(
                f"Unsupported figure format: {self.format}. Choose PNG, TIFF, SVG or PDF."
            )
        if self.unit not in UNIT_INCHES:
            raise ValueError("Dimensions must use mm, cm or in.")
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.width, self.actual_height, self.aspect_ratio, self.dpi)
        ):
            raise ValueError("Dimensions, aspect ratio and DPI must be positive finite values.")
        if not 72 <= self.dpi <= 1200:
            raise ValueError("Choose a DPI between 72 and 1200.")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.typography.base,
                self.typography.title,
                self.typography.subtitle,
                self.typography.axis_title,
                self.typography.tick,
                self.typography.legend,
                self.typography.annotation,
            )
        ):
            raise ValueError("Font sizes must be positive finite point sizes.")
        if not self.vector and math.prod(self.pixel_dimensions) > 100_000_000:
            raise ValueError(
                "This export exceeds 100 million pixels. Reduce the dimensions or DPI."
            )


PRESETS = {
    "Manuscript – single column": (85, "mm", 9, 600),
    "Manuscript – double column": (180, "mm", 10, 600),
    "Grant figure": (165, "mm", 12, 600),
    "Presentation": (13.33, "in", 18, 300),
    "Poster": (180, "mm", 16, 300),
    "Custom": (85, "mm", 9, 600),
}


def preset_config(name: str, aspect_ratio: float = 1.5) -> ExportConfig:
    width, unit, base, dpi = PRESETS[name]
    fonts = Typography(
        base=base,
        title=base * 4 / 3,
        subtitle=base * 13 / 12,
        axis_title=base * 13 / 12,
        tick=base * 11 / 12,
        legend=base * 11 / 12,
        annotation=base * 11 / 12,
    )
    return ExportConfig(
        preset=name, width=width, unit=unit, dpi=dpi, aspect_ratio=aspect_ratio, typography=fonts
    )


@dataclass(frozen=True)
class PlotContentConfig:
    title: str | None = None
    subtitle: str = ""
    show_title: bool = True
    show_subtitle: bool = True
    show_panel_titles: bool = True
    show_panel_subtitles: bool = True
    show_x_title: bool = True
    show_y_title: bool = True
    show_z_title: bool = True
    show_x_ticks: bool | None = None
    show_y_ticks: bool | None = None
    show_color_scale: bool = True
    show_region_annotations: bool = True
    shared_axis_titles: bool = True
    x_label: str | None = None
    y_label: str | None = None
    z_label: str | None = None
    caption: str = ""
    correlation_mode: str = "Original"
    annotation_groups: tuple[str, ...] | None = None
    statistic_fields: tuple[str, ...] = ("n", "coefficient", "p", "fdr")
    statistic_precision: int = 3
    statistic_position: str = "Original"
    show_observations: bool = True
    show_regression: bool = True
    show_intervals: bool = True
    show_statistics: bool = True
    show_selected_module_annotation: bool = True
    panel_order: tuple[str, ...] | None = None
    panel_labels: bool = False
    horizontal_reference: float | None = None
    vertical_reference: float | None = None
    bar_series: tuple[str, ...] | None = None
    bar_categories: tuple[str, ...] | None = None
    bar_items: tuple[str, ...] | None = None
    bar_series_colors: tuple[tuple[str, str], ...] = ()
    bar_item_colors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Panel:
    identifier: str
    label: str
    xaxis: str = "x"
    yaxis: str = "y"
    scene: str = ""


@dataclass
class FigureArtifact:
    figure: go.Figure
    panels: tuple[Panel, ...]
    warnings: tuple[str, ...] = ()
    # Minimum canvas height in CSS pixels, including physical text and panel gaps.
    minimum_height: float = 0


Builder = Callable[[Any, Any, ExportConfig, PlotContentConfig], FigureArtifact]


def decode_arrays(value: Any) -> Any:
    """Expand Plotly 6 typed arrays without losing values or NaN positions."""
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
            if value.get("shape"):
                shape = tuple(int(v) for v in str(value["shape"]).split(","))
                array = array.reshape(shape)
            return array.tolist()
        return {k: decode_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_arrays(v) for v in value]
    return value


def restore_figure(snapshot: str) -> go.Figure:
    return go.Figure(decode_arrays(json.loads(snapshot)))


def figure_panels(figure: go.Figure) -> tuple[Panel, ...]:
    panels: list[Panel] = []
    seen: set[str] = set()
    layout = figure.layout.to_plotly_json()
    for trace in figure.data:
        scene = getattr(trace, "scene", None)
        if trace.type in {
            "scatter3d",
            "surface",
            "mesh3d",
            "cone",
            "volume",
            "isosurface",
            "streamtube",
        }:
            scene = scene or "scene"
        x, y = getattr(trace, "xaxis", None) or "x", getattr(trace, "yaxis", None) or "y"
        identifier = scene or f"{x}:{y}"
        if identifier in seen:
            continue
        seen.add(identifier)
        label = identifier
        if not scene:
            domain = layout.get(x.replace("x", "xaxis", 1), {}).get("domain", [0, 1])
            yd = layout.get(y.replace("y", "yaxis", 1), {}).get("domain", [0, 1])
            for a in layout.get("annotations", []):
                if a.get("xref", "paper") == "paper" and a.get("yref", "paper") == "paper":
                    if (
                        abs(a.get("x", -100) - sum(domain) / 2) < 0.01
                        and abs(a.get("y", -100) - yd[1]) < 0.01
                    ):
                        heading = str(a.get("text", label)).split("<br>")[0]
                        label = html.unescape(re.sub("<[^>]+>", " ", heading)).strip()
                        break
        panels.append(Panel(identifier, label, x, y, scene or ""))
    return tuple(panels)


@dataclass(frozen=True)
class PlotExportSpec:
    plot_id: str
    figure_snapshot: str
    metadata: tuple[tuple[str, str], ...] = ()
    supported_formats: tuple[str, ...] = FORMATS
    builder: Builder | None = field(default=None, repr=False, compare=False)
    data_snapshot: Any = field(default=None, repr=False, compare=False)
    analysis_snapshot: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.builder is not None:
            object.__setattr__(self, "data_snapshot", deepcopy(self.data_snapshot))
            object.__setattr__(self, "analysis_snapshot", deepcopy(self.analysis_snapshot))

    @classmethod
    def from_figure(
        cls, plot_id: str, figure: go.Figure, metadata: dict[str, Any] | None = None
    ) -> PlotExportSpec:
        return cls(
            plot_id, figure.to_json(), tuple((str(k), str(v)) for k, v in (metadata or {}).items())
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((self.figure_snapshot + repr(self.metadata)).encode()).hexdigest()

    @property
    def aspect_ratio(self) -> float:
        figure = restore_figure(self.figure_snapshot)
        panel_count = len(figure_panels(figure))
        width = figure.layout.width or (1000 if panel_count > 1 else 700)
        height = figure.layout.height or 480
        return float(width) / float(height)


def widget_key(plot_id: str, context: str, control: str) -> str:
    identity = hashlib.sha256(plot_id.encode()).hexdigest()[:12]
    return f"figure_export::{sanitize_filename(plot_id)}::{identity}::{context[:16]}::{control}"


def sanitize_filename(value: str, extension: str | None = None) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"\.(png|tiff?|svg|pdf|zip)$", "", value, flags=re.I)
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_.-")[:170] or "figure"
    if value.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = "figure_" + value
    return value + ("." + extension if extension else "")


def default_filename(spec: PlotExportSpec, config: ExportConfig) -> str:
    fields = [
        v
        for k, v in spec.metadata
        if k in {"module", "module_set", "method", "component", "outcome", "feature"}
    ]
    width = f"{convert_length(config.width, config.unit, 'mm'):.3f}".rstrip("0").rstrip(".")
    return sanitize_filename(
        "_".join([*fields, spec.plot_id, width + "mm", f"{config.dpi}dpi"]), config.format
    )


def output_summary(config: ExportConfig, mixed: bool = False) -> str:
    size = f"{config.width:g} × {config.actual_height:.2f} {config.unit}"
    if not config.vector:
        w, h = config.pixel_dimensions
        size += f" | {w} × {h} px | {config.dpi} DPI"
    else:
        size += " | " + (f"vector + raster ({config.dpi} DPI)" if mixed else "vector")
    return f"{size} | {config.format.upper()} | aspect {config.width / config.actual_height:.2f}:1"


def render_fingerprint(
    spec: PlotExportSpec, config: ExportConfig, content: PlotContentConfig
) -> str:
    from importlib.metadata import PackageNotFoundError, version

    versions = []
    for library in ("plotly", "kaleido", "Pillow", "pypdf"):
        try:
            versions.append((library, version(library)))
        except PackageNotFoundError:
            versions.append((library, "unavailable"))
    # Caller-provided builders are not memoized; identity distinguishes prepared
    # files between distinct captured callback inputs without pickling live objects.
    callback_context = id(spec) if spec.builder is not None else None
    payload = json.dumps(
        [
            "export-v2",
            versions,
            spec.fingerprint,
            callback_context,
            asdict(config),
            asdict(content),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
