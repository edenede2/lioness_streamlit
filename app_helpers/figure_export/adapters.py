"""Pure Plotly rebuilding, vector adapters, and bounded static rendering."""

from __future__ import annotations

import html
import io
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
from plotly.colors import get_colorscale, sample_colorscale

from .config import (
    CSS_DPI,
    POINTS_PER_INCH,
    ExportConfig,
    FigureArtifact,
    Panel,
    PlotContentConfig,
    PlotExportSpec,
    convert_length,
    figure_panels,
    restore_figure,
)

SAFE_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
    "#F0E442",
]
GL_TYPES = {
    "scatter3d",
    "surface",
    "mesh3d",
    "cone",
    "streamtube",
    "volume",
    "isosurface",
    "scatterpolargl",
    "splom",
    "parcoords",
}


class ExportError(ValueError):
    """A user-actionable export failure, without exposing analysis payloads."""


@lru_cache(maxsize=32)
def resolve_font(requested: str) -> str:
    # Arial is proprietary and is deliberately not bundled with the application.
    if requested == "DejaVu Sans":
        return requested
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}", requested],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0 and requested.casefold() in result.stdout.casefold().split(","):
            return requested
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "DejaVu Sans"


def correlation_metadata(figure: go.Figure) -> dict[str, Any]:
    return (
        dict((figure.layout.meta or {}).get("figure_export", {}))
        if isinstance(figure.layout.meta, dict)
        else {}
    )


def trace_role(trace: Any) -> str:
    if isinstance(trace.meta, dict) and trace.meta.get("export_role"):
        return str(trace.meta["export_role"])
    if trace.type in {"surface", "mesh3d"}:
        return "regression"
    if (
        getattr(trace, "mode", None) == "lines+markers"
        and trace.x is not None
        and all(v is None for v in trace.x)
    ):
        return "legend"
    name = str(trace.name or "").lower()
    if trace.legendgroup == "__pooled__" and getattr(trace, "mode", None) == "lines":
        return "regression"
    if any(v in name for v in ("trend", "ols:", "ols plane", "regression")):
        return "regression"
    if getattr(trace, "fill", None) in {"tonexty", "tonextx"}:
        return "interval"
    return "observations" if "markers" in str(getattr(trace, "mode", "")) else "data"


def is_mixed(figure: go.Figure) -> bool:
    return any(t.type in GL_TYPES for t in figure.data)


def _axis_key(reference: str) -> str:
    return reference[0] + "axis" + reference[1:]


def can_share_axes(figure: go.Figure, panels: tuple[Panel, ...]) -> bool:
    """Prevent matching categorical, numeric and logarithmic axes to each other."""
    if any(p.scene for p in panels):
        return False
    for direction in ("x", "y"):
        kinds = set()
        for panel in panels:
            reference = getattr(panel, direction + "axis")
            explicit = figure.layout[_axis_key(reference)].type
            if explicit:
                kinds.add(explicit)
                continue
            values = [
                v
                for t in figure.data
                if (getattr(t, direction + "axis", None) or direction) == reference
                for v in (
                    getattr(t, direction, None) if getattr(t, direction, None) is not None else []
                )
                if v is not None
            ]
            kinds.add("category" if any(isinstance(v, str) for v in values) else "linear")
        if len(kinds) > 1:
            return False
    return True


def _annotation_panel(annotation: dict, panels: tuple[Panel, ...], layout: dict) -> str | None:
    xr, yr = annotation.get("xref", "paper"), annotation.get("yref", "paper")
    for panel in panels:
        if xr.split(" ")[0] == panel.xaxis or yr.split(" ")[0] == panel.yaxis:
            return panel.identifier
        if not panel.scene and xr == yr == "paper":
            xd = layout.get(_axis_key(panel.xaxis), {}).get("domain", [0, 1])
            yd = layout.get(_axis_key(panel.yaxis), {}).get("domain", [0, 1])
            if (
                abs(annotation.get("x", -100) - sum(xd) / 2) < 0.01
                and abs(annotation.get("y", -100) - yd[1]) < 0.01
            ):
                return panel.identifier
    return None


def _select_panels(
    figure: go.Figure, panels: tuple[Panel, ...], content: PlotContentConfig, config: ExportConfig
) -> tuple[Panel, ...]:
    order = (
        content.panel_order
        if content.panel_order is not None
        else tuple(p.identifier for p in panels)
    )
    if not order or len(order) != len(set(order)) or set(order) - {p.identifier for p in panels}:
        raise ExportError("Select at least one valid panel, without duplicate panel identifiers.")
    selected = tuple(next(p for p in panels if p.identifier == identifier) for identifier in order)
    if config.style.shared_axes and not can_share_axes(figure, selected):
        raise ExportError(
            "These panels use different axis types. Use independent or original axes."
        )
    old = figure.layout.to_plotly_json()
    figure.data = tuple(
        t
        for t in figure.data
        if (
            getattr(t, "scene", None)
            or (
                "scene"
                if hasattr(t, "scene")
                else f"{getattr(t, 'xaxis', None) or 'x'}:{getattr(t, 'yaxis', None) or 'y'}"
            )
        )
        in order
    )
    annotations = []
    for a in old.get("annotations", []):
        owner = _annotation_panel(a, panels, old)
        if owner is None or owner in order:
            annotations.append(dict(a))
    figure.layout.annotations = annotations
    annotation_owners = [_annotation_panel(a, panels, old) for a in annotations]
    # Preserve unequal panel widths and shared axes in the unchanged layout.
    reflow = (
        tuple(order) != tuple(p.identifier for p in panels)
        or config.style.panel_columns > 0
        or config.style.panel_spacing is not None
    )
    if reflow and not any(p.scene for p in selected):
        original_columns = len(
            {tuple(old.get(_axis_key(p.xaxis), {}).get("domain", [0, 1])) for p in panels}
        )
        columns = min(len(selected), config.style.panel_columns or original_columns)
        rows = math.ceil(len(selected) / columns)
        gap = config.style.panel_spacing if config.style.panel_spacing is not None else 0.08
        gap_x = min(gap, 0.8 / max(1, columns - 1))
        gap_y = min(gap, 0.8 / max(1, rows - 1))
        for i, panel in enumerate(selected):
            col, row = i % columns, i // columns
            dx, dy = (1 - gap_x * (columns - 1)) / columns, (1 - gap_y * (rows - 1)) / rows
            xd = [max(0, min(1, v)) for v in (col * (dx + gap_x), col * (dx + gap_x) + dx)]
            yd = [max(0, min(1, v)) for v in (1 - row * (dy + gap_y) - dy, 1 - row * (dy + gap_y))]
            figure.layout[_axis_key(panel.xaxis)].domain = xd
            figure.layout[_axis_key(panel.yaxis)].domain = yd
            for annotation_index, a in enumerate(figure.layout.annotations):
                if (
                    annotation_owners[annotation_index] == panel.identifier
                    and a.xref == a.yref == "paper"
                ):
                    a.x, a.y = sum(xd) / 2, yd[1]
    # Remove axes belonging only to omitted panels; otherwise their ticks still render.
    keep_axes = {p.xaxis for p in selected} | {p.yaxis for p in selected}
    for p in panels:
        for reference in (p.xaxis, p.yaxis):
            if reference not in keep_axes and not p.scene:
                figure.layout[_axis_key(reference)] = None
        if p.scene and p not in selected:
            figure.layout[p.scene] = None
    figure.layout.shapes = tuple(
        s
        for s in figure.layout.shapes
        if (not s.xref or s.xref == "paper" or s.xref.split(" ")[0] in keep_axes)
        and (not s.yref or s.yref == "paper" or s.yref.split(" ")[0] in keep_axes)
    )
    for p in selected:
        if not p.scene:
            for reference in (p.xaxis, p.yaxis):
                axis = figure.layout[_axis_key(reference)]
                if axis.matches and axis.matches not in keep_axes:
                    axis.matches = None
                if config.style.shared_axes is not None:
                    first = selected[0].xaxis if reference.startswith("x") else selected[0].yaxis
                    axis.matches = (
                        first if config.style.shared_axes and reference != first else None
                    )
        if content.panel_labels and not p.scene:
            label_index = selected.index(p)
            figure.add_annotation(
                x=0,
                y=1,
                xref=p.xaxis + " domain",
                yref=p.yaxis + " domain",
                text=f"<b>{chr(65 + label_index) if label_index < 26 else label_index + 1}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                yshift=3,
                name="publication-panel-label",
            )
    return selected


def _stat_text(record: dict, method: str, content: PlotContentConfig, minimum_n: int) -> str:
    n = int(record.get("n") or 0)
    if n < minimum_n or not record.get("eligible", True):
        return f"n={n}; statistics unavailable ({html.escape(str(record.get('unavailable_reason') or 'insufficient or constant values'))})"
    parts = []
    for field in content.statistic_fields:
        if field == "n":
            parts.append(f"n={n}")
            continue
        key = {
            "coefficient": "spearman_rho" if method == "spearman" else "pearson_r",
            "p": method + "_p",
            "fdr": method + "_fdr_across_modules",
        }.get(field)
        value = record.get(key)
        label = {
            "coefficient": "ρ" if method == "spearman" else "r",
            "p": "p",
            "fdr": "module-set FDR",
        }[field]
        if value is None or not np.isfinite(float(value)):
            rendered = "NA"
        else:
            rendered = (
                f"{float(value):.{content.statistic_precision}g}"
                if field != "coefficient"
                else f"{float(value):.{content.statistic_precision}f}"
            )
        parts.append(f"{label}={rendered}")
    return "; ".join(parts)


@lru_cache(maxsize=128)
def _measurement_font(family: str, font_pixels: float):
    """Use the export font's actual metrics, including proportional glyph widths."""
    from PIL import ImageFont

    try:
        match = subprocess.run(
            ["fc-match", "-f", "%{file}", family],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        path = match.stdout.strip() if match.returncode == 0 else "DejaVuSans.ttf"
        return ImageFont.truetype(path, max(1, round(font_pixels * 4)))
    except (OSError, subprocess.TimeoutExpired):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", max(1, round(font_pixels * 4)))
        except OSError:
            return None


def _text_width(text: str, font_pixels: float, family: str) -> float:
    plain = html.unescape(re.sub("<[^>]+>", "", str(text)))
    font = _measurement_font(family, font_pixels)
    width = font.getlength(plain) / 4 if font else len(plain) * font_pixels * 0.6
    return width * (1.08 if "<b>" in str(text) else 1)


def _wrap_annotation(
    text: str, available_pixels: float, font_pixels: float, family: str = "DejaVu Sans"
) -> str:
    """Wrap display text at physical panel width, preserving HTML and all words."""
    available_pixels = max(font_pixels * 1.5, available_pixels)
    lines = []
    for line in re.split(r"<br\s*/?>", str(text), flags=re.I):
        # Spaces inside a span's style attribute are not word boundaries.
        words, word = [], ""
        for token in re.findall(r"<[^>]+>|[^\s<]+|\s+", line):
            if token.isspace():
                if word:
                    words.append(word)
                    word = ""
            else:
                word += token
        if word:
            words.append(word)
        current = []
        for word in words:
            if (
                current
                and _text_width(" ".join([*current, word]), font_pixels, family) > available_pixels
            ):
                lines.append(" ".join(current))
                current = []
            if _text_width(word, font_pixels, family) > available_pixels:
                # Long pathway names and axis-label tokens must also fit. This
                # changes line breaks only, never the underlying label/value.
                tags = re.findall(r"<[^>]+>", word)
                opening = "".join(t for t in tags if not t.startswith("</"))
                closing = "".join(t for t in tags if t.startswith("</"))
                measured_font = font_pixels * (1.08 if any("b>" in t for t in tags) else 1)
                chunk = ""
                first = True
                for char in html.unescape(re.sub(r"<[^>]+>", "", word)):
                    if (
                        chunk
                        and _text_width(chunk + char, measured_font, family) > available_pixels
                    ):
                        lines.append((opening if first else "") + html.escape(chunk))
                        first = False
                        chunk = ""
                    chunk += char
                word = (opening if first else "") + html.escape(chunk) + closing
            current.append(word)
        lines.append(" ".join(current))
    return "<br>".join(lines)


def panel_title_annotations(figure: go.Figure, panels: tuple[Panel, ...] | None = None) -> tuple:
    """Identify builder-tagged headings and native make_subplots titles."""
    panels = figure_panels(figure) if panels is None else panels
    layout = figure.layout.to_plotly_json()
    return tuple(
        a
        for a in figure.layout.annotations
        if (a.name or "").startswith("publication-panel-title")
        or (
            a.xref == a.yref == "paper"
            and not a.showarrow
            and not (a.name or "").startswith(("publication-statistics", "publication-panel-label"))
            and _annotation_panel(a.to_plotly_json(), panels, layout) is not None
        )
    )


def _physical_panel_layout(
    figure: go.Figure,
    panels: tuple[Panel, ...],
    config: ExportConfig,
    content: PlotContentConfig,
    title_height: float,
    legend_height: float,
    bottom_legend_height: float,
    warnings: list[str],
) -> float:
    """Reserve physical text bands for every row and column of a Cartesian grid.

    Plotly's automargin only protects the outside of a figure. Internal subplot
    gaps must explicitly include the lower row's header and the upper row's
    axis labels. Unequal panel sizes keep their relative proportions.
    """
    if not config.style.tight or not panels or any(p.scene for p in panels):
        return 0
    layout = figure.layout.to_plotly_json()
    xdomains = sorted(
        {tuple(layout.get(_axis_key(p.xaxis), {}).get("domain", (0, 1))) for p in panels}
    )
    ydomains = sorted(
        {tuple(layout.get(_axis_key(p.yaxis), {}).get("domain", (0, 1))) for p in panels},
        reverse=True,
    )
    # Inset/overlaid plots are not a regular subplot grid; retain their domains.
    if any(a[1] > b[0] + 1e-6 for a, b in zip(xdomains, xdomains[1:])) or any(
        lower[1] > upper[0] + 1e-6 for upper, lower in zip(ydomains, ydomains[1:])
    ):
        return 0
    fonts, pt = config.typography, CSS_DPI / POINTS_PER_INCH
    family = resolve_font(fonts.family)
    width, height = figure.layout.width, figure.layout.height
    margin = figure.layout.margin.to_plotly_json()
    owners = [
        _annotation_panel(a.to_plotly_json(), panels, layout) for a in figure.layout.annotations
    ]
    headings = panel_title_annotations(figure, panels)
    axis_line = fonts.axis_title * pt * 1.3
    stat_line = max(fonts.annotation * 1.3, fonts.annotation + fonts.annotation_spacing) * pt
    subtitle_line = fonts.subtitle * pt * 1.3
    panel_minimum = max(fonts.axis_title * pt * 8, width * 0.15)
    shared_titles = {}
    if content.shared_axis_titles and len(panels) > 1:
        for direction in ("x", "y"):
            texts = {
                str(figure.layout[_axis_key(getattr(p, direction + "axis"))].title.text or "")
                for p in panels
            }
            if len(texts) == 1 and next(iter(texts)):
                shared_titles[direction] = next(iter(texts))
                for p in panels:
                    figure.layout[_axis_key(getattr(p, direction + "axis"))].title.text = ""
    shared_y = _wrap_annotation(
        shared_titles.get("y", ""),
        panel_minimum * max(1, len(ydomains)),
        fonts.axis_title * pt,
        family,
    )
    shared_y_band = axis_line * (shared_y.count("<br>") + 1) + 4 * pt if shared_y else 0
    # Two lines of a vertical title must fit alongside each data panel.
    panel_minimum = max(
        panel_minimum,
        *(
            _text_width(
                figure.layout[_axis_key(p.yaxis)].title.text or "", fonts.axis_title * pt, family
            )
            / 2
            + 8 * pt
            for p in panels
        ),
    )
    left_bands, right_bands = [0.0] * len(xdomains), [0.0] * len(xdomains)
    for p in panels:
        axis = figure.layout[_axis_key(p.yaxis)]
        axis.title.text = _wrap_annotation(
            axis.title.text or "", panel_minimum, fonts.axis_title * pt, family
        )
        axis.title.standoff = 4 * pt
        tick_width = 0.0
        if axis.showticklabels is not False:
            labels = list(axis.ticktext) if axis.ticktext is not None else []
            if not labels:
                values = [
                    v
                    for t in figure.data
                    if (getattr(t, "yaxis", None) or "y") == p.yaxis
                    for v in (getattr(t, "y", None) if getattr(t, "y", None) is not None else [])
                    if v is not None
                ]
                numeric = [
                    float(v)
                    for v in values
                    if isinstance(v, (int, float, np.number)) and np.isfinite(v)
                ]
                if numeric:
                    extremes = (min(numeric), max(numeric), 0)
                    fmt = axis.tickformat or ".3g"
                    try:
                        labels = [format(v, fmt) for v in extremes]
                    except ValueError:
                        labels = [format(v, ".3g") for v in extremes]
                else:
                    labels = [str(v) for v in values] or ["0.00"]
            tick_width = (
                max((_text_width(str(v), fonts.tick * pt, family) for v in labels), default=0)
                + 6 * pt
            )
        title_band = (
            axis_line * (str(axis.title.text).count("<br>") + 1) + 4 * pt if axis.title.text else 0
        )
        col = xdomains.index(tuple(layout.get(_axis_key(p.xaxis), {}).get("domain", (0, 1))))
        bands = right_bands if axis.side == "right" else left_bands
        bands[col] = max(bands[col], tick_width + title_band)
    margin["l"] = max(margin.get("l", 0), left_bands[0] + shared_y_band + 6 * pt)
    margin["r"] = max(margin.get("r", 0), right_bands[-1] + 6 * pt)
    canvas_width = max(20, width - margin["l"] - margin["r"])
    shared_x = _wrap_annotation(
        shared_titles.get("x", ""), canvas_width, fonts.axis_title * pt, family
    )
    shared_x_band = axis_line * (shared_x.count("<br>") + 1) + 8 * pt if shared_x else 0
    gaps_x = [
        max(
            right_bands[i] + left_bands[i + 1] + 8 * pt,
            (config.style.panel_spacing or 0) * canvas_width,
        )
        for i in range(len(xdomains) - 1)
    ]
    if sum(gaps_x) > canvas_width * 0.75:
        warnings.append(
            "The panels are too narrow for these labels. Increase width or use fewer panel columns."
        )
        scale = canvas_width * 0.75 / sum(gaps_x)
        gaps_x = [g * scale for g in gaps_x]
    weights_x = [d[1] - d[0] for d in xdomains]
    available_x = canvas_width - sum(gaps_x)
    minimum_width = fonts.base * pt * 6
    if (
        config.style.panel_columns == 0
        and len(xdomains) > 1
        and max(weights_x) - min(weights_x) < 1e-6
        and available_x / len(xdomains) < minimum_width
    ):
        gap = max(gaps_x, default=0)
        columns = max(1, min(len(xdomains) - 1, int((canvas_width + gap) / (minimum_width + gap))))
        # Keep observations, panel order and statistics; only the grid changes.
        for p in panels:
            for direction in ("x", "y"):
                key = _axis_key(getattr(p, direction + "axis"))
                figure.layout[key].title.text = layout.get(key, {}).get("title", {}).get("text", "")
        fitted_layout = replace(config, style=replace(config.style, panel_columns=columns))
        _select_panels(figure, panels, replace(content, panel_labels=False), fitted_layout)
        warnings.append(
            f"Using {columns} panel column(s) at this width to keep text legible. Customize panel layout to set the columns explicitly."
        )
        return _physical_panel_layout(
            figure,
            panels,
            fitted_layout,
            content,
            title_height,
            legend_height,
            bottom_legend_height,
            warnings,
        )
    new_x, position = {}, 0.0
    for i, domain in enumerate(xdomains):
        length = available_x * weights_x[i] / sum(weights_x)
        new_x[domain] = (
            max(0, min(1, position / canvas_width)),
            max(0, min(1, (position + length) / canvas_width)),
        )
        position += length + (gaps_x[i] if i < len(gaps_x) else 0)
    headers, footers = [0.0] * len(ydomains), [0.0] * len(ydomains)
    for p in panels:
        xd = tuple(layout.get(_axis_key(p.xaxis), {}).get("domain", (0, 1)))
        yd = tuple(layout.get(_axis_key(p.yaxis), {}).get("domain", (0, 1)))
        row = ydomains.index(yd)
        panel_width = (new_x[xd][1] - new_x[xd][0]) * canvas_width
        axis = figure.layout[_axis_key(p.xaxis)]
        axis.title.text = _wrap_annotation(
            axis.title.text or "", max(20, panel_width - 8 * pt), fonts.axis_title * pt, family
        )
        axis.title.standoff = 4 * pt
        ticks = fonts.tick * pt * 1.6 + 6 * pt if axis.showticklabels is not False else 0
        axis_band = (
            axis_line * (str(axis.title.text).count("<br>") + 1) + 8 * pt if axis.title.text else 0
        )
        if axis.tickangle:
            labels = list(axis.ticktext) if axis.ticktext is not None else []
            tick_extent = max(
                (_text_width(str(v), fonts.tick * pt, family) for v in labels), default=ticks
            )
            ticks = max(
                ticks, abs(math.sin(math.radians(axis.tickangle))) * tick_extent + fonts.tick * pt
            )
        panel_footer = ticks + axis_band
        owned = [a for a, owner in zip(figure.layout.annotations, owners) if owner == p.identifier]
        stats = [
            a
            for a in owned
            if (a.name or "").startswith("publication-statistics") and a.visible is not False
        ]
        stat_height = 0.0
        for a in stats:
            a.text = _wrap_annotation(
                a.text, max(20, panel_width - 8 * pt), fonts.annotation * pt, family
            )
            block = (str(a.text).count("<br>") + 1) * stat_line + 6 * pt
            if content.statistic_position == "Original":
                a.yshift = fonts.annotation_spacing * pt + stat_height
                stat_height += block
            elif content.statistic_position == "Below panel":
                a.yshift = -panel_footer - 4 * pt
                panel_footer += block
        footers[row] = max(footers[row], panel_footer)
        heading_height = 0.0
        for a in owned:
            if any(a is heading for heading in headings) and a.visible is not False:
                a.text = _wrap_annotation(
                    a.text, max(20, panel_width - 8 * pt), fonts.subtitle * pt, family
                )
                a.yanchor = "bottom"
                a.yshift = stat_height + 6 * pt
                heading_height = max(
                    heading_height, (str(a.text).count("<br>") + 1) * subtitle_line + 8 * pt
                )
        labels = [a for a in owned if a.name == "publication-panel-label"]
        for a in labels:
            a.yshift = stat_height + heading_height + 6 * pt
            a.yanchor = "bottom"
        label_height = subtitle_line + 6 * pt if labels else 0
        headers[row] = max(headers[row], stat_height + heading_height + label_height)
    for annotation, owner in zip(figure.layout.annotations, owners):
        if annotation.name == "publication-panel-label" and owner is not None:
            panel = next(p for p in panels if p.identifier == owner)
            row = ydomains.index(
                tuple(layout.get(_axis_key(panel.yaxis), {}).get("domain", (0, 1)))
            )
            annotation.yshift = headers[row] - subtitle_line
    margin["t"] = max(
        config.style.margin_top * pt, title_height + legend_height + headers[0] + 12 * pt
    )
    caption_height = (
        (str(content.caption).count("\n") + 1) * stat_line + 8 * pt if content.caption else 0
    )
    margin["b"] = max(
        config.style.margin_bottom * pt,
        footers[-1] + shared_x_band + bottom_legend_height + caption_height + 12 * pt,
    )
    if content.caption:
        caption = figure.layout.annotations[-1]  # Added after the source annotations.
        caption.yshift = -footers[-1] - shared_x_band - bottom_legend_height - 8 * pt
    if bottom_legend_height:
        figure.layout.legend.y = -(footers[-1] + shared_x_band + 6 * pt) / max(
            20, height - margin["t"] - margin["b"]
        )
    gaps_y = [footers[i] + headers[i + 1] + 12 * pt for i in range(len(ydomains) - 1)]
    weights_y = [d[1] - d[0] for d in ydomains]
    required_height = (
        margin["t"] + margin["b"] + sum(gaps_y) + panel_minimum * sum(weights_y) / min(weights_y)
    )
    canvas_height = max(20, height - margin["t"] - margin["b"])
    gaps_y = [max(g, (config.style.panel_spacing or 0) * canvas_height) for g in gaps_y]
    required_height = max(
        required_height,
        margin["t"] + margin["b"] + sum(gaps_y) + panel_minimum * sum(weights_y) / min(weights_y),
    )
    if height + 1 < required_height and config.height is not None:
        warnings.append(
            "This manual height cannot fit the panel titles and axis labels without crowding. Use automatic height, increase height, or hide unwanted titles/subtitles."
        )
    # An initial automatic-size pass may be too short. Leave a positive data
    # domain while reporting the full required height to fit_automatic_config.
    gap_scale = min(1, canvas_height * 0.85 / max(1, sum(gaps_y)))
    actual_gaps_y = [g * gap_scale for g in gaps_y]
    available_y = canvas_height - sum(actual_gaps_y)
    new_y, position = {}, canvas_height
    for i, domain in enumerate(ydomains):
        length = available_y * weights_y[i] / sum(weights_y)
        new_y[domain] = (
            max(0, min(1, (position - length) / canvas_height)),
            max(0, min(1, position / canvas_height)),
        )
        position -= length + (actual_gaps_y[i] if i < len(actual_gaps_y) else 0)
    for p in panels:
        xd = tuple(layout.get(_axis_key(p.xaxis), {}).get("domain", (0, 1)))
        yd = tuple(layout.get(_axis_key(p.yaxis), {}).get("domain", (0, 1)))
        figure.layout[_axis_key(p.xaxis)].domain = new_x[xd]
        figure.layout[_axis_key(p.yaxis)].domain = new_y[yd]
        for a, owner in zip(figure.layout.annotations, owners):
            if owner == p.identifier and a.xref == a.yref == "paper":
                a.x, a.y = sum(new_x[xd]) / 2, new_y[yd][1]
    figure.update_layout(margin=margin)
    shared_font = dict(size=fonts.axis_title * pt, family=family, color=fonts.color)
    if shared_x:
        figure.add_annotation(
            name="publication-shared-x-title",
            text=shared_x,
            x=0.5,
            y=0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            yshift=-footers[-1] - 6 * pt,
            showarrow=False,
            font=shared_font,
        )
    if shared_y:
        figure.add_annotation(
            name="publication-shared-y-title",
            text=shared_y,
            x=0,
            y=0.5,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            xshift=-left_bands[0] - shared_y_band / 2 - 6 * pt,
            textangle=-90,
            showarrow=False,
            font=shared_font,
        )
    return required_height


def _correlation_annotations(figure: go.Figure, content: PlotContentConfig) -> None:
    metadata = correlation_metadata(figure)
    if not metadata.get("correlation_records"):
        return
    mode = content.correlation_mode
    records = metadata["correlation_records"]
    method = metadata["correlation_method"]
    groups = (
        content.annotation_groups
        if content.annotation_groups is not None
        else tuple(str(v) for v in metadata["grouping_levels"])
    )
    labels = metadata["grouping_labels"]
    pooled = mode in {"Both", "Pooled only"} or (mode == "Original" and metadata["show_pooled"])
    grouped = mode in {"Both", "Group-specific only", "Original"}
    if pooled and not any(str(r.get("grouping_level")) == "__pooled__" for r in records):
        raise ExportError(
            "Pooled statistics are unavailable for this displayed analysis. Enable pooled results on the main page first."
        )
    for index, _component in enumerate(metadata["components"]):
        name = f"publication-statistics-{index}"
        if not any(a.name == name for a in figure.layout.annotations):
            suffix = str(index + 1) if index else ""
            figure.add_annotation(
                name=name,
                x=0.01,
                y=1,
                xref=f"x{suffix} domain",
                yref=f"y{suffix} domain",
                showarrow=False,
                align="left",
                xanchor="left",
                yanchor="bottom",
                text="",
            )
    for annotation in figure.layout.annotations:
        name = annotation.name or ""
        if not name.startswith("publication-statistics-"):
            continue
        index = int(name.rsplit("-", 1)[1])
        component = metadata["components"][index]
        lines = []
        for r in records:
            if str(r.get("component")) != str(component):
                continue
            level = str(r.get("grouping_level"))
            if (level == "__pooled__" and pooled) or (level in groups and grouped):
                label = (
                    metadata["pooled_label"] if level == "__pooled__" else labels.get(level, level)
                )
                text = _stat_text(r, method, content, metadata["minimum_group_n"])
                if text:
                    lines.append((level == "__pooled__", f"<b>{html.escape(label)}</b>: {text}"))
        lines.sort(key=lambda item: not item[0])
        annotation.text = "<br>".join(v for _, v in lines)
        annotation.visible = bool(lines) and mode != "None" and content.show_statistics


def build_figure(
    data: str, analysis_results: Any, export_config: ExportConfig, content_config: PlotContentConfig
) -> FigureArtifact:
    """Rebuild from an immutable figure/data snapshot; never recompute analyses."""
    export_config.validate()
    figure = restore_figure(data)
    panels = figure_panels(figure)
    content, config, fonts, style = (
        content_config,
        export_config,
        export_config.typography,
        export_config.style,
    )
    _correlation_annotations(figure, content)
    selected = _select_panels(figure, panels, content, config) if panels else ()
    warnings = []
    family = resolve_font(fonts.family)
    if family != fonts.family:
        warnings.append(
            f"{fonts.family} is unavailable on this server; using {family} in the preview and file."
        )
    pt = CSS_DPI / POINTS_PER_INCH
    width, height = (v * CSS_DPI for v in config.inches)
    background = "rgba(0,0,0,0)" if style.transparent else style.background
    # User labels are plain text. Original builder-generated markup is preserved.
    title = (
        html.escape(content.title)
        if content.title is not None
        else str(figure.layout.title.text or "")
    )
    title_parts = title.split("<br>", 1)
    title = title_parts[0] if content.show_title else ""
    if content.show_subtitle and len(title_parts) > 1:
        title += ("<br>" if title else "") + title_parts[1]
    if not content.show_subtitle:
        title = re.sub(r"<sup>.*?</sup>", "", title, flags=re.S)
    title = re.sub(r"</?b>", "", title, flags=re.I)
    if style.tight:
        title = _wrap_annotation(
            title, width * 0.94, fonts.title * pt * (1.08 if fonts.bold_title else 1), family
        )
    title = re.sub(
        r"<sup>(.*?)</sup>",
        lambda m: f"<span style='font-size:{fonts.subtitle * pt:g}px'>{m.group(1)}</span>",
        title,
        flags=re.S,
    )
    if fonts.bold_title and title:
        title = "<b>" + title + "</b>"
    if content.subtitle and content.show_subtitle:
        subtitle = html.escape(content.subtitle)
        if style.tight:
            subtitle = _wrap_annotation(subtitle, width * 0.94, fonts.subtitle * pt, family)
        title += (
            "<br>" if title else ""
        ) + f"<span style='font-size:{fonts.subtitle * pt:g}px'>{subtitle}</span>"
    figure.update_layout(
        width=width,
        height=height,
        autosize=False,
        font=dict(family=family, size=fonts.base * pt, color=fonts.color),
        title=dict(
            text=title,
            font=dict(size=fonts.title * pt, family=family, color=fonts.color),
            y=max(0.5, 1 - fonts.title * pt / height),
            yanchor="top",
            yref="container",
            x=0.02,
            xanchor="left",
            xref="container",
            pad=dict(t=0),
            automargin=True,
        ),
        paper_bgcolor=background,
        plot_bgcolor=background,
        showlegend=style.legend_visible,
    )
    headings = panel_title_annotations(figure, selected)
    for annotation in headings:
        parts = str(annotation.text or "").split("<br>", 1)
        heading = parts[0] if content.show_panel_titles else ""
        subtitle = (
            parts[1].replace("<br>", " ") if content.show_panel_subtitles and len(parts) > 1 else ""
        )
        annotation.text = heading + ("<br>" if heading and subtitle else "") + subtitle
        annotation.visible = bool(annotation.text) and annotation.visible is not False
    margin = {
        "l": style.margin_left * pt,
        "r": style.margin_right * pt,
        "t": style.margin_top * pt,
        "b": style.margin_bottom * pt,
    }
    outside_legend = style.legend_visible and (
        style.legend_position == "Top"
        or (
            style.legend_position == "Original"
            and figure.layout.legend.y is not None
            and figure.layout.legend.y > 1
        )
    )
    bottom_legend = style.legend_visible and (
        style.legend_position == "Bottom"
        or (
            style.legend_position == "Original"
            and figure.layout.legend.y is not None
            and figure.layout.legend.y < 0
        )
    )
    legend_items = [
        t for t in figure.data if t.showlegend is not False and t.type not in {"heatmap", "surface"}
    ]
    horizontal_legend = style.legend_orientation == "Horizontal" or (
        style.legend_orientation == "Original" and figure.layout.legend.orientation == "h"
    )
    legend_width = sum(
        len(re.sub("<[^>]+>", "", str(t.name or "trace"))) * fonts.legend * pt * 0.55 + 40
        for t in legend_items
    )
    legend_rows = (
        max(1, math.ceil(legend_width / max(40, width - margin["l"] - margin["r"])))
        if horizontal_legend
        else max(1, len(legend_items))
    )
    legend_height = (
        fonts.legend * (1.6 * legend_rows + (1.4 if figure.layout.legend.title.text else 0)) * pt
    )
    bottom_legend_row = legend_height if bottom_legend else 0
    annotation_line_height = (
        max(fonts.annotation * 1.3, fonts.annotation + fonts.annotation_spacing) * pt
    )
    title_row = fonts.title * (1.35 * (title.count("<br>") + 1) + 1) * pt if title else 0
    legend_row = legend_height if outside_legend else 0
    if style.tight:
        max_lines = max(
            (
                str(a.text).count("<br>") + 1
                for a in figure.layout.annotations
                if a.visible is not False and (a.name or "").startswith("publication-statistics")
            ),
            default=0,
        )
        if content.statistic_position == "Inside top-left":
            max_lines = 0
        margin["t"] = max(
            margin["t"],
            title_row
            + legend_row
            + annotation_line_height * max_lines
            + fonts.subtitle * 1.4 * pt
            + (16 if max_lines else 8) * pt,
        )
        if content.caption:
            margin["b"] += fonts.annotation * 2 * pt
        if bottom_legend:
            margin["b"] = max(margin["b"], fonts.axis_title * 3 * pt + bottom_legend_row + 8 * pt)
            if content.caption:
                margin["b"] += fonts.annotation * 2 * pt
    figure.update_layout(
        margin={**margin, "pad": 3 * pt}, legend=dict(font=dict(size=fonts.legend * pt))
    )
    figure.layout.legend.title.font.update(size=fonts.legend * pt, family=family, color=fonts.color)
    if style.legend_title is not None:
        figure.layout.legend.title = dict(
            text=html.escape(style.legend_title), font=dict(size=fonts.legend * pt)
        )
    if style.legend_orientation != "Original":
        figure.layout.legend.orientation = "h" if style.legend_orientation == "Horizontal" else "v"
    positions = {
        "Top": dict(x=0.5, y=1.02, xanchor="center", yanchor="bottom"),
        "Bottom": dict(x=0.5, y=-0.18, xanchor="center", yanchor="top"),
        "Right": dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        "Inside top-right": dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    }
    plot_height = max(20, height - margin["t"] - margin["b"])
    if bottom_legend:
        figure.layout.legend.update(
            y=-fonts.axis_title * 3 * pt / plot_height, yanchor="top", yref="paper"
        )
        if style.legend_position == "Bottom":
            figure.layout.legend.update(x=0.5, xanchor="center")
    elif outside_legend:
        # Reserve a row above panel titles/statistics, below the figure title.
        figure.layout.legend.update(
            x=0.5,
            xanchor="center",
            yanchor="top",
            yref="container",
            y=max(0, 1 - title_row / height),
        )
    elif style.legend_position in positions:
        figure.layout.legend.update(positions[style.legend_position])
    axis_style = dict(
        title_font=dict(size=fonts.axis_title * pt, family=family, color=fonts.color),
        tickfont=dict(size=fonts.tick * pt, family=family, color=fonts.color),
        showgrid=style.grid,
        gridcolor=f"rgba(100,100,100,{style.grid_opacity})",
        showline=style.axis_lines,
        linewidth=style.axis_width * pt,
        linecolor=fonts.color,
        automargin=True,
    )
    for panel in selected:
        axis_pairs = [("x", panel.xaxis), ("y", panel.yaxis)]
        if panel.scene:
            if style.tight:
                # A native 3D viewport clips at its own boundaries. Widen only
                # the export camera, preserving orientation, ranges and data.
                camera = figure.layout[panel.scene].camera
                eye = camera.eye.to_plotly_json()
                camera.eye = {axis: eye.get(axis, 1.25) * 1.4 for axis in ("x", "y", "z")}
            for label, value in (
                ("x", content.x_label),
                ("y", content.y_label),
                ("z", content.z_label),
            ):
                axis = figure.layout[panel.scene][label + "axis"]
                axis.update({k: v for k, v in axis_style.items() if k != "automargin"})
                axis.backgroundcolor = background
                if style.tick_angle is not None:
                    axis.tickangle = style.tick_angle
                if style.number_format != "Original":
                    axis.tickformat = (
                        f".{style.decimals}{'e' if style.number_format == 'Scientific' else 'f'}"
                    )
                if value is not None:
                    axis.title.text = html.escape(value)
                if not getattr(content, "show_" + label + "_title"):
                    axis.title.text = ""
            continue
        for direction, reference in axis_pairs:
            axis = figure.layout[_axis_key(reference)]
            axis.update(axis_style)
            label = content.x_label if direction == "x" else content.y_label
            if label is not None:
                axis.title.text = html.escape(label)
            if not getattr(content, "show_" + direction + "_title"):
                axis.title.text = ""
            if style.tick_angle is not None:
                axis.tickangle = style.tick_angle
            if style.number_format != "Original":
                axis.tickformat = (
                    f".{style.decimals}{'e' if style.number_format == 'Scientific' else 'f'}"
                )
    color_groups: dict[str, str] = {}

    def style_colorbar(colorbar):
        colorbar.tickfont.update(size=fonts.tick * pt, family=family, color=fonts.color)
        colorbar.title.font.update(size=fonts.axis_title * pt, family=family, color=fonts.color)

    for layout_key in figure.layout:
        if layout_key.startswith("coloraxis") and figure.layout[layout_key].to_plotly_json():
            style_colorbar(figure.layout[layout_key].colorbar)
            if not content.show_color_scale:
                figure.layout[layout_key].showscale = False
    for trace in figure.data:
        role = trace_role(trace)
        if not content.show_color_scale and hasattr(trace, "showscale"):
            trace.showscale = False
        if role == "observations" and not content.show_observations:
            trace.visible = False
        if role == "regression" and not content.show_regression:
            trace.visible = False
        if role == "interval" and not content.show_intervals:
            trace.visible = False
        # Error bars retain their exact supplied intervals.
        for attr in ("error_x", "error_y", "error_z"):
            error = getattr(trace, attr, None)
            if error and (error.array is not None or error.value is not None):
                error.visible = content.show_intervals
        group = str(trace.legendgroup or trace.name or "")
        if group not in color_groups:
            color_groups[group] = SAFE_COLORS[len(color_groups) % len(SAFE_COLORS)]
        if hasattr(trace, "textfont"):
            trace.textfont.update(size=fonts.annotation * pt, family=family)
        if hasattr(trace, "colorbar"):
            style_colorbar(trace.colorbar)
        if hasattr(trace, "marker"):
            marker = trace.marker
            if not content.show_color_scale and hasattr(marker, "showscale"):
                marker.showscale = False
            if hasattr(marker, "colorbar") and (
                marker.colorbar.to_plotly_json() or marker.showscale is True
            ):
                style_colorbar(marker.colorbar)
            if trace.type in {"box", "violin"}:
                if not content.show_observations:
                    if trace.type == "box":
                        trace.boxpoints = False
                    else:
                        trace.points = False
                marker.size, marker.opacity = style.point_size * pt, style.point_opacity
                if style.point_shape != "Original":
                    marker.symbol = style.point_shape
            if "markers" in str(getattr(trace, "mode", "")):
                marker.size, marker.opacity = style.point_size * pt, style.point_opacity
                if style.point_shape != "Original":
                    # Scatter3d has a smaller supported symbol set.
                    symbol = style.point_shape
                    if trace.type == "scatter3d" and symbol not in {
                        "circle",
                        "circle-open",
                        "cross",
                        "diamond",
                        "diamond-open",
                        "square",
                        "square-open",
                        "x",
                    }:
                        symbol = "circle"
                    marker.symbol = symbol
                if style.point_encoding == "One color":
                    marker.color, marker.coloraxis = style.single_color, None
                elif style.point_encoding in {"Group color", "Group color and shape"}:
                    marker.color, marker.coloraxis = color_groups[group], None
                elif style.point_encoding == "Group shape":
                    marker.color, marker.coloraxis = style.single_color, None
                if style.point_encoding in {"Group shape", "Group color and shape"}:
                    marker.symbol = ["circle", "diamond", "square", "cross", "x"][
                        list(color_groups).index(group) % 5
                    ]
            if (
                style.palette == "Color-blind safe"
                and not getattr(marker, "coloraxis", None)
                and isinstance(marker.color, str)
            ):
                marker.color = color_groups[group]
        if hasattr(trace, "line") and trace.type not in {"surface", "mesh3d"}:
            trace.line.width = (
                style.regression_width if role == "regression" else style.line_width
            ) * pt
            dash = style.regression_dash if role == "regression" else style.line_dash
            if dash != "Original" and hasattr(trace.line, "dash"):
                trace.line.dash = dash
            if role == "regression" and style.regression_color != "Original":
                trace.line.color = style.regression_color
            elif style.palette == "Color-blind safe":
                trace.line.color = color_groups[group]
        if role == "interval":
            trace.opacity = style.ci_opacity
        elif "lines" in str(getattr(trace, "mode", "")):
            trace.opacity = style.line_opacity
        if style.palette in {"Viridis", "Cividis", "Color-blind safe"}:
            continuous_palette = "Cividis" if style.palette == "Color-blind safe" else style.palette
            if trace.type == "heatmap":
                trace.colorscale = continuous_palette
            elif hasattr(trace, "marker") and getattr(trace.marker, "coloraxis", None):
                figure.layout[trace.marker.coloraxis].colorscale = continuous_palette
    for annotation in figure.layout.annotations:
        statistical = (annotation.name or "").startswith("publication-statistics")
        annotation.font.update(
            size=(fonts.annotation if statistical else fonts.subtitle) * pt,
            family=family,
            color=fonts.color,
        )
        if statistical:
            if not content.show_statistics:
                annotation.visible = False
            if content.statistic_position != "Original":
                annotation.y = 1 if content.statistic_position == "Inside top-left" else 0
                annotation.yanchor = "top"
                annotation.yshift = (
                    -3
                    if content.statistic_position == "Inside top-left"
                    else -fonts.axis_title * 2 * pt
                )
            else:
                annotation.yshift = fonts.annotation_spacing * pt
        elif any(annotation is heading for heading in headings):
            # Association titles must sit above their statistics block at final font size.
            statistics = next(
                (
                    a
                    for a in figure.layout.annotations
                    if (a.name or "").startswith("publication-statistics")
                    and _annotation_panel(
                        a.to_plotly_json(), panels, figure.layout.to_plotly_json()
                    )
                    == _annotation_panel(
                        annotation.to_plotly_json(), panels, figure.layout.to_plotly_json()
                    )
                ),
                None,
            )
            count = (
                str(statistics.text).count("<br>") + 1
                if statistics is not None
                and statistics.visible is not False
                and content.statistic_position == "Original"
                else 0
            )
            annotation.yshift = count * annotation_line_height + 7 * pt
    if content.caption:
        figure.add_annotation(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            yshift=-fonts.axis_title * 3 * pt - bottom_legend_row,
            text=html.escape(content.caption).replace("\n", "<br>"),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(size=fonts.annotation * pt, family=family, color=fonts.color),
        )
    for panel in selected:
        if panel.scene:
            continue
        if content.horizontal_reference is not None:
            figure.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=content.horizontal_reference,
                y1=content.horizontal_reference,
                xref=panel.xaxis + " domain",
                yref=panel.yaxis,
                line=dict(color=fonts.color, width=style.axis_width * pt, dash="dash"),
            )
        if content.vertical_reference is not None:
            figure.add_shape(
                type="line",
                y0=0,
                y1=1,
                x0=content.vertical_reference,
                x1=content.vertical_reference,
                yref=panel.yaxis + " domain",
                xref=panel.xaxis,
                line=dict(color=fonts.color, width=style.axis_width * pt, dash="dash"),
            )
    minimum_height = _physical_panel_layout(
        figure, selected, config, content, title_row, legend_row, bottom_legend_row, warnings
    )
    margin = figure.layout.margin.to_plotly_json()
    if min(fonts.tick, fonts.annotation, fonts.legend) < 7:
        warnings.append("Some text is smaller than 7 pt at the final physical size.")
    if margin["l"] + margin["r"] > width * 0.6 or (
        margin["t"] + margin["b"] > height * 0.65
        and (not minimum_height or height + 1 < minimum_height)
    ):
        warnings.append(
            "Labels or annotations may be clipped: increase the figure size, select fewer panels, or reduce margins/text."
        )
    if not config.vector and math.prod(config.pixel_dimensions) > 40_000_000:
        warnings.append("This is an unusually large raster export (over 40 million pixels).")
    if config.vector and is_mixed(figure):
        warnings.append(
            "This vector container includes unavoidable rasterized 3D/WebGL components at the selected DPI."
        )
    if is_mixed(figure) and height < width:
        warnings.append(
            "Wide 3D layouts may clip axis labels. Increase the height or reduce the tick-label font size if the preview is clipped."
        )
    if (
        config.vector
        and sum(np.asarray(t.z).size for t in figure.data if t.type == "heatmap") > 5000
    ):
        warnings.append(
            "This dense vector heatmap may produce a large file. No cells will be aggregated or omitted."
        )
    return FigureArtifact(figure, selected, tuple(warnings), minimum_height)


def _edges(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    return np.r_[
        values[0] - (values[1] - values[0]) / 2,
        (values[:-1] + values[1:]) / 2,
        values[-1] + (values[-1] - values[-2]) / 2,
    ]


def _heatmap_contrast(color: str) -> str:
    """Match Plotly's tinycolor isDark/contrast rule for automatic cell text."""
    if color.startswith("rgb"):
        channels = [float(v) for v in re.findall(r"[\d.]+", color)]
        rgb, alpha = channels[:3], channels[3] if len(channels) > 3 else 1.0
    else:
        from PIL import ImageColor

        channels = ImageColor.getcolor(color, "RGBA")
        rgb, alpha = channels[:3], channels[3] / 255
    rgb = [v * alpha + 255 * (1 - alpha) for v in rgb]
    brightness = sum(v * w for v, w in zip(rgb, (299, 587, 114))) / 1000
    return "#fff" if brightness < 128 else "#444"


def vectorize_figure(source: go.Figure) -> go.Figure:
    """Replace export-only WebGL scatters and heatmap cells with vector primitives."""
    figure = restore_figure(source.to_json())
    category_axes: dict[str, list] = {}
    new_traces = []
    cells = []
    for trace in figure.data:
        if trace.type == "scattergl":
            values = trace.to_plotly_json()
            values["type"] = "scatter"
            new_traces.append(go.Scatter(values))
            continue
        if trace.type != "heatmap":
            new_traces.append(trace)
            continue
        if trace.visible is False or trace.visible == "legendonly":
            continue
        if trace.zsmooth not in {None, False}:
            raise ExportError(
                "Smoothed heatmaps cannot be exported as exact vector cells. Choose PNG or TIFF."
            )
        z = np.asarray(trace.z, dtype=float)
        if z.ndim != 2:
            raise ExportError("Vector heatmaps require a two-dimensional numeric matrix.")
        if z.size > 100_000:
            raise ExportError(
                "This heatmap exceeds 100,000 vector cells. Select fewer panels or choose PNG/TIFF."
            )
        xr, yr = trace.xaxis or "x", trace.yaxis or "y"
        coords = []
        for reference, supplied, count, start, step in (
            (xr, trace.x, z.shape[1], trace.x0, trace.dx),
            (yr, trace.y, z.shape[0], trace.y0, trace.dy),
        ):
            values = (
                list(supplied)
                if supplied is not None
                else [float(start or 0) + i * float(step or 1) for i in range(count)]
            )
            if len(values) != count:
                raise ExportError(
                    "This heatmap uses unsupported explicit cell boundaries. Choose a raster format."
                )
            categorical = any(isinstance(v, str) for v in values)
            axis = figure.layout[_axis_key(reference)]
            if categorical:
                categories = list(axis.categoryarray) if axis.categoryarray is not None else values
                category_axes[reference] = categories
                positions = [categories.index(v) for v in values]
                axis.update(
                    type="linear",
                    tickmode="array",
                    tickvals=list(range(len(categories))),
                    ticktext=categories,
                    categoryarray=None,
                    categoryorder=None,
                )
            else:
                positions = [float(v) for v in values]
            edges = _edges(positions)
            if axis.range is None:
                axis.range = (
                    [float(edges[-1]), float(edges[0])]
                    if axis.autorange == "reversed"
                    else [float(edges[0]), float(edges[-1])]
                )
                axis.autorange = False
            coords.append((positions, edges))
        coloraxis = figure.layout[trace.coloraxis] if trace.coloraxis else None
        scale = coloraxis.colorscale if coloraxis and coloraxis.colorscale else trace.colorscale
        if scale is None:
            scale = figure.layout.template.layout.colorscale.sequential or get_colorscale("Viridis")
        if isinstance(scale, str):
            scale = get_colorscale(scale)
        scale = [list(v) for v in scale]
        if coloraxis.reversescale if coloraxis else trace.reversescale:
            scale = [[1 - v, color] for v, color in reversed(scale)]
        finite = z[np.isfinite(z)]
        low = coloraxis.cmin if coloraxis else trace.zmin
        high = coloraxis.cmax if coloraxis else trace.zmax
        mid = coloraxis.cmid if coloraxis else trace.zmid
        low = float(low) if low is not None else float(finite.min()) if finite.size else 0
        high = float(high) if high is not None else float(finite.max()) if finite.size else 1
        if (
            mid is not None
            and (trace.zmin is None and trace.zmax is None)
            and not (coloraxis and coloraxis.cmin is not None)
        ):
            span = max(abs(high - mid), abs(low - mid))
            low, high = mid - span, mid + span
        if high == low:
            low, high = low - 0.5, high + 0.5
        (xs, xe), (ys, ye) = coords
        xd = figure.layout[_axis_key(xr)].domain or (0, 1)
        yd = figure.layout[_axis_key(yr)].domain or (0, 1)
        usable_w = max(
            1, figure.layout.width - (figure.layout.margin.l or 0) - (figure.layout.margin.r or 0)
        )
        usable_h = max(
            1, figure.layout.height - (figure.layout.margin.t or 0) - (figure.layout.margin.b or 0)
        )
        gx = (trace.xgap or 0) * abs(xe[-1] - xe[0]) / max(1, usable_w * (xd[1] - xd[0])) / 2
        gy = (trace.ygap or 0) * abs(ye[-1] - ye[0]) / max(1, usable_h * (yd[1] - yd[0])) / 2
        texts = np.asarray(trace.text, dtype=object) if trace.text is not None else None
        tx, ty, tt, tc = [], [], [], []

        def add_cell_text(row, col, background):
            if trace.texttemplate and texts is not None:
                text = str(texts[row, col])
                if text not in {"", "None", "nan"}:
                    tx.append(xs[col])
                    ty.append(ys[row])
                    tt.append(text)
                    supplied = trace.textfont.color
                    tc.append(
                        supplied
                        if supplied and supplied != "auto"
                        else _heatmap_contrast(background)
                    )

        for row in range(z.shape[0]):
            for col in range(z.shape[1]):
                value = z[row, col]
                if not np.isfinite(value):
                    add_cell_text(row, col, figure.layout.plot_bgcolor or "white")
                    continue
                fraction = float(np.clip((value - low) / (high - low), 0, 1))
                # Plotly supports stepped (duplicate-stop) categorical scales.
                left = max(i for i, stop in enumerate(scale) if stop[0] <= fraction)
                right = min(left + 1, len(scale) - 1)
                if scale[right][0] == scale[left][0]:
                    color = scale[right][1]
                else:
                    color = sample_colorscale(
                        [[0, scale[left][1]], [1, scale[right][1]]],
                        [(fraction - scale[left][0]) / (scale[right][0] - scale[left][0])],
                    )[0]
                cells.append(
                    dict(
                        type="rect",
                        xref=xr,
                        yref=yr,
                        x0=float(min(xe[col : col + 2]) + gx),
                        x1=float(max(xe[col : col + 2]) - gx),
                        y0=float(min(ye[row : row + 2]) + gy),
                        y1=float(max(ye[row : row + 2]) - gy),
                        fillcolor=color,
                        line=dict(width=0),
                        opacity=trace.opacity if trace.opacity is not None else 1,
                        layer="between",
                        name="publication-heatmap-cell",
                    )
                )
                add_cell_text(row, col, color)
        showscale = coloraxis.showscale if coloraxis else trace.showscale
        marker = dict(
            color=[low, high],
            cmin=low,
            cmax=high,
            colorscale=scale,
            showscale=showscale is not False,
            colorbar=(coloraxis.colorbar if coloraxis else trace.colorbar).to_plotly_json(),
        )
        new_traces.append(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="markers",
                xaxis=xr,
                yaxis=yr,
                marker=marker,
                showlegend=False,
                hoverinfo="skip",
                name=trace.name,
            )
        )
        if tt:
            new_traces.append(
                go.Scatter(
                    x=tx,
                    y=ty,
                    text=tt,
                    mode="text",
                    textfont={**trace.textfont.to_plotly_json(), "color": tc},
                    opacity=trace.opacity,
                    xaxis=xr,
                    yaxis=yr,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    # All traces sharing converted categorical axes (e.g. KEGG text and bars) must follow.
    for trace in new_traces:
        for direction in ("x", "y"):
            reference = getattr(trace, direction + "axis", None) or direction
            categories = category_axes.get(reference)
            values = getattr(trace, direction, None)
            if categories and values is not None and any(isinstance(v, str) for v in values):
                setattr(
                    trace,
                    direction,
                    [categories.index(v) if v in categories else v for v in values],
                )
    figure.data = ()
    figure.add_traces(new_traces)
    figure.layout.shapes = [*(s.to_plotly_json() for s in figure.layout.shapes), *cells]
    return figure


def _scale_pixels(value: Any, factor: float, parent: str = "") -> Any:
    """Scale pixel-valued styling for high-DPI raster components in vector files."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            scale = (
                key
                in {
                    "xshift",
                    "yshift",
                    "borderwidth",
                    "borderpad",
                    "ticklen",
                    "tickwidth",
                    "linewidth",
                    "thickness",
                    "xgap",
                    "ygap",
                }
                or key == "size"
                and parent in {"font", "textfont", "tickfont", "title_font", "marker"}
                or key == "width"
                and parent in {"line", "error_x", "error_y", "error_z"}
                or key in {"l", "r", "t", "b", "pad"}
                and parent == "margin"
            )
            if scale and isinstance(item, (int, float)):
                result[key] = item * factor
            elif scale and isinstance(item, list):
                result[key] = [v * factor if isinstance(v, (int, float)) else v for v in item]
            else:
                result[key] = _scale_pixels(item, factor, key)
        return result
    if isinstance(value, list):
        return [_scale_pixels(v, factor, parent) for v in value]
    return value


def static_image(
    figure: go.Figure,
    format: str,
    width: float,
    height: float,
    scale: float = 1,
    timeout: float = 45,
) -> bytes:
    """Bound browser startup AND rendering; terminate the worker process group on timeout."""
    payload = json.dumps(
        dict(figure=figure.to_json(), format=format, width=width, height=height, scale=scale)
    ).encode()
    command = [sys.executable, "-m", "app_helpers.figure_export.adapters", "--render"]
    worker = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        output, error = worker.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(worker.pid, signal.SIGKILL)
        worker.communicate()
        raise ExportError(
            "Figure rendering timed out. Reduce the dimensions, DPI or selected panels and try again."
        ) from None
    if worker.returncode or not output:
        detail = error.decode(errors="replace")
        logging.getLogger(__name__).warning(
            "Publication renderer failed with exit code %s", worker.returncode
        )
        if "invalid args" in detail or "unexpected keyword" in detail:
            raise ExportError(
                "Plotly and Kaleido are incompatible. Install the versions pinned in the application's updated requirements.txt."
            )
        if "Chrome" in detail or "Browser" in detail or "chrom" in detail.lower():
            raise ExportError(
                "Static export could not start Chrome/Chromium. Install the browser and its system dependencies, or ask the server administrator to check the export setup."
            )
        raise ExportError(
            "The static renderer could not export this figure. Try PNG or fewer panels; check server logs for the renderer failure."
        )
    return output


def render_figure(
    spec: PlotExportSpec, config: ExportConfig, content: PlotContentConfig
) -> tuple[bytes, FigureArtifact]:
    config.validate()
    if config.format not in spec.supported_formats:
        raise ExportError(f"{config.format.upper()} is unsupported for this figure.")
    artifact = build_artifact(spec, config, content)
    figure = artifact.figure
    if config.vector:
        figure = vectorize_figure(figure)
    if config.vector and is_mixed(figure):
        if math.prod(config.pixel_dimensions) > 100_000_000:
            raise ExportError(
                "The raster components exceed 100 million pixels. Reduce dimensions or DPI."
            )
        factor = config.dpi / CSS_DPI
        figure = go.Figure(_scale_pixels(figure.to_plotly_json(), factor))
        width, height = config.pixel_dimensions
        scale = 1
    elif config.vector:
        width, height = (v * CSS_DPI for v in config.inches)
        scale = 1
    else:
        # Integer pixels first: native Kaleido otherwise truncates some fractional dimensions.
        w, h = config.pixel_dimensions
        scale = config.dpi / CSS_DPI
        width, height = w / scale + 1e-8, h / scale + 1e-8
    figure.update_layout(width=width, height=height)
    backend_format = "png" if config.format == "tiff" else config.format
    raw = static_image(figure, backend_format, width, height, scale)
    if config.format in {"png", "tiff"}:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        if image.size != config.pixel_dimensions:
            raise ExportError(
                f"Renderer returned {image.size}, expected {config.pixel_dimensions}. No image was resized."
            )
        buffer = io.BytesIO()
        if config.format == "tiff":
            image.save(buffer, format="TIFF", compression="tiff_lzw", dpi=(config.dpi, config.dpi))
        else:
            image.save(buffer, format="PNG", dpi=(config.dpi, config.dpi))
        raw = buffer.getvalue()
    elif config.format == "svg":
        root = ET.fromstring(raw)
        root.set("width", f"{convert_length(config.width, config.unit, 'mm'):.9g}mm")
        root.set("height", f"{convert_length(config.actual_height, config.unit, 'mm'):.9g}mm")
        # Namespace registration keeps standard svg element names for downstream editors.
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        raw = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    elif config.format == "pdf":
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            raise ExportError(
                "PDF export requires pypdf. Install the application's updated requirements."
            ) from None
        reader = PdfReader(io.BytesIO(raw))
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        page = writer.pages[0]
        page.scale_to(*(v * POINTS_PER_INCH for v in config.inches))
        writer.add_metadata(
            {
                "/Title": str(artifact.figure.layout.title.text or spec.plot_id),
                "/Creator": "Lioness ROSMAP publication figure export",
            }
        )
        buffer = io.BytesIO()
        writer.write(buffer)
        raw = buffer.getvalue()
    return raw, artifact


def build_artifact(
    spec: PlotExportSpec, config: ExportConfig, content: PlotContentConfig
) -> FigureArtifact:
    if spec.builder is None:
        return build_figure(spec.figure_snapshot, None, config, content)
    from copy import deepcopy

    return spec.builder(
        deepcopy(spec.data_snapshot), deepcopy(spec.analysis_snapshot), config, content
    )


def fit_automatic_config(
    spec: PlotExportSpec, config: ExportConfig, content: PlotContentConfig
) -> ExportConfig:
    """Keep the preferred aspect unless physical text leaves too little plotting space.

    Return the effective configuration before preview, summary or rendering, so
    pixel/page dimensions always correspond to the configuration the user sees.
    Manual dimensions and non-tight layouts are exact and never adjusted.
    """
    if config.height is not None or not config.style.tight:
        return config
    artifact = build_artifact(spec, config, content)
    figure = artifact.figure
    margin = figure.layout.margin
    if artifact.minimum_height:
        if artifact.minimum_height > figure.layout.height:
            fitted = replace(
                config, aspect_ratio=float(figure.layout.width) / artifact.minimum_height
            )
            # User-defined fractional panel spacing grows with the canvas. A
            # bounded second pass covers that dependency before showing sizes.
            for _ in range(4):
                required = build_artifact(spec, fitted, content).minimum_height
                if required <= fitted.inches[1] * CSS_DPI + 0.5:
                    break
                fitted = replace(fitted, aspect_ratio=float(figure.layout.width) / required)
            return fitted
        return config
    minimum_panel_fraction = min(
        (
            float(
                figure.layout[_axis_key(p.yaxis)].domain[1]
                - figure.layout[_axis_key(p.yaxis)].domain[0]
            )
            for p in artifact.panels
            if not p.scene and figure.layout[_axis_key(p.yaxis)].domain
        ),
        default=1.0,
    )
    minimum_plot_height = max(
        config.typography.axis_title * 8 * CSS_DPI / POINTS_PER_INCH,
        float(figure.layout.width) * 0.15,
    ) / max(0.05, minimum_panel_fraction)
    required = (margin.t or 0) + (margin.b or 0) + minimum_plot_height
    if required > figure.layout.height:
        return replace(config, aspect_ratio=float(figure.layout.width) / required)
    return config


def _worker_main() -> None:
    request = json.loads(sys.stdin.buffer.read())
    figure = restore_figure(request["figure"])
    output = figure.to_image(
        format=request["format"],
        width=request["width"],
        height=request["height"],
        scale=request["scale"],
    )
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    _worker_main()
