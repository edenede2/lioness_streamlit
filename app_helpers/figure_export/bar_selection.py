"""Export-only selection of categorical bars from immutable Plotly snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PIL import ImageColor


@dataclass(frozen=True)
class BarItem:
    key: str
    series: str
    category: str
    panel: str


@dataclass(frozen=True)
class BarOptions:
    series: tuple[str, ...]
    categories: tuple[str, ...]
    items: tuple[BarItem, ...]
    series_label: str = "Series"
    category_label: str = "Categories"


def _series(trace: Any, index: int) -> str:
    return str(trace.legendgroup or trace.name or f"Series {index + 1}")


def _category_axis(trace: Any) -> str:
    return "y" if trace.orientation == "h" else "x"


def _layout_axis(reference: str) -> str:
    return reference[0] + "axis" + reference[1:]


def bar_options(figure: go.Figure) -> BarOptions | None:
    """Only independent categorical bars are supported, not mixed KEGG matrices.

    Keeping this conservative avoids detaching annotations, heatmap strips, or
    independently drawn confidence intervals from their associated observations.
    Native Plotly error bars remain attached and are sliced with their bar.
    """
    if not figure.data or any(trace.type != "bar" for trace in figure.data):
        return None
    if len({_category_axis(trace) for trace in figure.data}) != 1:
        return None
    if any(reference != "paper" and not reference.endswith(" domain")
           for annotation in figure.layout.annotations
           for reference in (annotation.xref or "x", annotation.yref or "y")):
        return None  # Do not leave point-specific labels on omitted bars.
    items, series, observed, category_order = [], [], [], []
    for i, trace in enumerate(figure.data):
        axis = _category_axis(trace)
        values = getattr(trace, axis)
        other = getattr(trace, "x" if axis == "y" else "y")
        if values is None or other is None or len(values) != len(other):
            return None
        if not all(isinstance(value, str) for value in values):
            return None  # Never turn numeric/date positions into category indices.
        if any(getattr(error, "traceref", None) is not None or getattr(error, "tracerefminus", None) is not None
               for error in (trace.error_x, trace.error_y)):
            return None
        name = _series(trace, i)
        if name not in series:
            series.append(name)
        panel = f"{trace.xaxis or 'x'}:{trace.yaxis or 'y'}"
        reference = getattr(trace, axis + "axis") or axis
        category_order.extend(figure.layout[_layout_axis(reference)].categoryarray or ())
        for j, value in enumerate(values):
            observed.append(value)
            items.append(BarItem(f"bar:{i}:{j}", name, value, panel))
    categories = tuple(dict.fromkeys(value for value in [*category_order, *observed] if value in observed))
    if not items:
        return None
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    metadata = meta.get("figure_export", {})
    settings = metadata.get("bar_selection", {}) if isinstance(metadata, dict) else {}
    if not isinstance(settings, dict):
        settings = {}
    return BarOptions(tuple(series), categories, tuple(items),
                      settings.get("series_label", "Series"), settings.get("category_label", "Categories"))


def _slice_array(value: Any, positions: list[int], count: int) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == count:
        return [value[index] for index in positions]
    return value


def _original_colors(figure: go.Figure, trace: Any, index: int) -> list[str]:
    count = len(getattr(trace, _category_axis(trace)))
    colors = trace.marker.color
    if colors is None:
        palette = figure.layout.colorway or figure.layout.template.layout.colorway or ("#636EFA",)
        return [palette[index % len(palette)]] * count
    if isinstance(colors, str):
        return [colors] * count
    if isinstance(colors, (tuple, list, np.ndarray)) and len(colors) == count and all(isinstance(color, str) for color in colors):
        return list(colors)
    raise ValueError("Custom bar colors require categorical, rather than continuous numeric, colors.")


def bar_color_defaults(figure: go.Figure) -> dict[str, str]:
    """Hex defaults for bounded color pickers; no alteration of source colors."""
    defaults = {}
    for i, trace in enumerate(figure.data):
        for j, color in enumerate(_original_colors(figure, trace, i)):
            try:
                rgb = ImageColor.getrgb(color)[:3]
            except ValueError:
                channels = re.findall(r"[\d.]+", color)
                if not color.startswith("rgba(") or len(channels) != 4:
                    raise ValueError("This bar color cannot be represented by a color picker.") from None
                rgb = tuple(int(float(value)) for value in channels[:3])
            defaults[f"bar:{i}:{j}"] = "#" + "".join(f"{value:02x}" for value in rgb)
    return defaults


def apply_bar_colors(figure: go.Figure, content: Any) -> None:
    """Apply explicit colors before slicing so original point identities persist."""
    if not content.bar_series_colors and not content.bar_item_colors:
        return
    options = bar_options(figure)
    if options is None:
        raise ValueError("Custom bar colors are unavailable for this figure.")
    series_colors, item_colors = dict(content.bar_series_colors), dict(content.bar_item_colors)
    if (len(series_colors) != len(content.bar_series_colors) or len(item_colors) != len(content.bar_item_colors)
            or set(series_colors) - set(options.series)
            or set(item_colors) - {item.key for item in options.items}):
        raise ValueError("Invalid bar color identity. Reset the export settings.")
    if any(not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
           for color in [*series_colors.values(), *item_colors.values()]):
        raise ValueError("Bar colors must be six-digit hexadecimal colors, such as #0072B2.")
    for i, trace in enumerate(figure.data):
        series = _series(trace, i)
        colors = _original_colors(figure, trace, i)
        if series not in series_colors and not any(f"bar:{i}:{j}" in item_colors for j in range(len(colors))):
            continue
        trace.marker.color = [item_colors.get(f"bar:{i}:{j}", series_colors.get(series, color))
                              for j, color in enumerate(colors)]
        # Literal color arrays also prevent a global export palette from
        # overriding the user's explicit model/per-bar choices.
        trace.marker.coloraxis = None
    _refresh_group_key(figure)


def _refresh_group_key(figure: go.Figure) -> None:
    """Keep optional score keys aligned with selected bars and export colors."""
    metadata = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    settings = metadata.get("figure_export", {}).get("bar_selection", {})
    annotation_name = settings.get("group_label_annotation")
    if not annotation_name:
        return
    group_column = int(settings["group_customdata_index"])
    item_column = settings.get("group_item_customdata_index")
    display_labels = settings.get("category_display_labels", {})
    groups = {}
    for index, trace in enumerate(figure.data):
        colors = _original_colors(figure, trace, index) if settings.get("group_item_colors") else None
        for i, (category, record) in enumerate(zip(getattr(trace, _category_axis(trace)), trace.customdata)):
            label = str(record[int(item_column)]) if item_column is not None else display_labels.get(category, category)
            label = html.escape(label)
            if colors:
                label = f'<span style="color:{html.escape(colors[i], quote=True)}">■</span> {label}'
            groups.setdefault(str(record[group_column]), []).append(label)
    separator = "<br />" if settings.get("group_items_multiline") else ", "
    text = "<br />".join(f"<b>{html.escape(group)}</b>: {separator.join(labels)}"
                         for group, labels in groups.items())
    for annotation in figure.layout.annotations:
        if annotation.name == annotation_name:
            annotation.text = text


def _slice_trace(trace: Any, positions: list[int], count: int) -> None:
    # Explicit point-valued attributes; never slice scales, bins or other arrays
    # merely because their length coincidentally matches the number of bars.
    for name in ("x", "y", "base", "width", "offset", "text", "textposition", "ids",
                 "customdata", "hovertext", "hovertemplate"):
        value = getattr(trace, name, None)
        if value is not None:
            trace[name] = _slice_array(value, positions, count)
    for obj, fields in (
        (trace.marker, ("color", "opacity")),
        (trace.marker.line, ("color", "width")),
        (trace.marker.pattern, ("shape", "size", "solidity", "fgcolor", "bgcolor", "fgopacity")),
        (trace.textfont, ("family", "size", "color")),
        (trace.insidetextfont, ("family", "size", "color")),
        (trace.outsidetextfont, ("family", "size", "color")),
        (trace.error_x, ("array", "arrayminus")),
        (trace.error_y, ("array", "arrayminus")),
    ):
        for name in fields:
            value = getattr(obj, name, None)
            if value is not None:
                obj[name] = _slice_array(value, positions, count)
    if trace.selectedpoints is not None:
        original = set(trace.selectedpoints)
        trace.selectedpoints = [new for new, old in enumerate(positions) if old in original]


def select_bars(figure: go.Figure, content: Any) -> None:
    """Filter an export copy in place; keep numeric values and uncertainty intact."""
    if all(value is None for value in (content.bar_series, content.bar_categories, content.bar_items)):
        return
    options = bar_options(figure)
    if options is None:
        raise ValueError("Bar selection is unavailable for this figure's layout.")
    series = content.bar_series if content.bar_series is not None else options.series
    categories = content.bar_categories if content.bar_categories is not None else options.categories
    items = content.bar_items
    for selected, available in ((series, options.series), (categories, options.categories),
                                (items or (), tuple(item.key for item in options.items))):
        if len(selected) != len(set(selected)) or set(selected) - set(available):
            raise ValueError("Invalid bar selection. Reset the export settings and try again.")
    retained, axis_values = [], {}
    for i, trace in enumerate(figure.data):
        name = _series(trace, i)
        axis = _category_axis(trace)
        values = list(getattr(trace, axis))
        positions = [j for j, category in enumerate(values) if name in series and category in categories
                     and (items is None or f"bar:{i}:{j}" in items)]
        if not positions:
            continue
        _slice_trace(trace, positions, len(values))
        retained.append((series.index(name), trace))
        reference = getattr(trace, axis + "axis") or axis
        axis_values.setdefault(reference, set()).update(values[j] for j in positions)
    if not retained:
        raise ValueError("Select at least one bar: choose a series/model, a category/block and any individual bars.")
    figure.data = tuple(trace for _, trace in sorted(retained, key=lambda item: item[0]))
    # Optional structured group keys must describe the exported bars, not bars
    # removed by the existing series/category/item selectors.
    _refresh_group_key(figure)
    for reference, values in axis_values.items():
        axis = figure.layout[_layout_axis(reference)]
        reversed_axis = axis.autorange == "reversed" or (
            axis.range is not None and len(axis.range) == 2 and axis.range[0] > axis.range[1]
        )
        axis.update(type="category", categoryorder="array", categoryarray=[v for v in categories if v in values],
                    range=None, autorange="reversed" if reversed_axis else True)
        if axis.tickvals is not None:
            indices = [i for i, value in enumerate(axis.tickvals) if value in values]
            labels = list(axis.ticktext) if axis.ticktext is not None else None
            axis.tickvals = [axis.tickvals[i] for i in indices]
            if labels is not None and all(i < len(labels) for i in indices):
                axis.ticktext = [labels[i] for i in indices]
