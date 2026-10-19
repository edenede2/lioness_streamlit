"""Export-only selection of categorical bars from immutable Plotly snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go


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
