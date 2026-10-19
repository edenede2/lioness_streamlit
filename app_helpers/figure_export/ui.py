"""One collapsed export entry point and native dialog for every application plot."""

from __future__ import annotations

import html
import hashlib
import io
import re
import zipfile
from dataclasses import asdict, replace
from typing import Any

import streamlit as st
from app_helpers.streamlit_compat import supports_dialog
from .bar_selection import bar_color_defaults, bar_options

from .adapters import (
    ExportError,
    build_artifact,
    can_share_axes,
    fit_automatic_config,
    correlation_metadata,
    is_mixed,
    render_figure,
    panel_title_annotations,
    trace_role,
)
from .config import (
    FORMATS,
    MIME_TYPES,
    PRESETS,
    ExportConfig,
    PlotContentConfig,
    PlotExportSpec,
    Style,
    Typography,
    convert_length,
    default_filename,
    figure_panels,
    output_summary,
    preset_config,
    render_fingerprint,
    restore_figure,
    sanitize_filename,
    widget_key,
)


def _plain(value: Any) -> str:
    return html.unescape(re.sub("<[^>]+>", " ", str(value or ""))).strip()


def _prefix(spec: PlotExportSpec) -> str:
    return widget_key(spec.plot_id, spec.fingerprint, "")


def _reset(spec: PlotExportSpec, preset: str | None = None) -> None:
    prefix = _prefix(spec)
    selected = preset or st.session_state.get(prefix + "preset", "Manuscript – single column")
    # Clear only this export namespace, never main-page widgets.
    for key in list(st.session_state):
        if key.startswith(prefix):
            del st.session_state[key]
    st.session_state[prefix + "preset"] = selected
    st.session_state.pop("figure_export::prepared", None)


def _change_preset(spec: PlotExportSpec) -> None:
    prefix = _prefix(spec)
    selected = st.session_state[prefix + "preset"]
    if selected != "Custom":
        _reset(spec, selected)


def _convert_units(spec: PlotExportSpec) -> None:
    prefix = _prefix(spec)
    source, destination = (
        st.session_state.get(prefix + "previous_unit", "mm"),
        st.session_state[prefix + "unit"],
    )
    for control in ("width", "manual_height"):
        if prefix + control in st.session_state:
            st.session_state[prefix + control] = convert_length(
                st.session_state[prefix + control], source, destination
            )
    st.session_state[prefix + "previous_unit"] = destination


def _render_cached(spec: PlotExportSpec, config: ExportConfig, content: PlotContentConfig):
    if spec.builder is not None:
        data, artifact = render_figure(spec, config, content)
        return data, artifact.warnings
    fingerprint = render_fingerprint(spec, config, content)
    # One session-wide budget prevents many analysis snapshots accumulating
    # separate caches. The full snapshot/configuration fingerprint is the key.
    key = "figure_export::render_cache"
    cache = st.session_state.get(key, {})
    if fingerprint in cache:
        return cache[fingerprint]
    data, artifact = render_figure(spec, config, content)
    # Cache immutable bytes and warning strings, not figure objects or buffers.
    result = (data, artifact.warnings)
    if len(data) <= 32 * 1024 * 1024:
        while len(cache) >= 3:
            cache.pop(next(iter(cache)))
        if sum(len(v[0]) for v in cache.values()) + len(data) > 48 * 1024 * 1024:
            cache.clear()
        cache[fingerprint] = result
        st.session_state[key] = cache
    return result


def _bar_color_controls(source, choices, series, categories, items, key):
    if not st.checkbox("Customize bar colors", key=key("custom_bar_colors")):
        return (), ()
    try:
        defaults = bar_color_defaults(source)
    except (TypeError, ValueError) as error:
        st.info(str(error))
        return (), ()
    candidates = [item for item in choices.items if item.series in series and item.category in categories
                  and (items is None or item.key in items)]
    series_colors = []
    columns = st.columns(2)
    for i, series_name in enumerate(series):
        members = [item for item in candidates if item.series == series_name]
        if not members:
            continue
        with columns[i % 2]:
            series_colors.append((series_name, st.color_picker(
                f"{_plain(series_name)} color", value=defaults[members[0].key],
                key=key("series_color_" + hashlib.sha256(series_name.encode()).hexdigest()[:16]),
            )))
    item_colors = []
    if st.checkbox("Override individual bar colors", key=key("custom_individual_colors")):
        labels = {item.key: f"{_plain(item.category)} — {_plain(item.series)} ({item.panel})" for item in candidates}
        selection_key = key("recolored_bar_items")
        if selection_key in st.session_state:
            st.session_state[selection_key] = [value for value in st.session_state[selection_key] if value in labels]
        selected = st.multiselect("Bars to recolor individually", list(labels),
                                  default=None if selection_key in st.session_state else [],
                                  format_func=labels.get, key=selection_key)
        series_lookup = dict(series_colors)
        for i, item in enumerate(item for item in candidates if item.key in selected):
            with columns[i % 2]:
                item_colors.append((item.key, st.color_picker(
                    f"Bar color: {labels[item.key]}", value=series_lookup.get(item.series, defaults[item.key]),
                    key=key("item_color_" + item.key),
                )))
        st.caption("Individual bar colors override series colors and the export palette. Legend entries identify series; a series swatch does not show every per-bar override.")
    return tuple(series_colors), tuple(item_colors)


def _controls(spec: PlotExportSpec) -> tuple[ExportConfig, PlotContentConfig, bool]:
    source = restore_figure(spec.figure_snapshot)
    panels = figure_panels(source)
    meta = correlation_metadata(source)
    key = lambda name: _prefix(spec) + name
    preset = st.selectbox(
        "Publication preset",
        list(PRESETS),
        key=key("preset"),
        on_change=_change_preset,
        args=(spec,),
    )
    defaults = preset_config(preset, spec.aspect_ratio)
    # A custom builder owns its own presentation contract; do not offer filters
    # it may not implement. Ordinary application charts use snapshot rebuilding.
    bar_choices = bar_options(source) if spec.builder is None else None
    tabs = st.tabs(["Size and format", "Typography", "Appearance", "Content", "Panels"]
                   + (["Bars and groups"] if bar_choices else []))
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            unit = st.selectbox(
                "Dimension units",
                ["mm", "cm", "in"],
                index=["mm", "cm", "in"].index(defaults.unit),
                key=key("unit"),
                on_change=_convert_units,
                args=(spec,),
            )
            st.session_state[key("previous_unit")] = unit
            width = st.number_input(
                "Figure width",
                min_value=0.01,
                value=float(convert_length(defaults.width, defaults.unit, unit)),
                step=0.5,
                key=key("width"),
            )
            automatic = st.checkbox("Automatic height", value=True, key=key("automatic_height"))
            ratio = st.number_input(
                "Aspect ratio (width / height)",
                min_value=0.05,
                max_value=50.0,
                value=float(spec.aspect_ratio),
                step=0.05,
                key=key("aspect_ratio"),
            )
            manual_height = float(width / ratio)
            if automatic:
                st.caption(
                    "Height follows the preferred aspect ratio and expands when physical text needs more space."
                )
            else:
                manual_height = st.number_input(
                    "Figure height",
                    min_value=0.01,
                    value=manual_height,
                    step=0.5,
                    key=key("manual_height"),
                )
        with b:
            format = st.selectbox(
                "File format", list(FORMATS), format_func=str.upper, key=key("format")
            )
            dpi = st.number_input(
                "DPI (raster output or embedded raster components)",
                min_value=72,
                max_value=1200,
                value=defaults.dpi,
                step=50,
                key=key("dpi"),
            )
            transparent = st.checkbox("Transparent background", value=False, key=key("transparent"))
            background = st.color_picker("Background color", value="#FFFFFF", key=key("background"))
            tight = st.checkbox("Remove excessive whitespace", value=True, key=key("tight"))
    with tabs[1]:
        family = st.selectbox(
            "Font family",
            [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "Times New Roman",
                "DejaVu Serif",
                "Courier New",
            ],
            key=key("font_family"),
        )
        columns = st.columns(2)
        font_values = {}
        for i, (field, label) in enumerate(
            (
                ("base", "Base font"),
                ("title", "Title font"),
                ("subtitle", "Subtitle / panel-title font"),
                ("axis_title", "Axis-title font"),
                ("tick", "Tick-label font"),
                ("legend", "Legend font"),
                ("annotation", "Annotation / statistical font"),
            )
        ):
            with columns[i % 2]:
                font_values[field] = st.number_input(
                    label + " (pt)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(getattr(defaults.typography, field)),
                    step=0.5,
                    key=key("font_" + field),
                )
        bold = st.checkbox("Bold title", value=True, key=key("bold_title"))
        font_color = st.color_picker("Font color", value="#202124", key=key("font_color"))
        spacing = st.number_input(
            "Statistical annotation spacing (pt)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.5,
            key=key("annotation_spacing"),
        )
    with tabs[2]:
        styles = asdict(defaults.style)
        styles.update(background=background, transparent=transparent, tight=tight)
        for field, label in (
            ("margin_left", "Left margin"),
            ("margin_right", "Right margin"),
            ("margin_top", "Top margin"),
            ("margin_bottom", "Bottom margin"),
        ):
            styles[field] = st.number_input(
                label + " (pt)",
                min_value=0.0,
                max_value=500.0,
                value=float(styles[field]),
                step=1.0,
                key=key(field),
            )
        styles["grid"] = st.checkbox("Show grid", value=True, key=key("grid"))
        styles["grid_opacity"] = st.slider("Grid opacity", 0.0, 1.0, 0.18, key=key("grid_opacity"))
        styles["axis_lines"] = st.checkbox("Show axis lines", value=True, key=key("axis_lines"))
        styles["axis_width"] = st.number_input(
            "Axis-line thickness (pt)", min_value=0.0, value=0.6, step=0.1, key=key("axis_width")
        )
        angle = st.selectbox(
            "Tick-label angle", ["Original", "0", "-45", "45", "90"], key=key("tick_angle")
        )
        styles["tick_angle"] = None if angle == "Original" else float(angle)
        styles["number_format"] = st.selectbox(
            "Number formatting", ["Original", "Decimal", "Scientific"], key=key("number_format")
        )
        styles["decimals"] = st.slider("Decimal precision", 0, 10, 3, key=key("decimals"))
        styles["palette"] = st.selectbox(
            "Color palette",
            ["Original", "Color-blind safe", "Viridis", "Cividis"],
            key=key("palette"),
        )
        has_points = any(
            "markers" in str(getattr(t, "mode", "")) or t.type in {"box", "violin"}
            for t in source.data
        )
        if has_points:
            styles["point_size"] = st.number_input(
                "Point size (pt)",
                min_value=0.1,
                max_value=40.0,
                value=4.0,
                step=0.5,
                key=key("point_size"),
            )
            styles["point_opacity"] = st.slider(
                "Point opacity", 0.0, 1.0, 0.74, key=key("point_opacity")
            )
            styles["point_shape"] = st.selectbox(
                "Point shape",
                ["Original", "circle", "diamond", "square", "cross", "x"],
                key=key("point_shape"),
            )
            styles["point_encoding"] = st.selectbox(
                "Point group encoding",
                ["Original", "One color", "Group color", "Group shape", "Group color and shape"],
                key=key("point_encoding"),
            )
            if styles["point_encoding"] in {"One color", "Group shape"}:
                styles["single_color"] = st.color_picker(
                    "Point color", "#0072B2", key=key("single_color")
                )
        has_lines = any(
            "lines" in str(getattr(t, "mode", "")) or t.type in {"box", "violin"}
            for t in source.data
        )
        if has_lines:
            styles["line_width"] = st.number_input(
                "Line width (pt)", min_value=0.1, value=1.5, step=0.1, key=key("line_width")
            )
            styles["line_dash"] = st.selectbox(
                "Line type", ["Original", "solid", "dash", "dot", "dashdot"], key=key("line_dash")
            )
            styles["line_opacity"] = st.slider(
                "Line opacity", 0.0, 1.0, 1.0, key=key("line_opacity")
            )
        has_regression = any(trace_role(t) == "regression" for t in source.data)
        if has_regression:
            styles["regression_width"] = st.number_input(
                "Regression-line width (pt)",
                min_value=0.1,
                value=1.8,
                step=0.1,
                key=key("regression_width"),
            )
            styles["regression_dash"] = st.selectbox(
                "Regression-line type",
                ["Original", "solid", "dash", "dot", "dashdot"],
                key=key("regression_dash"),
            )
            if st.checkbox("Custom regression color", key=key("custom_regression_color")):
                styles["regression_color"] = st.color_picker(
                    "Regression color", "#202124", key=key("regression_color")
                )
        has_intervals = any(
            trace_role(t) == "interval"
            or any(
                getattr(getattr(t, e, None), "array", None) is not None
                for e in ("error_x", "error_y", "error_z")
            )
            for t in source.data
        )
        if any(trace_role(t) == "interval" for t in source.data):
            styles["ci_opacity"] = st.slider(
                "Confidence-band opacity", 0.0, 1.0, 0.2, key=key("ci_opacity")
            )
        styles["legend_position"] = st.selectbox(
            "Legend position",
            ["Original", "Top", "Bottom", "Right", "Inside top-right"],
            key=key("legend_position"),
        )
        styles["legend_orientation"] = st.selectbox(
            "Legend orientation",
            ["Original", "Horizontal", "Vertical"],
            key=key("legend_orientation"),
        )
        original_legend = str(source.layout.legend.title.text or "")
        legend_title = st.text_input(
            "Legend title", value=_plain(original_legend), key=key("legend_title")
        )
        styles["legend_title"] = legend_title if legend_title != _plain(original_legend) else None
    with tabs[3]:
        if spec.metadata:
            st.caption("Current analysis: " + " · ".join(f"{k}: {v}" for k, v in spec.metadata))
        st.caption(
            "Analysis filters and statistical methods stay matched to the main plot. These controls change the exported presentation."
        )
        original_title = _plain(source.layout.title.text)
        show_title = st.checkbox("Show plot title", value=True, key=key("show_title"))
        title = st.text_input("Plot title", value=original_title, key=key("title"))
        show_subtitle = st.checkbox("Show plot subtitle", value=True, key=key("show_subtitle"))
        subtitle = st.text_input("Subtitle", key=key("subtitle"))
        styles["legend_visible"] = st.checkbox(
            "Show legend", value=source.layout.showlegend is not False, key=key("legend_visible")
        )
        show_selected_module_annotation = (
            st.checkbox(
                "Show selected-module annotation",
                value=True,
                key=key("show_selected_module_annotation"),
                help=(
                    "Hide the selected module's highlight, label and legend entry in "
                    "the export. Its normal data point and all statistics remain unchanged."
                ),
            )
            if any(trace_role(t) == "selected_module" for t in source.data)
            else True
        )
        has_color_scale = any(
            k.startswith("coloraxis") and source.layout[k].to_plotly_json() for k in source.layout
        ) or any(
            (hasattr(t, "showscale") and t.showscale is not False)
            or getattr(getattr(t, "marker", None), "showscale", None) is True
            for t in source.data
        )
        show_color_scale = (
            st.checkbox("Show color scale", value=True, key=key("show_color_scale"))
            if has_color_scale
            else True
        )
        panel_headings = panel_title_annotations(source, panels)
        show_panel_titles = (
            st.checkbox("Show panel titles", value=True, key=key("show_panel_titles"))
            if panel_headings
            else True
        )
        show_panel_subtitles = (
            st.checkbox(
                "Show panel subtitles / enrichment", value=True, key=key("show_panel_subtitles")
            )
            if any("<br>" in str(a.text) for a in panel_headings)
            else True
        )
        axis_labels = {}
        axis_visibility = {}
        shared_axis_titles = (
            st.checkbox(
                "Use shared axis titles",
                value=True,
                key=key("shared_axis_titles"),
                help="Show an identical axis title once for the combined figure. Axis values and ranges remain unchanged.",
            )
            if len(panels) > 1 and not any(p.scene for p in panels)
            else True
        )
        for direction, label in (
            ("x", "X-axis label"),
            ("y", "Y-axis label"),
            ("z", "Z-axis label"),
        ):
            if direction == "z" and not is_mixed(source):
                continue
            originals = [
                str(
                    (
                        source.layout[p.scene][direction + "axis"]
                        if p.scene
                        else source.layout[
                            direction + "axis" + (getattr(p, direction + "axis")[1:])
                        ]
                    ).title.text
                    or ""
                )
                for p in panels
                if p.scene or direction != "z"
            ]
            original = originals[0] if originals and len(set(originals)) == 1 else ""
            axis_visibility["show_" + direction + "_title"] = st.checkbox(
                f"Show {direction.upper()}-axis title",
                value=True,
                key=key("show_" + direction + "_title"),
            )
            edited = st.text_input(label, value=_plain(original), key=key(direction + "_label"))
            axis_labels[direction + "_label"] = edited if edited != _plain(original) else None
        caption = st.text_area("Caption / footnote", key=key("caption"))
        show_observations = (
            st.checkbox("Show individual observations", value=True, key=key("observations"))
            if has_points
            else True
        )
        show_regression = (
            st.checkbox("Show existing regression guides", value=True, key=key("regression"))
            if has_regression
            else True
        )
        show_intervals = (
            st.checkbox("Show existing confidence intervals", value=True, key=key("intervals"))
            if has_intervals
            else True
        )
        show_statistics = True
        if any(
            (a.name or "").startswith("publication-statistics") for a in source.layout.annotations
        ):
            show_statistics = st.checkbox(
                "Show statistical annotations", value=True, key=key("statistics")
            )
        mode, groups, fields, precision, position = (
            "Original",
            None,
            tuple(meta.get("annotation_fields", ("n", "coefficient", "p", "fdr"))),
            3,
            "Original",
        )
        if meta.get("correlation_records"):
            st.caption(
                "Correlation method: "
                + meta["correlation_method"].capitalize()
                + ". Confidence bands are unavailable for these existing OLS guides."
            )
            mode = st.selectbox(
                "Correlation annotations",
                ["Original", "Pooled only", "Group-specific only", "Both", "None"],
                key=key("correlation_mode"),
            )
            levels = [str(v) for v in meta["grouping_levels"]]
            groups = tuple(
                st.multiselect(
                    "Groups to annotate",
                    levels,
                    default=levels,
                    format_func=lambda v: meta["grouping_labels"].get(v, v),
                    key=key("annotation_groups"),
                )
            )
            fields = tuple(
                st.multiselect(
                    "Statistical fields",
                    ["n", "coefficient", "p", "fdr"],
                    default=list(fields),
                    key=key("statistic_fields"),
                )
            )
            precision = st.slider("Statistical precision", 1, 8, 3, key=key("statistic_precision"))
        if meta.get("correlation_records") or any(
            (a.name or "").startswith("publication-statistics") for a in source.layout.annotations
        ):
            position = st.selectbox(
                "Statistical annotation position",
                ["Original", "Inside top-left", "Below panel"],
                key=key("statistic_position"),
            )
        horizontal = vertical = None
        if not is_mixed(source):
            if st.checkbox("Add horizontal reference line", key=key("add_hline")):
                horizontal = st.number_input(
                    "Horizontal reference value", value=0.0, key=key("hline")
                )
            if st.checkbox("Add vertical reference line", key=key("add_vline")):
                vertical = st.number_input("Vertical reference value", value=0.0, key=key("vline"))
    with tabs[4]:
        order, panel_labels, separately = None, False, False
        if len(panels) > 1:
            identifiers = [p.identifier for p in panels]
            labels = {p.identifier: p.label for p in panels}
            order = tuple(
                st.multiselect(
                    "Panels in export order",
                    identifiers,
                    default=identifiers,
                    format_func=lambda v: labels[v],
                    key=key("panel_order"),
                )
            )
            st.caption(
                "Remove and re-add a panel to change its order. Group observations and statistics remain unchanged."
            )
            panel_labels = st.checkbox("Add panel labels (A, B, C…)", key=key("panel_labels"))
            if st.checkbox("Customize panel layout", key=key("custom_panels")):
                styles["panel_columns"] = st.number_input(
                    "Panel columns",
                    min_value=1,
                    max_value=max(1, len(panels)),
                    value=min(3, len(panels)),
                    key=key("panel_columns"),
                )
                styles["panel_spacing"] = st.slider(
                    "Panel spacing",
                    0.0,
                    0.3,
                    0.08,
                    key=key("panel_spacing"),
                    help="Minimum gap as a fraction of the plotting area. Tight layout adds room when needed for titles and axis labels.",
                )
                axes = st.selectbox(
                    "Panel axes",
                    (
                        ["Original", "Independent", "Shared"]
                        if can_share_axes(source, panels)
                        else ["Original", "Independent"]
                    ),
                    key=key("shared_axes"),
                )
                styles["shared_axes"] = None if axes == "Original" else axes == "Shared"
            separately = st.checkbox(
                "Export each selected panel separately (ZIP)", key=key("separately")
            )
        else:
            st.caption("This figure has one panel.")
    bar_series = bar_categories = bar_items = None
    bar_series_colors = bar_item_colors = ()
    if bar_choices:
        with tabs[5]:
            st.caption("Export only: hide models, blocks or individual bars without changing the analysis. Retained values and uncertainty are unchanged. Remove and re-add model/block selections to change their order.")
            series = tuple(st.multiselect(
                f"{bar_choices.series_label} to include", bar_choices.series,
                default=bar_choices.series, format_func=_plain, key=key("bar_series"),
            ))
            categories = tuple(st.multiselect(
                f"{bar_choices.category_label} to include", bar_choices.categories,
                default=bar_choices.categories, format_func=_plain, key=key("bar_categories"),
                help=("For example, retain CT pooled and TS pooled and remove CT + TS pooled."
                      if bar_choices.category_label == "Predictor blocks"
                      else "Only selected categories appear in the export; retained values are unchanged."),
            ))
            bar_series = series if series != bar_choices.series else None
            bar_categories = categories if categories != bar_choices.categories else None
            if st.checkbox("Choose individual bars", key=key("choose_bar_items")):
                candidates = [item for item in bar_choices.items if item.series in series and item.category in categories]
                labels = {item.key: f"{_plain(item.category)} — {_plain(item.series)}"
                          + (f" ({item.panel})" if len(panels) > 1 else "") for item in candidates}
                item_key = key("bar_items")
                if item_key in st.session_state:
                    st.session_state[item_key] = [value for value in st.session_state[item_key] if value in labels]
                bar_items = tuple(st.multiselect(
                    "Individual bars to include", list(labels),
                    default=None if item_key in st.session_state else list(labels),
                    format_func=labels.get, key=item_key,
                ))
                count = len(bar_items)
            else:
                count = sum(item.series in series and item.category in categories for item in bar_choices.items)
            st.caption(f"{count} of {len(bar_choices.items)} bars selected before panel filtering.")
            bar_series_colors, bar_item_colors = _bar_color_controls(
                source, bar_choices, series, categories, bar_items, key,
            )
    config = ExportConfig(
        preset,
        width,
        None if automatic else manual_height,
        unit,
        ratio,
        int(dpi),
        format,
        Typography(
            family=family,
            **font_values,
            bold_title=bold,
            color=font_color,
            annotation_spacing=spacing,
        ),
        Style(**styles),
    )
    content = PlotContentConfig(
        title=title if title != original_title else None,
        subtitle=subtitle,
        show_title=show_title,
        show_subtitle=show_subtitle,
        show_panel_titles=show_panel_titles,
        show_panel_subtitles=show_panel_subtitles,
        show_color_scale=show_color_scale,
        shared_axis_titles=shared_axis_titles,
        **axis_visibility,
        caption=caption,
        **axis_labels,
        correlation_mode=mode,
        annotation_groups=groups,
        statistic_fields=fields,
        statistic_precision=precision,
        statistic_position=position,
        show_observations=show_observations,
        show_regression=show_regression,
        show_intervals=show_intervals,
        show_statistics=show_statistics,
        show_selected_module_annotation=show_selected_module_annotation,
        panel_order=order,
        panel_labels=panel_labels,
        horizontal_reference=horizontal,
        vertical_reference=vertical,
        bar_series=bar_series,
        bar_categories=bar_categories,
        bar_items=bar_items,
        bar_series_colors=bar_series_colors,
        bar_item_colors=bar_item_colors,
    )
    return config, content, separately


def export_dialog_body(spec: PlotExportSpec) -> None:
    config, content, separately = _controls(spec)
    # Keep recovery available even when no bars/panels are selected.
    st.button("Reset to defaults", key=_prefix(spec) + "reset", on_click=_reset, args=(spec,))
    try:
        fitted = fit_automatic_config(spec, config, content)
    except (ExportError, ValueError, ImportError, OSError) as error:
        st.error(str(error))
        return
    except Exception:
        st.error("The figure could not be rebuilt. Try fewer panels or reset the export settings.")
        return
    if fitted.actual_height > config.actual_height:
        st.caption(
            "Automatic height expanded to accommodate the selected labels and annotations. Use manual height for an exact aspect ratio."
        )
    config = fitted
    st.caption(output_summary(config, is_mixed(restore_figure(spec.figure_snapshot))))
    filename_default = default_filename(spec, config)
    filename_key = _prefix(spec) + "filename_" + config.format
    auto_key = filename_key + "_automatic"
    if filename_key in st.session_state and st.session_state[filename_key] == st.session_state.get(
        auto_key
    ):
        st.session_state[filename_key] = filename_default
    st.session_state[auto_key] = filename_default
    filename = st.text_input(
        "Download filename",
        value=None if filename_key in st.session_state else filename_default,
        key=filename_key,
    )
    try:
        artifact = build_artifact(spec, config, content)
        for warning in artifact.warnings:
            st.warning(warning)
        preview_config = replace(config, format="png", dpi=min(config.dpi, 120))
        with st.spinner("Rendering figure preview…"):
            preview, _ = _render_cached(spec, preview_config, content)
        st.image(
            preview,
            width=550,
            caption="Scaled preview; font sizes refer to the final physical figure.",
        )
        # Lazy preparation keeps large final files and per-panel archives off ordinary dialog reruns.
        if st.button("Prepare download", key=_prefix(spec) + "prepare"):
            with st.spinner("Rendering publication file…"):
                if separately:
                    panel_ids = tuple(p.identifier for p in artifact.panels)
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        labels = {p.identifier: p.label for p in artifact.panels}
                        for i, identifier in enumerate(panel_ids):
                            panel_bytes, _ = _render_cached(
                                spec, config, replace(content, panel_order=(identifier,))
                            )
                            archive.writestr(
                                sanitize_filename(
                                    f"panel_{i + 1}_{labels[identifier]}_{sanitize_filename(filename)}",
                                    config.format,
                                ),
                                panel_bytes,
                            )
                    data, mime, name = (
                        buffer.getvalue(),
                        "application/zip",
                        sanitize_filename(filename, "zip"),
                    )
                else:
                    data, _ = _render_cached(spec, config, content)
                    mime, name = MIME_TYPES[config.format], sanitize_filename(
                        filename, config.format
                    )
            st.session_state["figure_export::prepared"] = (
                render_fingerprint(spec, config, content),
                separately,
                data,
                mime,
                name,
            )
        prepared = st.session_state.get("figure_export::prepared")
        if prepared and prepared[:2] == (render_fingerprint(spec, config, content), separately):
            st.download_button(
                "Download figure",
                data=prepared[2],
                mime=prepared[3],
                file_name=sanitize_filename(filename, "zip" if separately else config.format),
                key=_prefix(spec) + "download",
                on_click="ignore",
            )
    except (ExportError, ValueError, ImportError, OSError) as error:
        st.error(str(error))
    except Exception:
        st.error(
            "The figure could not be exported. Try a raster format, smaller dimensions or fewer panels."
        )


if supports_dialog():
    export_dialog = st.dialog("Export figure", width="large")(export_dialog_body)
else:
    export_dialog = None


def render_exportable_plot(
    figure: Any,
    *,
    plot_id: str | None = None,
    metadata: dict | None = None,
    export_spec: PlotExportSpec | None = None,
    **chart_kwargs: Any,
) -> Any:
    from app_helpers.streamlit_compat import plotly_chart

    identifier = plot_id or chart_kwargs.get("key")
    if not identifier:
        raise ValueError("Exportable figures require a stable plot_id or chart key.")
    chart_kwargs.setdefault("key", widget_key(str(identifier), "chart", "display"))
    result = plotly_chart(figure, **chart_kwargs)
    # No JSON snapshots or renderer/browser work until a user opens the exporter.
    button_key = widget_key(str(identifier), "entry", "open")
    if export_dialog is not None:
        if st.button("Export figure", key=button_key, icon="📥"):
            meta = dict(metadata or {})
            if "module" not in meta:
                module = re.search(r"\bM\d+\b", str(figure.layout.title.text or ""))
                if module:
                    meta["module"] = module.group()
            export_dialog(export_spec or PlotExportSpec.from_figure(str(identifier), figure, meta))
    else:
        with st.popover("Export figure"):
            enabled = st.checkbox("Configure export", key=button_key + "_enabled")
            snapshot_key = button_key + "_snapshot"
            if enabled:
                if snapshot_key not in st.session_state:
                    st.session_state[snapshot_key] = export_spec or PlotExportSpec.from_figure(
                        str(identifier), figure, metadata
                    )
                export_dialog_body(st.session_state[snapshot_key])
            else:
                st.session_state.pop(snapshot_key, None)
    return result
