"""Plotly figures for associations and donor-level feature distributions."""

from __future__ import annotations

import html
import math
import textwrap
from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


DIAGNOSIS_COLORS = {
    "Control": "#2C7FB8",
    "MCI": "#D8A500",
    "AD": "#E66101",
}
DIAGNOSIS_SYMBOLS = {"Control": "circle", "MCI": "diamond", "AD": "square"}
MDC_DIRECTION_COLORS = {
    "Higher in AD": "#E66101",
    "Higher in Control": "#2C7FB8",
    "Equal": "#7A8793",
    "Not available": "#C6CCD2",
}
MODULE_TYPE_COLORS = {
    "CT": "#2C7FB8",
    "TS": "#E66101",
}
REGION_COLORS = {
    "AC": "#2C7FB8",
    "DLPFC": "#D8A500",
    "PCG": "#E66101",
}
EDGE_COMPONENT_ORDER = [
    "TS_AC",
    "TS_DLPFC",
    "TS_PCGBA23",
    "CT_AC__DLPFC",
    "CT_AC__PCGBA23",
    "CT_DLPFC__PCGBA23",
]
EDGE_COMPONENT_LABELS = {
    "TS_AC": "AC",
    "TS_DLPFC": "DLPFC",
    "TS_PCGBA23": "PCG",
    "CT_AC__DLPFC": "AC vs DLPFC",
    "CT_AC__PCGBA23": "AC vs PCG",
    "CT_DLPFC__PCGBA23": "DLPFC vs PCG",
}
EDGE_COMPONENT_COLORS = {
    "TS_AC": "#2C7FB8",
    "TS_DLPFC": "#D8A500",
    "TS_PCGBA23": "#E66101",
    "CT_AC__DLPFC": "#6A51A3",
    "CT_AC__PCGBA23": "#2A9D8F",
    "CT_DLPFC__PCGBA23": "#C44E8A",
}
MDC_COMPONENT_LABEL_ORDER = [
    "Total",
    "TS pooled",
    "CT pooled",
    "TS: AC",
    "TS: DLPFC",
    "TS: PCG",
    "CT: AC - DLPFC",
    "CT: AC - PCG",
    "CT: DLPFC - PCG",
]
CONTINUOUS_COLOR_SCALES = {
    "Blue–white–orange": [
        [0.0, "#2C7FB8"],
        [0.5, "#F4F4F2"],
        [1.0, "#E66101"],
    ],
    "Viridis": "Viridis",
    "Cividis": "Cividis",
    "Plasma": "Plasma",
    "Turbo": "Turbo",
    "Blues": "Blues",
    "Reds": "Reds",
    "RdBu": "RdBu",
    "Spectral": "Spectral",
}
MDC_ENRICHMENT_RESOLUTION_TITLES = {
    "pathway": ("Pathway", "pathways"),
    "subcategory": ("KEGG sub-category", "KEGG sub-categories"),
    "category": ("KEGG category", "KEGG categories"),
}


def _mdc_scale_settings(scale: str) -> tuple[str, str, float]:
    """Return the MDC value-column prefix, axis label, and equality reference."""

    if scale == "log2":
        return "log2_mdc", "log2 MDC (AD / Control)", 0.0
    if scale == "raw":
        return "mdc", "MDC ratio (AD / Control)", 1.0
    raise ValueError(f"Unknown MDC display scale: {scale}")


def aggregate_to_long(frame: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Convert CT/TS wide values into the same component schema as resolved data."""
    value_columns = {"CT": f"CT_{scale}", "TS": f"TS_{scale}"}
    parts = []
    for component, value_column in value_columns.items():
        part = frame.copy()
        part["component"] = component
        part["component_class"] = component
        part["component_label"] = f"{component} aggregate"
        part["metric_value"] = part[value_column]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def resolved_to_long(frame: pd.DataFrame, scale: str) -> pd.DataFrame:
    result = frame.copy()
    result["metric_value"] = result[f"metric_{scale}"]
    return result


def edge_volcano_figure(
    candidates: pd.DataFrame,
    bins: pd.DataFrame,
    *,
    module: int,
    scope: str,
    analysis_set: str = "Discovery",
    fdr_scope: str = "global",
    x_metric: str = "hedges_g",
    y_metric: str = "fdr",
    significant_only: bool = False,
    significance_threshold: float = 0.05,
    direction: str = "Either",
    prevalence_column: str | None = None,
    minimum_prevalence: float = 0.0,
    module_definition: str | None = None,
) -> go.Figure:
    """Build a scalable edge-level volcano from density bins and exact candidates."""

    prefix = analysis_set.lower()
    x_column = f"{prefix}_{x_metric}"
    probability_column = (
        f"{prefix}_fdr_{fdr_scope}"
        if y_metric == "fdr"
        else f"{prefix}_p_value"
    )
    if scope == "total":
        points = candidates.copy()
    elif scope in {"CT", "TS"}:
        points = candidates.loc[candidates["component_class"].eq(scope)].copy()
    else:
        points = candidates.loc[candidates["component"].eq(scope)].copy()
    if direction == "Higher in AD":
        points = points.loc[points[f"{prefix}_mean_difference"].gt(0)]
    elif direction == "Higher in Control":
        points = points.loc[points[f"{prefix}_mean_difference"].lt(0)]
    if significant_only:
        points = points.loc[points[probability_column].le(significance_threshold)]
    if prevalence_column and prevalence_column in points:
        points = points.loc[points[prevalence_column].ge(float(minimum_prevalence))]

    figure = go.Figure()
    matching_bins = bins.loc[
        bins["analysis_set"].eq(analysis_set)
        & bins["fdr_scope"].eq(fdr_scope)
        & bins["scope"].eq(scope)
        & bins["x_metric"].eq(x_metric)
        & bins["y_metric"].eq(y_metric)
    ]
    if not significant_only and not matching_bins.empty:
        row = matching_bins.iloc[0]
        x_edges = np.asarray(row["x_edges"], dtype=float)
        y_edges = np.asarray(row["y_edges"], dtype=float)
        counts = np.asarray(row["counts"], dtype=float).reshape(
            len(x_edges) - 1, len(y_edges) - 1
        )
        figure.add_trace(
            go.Heatmap(
                x=(x_edges[:-1] + x_edges[1:]) / 2,
                y=(y_edges[:-1] + y_edges[1:]) / 2,
                z=counts.T,
                colorscale="Greys",
                opacity=0.55,
                colorbar={"title": "All-edge density"},
                hovertemplate=(
                    "Effect bin: %{x:.3g}<br>−log10 significance bin: %{y:.3g}"
                    "<br>Edges: %{z:,.0f}<extra></extra>"
                ),
                name="All edges",
            )
        )

    if not points.empty:
        probability = pd.to_numeric(points[probability_column], errors="coerce")
        points = points.assign(
            _volcano_y=-np.log10(
                np.clip(probability, np.finfo(float).tiny, 1.0)
            ),
            _component_label=points["component"].map(EDGE_COMPONENT_LABELS).fillna(
                points["component"].astype(str)
            ),
        )
        hover_columns = [
            "tissue_a",
            "gene_a",
            "tissue_b",
            "gene_b",
            "_component_label",
            f"{prefix}_mean_ad",
            f"{prefix}_mean_control",
            f"{prefix}_mean_difference",
            f"{prefix}_hedges_g",
            f"{prefix}_p_value",
            f"{prefix}_fdr_global",
            f"{prefix}_fdr_per_module",
            "validation_direction_concordant",
        ]
        if prevalence_column and prevalence_column in points:
            hover_columns.append(prevalence_column)
        template = (
            "%{customdata[0]}:%{customdata[1]} ↔ %{customdata[2]}:%{customdata[3]}"
            "<br>Component: %{customdata[4]}"
            "<br>AD mean: %{customdata[5]:.4g}; Control mean: %{customdata[6]:.4g}"
            "<br>AD−Control: %{customdata[7]:.4g}; Hedges g: %{customdata[8]:.4g}"
            "<br>p=%{customdata[9]:.3g}; Global BH FDR=%{customdata[10]:.3g}"
            "<br>Per-module BH FDR=%{customdata[11]:.3g}"
            "<br>Validation direction concordant: %{customdata[12]}"
        )
        if prevalence_column and prevalence_column in points:
            template += "<br>BONOBO significance prevalence: %{customdata[13]:.1%}"
        template += "<extra></extra>"
        observed_components = points["component"].dropna().astype(str).unique().tolist()
        ordered_components = [
            component for component in EDGE_COMPONENT_ORDER if component in observed_components
        ] + [
            component
            for component in observed_components
            if component not in EDGE_COMPONENT_ORDER
        ]
        for component in ordered_components:
            component_points = points.loc[points["component"].eq(component)]
            marker = {
                "color": EDGE_COMPONENT_COLORS.get(component, "#7A8793"),
                "symbol": "circle" if component.startswith("TS_") else "diamond",
                "size": 7,
                "opacity": 0.82,
                "line": {"width": 0.4, "color": "white"},
            }
            figure.add_trace(
                go.Scattergl(
                    x=component_points[x_column],
                    y=component_points["_volcano_y"],
                    mode="markers",
                    marker=marker,
                    customdata=component_points[hover_columns].astype(object).to_numpy(),
                    hovertemplate=template,
                    name=EDGE_COMPONENT_LABELS.get(component, component),
                    legendgroup=component,
                )
            )

    figure.add_hline(
        y=-math.log10(significance_threshold),
        line_dash="dash",
        line_color="#A33A2B",
        annotation_text=f"{y_metric.upper()}={significance_threshold:g}",
    )
    figure.add_vline(x=0, line_color="#6E7A86", line_width=1)
    x_label = "Hedges’ g (AD − Control)" if x_metric == "hedges_g" else "Mean edge-weight difference (AD − Control)"
    y_label = "−log10(BH FDR)" if y_metric == "fdr" else "−log10(Welch p-value)"
    title = f"Module M{int(module)}: {analysis_set} AD–Control edge volcano"
    subtitle = "Global BH" if fdr_scope == "global" else "Per-module BH"
    if module_definition:
        subtitle = f"{module_definition} · {subtitle}"
    title += f"<br><sup>{subtitle}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=650,
        margin={"l": 65, "r": 35, "t": 95, "b": 115},
        legend={
            "title": {"text": "Tissue component"},
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0.0,
        },
        hoverlabel={"font_size": 13},
    )
    return figure


def module_finder_figure(
    frame: pd.DataFrame,
    *,
    phenotype_label: str,
    feature_label: str,
    correlation_method: str,
    ct_ts_diagnosis: str,
    criterion_label: str,
    selected_module: int | None = None,
    label_count: int = 10,
    minimum_ct_ts_difference: float = 0.0,
    minimum_ad_control_difference: float = 0.0,
) -> go.Figure:
    """Map CT–TS divergence against Control–AD association divergence."""

    points = frame.copy()
    points["module_label"] = points["module"].map(lambda value: f"M{int(value)}")
    points["top_label"] = np.where(
        points["display_rank"].le(int(label_count)), points["module_label"], ""
    )
    hover_columns = [
        "module_label",
        "finder_score",
        "ct_ts_correlation_CT",
        "ct_ts_correlation_TS",
        "ct_ts_delta_correlation",
        "ct_ts_fdr_within_phenotype",
        "correlation_CT_control",
        "correlation_CT_ad",
        "ad_control_delta_correlation_CT",
        "ad_control_fdr_CT",
        "correlation_TS_control",
        "correlation_TS_ad",
        "ad_control_delta_correlation_TS",
        "ad_control_fdr_TS",
        "ad_control_best_component",
        "displayed_pathway",
        "displayed_fdr",
    ]
    for column in hover_columns:
        if column not in points:
            points[column] = np.nan
    correlation_symbol = "ρ" if correlation_method == "Spearman" else "r"
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=points["ct_ts_abs_delta_correlation"],
            y=points["ad_control_max_abs_delta_correlation"],
            mode="markers+text",
            text=points["top_label"],
            textposition="top center",
            textfont={"size": 11, "color": "#263746"},
            marker={
                "color": points["finder_score"],
                "colorscale": "Cividis",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "Rank score"},
                "size": 9,
                "opacity": 0.78,
                "line": {"width": 0.5, "color": "white"},
            },
            customdata=points[hover_columns].astype(object).to_numpy(),
            hovertemplate=(
                "%{customdata[0]} · rank score=%{customdata[1]:.3f}"
                f"<br>{ct_ts_diagnosis} CT {correlation_symbol}=%{{customdata[2]:.3f}}; "
                f"TS {correlation_symbol}=%{{customdata[3]:.3f}}"
                f"<br>CT−TS Δ{correlation_symbol}=%{{customdata[4]:.3f}}; "
                "FDR=%{customdata[5]:.3g}"
                f"<br>Control/AD CT {correlation_symbol}=%{{customdata[6]:.3f}}/"
                "%{customdata[7]:.3f}"
                f"<br>AD−Control CT Δ{correlation_symbol}=%{{customdata[8]:.3f}}; "
                "FDR=%{customdata[9]:.3g}"
                f"<br>Control/AD TS {correlation_symbol}=%{{customdata[10]:.3f}}/"
                "%{customdata[11]:.3f}"
                f"<br>AD−Control TS Δ{correlation_symbol}=%{{customdata[12]:.3f}}; "
                "FDR=%{customdata[13]:.3g}"
                "<br>Stronger diagnosis contrast: %{customdata[14]}"
                "<br>KEGG: %{customdata[15]} · FDR=%{customdata[16]:.3g}"
                "<extra></extra>"
            ),
            name="Modules",
            showlegend=False,
        )
    )
    if selected_module is not None:
        selected = points.loc[points["module"].astype(int).eq(int(selected_module))]
        if not selected.empty:
            figure.add_trace(
                go.Scattergl(
                    x=selected["ct_ts_abs_delta_correlation"],
                    y=selected["ad_control_max_abs_delta_correlation"],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 18,
                        "color": "#2C7FB8",
                        "line": {"width": 3, "color": "#2C7FB8"},
                    },
                    hoverinfo="skip",
                    name=f"Selected M{int(selected_module)}",
                )
            )
    if minimum_ct_ts_difference > 0:
        figure.add_vline(
            x=float(minimum_ct_ts_difference),
            line_dash="dash",
            line_color="#6E7A86",
        )
    if minimum_ad_control_difference > 0:
        figure.add_hline(
            y=float(minimum_ad_control_difference),
            line_dash="dash",
            line_color="#6E7A86",
        )
    figure.update_layout(
        title={
            "text": (
                "Module association-difference map"
                f"<br><sup>{phenotype_label} · {feature_label} · {correlation_method} · "
                f"ranking: {criterion_label}</sup>"
            ),
            "x": 0.01,
        },
        xaxis={
            "title": f"Absolute CT−TS correlation difference ({ct_ts_diagnosis})",
            "rangemode": "tozero",
        },
        yaxis={
            "title": "Maximum absolute Control−AD correlation difference (CT or TS)",
            "rangemode": "tozero",
        },
        template="plotly_white",
        height=690,
        margin={"l": 80, "r": 45, "t": 95, "b": 75},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right"},
    )
    return figure


def _clean_xy(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    clean = frame.dropna(subset=[x, y]).copy()
    if clean.empty:
        return clean
    return clean.loc[np.isfinite(clean[x]) & np.isfinite(clean[y])]


def _hover_payload(
    frame: pd.DataFrame,
    hover_fields: dict[str, str],
) -> tuple[np.ndarray, str]:
    values: list[np.ndarray] = [
        frame["sample_id"].astype(str).to_numpy(),
        frame["diagnosis_group"].astype(str).to_numpy(),
    ]
    template = "Sample: %{customdata[0]}<br>Diagnosis: %{customdata[1]}<br>"
    for index, (field, label) in enumerate(hover_fields.items(), start=2):
        series = frame[field] if field in frame else pd.Series(pd.NA, index=frame.index)
        if pd.api.types.is_numeric_dtype(series):
            displayed = series.map(lambda value: "NA" if pd.isna(value) else f"{value:.4g}")
        else:
            displayed = series.astype("string").fillna("NA")
        values.append(displayed.astype(str).to_numpy())
        template += f"{label}: %{{customdata[{index}]}}<br>"
    return np.stack(values, axis=-1), template


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value != 0 and abs(value) < 0.001:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def _stat_text(
    row: pd.Series | None,
    scale: str,
    resolved: bool,
    correlation_method: str,
) -> str:
    if row is None:
        return "statistics unavailable"
    correlation_method = correlation_method.lower()
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError(f"Unsupported correlation method: {correlation_method}")
    q_available = False
    if resolved:
        if correlation_method == "spearman":
            coefficient = row.get("rho")
            p = row.get("p_spearman")
            q = row.get("q_spearman_all12_global", row.get("q_spearman_global"))
            q_available = pd.notna(q)
            coefficient_label = "ρ"
        else:
            coefficient = row.get(f"r_{scale}")
            p = row.get(f"p_{scale}")
            q = (
                row.get("q_rint_all12_global", row.get("q_rint_within_phenotype"))
                if scale == "rint" else np.nan
            )
            q_available = scale == "rint" and pd.notna(q)
            coefficient_label = "r"
    else:
        component = str(row.get("_display_component", "CT"))
        if correlation_method == "spearman":
            coefficient = row.get(f"rho_{component}")
            p = row.get(f"p_spearman_{component}")
            q = row.get(
                f"q_spearman_{component}_all12_global",
                row.get(f"q_spearman_{component}_global"),
            )
            q_available = pd.notna(q)
            coefficient_label = "ρ"
        else:
            coefficient = row.get(f"r_{scale}_{component}")
            p = row.get(f"p_{scale}_{component}")
            q = (
                row.get(
                    f"q_rint_{component}_all12_global",
                    row.get(f"q_rint_{component}_global"),
                )
                if scale == "rint" else np.nan
            )
            q_available = scale == "rint" and pd.notna(q)
            coefficient_label = "r"
    text = (
        f"n={int(row.get('n', 0))}; "
        f"{coefficient_label}={_format_number(coefficient)}; "
        f"p={_format_number(p)}"
    )
    if q_available:
        text += f"; FDR={_format_number(q)}"
    return text


def _statistics_lookup(
    statistics: pd.DataFrame,
    component: str,
    diagnosis: str,
    resolved: bool,
) -> pd.Series | None:
    if statistics.empty:
        return None
    mask = statistics["diagnosis_group"].eq(diagnosis)
    if resolved:
        mask &= statistics["component"].eq(component)
    match = statistics.loc[mask]
    if match.empty:
        return None
    row = match.iloc[0].copy()
    if not resolved:
        row["_display_component"] = component
    return row


def association_figure(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    phenotype: str,
    phenotype_label: str,
    feature_label: str,
    scale: str,
    scale_label: str,
    diagnoses: Iterable[str],
    module: int,
    resolved: bool,
    color_by: str,
    color_label: str,
    hover_fields: dict[str, str],
    correlation_method: str = "spearman",
    module_definition: str | None = None,
    continuous_colorscale: str = "Blue–white–orange",
    reverse_colorscale: bool = False,
    kegg_subtitles: dict[str, str] | None = None,
) -> go.Figure:
    """Build faceted scatter plots with selected correlation annotations and OLS lines."""
    diagnoses = list(diagnoses)
    components = (
        frame[["component", "component_label"]]
        .drop_duplicates()
        .sort_values("component")
    )
    component_pairs = list(components.itertuples(index=False, name=None))
    ncols = 2 if len(component_pairs) <= 2 else 3
    nrows = math.ceil(len(component_pairs) / ncols)
    kegg_subtitles = kegg_subtitles or {}
    subplot_titles = []
    for component, component_label in component_pairs:
        subtitle = kegg_subtitles.get(str(component))
        if not subtitle:
            subplot_titles.append(html.escape(str(component_label)))
            continue
        wrapped = textwrap.wrap(
            str(subtitle), width=58, break_long_words=False, break_on_hyphens=False
        )
        subtitle_html = "<br>".join(html.escape(line) for line in wrapped)
        subplot_titles.append(
            f"<b>{html.escape(str(component_label))}</b><br>{subtitle_html}"
        )
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.25 if nrows > 1 else 0.08,
    )
    for annotation in fig.layout.annotations:
        annotation.update(font={"size": 12, "color": "#27364B"})

    continuous_color = color_by != "diagnosis_group"
    color_min = color_max = None
    if continuous_color:
        color_values = pd.to_numeric(frame[color_by], errors="coerce")
        finite_colors = color_values.loc[np.isfinite(color_values)]
        if not finite_colors.empty:
            color_min = float(finite_colors.min())
            color_max = float(finite_colors.max())
            if color_min == color_max:
                color_min -= 0.5
                color_max += 0.5

    for index, (component, component_label) in enumerate(component_pairs):
        row = index // ncols + 1
        col = index % ncols + 1
        panel = frame.loc[frame["component"].eq(component)]
        stat_lines = []
        for diagnosis in diagnoses:
            group = _clean_xy(
                panel.loc[panel["diagnosis_group"].eq(diagnosis)],
                "metric_value",
                phenotype,
            )
            if group.empty:
                continue
            if continuous_color:
                valid_color = pd.to_numeric(group[color_by], errors="coerce").notna()
                point_subsets = [(group.loc[valid_color], True), (group.loc[~valid_color], False)]
            else:
                point_subsets = [(group, True)]
            legend_added = False
            for point_group, has_color_value in point_subsets:
                if point_group.empty:
                    continue
                customdata, hover_template = _hover_payload(point_group, hover_fields)
                marker: dict[str, object] = {
                    "symbol": DIAGNOSIS_SYMBOLS[diagnosis],
                    "size": 8,
                    "opacity": 0.74,
                    "line": {"width": 0.5, "color": "white"},
                }
                if continuous_color and has_color_value:
                    marker.update(
                        {
                            "color": pd.to_numeric(point_group[color_by], errors="coerce"),
                            "coloraxis": "coloraxis",
                        }
                    )
                elif continuous_color:
                    marker["color"] = "#A9B1BA"
                else:
                    marker["color"] = DIAGNOSIS_COLORS[diagnosis]
                fig.add_trace(
                    go.Scatter(
                        x=point_group["metric_value"],
                        y=point_group[phenotype],
                        mode="markers",
                        name=diagnosis,
                        legendgroup=diagnosis,
                        showlegend=index == 0 and not legend_added,
                        marker=marker,
                        customdata=customdata,
                        hovertemplate=(
                            hover_template
                            + f"{feature_label}: %{{x:.3f}}<br>"
                            + f"{phenotype_label}: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                legend_added = True
            if len(group) >= 3 and group["metric_value"].nunique() > 1:
                slope, intercept = np.polyfit(group["metric_value"], group[phenotype], 1)
                x_line = np.array([group["metric_value"].min(), group["metric_value"].max()])
                y_line = intercept + slope * x_line
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"{diagnosis} trend",
                        legendgroup=diagnosis,
                        showlegend=False,
                        hoverinfo="skip",
                        line={"color": DIAGNOSIS_COLORS[diagnosis], "width": 2.5},
                    ),
                    row=row,
                    col=col,
                )
            stat_row = _statistics_lookup(statistics, component, diagnosis, resolved)
            stat_lines.append(
                f"<b>{diagnosis}</b>: "
                f"{_stat_text(stat_row, scale, resolved, correlation_method)}"
            )

        axis_number = index + 1
        xref = "x domain" if axis_number == 1 else f"x{axis_number} domain"
        yref = "y domain" if axis_number == 1 else f"y{axis_number} domain"
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref=xref,
            yref=yref,
            text="<br>".join(stat_lines),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font={"size": 10, "color": "#27364B"},
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(90,110,130,0.28)",
            borderwidth=1,
        )

    fig.update_xaxes(title_text=f"{feature_label} ({scale_label})", zeroline=True)
    fig.update_yaxes(title_text=phenotype_label, zeroline=True)
    title_text = f"Module M{int(module)}: {phenotype_label} vs {feature_label}"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=590 if nrows == 1 else 1040,
        margin={"l": 55, "r": 25, "t": 175, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.14,
            "xanchor": "right",
            "x": 1,
            "groupclick": "togglegroup",
        },
        hoverlabel={"font_size": 13},
    )
    if continuous_color and color_min is not None and color_max is not None:
        colorscale = CONTINUOUS_COLOR_SCALES.get(
            continuous_colorscale, CONTINUOUS_COLOR_SCALES["Blue–white–orange"]
        )
        fig.update_layout(
            coloraxis={
                "cmin": color_min,
                "cmax": color_max,
                "colorscale": colorscale,
                "reversescale": bool(reverse_colorscale),
                "colorbar": {"title": {"text": color_label}, "thickness": 16},
            }
        )
    return fig


def distribution_figure(
    frame: pd.DataFrame,
    feature_label: str,
    scale_label: str,
    diagnoses: Iterable[str],
    module: int,
    chart_type: str,
    bins: int = 30,
    module_definition: str | None = None,
) -> go.Figure:
    """Build diagnosis-colored feature distributions without a phenotype axis."""
    diagnoses = list(diagnoses)
    components = (
        frame[["component", "component_label"]]
        .drop_duplicates()
        .sort_values("component")
    )
    component_pairs = list(components.itertuples(index=False, name=None))
    ncols = 2 if len(component_pairs) <= 2 else 3
    nrows = math.ceil(len(component_pairs) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[label for _, label in component_pairs],
        horizontal_spacing=0.08,
        vertical_spacing=0.16 if nrows > 1 else 0.08,
    )

    for index, (component, _) in enumerate(component_pairs):
        row = index // ncols + 1
        col = index % ncols + 1
        panel = frame.loc[frame["component"].eq(component)]
        for diagnosis in diagnoses:
            group = panel.loc[panel["diagnosis_group"].eq(diagnosis)].dropna(
                subset=["metric_value"]
            )
            if group.empty:
                continue
            common = {
                "name": diagnosis,
                "legendgroup": diagnosis,
                "showlegend": index == 0,
            }
            if chart_type == "Histogram":
                trace = go.Histogram(
                    x=group["metric_value"],
                    histnorm="probability density",
                    nbinsx=bins,
                    opacity=0.52,
                    marker_color=DIAGNOSIS_COLORS[diagnosis],
                    hovertemplate=(
                        f"Diagnosis: {diagnosis}<br>"
                        "Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"
                    ),
                    **common,
                )
            else:
                trace = go.Violin(
                    x=group["metric_value"],
                    y=[diagnosis] * len(group),
                    orientation="h",
                    side="positive",
                    width=1.6,
                    points="outliers",
                    box_visible=True,
                    meanline_visible=True,
                    line_color=DIAGNOSIS_COLORS[diagnosis],
                    fillcolor=DIAGNOSIS_COLORS[diagnosis],
                    opacity=0.55,
                    hovertemplate=(
                        f"Diagnosis: {diagnosis}<br>"
                        "Value: %{x:.3f}<extra></extra>"
                    ),
                    **common,
                )
            fig.add_trace(trace, row=row, col=col)

    fig.update_xaxes(title_text=f"{feature_label} ({scale_label})")
    if chart_type == "Histogram":
        fig.update_yaxes(title_text="Probability density")
    title_text = f"Module M{int(module)}: {feature_label} distributions"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    fig.update_layout(
        title={"text": title_text, "x": 0.01},
        template="plotly_white",
        barmode="overlay",
        violinmode="overlay",
        height=470 if nrows == 1 else 790,
        margin={"l": 55, "r": 25, "t": 95, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "right",
            "x": 1,
            "groupclick": "togglegroup",
        },
    )
    return fig


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.dropna(subset=["metric_value"])
        .groupby(["component_label", "diagnosis_group"], observed=True)["metric_value"]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            minimum="min",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    numeric = summary.select_dtypes(include="number").columns.difference(["n"])
    summary[numeric] = summary[numeric].round(4)
    return summary


def module_size_distribution_figure(
    frame: pd.DataFrame,
    module_definition: str | None = None,
) -> go.Figure:
    """Show the module-size distribution for the selected SE2 module types."""
    data = frame.copy()
    data["cluster_type"] = data["cluster_type"].astype(str).str.upper()
    data["module_size"] = pd.to_numeric(data["module_size"], errors="coerce")
    data = data.dropna(subset=["module_size"])
    if data.empty:
        return go.Figure()

    minimum = float(data["module_size"].min())
    maximum = float(data["module_size"].max())
    n_bins = max(3, min(30, int(math.ceil(math.sqrt(len(data))))))
    bin_width = max(1.0, math.ceil((maximum - minimum + 1.0) / n_bins))
    bin_start = math.floor(minimum / bin_width) * bin_width
    bin_end = math.ceil((maximum + 1.0) / bin_width) * bin_width

    figure = go.Figure()
    ordered_types = [
        module_type
        for module_type in ["CT", "TS"]
        if module_type in set(data["cluster_type"])
    ]
    ordered_types.extend(
        sorted(set(data["cluster_type"]).difference(ordered_types))
    )
    for module_type in ordered_types:
        group = data.loc[data["cluster_type"].eq(module_type)]
        figure.add_trace(
            go.Histogram(
                x=group["module_size"],
                name=f"{module_type} modules (n={len(group)})",
                marker={
                    "color": MODULE_TYPE_COLORS.get(module_type, "#7A8793"),
                    "line": {"color": "white", "width": 0.6},
                },
                opacity=0.66,
                xbins={"start": bin_start, "end": bin_end, "size": bin_width},
                hovertemplate=(
                    f"Module type: {module_type}<br>"
                    "Size interval: %{x}<br>Modules: %{y}<extra></extra>"
                ),
            )
        )

    title_text = "Module-size distribution"
    if module_definition:
        title_text += f"<br><sup>{module_definition} · {len(data)} modules shown</sup>"
    figure.update_layout(
        title={"text": title_text, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        barmode="overlay",
        bargap=0.06,
        height=470,
        margin={"l": 65, "r": 25, "t": 90, "b": 60},
        xaxis={"title": "Module size (genes)", "rangemode": "tozero"},
        yaxis={"title": "Number of modules", "rangemode": "tozero"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
    )
    return figure


def module_region_composition_figure(
    frame: pd.DataFrame,
    module_definition: str | None = None,
) -> go.Figure:
    """Rank modules by size and show their AC, DLPFC, and PCG gene composition."""
    required = {
        "module",
        "module_size",
        "cluster_type",
        "n_genes_ac",
        "n_genes_dlpfc",
        "n_genes_pcg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing module-composition columns: {sorted(missing)}")

    data = frame.copy()
    numeric_columns = [
        "module",
        "module_size",
        "n_genes_ac",
        "n_genes_dlpfc",
        "n_genes_pcg",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=["module", "module_size"])
    data = data.sort_values(
        ["module_size", "module"], ascending=[False, True], kind="stable"
    )
    if data.empty:
        return go.Figure()

    module_labels = data["module"].astype(int).map(lambda value: f"M{value}")
    figure = go.Figure()
    region_columns = [
        ("AC", "n_genes_ac"),
        ("DLPFC", "n_genes_dlpfc"),
        ("PCG", "n_genes_pcg"),
    ]
    for region, count_column in region_columns:
        counts = data[count_column].fillna(0).astype(int)
        proportions = np.divide(
            counts.to_numpy(dtype=float),
            data["module_size"].to_numpy(dtype=float),
            out=np.zeros(len(data), dtype=float),
            where=data["module_size"].to_numpy(dtype=float) > 0,
        )
        customdata = np.column_stack(
            [
                data["module_size"].astype(int).astype(str),
                data["cluster_type"].astype(str),
                np.char.mod("%.1f%%", proportions * 100.0),
            ]
        )
        figure.add_trace(
            go.Bar(
                x=counts,
                y=module_labels,
                orientation="h",
                name=region,
                marker={
                    "color": REGION_COLORS[region],
                    "line": {"color": "white", "width": 0.35},
                },
                customdata=customdata,
                hovertemplate=(
                    "Module: %{y}<br>"
                    f"Region: {region}<br>"
                    "Region genes: %{x:,}<br>"
                    "Region proportion: %{customdata[2]}<br>"
                    "Total module genes: %{customdata[0]}<br>"
                    "Module type: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    title_text = "Brain-region composition by module"
    if module_definition:
        title_text += (
            f"<br><sup>{module_definition} · sorted from largest to smallest · "
            f"{len(data)} modules shown</sup>"
        )
    figure.update_layout(
        title={"text": title_text, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        barmode="stack",
        height=max(620, min(2600, 230 + 16 * len(data))),
        margin={"l": 85, "r": 25, "t": 95, "b": 65},
        xaxis={"title": "Genes in module", "rangemode": "tozero"},
        yaxis={
            "title": "Module",
            "categoryorder": "array",
            "categoryarray": module_labels.tolist(),
            "autorange": "reversed",
            "tickfont": {"size": 10},
        },
        legend={
            "title": {"text": "Brain region"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="closest",
    )
    return figure


def module_entropy_figure(
    frame: pd.DataFrame,
    selected_module: int,
    module_definition: str | None = None,
) -> go.Figure:
    """Show module size against normalized tissue-mixing entropy."""

    data = frame.copy()
    data["module_size"] = pd.to_numeric(data["module_size"], errors="coerce")
    data["tissue_entropy_normalized"] = pd.to_numeric(
        data["tissue_entropy_normalized"], errors="coerce"
    )
    data = data.dropna(subset=["module_size", "tissue_entropy_normalized"])
    figure = go.Figure()
    for module_type in ["CT", "TS"]:
        group = data.loc[data["cluster_type"].astype(str).str.upper().eq(module_type)]
        if group.empty:
            continue
        customdata = np.column_stack(
            [
                group["module"].astype(int).map(lambda value: f"M{value}"),
                group["n_tissues"].astype(int),
                group["tissue_entropy"].map(lambda value: f"{float(value):.4f}"),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=group["module_size"],
                y=group["tissue_entropy_normalized"],
                mode="markers",
                name=module_type,
                marker={
                    "color": MODULE_TYPE_COLORS.get(module_type, "#7A8793"),
                    "size": 8,
                    "opacity": 0.72,
                    "line": {"color": "white", "width": 0.5},
                },
                customdata=customdata,
                hovertemplate=(
                    "Module: %{customdata[0]}<br>Module genes: %{x:,}<br>"
                    "Normalized entropy: %{y:.4f}<br>Raw entropy: %{customdata[2]}<br>"
                    "Tissues represented: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    selected = data.loc[data["module"].astype(int).eq(int(selected_module))]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["module_size"],
                y=selected["tissue_entropy_normalized"],
                mode="markers+text",
                text=[f"M{int(selected_module)}"],
                textposition="top center",
                name="Selected module",
                marker={
                    "symbol": "star",
                    "size": 16,
                    "color": "#222222",
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    f"Selected module: M{int(selected_module)}<br>"
                    "Module genes: %{x:,}<br>Normalized entropy: %{y:.4f}<extra></extra>"
                ),
            )
        )
    title = "Module size and continuous tissue-mixing entropy"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_white",
        height=520,
        xaxis={"title": "Module size (genes)", "type": "log"},
        yaxis={"title": "Normalized Shannon entropy", "range": [-0.03, 1.03]},
        legend={"orientation": "h", "y": 1.04, "x": 1, "xanchor": "right"},
        margin={"l": 65, "r": 25, "t": 90, "b": 60},
    )
    return figure


def mdc_entropy_figure(
    frame: pd.DataFrame,
    *,
    scope: str,
    selected_module: int,
    threshold: float,
    scale: str = "log2",
    module_definition: str | None = None,
) -> go.Figure:
    """Show TS or CT MDC against normalized tissue-mixing entropy."""

    scope = scope.lower()
    if scope not in {"ts", "ct"}:
        raise ValueError("MDC entropy scope must be 'ts' or 'ct'")
    value_prefix, axis_title, reference = _mdc_scale_settings(scale)
    value_column = f"{value_prefix}_{scope}"
    raw_column = f"mdc_{scope}"
    log_column = f"log2_mdc_{scope}"
    fdr_column = f"directional_fdr_{scope}"
    direction_column = f"direction_{scope}"

    data = frame.copy()
    numeric_columns = [
        "tissue_entropy_normalized",
        "tissue_entropy",
        "module_size",
        value_column,
        raw_column,
        log_column,
        fdr_column,
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["tissue_entropy_normalized", value_column]).copy()
    data["significant"] = data[fdr_column].lt(threshold)

    figure = go.Figure()
    for direction in ["Higher in Control", "Higher in AD", "Equal", "Not available"]:
        group = data.loc[data[direction_column].fillna("Not available").eq(direction)]
        if group.empty:
            continue
        customdata = np.column_stack(
            [
                group["module"].astype(int).map(lambda value: f"M{value}"),
                group[raw_column],
                group[log_column],
                group[fdr_column],
                group["tissue_entropy"],
                group["module_size"],
                group.get("cluster_type", pd.Series("NA", index=group.index)),
                group["significant"].map(lambda value: "Yes" if value else "No"),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=group["tissue_entropy_normalized"],
                y=group[value_column],
                mode="markers",
                name=direction,
                marker={
                    "color": MDC_DIRECTION_COLORS.get(direction, "#7A8793"),
                    "size": 9,
                    "opacity": 0.78,
                    "symbol": [
                        "diamond" if significant else "circle-open"
                        for significant in group["significant"]
                    ],
                    "line": {"color": "#FFFFFF", "width": 0.6},
                },
                customdata=customdata,
                hovertemplate=(
                    "Module: %{customdata[0]}<br>"
                    "Normalized Shannon entropy: %{x:.4f}<br>"
                    "Raw entropy: %{customdata[4]:.4f}<br>"
                    f"{scope.upper()} MDC: %{{customdata[1]:.4f}}<br>"
                    f"{scope.upper()} log2 MDC: %{{customdata[2]:.4f}}<br>"
                    "Directional FDR: %{customdata[3]:.4g}<br>"
                    f"Significant at FDR &lt; {threshold:.2f}: %{{customdata[7]}}<br>"
                    "Module genes: %{customdata[5]:,}<br>Module type: %{customdata[6]}"
                    "<extra></extra>"
                ),
            )
        )

    x = data["tissue_entropy_normalized"].to_numpy(dtype=float)
    y = data[value_column].to_numpy(dtype=float)
    if len(data) >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([float(np.min(x)), float(np.max(x))])
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=intercept + slope * line_x,
                mode="lines",
                name="OLS trend",
                line={"color": "#343A40", "width": 2, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    selected = data.loc[data["module"].astype(int).eq(int(selected_module))]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["tissue_entropy_normalized"],
                y=selected[value_column],
                mode="markers+text",
                text=[f"M{int(selected_module)}"],
                textposition="top center",
                name=f"Selected M{int(selected_module)}",
                marker={
                    "symbol": "star",
                    "size": 17,
                    "color": "#FFD92F",
                    "line": {"color": "#202124", "width": 1.4},
                },
                hovertemplate=(
                    f"Selected module: M{int(selected_module)}<br>"
                    "Normalized Shannon entropy: %{x:.4f}<br>"
                    f"{scope.upper()} {axis_title}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    rho = np.nan
    p_value = np.nan
    if len(data) >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        result = spearmanr(x, y, nan_policy="omit")
        rho = float(result.statistic)
        p_value = float(result.pvalue)
    statistic_text = (
        "Spearman unavailable"
        if not np.isfinite(rho) or not np.isfinite(p_value)
        else f"Spearman ρ={rho:.3f}, p={p_value:.3g}"
    )
    scale_text = "log2 scale" if scale == "log2" else "raw AD/Control ratio"
    title = (
        f"{scope.upper()} MDC vs normalized tissue-mixing entropy"
        f"<br><sup>{scale_text} · n={len(data)} modules · {statistic_text}"
    )
    if module_definition:
        title += f" · {module_definition}"
    title += "</sup>"

    figure.add_hline(
        y=reference, line_dash="dash", line_color="#657584", line_width=1
    )
    yaxis: dict[str, object] = {"title": axis_title, "zeroline": False}
    if scale == "raw":
        yaxis["rangemode"] = "tozero"
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=540,
        margin={"l": 70, "r": 25, "t": 105, "b": 65},
        xaxis={
            "title": "Normalized Shannon entropy",
            "range": [-0.03, 1.03],
        },
        yaxis=yaxis,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return figure


def pathway_mdc_heatmap_figure(
    frame: pd.DataFrame,
    *,
    scale: str = "log2",
    top_n: int = 25,
    selected_pathway_id: str | None = None,
    module_definition: str | None = None,
    resolution: str = "pathway",
) -> go.Figure:
    """Summarize KEGG-annotated module MDC across tissue components."""

    if frame.empty:
        return go.Figure()
    if resolution not in MDC_ENRICHMENT_RESOLUTION_TITLES:
        raise ValueError(f"Unknown MDC enrichment resolution: {resolution}")
    resolution_singular, resolution_plural = MDC_ENRICHMENT_RESOLUTION_TITLES[
        resolution
    ]
    _, _, reference = _mdc_scale_settings(scale)
    value_column = "mean_log2_mdc" if scale == "log2" else "geometric_mean_mdc"
    axis_title = "Mean log2 MDC" if scale == "log2" else "Geometric mean MDC"
    data = frame.dropna(subset=[value_column]).copy()
    data["absolute_log2_effect"] = pd.to_numeric(
        data["mean_log2_mdc"], errors="coerce"
    ).abs()
    ranking = (
        data.groupby(["pathway_id", "pathway_label"], observed=True)
        .agg(
            maximum_absolute_log2_mdc=("absolute_log2_effect", "max"),
            minimum_enrichment_fdr=("minimum_enrichment_fdr", "min"),
            maximum_module_support=("n_modules", "max"),
        )
        .reset_index()
        .sort_values(
            [
                "maximum_absolute_log2_mdc",
                "minimum_enrichment_fdr",
                "maximum_module_support",
            ],
            ascending=[False, True, False],
            kind="stable",
        )
    )
    selected_ids = ranking.head(max(1, int(top_n)))["pathway_id"].astype(str).tolist()
    if selected_pathway_id is not None and str(selected_pathway_id) not in selected_ids:
        selected_ids = [str(selected_pathway_id), *selected_ids[:-1]]
    data = data.loc[data["pathway_id"].astype(str).isin(selected_ids)].copy()

    component_order = [
        label
        for label in MDC_COMPONENT_LABEL_ORDER
        if label in set(data["component_label"].astype(str))
    ]
    pathway_order = [
        pathway_id
        for pathway_id in selected_ids
        if pathway_id in set(data["pathway_id"].astype(str))
    ]
    data["pathway_id"] = data["pathway_id"].astype(str)
    pivot = data.pivot(
        index="pathway_id", columns="component_label", values=value_column
    ).reindex(index=pathway_order, columns=component_order)
    n_modules = data.pivot(
        index="pathway_id", columns="component_label", values="n_modules"
    ).reindex(index=pathway_order, columns=component_order)
    enrichment_fdr = data.pivot(
        index="pathway_id", columns="component_label", values="minimum_enrichment_fdr"
    ).reindex(index=pathway_order, columns=component_order)
    mdc_significant = data.pivot(
        index="pathway_id",
        columns="component_label",
        values="proportion_mdc_significant",
    ).reindex(index=pathway_order, columns=component_order)
    minimum_mdc_fdr = data.pivot(
        index="pathway_id", columns="component_label", values="minimum_mdc_fdr"
    ).reindex(index=pathway_order, columns=component_order)
    if "n_pathways" in data:
        n_pathways = data.pivot(
            index="pathway_id", columns="component_label", values="n_pathways"
        ).reindex(index=pathway_order, columns=component_order)
    else:
        n_pathways = n_modules.copy()
    pathway_labels = (
        data[["pathway_id", "pathway_label"]]
        .drop_duplicates("pathway_id")
        .set_index("pathway_id")["pathway_label"]
        .to_dict()
    )
    display_labels = []
    for pathway_id in pathway_order:
        label = "<br>".join(textwrap.wrap(str(pathway_labels[pathway_id]), width=46))
        if selected_pathway_id is not None and pathway_id == str(selected_pathway_id):
            label = "★ " + label
        display_labels.append(label)
    customdata = np.stack(
        [
            n_modules.to_numpy(),
            enrichment_fdr.to_numpy(),
            mdc_significant.to_numpy(),
            minimum_mdc_fdr.to_numpy(),
            n_pathways.to_numpy(),
        ],
        axis=-1,
    )
    text = np.empty(pivot.shape, dtype=object)
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            support = n_modules.iat[row_index, column_index]
            text[row_index, column_index] = (
                ""
                if pd.isna(value)
                else f"{float(value):.2f}<br>n={int(support)}"
            )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=component_order,
            y=display_labels,
            colorscale="RdBu_r",
            zmid=reference,
            text=text,
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate=(
                resolution_singular
                + ": %{y}<br>Component: %{x}<br>"
                + axis_title
                + ": %{z:.4f}<br>Enriched modules: %{customdata[0]:.0f}<br>"
                "Supporting pathways: %{customdata[4]:.0f}<br>"
                "Minimum KEGG FDR: %{customdata[1]:.4g}<br>"
                "MDC-significant module proportion: %{customdata[2]:.1%}<br>"
                "Minimum directional MDC FDR: %{customdata[3]:.4g}"
                "<extra></extra>"
            ),
            colorbar={"title": axis_title},
        )
    )
    scale_text = "mean log2 MDC" if scale == "log2" else "geometric mean MDC ratio"
    title = (
        f"{resolution_singular}-annotated MDC across regions and tissue pairs"
        f"<br><sup>Top {len(pathway_order)} {resolution_plural} by maximum absolute mean log2 MDC · "
        f"cells show {scale_text} and enriched-module n"
    )
    if module_definition:
        title += f" · {module_definition}"
    title += "</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(650, min(1800, 260 + 34 * len(pathway_order))),
        margin={"l": 280, "r": 35, "t": 110, "b": 120},
        xaxis={"side": "top", "tickangle": -30},
        yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
    )
    return figure


def pathway_mdc_detail_figure(
    frame: pd.DataFrame,
    *,
    pathway_id: str,
    selected_module: int,
    threshold: float,
    scale: str = "log2",
    module_definition: str | None = None,
    resolution: str = "pathway",
) -> go.Figure:
    """Show module-level MDC values for one selected KEGG group."""

    if resolution not in MDC_ENRICHMENT_RESOLUTION_TITLES:
        raise ValueError(f"Unknown MDC enrichment resolution: {resolution}")
    resolution_singular, _ = MDC_ENRICHMENT_RESOLUTION_TITLES[resolution]
    value_column = "log2_mdc" if scale == "log2" else "mdc"
    _, axis_title, reference = _mdc_scale_settings(scale)
    data = frame.loc[frame["pathway_id"].astype(str).eq(str(pathway_id))].copy()
    data = data.dropna(subset=[value_column])
    data["mdc_significant"] = pd.to_numeric(
        data["directional_fdr"], errors="coerce"
    ).lt(float(threshold))
    component_order = [
        label
        for label in MDC_COMPONENT_LABEL_ORDER
        if label in set(data["component_label"].astype(str))
    ]
    figure = go.Figure()
    for direction in ["Higher in Control", "Higher in AD", "Equal", "Not available"]:
        group = data.loc[data["direction"].fillna("Not available").eq(direction)]
        if group.empty:
            continue
        customdata = np.column_stack(
            [
                group["module"].astype(int).map(lambda value: f"M{value}"),
                group["mdc"],
                group["log2_mdc"],
                group["directional_fdr"],
                group["enrichment_fdr"],
                group["enrichment_scope"],
                group["n_edges"],
                group["mdc_significant"].map(lambda value: "Yes" if value else "No"),
                group.get(
                    "supporting_pathway_count", pd.Series(1, index=group.index)
                ),
                group.get(
                    "supporting_pathway_names",
                    group.get("pathway_label", pd.Series("", index=group.index)),
                ),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=group["component_label"],
                y=group[value_column],
                mode="markers",
                name=direction,
                marker={
                    "color": MDC_DIRECTION_COLORS.get(direction, "#7A8793"),
                    "size": 9,
                    "opacity": 0.75,
                    "symbol": [
                        "diamond" if significant else "circle-open"
                        for significant in group["mdc_significant"]
                    ],
                    "line": {"color": "white", "width": 0.6},
                },
                customdata=customdata,
                hovertemplate=(
                    "Module: %{customdata[0]}<br>Component: %{x}<br>"
                    "MDC: %{customdata[1]:.4f}<br>log2 MDC: %{customdata[2]:.4f}<br>"
                    "Directional MDC FDR: %{customdata[3]:.4g}<br>"
                    "KEGG FDR for component: %{customdata[4]:.4g}<br>"
                    "KEGG scope: %{customdata[5]}<br>Edges: %{customdata[6]:,}<br>"
                    f"MDC significant at FDR &lt; {threshold:.2f}: %{{customdata[7]}}<br>"
                    "Supporting pathways: %{customdata[8]:.0f}<br>"
                    "Pathway names: %{customdata[9]}"
                    "<extra></extra>"
                ),
            )
        )
    selected = data.loc[data["module"].astype(int).eq(int(selected_module))]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["component_label"],
                y=selected[value_column],
                mode="markers+text",
                text=[f"M{int(selected_module)}"] * len(selected),
                textposition="top center",
                name=f"Selected M{int(selected_module)}",
                marker={
                    "symbol": "star",
                    "size": 17,
                    "color": "#FFD92F",
                    "line": {"color": "#202124", "width": 1.4},
                },
                hovertemplate=(
                    f"Selected module: M{int(selected_module)}<br>"
                    "Component: %{x}<br>MDC value: %{y:.4f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(
        y=reference, line_dash="dash", line_color="#657584", line_width=1
    )
    pathway_name = (
        str(data["pathway_label"].iloc[0]) if not data.empty else str(pathway_id)
    )
    title = (
        f"Module MDCs annotated to {resolution_singular}: "
        f"{html.escape(pathway_name)}"
    )
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    yaxis: dict[str, object] = {"title": axis_title, "zeroline": False}
    if scale == "raw":
        yaxis["rangemode"] = "tozero"
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=570,
        margin={"l": 70, "r": 25, "t": 90, "b": 120},
        xaxis={
            "title": "MDC edge component",
            "categoryorder": "array",
            "categoryarray": component_order,
            "tickangle": -30,
        },
        yaxis=yaxis,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return figure


def mdc_resolved_module_figure(
    frame: pd.DataFrame,
    threshold: float,
    module_definition: str | None = None,
    scale: str = "log2",
) -> go.Figure:
    """Resolved MDC bars for one selected module."""

    data = frame.copy()
    value_column, axis_title, reference = _mdc_scale_settings(scale)
    data["significant"] = pd.to_numeric(
        data["directional_fdr"], errors="coerce"
    ).lt(threshold)
    colors = [
        MDC_DIRECTION_COLORS.get(str(direction), "#7A8793")
        for direction in data["direction"]
    ]
    figure = go.Figure(
        go.Bar(
            x=data["component_label"],
            y=data[value_column],
            marker={"color": colors},
            text=[
                ("NA" if pd.isna(value) else f"{float(value):.3f}")
                + (" ★" if significant else "")
                for value, significant in zip(
                    data["mdc"], data["significant"], strict=True
                )
            ],
            textposition="outside",
            customdata=np.column_stack(
                [
                    data["mdc"],
                    data["log2_mdc"],
                    data["directional_fdr"],
                    data["direction"],
                    data["n_edges"],
                ]
            ),
            hovertemplate=(
                "Component: %{x}<br>MDC: %{customdata[0]:.4f}<br>"
                "log2 MDC: %{customdata[1]:.4f}<br>Direction: %{customdata[3]}<br>"
                "Directional FDR: %{customdata[2]:.4g}<br>Edges: %{customdata[4]:,}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=reference, line_dash="dash", line_color="#657584")
    title = f"Module M{int(data['module'].iloc[0])}: tissue-resolved MDC"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_white",
        height=480,
        yaxis={
            "title": axis_title,
            **({"rangemode": "tozero"} if scale == "raw" else {}),
        },
        xaxis={"title": "Resolved edge component"},
        showlegend=False,
        margin={"l": 65, "r": 25, "t": 85, "b": 100},
    )
    return figure


def mdc_resolved_heatmap_figure(
    frame: pd.DataFrame,
    threshold: float,
    selected_module: int,
    module_definition: str | None = None,
    scale: str = "log2",
) -> go.Figure:
    """All-module resolved MDC heatmap with FDR significance markers."""

    data = frame.copy()
    value_column, axis_title, reference = _mdc_scale_settings(scale)
    pivot = data.pivot(index="module", columns="component_label", values=value_column)
    raw = data.pivot(index="module", columns="component_label", values="mdc")
    log2 = data.pivot(index="module", columns="component_label", values="log2_mdc")
    fdr = data.pivot(index="module", columns="component_label", values="directional_fdr")
    module_order = (
        pivot.sub(reference).abs().max(axis=1).sort_values(ascending=False)
        .index.astype(int).tolist()
    )
    if int(selected_module) in module_order:
        module_order.remove(int(selected_module))
        module_order.insert(0, int(selected_module))
    component_order = [
        value
        for value in [
            "TS: AC", "TS: DLPFC", "TS: PCG",
            "CT: AC - DLPFC", "CT: AC - PCG", "CT: DLPFC - PCG",
        ]
        if value in pivot.columns
    ]
    pivot = pivot.reindex(index=module_order, columns=component_order)
    raw = raw.reindex(index=module_order, columns=component_order)
    log2 = log2.reindex(index=module_order, columns=component_order)
    fdr = fdr.reindex(index=module_order, columns=component_order)
    text = np.empty(pivot.shape, dtype=object)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iat[row, column]
            q = fdr.iat[row, column]
            text[row, column] = (
                "NA"
                if pd.isna(value)
                else f"{float(value):.2f}" + (" ★" if pd.notna(q) and q < threshold else "")
            )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=component_order,
            y=[f"M{value}" for value in module_order],
            colorscale="RdBu_r",
            zmid=reference,
            text=text,
            texttemplate="%{text}",
            customdata=np.stack(
                [raw.to_numpy(), log2.to_numpy(), fdr.to_numpy()], axis=-1
            ),
            hovertemplate=(
                "Module: %{y}<br>Component: %{x}<br>MDC: %{customdata[0]:.4f}<br>"
                "log2 MDC: %{customdata[1]:.4f}<br>"
                "Directional FDR: %{customdata[2]:.4g}<extra></extra>"
            ),
            colorbar={"title": axis_title},
        )
    )
    title = "Resolved MDC across modules"
    if module_definition:
        title += f"<br><sup>{module_definition} · ★ FDR &lt; {threshold:.2f}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_white",
        height=max(650, min(2800, 240 + 15 * len(module_order))),
        margin={"l": 85, "r": 30, "t": 90, "b": 95},
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed"},
    )
    return figure


def edge_summary_figure(
    frame: pd.DataFrame,
    metric: str,
    metric_label: str,
    module: int,
    module_definition: str | None = None,
) -> go.Figure:
    """Diagnosis-level box plots for one donor-level edge-summary metric."""

    figure = go.Figure()
    for diagnosis in ["Control", "MCI", "AD"]:
        group = frame.loc[frame["diagnosis_group"].eq(diagnosis)]
        if group.empty:
            continue
        figure.add_trace(
            go.Box(
                x=group["scope_label"],
                y=pd.to_numeric(group[metric], errors="coerce"),
                name=diagnosis,
                legendgroup=diagnosis,
                marker_color=DIAGNOSIS_COLORS[diagnosis],
                boxpoints="outliers",
                hovertemplate=(
                    f"Diagnosis: {diagnosis}<br>Scope: %{{x}}<br>"
                    f"{metric_label}: %{{y:.4g}}<extra></extra>"
                ),
            )
        )
    title = f"Module M{int(module)}: {metric_label} by edge scope and diagnosis"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_white",
        boxmode="group",
        height=540,
        yaxis={"title": metric_label},
        xaxis={"title": "Edge scope", "tickangle": -25},
        legend={"orientation": "h", "y": 1.04, "x": 1, "xanchor": "right"},
        margin={"l": 70, "r": 25, "t": 90, "b": 110},
    )
    return figure


def mdc_module_figure(
    row: pd.Series,
    threshold: float,
    module_definition: str | None = None,
    scale: str = "log2",
) -> go.Figure:
    """Compare total, TS, and CT MDC for the selected module."""
    value_prefix, axis_title, reference = _mdc_scale_settings(scale)
    scopes = [("total", "Total"), ("ts", "Tissue-specific (TS)"), ("ct", "Cross-tissue (CT)")]
    ratios = [row.get(f"mdc_{scope}") for scope, _ in scopes]
    log_ratios = [row.get(f"log2_mdc_{scope}") for scope, _ in scopes]
    displayed_values = [row.get(f"{value_prefix}_{scope}") for scope, _ in scopes]
    fdrs = [row.get(f"directional_fdr_{scope}") for scope, _ in scopes]
    directions = [str(row.get(f"direction_{scope}", "Not available")) for scope, _ in scopes]
    significant = [pd.notna(fdr) and float(fdr) < threshold for fdr in fdrs]
    labels = [label for _, label in scopes]
    colors = [MDC_DIRECTION_COLORS.get(direction, "#7A8793") for direction in directions]
    text = [
        (
            "NA"
            if pd.isna(ratio)
            else f"MDC={float(ratio):.3f}<br>FDR={_format_number(fdr, 3)}"
            + (" ★" if is_significant else "")
        )
        for ratio, fdr, is_significant in zip(ratios, fdrs, significant, strict=True)
    ]
    customdata = np.column_stack(
        [
            ["NA" if pd.isna(value) else f"{float(value):.4f}" for value in ratios],
            ["NA" if pd.isna(value) else f"{float(value):.4f}" for value in log_ratios],
            ["NA" if pd.isna(value) else f"{float(value):.4g}" for value in fdrs],
            directions,
            ["Yes" if value else "No" for value in significant],
        ]
    )
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=displayed_values,
            marker={"color": colors},
            text=text,
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "Component: %{x}<br>MDC (AD / Control): %{customdata[0]}<br>"
                "log2 MDC: %{customdata[1]}<br>Direction: %{customdata[3]}<br>"
                "Directional FDR: %{customdata[2]}<br>"
                f"Significant at FDR &lt; {threshold:.2f}: %{{customdata[4]}}"
                "<extra></extra>"
            ),
        )
    )
    finite = np.asarray(
        [value for value in displayed_values if pd.notna(value)], dtype=float
    )
    if scale == "log2":
        extent = max(0.35, float(np.max(np.abs(finite))) * 1.32) if finite.size else 0.35
        yaxis = {"title": axis_title, "range": [-extent, extent], "zeroline": False}
    else:
        maximum = max(1.15, float(np.max(finite)) * 1.28) if finite.size else 1.15
        yaxis = {"title": axis_title, "range": [0, maximum], "zeroline": False}
    figure.add_hline(
        y=reference, line_dash="dash", line_color="#657584", line_width=1
    )
    title_text = f"Module M{int(row['module'])}: MDC by edge scope"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={
            "text": title_text,
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=430,
        margin={"l": 65, "r": 30, "t": 75, "b": 75},
        yaxis=yaxis,
        showlegend=False,
    )
    return figure


def mdc_overview_figure(
    frame: pd.DataFrame,
    selected_module: int,
    threshold: float,
    module_definition: str | None = None,
    scale: str = "log2",
) -> go.Figure:
    """Show tissue-specific versus cross-tissue MDC across modules."""
    value_prefix, axis_title, reference = _mdc_scale_settings(scale)
    ts_column = f"{value_prefix}_ts"
    ct_column = f"{value_prefix}_ct"
    data = frame.dropna(subset=[ts_column, ct_column]).copy()
    data["ts_significant"] = data["directional_fdr_ts"].lt(threshold)
    data["ct_significant"] = data["directional_fdr_ct"].lt(threshold)
    data["significance"] = np.select(
        [
            data["ts_significant"] & data["ct_significant"],
            data["ts_significant"],
            data["ct_significant"],
        ],
        ["TS and CT", "TS only", "CT only"],
        default="Neither",
    )
    category_colors = {
        "TS and CT": "#6A3D9A",
        "TS only": "#2C7FB8",
        "CT only": "#E66101",
        "Neither": "#A8B0B8",
    }
    figure = go.Figure()
    for category in ["Neither", "TS only", "CT only", "TS and CT"]:
        group = data.loc[data["significance"].eq(category)]
        if group.empty:
            continue
        customdata = [
            [
                f"M{int(row.module)}",
                row.mdc_total,
                row.directional_fdr_total,
                row.mdc_ts,
                row.directional_fdr_ts,
                row.mdc_ct,
                row.directional_fdr_ct,
            ]
            for row in group.itertuples(index=False)
        ]
        figure.add_trace(
            go.Scatter(
                x=group[ts_column],
                y=group[ct_column],
                mode="markers",
                name=category,
                marker={
                    "color": category_colors[category],
                    "size": 8,
                    "opacity": 0.76,
                    "line": {"color": "white", "width": 0.5},
                },
                customdata=customdata,
                hovertemplate=(
                    "Module: %{customdata[0]}<br>"
                    "Total MDC: %{customdata[1]:.3f} · FDR=%{customdata[2]:.3g}<br>"
                    "TS MDC: %{customdata[3]:.3f} · FDR=%{customdata[4]:.3g}<br>"
                    "CT MDC: %{customdata[5]:.3f} · FDR=%{customdata[6]:.3g}<br>"
                    "TS significance: "
                    + ("yes" if category in {"TS only", "TS and CT"} else "no")
                    + "<br>CT significance: "
                    + ("yes" if category in {"CT only", "TS and CT"} else "no")
                    + "<extra></extra>"
                ),
            )
        )

    selected = data.loc[data["module"].astype(int).eq(int(selected_module))]
    if not selected.empty:
        row = selected.iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[row[ts_column]],
                y=[row[ct_column]],
                mode="markers+text",
                name=f"Selected M{int(selected_module)}",
                text=[f"M{int(selected_module)}"],
                textposition="top center",
                marker={
                    "symbol": "star",
                    "size": 17,
                    "color": "#FFD92F",
                    "line": {"color": "#202124", "width": 1.5},
                },
                hovertemplate=(
                    f"Selected module: M{int(selected_module)}<br>"
                    f"TS {axis_title}: %{{x:.3f}}<br>CT {axis_title}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )

    finite = data[[ts_column, ct_column]].to_numpy(dtype=float).ravel()
    if scale == "log2":
        extent = max(0.5, float(np.max(np.abs(finite))) * 1.08) if finite.size else 0.5
        lower, upper = -extent, extent
    else:
        lower = 0.0
        upper = max(1.1, float(np.max(finite)) * 1.08) if finite.size else 1.1
    figure.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"color": "#AAB2BA", "dash": "dot", "width": 1},
        layer="below",
    )
    figure.add_vline(
        x=reference, line_dash="dash", line_color="#657584", line_width=1
    )
    figure.add_hline(
        y=reference, line_dash="dash", line_color="#657584", line_width=1
    )
    title_text = "TS versus CT MDC across modules"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={"text": title_text, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=610,
        margin={"l": 70, "r": 30, "t": 90, "b": 65},
        xaxis={"title": f"TS {axis_title}", "range": [lower, upper]},
        yaxis={"title": f"CT {axis_title}", "range": [lower, upper]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
        },
    )
    return figure


def _hierarchical_order(matrix: pd.DataFrame, axis: str) -> list[str]:
    """Order one heatmap axis by average-linkage clustering of correlation profiles."""
    if axis not in {"rows", "columns"}:
        raise ValueError("axis must be 'rows' or 'columns'")
    labels = matrix.index.tolist() if axis == "rows" else matrix.columns.tolist()
    if len(labels) <= 1:
        return labels

    profiles = matrix.to_numpy(dtype=float)
    if axis == "columns":
        profiles = profiles.T
    # Missing correlations are neutral only for calculating the display order.
    profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
    distances = pdist(profiles, metric="euclidean")
    if distances.size == 0 or np.allclose(distances, 0):
        return labels
    hierarchy = linkage(distances, method="average", optimal_ordering=True)
    return [labels[index] for index in leaves_list(hierarchy)]


def correlation_heatmap_figure(
    frame: pd.DataFrame,
    value_column: str,
    p_column: str,
    fdr_column: str,
    title: str,
    row_order: list[str] | None = None,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
) -> go.Figure:
    """Render a correlation matrix, optionally clustering its display order."""
    if frame.empty:
        return go.Figure()
    outcomes = frame[["outcome", "outcome_label"]].drop_duplicates()
    outcome_order = outcomes["outcome"].tolist()
    outcome_labels = outcomes.set_index("outcome")["outcome_label"].to_dict()
    if row_order is None:
        row_order = frame["heatmap_row"].drop_duplicates().tolist()

    def pivot(column: str, rows: list[str], columns: list[str]) -> pd.DataFrame:
        return (
            frame.pivot_table(
                index="heatmap_row",
                columns="outcome",
                values=column,
                aggfunc="first",
            )
            .reindex(index=rows, columns=columns)
        )

    values = pivot(value_column, row_order, outcome_order)
    if cluster_rows:
        row_order = _hierarchical_order(values, "rows")
        values = values.reindex(index=row_order)
    if cluster_columns:
        outcome_order = _hierarchical_order(values, "columns")
        values = values.reindex(columns=outcome_order)

    n_values = pivot("n", row_order, outcome_order)
    p_values = pivot(p_column, row_order, outcome_order)
    fdr_values = pivot(fdr_column, row_order, outcome_order)
    customdata = np.dstack([n_values.to_numpy(), p_values.to_numpy(), fdr_values.to_numpy()])
    coefficient_label = "Pearson r" if value_column == "pearson_r" else "Spearman ρ"
    figure = go.Figure(
        data=go.Heatmap(
            z=values.to_numpy(),
            x=[outcome_labels[outcome] for outcome in outcome_order],
            y=row_order,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, "#2C7FB8"],
                [0.5, "#F7F7F7"],
                [1.0, "#E66101"],
            ],
            colorbar={"title": coefficient_label, "thickness": 16},
            customdata=customdata,
            hovertemplate=(
                "LIONESS score: %{y}<br>Outcome: %{x}<br>"
                + f"{coefficient_label}: %{{z:.3f}}<br>"
                + "n: %{customdata[0]:.0f}<br>"
                + "p: %{customdata[1]:.3g}<br>"
                + "FDR: %{customdata[2]:.3g}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(560, min(2400, 180 + 12 * len(row_order))),
        margin={"l": 235, "r": 35, "t": 80, "b": 165},
        xaxis={"tickangle": -42, "side": "bottom"},
        yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
    )
    return figure


def prediction_performance_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    block_labels: dict[str, str],
    model_labels: dict[str, str],
    title: str,
) -> go.Figure:
    """Held-out performance across predictor blocks and model variants."""

    selected = frame.loc[frame["metric"].eq(metric) & frame["value"].notna()].copy()
    figure = go.Figure()
    colors = {
        "dummy": "#9AA5B1",
        "covariates": "#6A51A3",
        "network_only": "#2C7FB8",
        "covariates_plus_network": "#E66101",
    }
    for variant, subset in selected.groupby("model_variant", observed=True):
        figure.add_trace(
            go.Bar(
                name=model_labels.get(str(variant), str(variant)),
                x=[block_labels.get(str(value), str(value)) for value in subset["predictor_block"]],
                y=subset["value"],
                marker_color=colors.get(str(variant), "#526273"),
                customdata=np.column_stack([subset["n_held_out"], subset["status"]]),
                hovertemplate=(
                    "%{x}<br>Held-out " + metric + ": %{y:.3f}<br>n=%{customdata[0]}"
                    "<br>Status=%{customdata[1]}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        barmode="group",
        height=560,
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 70, "r": 30, "t": 100, "b": 150},
        xaxis={"tickangle": -35},
        yaxis_title=metric,
    )
    return figure


def prediction_ct_ts_figure(
    frame: pd.DataFrame,
    *,
    outcome_labels: dict[str, str],
    title: str,
) -> go.Figure:
    """Forest plot of held-out CT-minus-TS primary-metric differences."""

    selected = frame.loc[frame["performance_difference"].notna()].copy()
    selected["display"] = selected["outcome"].map(outcome_labels).fillna(selected["outcome"])
    selected["display"] += " · " + selected["comparison"].str.replace("_", " ")
    selected = selected.sort_values("performance_difference")
    figure = go.Figure(
        go.Scatter(
            x=selected["performance_difference"],
            y=selected["display"],
            mode="markers",
            marker={"size": 10, "color": "#2C7FB8"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": selected["ci_high"] - selected["performance_difference"],
                "arrayminus": selected["performance_difference"] - selected["ci_low"],
            },
            customdata=np.column_stack(
                [
                    selected["ci_low"],
                    selected["ci_high"],
                    selected["p_value"],
                    selected["fdr_global"],
                    selected["fdr_within_outcome"],
                ]
            ),
            hovertemplate=(
                "%{y}<br>CT advantage: %{x:.3f}<br>95% CI: %{customdata[0]:.3f} to "
                "%{customdata[1]:.3f}<br>p=%{customdata[2]:.3g}<br>global FDR=%{customdata[3]:.3g}"
                "<br>within-outcome FDR=%{customdata[4]:.3g}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#526273")
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(480, 70 + 36 * len(selected)),
        margin={"l": 260, "r": 30, "t": 80, "b": 70},
        xaxis_title="CT minus TS performance (positive favors CT)",
    )
    return figure


def prediction_heatmap_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    block_labels: dict[str, str],
    outcome_labels: dict[str, str],
    title: str,
) -> go.Figure:
    selected = frame.loc[frame["metric"].eq(metric) & frame["value"].notna()].copy()
    pivot = selected.pivot_table(
        index="predictor_block", columns="outcome", values="value", aggfunc="first"
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[outcome_labels.get(str(value), str(value)) for value in pivot.columns],
            y=[block_labels.get(str(value), str(value)) for value in pivot.index],
            colorscale="Viridis",
            colorbar={"title": metric},
            hovertemplate="Block: %{y}<br>Outcome: %{x}<br>Performance: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(520, 150 + 34 * len(pivot)),
        margin={"l": 220, "r": 35, "t": 80, "b": 170},
        xaxis={"tickangle": -35},
    )
    return figure


def prediction_curve_figure(frame: pd.DataFrame, *, curve: str, title: str) -> go.Figure:
    selected = frame.loc[frame["curve"].eq(curve)].copy()
    figure = go.Figure()
    for label, subset in selected.groupby("class_label", observed=True):
        figure.add_trace(
            go.Scatter(
                x=subset["x"], y=subset["y"], mode="lines+markers",
                name=str(label), marker={"size": 5},
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    if curve in {"ROC", "Calibration"}:
        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#9AA5B1"}, name="Reference")
        )
    axis = {
        "ROC": ("False-positive rate", "True-positive rate"),
        "Precision-recall": ("Recall", "Precision"),
        "Calibration": ("Predicted probability", "Observed fraction"),
    }.get(curve, ("x", "y"))
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=520, xaxis_title=axis[0], yaxis_title=axis[1],
        legend={"orientation": "h", "y": 1.1},
    )
    return figure


def prediction_observed_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    value_column = "predicted_expected" if "predicted_expected" in frame and frame["predicted_expected"].notna().any() else "predicted"
    truth = pd.to_numeric(frame["target"], errors="coerce")
    estimate = pd.to_numeric(frame[value_column], errors="coerce")
    sample_ids = frame.get("sample_id", pd.Series("Anonymous", index=frame.index)).astype(str)
    figure = go.Figure(
        go.Scatter(
            x=truth, y=estimate, mode="markers", marker={"size": 8, "opacity": 0.75, "color": "#2C7FB8"},
            customdata=sample_ids.to_numpy()[:, None],
            hovertemplate=(
                "Sample=%{customdata[0]}<br>Observed=%{x:.3f}<br>"
                "Predicted=%{y:.3f}<extra></extra>"
            ),
        )
    )
    finite = np.r_[truth[np.isfinite(truth)], estimate[np.isfinite(estimate)]]
    if finite.size:
        bounds = [float(np.min(finite)), float(np.max(finite))]
        figure.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", line={"dash": "dash", "color": "#9AA5B1"}, name="Identity"))
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=520, xaxis_title="Observed", yaxis_title="Held-out prediction",
    )
    return figure


def prediction_error_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Held-out residual versus fitted and error-distribution diagnostics."""

    value_column = "predicted_expected" if "predicted_expected" in frame and frame["predicted_expected"].notna().any() else "predicted"
    truth = pd.to_numeric(frame["target"], errors="coerce")
    estimate = pd.to_numeric(frame[value_column], errors="coerce")
    residual = truth - estimate
    sample_ids = frame.get("sample_id", pd.Series("Anonymous", index=frame.index)).astype(str)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Residual versus predicted", "Error distribution"))
    figure.add_trace(
        go.Scatter(
            x=estimate, y=residual, mode="markers",
            marker={"size": 7, "opacity": 0.7, "color": "#2C7FB8"},
            customdata=sample_ids.to_numpy()[:, None],
            hovertemplate=(
                "Sample=%{customdata[0]}<br>Predicted=%{x:.3f}<br>"
                "Residual=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#9AA5B1", row=1, col=1)
    figure.add_trace(
        go.Histogram(x=residual, nbinsx=24, marker_color="#E66101", showlegend=False),
        row=1, col=2,
    )
    figure.update_xaxes(title_text="Held-out prediction", row=1, col=1)
    figure.update_yaxes(title_text="Observed − predicted", row=1, col=1)
    figure.update_xaxes(title_text="Observed − predicted", row=1, col=2)
    figure.update_yaxes(title_text="Donors", row=1, col=2)
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white", height=480, margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    return figure


def prediction_confusion_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    matrix = frame.pivot_table(index="actual", columns="predicted_class", values="n", aggfunc="sum", fill_value=0)
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(), x=matrix.columns.astype(str), y=matrix.index.astype(str),
            colorscale="Blues", text=matrix.to_numpy(), texttemplate="%{text}",
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>n=%{z}<extra></extra>",
            colorbar={"title": "Donors"},
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=460, xaxis_title="Predicted", yaxis_title="Actual",
        margin={"l": 80, "r": 30, "t": 80, "b": 70},
    )
    return figure


def prediction_threshold_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Binary held-out sensitivity/specificity across probability thresholds."""

    probability_columns = [column for column in frame if column.startswith("probability_")]
    if len(probability_columns) != 2:
        return go.Figure()
    positive_column = probability_columns[-1]
    positive_label = positive_column.removeprefix("probability_")
    score = pd.to_numeric(frame[positive_column], errors="coerce").to_numpy()
    numeric_truth = pd.to_numeric(frame["target"], errors="coerce")
    try:
        numeric_positive = float(positive_label)
    except ValueError:
        truth = frame["target"].astype(str).eq(positive_label).to_numpy()
    else:
        truth = numeric_truth.eq(numeric_positive).to_numpy()
    thresholds = np.linspace(0, 1, 101)
    sensitivity, specificity, balanced = [], [], []
    for threshold in thresholds:
        predicted = score >= threshold
        tp = np.sum(predicted & truth)
        fn = np.sum((~predicted) & truth)
        tn = np.sum((~predicted) & (~truth))
        fp = np.sum(predicted & (~truth))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        sensitivity.append(sens)
        specificity.append(spec)
        balanced.append((sens + spec) / 2)
    figure = go.Figure()
    for label, values, color in (
        ("Sensitivity", sensitivity, "#E66101"),
        ("Specificity", specificity, "#2C7FB8"),
        ("Balanced accuracy", balanced, "#6A51A3"),
    ):
        figure.add_trace(
            go.Scatter(x=thresholds, y=values, mode="lines", name=label, line={"color": color})
        )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=460, xaxis_title=f"Threshold for {positive_label}", yaxis_title="Held-out rate",
        yaxis={"range": [0, 1]}, legend={"orientation": "h", "y": 1.1},
    )
    return figure


def prediction_coefficient_figure(frame: pd.DataFrame, *, title: str, limit: int = 30) -> go.Figure:
    selected = frame.nlargest(limit, "abs_standardized_coefficient").sort_values("standardized_coefficient")
    labels = selected.get("display_feature", selected["feature_name"])
    figure = go.Figure(
        go.Bar(
            x=selected["standardized_coefficient"], y=labels, orientation="h",
            marker_color=np.where(selected["standardized_coefficient"] >= 0, "#E66101", "#2C7FB8"),
            customdata=np.column_stack([selected.get("kegg_annotation", pd.Series([""] * len(selected)))]),
            hovertemplate="%{y}<br>Coefficient=%{x:.3g}<br>%{customdata[0]}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=max(520, 100 + 24 * len(selected)), margin={"l": 260, "r": 30, "t": 80, "b": 60},
        xaxis_title="Standardized elastic-net coefficient",
    )
    return figure
