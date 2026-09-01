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
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


PREDICTION_BLOCK_ORDERING_API_VERSION = 1
DISTRIBUTION_GROUPING_API_VERSION = 1


DIAGNOSIS_COLORS = {
    "Control": "#2C7FB8",
    "MCI": "#D8A500",
    "AD": "#E66101",
    "Unclassified": "#7A8793",
}
DIAGNOSIS_SYMBOLS = {
    "Control": "circle", "MCI": "diamond", "AD": "square",
    "Unclassified": "x",
}
CLUSTER_COLORS = {1: "#3B4CC0", 2: "#20A486", 3: "#F6C141", 4: "#D1495B"}
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
ASSOCIATION_GROUP_COLORS = [
    "#2C7FB8", "#E66101", "#20A486", "#8C6BB1", "#D8A500",
    "#C44E8A", "#4D908E", "#6C757D", "#E76F51", "#577590",
]
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


def _mdc_heatmap_color_settings(
    log2_values: np.ndarray,
    *,
    scale: str,
    log2_title: str,
) -> dict[str, object]:
    """Return a reciprocal color scale centered on MDC=1 for MDC heatmaps.

    Heatmap color always uses log2 MDC so reciprocal raw ratios have equal visual
    distance from the neutral value. In raw mode, colorbar ticks are converted
    back to MDC ratios, while cells and hovers continue to show raw values.
    """

    numeric = np.asarray(log2_values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    extent = max(0.05, float(np.max(np.abs(finite)))) if finite.size else 1.0
    colorbar: dict[str, object] = {"title": log2_title}
    if scale == "raw":
        tick_values = np.linspace(-extent, extent, 5)
        colorbar = {
            "title": "MDC ratio<br>(log-symmetric)",
            "tickvals": tick_values.tolist(),
            "ticktext": [f"{float(np.exp2(value)):.3g}" for value in tick_values],
        }
    elif scale != "log2":
        raise ValueError(f"Unknown MDC display scale: {scale}")
    return {
        "zmin": -extent,
        "zmax": extent,
        "zmid": 0.0,
        "colorbar": colorbar,
    }


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
    module_fdr_row: pd.Series | None = None,
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
    if module_fdr_row is not None:
        module_fdr = module_fdr_row.get(
            f"{correlation_method}_fdr_across_modules"
        )
        if pd.notna(module_fdr):
            text = text.rsplit("; FDR=", 1)[0] if "; FDR=" in text else text
            text += f"; module-set FDR={_format_number(module_fdr)}"
    return text


def _pooled_stat_text(
    row: pd.Series | None,
    correlation_method: str,
) -> str:
    """Format statistics recalculated across the currently displayed donors."""

    if row is None:
        return "statistics unavailable"
    correlation_method = correlation_method.lower()
    if correlation_method == "spearman":
        coefficient = row.get("spearman_rho")
        p = row.get("spearman_p")
        q = row.get(
            "spearman_fdr_across_modules",
            row.get("spearman_fdr_displayed_family"),
        )
        coefficient_label = "ρ"
    elif correlation_method == "pearson":
        coefficient = row.get("pearson_r")
        p = row.get("pearson_p")
        q = row.get(
            "pearson_fdr_across_modules",
            row.get("pearson_fdr_displayed_family"),
        )
        coefficient_label = "r"
    else:
        raise ValueError(f"Unsupported correlation method: {correlation_method}")
    text = (
        f"n={int(row.get('n', 0))}; "
        f"{coefficient_label}={_format_number(coefficient)}; "
        f"p={_format_number(p)}"
    )
    if pd.notna(q):
        fdr_label = (
            "module-set FDR"
            if f"{correlation_method}_fdr_across_modules" in row.index
            else "panel FDR"
        )
        text += f"; {fdr_label}={_format_number(q)}"
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


def _pooled_statistics_lookup(
    statistics: pd.DataFrame,
    component: str,
) -> pd.Series | None:
    if statistics.empty or "component" not in statistics:
        return None
    match = statistics.loc[statistics["component"].eq(component)]
    if match.empty:
        return None
    return match.iloc[0]


def _association_subplot_titles(
    component_pairs: Iterable[tuple[object, object]],
    kegg_subtitles: dict[str, str],
    *,
    ncols: int,
) -> tuple[list[str], int]:
    """Return compact multi-line panel titles and the maximum subtitle depth.

    KEGG annotations contain a scope, category, sub-category, pathway, and FDR.
    Keeping that entire string on one line causes titles from adjacent panels to
    overlap, especially in the three-column tissue-resolved layout.  The scope is
    therefore placed on its own line and the biological annotation is wrapped to
    the width available for the current number of subplot columns.
    """

    wrap_width = 42 if int(ncols) >= 3 else 66
    titles: list[str] = []
    maximum_subtitle_lines = 0
    for component, component_label in component_pairs:
        subtitle = str(kegg_subtitles.get(str(component), "") or "").strip()
        if not subtitle:
            titles.append(html.escape(str(component_label)))
            continue
        if ": " in subtitle:
            scope, detail = subtitle.split(": ", 1)
            subtitle_lines = [f"{scope}:"]
            subtitle_lines.extend(
                textwrap.wrap(
                    detail,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [detail]
            )
        else:
            subtitle_lines = textwrap.wrap(
                subtitle,
                width=wrap_width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [subtitle]
        maximum_subtitle_lines = max(maximum_subtitle_lines, len(subtitle_lines))
        subtitle_html = "<br>".join(html.escape(line) for line in subtitle_lines)
        titles.append(
            f"<b>{html.escape(str(component_label))}</b><br>{subtitle_html}"
        )
    return titles, maximum_subtitle_lines


def _association_vertical_spacing(nrows: int, annotation_lines: int) -> float:
    if int(nrows) <= 1:
        return 0.08
    return min(0.46, 0.24 + 0.022 * max(0, int(annotation_lines) - 1))


def _format_association_subplot_title_annotations(figure: go.Figure) -> None:
    """Keep multi-line subplot titles above, rather than inside, plot domains."""

    for annotation in figure.layout.annotations:
        annotation.update(
            font={"size": 12, "color": "#27364B"},
            yanchor="bottom",
            yshift=5,
            align="center",
        )


def _association_stat_line_count(text: str | None) -> int:
    """Count rendered lines in a Plotly HTML annotation."""

    if not text:
        return 0
    return str(text).count("<br>") + 1


def _place_association_statistics_above_panel(
    figure: go.Figure,
    *,
    panel_index: int,
    text: str,
    bgcolor: str = "rgba(255,255,255,0.86)",
    border: bool = True,
) -> int:
    """Place component statistics above the axes and below the KEGG title."""

    line_count = _association_stat_line_count(text)
    if line_count == 0:
        return 0
    axis_number = int(panel_index) + 1
    xref = "x domain" if axis_number == 1 else f"x{axis_number} domain"
    yref = "y domain" if axis_number == 1 else f"y{axis_number} domain"
    # make_subplots creates the component/KEGG titles first, in panel order.
    # Raise each title by the rendered height of its statistics block.
    figure.layout.annotations[panel_index].update(
        yshift=11 + 12 * line_count,
    )
    annotation: dict[str, object] = {
        "x": 0.01,
        "y": 1.0,
        "xref": xref,
        "yref": yref,
        "text": text,
        "showarrow": False,
        "align": "left",
        "xanchor": "left",
        "yanchor": "bottom",
        "yshift": 5,
        "font": {"size": 9, "color": "#27364B"},
        "bgcolor": bgcolor,
    }
    if border:
        annotation.update(
            {
                "bordercolor": "rgba(90,110,130,0.28)",
                "borderwidth": 1,
            }
        )
    figure.add_annotation(**annotation)
    return line_count


def _association_layout_metrics(
    *,
    nrows: int,
    subtitle_lines: int,
    statistic_lines: int,
    single_row_base_height: int,
    multi_row_base_height: int,
    base_top_margin: int,
    bottom_margin: int,
) -> tuple[int, int, float]:
    """Reserve pixel space and position the legend above every panel annotation."""

    subtitle_extra = max(0, int(subtitle_lines) - 1)
    statistic_lines = max(0, int(statistic_lines))
    if int(nrows) == 1:
        height = single_row_base_height + 22 * subtitle_extra + 13 * statistic_lines
        minimum_plot_height = 340
    else:
        height = multi_row_base_height + 42 * subtitle_extra + 24 * statistic_lines
        minimum_plot_height = 720
    panel_stack_pixels = 18 + 14 * (int(subtitle_lines) + 1) + 12 * statistic_lines
    top_margin = max(
        base_top_margin + 14 * subtitle_extra + 12 * statistic_lines,
        panel_stack_pixels + 100,
    )
    height = max(height, top_margin + int(bottom_margin) + minimum_plot_height)
    plot_height = max(1, height - top_margin - int(bottom_margin))
    legend_y = 1.0 + (panel_stack_pixels + 12) / plot_height
    return int(height), int(top_margin), float(legend_y)


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
    pooled_statistics: pd.DataFrame | None = None,
    pooled_label: str = "All donors (pooled)",
    module_fdr_statistics: pd.DataFrame | None = None,
) -> go.Figure:
    """Build faceted scatter plots with diagnosis and optional pooled associations."""
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
    pooled_statistics = (
        pooled_statistics if pooled_statistics is not None else pd.DataFrame()
    )
    module_fdr_statistics = (
        module_fdr_statistics
        if module_fdr_statistics is not None
        else pd.DataFrame()
    )
    maximum_statistic_lines = len(diagnoses) + int(not pooled_statistics.empty)
    subplot_titles, subtitle_lines = _association_subplot_titles(
        component_pairs, kegg_subtitles, ncols=ncols
    )
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=_association_vertical_spacing(
            nrows, subtitle_lines + maximum_statistic_lines
        ),
    )
    _format_association_subplot_title_annotations(fig)

    categorical_cluster_color = color_by == "clusters"
    continuous_color = color_by not in {"diagnosis_group", "clusters"}
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
            if categorical_cluster_color:
                cluster_values = pd.to_numeric(group[color_by], errors="coerce")
                point_subsets = [
                    (group.loc[cluster_values.eq(cluster)], cluster)
                    for cluster in (1, 2, 3, 4)
                ]
                point_subsets.append((group.loc[cluster_values.isna()], None))
            elif continuous_color:
                valid_color = pd.to_numeric(group[color_by], errors="coerce").notna()
                point_subsets = [(group.loc[valid_color], True), (group.loc[~valid_color], False)]
            else:
                point_subsets = [(group, True)]
            legend_added = False
            for point_group, color_value in point_subsets:
                if point_group.empty:
                    continue
                customdata, hover_template = _hover_payload(point_group, hover_fields)
                marker: dict[str, object] = {
                    "symbol": DIAGNOSIS_SYMBOLS[diagnosis],
                    "size": 8,
                    "opacity": 0.74,
                    "line": {"width": 0.5, "color": "white"},
                }
                if categorical_cluster_color:
                    marker["color"] = (
                        CLUSTER_COLORS[int(color_value)]
                        if color_value is not None else "#A9B1BA"
                    )
                elif continuous_color and color_value:
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
                        name=(
                            f"Cluster {int(color_value)}"
                            if categorical_cluster_color and color_value is not None
                            else "Cluster unavailable"
                            if categorical_cluster_color
                            else diagnosis
                        ),
                        legendgroup=(
                            f"cluster_{color_value}"
                            if categorical_cluster_color else diagnosis
                        ),
                        showlegend=(
                            index == 0 and diagnosis == diagnoses[0]
                            if categorical_cluster_color
                            else index == 0 and not legend_added
                        ),
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
            module_fdr_row = _statistics_lookup(
                module_fdr_statistics, component, diagnosis, resolved=True
            )
            stat_lines.append(
                f"<b>{diagnosis}</b>: "
                f"{_stat_text(stat_row, scale, resolved, correlation_method, module_fdr_row)}"
            )

        if not pooled_statistics.empty:
            pooled = _clean_xy(panel, "metric_value", phenotype)
            if len(pooled) >= 3 and pooled["metric_value"].nunique() > 1:
                slope, intercept = np.polyfit(
                    pooled["metric_value"], pooled[phenotype], 1
                )
                x_line = np.array(
                    [pooled["metric_value"].min(), pooled["metric_value"].max()]
                )
                y_line = intercept + slope * x_line
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=pooled_label,
                        legendgroup="__pooled__",
                        showlegend=index == 0,
                        hoverinfo="skip",
                        line={"color": "#1F2937", "width": 3, "dash": "dash"},
                    ),
                    row=row,
                    col=col,
                )
            pooled_stat = _pooled_statistics_lookup(pooled_statistics, component)
            stat_lines.insert(
                0,
                f"<b>{pooled_label}</b>: "
                f"{_pooled_stat_text(pooled_stat, correlation_method)}",
            )

        _place_association_statistics_above_panel(
            fig,
            panel_index=index,
            text="<br>".join(stat_lines),
            bgcolor="rgba(255,255,255,0.86)",
        )

    fig.update_xaxes(title_text=f"{feature_label} ({scale_label})", zeroline=True)
    fig.update_yaxes(title_text=phenotype_label, zeroline=True)
    title_text = f"Module M{int(module)}: {phenotype_label} vs {feature_label}"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    figure_height, top_margin, legend_y = _association_layout_metrics(
        nrows=nrows,
        subtitle_lines=subtitle_lines,
        statistic_lines=maximum_statistic_lines,
        single_row_base_height=590,
        multi_row_base_height=1040,
        base_top_margin=195,
        bottom_margin=55,
    )
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        height=figure_height,
        margin={
            "l": 55,
            "r": 25,
            "t": top_margin,
            "b": 55,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "center",
            "x": 0.5,
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


def _group_stat_lookup(
    statistics: pd.DataFrame,
    component: str,
    grouping_level: object,
) -> pd.Series | None:
    if statistics.empty:
        return None
    mask = statistics["component"].astype(str).eq(str(component))
    mask &= statistics["grouping_level"].astype(str).eq(str(grouping_level))
    selected = statistics.loc[mask]
    return None if selected.empty else selected.iloc[0]


def _configurable_correlation_text(
    row: pd.Series | None,
    correlation_method: str,
    annotation_fields: Iterable[str],
    minimum_group_n: int,
) -> str:
    fields = set(annotation_fields)
    if row is None:
        return "statistics unavailable"
    n = int(row.get("n", 0))
    if n < int(minimum_group_n) or not bool(row.get("eligible", True)):
        reason = row.get("unavailable_reason", "") or "constant or unavailable values"
        return f"n={n}; statistics unavailable ({html.escape(str(reason))})"
    method = correlation_method.lower()
    coefficient_column = "spearman_rho" if method == "spearman" else "pearson_r"
    p_column = f"{method}_p"
    q_column = f"{method}_fdr_across_modules"
    parts: list[str] = []
    if "n" in fields:
        parts.append(f"n={n}")
    if "coefficient" in fields:
        symbol = "ρ" if method == "spearman" else "r"
        parts.append(f"{symbol}={_format_number(row.get(coefficient_column))}")
    if "p" in fields:
        parts.append(f"p={_format_number(row.get(p_column))}")
    if "fdr" in fields:
        parts.append(f"module-set FDR={_format_number(row.get(q_column))}")
    return "; ".join(parts) if parts else ""


def _trend_is_visible(
    row: pd.Series | None,
    correlation_method: str,
    rule: str,
    cutoff: float,
    minimum_group_n: int,
) -> bool:
    if rule == "none" or row is None:
        return False
    if int(row.get("n", 0)) < int(minimum_group_n) or not bool(row.get("eligible", True)):
        return False
    if rule == "all":
        return True
    column = (
        f"{correlation_method.lower()}_p"
        if rule == "p"
        else f"{correlation_method.lower()}_fdr_across_modules"
    )
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return bool(pd.notna(value) and value < float(cutoff))


def grouped_association_figure(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    phenotype: str,
    phenotype_label: str,
    feature_label: str,
    scale_label: str,
    grouping_variable: str,
    grouping_levels: Iterable[object],
    grouping_labels: dict[str, str],
    module: int,
    color_by: str,
    color_label: str,
    hover_fields: dict[str, str],
    correlation_method: str = "spearman",
    annotation_fields: Iterable[str] = ("n", "coefficient", "p", "fdr"),
    trend_line_rule: str = "all",
    significance_cutoff: float = 0.05,
    minimum_group_n: int = 10,
    show_pooled: bool = True,
    pooled_label: str = "All displayed donors (pooled)",
    module_definition: str | None = None,
    continuous_colorscale: str = "Blue–white–orange",
    reverse_colorscale: bool = False,
    categorical_color_fields: Iterable[str] = (),
    kegg_subtitles: dict[str, str] | None = None,
) -> go.Figure:
    """Grouped numeric associations with configurable annotations and trends."""

    levels = list(grouping_levels)
    components = frame[["component", "component_label"]].drop_duplicates()
    component_pairs = list(components.itertuples(index=False, name=None))
    ncols = 2 if len(component_pairs) <= 2 else 3
    nrows = math.ceil(len(component_pairs) / ncols)
    maximum_statistic_lines = len(levels) + int(show_pooled)
    subtitles = kegg_subtitles or {}
    subplot_titles, subtitle_lines = _association_subplot_titles(
        component_pairs, subtitles, ncols=ncols
    )
    figure = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=_association_vertical_spacing(
            nrows, subtitle_lines + maximum_statistic_lines
        ),
    )
    _format_association_subplot_title_annotations(figure)

    group_colors = {
        str(level): (
            DIAGNOSIS_COLORS[str(level)]
            if grouping_variable == "diagnosis_group" and str(level) in DIAGNOSIS_COLORS
            else CLUSTER_COLORS[int(float(level))]
            if grouping_variable == "clusters" and int(float(level)) in CLUSTER_COLORS
            else ASSOCIATION_GROUP_COLORS[index % len(ASSOCIATION_GROUP_COLORS)]
        )
        for index, level in enumerate(levels)
    }
    categorical_color_fields = set(categorical_color_fields)
    discrete_color = color_by in categorical_color_fields or color_by == grouping_variable
    continuous_color = not discrete_color
    color_min = color_max = None
    if continuous_color and color_by in frame:
        color_values = pd.to_numeric(frame[color_by], errors="coerce")
        finite = color_values.loc[np.isfinite(color_values)]
        if not finite.empty:
            color_min, color_max = float(finite.min()), float(finite.max())
            if color_min == color_max:
                color_min -= 0.5
                color_max += 0.5
    discrete_values = []
    if discrete_color and color_by in frame:
        discrete_values = frame[color_by].dropna().drop_duplicates().tolist()
    if color_by == "diagnosis_group":
        discrete_colors = {str(value): DIAGNOSIS_COLORS.get(str(value), "#A9B1BA") for value in discrete_values}
    elif color_by == "clusters":
        discrete_colors = {
            str(value): CLUSTER_COLORS.get(int(float(value)), "#A9B1BA")
            for value in discrete_values
        }
    else:
        discrete_colors = {
            str(value): ASSOCIATION_GROUP_COLORS[index % len(ASSOCIATION_GROUP_COLORS)]
            for index, value in enumerate(discrete_values)
        }

    for panel_index, (component, _component_label) in enumerate(component_pairs):
        row = panel_index // ncols + 1
        col = panel_index % ncols + 1
        panel = frame.loc[frame["component"].eq(component)]
        annotation_lines: list[str] = []
        for level in levels:
            if grouping_variable == "__all__":
                selected = panel.copy()
            else:
                selected = panel.loc[panel[grouping_variable].eq(level)].copy()
            selected = _clean_xy(selected, "metric_value", phenotype)
            if selected.empty:
                continue
            group_key = str(level)
            group_label = grouping_labels.get(group_key, group_key)
            group_color = group_colors[group_key]
            stat = _group_stat_lookup(statistics, component, level)

            # One legend item controls every diagnosis-shaped point trace and its line.
            figure.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="lines+markers", name=group_label,
                    legendgroup=f"association_group::{group_key}",
                    showlegend=panel_index == 0,
                    line={"color": group_color, "width": 2.5},
                    marker={"color": group_color, "size": 8}, hoverinfo="skip",
                ), row=row, col=col,
            )
            for diagnosis in [*DIAGNOSIS_COLORS, None]:
                diagnosis_rows = (
                    selected.loc[selected["diagnosis_group"].eq(diagnosis)]
                    if diagnosis is not None
                    else selected.loc[~selected["diagnosis_group"].isin(DIAGNOSIS_COLORS)]
                )
                if diagnosis_rows.empty:
                    continue
                if discrete_color and color_by != grouping_variable:
                    color_subsets = [
                        (diagnosis_rows.loc[diagnosis_rows[color_by].eq(value)], value)
                        for value in discrete_values
                    ]
                    color_subsets.append((diagnosis_rows.loc[diagnosis_rows[color_by].isna()], None))
                elif continuous_color:
                    numeric_color = pd.to_numeric(diagnosis_rows[color_by], errors="coerce")
                    color_subsets = [
                        (diagnosis_rows.loc[numeric_color.notna()], "__continuous__"),
                        (diagnosis_rows.loc[numeric_color.isna()], None),
                    ]
                else:
                    color_subsets = [(diagnosis_rows, level)]
                for point_rows, point_color in color_subsets:
                    if point_rows.empty:
                        continue
                    customdata, hover_template = _hover_payload(point_rows, hover_fields)
                    marker: dict[str, object] = {
                        "symbol": DIAGNOSIS_SYMBOLS.get(diagnosis, "circle-open"),
                        "size": 8, "opacity": 0.74,
                        "line": {"width": 0.5, "color": "white"},
                    }
                    if point_color == "__continuous__":
                        marker.update({
                            "color": pd.to_numeric(point_rows[color_by], errors="coerce"),
                            "coloraxis": "coloraxis",
                        })
                    elif point_color is None:
                        marker["color"] = "#A9B1BA"
                    elif color_by == grouping_variable:
                        marker["color"] = group_color
                    elif discrete_color:
                        marker["color"] = discrete_colors.get(str(point_color), "#A9B1BA")
                    else:
                        marker["color"] = group_color
                    figure.add_trace(
                        go.Scatter(
                            x=point_rows["metric_value"], y=point_rows[phenotype],
                            mode="markers", name=group_label,
                            legendgroup=f"association_group::{group_key}", showlegend=False,
                            marker=marker, customdata=customdata,
                            hovertemplate=(
                                hover_template + f"Correlation group: {html.escape(group_label)}<br>"
                                + f"{feature_label}: %{{x:.3f}}<br>"
                                + f"{phenotype_label}: %{{y:.3f}}<extra></extra>"
                            ),
                        ), row=row, col=col,
                    )
            if _trend_is_visible(
                stat, correlation_method, trend_line_rule,
                significance_cutoff, minimum_group_n,
            ) and selected["metric_value"].nunique() > 1:
                slope, intercept = np.polyfit(selected["metric_value"], selected[phenotype], 1)
                x_line = np.array([selected["metric_value"].min(), selected["metric_value"].max()])
                figure.add_trace(
                    go.Scatter(
                        x=x_line, y=intercept + slope * x_line, mode="lines",
                        name=f"{group_label} trend",
                        legendgroup=f"association_group::{group_key}", showlegend=False,
                        hoverinfo="skip", line={"color": group_color, "width": 2.5},
                    ), row=row, col=col,
                )
            statistic_text = _configurable_correlation_text(
                stat, correlation_method, annotation_fields, minimum_group_n,
            )
            if statistic_text:
                annotation_lines.append(f"<b>{html.escape(group_label)}</b>: {statistic_text}")

        pooled_stat = _group_stat_lookup(statistics, component, "__pooled__")
        if show_pooled and pooled_stat is not None:
            pooled = _clean_xy(panel, "metric_value", phenotype)
            if _trend_is_visible(
                pooled_stat, correlation_method, trend_line_rule,
                significance_cutoff, minimum_group_n,
            ) and pooled["metric_value"].nunique() > 1:
                slope, intercept = np.polyfit(pooled["metric_value"], pooled[phenotype], 1)
                x_line = np.array([pooled["metric_value"].min(), pooled["metric_value"].max()])
                figure.add_trace(
                    go.Scatter(
                        x=x_line, y=intercept + slope * x_line, mode="lines",
                        name=pooled_label, legendgroup="__pooled__",
                        showlegend=panel_index == 0, hoverinfo="skip",
                        line={"color": "#1F2937", "width": 3, "dash": "dash"},
                    ), row=row, col=col,
                )
            pooled_text = _configurable_correlation_text(
                pooled_stat, correlation_method, annotation_fields, minimum_group_n,
            )
            if pooled_text:
                annotation_lines.insert(0, f"<b>{html.escape(pooled_label)}</b>: {pooled_text}")

        if annotation_lines:
            _place_association_statistics_above_panel(
                figure,
                panel_index=panel_index,
                text="<br>".join(annotation_lines),
                bgcolor="rgba(255,255,255,0.86)",
            )

    figure.update_xaxes(title_text=f"{feature_label} ({scale_label})", zeroline=True)
    figure.update_yaxes(title_text=phenotype_label, zeroline=True)
    title = f"Module M{int(module)}: {phenotype_label} vs {feature_label}"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure_height, top_margin, legend_y = _association_layout_metrics(
        nrows=nrows,
        subtitle_lines=subtitle_lines,
        statistic_lines=maximum_statistic_lines,
        single_row_base_height=610,
        multi_row_base_height=1060,
        base_top_margin=200,
        bottom_margin=55,
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=figure_height,
        margin={
            "l": 55,
            "r": 25,
            "t": top_margin,
            "b": 55,
        },
        legend={
            "orientation": "h", "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "center", "x": 0.5, "groupclick": "togglegroup",
        },
        hoverlabel={"font_size": 13},
    )
    if continuous_color and color_min is not None and color_max is not None:
        figure.update_layout(coloraxis={
            "cmin": color_min, "cmax": color_max,
            "colorscale": CONTINUOUS_COLOR_SCALES.get(
                continuous_colorscale, CONTINUOUS_COLOR_SCALES["Blue–white–orange"]
            ),
            "reversescale": bool(reverse_colorscale),
            "colorbar": {"title": {"text": color_label}, "thickness": 16},
        })
    return figure


def categorical_association_figure(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    category_variable: str,
    category_label: str,
    category_levels: Iterable[object],
    category_labels: dict[str, str],
    feature_label: str,
    scale_label: str,
    module: int,
    minimum_group_n: int = 10,
    annotation_fields: Iterable[str] = ("n", "effect", "p", "fdr"),
    module_definition: str | None = None,
    kegg_subtitles: dict[str, str] | None = None,
    hover_fields: dict[str, str] | None = None,
) -> go.Figure:
    """Generic nominal/ordinal category comparison without numeric correlations."""

    levels = list(category_levels)
    components = frame[["component", "component_label"]].drop_duplicates()
    component_pairs = list(components.itertuples(index=False, name=None))
    ncols = 2 if len(component_pairs) <= 2 else 3
    nrows = math.ceil(len(component_pairs) / ncols)
    maximum_statistic_lines = 2
    subtitles = kegg_subtitles or {}
    titles, subtitle_lines = _association_subplot_titles(
        component_pairs, subtitles, ncols=ncols
    )
    figure = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=_association_vertical_spacing(
            nrows, subtitle_lines + maximum_statistic_lines
        ),
    )
    _format_association_subplot_title_annotations(figure)
    hover_fields = hover_fields or {}
    fields = set(annotation_fields)
    colors = {
        str(level): (
            CLUSTER_COLORS[int(float(level))]
            if category_variable == "clusters" and int(float(level)) in CLUSTER_COLORS
            else DIAGNOSIS_COLORS[str(level)]
            if category_variable == "diagnosis_group" and str(level) in DIAGNOSIS_COLORS
            else ASSOCIATION_GROUP_COLORS[index % len(ASSOCIATION_GROUP_COLORS)]
        )
        for index, level in enumerate(levels)
    }
    for panel_index, (component, _label) in enumerate(component_pairs):
        row = panel_index // ncols + 1
        col = panel_index % ncols + 1
        panel = frame.loc[frame["component"].eq(component)]
        for level in levels:
            selected = panel.loc[panel[category_variable].eq(level)].copy()
            selected = selected.loc[pd.to_numeric(selected["metric_value"], errors="coerce").notna()]
            if selected.empty:
                continue
            label = category_labels.get(str(level), str(level))
            customdata, hover_template = _hover_payload(selected, hover_fields)
            figure.add_trace(
                go.Box(
                    x=np.repeat(label, len(selected)), y=selected["metric_value"],
                    name=label, legendgroup=f"category::{level}",
                    showlegend=panel_index == 0, boxpoints=False,
                    fillcolor="rgba(255,255,255,0)",
                    line={"color": colors[str(level)], "width": 2},
                    hoverinfo="skip",
                ), row=row, col=col,
            )
            for diagnosis in DIAGNOSIS_COLORS:
                points = selected.loc[selected["diagnosis_group"].eq(diagnosis)]
                if points.empty:
                    continue
                point_customdata, point_hover = _hover_payload(points, hover_fields)
                figure.add_trace(
                    go.Scatter(
                        x=np.repeat(label, len(points)), y=points["metric_value"],
                        mode="markers", name=label,
                        legendgroup=f"category::{level}", showlegend=False,
                        marker={
                            "color": colors[str(level)],
                            "symbol": DIAGNOSIS_SYMBOLS[diagnosis],
                            "size": 7, "opacity": 0.72,
                            "line": {"color": "white", "width": 0.4},
                        },
                        customdata=point_customdata,
                        hovertemplate=(
                            point_hover + f"{html.escape(category_label)}: {html.escape(label)}<br>"
                            + f"{feature_label}: %{{y:.3f}}<extra></extra>"
                        ),
                    ), row=row, col=col,
                )
        stat = statistics.loc[statistics["component"].astype(str).eq(str(component))]
        if not stat.empty:
            value = stat.iloc[0]
            pieces: list[str] = []
            if "n" in fields:
                pieces.append(f"n={int(value.get('n', 0))}")
            if "h" in fields:
                pieces.append(f"H={_format_number(value.get('kruskal_h'))}")
            if "effect" in fields:
                pieces.append(f"ε²={_format_number(value.get('epsilon_squared'))}")
            if "p" in fields:
                pieces.append(f"p={_format_number(value.get('categorical_p'))}")
            if "fdr" in fields:
                pieces.append(
                    "module-set FDR="
                    + _format_number(value.get("categorical_fdr_across_modules"))
                )
            tested = value.get("levels_tested", value.get("clusters_tested", "")) or "none"
            excluded = value.get(
                "levels_excluded_small_n", value.get("clusters_excluded_small_n", "")
            ) or "none"
            pieces.append(
                f"<br>tested: {html.escape(str(tested))}; "
                f"excluded (&lt;{int(minimum_group_n)}): {html.escape(str(excluded))}"
            )
            _place_association_statistics_above_panel(
                figure,
                panel_index=panel_index,
                text="; ".join(pieces),
                bgcolor="rgba(255,255,255,0.88)",
                border=False,
            )
    figure.update_xaxes(title_text=category_label)
    figure.update_yaxes(title_text=f"{feature_label} ({scale_label})", zeroline=True)
    title = f"Module M{int(module)}: network-score distributions by {category_label}"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure_height, top_margin, legend_y = _association_layout_metrics(
        nrows=nrows,
        subtitle_lines=subtitle_lines,
        statistic_lines=maximum_statistic_lines,
        single_row_base_height=610,
        multi_row_base_height=1050,
        base_top_margin=195,
        bottom_margin=60,
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white",
        height=figure_height,
        margin={
            "l": 60,
            "r": 25,
            "t": top_margin,
            "b": 60,
        },
        legend={
            "orientation": "h", "yanchor": "bottom",
            "y": legend_y,
            "x": 0.5, "xanchor": "center",
            "groupclick": "togglegroup",
        },
    )
    return figure


def cluster_association_figure(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    feature_label: str,
    scale_label: str,
    module: int,
    module_definition: str | None = None,
    kegg_subtitles: dict[str, str] | None = None,
    hover_fields: dict[str, str] | None = None,
) -> go.Figure:
    """Render nominal Cluster 1–4 score distributions without numeric trends."""

    components = frame[["component", "component_label"]].drop_duplicates()
    component_pairs = list(components.itertuples(index=False, name=None))
    ncols = 2 if len(component_pairs) <= 2 else 3
    nrows = math.ceil(len(component_pairs) / ncols)
    maximum_statistic_lines = 2
    subtitles = kegg_subtitles or {}
    titles, subtitle_lines = _association_subplot_titles(
        component_pairs, subtitles, ncols=ncols
    )
    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=_association_vertical_spacing(
            nrows, subtitle_lines + maximum_statistic_lines
        ),
    )
    _format_association_subplot_title_annotations(figure)
    hover_fields = hover_fields or {}
    for index, (component, _label) in enumerate(component_pairs):
        row = index // ncols + 1
        col = index % ncols + 1
        panel = frame.loc[frame["component"].eq(component)].copy()
        panel["clusters"] = pd.to_numeric(panel["clusters"], errors="coerce")
        for cluster in (1, 2, 3, 4):
            selected = panel.loc[panel["clusters"].eq(cluster)].copy()
            if selected.empty:
                continue
            customdata, hover_template = _hover_payload(selected, hover_fields)
            figure.add_trace(
                go.Box(
                    x=np.repeat(f"Cluster {cluster}", len(selected)),
                    y=selected["metric_value"],
                    name=f"Cluster {cluster}",
                    legendgroup=f"cluster_{cluster}",
                    showlegend=index == 0,
                    boxpoints="all",
                    jitter=0.36,
                    pointpos=0,
                    fillcolor="rgba(255,255,255,0)",
                    line={"color": CLUSTER_COLORS[cluster], "width": 2},
                    marker={
                        "color": CLUSTER_COLORS[cluster],
                        "size": 7,
                        "opacity": 0.72,
                        "line": {"color": "white", "width": 0.4},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        hover_template
                        + f"Cluster: {cluster}<br>{feature_label}: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        stat = statistics.loc[statistics["component"].eq(component)]
        if not stat.empty:
            value = stat.iloc[0]
            tested = value.get("clusters_tested", "") or "none"
            excluded = value.get("clusters_excluded_small_n", "") or "none"
            text = (
                f"n={int(value.get('n', 0))}; H={_format_number(value.get('kruskal_h'))}; "
                f"ε²={_format_number(value.get('epsilon_squared'))}; "
                f"p={_format_number(value.get('categorical_p'))}; "
                f"module-set FDR={_format_number(value.get('categorical_fdr_across_modules'))}"
                f"<br>tested clusters: {tested}; excluded (&lt;5): {excluded}"
            )
            _place_association_statistics_above_panel(
                figure,
                panel_index=index,
                text=text,
                bgcolor="rgba(255,255,255,0.88)",
                border=False,
            )
    figure.update_xaxes(title_text="Nominal donor cluster")
    figure.update_yaxes(title_text=f"{feature_label} ({scale_label})", zeroline=True)
    title = f"Module M{int(module)}: network-score distributions by ROSMAP cluster"
    if module_definition:
        title += f"<br><sup>{module_definition}</sup>"
    figure_height, top_margin, legend_y = _association_layout_metrics(
        nrows=nrows,
        subtitle_lines=subtitle_lines,
        statistic_lines=maximum_statistic_lines,
        single_row_base_height=610,
        multi_row_base_height=1050,
        base_top_margin=195,
        bottom_margin=60,
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=figure_height,
        margin={
            "l": 60,
            "r": 25,
            "t": top_margin,
            "b": 60,
        },
        legend={
            "orientation": "h", "yanchor": "bottom",
            "y": legend_y,
            "x": 0.5, "xanchor": "center",
        },
    )
    return figure


def cluster_association_heatmap_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    significance_threshold: float = 0.05,
) -> go.Figure:
    """Sequential epsilon-squared heatmap kept separate from signed correlations."""

    if frame.empty:
        return go.Figure()
    rows = frame["heatmap_row"].drop_duplicates().tolist()
    columns = frame["component_label"].drop_duplicates().tolist()
    values = frame.pivot_table(
        index="heatmap_row", columns="component_label",
        values="epsilon_squared", aggfunc="first",
    ).reindex(index=rows, columns=columns)
    if cluster_rows:
        rows = _hierarchical_order(values, "rows")
        values = values.reindex(index=rows)
    if cluster_columns:
        columns = _hierarchical_order(values, "columns")
        values = values.reindex(columns=columns)
    def pivot(column: str) -> pd.DataFrame:
        return frame.pivot_table(
            index="heatmap_row", columns="component_label", values=column,
            aggfunc="first",
        ).reindex(index=rows, columns=columns)
    p_values = pivot("categorical_p")
    fdr_values = pivot("categorical_fdr_across_modules")
    n_values = pivot("n")
    stars = np.where(
        np.isfinite(fdr_values.to_numpy(float))
        & (fdr_values.to_numpy(float) < float(significance_threshold)),
        "*",
        "",
    )
    customdata = np.dstack(
        [n_values.to_numpy(), p_values.to_numpy(), fdr_values.to_numpy()]
    )
    figure = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=columns,
            y=rows,
            zmin=0,
            zmax=max(0.10, float(np.nanmax(values.to_numpy(float)))),
            colorscale="Viridis",
            colorbar={"title": "Kruskal ε²", "thickness": 16},
            text=stars,
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate=(
                "Score: %{y}<br>Component: %{x}<br>ε²: %{z:.3f}<br>"
                "n: %{customdata[0]:.0f}<br>p: %{customdata[1]:.3g}<br>"
                "module-set FDR: %{customdata[2]:.3g}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(560, min(2400, 180 + 12 * len(rows))),
        margin={"l": 250, "r": 35, "t": 80, "b": 130},
        xaxis={"tickangle": -35},
        yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
    )
    return figure


def distribution_figure(
    frame: pd.DataFrame,
    feature_label: str,
    scale_label: str,
    diagnoses: Iterable[str],
    module: int,
    chart_type: str,
    bins: int = 30,
    module_definition: str | None = None,
    group_column: str = "diagnosis_group",
    group_label: str = "Diagnosis group",
) -> go.Figure:
    """Build category-colored feature distributions without a phenotype axis.

    ``diagnoses`` is retained as the public argument name for backward
    compatibility; when ``group_column`` differs from ``diagnosis_group`` it
    contains the ordered display values of the selected grouping variable.
    """
    groups = list(diagnoses)
    qualitative_colors = [
        "#2C7FB8", "#E66101", "#20A486", "#D1495B", "#6A51A3",
        "#D8A500", "#C44E8A", "#7A8793", "#1B9E77", "#7570B3",
    ]
    group_colors = {
        group: DIAGNOSIS_COLORS.get(str(group), qualitative_colors[index % len(qualitative_colors)])
        for index, group in enumerate(groups)
    }
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
        for group_value in groups:
            group = panel.loc[panel[group_column].eq(group_value)].dropna(
                subset=["metric_value"]
            )
            if group.empty:
                continue
            common = {
                "name": str(group_value),
                "legendgroup": str(group_value),
                "showlegend": index == 0,
            }
            if chart_type == "Histogram":
                trace = go.Histogram(
                    x=group["metric_value"],
                    histnorm="probability density",
                    nbinsx=bins,
                    opacity=0.52,
                    marker_color=group_colors[group_value],
                    hovertemplate=(
                        f"{group_label}: {group_value}<br>"
                        "Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"
                    ),
                    **common,
                )
            else:
                trace = go.Violin(
                    x=group["metric_value"],
                    y=[str(group_value)] * len(group),
                    orientation="h",
                    side="positive",
                    width=1.6,
                    points="outliers",
                    box_visible=True,
                    meanline_visible=True,
                    line_color=group_colors[group_value],
                    fillcolor=group_colors[group_value],
                    opacity=0.55,
                    hovertemplate=(
                        f"{group_label}: {group_value}<br>"
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


def distribution_summary(
    frame: pd.DataFrame,
    group_column: str = "diagnosis_group",
    group_label: str | None = None,
) -> pd.DataFrame:
    """Summarize distributions by component and the selected category."""

    output_group_column = group_label or group_column
    summary = (
        frame.dropna(subset=["metric_value"])
        .groupby(["component_label", group_column], observed=True)["metric_value"]
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
    if output_group_column != group_column:
        summary = summary.rename(columns={group_column: output_group_column})
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
    value_column = "mean_log2_mdc" if scale == "log2" else "geometric_mean_mdc"
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
    raw_mdc = data.pivot(
        index="pathway_id", columns="component_label", values="geometric_mean_mdc"
    ).reindex(index=pathway_order, columns=component_order)
    mean_log2_mdc = data.pivot(
        index="pathway_id", columns="component_label", values="mean_log2_mdc"
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
            raw_mdc.to_numpy(),
            mean_log2_mdc.to_numpy(),
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
    color_settings = _mdc_heatmap_color_settings(
        mean_log2_mdc.to_numpy(),
        scale=scale,
        log2_title="Mean log2 MDC",
    )
    figure = go.Figure(
        go.Heatmap(
            z=mean_log2_mdc.to_numpy(),
            x=component_order,
            y=display_labels,
            colorscale="RdBu_r",
            text=text,
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate=(
                resolution_singular
                + ": %{y}<br>Component: %{x}<br>"
                "MDC ratio: %{customdata[5]:.4f}<br>"
                "Mean log2 MDC: %{customdata[6]:.4f}<br>"
                "Enriched modules: %{customdata[0]:.0f}<br>"
                "Supporting pathways: %{customdata[4]:.0f}<br>"
                "Minimum KEGG FDR: %{customdata[1]:.4g}<br>"
                "MDC-significant module proportion: %{customdata[2]:.1%}<br>"
                "Minimum directional MDC FDR: %{customdata[3]:.4g}"
                "<extra></extra>"
            ),
            hoverongaps=False,
            **color_settings,
        )
    )
    scale_text = (
        "mean log2 MDC"
        if scale == "log2"
        else "raw geometric mean MDC ratio · log-symmetric colors centered at MDC=1"
    )
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
    value_column, _, _ = _mdc_scale_settings(scale)
    pivot = data.pivot(index="module", columns="component_label", values=value_column)
    raw = data.pivot(index="module", columns="component_label", values="mdc")
    log2 = data.pivot(index="module", columns="component_label", values="log2_mdc")
    fdr = data.pivot(index="module", columns="component_label", values="directional_fdr")
    module_order = (
        log2.abs().max(axis=1).sort_values(ascending=False).index.astype(int).tolist()
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
    color_settings = _mdc_heatmap_color_settings(
        log2.to_numpy(),
        scale=scale,
        log2_title="log2 MDC",
    )
    figure = go.Figure(
        go.Heatmap(
            z=log2.to_numpy(),
            x=component_order,
            y=[f"M{value}" for value in module_order],
            colorscale="RdBu_r",
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
            hoverongaps=False,
            **color_settings,
        )
    )
    title = "Resolved MDC across modules"
    subtitle_parts = [f"★ FDR &lt; {threshold:.2f}"]
    if scale == "raw":
        subtitle_parts.append("raw MDC labels · log-symmetric colors centered at MDC=1")
    if module_definition:
        subtitle_parts.insert(0, module_definition)
    title += f"<br><sup>{' · '.join(subtitle_parts)}</sup>"
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


def _hierarchical_order(matrix: pd.DataFrame, axis: str) -> list[object]:
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


def clustered_correlation_group_order(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_column: str = "module",
    subgroup_columns: tuple[str, ...] = ("metric_family", "diagnosis_group"),
) -> list[object]:
    """Cluster modules while keeping all of each module's score rows together."""

    if frame.empty:
        return []
    required = {group_column, "outcome", value_column, *subgroup_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Grouped correlation clustering is missing columns: "
            + ", ".join(sorted(missing))
        )
    profiles = frame.pivot_table(
        index=group_column,
        columns=[*subgroup_columns, "outcome"],
        values=value_column,
        aggfunc="first",
    )
    return _hierarchical_order(profiles, "rows")


def correlation_heatmap_figure(
    frame: pd.DataFrame,
    value_column: str,
    p_column: str,
    fdr_column: str,
    title: str,
    row_order: list[str] | None = None,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    row_group_labels: dict[str, str] | None = None,
    significance_threshold: float = 0.05,
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
    significance_text = np.where(
        np.isfinite(fdr_values.to_numpy(dtype=float))
        & (fdr_values.to_numpy(dtype=float) < float(significance_threshold)),
        "*",
        "",
    )
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
            text=significance_text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#111111"},
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
    if row_group_labels:
        block_start = 0
        previous_group = row_group_labels.get(row_order[0], "")
        blocks: list[tuple[int, int]] = []
        for index, row in enumerate(row_order[1:], start=1):
            group = row_group_labels.get(row, "")
            if group != previous_group:
                blocks.append((block_start, index - 1))
                block_start = index
                previous_group = group
        blocks.append((block_start, len(row_order) - 1))
        for start, end in blocks:
            figure.add_shape(
                type="rect",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=start - 0.5,
                y1=end + 0.5,
                line={"color": "rgba(20, 35, 50, 0.55)", "width": 1.1},
                fillcolor="rgba(0, 0, 0, 0)",
                layer="above",
            )
    return figure


def prediction_performance_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    block_labels: dict[str, str],
    block_order: Iterable[str] | None = None,
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
        "transcriptomics_only": "#1B9E77",
        "covariates_plus_transcriptomics": "#66A61E",
        "network_plus_transcriptomics": "#D95F02",
        "covariates_plus_network_plus_transcriptomics": "#A6761D",
    }
    model_order = [
        "dummy",
        "covariates",
        "transcriptomics_only",
        "network_only",
        "covariates_plus_transcriptomics",
        "covariates_plus_network",
        "network_plus_transcriptomics",
        "covariates_plus_network_plus_transcriptomics",
    ]
    n_column = "n_held_out" if "n_held_out" in selected else "n_oof"
    status = (
        selected["status"].astype(str)
        if "status" in selected
        else pd.Series("available", index=selected.index)
    )
    selected = selected.assign(_display_status=status)
    selected["_network_predictors"] = pd.to_numeric(
        selected.get(
            "n_network_predictors", pd.Series(0, index=selected.index)
        ), errors="coerce"
    ).fillna(0)
    selected["_transcriptomic_predictors"] = pd.to_numeric(
        selected.get(
            "n_transcriptomic_predictors", pd.Series(0, index=selected.index)
        ), errors="coerce"
    ).fillna(0)
    observed_variants = selected["model_variant"].astype(str).drop_duplicates().tolist()
    ordered_variants = [value for value in model_order if value in observed_variants]
    ordered_variants.extend(
        value for value in observed_variants if value not in ordered_variants
    )
    ordered_blocks = _ordered_prediction_blocks(
        selected["predictor_block"], block_order
    )
    block_rank = {value: index for index, value in enumerate(ordered_blocks)}
    for variant in ordered_variants:
        subset = selected.loc[
            selected["model_variant"].astype(str).eq(variant)
        ].copy()
        subset["_block_rank"] = subset["predictor_block"].astype(str).map(block_rank)
        subset = subset.sort_values("_block_rank", kind="stable")
        figure.add_trace(
            go.Bar(
                name=model_labels.get(str(variant), str(variant)),
                x=[block_labels.get(str(value), str(value)) for value in subset["predictor_block"]],
                y=subset["value"],
                marker_color=colors.get(str(variant), "#526273"),
                customdata=np.column_stack(
                    [
                        subset[n_column], subset["_display_status"],
                        subset["_network_predictors"],
                        subset["_transcriptomic_predictors"],
                    ]
                ),
                hovertemplate=(
                    "%{x}<br>Held-out " + metric + ": %{y:.3f}<br>n=%{customdata[0]}"
                    "<br>Network predictors=%{customdata[2]:.0f}"
                    "<br>Eigengene predictors=%{customdata[3]:.0f}"
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
        xaxis={
            "tickangle": -35,
            "categoryorder": "array",
            "categoryarray": [
                block_labels.get(value, value) for value in ordered_blocks
            ],
        },
        yaxis_title=metric,
    )
    return figure


def _ordered_prediction_blocks(
    values: Iterable[object],
    block_order: Iterable[str] | None,
) -> list[str]:
    """Return observed predictor blocks in the requested biological order."""

    observed = [str(value) for value in pd.Series(values).dropna().drop_duplicates()]
    preferred = [str(value) for value in (block_order or ())]
    ordered = [value for value in preferred if value in observed]
    ordered.extend(value for value in observed if value not in ordered)
    return ordered


def targeted_primary_comparison_figure(
    frame: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    """Forest plot for the three prespecified repeated-CV primary hypotheses."""

    labels = {
        "targeted_CT_minus_all_module_CT": "Targeted CT − all-module CT",
        "targeted_CT_plus_covariates_minus_covariates": "Targeted CT − covariates only",
        "tissue_neutral_targeted_CT_minus_TS": "Tissue-neutral targeted CT − TS",
    }
    selected = frame.loc[frame["performance_difference"].notna()].copy()
    selected["display"] = selected["comparison"].map(labels).fillna(selected["comparison"])
    selected = selected.sort_values("performance_difference")
    fdr_column = (
        "fdr_primary_family" if "fdr_primary_family" in selected else "fdr_global"
    )
    custom = np.column_stack(
        [
            selected["ci_low"], selected["ci_high"], selected["p_value"],
            selected[fdr_column], selected["n_oof"],
        ]
    )
    figure = go.Figure(
        go.Scatter(
            x=selected["performance_difference"],
            y=selected["display"],
            mode="markers",
            marker={"size": 11, "color": "#2C7FB8"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": selected["ci_high"] - selected["performance_difference"],
                "arrayminus": selected["performance_difference"] - selected["ci_low"],
            },
            customdata=custom,
            hovertemplate=(
                "%{y}<br>ROC-AUC difference: %{x:.3f}<br>95% CI: "
                "%{customdata[0]:.3f} to %{customdata[1]:.3f}<br>p=%{customdata[2]:.3g}"
                "<br>Primary-family FDR=%{customdata[3]:.3g}<br>OOF n=%{customdata[4]:.0f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#526273")
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(430, 130 + 70 * len(selected)),
        margin={"l": 275, "r": 35, "t": 80, "b": 65},
        xaxis_title="Paired ROC-AUC difference (positive favors the first model)",
    )
    return figure


def targeted_transform_comparison_figure(
    frame: pd.DataFrame,
    *,
    block_labels: dict[str, str],
    model_labels: dict[str, str],
    title: str,
) -> go.Figure:
    """Forest plot of exploratory transformed-minus-Raw OOF differences."""

    selected = frame.loc[frame["performance_difference"].notna()].copy()
    selected["display"] = selected.apply(
        lambda row: (
            f"{str(row['transformed_scale']).upper()} · "
            f"{block_labels.get(str(row['predictor_block']), row['predictor_block'])} · "
            f"{model_labels.get(str(row['model_variant']), row['model_variant'])}"
        ),
        axis=1,
    )
    selected = selected.sort_values("performance_difference")
    custom = selected[
        ["ci_low", "ci_high", "p_value", "fdr_transform_global", "fdr_transform_within_outcome", "n_oof"]
    ].to_numpy(float)
    figure = go.Figure(
        go.Scatter(
            x=selected["performance_difference"],
            y=selected["display"],
            mode="markers",
            marker={
                "size": 10,
                "color": selected["transformed_scale"].map(
                    {"asinh": "#2C7FB8", "rint": "#E66101"}
                ),
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": selected["ci_high"] - selected["performance_difference"],
                "arrayminus": selected["performance_difference"] - selected["ci_low"],
            },
            customdata=custom,
            hovertemplate=(
                "%{y}<br>Oriented difference: %{x:.3f}<br>95% CI: "
                "%{customdata[0]:.3f} to %{customdata[1]:.3f}"
                "<br>p=%{customdata[2]:.3g}<br>Global FDR=%{customdata[3]:.3g}"
                "<br>Within-outcome FDR=%{customdata[4]:.3g}"
                "<br>OOF n=%{customdata[5]:.0f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#526273")
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(480, 150 + 34 * len(selected)),
        margin={"l": 300, "r": 35, "t": 80, "b": 65},
        xaxis_title="Paired primary-metric difference (positive favors transformed scale)",
    )
    return figure


def targeted_transform_heatmap_figure(
    frame: pd.DataFrame,
    *,
    block_labels: dict[str, str],
    block_order: Iterable[str] | None = None,
    model_labels: dict[str, str],
    title: str,
) -> go.Figure:
    """Heatmap of transformed-minus-Raw primary-metric differences."""

    selected = frame.copy()
    ordered_blocks = _ordered_prediction_blocks(
        selected["predictor_block"], block_order
    )
    block_rank = {value: index for index, value in enumerate(ordered_blocks)}
    model_rank = {value: index for index, value in enumerate(model_labels)}
    selected["_block_rank"] = selected["predictor_block"].astype(str).map(block_rank)
    selected["_model_rank"] = selected["model_variant"].astype(str).map(model_rank)
    selected = selected.sort_values(
        ["_block_rank", "_model_rank"], kind="stable"
    )
    selected["row_label"] = selected.apply(
        lambda row: (
            f"{block_labels.get(str(row['predictor_block']), row['predictor_block'])} · "
            f"{model_labels.get(str(row['model_variant']), row['model_variant'])}"
        ),
        axis=1,
    )
    matrix = selected.pivot_table(
        index="row_label", columns="transformed_scale",
        values="performance_difference", aggfunc="first", sort=False,
    ).reindex(
        index=selected["row_label"].drop_duplicates(),
        columns=[
            value for value in ("asinh", "rint")
            if value in set(selected["transformed_scale"])
        ],
    )
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=[str(value).upper() for value in matrix.columns],
            y=matrix.index,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Δ metric"},
            hovertemplate="%{y}<br>%{x} − Raw: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(480, 150 + 28 * len(matrix)),
        margin={"l": 270, "r": 35, "t": 80, "b": 65},
    )
    return figure


def targeted_eigengene_source_comparison_figure(
    frame: pd.DataFrame,
    *,
    block_labels: dict[str, str],
    model_labels: dict[str, str],
    title: str,
) -> go.Figure:
    """Forest plot of paired OOF performance differences between eigengene sources."""

    selected = frame.loc[frame["performance_difference"].notna()].copy()
    if selected.empty:
        figure = go.Figure()
        figure.update_layout(title=title, template="plotly_white", height=420)
        return figure
    selected["display"] = selected.apply(
        lambda row: (
            f"{block_labels.get(str(row['predictor_block']), row['predictor_block'])} · "
            f"{model_labels.get(str(row['model_variant']), row['model_variant'])}"
        ),
        axis=1,
    )
    selected = selected.sort_values("performance_difference", kind="stable")
    colors = {
        "single_region_full_tissue_l3": "#1B9E77",
        "single_region_complete_case_l3": "#D95F02",
    }
    figure = go.Figure()
    for source, subset in selected.groupby("source_a", observed=True):
        figure.add_trace(
            go.Scatter(
                x=subset["performance_difference"],
                y=subset["display"],
                mode="markers",
                name=str(subset["source_a_label"].iloc[0]),
                marker={"size": 9, "color": colors.get(str(source), "#2C7FB8")},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": subset["ci_high"] - subset["performance_difference"],
                    "arrayminus": subset["performance_difference"] - subset["ci_low"],
                    "thickness": 1.3,
                },
                customdata=np.column_stack(
                    [
                        subset["source_b_label"],
                        subset["p_value"],
                        subset["fdr_global"],
                        subset["fdr_within_outcome"],
                        subset["n_oof"],
                    ]
                ),
                hovertemplate=(
                    "%{y}<br>Source difference: %{x:.3f}"
                    "<br>Reference: %{customdata[0]}"
                    "<br>p=%{customdata[1]:.3g}"
                    "<br>Global FDR=%{customdata[2]:.3g}"
                    "<br>Within-outcome FDR=%{customdata[3]:.3g}"
                    "<br>OOF n=%{customdata[4]:.0f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.add_vline(x=0, line_dash="dash", line_color="#526273")
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(480, 150 + 30 * len(selected)),
        margin={"l": 310, "r": 35, "t": 85, "b": 65},
        xaxis_title=(
            "Paired primary-metric difference (Source A − Source B; "
            "positive favors Source A)"
        ),
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def targeted_panel_overlap_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Show fold-level panel Jaccard overlap with Raw selections."""

    selected = frame.loc[frame["jaccard"].notna()].copy()
    figure = go.Figure()
    for transform, subset in selected.groupby("transformed_scale", observed=True):
        figure.add_trace(
            go.Box(
                name=str(transform).upper(),
                y=subset["jaccard"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                customdata=subset[["raw_k", "transformed_k", "intersection_modules"]].to_numpy(),
                hovertemplate=(
                    "%{fullData.name}<br>Jaccard=%{y:.3f}<br>Raw K=%{customdata[0]}"
                    "<br>Transformed K=%{customdata[1]}<br>Shared=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=480,
        yaxis={"title": "Selected-panel Jaccard with Raw", "range": [0, 1.02]},
    )
    return figure


def targeted_selection_frequency_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    limit: int = 40,
) -> go.Figure:
    """Show outer-fold stability of the display-only consensus panel."""

    selected = frame.copy()
    selected["module_label"] = selected["module"].map(lambda value: f"M{int(value)}")
    selected = selected.sort_values(
        ["outer_selection_frequency", "mean_stable_rank"],
        ascending=[False, True],
    ).head(int(limit)).sort_values("outer_selection_frequency")
    custom_columns = ["mean_stable_rank", "median_incremental_score"]
    if "kegg_annotation" in selected:
        custom_columns.append("kegg_annotation")
    custom = selected[custom_columns].fillna("Unavailable").to_numpy(object)
    hover = (
        "Module %{y}<br>Outer-fold selection frequency: %{x:.1%}"
        "<br>Mean stable rank: %{customdata[0]}"
        "<br>Median incremental score: %{customdata[1]}"
    )
    if len(custom_columns) == 3:
        hover += "<br>%{customdata[2]}"
    hover += "<extra></extra>"
    figure = go.Figure(
        go.Bar(
            x=selected["outer_selection_frequency"],
            y=selected["module_label"],
            orientation="h",
            marker_color=np.where(selected["consensus_selected"], "#E66101", "#2C7FB8"),
            customdata=custom,
            hovertemplate=hover,
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=max(500, 145 + 24 * len(selected)),
        margin={"l": 90, "r": 35, "t": 80, "b": 65},
        xaxis={"title": "Outer-fold selection frequency", "tickformat": ".0%", "range": [0, 1.02]},
        yaxis_title="Module",
    )
    return figure


def targeted_fold_robustness_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    block_labels: dict[str, str],
    block_order: Iterable[str] | None = None,
    title: str,
) -> go.Figure:
    """Repeat/fold metric distributions for selected targeted models."""

    selected = frame.loc[frame["metric"].eq(metric) & frame["value"].notna()].copy()
    figure = go.Figure()
    ordered_blocks = _ordered_prediction_blocks(
        selected["predictor_block"], block_order
    )
    for block in ordered_blocks:
        subset = selected.loc[selected["predictor_block"].astype(str).eq(block)]
        figure.add_trace(
            go.Box(
                name=block_labels.get(str(block), str(block)),
                y=subset["value"],
                boxpoints="all",
                jitter=0.28,
                pointpos=0,
                customdata=np.column_stack([subset["outer_repeat"], subset["outer_fold"]]),
                hovertemplate=(
                    "%{fullData.name}<br>" + metric + ": %{y:.3f}"
                    "<br>Repeat %{customdata[0]}, fold %{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=540,
        showlegend=False,
        margin={"l": 70, "r": 35, "t": 80, "b": 150},
        xaxis={"tickangle": -30},
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
    block_order: Iterable[str] | None = None,
    outcome_labels: dict[str, str],
    title: str,
) -> go.Figure:
    selected = frame.loc[frame["metric"].eq(metric) & frame["value"].notna()].copy()
    pivot = selected.pivot_table(
        index="predictor_block", columns="outcome", values="value", aggfunc="first"
    )
    pivot = pivot.reindex(
        _ordered_prediction_blocks(pivot.index, block_order)
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


def classification_diagnostic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one-vs-rest ROC, PR, and calibration rows from public OOF predictions."""

    rows: list[dict[str, object]] = []
    truth = frame["target"].astype(str)
    for column in sorted(value for value in frame if value.startswith("probability_")):
        label = column.removeprefix("probability_")
        score = pd.to_numeric(frame[column], errors="coerce")
        valid = score.notna() & truth.notna()
        binary = truth.loc[valid].eq(label).astype(int).to_numpy()
        probability = score.loc[valid].to_numpy(dtype=float)
        if len(binary) < 2 or np.unique(binary).size != 2:
            continue
        false_positive, true_positive, _ = roc_curve(binary, probability)
        rows.extend(
            {"curve": "ROC", "class_label": f"Class {label}", "x": x, "y": y}
            for x, y in zip(false_positive, true_positive, strict=True)
        )
        precision, recall, _ = precision_recall_curve(binary, probability)
        rows.extend(
            {"curve": "Precision-recall", "class_label": f"Class {label}", "x": x, "y": y}
            for x, y in zip(recall, precision, strict=True)
        )
        observed, predicted = calibration_curve(binary, probability, n_bins=10, strategy="quantile")
        rows.extend(
            {"curve": "Calibration", "class_label": f"Class {label}", "x": x, "y": y}
            for x, y in zip(predicted, observed, strict=True)
        )
    return pd.DataFrame(rows)


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
    positive_column = next(
        (
            column
            for column in ("probability_AD", "probability_1", "probability_1.0")
            if column in probability_columns
        ),
        probability_columns[-1],
    )
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
