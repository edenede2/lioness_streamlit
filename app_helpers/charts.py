"""Plotly figures for associations and donor-level feature distributions."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


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
            q = row.get("q_spearman_global")
            q_available = "q_spearman_global" in row.index
            coefficient_label = "ρ"
        else:
            coefficient = row.get(f"r_{scale}")
            p = row.get(f"p_{scale}")
            q = row.get("q_rint_within_phenotype") if scale == "rint" else np.nan
            q_available = scale == "rint"
            coefficient_label = "r"
    else:
        component = str(row.get("_display_component", "CT"))
        if correlation_method == "spearman":
            coefficient = row.get(f"rho_{component}")
            p = row.get(f"p_spearman_{component}")
            q = np.nan
            coefficient_label = "ρ"
        else:
            coefficient = row.get(f"r_{scale}_{component}")
            p = row.get(f"p_{scale}_{component}")
            q = row.get(f"q_rint_{component}_global") if scale == "rint" else np.nan
            q_available = scale == "rint"
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
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[label for _, label in component_pairs],
        horizontal_spacing=0.08,
        vertical_spacing=0.16 if nrows > 1 else 0.08,
    )

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
        height=510 if nrows == 1 else 840,
        margin={"l": 55, "r": 25, "t": 95, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "right",
            "x": 1,
            "groupclick": "togglegroup",
        },
        hoverlabel={"font_size": 13},
    )
    if continuous_color and color_min is not None and color_max is not None:
        fig.update_layout(
            coloraxis={
                "cmin": color_min,
                "cmax": color_max,
                "colorscale": [
                    [0.0, "#2C7FB8"],
                    [0.5, "#F4F4F2"],
                    [1.0, "#E66101"],
                ],
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


def mdc_module_figure(
    row: pd.Series,
    threshold: float,
    module_definition: str | None = None,
) -> go.Figure:
    """Compare total, TS, and CT MDC for the selected module on a centered log2 scale."""
    scopes = [("total", "Total"), ("ts", "Tissue-specific (TS)"), ("ct", "Cross-tissue (CT)")]
    ratios = [row.get(f"mdc_{scope}") for scope, _ in scopes]
    log_ratios = [row.get(f"log2_mdc_{scope}") for scope, _ in scopes]
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
            ["NA" if pd.isna(value) else f"{float(value):.4g}" for value in fdrs],
            directions,
            ["Yes" if value else "No" for value in significant],
        ]
    )
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=log_ratios,
            marker={"color": colors},
            text=text,
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "Component: %{x}<br>MDC (AD / Control): %{customdata[0]}<br>"
                "log2 MDC: %{y:.3f}<br>Direction: %{customdata[2]}<br>"
                "Directional FDR: %{customdata[1]}<br>"
                f"Significant at FDR &lt; {threshold:.2f}: %{{customdata[3]}}"
                "<extra></extra>"
            ),
        )
    )
    finite = np.asarray([value for value in log_ratios if pd.notna(value)], dtype=float)
    extent = max(0.35, float(np.max(np.abs(finite))) * 1.32) if finite.size else 0.35
    figure.add_hline(y=0, line_dash="dash", line_color="#657584", line_width=1)
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
        yaxis={
            "title": "log2 MDC (AD / Control)",
            "range": [-extent, extent],
            "zeroline": False,
        },
        showlegend=False,
    )
    return figure


def mdc_overview_figure(
    frame: pd.DataFrame,
    selected_module: int,
    threshold: float,
    module_definition: str | None = None,
) -> go.Figure:
    """Show tissue-specific versus cross-tissue MDC across modules."""
    data = frame.dropna(subset=["log2_mdc_ts", "log2_mdc_ct"]).copy()
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
                x=group["log2_mdc_ts"],
                y=group["log2_mdc_ct"],
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
                x=[row["log2_mdc_ts"]],
                y=[row["log2_mdc_ct"]],
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
                    "TS log2 MDC: %{x:.3f}<br>CT log2 MDC: %{y:.3f}<extra></extra>"
                ),
            )
        )

    finite = data[["log2_mdc_ts", "log2_mdc_ct"]].to_numpy(dtype=float).ravel()
    extent = max(0.5, float(np.max(np.abs(finite))) * 1.08) if finite.size else 0.5
    figure.add_shape(
        type="line",
        x0=-extent,
        y0=-extent,
        x1=extent,
        y1=extent,
        line={"color": "#AAB2BA", "dash": "dot", "width": 1},
        layer="below",
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#657584", line_width=1)
    figure.add_hline(y=0, line_dash="dash", line_color="#657584", line_width=1)
    title_text = "TS versus CT MDC across modules"
    if module_definition:
        title_text += f"<br><sup>{module_definition}</sup>"
    figure.update_layout(
        title={"text": title_text, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=610,
        margin={"l": 70, "r": 30, "t": 90, "b": 65},
        xaxis={"title": "TS log2 MDC (AD / Control)", "range": [-extent, extent]},
        yaxis={"title": "CT log2 MDC (AD / Control)", "range": [-extent, extent]},
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
