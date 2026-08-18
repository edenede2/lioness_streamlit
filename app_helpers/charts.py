"""Plotly figures for associations and donor-level feature distributions."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DIAGNOSIS_COLORS = {
    "Control": "#2C7FB8",
    "MCI": "#D8A500",
    "AD": "#E66101",
}
DIAGNOSIS_SYMBOLS = {"Control": "circle", "MCI": "diamond", "AD": "square"}


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


def _stat_text(row: pd.Series | None, scale: str, resolved: bool) -> str:
    if row is None:
        return "statistics unavailable"
    if resolved:
        r = row.get(f"r_{scale}")
        p = row.get(f"p_{scale}")
        rho = row.get("rho")
        q = row.get("q_rint_within_phenotype") if scale == "rint" else np.nan
    else:
        component = str(row.get("_display_component", "CT"))
        r = row.get(f"r_{scale}_{component}")
        p = row.get(f"p_{scale}_{component}")
        rho = row.get(f"rho_{component}")
        q = row.get(f"q_rint_{component}_global") if scale == "rint" else np.nan
    text = (
        f"n={int(row.get('n', 0))}; r={_format_number(r)}; "
        f"p={_format_number(p)}; ρ={_format_number(rho)}"
    )
    if scale == "rint":
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
) -> go.Figure:
    """Build faceted scatter plots with diagnosis-specific OLS lines."""
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
            stat_lines.append(f"<b>{diagnosis}</b>: {_stat_text(stat_row, scale, resolved)}")

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
    fig.update_layout(
        title={
            "text": f"Module M{int(module)}: {phenotype_label} vs {feature_label}",
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
    fig.update_layout(
        title={"text": f"Module M{int(module)}: {feature_label} distributions", "x": 0.01},
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


def correlation_heatmap_figure(
    frame: pd.DataFrame,
    value_column: str,
    p_column: str,
    fdr_column: str,
    title: str,
    row_order: list[str] | None = None,
) -> go.Figure:
    """Render a correlation matrix with n, p, and displayed-family FDR in hover."""
    if frame.empty:
        return go.Figure()
    outcomes = frame[["outcome", "outcome_label"]].drop_duplicates()
    outcome_order = outcomes["outcome"].tolist()
    outcome_labels = outcomes.set_index("outcome")["outcome_label"].to_dict()
    if row_order is None:
        row_order = frame["heatmap_row"].drop_duplicates().tolist()

    def pivot(column: str) -> pd.DataFrame:
        return (
            frame.pivot_table(
                index="heatmap_row",
                columns="outcome",
                values=column,
                aggfunc="first",
            )
            .reindex(index=row_order, columns=outcome_order)
        )

    values = pivot(value_column)
    n_values = pivot("n")
    p_values = pivot(p_column)
    fdr_values = pivot(fdr_column)
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
