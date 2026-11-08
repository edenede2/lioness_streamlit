#!/usr/bin/env python3
"""Meeting figures from saved evidence using the app's existing exporter."""

from pathlib import Path
from dataclasses import replace
import json
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from app_helpers import validation_evidence as E
from app_helpers.validation_evidence_ui import forest, _gene_symbols
from app_helpers.figure_export import PlotExportSpec, PlotContentConfig, preset_config
from app_helpers.figure_export.adapters import render_figure, fit_automatic_config


def export(fig, name, root):
    fig.update_layout(meta={"evidence_version": E.VERSION, "saved_source_only": True})
    spec = PlotExportSpec.from_figure(name, fig, {"evidence_version": E.VERSION})
    root.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        config = replace(
            preset_config("Grant figure", spec.aspect_ratio),
            width=210,
            height=165,
            format=extension,
            dpi=300,
        )
        config = replace(
            config,
            style=replace(
                config.style,
                legend_visible=fig.layout.showlegend is not False,
                legend_position="Bottom" if name.startswith("same_edge") else "Top",
                legend_orientation=(
                    "Horizontal" if name.startswith("same_edge") else "Original"
                ),
            ),
        )
        content = PlotContentConfig()
        config = fit_automatic_config(spec, config, content)
        data, _ = render_figure(spec, config, content)
        (root / (name + "." + extension)).write_bytes(data)


def main():
    root = E.PUBLIC / "figures"
    s = E.table("structure")
    labels = {"control_derived": "Control-derived", "full_cohort": "Full-cohort"}
    s["row_label"] = (
        s.partition.map(labels)
        + " M"
        + s.module.astype(str)
        + " · "
        + s.component.map(E.COMPONENT_LABELS)
    )
    s["cohort"] = "Observed"
    figure = forest(
        s,
        effect="magnitude_concordance",
        low="magnitude_concordance_resampling_low",
        high="magnitude_concordance_resampling_high",
        title="Controls-only structure<br><sup>Fixed edges; donor-resampling stability ranges, not formal preservation CIs</sup>",
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(
        range=[-0.05, 1], title="Corresponding absolute-adjacency correlation"
    )
    export(figure, "controls_structure_summary", root)
    m = E.table("mdc_decomposition")
    rows = []
    for _, r in m.iterrows():
        for key, label in [
            ("log2_ad_density_change", "AD change"),
            ("log2_control_density_change", "Control change"),
            ("delta_log2_mdc", "MDC change (AD − Control)"),
        ]:
            rows.append(
                dict(
                    row_label=labels[r.partition]
                    + " M"
                    + str(r.module)
                    + " · "
                    + E.COMPONENT_LABELS[r.component],
                    cohort=label,
                    effect=r[key],
                    ci_low=r[key + "_ci_low"],
                    ci_high=r[key + "_ci_high"],
                )
            )
    figure = forest(
        pd.DataFrame(rows),
        title="MDC decomposition: original → incomplete donors<br><sup>Log2 changes with donor-bootstrap intervals; descriptive, not causal</sup>",
    )
    figure.update_xaxes(title="Log2 validation/original change")
    export(figure, "mdc_decomposition_summary", root)
    f = E.table("edge_summary")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["AD–Control g", "Cognition ρ", "gpath ρ"],
        shared_yaxes=True,
    )
    for col, outcome in enumerate(("diagnosis", "cogn_global", "gpath"), 1):
        subset = f.loc[f.outcome.eq(outcome)].copy()
        subset["label"] = (
            subset.partition.map(labels)
            + " M"
            + subset.module.astype(str)
            + " · "
            + subset.component.map(E.COMPONENT_LABELS)
        )
        categories = [
            ("Same + FDR<0.05", subset.adjusted_support, "#228833"),
            (
                "Same + p<0.05 only",
                subset.nominal_support - subset.adjusted_support,
                "#4477AA",
            ),
            (
                "Same direction only",
                subset.same_direction - subset.nominal_support,
                "#DDCC77",
            ),
            ("Reversed", subset.direction_reversed, "#CC6677"),
        ]
        for label, values, color in categories:
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=subset.label,
                    orientation="h",
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                    marker_color=color,
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        height=760,
        title="Frozen original-selected edges: validation evidence<br><sup>Selected-union FDR; no validation reranking; edges are not independent replications</sup>",
        legend=dict(orientation="v", y=1.15, yanchor="bottom", x=0),
        margin=dict(t=190, l=240),
    )
    fig.update_xaxes(title="Selected edges", rangemode="tozero")
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=list(subset.label),
        ticktext=list(subset.label),
    )
    export(fig, "same_edge_support_summary", root)
    local = _gene_symbols(E.private_table("same_edges"))
    if not local.empty:
        local["module_label"] = (
            local.partition.map(labels) + " M" + local.module.astype(str)
        )
        local["physical_component"] = local.component.map(E.COMPONENT_LABELS)
        local["meaning"] = local.outcome.map(
            {
                "diagnosis": "AD–Control Hedges g",
                "cogn_global": "Cognition partial ranks",
                "gpath": "gpath partial ranks",
            }
        )
        fig = px.scatter(
            local,
            x="original_effect",
            y="effect",
            color="module_label",
            symbol="physical_component",
            color_discrete_map={
                "Control-derived M747": "#0072B2",
                "Control-derived M1064": "#D55E00",
                "Control-derived M1116": "#009E73",
                "Full-cohort M1643": "#CC79A7",
            },
            facet_col="meaning",
            hover_data=[
                "gene_a",
                "gene_b",
                "tissue_a",
                "tissue_b",
                "n",
                "p",
                "selected_family_q",
            ],
            height=650,
            title="Same fixed edges: original versus incomplete targets<br><sup>Local-only edge summaries; raw frozen-R82 beta3 readout; no edge-as-replicate p-value</sup>",
            labels={
                "effect": "Incomplete validation effect",
                "original_effect": "Original effect",
            },
        )
        fig.update_layout(
            template="plotly_white",
            legend_title_text="Module · component",
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=150),
        )
        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("meaning=", "")))
        # Facet order follows the observed fixed table, not favorable results.
        for column, meaning in enumerate(local.meaning.drop_duplicates(), 1):
            g = local.loc[local.meaning.eq(meaning)]
            low = float(g[["original_effect", "effect"]].min().min())
            high = float(g[["original_effect", "effect"]].max().max())
            fig.add_trace(
                go.Scatter(
                    x=[low, high],
                    y=[low, high],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
        export(fig, "same_edges_original_validation", E.PRIVATE)
    print(
        json.dumps(
            {
                "public_figures": 6,
                "local_edge_figures": 2,
                "exporter": "existing PlotExportSpec/Kaleido pipeline",
            }
        )
    )


if __name__ == "__main__":
    main()
