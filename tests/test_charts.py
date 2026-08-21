from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (  # noqa: E402
    CONTINUOUS_COLOR_SCALES,
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
    edge_summary_figure,
    mdc_module_figure,
    mdc_overview_figure,
    mdc_resolved_heatmap_figure,
    mdc_resolved_module_figure,
    module_entropy_figure,
    module_region_composition_figure,
    module_size_distribution_figure,
    resolved_to_long,
)
from app_helpers.correlations import calculate_correlations  # noqa: E402
from app_helpers.data import (  # noqa: E402
    association_kegg_subtitles,
    load_aggregate,
    load_aggregate_statistics,
    load_kegg,
    load_mdc_summary,
    load_mdc_resolved,
    load_edge_summaries,
    load_module_annotations,
    load_module_details,
    load_resolved,
    load_resolved_statistics,
    selected_annotation,
)


def test_scatter_and_distribution_chart_paths() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    long = aggregate_to_long(aggregate, "rint")
    stats = load_aggregate_statistics(
        "control_anchored", 935, "motor10_demog_slope", "connectivity"
    )
    scatter = association_figure(
        long,
        stats,
        phenotype="motor10_demog_slope",
        phenotype_label="Demographic-adjusted motor slope",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=False,
        color_by="diagnosis_group",
        color_label="Diagnosis group",
        hover_fields={"cogn_global": "Global cognition"},
    )
    assert len(scatter.data) >= 6
    assert "Module M935" in scatter.layout.title.text
    assert all("projid" not in str(trace.hovertemplate) for trace in scatter.data)
    statistic_annotations = [
        str(annotation.text)
        for annotation in scatter.layout.annotations
        if "<b>Control</b>" in str(annotation.text)
    ]
    assert statistic_annotations
    assert all("ρ=" in text and "; r=" not in text for text in statistic_annotations)

    histogram = distribution_figure(
        long,
        feature_label="Connectivity",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        chart_type="Histogram",
        bins=30,
    )
    assert len(histogram.data) == 6
    assert all(trace.histnorm == "probability density" for trace in histogram.data)


def test_association_panels_show_scope_matched_kegg_subtitles() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    aggregate_long = aggregate_to_long(aggregate, "rint")
    aggregate_stats = load_aggregate_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    annotation = selected_annotation(load_module_annotations(), 935)
    kegg = load_kegg(935)
    aggregate_subtitles = association_kegg_subtitles(
        kegg,
        aggregate_long["component"].unique(),
        resolved=False,
        aggregate_annotation=annotation,
    )
    aggregate_figure = association_figure(
        aggregate_long,
        aggregate_stats,
        phenotype="cogn_global",
        phenotype_label="Global cognition",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=False,
        color_by="diagnosis_group",
        color_label="Diagnosis group",
        hover_fields={},
        kegg_subtitles=aggregate_subtitles,
    )
    aggregate_titles = [
        str(value.text) for value in aggregate_figure.layout.annotations[:2]
    ]
    assert all("KEGG enrichment (tissue-expanded)" in text for text in aggregate_titles)

    resolved = load_resolved("control_anchored", 935, "connectivity")
    resolved_long = resolved_to_long(resolved, "rint")
    resolved_stats = load_resolved_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    resolved_subtitles = association_kegg_subtitles(
        kegg, resolved_long["component"].unique(), resolved=True
    )
    resolved_figure = association_figure(
        resolved_long,
        resolved_stats,
        phenotype="cogn_global",
        phenotype_label="Global cognition",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=True,
        color_by="diagnosis_group",
        color_label="Diagnosis group",
        hover_fields={},
        kegg_subtitles=resolved_subtitles,
    )
    resolved_titles = {
        component: str(annotation.text)
        for component, annotation in zip(
            sorted(resolved_long["component"].unique()),
            resolved_figure.layout.annotations[:6],
        )
    }
    assert "KEGG enrichment (AC)" in resolved_titles["TS_AC"]
    pair_title = resolved_titles["CT_AC__DLPFC"]
    assert "AC FDR=" in pair_title and "DLPFC FDR=" in pair_title


def test_association_scatter_can_switch_from_default_spearman_to_pearson() -> None:
    resolved = load_resolved("control_anchored", 263, "negative_density")
    long = resolved_to_long(resolved, "rint")
    statistics = load_resolved_statistics(
        "control_anchored", 263, "cogn_global", "negative_density"
    )
    common = {
        "frame": long,
        "statistics": statistics,
        "phenotype": "cogn_global",
        "phenotype_label": "Global cognition",
        "feature_label": "Negative density",
        "scale": "rint",
        "scale_label": "Z-score",
        "diagnoses": ["Control", "MCI", "AD"],
        "module": 263,
        "resolved": True,
        "color_by": "diagnosis_group",
        "color_label": "Diagnosis group",
        "hover_fields": {},
    }
    spearman = association_figure(**common)
    spearman_text = next(
        str(annotation.text)
        for annotation in spearman.layout.annotations
        if "<b>Control</b>" in str(annotation.text)
    )
    assert "ρ=0.04" in spearman_text
    assert "p=0.59" in spearman_text
    assert "FDR=" in spearman_text
    assert "; r=" not in spearman_text

    pearson = association_figure(**common, correlation_method="pearson")
    pearson_text = next(
        str(annotation.text)
        for annotation in pearson.layout.annotations
        if "<b>Control</b>" in str(annotation.text)
    )
    assert "r=0.05" in pearson_text
    assert "p=0.54" in pearson_text
    assert "ρ=" not in pearson_text


def test_continuous_color_and_correlation_heatmap() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    long = aggregate_to_long(aggregate, "rint")
    stats = load_aggregate_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    scatter = association_figure(
        long,
        stats,
        phenotype="cogn_global",
        phenotype_label="Global cognition",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=False,
        color_by="motor10_demog_slope",
        color_label="Demographic-adjusted motor slope",
        hover_fields={
            "cogn_global": "Global cognition",
            "motor10_demog_slope": "Demographic-adjusted motor slope",
        },
        continuous_colorscale="Viridis",
        reverse_colorscale=True,
    )
    assert scatter.layout.coloraxis.colorbar.title.text == "Demographic-adjusted motor slope"
    assert any("motor slope" in str(trace.hovertemplate) for trace in scatter.data)
    assert bool(scatter.layout.coloraxis.reversescale)

    for palette in CONTINUOUS_COLOR_SCALES:
        palette_figure = association_figure(
            long,
            stats,
            phenotype="cogn_global",
            phenotype_label="Global cognition",
            feature_label="Connectivity",
            scale="rint",
            scale_label="Z-score",
            diagnoses=["Control", "MCI", "AD"],
            module=935,
            resolved=False,
            color_by="motor10_demog_slope",
            color_label="Demographic-adjusted motor slope",
            hover_fields={},
            continuous_colorscale=palette,
        )
        assert palette_figure.layout.coloraxis.colorscale is not None

    correlations = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=["cogn_global", "motor10_demog_slope"],
    )
    correlations["outcome_label"] = correlations["outcome"].map(
        {
            "cogn_global": "Global cognition",
            "motor10_demog_slope": "Motor slope",
        }
    )
    correlations["heatmap_row"] = (
        correlations["metric_family"] + " · " + correlations["component_label"]
    )
    selected = correlations.loc[correlations["diagnosis_group"].eq("AD")]
    heatmap = correlation_heatmap_figure(
        selected,
        value_column="pearson_r",
        p_column="pearson_p",
        fdr_column="pearson_fdr_displayed_family",
        title="Test correlations",
    )
    assert len(heatmap.data) == 1
    assert heatmap.data[0].zmin == -1
    assert heatmap.data[0].zmax == 1


def test_correlation_heatmap_can_cluster_rows_and_columns() -> None:
    rows = ["A", "B", "C", "D"]
    outcomes = ["outcome_1", "outcome_3", "outcome_2"]
    coefficients = {
        "A": [1.0, -1.0, 0.9],
        "B": [-1.0, 1.0, -0.9],
        "C": [0.9, -0.8, 1.0],
        "D": [-0.9, 0.8, -1.0],
    }
    records = []
    for row in rows:
        for outcome, coefficient in zip(outcomes, coefficients[row], strict=True):
            records.append(
                {
                    "heatmap_row": row,
                    "outcome": outcome,
                    "outcome_label": outcome,
                    "pearson_r": coefficient,
                    "pearson_p": 0.01,
                    "pearson_fdr_displayed_family": 0.02,
                    "n": 100,
                }
            )

    figure = correlation_heatmap_figure(
        pd.DataFrame.from_records(records),
        value_column="pearson_r",
        p_column="pearson_p",
        fdr_column="pearson_fdr_displayed_family",
        title="Clustered correlations",
        row_order=rows,
        cluster_rows=True,
        cluster_columns=True,
    )
    clustered_rows = list(figure.data[0].y)
    clustered_columns = list(figure.data[0].x)

    assert set(clustered_rows) == set(rows)
    assert set(clustered_columns) == set(outcomes)
    assert abs(clustered_rows.index("A") - clustered_rows.index("C")) == 1
    assert abs(clustered_rows.index("B") - clustered_rows.index("D")) == 1
    assert abs(clustered_columns.index("outcome_1") - clustered_columns.index("outcome_2")) == 1


def test_mdc_selected_module_and_overview_charts() -> None:
    mdc = load_mdc_summary()
    selected = mdc.loc[mdc["module"].eq(1918)].iloc[0]

    module_figure = mdc_module_figure(selected, threshold=0.05)
    assert len(module_figure.data) == 1
    assert list(module_figure.data[0].x) == [
        "Total",
        "Tissue-specific (TS)",
        "Cross-tissue (CT)",
    ]
    assert "M1918" in module_figure.layout.title.text

    overview = mdc_overview_figure(mdc, selected_module=1918, threshold=0.05)
    assert len(overview.data) >= 2
    assert any(trace.name == "Selected M1918" for trace in overview.data)
    assert overview.layout.xaxis.title.text == "TS log2 MDC (AD / Control)"


def test_module_size_and_region_composition_charts_filter_and_sort() -> None:
    details = load_module_details()
    ct_only = details.loc[details["cluster_type"].eq("CT")].copy()
    ts_only = details.loc[details["cluster_type"].eq("TS")].copy()

    distribution = module_size_distribution_figure(
        details, module_definition="Full-cohort L4 modules (154)"
    )
    assert len(distribution.data) == 2
    assert {trace.name.split()[0] for trace in distribution.data} == {"CT", "TS"}
    assert sum(len(trace.x) for trace in distribution.data) == len(details)
    assert "154 modules shown" in distribution.layout.title.text

    ct_composition = module_region_composition_figure(ct_only)
    assert len(ct_composition.data) == 3
    assert [trace.name for trace in ct_composition.data] == ["AC", "DLPFC", "PCG"]
    expected_order = (
        ct_only.sort_values(
            ["module_size", "module"], ascending=[False, True], kind="stable"
        )["module"]
        .astype(int)
        .map(lambda value: f"M{value}")
        .tolist()
    )
    assert list(ct_composition.layout.yaxis.categoryarray) == expected_order
    stacked_totals = sum(pd.Series(trace.x, dtype="int64") for trace in ct_composition.data)
    expected_sizes = (
        ct_only.sort_values(
            ["module_size", "module"], ascending=[False, True], kind="stable"
        )["module_size"]
        .astype(int)
        .reset_index(drop=True)
    )
    assert stacked_totals.reset_index(drop=True).equals(expected_sizes)

    ts_composition = module_region_composition_figure(ts_only)
    assert len(ts_composition.layout.yaxis.categoryarray) == len(ts_only) == 28
    assert set(ts_composition.data[0].customdata[:, 1]) == {"TS"}

    entropy = module_entropy_figure(details, selected_module=935)
    assert len(entropy.data) >= 3
    assert any(trace.name == "Selected module" for trace in entropy.data)
    assert entropy.layout.yaxis.range == (-0.03, 1.03)


def test_resolved_mdc_and_edge_summary_charts() -> None:
    resolved = load_mdc_resolved()
    selected = resolved.loc[resolved["module"].eq(1918)]
    module_figure = mdc_resolved_module_figure(selected, threshold=0.05)
    assert len(module_figure.data) == 1
    assert len(module_figure.data[0].x) == 6
    overview = mdc_resolved_heatmap_figure(
        resolved, threshold=0.05, selected_module=1918
    )
    assert len(overview.data) == 1
    assert list(overview.data[0].y)[0] == "M1918"

    edges = load_edge_summaries("lioness", "control_anchored", 935)
    edge_figure = edge_summary_figure(
        edges, "absolute_weight_sum", "Absolute weight sum", 935
    )
    assert len(edge_figure.data) == 3
