from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (  # noqa: E402
    CONTINUOUS_COLOR_SCALES,
    EDGE_COMPONENT_COLORS,
    EDGE_COMPONENT_LABELS,
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
    edge_volcano_figure,
    edge_summary_figure,
    mdc_entropy_figure,
    mdc_module_figure,
    mdc_overview_figure,
    mdc_resolved_heatmap_figure,
    mdc_resolved_module_figure,
    module_entropy_figure,
    module_region_composition_figure,
    module_size_distribution_figure,
    pathway_mdc_detail_figure,
    pathway_mdc_heatmap_figure,
    prediction_confusion_figure,
    prediction_ct_ts_figure,
    prediction_error_figure,
    prediction_observed_figure,
    prediction_threshold_figure,
    resolved_to_long,
)
from app_helpers.correlations import calculate_correlations  # noqa: E402
from app_helpers.data import (  # noqa: E402
    association_kegg_subtitles,
    build_pathway_mdc_rows,
    collapse_pathway_mdc_rows,
    load_aggregate,
    load_aggregate_statistics,
    load_kegg,
    load_mdc_summary,
    load_mdc_resolved,
    load_edge_summaries,
    load_volcano_bins,
    load_volcano_candidates,
    load_module_annotations,
    load_module_details,
    load_resolved,
    load_resolved_statistics,
    selected_annotation,
    summarize_pathway_mdc_rows,
)


def test_edge_volcano_switches_between_global_and_per_module_bh() -> None:
    candidates = load_volcano_candidates(
        "full_cohort", "lioness", "control_anchored", 935
    )
    for fdr_scope, label in (("global", "Global BH"), ("per_module", "Per-module BH")):
        bins = load_volcano_bins(
            "full_cohort", "lioness", "control_anchored", 935, fdr_scope
        )
        figure = edge_volcano_figure(
            candidates,
            bins,
            module=935,
            scope="CT",
            analysis_set="Discovery",
            fdr_scope=fdr_scope,
            x_metric="hedges_g",
            y_metric="fdr",
            significant_only=False,
            significance_threshold=0.10,
            direction="Either",
            prevalence_column=None,
            minimum_prevalence=0.0,
            module_definition="Full-cohort L4 modules (154)",
        )
        assert label in str(figure.layout.title.text)
        assert any(
            "Global BH FDR" in str(trace.hovertemplate)
            and "Per-module BH FDR" in str(trace.hovertemplate)
            for trace in figure.data
        )


def test_edge_volcano_colors_exact_dots_by_tissue_component() -> None:
    candidates = load_volcano_candidates(
        "control_derived", "lioness", "control_anchored", 935
    )
    bins = load_volcano_bins(
        "control_derived", "lioness", "control_anchored", 935, "global"
    )
    figure = edge_volcano_figure(
        candidates,
        bins,
        module=935,
        scope="total",
        module_definition="Control-derived L4 modules (186)",
    )
    exact_traces = [trace for trace in figure.data if trace.type == "scattergl"]
    expected_components = set(candidates["component"].astype(str))
    assert {trace.name for trace in exact_traces} == {
        EDGE_COMPONENT_LABELS[component] for component in expected_components
    }
    for trace in exact_traces:
        component = next(
            key for key, label in EDGE_COMPONENT_LABELS.items() if label == trace.name
        )
        assert trace.marker.color == EDGE_COMPONENT_COLORS[component]
        assert trace.marker.symbol == (
            "circle" if component.startswith("TS_") else "diamond"
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

    raw_module = mdc_module_figure(selected, threshold=0.05, scale="raw")
    assert raw_module.layout.yaxis.title.text == "MDC ratio (AD / Control)"
    assert list(raw_module.data[0].y) == [
        selected["mdc_total"],
        selected["mdc_ts"],
        selected["mdc_ct"],
    ]
    raw_overview = mdc_overview_figure(
        mdc, selected_module=1918, threshold=0.05, scale="raw"
    )
    assert raw_overview.layout.xaxis.title.text == "TS MDC ratio (AD / Control)"
    assert raw_overview.layout.yaxis.title.text == "CT MDC ratio (AD / Control)"


def test_mdc_entropy_scatter_supports_ts_ct_and_both_scales() -> None:
    mdc = load_mdc_summary()
    details = load_module_details()
    data = mdc.merge(
        details[
            [
                "module",
                "module_size",
                "cluster_type",
                "tissue_entropy",
                "tissue_entropy_normalized",
            ]
        ],
        on="module",
        validate="one_to_one",
    )
    for scope in ("ts", "ct"):
        for scale, expected_reference in (("log2", 0.0), ("raw", 1.0)):
            figure = mdc_entropy_figure(
                data,
                scope=scope,
                selected_module=1918,
                threshold=0.05,
                scale=scale,
                module_definition="Full-cohort L4 modules (154)",
            )
            assert f"{scope.upper()} MDC vs normalized" in figure.layout.title.text
            assert "Spearman" in figure.layout.title.text
            assert figure.layout.xaxis.title.text == "Normalized Shannon entropy"
            assert any(trace.name == "Selected M1918" for trace in figure.data)
            assert any(trace.name == "OLS trend" for trace in figure.data)
            horizontal_lines = [
                shape
                for shape in figure.layout.shapes
                if shape.y0 == shape.y1 == expected_reference
            ]
            assert horizontal_lines


def test_pathway_resolved_mdc_heatmap_and_detail_charts() -> None:
    rows = build_pathway_mdc_rows(
        load_mdc_summary(),
        load_mdc_resolved(),
        load_kegg(),
        enrichment_fdr_threshold=0.05,
    )
    rows = rows.loc[
        rows["component"].isin(
            [
                "TS_AC",
                "TS_DLPFC",
                "TS_PCGBA23",
                "CT_AC__DLPFC",
                "CT_AC__PCGBA23",
                "CT_DLPFC__PCGBA23",
            ]
        )
    ]
    summary = summarize_pathway_mdc_rows(rows, minimum_modules=1)
    selected_pathway = str(summary.iloc[0]["pathway_id"])
    for scale, expected_reference in (("log2", 0.0), ("raw", 1.0)):
        heatmap = pathway_mdc_heatmap_figure(
            summary,
            scale=scale,
            top_n=20,
            selected_pathway_id=selected_pathway,
            module_definition="Full-cohort L4 modules (154)",
        )
        assert len(heatmap.data) == 1
        assert len(heatmap.data[0].y) == 20
        assert any(str(label).startswith("★") for label in heatmap.data[0].y)
        detail = pathway_mdc_detail_figure(
            rows,
            pathway_id=selected_pathway,
            selected_module=935,
            threshold=0.05,
            scale=scale,
        )
        assert len(detail.data) >= 1
        horizontal_lines = [
            shape
            for shape in detail.layout.shapes
            if shape.y0 == shape.y1 == expected_reference
        ]
        assert horizontal_lines

    for resolution, expected_title in (
        ("subcategory", "KEGG sub-category-annotated MDC"),
        ("category", "KEGG category-annotated MDC"),
    ):
        grouped_rows = collapse_pathway_mdc_rows(rows, resolution=resolution)
        grouped_summary = summarize_pathway_mdc_rows(
            rows, minimum_modules=1, resolution=resolution
        )
        selected_group = str(grouped_summary.iloc[0]["pathway_id"])
        heatmap = pathway_mdc_heatmap_figure(
            grouped_summary,
            top_n=20,
            selected_pathway_id=selected_group,
            resolution=resolution,
        )
        assert expected_title in heatmap.layout.title.text
        assert any(str(label).startswith("★") for label in heatmap.data[0].y)
        detail = pathway_mdc_detail_figure(
            grouped_rows,
            pathway_id=selected_group,
            selected_module=935,
            threshold=0.05,
            resolution=resolution,
        )
        assert f"annotated to {expected_title.removesuffix('-annotated MDC')}" in (
            detail.layout.title.text
        )


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
    raw_module_figure = mdc_resolved_module_figure(
        selected, threshold=0.05, scale="raw"
    )
    assert raw_module_figure.layout.yaxis.title.text == "MDC ratio (AD / Control)"
    raw_overview = mdc_resolved_heatmap_figure(
        resolved, threshold=0.05, selected_module=1918, scale="raw"
    )
    assert raw_overview.data[0].colorbar.title.text == "MDC ratio (AD / Control)"

    edges = load_edge_summaries("lioness", "control_anchored", 935)
    edge_figure = edge_summary_figure(
        edges, "absolute_weight_sum", "Absolute weight sum", 935
    )
    assert len(edge_figure.data) == 3


def test_prediction_diagnostic_charts_use_sanitized_rows() -> None:
    diagnostics = pd.DataFrame(
        {
            "sample_id": ["P-A", "P-B", "P-C", "P-D"],
            "target": ["Control", "Control", "AD", "AD"],
            "predicted": ["Control", "AD", "AD", "AD"],
            "probability_Control": [0.8, 0.4, 0.2, 0.1],
            "probability_AD": [0.2, 0.6, 0.8, 0.9],
        }
    )
    threshold = prediction_threshold_figure(diagnostics, title="Threshold")
    assert len(threshold.data) == 3
    confusion = prediction_confusion_figure(
        pd.DataFrame(
            {
                "actual": ["Control", "Control", "AD", "AD"],
                "predicted_class": ["Control", "AD", "Control", "AD"],
                "n": [1, 1, 0, 2],
            }
        ),
        title="Confusion",
    )
    assert int(confusion.data[0].z.sum()) == 4
    continuous = pd.DataFrame(
        {"sample_id": ["P-A", "P-B"], "target": [1.0, 2.0], "predicted": [1.2, 1.8]}
    )
    assert len(prediction_observed_figure(continuous, title="Observed").data) == 2
    assert len(prediction_error_figure(continuous, title="Errors").data) == 2


def test_prediction_ct_ts_forest_uses_absolute_confidence_limits_in_hover() -> None:
    frame = pd.DataFrame(
        {
            "outcome": ["motor10_demog_slope"],
            "comparison": ["pooled_CT_minus_TS"],
            "performance_difference": [0.12],
            "ci_low": [0.03],
            "ci_high": [0.21],
            "p_value": [0.01],
            "fdr_global": [0.04],
            "fdr_within_outcome": [0.02],
        }
    )
    figure = prediction_ct_ts_figure(
        frame,
        outcome_labels={"motor10_demog_slope": "Motor slope"},
        title="CT versus TS",
    )
    assert figure.data[0].customdata[0, 0] == 0.03
    assert figure.data[0].customdata[0, 1] == 0.21
    assert "error_x" not in figure.data[0].hovertemplate
