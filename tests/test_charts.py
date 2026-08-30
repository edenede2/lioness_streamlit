from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (  # noqa: E402
    CONTINUOUS_COLOR_SCALES,
    EDGE_COMPONENT_COLORS,
    EDGE_COMPONENT_LABELS,
    aggregate_to_long,
    association_figure,
    categorical_association_figure,
    clustered_correlation_group_order,
    correlation_heatmap_figure,
    distribution_figure,
    edge_volcano_figure,
    edge_summary_figure,
    grouped_association_figure,
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
    prediction_heatmap_figure,
    prediction_observed_figure,
    prediction_performance_figure,
    prediction_threshold_figure,
    resolved_to_long,
    targeted_fold_robustness_figure,
    targeted_eigengene_source_comparison_figure,
    targeted_primary_comparison_figure,
    targeted_selection_frequency_figure,
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
    PREDICTION_BLOCK_LABELS,
    PREDICTION_BLOCK_ORDER,
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


def test_configurable_grouped_scatter_controls_annotations_lines_and_legend() -> None:
    rows = []
    for cluster, diagnosis, offset in [(1, "Control", 0.0), (2, "AD", 1.0)]:
        for index in range(12):
            rows.append(
                {
                    "sample_id": f"S-{cluster}-{index}", "module": 1,
                    "component": "CT", "component_label": "CT aggregate",
                    "metric_value": float(index), "cogn_global": offset + index,
                    "clusters": cluster, "diagnosis_group": diagnosis,
                }
            )
    frame = pd.DataFrame(rows)
    statistics = calculate_correlations(
        frame.assign(grouping_variable="clusters", grouping_level=frame["clusters"]),
        ["module", "component", "component_label", "grouping_variable", "grouping_level"],
        ["cogn_global"], min_group_n=10,
    )
    statistics["spearman_fdr_across_modules"] = [0.01, 0.20]
    statistics["pearson_fdr_across_modules"] = [0.01, 0.20]
    figure = grouped_association_figure(
        frame, statistics, phenotype="cogn_global", phenotype_label="Global cognition",
        feature_label="Connectivity", scale_label="Raw", grouping_variable="clusters",
        grouping_levels=[1, 2], grouping_labels={"1": "Cluster 1", "2": "Cluster 2"},
        module=1, color_by="clusters", color_label="Cluster", hover_fields={},
        trend_line_rule="fdr", significance_cutoff=0.05,
        annotation_fields=["coefficient", "fdr"], minimum_group_n=10,
        categorical_color_fields={"clusters"}, show_pooled=False,
    )
    legend_groups = {
        trace.legendgroup for trace in figure.data if bool(trace.showlegend)
    }
    assert legend_groups == {"association_group::1", "association_group::2"}
    trend_groups = {
        trace.legendgroup
        for trace in figure.data
        if trace.mode == "lines" and "trend" in str(trace.name)
    }
    assert trend_groups == {"association_group::1"}
    statistic_text = " ".join(str(value.text) for value in figure.layout.annotations)
    assert "ρ=" in statistic_text and "module-set FDR=" in statistic_text
    assert "n=" not in statistic_text


def test_generic_categorical_figure_uses_diagnosis_marker_shapes() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"S-{index}" for index in range(20)],
            "component": ["TS"] * 20, "component_label": ["TS aggregate"] * 20,
            "metric_value": np.arange(20, dtype=float),
            "braak_stage": [1] * 10 + [2] * 10,
            "diagnosis_group": ["Control"] * 10 + ["AD"] * 10,
        }
    )
    statistics = pd.DataFrame(
        {
            "component": ["TS"], "n": [20], "n_tested": [20],
            "levels_tested": ["1,2"], "levels_excluded_small_n": [""],
            "kruskal_h": [8.0], "epsilon_squared": [0.4],
            "categorical_p": [0.004], "categorical_fdr_across_modules": [0.02],
        }
    )
    figure = categorical_association_figure(
        frame, statistics, category_variable="braak_stage", category_label="Braak stage",
        category_levels=[1, 2], category_labels={"1": "Braak 1", "2": "Braak 2"},
        feature_label="Connectivity", scale_label="Raw", module=1,
    )
    point_symbols = {
        trace.marker.symbol for trace in figure.data if trace.type == "scatter"
    }
    assert point_symbols == {"circle", "square"}
    annotation = " ".join(str(value.text) for value in figure.layout.annotations)
    assert "ε²=" in annotation and "module-set FDR=" in annotation


def test_association_scatter_adds_pooled_all_donor_statistics_and_trends() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    long = aggregate_to_long(aggregate, "rint")
    stats = load_aggregate_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    pooled_input = long.copy()
    pooled_input["diagnosis_group"] = "All donors"
    pooled = calculate_correlations(
        pooled_input,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=["cogn_global"],
    )
    figure = association_figure(
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
        color_by="diagnosis_group",
        color_label="Diagnosis group",
        hover_fields={},
        pooled_statistics=pooled,
    )

    pooled_traces = [
        trace for trace in figure.data if trace.legendgroup == "__pooled__"
    ]
    assert len(pooled_traces) == 2
    assert sum(bool(trace.showlegend) for trace in pooled_traces) == 1
    assert all(trace.line.dash == "dash" for trace in pooled_traces)
    pooled_annotations = [
        str(annotation.text)
        for annotation in figure.layout.annotations
        if "All donors (pooled)" in str(annotation.text)
    ]
    assert len(pooled_annotations) == 2
    assert all(
        "ρ=" in text and "p=" in text and "panel FDR=" in text
        for text in pooled_annotations
    )


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


def test_correlation_heatmap_marks_fdr_and_outlines_clustered_module_blocks() -> None:
    records = []
    for module, profile in ((10, [0.9, 0.8]), (20, [-0.8, -0.7]), (30, [0.85, 0.75])):
        for feature_index, feature in enumerate(("connectivity", "abs_sum")):
            for outcome_index, outcome in enumerate(("cognition", "motor")):
                coefficient = profile[outcome_index] - feature_index * 0.05
                records.append(
                    {
                        "module": module,
                        "metric_family": feature,
                        "diagnosis_group": "AD",
                        "heatmap_row": f"M{module} · {feature}",
                        "outcome": outcome,
                        "outcome_label": outcome.title(),
                        "spearman_rho": coefficient,
                        "spearman_p": 0.001 if module != 20 else 0.4,
                        "spearman_fdr_across_modules": 0.01 if module != 20 else 0.6,
                        "n": 100,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    module_order = clustered_correlation_group_order(
        frame,
        value_column="spearman_rho",
    )
    assert set(module_order) == {10, 20, 30}
    assert abs(module_order.index(10) - module_order.index(30)) == 1

    row_order = [
        f"M{module} · {feature}"
        for module in module_order
        for feature in ("connectivity", "abs_sum")
    ]
    groups = {
        row: row.split(" · ", maxsplit=1)[0]
        for row in row_order
    }
    figure = correlation_heatmap_figure(
        frame,
        value_column="spearman_rho",
        p_column="spearman_p",
        fdr_column="spearman_fdr_across_modules",
        title="Module blocks",
        row_order=row_order,
        row_group_labels=groups,
    )
    assert len(figure.layout.shapes) == 3
    assert np.count_nonzero(np.asarray(figure.data[0].text) == "*") == 8


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
    heatmap_colors = {}
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
        heatmap_colors[scale] = np.asarray(heatmap.data[0].z, dtype=float)
        assert float(heatmap.data[0].zmid) == 0.0
        np.testing.assert_allclose(
            heatmap_colors[scale],
            np.asarray(heatmap.data[0].customdata[:, :, 6], dtype=float),
            equal_nan=True,
        )
        if scale == "raw":
            assert "log-symmetric" in heatmap.data[0].colorbar.title.text
            assert "1" in list(heatmap.data[0].colorbar.ticktext)
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
    np.testing.assert_allclose(
        heatmap_colors["raw"], heatmap_colors["log2"], equal_nan=True
    )

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
    assert "log-symmetric" in raw_overview.data[0].colorbar.title.text
    assert float(raw_overview.data[0].zmid) == 0.0
    np.testing.assert_allclose(
        np.asarray(raw_overview.data[0].z, dtype=float),
        np.asarray(raw_overview.data[0].customdata[:, :, 1], dtype=float),
        equal_nan=True,
    )
    assert "raw MDC labels" in raw_overview.layout.title.text

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


def test_targeted_prediction_charts_render_nested_cv_contract() -> None:
    comparisons = pd.DataFrame(
        {
            "comparison": [
                "targeted_CT_minus_all_module_CT",
                "targeted_CT_plus_covariates_minus_covariates",
                "tissue_neutral_targeted_CT_minus_TS",
            ],
            "performance_difference": [0.04, 0.08, 0.02],
            "ci_low": [-0.01, 0.03, -0.02],
            "ci_high": [0.09, 0.13, 0.06],
            "p_value": [0.10, 0.01, 0.30],
            "fdr_primary_family": [0.15, 0.03, 0.30],
            "n_oof": [331, 331, 331],
        }
    )
    comparison = targeted_primary_comparison_figure(comparisons, title="Primary")
    assert len(comparison.data) == 1
    assert len(comparison.data[0].x) == 3

    consensus = pd.DataFrame(
        {
            "module": [935, 1918],
            "outer_selection_frequency": [0.88, 0.64],
            "mean_stable_rank": [2.0, 6.0],
            "median_incremental_score": [0.03, 0.01],
            "consensus_selected": [True, False],
            "kegg_annotation": ["KEGG enrichment: A", "KEGG enrichment: B"],
        }
    )
    stability = targeted_selection_frequency_figure(consensus, title="Stability")
    assert set(stability.data[0].y) == {"M935", "M1918"}

    folds = pd.DataFrame(
        {
            "metric": ["roc_auc"] * 4,
            "value": [0.66, 0.70, 0.62, 0.69],
            "predictor_block": ["CT_pooled", "CT_pooled", "TS_pooled", "TS_pooled"],
            "outer_repeat": [0, 1, 0, 1],
            "outer_fold": [0, 0, 0, 0],
        }
    )
    robustness = targeted_fold_robustness_figure(
        folds,
        metric="roc_auc",
        block_labels={"CT_pooled": "CT pooled", "TS_pooled": "TS pooled"},
        title="Robustness",
    )
    assert len(robustness.data) == 2


def test_eigengene_source_comparison_chart_uses_paired_intervals() -> None:
    frame = pd.DataFrame(
        {
            "predictor_block": ["AC", "CT_pooled"],
            "model_variant": [
                "transcriptomics_only",
                "covariates_plus_network_plus_transcriptomics",
            ],
            "source_a": ["single_region_full_tissue_l3"] * 2,
            "source_a_label": ["Single-region modules: full tissue cohorts"] * 2,
            "source_b_label": ["Matched multi-tissue modules"] * 2,
            "performance_difference": [0.03, -0.01],
            "ci_low": [-0.01, -0.04],
            "ci_high": [0.07, 0.02],
            "p_value": [0.12, 0.40],
            "fdr_global": [0.24, 0.40],
            "fdr_within_outcome": [0.20, 0.40],
            "n_oof": [331, 331],
        }
    )
    figure = targeted_eigengene_source_comparison_figure(
        frame,
        block_labels={"AC": "AC", "CT_pooled": "Pooled CT"},
        model_labels={
            "transcriptomics_only": "Transcriptomics only",
            "covariates_plus_network_plus_transcriptomics": "All predictors",
        },
        title="Eigengene sources",
    )
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 2
    assert np.all(np.asarray(figure.data[0].error_x.array) >= 0)
    assert np.all(np.asarray(figure.data[0].error_x.arrayminus) >= 0)
    assert "Source A − Source B" in figure.layout.xaxis.title.text


def test_prediction_block_charts_use_ts_ct_pool_all_order() -> None:
    shuffled_blocks = list(reversed(PREDICTION_BLOCK_ORDER))
    performance = pd.DataFrame(
        {
            "metric": ["roc_auc"] * len(shuffled_blocks),
            "value": np.linspace(0.55, 0.75, len(shuffled_blocks)),
            "predictor_block": shuffled_blocks,
            "model_variant": ["network_only"] * len(shuffled_blocks),
            "n_oof": [300] * len(shuffled_blocks),
            "status": ["available"] * len(shuffled_blocks),
        }
    )
    figure = prediction_performance_figure(
        performance,
        metric="roc_auc",
        block_labels=PREDICTION_BLOCK_LABELS,
        block_order=PREDICTION_BLOCK_ORDER,
        model_labels={"network_only": "Network only"},
        title="Ordered blocks",
    )
    expected_labels = [PREDICTION_BLOCK_LABELS[value] for value in PREDICTION_BLOCK_ORDER]
    assert list(figure.data[0].x) == expected_labels
    assert list(figure.layout.xaxis.categoryarray) == expected_labels

    heatmap_data = performance.assign(outcome="diagnosis_binary")
    heatmap = prediction_heatmap_figure(
        heatmap_data,
        metric="roc_auc",
        block_labels=PREDICTION_BLOCK_LABELS,
        block_order=PREDICTION_BLOCK_ORDER,
        outcome_labels={"diagnosis_binary": "AD versus Control"},
        title="Ordered heatmap",
    )
    assert list(heatmap.data[0].y) == expected_labels
