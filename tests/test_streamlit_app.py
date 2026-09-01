from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_helpers.correlations import (
    add_categorical_across_module_fdr,
    calculate_categorical_associations,
)

import app_helpers.charts as chart_helpers
import app_helpers.correlations as correlation_helpers


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def widget_with_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def pills(app: AppTest):
    """Support Streamlit releases that expose pills as a generic button group."""
    return app.pills if hasattr(app, "pills") else app.get("button_group")


def preserve_legacy_pills_state(app: AppTest) -> AppTest:
    if not hasattr(app, "pills"):
        selector = widget_with_label(pills(app), "Analysis view")
        if isinstance(selector.value, str):
            selector._value = [selector.value]
    return app


def assert_app_clean(app: AppTest) -> None:
    preserve_legacy_pills_state(app)
    assert not app.exception
    assert not [error for error in app.error if "Traceback" in str(error.value)]


def select_view(app: AppTest, label: str) -> AppTest:
    label = getattr(label, "content", label)
    selector = widget_with_label(pills(app), "Analysis view")
    value = label if hasattr(app, "pills") else [label]
    result = selector.set_value(value).run()
    if not hasattr(result, "pills"):
        # Streamlit 1.44 exposes st.pills as ButtonGroup but decodes its single
        # selection as a scalar; preserve the list-form widget state for reruns.
        widget_with_label(pills(result), "Analysis view")._value = [label]
    return result


def test_streamlit_hot_reload_recovers_stale_chart_helper(monkeypatch) -> None:
    monkeypatch.delattr(chart_helpers, "CONTINUOUS_COLOR_SCALES")
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    assert hasattr(chart_helpers, "CONTINUOUS_COLOR_SCALES")


def test_streamlit_hot_reload_recovers_stale_correlation_helper(monkeypatch) -> None:
    monkeypatch.delattr(correlation_helpers, "GROUPED_ASSOCIATION_API_VERSION")
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    assert correlation_helpers.GROUPED_ASSOCIATION_API_VERSION == 1
    assert "min_group_n" in inspect.signature(
        correlation_helpers.calculate_correlations
    ).parameters


def test_streamlit_table_value_filter_filters_rows_and_resets() -> None:
    assert "st.dataframe(" not in APP.read_text(encoding="utf-8")
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    all_rows = len(app.dataframe[0].value)
    picker = widget_with_label(
        app.multiselect, "Module composition filter columns"
    )
    app = picker.set_value(["Tissue"]).run()
    assert_app_clean(app)
    values = widget_with_label(app.multiselect, "Tissue values")
    app = values.set_value(["AC"]).run()
    assert_app_clean(app)
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[0].value.iloc[0]["Tissue"] == "AC"
    reset = next(button for button in app.button if button.label == "Reset row filters")
    app = reset.click().run()
    assert_app_clean(app)
    assert len(app.dataframe[0].value) == all_rows
    picker = widget_with_label(
        app.multiselect, "Module composition filter columns"
    )
    app = picker.set_value(["Genes"]).run()
    assert_app_clean(app)
    minimum = widget_with_label(app.number_input, "Genes minimum")
    app = minimum.set_value(1).run()
    assert_app_clean(app)
    assert app.dataframe[0].value["Genes"].ge(1).all()
    assert len(app.dataframe[0].value) < all_rows


def test_every_lazy_analysis_view_renders_cleanly() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    view_selector = widget_with_label(pills(app), "Analysis view")
    assert view_selector.value == "Associations" or view_selector.value == ["Associations"]
    for view in view_selector.options:
        app = select_view(app, view)
        assert_app_clean(app)


def test_feature_distributions_support_non_diagnosis_grouping() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "Feature distributions")
    assert_app_clean(app)
    grouping = widget_with_label(app.selectbox, "Group distributions by")
    assert grouping.value == "diagnosis_group"
    app = grouping.set_value("clusters").run()
    assert_app_clean(app)
    levels = widget_with_label(app.multiselect, "Distribution group levels")
    assert set(levels.options) == {"Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"}
    app = levels.set_value([1.0, 4.0]).run()
    assert_app_clean(app)
    summary = next(
        dataframe.value for dataframe in app.dataframe
        if "grouping_variable" in dataframe.value.columns
        and "distribution_group" in dataframe.value.columns
    )
    assert set(summary["grouping_variable"]) == {"clusters"}
    assert set(summary["distribution_group"]) == {"Cluster 1", "Cluster 4"}


def test_feature_distributions_can_use_tissue_module_eigengenes() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = select_view(app, "Feature distributions")
    assert_app_clean(app)
    app = widget_with_label(app.selectbox, "Module").set_value(1918).run()
    assert_app_clean(app)
    feature = widget_with_label(app.selectbox, "Module feature")
    assert "Module eigengene (PCA1 expression)" in feature.options
    app = feature.set_value("eigengene").run()
    assert_app_clean(app)
    resolution = widget_with_label(app.radio, "Resolution")
    assert resolution.options == ["Tissue resolved"]
    assert resolution.value == "Tissue resolved"
    components = widget_with_label(app.multiselect, "Tissue components")
    assert set(components.options) == {"TS: AC", "TS: DLPFC", "TS: PCG"}
    assert any(
        "cross-tissue pair eigengene is not defined" in str(caption.value)
        for caption in app.caption
    )


def test_associations_can_use_tissue_module_eigengenes() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = widget_with_label(app.selectbox, "Module").set_value(1918).run()
    assert_app_clean(app)
    feature = widget_with_label(app.selectbox, "Module feature")
    assert "Module eigengene (PCA1 expression)" in feature.options
    app = feature.set_value("eigengene").run(timeout=180)
    assert_app_clean(app)
    assert widget_with_label(app.radio, "Resolution").options == ["Tissue resolved"]
    components = widget_with_label(app.multiselect, "Tissue components")
    assert set(components.options) == {"TS: AC", "TS: DLPFC", "TS: PCG"}
    association_table = next(
        dataframe.value
        for dataframe in app.dataframe
        if "spearman_fdr_across_modules" in dataframe.value.columns
    )
    assert set(association_table["metric_family"]) == {"eigengene"}
    assert association_table["spearman_fdr_module_family_n"].between(1, 154).all()


def test_association_view_defaults_to_pooled_all_donor_association() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    pooled = widget_with_label(
        app.checkbox, "Show pooled association across displayed donors"
    )
    assert pooled.value is True
    assert any(
        "BH-adjusted only across the 154 modules" in str(caption.value)
        for caption in app.caption
    )
    association_table = next(
        dataframe.value
        for dataframe in app.dataframe
        if "spearman_fdr_across_modules" in dataframe.value.columns
    )
    assert len(association_table) == 8
    assert set(association_table["diagnosis_group"]) == {
        "Control",
        "MCI",
        "AD",
        "All donors (pooled)",
    }
    assert association_table["pearson_fdr_module_family_n"].between(1, 154).all()
    assert association_table["spearman_fdr_module_family_n"].between(1, 154).all()


def test_correlation_heatmap_supports_all_features_and_module_blocks() -> None:
    app = AppTest.from_file(APP, default_timeout=240).run()
    app = select_view(app, "Correlation heatmaps")
    assert_app_clean(app)
    scope = widget_with_label(app.radio, "Heatmap scope")
    app = preserve_legacy_pills_state(
        scope.set_value("All 154 modules: selected or all feature scores").run()
    )
    app = widget_with_label(
        app.selectbox, "Features for all-module heatmap and table"
    ).set_value("__all__").run()
    app = preserve_legacy_pills_state(app)
    component_selector = widget_with_label(
        app.selectbox, "Components for all-module heatmap and table"
    )
    assert "Both aggregate components (CT and TS)" in component_selector.options
    app = component_selector.set_value("__all__").run()
    app = preserve_legacy_pills_state(app)
    app = widget_with_label(app.selectbox, "Heatmap clustering").set_value("Modules").run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)

    table = next(
        dataframe.value
        for dataframe in app.dataframe
        if "absolute_correlation" in dataframe.value.columns
    )
    assert table["module"].nunique() == 154
    assert table["metric_family"].nunique() == 6
    assert set(table["component"]) == {"CT", "TS"}
    assert set(table["diagnosis_group"]) == {"AD"}
    assert {"correlation", "p_value", "fdr"}.issubset(table.columns)

    app = widget_with_label(app.radio, "Correlation rows").set_value(
        "At least one FDR < 0.05"
    ).run()
    assert_app_clean(app)
    significant = next(
        dataframe.value
        for dataframe in app.dataframe
        if "absolute_correlation" in dataframe.value.columns
    )
    assert significant["fdr"].lt(0.05).all()


def test_correlation_heatmap_supports_all_tissue_resolved_components() -> None:
    app = AppTest.from_file(APP, default_timeout=240).run()
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Resolution").set_value(
        "Tissue resolved"
    ).run()
    app = select_view(app, "Correlation heatmaps")
    assert_app_clean(app)
    scope = widget_with_label(app.radio, "Heatmap scope")
    app = preserve_legacy_pills_state(
        scope.set_value("All 154 modules: selected or all feature scores").run()
    )
    app = widget_with_label(
        app.selectbox, "Components for all-module heatmap and table"
    ).set_value("__all__").run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)

    table = next(
        dataframe.value
        for dataframe in app.dataframe
        if "absolute_correlation" in dataframe.value.columns
    )
    assert table["module"].nunique() == 154
    assert set(table["component"]) == {
        "TS_AC",
        "TS_DLPFC",
        "TS_PCGBA23",
        "CT_AC__DLPFC",
        "CT_AC__PCGBA23",
        "CT_DLPFC__PCGBA23",
    }
    assert table["component_label"].astype(str).str.contains("DLPFC").any()


def test_prediction_view_uses_leakage_reduced_exploratory_default() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = select_view(app, "Prediction")
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Prediction mode").set_value("benchmark").run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "Prediction reference design").value == (
        "development_frozen"
    )
    assert widget_with_label(app.selectbox, "Development AD–Control edge mask").value == (
        "per_module_fdr10"
    )
    predictor_design = widget_with_label(app.radio, "Predictor design")
    assert len(predictor_design.options) == 2
    app = predictor_design.set_value("module_connectivity").run()
    assert_app_clean(app)
    assert {tab.label for tab in app.tabs}.issuperset(
        {"Performance", "CT versus TS", "Diagnostics", "Coefficients", "Tables", "Methods"}
    )


def test_targeted_prediction_view_defaults_to_primary_control_derived_analysis() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = select_view(app, "Prediction")
    assert_app_clean(app)

    assert widget_with_label(app.radio, "Prediction mode").value == "targeted"
    assert widget_with_label(app.selectbox, "Targeted edge set").value == "all"
    assert widget_with_label(
        app.selectbox, "Module-score transformation"
    ).value in {"raw", "asinh", "rint"}
    assert widget_with_label(
        app.selectbox, "Targeted module definition"
    ).value == "control_derived"
    assert widget_with_label(
        app.selectbox, "Targeted LIONESS method"
    ).value == "control_anchored"
    assert widget_with_label(
        app.selectbox, "Targeted prediction outcome"
    ).value == "diagnosis_binary"
    assert widget_with_label(
        app.selectbox, "Eigengene source"
    ).value == "matched_multitissue"
    assert {tab.label for tab in app.tabs}.issuperset(
        {
            "Summary",
            "CT versus TS",
            "Transformation sensitivity",
            "Eigengene sources",
            "Panel selection",
            "OOF diagnostics",
            "Coefficients & KEGG",
            "Tables",
            "Methods",
        }
    )


def test_targeted_cluster_prediction_renders_all_resolved_blocks() -> None:
    app = AppTest.from_file(APP, default_timeout=300).run()
    app = select_view(app, "Prediction")
    app = preserve_legacy_pills_state(app)
    app = widget_with_label(app.selectbox, "Evidence tier").set_value(
        "exploratory"
    ).run()
    app = preserve_legacy_pills_state(app)
    app = widget_with_label(app.selectbox, "Targeted prediction outcome").set_value(
        "clusters"
    ).run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)
    assert any("Cluster prediction is exploratory" in warning.value for warning in app.warning)

    block = widget_with_label(app.selectbox, "OOF diagnostic predictor block")
    assert {
        "AC",
        "DLPFC",
        "PCG",
        "AC–DLPFC",
        "AC–PCG",
        "DLPFC–PCG",
        "All three CT tissue pairs",
        "All three TS tissues",
        "All six resolved components",
    }.issubset(set(block.options))
    app = block.set_value("all_resolved").run()
    assert_app_clean(app)


def test_streamlit_smoke_lioness_and_bonobo_module_definitions() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = preserve_legacy_pills_state(app)

    outcome = widget_with_label(app.selectbox, "Association outcome")
    assert len(outcome.options) == 16
    outcome.set_value("adnc")
    app = widget_with_label(app.selectbox, "Color points by").set_value(
        "age_at_death"
    ).run()
    app = preserve_legacy_pills_state(app)
    widget_with_label(app.selectbox, "Continuous color scale").set_value("Viridis")
    app = widget_with_label(app.radio, "Network estimator").set_value("bonobo").run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)

    edge_subset = widget_with_label(app.radio, "BONOBO edge subset")
    app = edge_subset.set_value("Significant edges").run()
    app = preserve_legacy_pills_state(app)
    app = widget_with_label(app.radio, "Significant-edge rule").set_value(
        "bh_fdr05"
    ).run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)

    app = widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    app = preserve_legacy_pills_state(app)
    assert_app_clean(app)
    network_method = widget_with_label(app.radio, "Network method")
    assert network_method.value == "bonobo"
    assert network_method.options == ["All-donor empirical-Bayes BONOBO"]


def test_streamlit_differential_filter_can_switch_bh_scope_and_cutoff() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)

    app = widget_with_label(app.radio, "AD–Control edge subset").set_value(
        "ad_control_discovery_fdr05"
    ).run()
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Differential-edge FDR scope").set_value(
        "per_module"
    ).run()
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Differential-edge FDR cutoff").set_value(
        0.10
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "Evaluation cohort").value == (
        "validation_ad_control"
    )

    widget_with_label(app.radio, "AD–Control edge subset").set_value("all").run()
    assert_app_clean(app)


def test_streamlit_module_finder_supports_each_ranking_criterion() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = select_view(app, "Module finder")
    assert_app_clean(app)
    cohort = widget_with_label(app.selectbox, "Cohort for CT–TS comparison")
    assert cohort.value == "All donors"
    assert set(cohort.options).issuperset({"All donors", "Control", "MCI", "AD"})
    pooled_table = next(
        dataframe.value
        for dataframe in app.dataframe
        if "ct_ts_diagnosis" in dataframe.value.columns
    )
    assert pooled_table["ct_ts_diagnosis"].eq("All donors").all()
    assert pooled_table["ct_ts_n"].min() > 300

    app = widget_with_label(
        app.selectbox, "Cohort for CT–TS comparison"
    ).set_value("AD").run()
    assert_app_clean(app)
    criterion = widget_with_label(app.radio, "Ranking criterion")
    assert criterion.value == "ct_ts"
    criterion.set_value("ad_control").run()
    assert_app_clean(app)
    widget_with_label(app.radio, "Ranking criterion").set_value("both").run()
    assert_app_clean(app)
    widget_with_label(app.selectbox, "Association-difference FDR filter").set_value(
        0.10
    ).run()
    assert_app_clean(app)


def test_streamlit_mdc_can_switch_to_raw_scale() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "MDC")
    assert_app_clean(app)

    scale = widget_with_label(app.radio, "MDC display scale")
    assert scale.value == "log2"
    scale.set_value("raw").run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "MDC display scale").value == "raw"

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "MDC display scale").value == "raw"


def test_streamlit_mdc_uses_selected_ad_control_edge_mask() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "MDC")
    assert_app_clean(app)

    app = widget_with_label(app.radio, "AD–Control edge subset").set_value(
        "ad_control_discovery_fdr05"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "Differential-edge FDR scope").value == "global"
    app = widget_with_label(app.radio, "Differential-edge FDR scope").set_value(
        "per_module"
    ).run()
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Differential-edge FDR cutoff").set_value(
        0.10
    ).run()
    assert_app_clean(app)
    assert any(
        "Filtered MDC is exploratory post-selection context" in warning.value
        for warning in app.warning
    ) or any(
        "AD–Control-filtered MDC is currently being generated" in info.value
        for info in app.info
    )

    widget_with_label(app.radio, "Network estimator").set_value("bonobo").run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "Network method").value == "bonobo"

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "AD–Control edge subset").value == (
        "ad_control_discovery_fdr05"
    )


def test_streamlit_pathway_resolved_mdc_controls_both_module_sets() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "MDC")
    assert_app_clean(app)
    assert any(tab.label == "Region-resolved MDC" for tab in app.tabs)
    assert any(tab.label == "Pathway-resolved MDC" for tab in app.tabs)
    assert widget_with_label(app.selectbox, "MDC enrichment resolution").value == (
        "pathway"
    )
    assert widget_with_label(app.selectbox, "Pathway detail").value

    app = widget_with_label(app.selectbox, "MDC enrichment resolution").set_value(
        "subcategory"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "KEGG sub-category detail").value

    app = widget_with_label(app.radio, "KEGG enrichment threshold").set_value(
        0.10
    ).run()
    assert_app_clean(app)
    app = widget_with_label(app.radio, "Module MDC rows").set_value(
        "MDC FDR-significant only"
    ).run()
    assert_app_clean(app)

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "MDC enrichment resolution").value == (
        "pathway"
    )
    widget_with_label(app.selectbox, "MDC enrichment resolution").set_value(
        "category"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "KEGG category detail").value


def test_nominal_cluster_statistics_never_use_numeric_correlation() -> None:
    rows = []
    for module in (1, 2):
        for cluster, values in {
            1: [0, 1, 2, 3, 4],
            2: [5, 6, 7, 8, 9],
            3: [10, 11, 12, 13],  # displayed but excluded: n < 5
        }.items():
            rows.extend(
                {
                    "module": module,
                    "component": "CT",
                    "diagnosis_group": "All donors",
                    "metric_value": value + module,
                    "clusters": cluster,
                }
                for value in values
            )
    source = pd.DataFrame(rows)
    result = calculate_categorical_associations(
        source,
        ["module", "component", "diagnosis_group"],
    )
    assert set(result["clusters_tested"]) == {"1,2"}
    assert set(result["clusters_excluded_small_n"]) == {"3"}
    assert result["epsilon_squared"].between(0, 1).all()
    assert not {"pearson_r", "spearman_rho"}.intersection(result.columns)
    adjusted = add_categorical_across_module_fdr(
        result,
        family_columns=["component", "diagnosis_group", "outcome"],
    )
    assert adjusted["categorical_fdr_module_family_n"].eq(2).all()


def test_streamlit_nominal_cluster_association_and_heatmap_render() -> None:
    app = AppTest.from_file(APP, default_timeout=240).run()
    assert_app_clean(app)
    app = preserve_legacy_pills_state(app)
    outcome = widget_with_label(app.selectbox, "Association outcome")
    app = outcome.set_value("clusters").run()
    assert_app_clean(app)
    assert any("Cluster labels are nominal" in caption.value for caption in app.caption)

    app = select_view(app, "Correlation heatmaps")
    assert_app_clean(app)
    association_type = widget_with_label(app.radio, "Association type")
    app = association_type.set_value("Nominal ROSMAP clusters").run()
    assert_app_clean(app)
    assert widget_with_label(app.checkbox, "Show only rows with at least one cluster FDR < 0.05")


def test_streamlit_configurable_grouped_and_ordinal_associations_render() -> None:
    app = AppTest.from_file(APP, default_timeout=240).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "Group correlations by").value == "diagnosis_group"
    app = widget_with_label(app.selectbox, "Group correlations by").set_value(
        "clusters"
    ).run()
    assert_app_clean(app)
    levels = widget_with_label(app.multiselect, "Group levels")
    assert set(levels.options) == {"Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"}
    app = levels.set_value([1.0, 2.0]).run()
    assert_app_clean(app)
    app = widget_with_label(app.selectbox, "Trend-line rule").set_value("none").run()
    assert_app_clean(app)
    grouped_table = next(
        dataframe.value for dataframe in app.dataframe
        if "grouping_variable" in dataframe.value.columns
    )
    assert set(grouped_table["grouping_variable"]) == {"clusters"}
    assert set(grouped_table.loc[~grouped_table["is_pooled"], "grouping_level"]) == {"1", "2"}
    assert {"pearson_r", "spearman_rho", "pearson_fdr_across_modules",
            "spearman_fdr_across_modules"}.issubset(grouped_table.columns)

    app = widget_with_label(app.selectbox, "Association outcome").set_value("cogdx").run()
    assert_app_clean(app)
    interpretation = widget_with_label(app.radio, "Outcome interpretation")
    assert interpretation.value == "numeric"
    app = interpretation.set_value("categorical").run()
    assert_app_clean(app)
    categorical_table = next(
        dataframe.value for dataframe in app.dataframe
        if "category_variable" in dataframe.value.columns
    )
    assert set(categorical_table["category_variable"]) == {"cogdx"}
    assert not {"pearson_r", "spearman_rho"}.intersection(categorical_table.columns)


def test_numeric_correlation_heatmap_supports_selected_category_levels() -> None:
    app = AppTest.from_file(APP, default_timeout=240).run()
    assert_app_clean(app)
    app = select_view(app, "Correlation heatmaps")
    assert_app_clean(app)
    grouping = widget_with_label(app.selectbox, "Group heatmap correlations by")
    assert grouping.value == "diagnosis_group"
    app = grouping.set_value("clusters").run()
    assert_app_clean(app)
    levels = widget_with_label(app.multiselect, "Heatmap group levels")
    assert set(levels.options) == {"Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"}
    app = levels.set_value([1.0, 4.0]).run()
    assert_app_clean(app)
    grouped_table = next(
        dataframe.value for dataframe in app.dataframe
        if {"grouping_variable", "grouping_label", "correlation_method"}.issubset(
            dataframe.value.columns
        )
    )
    assert set(grouped_table["grouping_variable"]) == {"clusters"}
    assert set(grouped_table["grouping_label"]) == {"Cluster 1", "Cluster 4"}
