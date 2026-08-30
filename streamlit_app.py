"""Public explorer for ROSMAP LIONESS and BONOBO module networks."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Community Cloud may hot-reload this entrypoint while retaining an older
# app_helpers.charts module in memory. Reload the helper when its current public API
# is absent so a repository update cannot leave the two modules out of sync.
from app_helpers import charts as _chart_helpers

if not all(
    hasattr(_chart_helpers, name)
    for name in (
        "CONTINUOUS_COLOR_SCALES",
        "categorical_association_figure",
        "EDGE_COMPONENT_LABELS",
        "edge_volcano_figure",
        "module_finder_figure",
        "mdc_entropy_figure",
        "pathway_mdc_heatmap_figure",
        "prediction_performance_figure",
        "PREDICTION_BLOCK_ORDERING_API_VERSION",
        "targeted_primary_comparison_figure",
        "targeted_selection_frequency_figure",
        "targeted_fold_robustness_figure",
        "targeted_transform_comparison_figure",
        "targeted_transform_heatmap_figure",
        "targeted_eigengene_source_comparison_figure",
        "targeted_panel_overlap_figure",
        "clustered_correlation_group_order",
        "grouped_association_figure",
    )
):
    _chart_helpers = importlib.reload(_chart_helpers)

from app_helpers.charts import (
    CONTINUOUS_COLOR_SCALES,
    EDGE_COMPONENT_LABELS,
    aggregate_to_long,
    categorical_association_figure,
    cluster_association_heatmap_figure,
    classification_diagnostic_rows,
    correlation_heatmap_figure,
    clustered_correlation_group_order,
    distribution_figure,
    distribution_summary,
    edge_volcano_figure,
    edge_summary_figure,
    mdc_entropy_figure,
    mdc_module_figure,
    mdc_overview_figure,
    mdc_resolved_heatmap_figure,
    mdc_resolved_module_figure,
    grouped_association_figure,
    module_finder_figure,
    module_entropy_figure,
    module_region_composition_figure,
    module_size_distribution_figure,
    pathway_mdc_detail_figure,
    pathway_mdc_heatmap_figure,
    prediction_coefficient_figure,
    prediction_confusion_figure,
    prediction_ct_ts_figure,
    prediction_curve_figure,
    prediction_heatmap_figure,
    prediction_observed_figure,
    prediction_error_figure,
    prediction_performance_figure,
    prediction_threshold_figure,
    targeted_fold_robustness_figure,
    targeted_panel_overlap_figure,
    targeted_primary_comparison_figure,
    targeted_selection_frequency_figure,
    targeted_transform_comparison_figure,
    targeted_transform_heatmap_figure,
    targeted_eigengene_source_comparison_figure,
    resolved_to_long,
)

from app_helpers import correlations as _correlation_helpers

if getattr(_correlation_helpers, "GROUPED_ASSOCIATION_API_VERSION", 0) < 1:
    _correlation_helpers = importlib.reload(_correlation_helpers)

from app_helpers.correlations import (
    add_across_module_fdr,
    add_categorical_across_module_fdr,
    calculate_categorical_associations,
    calculate_correlations,
)
from app_helpers.module_finder import (
    FINDER_CRITERIA,
    build_module_finder_table,
    build_pooled_ct_ts_statistics,
)
from app_helpers.table_controls import filterable_dataframe
from app_helpers.streamlit_compat import plotly_chart as render_plotly_chart
from app_helpers.streaming_associations import (
    stream_categorical_associations,
    stream_diagnosis_categorical_associations,
    stream_grouped_correlations,
    stream_pooled_correlations,
)
from app_helpers.drive_data import data_source_label, ensure_data_path
from app_helpers import data as _data_helpers

if not all(
    hasattr(_data_helpers, name)
    for name in (
        "collapse_pathway_mdc_rows", "SCORE_TRANSFORM_LABELS",
        "EIGENGENE_SOURCE_LABELS",
        "ASSOCIATION_GROUP_LABELS", "association_level_label",
        "PREDICTION_BLOCK_ORDER",
    )
):
    _data_helpers = importlib.reload(_data_helpers)

from app_helpers.data import (
    ANALYSIS_SUBSET_LABELS,
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_LEVEL_ORDERS,
    ASSOCIATION_OUTCOME_LABELS,
    BONOBO_EDGE_RULE_LABELS,
    BONOBO_FEATURE_LABELS,
    COLOR_LABELS,
    COMPONENT_ORDER,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    DATA_DIR,
    DIFFERENTIAL_EDGE_RULE_LABELS,
    DIAGNOSIS_ORDER,
    FEATURE_LABELS,
    HOVER_LABELS,
    METHOD_LABELS,
    MDC_ENRICHMENT_RESOLUTION_LABELS,
    ESTIMATOR_LABELS,
    EIGENGENE_SOURCE_LABELS,
    FDR_SCOPE_LABELS,
    FDR_THRESHOLD_LABELS,
    MODULE_SET_LABELS,
    MODULE_SET_METHODS,
    NUMERIC_OUTCOMES,
    OUTCOME_LABELS,
    PHENOTYPE_LABELS,
    PREDICTION_BLOCK_LABELS,
    PREDICTION_BLOCK_ORDER,
    PREDICTION_DESIGN_LABELS,
    PREDICTION_MASK_LABELS,
    PREDICTION_MODEL_LABELS,
    PREDICTION_OUTCOME_LABELS,
    PREDICTION_REFERENCE_LABELS,
    TARGETED_PANEL_LABELS,
    TARGETED_PREDICTION_MODE_LABELS,
    TARGETED_TIER_LABELS,
    SCALE_LABELS,
    SCORE_TRANSFORM_LABELS,
    SCORE_NORMALIZATION_LABELS,
    SELECTABLE_ASSOCIATION_OUTCOMES,
    association_level_label,
    build_pathway_mdc_rows,
    collapse_pathway_mdc_rows,
    association_kegg_subtitles,
    dataframe_to_tsv_bytes,
    differential_data_available,
    differential_mdc_data_available,
    filter_kegg_enrichments,
    load_aggregate,
    load_aggregate_scope,
    load_aggregate_statistics,
    load_data_manifest,
    load_feature_definitions,
    load_edge_summaries,
    load_kegg,
    load_kegg_tsv_bytes,
    load_cluster_association_statistics,
    load_mdc_summary,
    load_mdc_resolved,
    load_module_annotations,
    load_module_details,
    load_resolved,
    load_resolved_scope,
    load_resolved_statistics,
    load_sample_metadata,
    load_tissue_mapping,
    load_prediction_bootstrap,
    load_prediction_coefficients,
    load_prediction_confusion,
    load_prediction_curves,
    load_prediction_diagnostics,
    load_prediction_manifest,
    load_prediction_performance,
    load_prediction_whole_network_features,
    load_targeted_prediction_manifest,
    load_targeted_prediction_table,
    load_volcano_bins,
    load_volcano_candidates,
    module_label,
    prediction_data_available,
    targeted_prediction_data_available,
    require_data_files,
    selected_annotation,
    summarize_pathway_mdc_rows,
)


st.set_page_config(
    page_title="ROSMAP Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .app-note {color: #526273; font-size: 0.93rem;}
      .kegg-note {padding: .6rem .85rem; background: #f3f7fb; border-left: 4px solid #2c7fb8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_annotations(module_set: str) -> pd.DataFrame:
    return load_module_annotations(module_set)


@st.cache_data(show_spinner=False)
def cached_module_details(module_set: str) -> pd.DataFrame:
    return load_module_details(module_set=module_set)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_aggregate(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    feature: str,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
) -> pd.DataFrame:
    return load_aggregate(
        method, module, feature, module_set=module_set,
        estimator=estimator, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_resolved(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    feature: str,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
) -> pd.DataFrame:
    return load_resolved(
        method, module, feature, module_set=module_set,
        estimator=estimator, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    )


@st.cache_data(show_spinner=False)
def cached_sample_metadata() -> pd.DataFrame:
    return load_sample_metadata()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_mdc_summary(
    module_set: str,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    return load_mdc_summary(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_mdc_resolved(
    module_set: str,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    return load_mdc_resolved(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
    )


@st.cache_data(show_spinner=False, max_entries=4)
def cached_pathway_mdc_rows(
    module_set: str,
    enrichment_fdr_threshold: float,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    return build_pathway_mdc_rows(
        load_mdc_summary(
            module_set,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        ),
        load_mdc_resolved(
            module_set,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        ),
        load_kegg(module_set=module_set),
        enrichment_fdr_threshold=enrichment_fdr_threshold,
    )


@st.cache_data(show_spinner=False, max_entries=6)
def cached_edge_summaries(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    return load_edge_summaries(
        estimator, method, module, module_set=module_set, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
    )


@st.cache_data(show_spinner=False)
def cached_data_manifest() -> dict[str, object]:
    return load_data_manifest()


@st.cache_data(show_spinner=False, max_entries=16)
def cached_aggregate_stats(
    module_set: str,
    estimator: str,
    method: str,
    module: int | None,
    phenotype: str | None,
    feature: str | None,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    return load_aggregate_statistics(
        method, module, phenotype, feature, module_set=module_set,
        estimator=estimator, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
    )


@st.cache_data(
    show_spinner="Calculating pooled-donor CT–TS module statistics…",
    max_entries=4,
)
def cached_pooled_module_finder_stats(
    module_set: str,
    estimator: str,
    method: str,
    phenotype: str,
    feature: str,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    source = load_aggregate_scope(
        method,
        module=None,
        metric_family=feature,
        module_set=module_set,
        estimator=estimator,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        include_embedded_metadata=False,
    )
    source = attach_metadata(source)
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        source = source.loc[source["ad_control_split"].eq(split_value)].copy()
    return build_pooled_ct_ts_statistics(source, phenotype=phenotype)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_resolved_stats(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    phenotype: str,
    feature: str,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    return load_resolved_statistics(
        method, module, phenotype, feature, module_set=module_set,
        estimator=estimator, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_volcano_candidates(
    module_set: str, estimator: str, method: str, module: int
) -> pd.DataFrame:
    return load_volcano_candidates(module_set, estimator, method, module)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_volcano_bins(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    differential_fdr_scope: str,
) -> pd.DataFrame:
    return load_volcano_bins(
        module_set, estimator, method, module, differential_fdr_scope
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_kegg(module_set: str, module: int | None) -> pd.DataFrame:
    return load_kegg(module, module_set=module_set)


@st.cache_data(show_spinner=False)
def cached_kegg_tsv(module_set: str) -> bytes:
    return load_kegg_tsv_bytes(module_set)


@st.cache_data(show_spinner=False)
def cached_feature_definitions() -> pd.DataFrame:
    return load_feature_definitions()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prediction_performance(
    reference_provenance: str,
    module_definition: str,
    network_method: str,
    predictor_design: str,
    edge_mask: str,
    score_normalization: str,
    outcome: str | None = None,
) -> pd.DataFrame:
    return load_prediction_performance(
        reference_provenance, module_definition, network_method, predictor_design, edge_mask,
        score_normalization, outcome,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prediction_bootstrap(
    reference_provenance: str,
    module_definition: str,
    network_method: str,
    predictor_design: str,
    edge_mask: str,
    score_normalization: str,
    outcome: str | None = None,
) -> pd.DataFrame:
    return load_prediction_bootstrap(
        reference_provenance, module_definition, network_method, predictor_design, edge_mask,
        score_normalization, outcome,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prediction_curves(**filters: str | None) -> pd.DataFrame:
    return load_prediction_curves(**filters)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prediction_confusion(**filters: str | None) -> pd.DataFrame:
    return load_prediction_confusion(**filters)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prediction_coefficients(**filters: str | None) -> pd.DataFrame:
    return load_prediction_coefficients(**filters)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_prediction_diagnostics(**filters: str | None) -> pd.DataFrame:
    return load_prediction_diagnostics(**filters)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_prediction_whole_network(**filters: str | None) -> pd.DataFrame:
    return load_prediction_whole_network_features(**filters)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_targeted_prediction_table(
    table: str,
    filters: tuple[tuple[str, object], ...] = (),
) -> pd.DataFrame:
    return load_targeted_prediction_table(table, **dict(filters))


@st.cache_data(show_spinner=False)
def cached_tissue_mapping() -> pd.DataFrame:
    return load_tissue_mapping()


def attach_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = cached_sample_metadata()
    additional = [
        column for column in metadata.columns if column == "sample_id" or column not in frame.columns
    ]
    return frame.merge(metadata[additional], on="sample_id", how="left", validate="many_to_one")


def ordered_association_levels(frame: pd.DataFrame, variable: str) -> list[object]:
    """Return observed non-missing category levels in a stable biological order."""

    if variable == "__all__":
        return ["__all__"]
    observed = frame[variable].dropna().drop_duplicates().tolist()
    observed_by_text = {str(value): value for value in observed}
    ordered = [
        observed_by_text[str(value)]
        for value in ASSOCIATION_LEVEL_ORDERS.get(variable, [])
        if str(value) in observed_by_text
    ]
    included = {str(value) for value in ordered}
    extras = sorted(
        (value for value in observed if str(value) not in included),
        key=lambda value: str(value),
    )
    return [*ordered, *extras]


def standardize_stored_associations(
    statistics: pd.DataFrame,
    *,
    resolved: bool,
    scale: str,
    components: tuple[str, ...],
    phenotype: str | None,
) -> pd.DataFrame:
    """Convert stored group statistics to one row per module and component."""

    if resolved:
        result = statistics.loc[statistics["component"].isin(components)].copy()
        result["pearson_r"] = result[f"r_{scale}"]
        result["pearson_p"] = result[f"p_{scale}"]
        result["spearman_rho"] = result["rho"]
        result["spearman_p"] = result["p_spearman"]
    else:
        parts: list[pd.DataFrame] = []
        for component in components:
            part = statistics.copy()
            part["component"] = component
            part["component_label"] = f"{component} aggregate"
            part["pearson_r"] = part[f"r_{scale}_{component}"]
            part["pearson_p"] = part[f"p_{scale}_{component}"]
            part["spearman_rho"] = part[f"rho_{component}"]
            part["spearman_p"] = part[f"p_spearman_{component}"]
            parts.append(part)
        result = pd.concat(parts, ignore_index=True)
    if phenotype is None:
        if "phenotype" not in result:
            raise ValueError("Stored association table is missing its phenotype column")
        result["outcome"] = result["phenotype"]
    else:
        result["outcome"] = phenotype
    columns = [
        "module",
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "outcome",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
    ]
    return result[columns]


@st.cache_data(
    show_spinner="Calculating nominal cluster associations across modules…",
    max_entries=6,
)
def cached_module_set_cluster_associations(
    module_set: str,
    module_count: int,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    include_pooled: bool,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    """Return Kruskal/epsilon-squared rows; never correlate cluster codes."""

    selected_groups = list(diagnoses)
    if include_pooled:
        selected_groups.append("All donors")
    if differential_edge_rule == "all":
        result = load_cluster_association_statistics(
            module_set=module_set,
            estimator=estimator,
            method=method,
            resolution="resolved" if resolved else "aggregate",
            metric_family=feature,
            edge_rule=edge_rule,
        )
        result = result.loc[
            result["component"].isin(components)
            & result["diagnosis_group"].isin(selected_groups)
        ].copy()
    else:
        module_ids = tuple(
            sorted(cached_annotations(module_set)["module"].astype(int).unique())
        )
        result = stream_diagnosis_categorical_associations(
            module_ids, cached_sample_metadata(), module_set=module_set,
            estimator=estimator, method=method, resolved=resolved,
            feature=feature, category_variable="clusters", scale="raw",
            components=components, diagnoses=diagnoses,
            include_pooled=include_pooled, min_group_n=5,
            edge_rule=edge_rule, differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
            analysis_subset=analysis_subset,
        )
        result = add_categorical_across_module_fdr(
            result,
            family_columns=["metric_family", "component", "diagnosis_group", "outcome"],
        )
    families = result.groupby(
        ["component", "diagnosis_group"], observed=True, sort=False
    )["module"].nunique()
    if not families.empty and not families.eq(int(module_count)).all():
        raise ValueError(
            "Nominal association families do not contain every module in the selected "
            f"definition ({module_count} expected): {families.to_dict()}"
        )
    return result


@st.cache_data(
    show_spinner="Calculating association FDR across the selected module set…",
    max_entries=6,
)
def cached_module_set_associations(
    module_set: str,
    module_count: int,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    phenotype: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    include_pooled: bool,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    """Return Pearson/Spearman statistics with BH applied across modules only."""

    if phenotype == "clusters":
        return cached_module_set_cluster_associations(
            module_set, module_count, estimator, method, resolved, feature,
            components, diagnoses, include_pooled, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )

    statistic_arguments = {
        "method": method,
        "module": None,
        "phenotype": phenotype,
        "metric_family": feature,
        "module_set": module_set,
        "estimator": estimator,
        "edge_rule": edge_rule,
        "differential_edge_rule": differential_edge_rule,
        "differential_fdr_scope": differential_fdr_scope,
        "differential_fdr_threshold": differential_fdr_threshold,
        "score_normalization": score_normalization,
        "analysis_subset": analysis_subset,
    }
    if resolved:
        stored = load_resolved_statistics(**statistic_arguments)
    else:
        stored = load_aggregate_statistics(**statistic_arguments)
    group_statistics = standardize_stored_associations(
        stored,
        resolved=resolved,
        scale=scale,
        components=components,
        phenotype=phenotype,
    )
    group_statistics = group_statistics.loc[
        group_statistics["diagnosis_group"].isin(diagnoses)
    ].copy()
    group_statistics = add_across_module_fdr(
        group_statistics,
        family_columns=["component", "diagnosis_group", "outcome"],
    )

    families = group_statistics.groupby(
        ["component", "diagnosis_group", "outcome"],
        observed=True,
        sort=False,
    )["module"].nunique()
    if not families.empty and not families.eq(int(module_count)).all():
        raise ValueError(
            "Stored association families do not contain every module in the selected "
            f"definition ({module_count} expected): {families.to_dict()}"
        )
    if not include_pooled:
        return group_statistics

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    pooled_statistics = stream_pooled_correlations(
        module_ids, cached_sample_metadata(), module_set=module_set,
        estimator=estimator, method=method, resolved=resolved, feature=feature,
        component=None, outcomes=(phenotype,), scale=scale,
        diagnoses=diagnoses, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization, analysis_subset=analysis_subset,
    )
    pooled_statistics = pooled_statistics.loc[
        pooled_statistics["component"].isin(components)
    ].copy()
    pooled_statistics = add_across_module_fdr(
        pooled_statistics,
        family_columns=["component", "diagnosis_group", "outcome"],
    )
    pooled_families = pooled_statistics.groupby(
        ["component", "diagnosis_group", "outcome"],
        observed=True,
        sort=False,
    )["module"].nunique()
    if not pooled_families.empty and not pooled_families.eq(int(module_count)).all():
        raise ValueError(
            "Pooled association families do not contain every module in the selected "
            f"definition ({module_count} expected): {pooled_families.to_dict()}"
        )
    return pd.concat([group_statistics, pooled_statistics], ignore_index=True)


@st.cache_data(
    show_spinner="Calculating grouped associations across the selected module set…",
    max_entries=4,
)
def cached_grouped_module_set_associations(
    module_set: str,
    module_count: int,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    phenotype: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    grouping_variable: str,
    grouping_levels: tuple[object, ...],
    include_pooled: bool,
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    """Return grouped Pearson/Spearman statistics with module-set BH FDR.

    Diagnosis grouping reuses the compact stored correlation catalog. Other
    grouping variables are calculated lazily from one feature/resolution scope.
    Every cache-key field fixes a statistical family or donor filter.
    """

    min_group_n = int(min_group_n)
    if grouping_variable == "diagnosis_group":
        result = cached_module_set_associations(
            module_set, module_count, estimator, method, resolved, feature,
            phenotype, scale, components, tuple(str(value) for value in grouping_levels),
            include_pooled, edge_rule, differential_edge_rule,
            differential_fdr_scope, differential_fdr_threshold,
            score_normalization, analysis_subset,
        ).copy()
        result["grouping_variable"] = "diagnosis_group"
        result["grouping_level"] = result["diagnosis_group"].where(
            ~result["diagnosis_group"].eq("All donors"), "__pooled__"
        )
        result["grouping_label"] = result["grouping_level"].replace(
            {"__pooled__": "All displayed donors (pooled)"}
        )
        result["is_pooled"] = result["grouping_level"].eq("__pooled__")
        result["minimum_group_n"] = min_group_n
        result["eligible"] = result["n"].ge(min_group_n) & (
            result["pearson_r"].notna() | result["spearman_rho"].notna()
        )
        result["unavailable_reason"] = np.where(
            result["n"].lt(min_group_n), f"n < {min_group_n}",
            np.where(result["eligible"], "", "constant or unavailable values"),
        )
        return result

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    result = stream_grouped_correlations(
        module_ids, cached_sample_metadata(), module_set=module_set,
        estimator=estimator, method=method, resolved=resolved, feature=feature,
        phenotype=phenotype, scale=scale, components=components,
        diagnoses=diagnoses, grouping_variable=grouping_variable,
        grouping_levels=grouping_levels, include_pooled=include_pooled,
        min_group_n=min_group_n, edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization, analysis_subset=analysis_subset,
    )
    if not result.empty:
        result = add_across_module_fdr(
            result,
            family_columns=[
                "component", "outcome", "grouping_variable", "grouping_level",
            ],
        )
        result["is_pooled"] = result["grouping_level"].astype(str).eq("__pooled__")
        result["grouping_label"] = result["grouping_level"].map(
            lambda value: (
                "All displayed donors (pooled)"
                if str(value) == "__pooled__"
                else association_level_label(grouping_variable, value)
            )
        )
        # A string key avoids mixed numeric/"__pooled__" object columns in
        # Streamlit/Arrow tables while preserving the human-readable label.
        result["grouping_level"] = result["grouping_level"].astype(str)

    if not result.empty:
        family_sizes = result.loc[~result["is_pooled"]].groupby(
            ["component", "outcome", "grouping_variable", "grouping_level"],
            observed=True, dropna=False,
        )["module"].nunique()
        if not family_sizes.empty and not family_sizes.eq(int(module_count)).all():
            raise ValueError(
                "Grouped association families do not contain every module in the "
                f"selected definition ({module_count} expected): {family_sizes.to_dict()}"
            )
    return result


@st.cache_data(
    show_spinner="Calculating categorical associations across the selected module set…",
    max_entries=4,
)
def cached_categorical_module_set_associations(
    module_set: str,
    module_count: int,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    category_variable: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    category_levels: tuple[object, ...],
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    """Return a generic Kruskal–Wallis/epsilon-squared module-set catalog."""

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    result = stream_categorical_associations(
        module_ids, cached_sample_metadata(), module_set=module_set,
        estimator=estimator, method=method, resolved=resolved, feature=feature,
        category_variable=category_variable, scale=scale, components=components,
        diagnoses=diagnoses, category_levels=category_levels,
        min_group_n=int(min_group_n), edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization, analysis_subset=analysis_subset,
    )
    if result.empty:
        return result
    result["category_variable"] = category_variable
    result = add_categorical_across_module_fdr(
        result,
        family_columns=["metric_family", "component", "outcome", "category_variable"],
    )
    family_sizes = result.groupby(
        ["component", "outcome", "category_variable"], observed=True, sort=False,
    )["module"].nunique()
    if not family_sizes.empty and not family_sizes.eq(int(module_count)).all():
        raise ValueError(
            "Categorical association families do not contain every module in the "
            f"selected definition ({module_count} expected): {family_sizes.to_dict()}"
        )
    return result


def add_correlation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["outcome_label"] = result["outcome"].map(OUTCOME_LABELS)
    result["feature_label"] = result["metric_family"].map(FEATURE_LABELS)
    result["heatmap_row"] = result["feature_label"] + " · " + result["component_label"]
    return result


@st.cache_data(show_spinner="Calculating the selected module correlation matrix…", max_entries=6)
def cached_module_correlations(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    resolved: bool,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    if resolved:
        source = load_resolved_scope(
            method, module=module, module_set=module_set,
            estimator=estimator, edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(
            method, module=module, module_set=module_set,
            estimator=estimator, edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = aggregate_to_long(source, "rint")
    long = attach_metadata(long)
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        long = long.loc[long["ad_control_split"].eq(split_value)]
    all_donors = long.copy()
    all_donors["diagnosis_group"] = "All donors"
    long = pd.concat([long, all_donors], ignore_index=True)
    summary = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=NUMERIC_OUTCOMES,
    )
    return add_correlation_labels(summary)


@st.cache_data(show_spinner="Calculating correlations across all modules…", max_entries=3)
def cached_all_module_correlations(
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    component: str,
    diagnosis: str,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    feature_filter = None if feature == "__all__" else feature
    component_filter = None if component == "__all__" else component
    summaries: list[pd.DataFrame] = []

    # The diagnosis-stratified statistics are already stored. Reading them avoids
    # loading all donor-level values just to reconstruct Control/MCI/AD summaries.
    if diagnosis != "All donors":
        statistic_arguments = {
            "method": method,
            "module": None,
            "phenotype": None,
            "metric_family": feature_filter,
            "module_set": module_set,
            "estimator": estimator,
            "edge_rule": edge_rule,
            "differential_edge_rule": differential_edge_rule,
            "differential_fdr_scope": differential_fdr_scope,
            "differential_fdr_threshold": differential_fdr_threshold,
            "score_normalization": score_normalization,
            "analysis_subset": analysis_subset,
            "diagnosis_group": None if diagnosis == "All diagnosis groups" else diagnosis,
        }
        if resolved:
            stored = load_resolved_statistics(
                **statistic_arguments,
                component=component_filter,
            )
            selected_components = tuple(
                sorted(
                    stored["component"].dropna().astype(str).unique(),
                    key=lambda value: (
                        COMPONENT_ORDER.index(value)
                        if value in COMPONENT_ORDER
                        else len(COMPONENT_ORDER),
                        value,
                    ),
                )
            )
        else:
            stored = load_aggregate_statistics(**statistic_arguments)
            selected_components = (
                ("CT", "TS") if component_filter is None else (component_filter,)
            )
        group_summary = standardize_stored_associations(
            stored,
            resolved=resolved,
            scale="rint",
            components=selected_components,
            phenotype=None,
        )
        if diagnosis == "All diagnosis groups":
            group_summary = group_summary.loc[
                group_summary["diagnosis_group"].isin(DIAGNOSIS_ORDER)
            ]
        summaries.append(group_summary)

    # Pooled-donor correlations are intentionally calculated from the anonymous
    # donor rows because the original robustness tables are diagnosis-stratified.
    if diagnosis in {"All donors", "All diagnosis groups"}:
        module_ids = tuple(
            sorted(cached_annotations(module_set)["module"].astype(int).unique())
        )
        summaries.append(
            stream_pooled_correlations(
                module_ids, cached_sample_metadata(), module_set=module_set,
                estimator=estimator, method=method, resolved=resolved,
                feature=feature_filter, component=component_filter,
                outcomes=tuple(NUMERIC_OUTCOMES), scale="rint",
                diagnoses=tuple(DIAGNOSIS_ORDER), edge_rule=edge_rule,
                differential_edge_rule=differential_edge_rule,
                differential_fdr_scope=differential_fdr_scope,
                differential_fdr_threshold=differential_fdr_threshold,
                score_normalization=score_normalization,
                analysis_subset=analysis_subset,
            )
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary = add_across_module_fdr(
        summary,
        family_columns=[
            "metric_family",
            "component",
            "diagnosis_group",
            "outcome",
        ],
    )
    summary = add_correlation_labels(summary)
    for column in (
        "metric_family", "component", "component_label", "diagnosis_group",
        "outcome", "outcome_label", "feature_label", "heatmap_row",
        "unavailable_reason",
    ):
        if column in summary:
            summary[column] = summary[column].astype("category")
    return summary


def readable_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def ordered_prediction_blocks(values: Iterable[object]) -> list[str]:
    """Order available prediction blocks from resolved TS through combined scopes."""

    observed = [str(value) for value in pd.Series(values).dropna().drop_duplicates()]
    ordered = [value for value in PREDICTION_BLOCK_ORDER if value in observed]
    ordered.extend(value for value in observed if value not in ordered)
    return ordered


def fdr_text(module_manifest: dict[str, object], module_count: int) -> str:
    return (
        f"Association-plot module-set FDR uses Benjamini–Hochberg across the {module_count} "
        "modules only, separately for every fixed phenotype, feature, component, diagnosis/"
        "cohort, estimator/network method, edge rule, score scale, and Pearson/Spearman test. "
        "Missing or constant correlations are excluded from the tested-hypothesis count. "
        "The robust-statistics downloads retain the older all-12, primary-five, and broad "
        "within-outcome correction columns for provenance and backward comparison. KEGG FDR "
        "is independent and comes from the supplied enrichment analysis."
    )


def _targeted_filters(**values: object | None) -> tuple[tuple[str, object], ...]:
    return tuple((key, value) for key, value in values.items() if value is not None)


def render_targeted_prediction_view() -> None:
    """Render repeated nested-CV targeted-module results without touching the benchmark."""

    manifest = load_targeted_prediction_manifest()
    transcriptomic_variants = {
        "transcriptomics_only",
        "covariates_plus_transcriptomics",
        "network_plus_transcriptomics",
        "covariates_plus_network_plus_transcriptomics",
    }
    masked_available = bool(manifest.get("masked_sensitivity_available", False))
    with st.sidebar:
        edge_options = ["all"]
        if masked_available:
            edge_options.extend(
                ["global_fdr05", "global_fdr10", "per_module_fdr05", "per_module_fdr10"]
            )
        edge_mask = st.selectbox(
            "Targeted edge set",
            options=edge_options,
            format_func=lambda value: PREDICTION_MASK_LABELS.get(value, value),
            key="targeted_edge_mask",
        )
        if edge_mask == "all":
            score_normalization = "standard_pruned"
        else:
            score_normalization = st.radio(
                "Targeted score normalization",
                options=["standard_pruned", "retained_edge"],
                format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
                key="targeted_score_normalization",
            )
    masked_selection = edge_mask != "all"
    performance_table = "masked_oof_performance" if masked_selection else "oof_performance"
    fold_table = "masked_fold_performance" if masked_selection else "fold_performance"
    coefficient_table = "masked_coefficients" if masked_selection else "coefficients"
    diagnostic_table = "masked_oof_predictions" if masked_selection else "oof_predictions"
    performance_catalog = cached_targeted_prediction_table(
        performance_table,
        _targeted_filters(
            edge_mask=edge_mask,
            score_normalization=score_normalization,
        ),
    )
    if performance_catalog.empty:
        st.info("The targeted-prediction manifest is present, but no completed OOF results exist.")
        return
    if masked_selection:
        score_transform = "asinh"
    else:
        available_transforms = [
            value for value in ("raw", "asinh", "rint")
            if value in set(performance_catalog["score_transform"].astype(str))
        ]
        with st.sidebar:
            score_transform = st.selectbox(
                "Module-score transformation",
                options=available_transforms,
                format_func=lambda value: SCORE_TRANSFORM_LABELS.get(value, value),
                index=available_transforms.index("raw") if "raw" in available_transforms else 0,
                key="targeted_score_transform",
                help=(
                    "Raw is the prespecified primary scale. asinh and RINT are exploratory "
                    "robustness analyses fitted within the nested-CV training partitions."
                ),
            )
        performance_catalog = performance_catalog.loc[
            performance_catalog["score_transform"].eq(score_transform)
        ].copy()
    source_registry = manifest.get("eigengene_source_registry", {})
    observed_sources = [
        value
        for value in performance_catalog.loc[
            performance_catalog["model_variant"].isin(transcriptomic_variants),
            "eigengene_source",
        ].dropna().astype(str).drop_duplicates().tolist()
        if value != "not_applicable"
    ]
    selectable_sources = [
        source
        for source in source_registry
        if bool(source_registry[source].get("selectable", False))
        and source in observed_sources
    ]
    selectable_sources.extend(
        source
        for source in observed_sources
        if source not in source_registry and source not in selectable_sources
    )
    if masked_selection or score_transform != "raw":
        eigengene_source = (
            "matched_multitissue"
            if "matched_multitissue" in observed_sources
            else observed_sources[0] if observed_sources else "not_applicable"
        )
    else:
        with st.sidebar:
            eigengene_source = st.selectbox(
                "Eigengene source",
                options=selectable_sources or ["matched_multitissue"],
                format_func=lambda value: EIGENGENE_SOURCE_LABELS.get(value, value),
                index=(
                    (selectable_sources or ["matched_multitissue"]).index(
                        "matched_multitissue"
                    )
                    if "matched_multitissue" in (selectable_sources or ["matched_multitissue"])
                    else 0
                ),
                key="targeted_eigengene_source",
                help=(
                    "This selection changes transcriptomic and joint models only. Dummy, "
                    "demographic, and network-only baselines remain source-neutral."
                ),
            )
    performance_catalog = performance_catalog.loc[
        ~performance_catalog["model_variant"].isin(transcriptomic_variants)
        | performance_catalog["eigengene_source"].eq(eigengene_source)
    ].copy()
    st.subheader(
        "Targeted differential-edge sensitivity"
        if masked_selection
        else "Fully nested targeted-module LIONESS prediction"
    )
    st.warning(
        "Repeated nested cross-validation is internal validation conditional on fixed module "
        "definitions. The earlier 70/30 cohort has already been inspected and is shown only "
        "as sensitivity evidence; it is not independent validation."
    )
    if not bool(manifest.get("complete", False)):
        st.warning(
            "This is an incremental transformation catalog. Only completed configurations are "
            "shown; the full Raw/asinh/RINT reconciliation is still interim."
        )
    if masked_selection:
        st.warning(
            "Exploratory differential-edge sensitivity: masks are learned from outer-training "
            "AD/Control donors in one five-fold outer cycle. The fold-specific all-edge panel "
            "and regularization settings are frozen; no new selection search is performed."
        )
        st.caption(
            "The differential-edge masked catalog is preserved unchanged and uses the existing "
            "asinh-based module scores."
        )
    elif score_transform != "raw":
        st.info(
            f"{SCORE_TRANSFORM_LABELS.get(score_transform, score_transform)} is an exploratory "
            "robustness analysis. The three prespecified primary hypotheses remain Raw-only."
        )
    with st.sidebar:
        st.header("Targeted prediction controls")
        tier_order = [
            value for value in ("primary", "secondary", "exploratory")
            if value in set(performance_catalog["evidence_tier"])
        ]
        evidence_tier = st.selectbox(
            "Evidence tier",
            options=tier_order,
            format_func=lambda value: TARGETED_TIER_LABELS.get(value, value),
            key="targeted_evidence_tier",
        )
        tier_catalog = performance_catalog.loc[
            performance_catalog["evidence_tier"].eq(evidence_tier)
        ]
        module_options = tier_catalog["module_definition"].drop_duplicates().tolist()
        module_definition = st.selectbox(
            "Targeted module definition",
            options=module_options,
            format_func=lambda value: MODULE_SET_LABELS.get(value, value),
            index=module_options.index("control_derived") if "control_derived" in module_options else 0,
            key="targeted_module_definition",
        )
        module_catalog = tier_catalog.loc[
            tier_catalog["module_definition"].eq(module_definition)
        ]
        methods = module_catalog["network_method"].drop_duplicates().tolist()
        network_method = st.selectbox(
            "Targeted LIONESS method",
            options=methods,
            format_func=readable_method,
            key="targeted_network_method",
        )
        method_catalog = module_catalog.loc[
            module_catalog["network_method"].eq(network_method)
        ]
        outcome_options = method_catalog["model_outcome"].drop_duplicates().tolist()
        outcome = st.selectbox(
            "Targeted prediction outcome",
            options=outcome_options,
            format_func=lambda value: PREDICTION_OUTCOME_LABELS.get(value, value),
            index=outcome_options.index("diagnosis_binary") if "diagnosis_binary" in outcome_options else 0,
            key="targeted_outcome",
        )
        outcome_catalog = method_catalog.loc[method_catalog["model_outcome"].eq(outcome)]
        panel_options = outcome_catalog["panel_strategy"].drop_duplicates().tolist()
        panel_strategy = st.selectbox(
            "Module panel",
            options=panel_options,
            format_func=lambda value: TARGETED_PANEL_LABELS.get(value, value),
            index=panel_options.index("tissue_neutral_ad") if "tissue_neutral_ad" in panel_options else 0,
            key="targeted_panel_strategy",
        )

    if masked_selection and outcome == "cogn_global":
        st.info(
            "Global cognition was not an outcome in the completed all-edge prediction "
            "catalog. Its diagnosis-derived module panel and K remain frozen; only "
            "elastic-net regularization was calibrated by inner CV within each "
            "outer-training fold on all-edge scores. Outer-test outcomes and masked scores "
            "were not used for that calibration."
        )
    if outcome == "clusters":
        st.warning(
            "Cluster prediction is exploratory. The reused tissue-neutral panel was selected "
            "for AD-versus-Control, not for cluster membership, and the four nominal clusters "
            "are strongly associated with diagnosis in this cohort. This analysis describes "
            "transfer of a diagnosis-derived panel; it is not an independent replication."
        )

    selected = performance_catalog.loc[
        performance_catalog["evidence_tier"].eq(evidence_tier)
        & performance_catalog["module_definition"].eq(module_definition)
        & performance_catalog["network_method"].eq(network_method)
        & performance_catalog["model_outcome"].eq(outcome)
        & performance_catalog["panel_strategy"].eq(panel_strategy)
        & performance_catalog["edge_mask"].eq(edge_mask)
        & performance_catalog["score_normalization"].eq(score_normalization)
        & performance_catalog["score_transform"].eq(score_transform)
    ].copy()
    if selected.empty:
        st.warning("No repeated-CV models match the selected targeted analysis.")
        return
    primary_metric = str(
        selected.loc[selected["metric"].notna(), "metric"].iloc[0]
        if "primary_metric" not in selected
        else selected["primary_metric"].dropna().iloc[0]
    )
    if "primary_metric" not in selected:
        preferred = {
            "diagnosis_binary": "roc_auc",
            "diagnosis_three_class": "macro_roc_auc",
            "cogdx": "mae",
            "parkinsonism": "roc_auc",
        }.get(outcome, "r2")
        if preferred in set(selected["metric"]):
            primary_metric = preferred
    repeats = (
        1 if masked_selection else int(manifest.get("selection", {}).get("outer_repeats", 5))
    )
    folds = int(manifest.get("selection", {}).get("outer_folds", 5))
    metrics = st.columns(6)
    metrics[0].metric(
        "Evaluation",
        f"{repeats} × {folds} outer CV sensitivity"
        if masked_selection
        else f"{repeats} × {folds} nested CV",
    )
    metrics[1].metric("OOF donors", int(selected["n_oof"].max()))
    metrics[2].metric("Primary metric", primary_metric)
    metrics[3].metric("Score scale", SCORE_TRANSFORM_LABELS.get(score_transform, score_transform))
    metrics[4].metric("Edge set", "All edges" if set(selected["edge_mask"]) == {"all"} else "Sensitivity mask")
    metrics[5].metric(
        "Eigengenes",
        EIGENGENE_SOURCE_LABELS.get(eigengene_source, eigengene_source),
    )

    (
        summary_tab, comparison_tab, transformation_tab, source_tab, panel_tab,
        diagnostic_tab, coefficient_tab, tables_tab, methods_tab,
    ) = st.tabs(
        [
            "Summary", "CT versus TS", "Transformation sensitivity", "Eigengene sources", "Panel selection",
            "OOF diagnostics", "Coefficients & KEGG", "Tables", "Methods",
        ]
    )
    with summary_tab:
        st.caption(
            "Performance is calculated from fold-specific panels and outer-test predictions. "
            "The consensus panel below is display-only and is never used to estimate performance."
        )
        if (
            not masked_selection
            and score_transform == "raw"
            and evidence_tier == "primary"
            and module_definition == "control_derived"
            and outcome == "diagnosis_binary"
        ):
            absolute = performance_catalog.loc[
                performance_catalog["evidence_tier"].eq("primary")
                & performance_catalog["score_transform"].eq("raw")
                & performance_catalog["edge_mask"].eq("all")
                & performance_catalog["score_normalization"].eq("standard_pruned")
                & performance_catalog["module_definition"].eq("control_derived")
                & performance_catalog["network_method"].eq("control_anchored")
                & performance_catalog["model_outcome"].eq("diagnosis_binary")
                & performance_catalog["predictor_block"].eq("CT_pooled")
                & performance_catalog["metric"].eq("roc_auc")
            ]
            absolute_specs = (
                (
                    "Unadjusted comparators",
                    (
                        ("Covariates", "tissue_neutral_ad", "covariates"),
                        ("Connectivity", "tissue_neutral_ad", "network_only"),
                        ("Transcriptomics", "tissue_neutral_ad", "transcriptomics_only"),
                        ("Connectivity + transcriptomics", "tissue_neutral_ad", "network_plus_transcriptomics"),
                    ),
                ),
                (
                    "Covariate-adjusted comparators",
                    (
                        ("Adjusted connectivity", "tissue_neutral_ad", "covariates_plus_network"),
                        ("Adjusted transcriptomics", "tissue_neutral_ad", "covariates_plus_transcriptomics"),
                        ("Fully adjusted joint", "tissue_neutral_ad", "covariates_plus_network_plus_transcriptomics"),
                        ("All-module adjusted connectivity", "all_modules", "covariates_plus_network"),
                    ),
                ),
            )
            for row_label, specifications in absolute_specs:
                st.caption(row_label)
                absolute_metrics = st.columns(len(specifications))
                for column, (label, strategy, variant) in zip(
                    absolute_metrics, specifications, strict=True
                ):
                    value = absolute.loc[
                        absolute["panel_strategy"].eq(strategy)
                        & absolute["model_variant"].eq(variant),
                        "value",
                    ]
                    column.metric(
                        f"{label} ROC-AUC",
                        f"{float(value.iloc[0]):.3f}" if len(value) else "NA",
                    )
        render_plotly_chart(
            prediction_performance_figure(
                selected,
                metric=primary_metric,
                block_labels=PREDICTION_BLOCK_LABELS,
                block_order=PREDICTION_BLOCK_ORDER,
                model_labels=PREDICTION_MODEL_LABELS,
                title=f"Nested-CV OOF performance: {PREDICTION_OUTCOME_LABELS.get(outcome, outcome)}",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        if (
            not masked_selection and score_transform == "raw"
            and evidence_tier == "primary" and outcome == "diagnosis_binary"
        ):
            comparisons = cached_targeted_prediction_table("primary_comparisons")
            if not comparisons.empty:
                render_plotly_chart(
                    targeted_primary_comparison_figure(
                        comparisons,
                        title="Three prespecified primary paired hypotheses",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                st.caption(
                    "Positive differences favor the first model. BH FDR is corrected across "
                    "exactly these three primary hypotheses."
                )
                filterable_dataframe(
                    comparisons,
                    table_key="targeted_primary_comparisons",
                    table_name="Three primary paired comparisons",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download primary comparisons (TSV)",
                    data=dataframe_to_tsv_bytes(comparisons),
                    file_name="targeted_lioness_primary_comparisons.tsv",
                    mime="text/tab-separated-values",
                )

    with comparison_tab:
        adjusted = selected.loc[
            selected["model_variant"].isin(
                [
                    "network_only",
                    "covariates_plus_network",
                    "transcriptomics_only",
                    "covariates_plus_transcriptomics",
                    "network_plus_transcriptomics",
                    "covariates_plus_network_plus_transcriptomics",
                ]
            )
        ].copy()
        if adjusted.empty:
            st.info("No CT/TS comparison is available for this selection.")
        else:
            render_plotly_chart(
                prediction_performance_figure(
                    adjusted,
                    metric=primary_metric,
                    block_labels=PREDICTION_BLOCK_LABELS,
                    block_order=PREDICTION_BLOCK_ORDER,
                    model_labels=PREDICTION_MODEL_LABELS,
                    title="CT, TS, and resolved-component OOF comparison",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            fold_performance = cached_targeted_prediction_table(
                fold_table,
                _targeted_filters(
                    evidence_tier=evidence_tier,
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    model_outcome=outcome,
                    model_variant="covariates_plus_network",
                    edge_mask=edge_mask,
                    score_normalization=score_normalization,
                    score_transform=score_transform,
                ),
            )
            if not fold_performance.empty:
                render_plotly_chart(
                    targeted_fold_robustness_figure(
                        fold_performance,
                        metric=primary_metric,
                        block_labels=PREDICTION_BLOCK_LABELS,
                        block_order=PREDICTION_BLOCK_ORDER,
                        title="Outer-fold robustness",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            if not masked_selection and evidence_tier in {"secondary", "exploratory"}:
                tier_comparisons = cached_targeted_prediction_table(
                    "tier_comparisons",
                    _targeted_filters(
                        evidence_tier=evidence_tier,
                        module_definition=module_definition,
                        network_method=network_method,
                        model_outcome=outcome,
                        score_transform=score_transform,
                    ),
                )
                if not tier_comparisons.empty:
                    st.caption(
                        "Secondary and exploratory paired comparisons use separate BH families; "
                        "exploratory rows retain both global-tier and within-outcome FDR."
                    )
                    filterable_dataframe(
                        tier_comparisons,
                        table_key="targeted_tier_comparisons",
                        table_name="Nested-CV paired performance comparisons",
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download paired comparisons (TSV)",
                        data=dataframe_to_tsv_bytes(tier_comparisons),
                        file_name="targeted_lioness_nested_cv_paired_comparisons.tsv",
                        mime="text/tab-separated-values",
                    )

    with transformation_tab:
        if masked_selection:
            st.info(
                "Transformation sensitivity applies to the all-edge targeted catalog. "
                "The differential-edge masked catalog remains the existing asinh analysis."
            )
        elif int(manifest.get("schema_version", 2)) < 3:
            st.info(
                "This deployed snapshot contains the corrected legacy asinh catalog only. "
                "Raw will appear here after its first validated incremental publication."
            )
        else:
            transform_status = cached_targeted_prediction_table("transformation_status")
            if not transform_status.empty:
                filterable_dataframe(
                    transform_status,
                    table_key="targeted_transform_status",
                    table_name="Transformation analysis status",
                    use_container_width=True,
                    hide_index=True,
                )
            transform_comparisons = cached_targeted_prediction_table(
                "transformation_comparisons",
                _targeted_filters(
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    model_outcome=outcome,
                ),
            )
            if transform_comparisons.empty:
                st.info(
                    "Paired transformed-versus-Raw comparisons will appear after a robustness "
                    "transformation completes for this configuration."
                )
            else:
                st.caption(
                    "These comparisons are exploratory. Positive oriented differences favor "
                    "asinh or RINT over Raw; global and within-outcome BH FDRs are both shown."
                )
                render_plotly_chart(
                    targeted_transform_comparison_figure(
                        transform_comparisons,
                        block_labels=PREDICTION_BLOCK_LABELS,
                        model_labels=PREDICTION_MODEL_LABELS,
                        title="Transformation sensitivity: paired donor-averaged OOF performance",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                render_plotly_chart(
                    targeted_transform_heatmap_figure(
                        transform_comparisons,
                        block_labels=PREDICTION_BLOCK_LABELS,
                        block_order=PREDICTION_BLOCK_ORDER,
                        model_labels=PREDICTION_MODEL_LABELS,
                        title="Oriented primary-metric difference versus Raw",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                filterable_dataframe(
                    transform_comparisons,
                    table_key="targeted_transform_comparisons",
                    table_name="Paired transformation comparisons",
                    use_container_width=True,
                    hide_index=True,
                )
            if panel_strategy != "all_modules":
                selection_outcome = str(
                    outcome_catalog.loc[
                        outcome_catalog["panel_strategy"].eq(panel_strategy),
                        "selection_outcome",
                    ].iloc[0]
                )
                overlap = cached_targeted_prediction_table(
                    "panel_transform_overlap",
                    _targeted_filters(
                        module_definition=module_definition,
                        network_method=network_method,
                        panel_strategy=panel_strategy,
                        selection_outcome=selection_outcome,
                    ),
                )
                if not overlap.empty:
                    render_plotly_chart(
                        targeted_panel_overlap_figure(
                            overlap,
                            title="Fold-specific panel overlap with Raw",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                    k_summary = overlap.groupby(
                        "transformed_scale", observed=True
                    ).agg(
                        folds=("outer_fold", "size"),
                        median_raw_k=("raw_k", "median"),
                        median_transformed_k=("transformed_k", "median"),
                        median_jaccard=("jaccard", "median"),
                    ).reset_index()
                    filterable_dataframe(
                        k_summary,
                        table_key="targeted_transform_k_summary",
                        table_name="K and panel-overlap summary",
                        use_container_width=True,
                        hide_index=True,
                    )
            if not transform_comparisons.empty:
                st.download_button(
                    "Download transformation sensitivity (TSV)",
                    data=dataframe_to_tsv_bytes(transform_comparisons),
                    file_name="targeted_lioness_transformation_sensitivity.tsv",
                    mime="text/tab-separated-values",
                )

    with source_tab:
        source_rows = []
        for source_id, values in source_registry.items():
            source_rows.append(
                {
                    "eigengene_source": source_id,
                    "label": values.get(
                        "label", EIGENGENE_SOURCE_LABELS.get(source_id, source_id)
                    ),
                    "status": values.get("status", "unavailable"),
                    "selectable": bool(values.get("selectable", False)),
                    "partition_level": values.get("partition_level"),
                    "tissue_scoped_modules": values.get("tissue_scoped_modules"),
                    "raw_configurations": values.get("raw_configurations", 0),
                }
            )
        if source_rows:
            filterable_dataframe(
                pd.DataFrame(source_rows),
                table_key="targeted_eigengene_source_registry",
                table_name="Eigengene-source availability",
                use_container_width=True,
                hide_index=True,
            )
        if masked_selection or score_transform != "raw":
            st.info(
                "Independent eigengene-source comparisons are available only for the "
                "all-edge Raw catalog. This selection retains the matched multi-tissue source."
            )
        elif "targeted_eigengene_source_comparisons.parquet" not in manifest.get(
            "files", {}
        ):
            st.info(
                "Source-comparison intervals will appear after the first independent "
                "regional eigengene source finishes and passes validation."
            )
        else:
            source_comparisons = cached_targeted_prediction_table(
                "eigengene_source_comparisons",
                _targeted_filters(
                    evidence_tier=evidence_tier,
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    model_outcome=outcome,
                    score_transform="raw",
                ),
            )
            source_comparisons = source_comparisons.loc[
                source_comparisons["source_a"].eq(eigengene_source)
                | source_comparisons["source_b"].eq(eigengene_source)
            ].copy()
            if source_comparisons.empty:
                st.info("No paired source comparison matches the selected analysis.")
            else:
                st.caption(
                    "Exploratory paired comparisons use identical donors and outer folds. "
                    "Positive differences favor Source A. Global and within-outcome BH FDRs "
                    "are retained; these comparisons do not change the prespecified primary tests."
                )
                render_plotly_chart(
                    targeted_eigengene_source_comparison_figure(
                        source_comparisons,
                        block_labels=PREDICTION_BLOCK_LABELS,
                        model_labels=PREDICTION_MODEL_LABELS,
                        title="Eigengene-source sensitivity: paired OOF performance",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                filterable_dataframe(
                    source_comparisons,
                    table_key="targeted_eigengene_source_comparisons",
                    table_name="Paired eigengene-source comparisons",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download eigengene-source comparisons (TSV)",
                    data=dataframe_to_tsv_bytes(source_comparisons),
                    file_name="targeted_lioness_eigengene_source_comparisons.tsv",
                    mime="text/tab-separated-values",
                )

    with panel_tab:
        if panel_strategy == "all_modules":
            st.info("All-module benchmark uses no data-derived targeted panel.")
        else:
            panel_filters = _targeted_filters(
                module_definition=module_definition,
                network_method=network_method,
                panel_strategy=panel_strategy,
                selection_outcome=str(outcome_catalog.loc[
                    outcome_catalog["panel_strategy"].eq(panel_strategy), "selection_outcome"
                ].iloc[0]),
                score_transform=score_transform,
            )
            consensus = cached_targeted_prediction_table("consensus_panels", panel_filters)
            selection = cached_targeted_prediction_table("panel_selection", panel_filters)
            if consensus.empty:
                st.info("Consensus selection results are unavailable for this panel.")
            else:
                render_plotly_chart(
                    targeted_selection_frequency_figure(
                        consensus,
                        title="Display-only consensus panel stability",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                st.caption(
                    "KEGG annotations were joined only after selection and are interpretive only."
                )
                filterable_dataframe(
                    consensus,
                    table_key="targeted_consensus_panel",
                    table_name="Targeted consensus panel",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download consensus panel (TSV)",
                    data=dataframe_to_tsv_bytes(consensus),
                    file_name="targeted_lioness_consensus_panel.tsv",
                    mime="text/tab-separated-values",
                )
            if not selection.empty:
                k_distribution = (
                    selection[["outer_repeat", "outer_fold", "selected_k"]]
                    .drop_duplicates()
                    .value_counts("selected_k")
                    .rename("outer_folds")
                    .reset_index()
                )
                filterable_dataframe(
                    k_distribution,
                    table_key="targeted_k_distribution",
                    table_name="One-standard-error K distribution",
                    use_container_width=True,
                    hide_index=True,
                )

    with diagnostic_tab:
        blocks = ordered_prediction_blocks(selected["predictor_block"])
        diagnostic_block = st.selectbox(
            "OOF diagnostic predictor block",
            options=blocks,
            format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
            key="targeted_diagnostic_block",
        )
        variants = selected.loc[
            selected["predictor_block"].eq(diagnostic_block), "model_variant"
        ].drop_duplicates().tolist()
        diagnostic_variant = st.selectbox(
            "OOF diagnostic model",
            options=variants,
            format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
            index=variants.index("covariates_plus_network") if "covariates_plus_network" in variants else 0,
            key="targeted_diagnostic_variant",
        )
        diagnostics = cached_targeted_prediction_table(
            diagnostic_table,
            _targeted_filters(
                evidence_tier=evidence_tier,
                module_definition=module_definition,
                network_method=network_method,
                panel_strategy=panel_strategy,
                model_outcome=outcome,
                predictor_block=diagnostic_block,
                model_variant=diagnostic_variant,
                edge_mask=edge_mask,
                score_normalization=score_normalization,
                score_transform=score_transform,
                eigengene_source=(
                    eigengene_source
                    if diagnostic_variant in transcriptomic_variants
                    else "not_applicable"
                ),
            ),
        )
        if diagnostics.empty:
            st.info("OOF donor diagnostics are unavailable for this model.")
        elif outcome in {"diagnosis_binary", "diagnosis_three_class", "parkinsonism", "clusters"}:
            confusion = (
                diagnostics.groupby(["target", "predicted"], observed=True)
                .size().rename("n").reset_index()
                .rename(columns={"target": "actual", "predicted": "predicted_class"})
            )
            render_plotly_chart(
                prediction_confusion_figure(confusion, title="Donor-averaged OOF confusion matrix"),
                use_container_width=True,
                config={"displaylogo": False},
            )
            diagnostic_curves = classification_diagnostic_rows(diagnostics)
            if not diagnostic_curves.empty:
                curve_columns = st.columns(3)
                for curve_column, curve_name in zip(
                    curve_columns,
                    ("ROC", "Precision-recall", "Calibration"),
                    strict=True,
                ):
                    curve_column.plotly_chart(
                        prediction_curve_figure(
                            diagnostic_curves,
                            curve=curve_name,
                            title=f"OOF {curve_name}",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
            if len([column for column in diagnostics if column.startswith("probability_")]) == 2:
                render_plotly_chart(
                    prediction_threshold_figure(diagnostics, title="Donor-averaged OOF threshold diagnostics"),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
        else:
            render_plotly_chart(
                prediction_observed_figure(
                    diagnostics,
                    title="Observed versus donor-averaged OOF prediction",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            render_plotly_chart(
                prediction_error_figure(diagnostics, title="OOF residual diagnostics"),
                use_container_width=True,
                config={"displaylogo": False},
            )
        if not diagnostics.empty:
            filterable_dataframe(
                diagnostics,
                table_key="targeted_oof_diagnostics",
                table_name="Sanitized donor-averaged OOF predictions",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected OOF predictions (TSV)",
                data=dataframe_to_tsv_bytes(diagnostics),
                file_name="targeted_lioness_oof_predictions.tsv",
                mime="text/tab-separated-values",
            )

    with coefficient_tab:
        coefficients = cached_targeted_prediction_table(
            coefficient_table,
            _targeted_filters(
                evidence_tier=evidence_tier,
                module_definition=module_definition,
                network_method=network_method,
                panel_strategy=panel_strategy,
                model_outcome=outcome,
                edge_mask=edge_mask,
                score_normalization=score_normalization,
                score_transform=score_transform,
            ),
        )
        coefficients = coefficients.loc[
            ~coefficients["model_variant"].isin(transcriptomic_variants)
            | coefficients["eigengene_source"].eq(eigengene_source)
        ].copy()
        if coefficients.empty:
            st.info("No nonzero coefficients are available for this model family.")
        else:
            coefficient_block = st.selectbox(
                "Targeted coefficient block",
                options=ordered_prediction_blocks(coefficients["predictor_block"]),
                format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
                key="targeted_coefficient_block",
            )
            coefficient_variants = coefficients.loc[
                coefficients["predictor_block"].eq(coefficient_block),
                "model_variant",
            ].drop_duplicates().tolist()
            coefficient_variant = st.selectbox(
                "Targeted coefficient model",
                options=coefficient_variants,
                format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
                index=(
                    coefficient_variants.index("covariates_plus_network")
                    if "covariates_plus_network" in coefficient_variants else 0
                ),
                key="targeted_coefficient_variant",
            )
            shown = coefficients.loc[
                coefficients["predictor_block"].eq(coefficient_block)
                & coefficients["model_variant"].eq(coefficient_variant)
            ].copy()
            render_plotly_chart(
                prediction_coefficient_figure(
                    shown,
                    title="Largest standardized outer-fold coefficients",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption("KEGG enrichment is interpretive only and never enters ranking or fitting.")
            filterable_dataframe(
                shown,
                table_key="targeted_coefficients",
                table_name="Fold coefficients with post-selection KEGG annotations",
                use_container_width=True,
                hide_index=True,
            )
            module_coefficients = shown.loc[shown["module"].notna()].copy()
            if not module_coefficients.empty:
                module_coefficients["component"] = (
                    module_coefficients["feature_name"].astype(str)
                    .str.split("__", n=1).str[1]
                    .str.replace("MFBA9BA46", "DLPFC", regex=False)
                )
                component_summary = module_coefficients.groupby(
                    ["coefficient_label", "component"], observed=True
                ).agg(
                    outer_fold_coefficients=("standardized_coefficient", "size"),
                    coefficient_sum=("standardized_coefficient", "sum"),
                    absolute_coefficient_sum=("abs_standardized_coefficient", "sum"),
                    median_absolute_coefficient=("abs_standardized_coefficient", "median"),
                ).reset_index()
                filterable_dataframe(
                    component_summary,
                    table_key="targeted_component_coefficients",
                    table_name="Component-level coefficient summary",
                    use_container_width=True,
                    hide_index=True,
                )
            st.download_button(
                "Download selected coefficients (TSV)",
                data=dataframe_to_tsv_bytes(shown),
                file_name="targeted_lioness_coefficients.tsv",
                mime="text/tab-separated-values",
            )

    with tables_tab:
        filterable_dataframe(
            selected,
            table_key="targeted_oof_performance",
            table_name="Donor-averaged OOF performance",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download selected OOF metrics (TSV)",
            data=dataframe_to_tsv_bytes(selected),
            file_name="targeted_lioness_nested_cv_metrics.tsv",
            mime="text/tab-separated-values",
        )

    with methods_tab:
        st.markdown(
            "Each outer-test fold is excluded from the LIONESS reference, panel ranking, "
            "redundancy pruning, K selection, preprocessing, and elastic-net tuning. Standard "
            "LIONESS uses outer-training donors; Control-referenced LIONESS uses outer-training "
            "Controls. Test donors are add-one scored against the frozen reference. Panel size "
            "and regularization are selected inside the outer-training partition."
        )
        st.markdown(
            "**Score transformations.** Raw uses `metric_raw` and is the only primary scale. "
            "asinh applies `arcsinh(metric_raw)`. RINT uses average ranks and is fitted anew "
            "inside every stability-selection subsample, inner-training fold, and outer-training "
            "fold; validation/test values are mapped through the training-derived monotone "
            "interpolation and clipped outside the training range. Panels and K may differ by "
            "transformation."
        )
        st.markdown(
            "**Transcriptomic comparator.** For every module, tissue-specific PCA1 "
            "eigengenes use training-only gene medians, scaling, loadings, and deterministic "
            "sign alignment. A single-tissue block uses that tissue; a tissue-pair block uses "
            "the two represented tissues; pooled CT/TS and resolved blocks concatenate AC, "
            "DLPFC, and PCG eigengenes without a second PCA. The matched source uses the same "
            "frozen LIONESS panel. Independent single-region sources instead use all level-3 "
            "regional modules from the represented tissues; joint models combine those "
            "eigengenes with the unchanged fold-selected LIONESS panel. Expression never "
            "contributes to LIONESS panel selection."
        )
        st.caption(
            "Regional partitions are fixed unsupervised structures discovered from broader "
            "expression cohorts. Eigengene preprocessing and PCA loadings are outer-training "
            "only, but partition discovery itself is not external validation."
        )
        st.caption(
            "The current fixed 70/30 held-out results are previously inspected sensitivity "
            "data and cannot sort, tune, or modify a targeted panel. Module discovery used the "
            "broader cohort, so this is not external validation."
        )
        st.json(manifest, expanded=False)


def render_prediction_view() -> None:
    """Render the leakage-reduced LIONESS held-out prediction catalog lazily."""

    targeted_available = targeted_prediction_data_available()
    if targeted_available:
        prediction_mode = st.radio(
            "Prediction mode",
            options=["targeted", "benchmark"],
            format_func=lambda value: TARGETED_PREDICTION_MODE_LABELS[value],
            horizontal=True,
            key="prediction_mode",
        )
        if prediction_mode == "targeted":
            render_targeted_prediction_view()
            return

    st.subheader("Leakage-reduced LIONESS prediction")
    if targeted_available:
        st.warning(
            "Previously inspected sensitivity data: this benchmark catalog remains useful for "
            "comparison, but it is not independent validation and cannot define or tune a "
            "targeted module panel."
        )
    st.caption(
        "The module definitions are fixed from the prior analysis, but reference networks, "
        "AD–Control edge masks, preprocessing, and elastic-net tuning use development data "
        "only. Held-out donors are add-one scored against frozen references. This is therefore "
        "leakage-reduced conditional on the module definitions, not a fully external validation."
    )
    if not prediction_data_available():
        st.info(
            "Prediction results have not yet been added to this deployed data bundle. "
            "The rest of the network explorer remains available."
        )
        return
    manifest = load_prediction_manifest()
    with st.sidebar:
        st.header("Prediction controls")
        reference_provenance = st.selectbox(
            "Prediction reference design",
            options=list(PREDICTION_REFERENCE_LABELS),
            format_func=lambda value: PREDICTION_REFERENCE_LABELS[value],
            index=0,
            key="prediction_reference_provenance",
        )
        module_definition = st.selectbox(
            "Prediction module definition",
            options=list(MODULE_SET_LABELS),
            format_func=lambda value: MODULE_SET_LABELS[value],
            key="prediction_module_definition",
        )
        methods = MODULE_SET_METHODS[module_definition]
        network_method = st.selectbox(
            "LIONESS method",
            options=methods,
            format_func=lambda value: (
                (
                    "Development-standard LIONESS (314-donor frozen reference)"
                    if value == "standard"
                    else "Development-Control LIONESS (114-Control frozen reference)"
                )
                if reference_provenance == "development_frozen"
                else readable_method(value)
            ),
            key="prediction_network_method",
        )
        predictor_design = st.radio(
            "Predictor design",
            options=list(PREDICTION_DESIGN_LABELS),
            format_func=lambda value: PREDICTION_DESIGN_LABELS[value],
            key="prediction_design",
        )
        edge_mask = st.selectbox(
            "Development AD–Control edge mask",
            options=list(PREDICTION_MASK_LABELS),
            format_func=lambda value: PREDICTION_MASK_LABELS[value],
            index=list(PREDICTION_MASK_LABELS).index("per_module_fdr10"),
            key="prediction_edge_mask",
        )
        if predictor_design == "module_connectivity":
            score_normalization = st.radio(
                "Connectivity score",
                options=["standard_pruned", "retained_edge"],
                format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
                key="prediction_normalization",
            )
        else:
            score_normalization = "not_applicable"
        outcome = st.selectbox(
            "Prediction target",
            options=[
                value for value in PREDICTION_OUTCOME_LABELS
                if value != "clusters"
            ],
            format_func=lambda value: PREDICTION_OUTCOME_LABELS[value],
            key="prediction_outcome",
        )

    performance = cached_prediction_performance(
        reference_provenance, module_definition, network_method, predictor_design, edge_mask,
        score_normalization, None,
    )
    if performance.empty:
        st.warning("No fitted models are available for these prediction controls.")
        return
    if reference_provenance == "existing_sensitivity":
        st.warning(
            "Exploratory sensitivity only: these scores use the previously established "
            "all-donor Standard reference or all-Control Control reference. Held-out donor "
            "expression therefore contributed to the unsupervised reference construction; "
            "do not interpret this option as the primary leakage-reduced evaluation."
        )
    selected_performance = performance.loc[performance["outcome"].eq(outcome)].copy()
    if selected_performance.empty:
        st.warning("The selected target is unavailable for this predictor configuration.")
        return
    primary_metric = str(selected_performance["primary_metric"].dropna().iloc[0])
    unavailable = selected_performance.loc[selected_performance["status"].ne("available")]
    if not unavailable.empty:
        st.warning(
            f"{len(unavailable)} model result(s) are unavailable because the selected "
            "edge mask left no variable development predictors or fitting was not estimable."
        )
    st.info(
        "CogDx and diagnosis are not independent replications: diagnosis is essentially a "
        "grouped version of CogDx in this cohort. Outcomes with missing values are evaluated "
        "only in donors with an observed target; outcomes are never imputed."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Development donors", int(selected_performance["n_development"].max()))
    summary_columns[1].metric("Held-out donors", int(selected_performance["n_held_out"].max()))
    summary_columns[2].metric("Primary metric", primary_metric)
    summary_columns[3].metric("Edge mask", PREDICTION_MASK_LABELS[edge_mask])

    overview_tab, comparison_tab, diagnostics_tab, coefficient_tab, tables_tab, methods_tab = st.tabs(
        ["Performance", "CT versus TS", "Diagnostics", "Coefficients", "Tables", "Methods"]
    )
    with overview_tab:
        figure = prediction_performance_figure(
            selected_performance,
            metric=primary_metric,
            block_labels=PREDICTION_BLOCK_LABELS,
            block_order=PREDICTION_BLOCK_ORDER,
            model_labels=PREDICTION_MODEL_LABELS,
            title=f"Held-out {PREDICTION_OUTCOME_LABELS[outcome]} performance",
        )
        render_plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
        st.caption(
            "Lower is better for CogDx MAE. Higher is better for all other primary "
            "metrics; held-out R² may be negative when a model performs worse than "
            "predicting the held-out outcome mean."
        )
        adjusted = performance.loc[
            performance["model_variant"].eq("covariates_plus_network")
            & performance["status"].eq("available")
            & performance["metric"].eq(performance["primary_metric"])
        ].copy()
        adjusted.loc[~adjusted["higher_is_better"].astype(bool), "value"] *= -1
        adjusted["metric"] = "primary_performance"
        if not adjusted.empty:
            render_plotly_chart(
                prediction_heatmap_figure(
                    adjusted,
                    metric="primary_performance",
                    block_labels=PREDICTION_BLOCK_LABELS,
                    block_order=PREDICTION_BLOCK_ORDER,
                    outcome_labels=PREDICTION_OUTCOME_LABELS,
                    title="Covariate-adjusted held-out primary performance by outcome and component block",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Cells use each outcome's prespecified primary metric. CogDx is displayed as "
                "negative MAE so that larger values consistently mean better performance; the "
                "numerical scales are still not directly interchangeable across columns."
            )
        if predictor_design == "edge_sums":
            with st.expander("Whole-network retained-edge sums", expanded=False):
                whole_network = cached_prediction_whole_network(
                    reference_provenance=reference_provenance,
                    module_definition=module_definition,
                    network_method=network_method,
                    edge_mask=edge_mask,
                )
                st.caption(
                    "Positive and negative magnitudes are the two independent model inputs. "
                    "Signed and absolute sums are retained here for interpretation and download. "
                    "These totals aggregate only retained within-module edges; no between-module "
                    "edges are introduced."
                )
                filterable_dataframe(
                    whole_network,
                    table_key="prediction_whole_network_features",
                    table_name="Whole-network edge-sum predictors",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download whole-network predictor values (TSV)",
                    data=dataframe_to_tsv_bytes(whole_network),
                    file_name="lioness_prediction_whole_network_edge_sums.tsv",
                    mime="text/tab-separated-values",
                )

    with comparison_tab:
        bootstrap = cached_prediction_bootstrap(
            reference_provenance, module_definition, network_method, predictor_design, edge_mask,
            score_normalization, None,
        )
        if bootstrap.empty:
            st.info("CT-versus-TS bootstrap comparisons are unavailable for this selection.")
        else:
            render_plotly_chart(
                prediction_ct_ts_figure(
                    bootstrap,
                    outcome_labels=PREDICTION_OUTCOME_LABELS,
                    title="Paired held-out CT-minus-TS performance with 95% bootstrap intervals",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Positive values favor CT. For CogDx, the sign is reversed after calculating "
                "MAE so that positive still means better CT performance. Global BH covers all "
                "displayed prediction comparisons; within-outcome BH is also retained."
            )
            filterable_dataframe(
                bootstrap,
                table_key="prediction_bootstrap",
                table_name="CT versus TS bootstrap comparisons",
                use_container_width=True,
                hide_index=True,
            )

    with diagnostics_tab:
        available_blocks = [
            block for block in PREDICTION_BLOCK_ORDER
            if block in set(selected_performance["predictor_block"])
        ]
        diagnostic_block = st.selectbox(
            "Diagnostic predictor block",
            options=available_blocks,
            format_func=lambda value: PREDICTION_BLOCK_LABELS[value],
        )
        diagnostic_variant = st.radio(
            "Diagnostic model",
            options=[
                value
                for value in (
                    "network_only",
                    "covariates_plus_network",
                    "transcriptomics_only",
                    "covariates_plus_transcriptomics",
                    "network_plus_transcriptomics",
                    "covariates_plus_network_plus_transcriptomics",
                )
                if value in set(selected_performance["model_variant"])
            ],
            format_func=lambda value: PREDICTION_MODEL_LABELS[value],
            horizontal=True,
        )
        filters = {
            "reference_provenance": reference_provenance,
            "module_definition": module_definition,
            "network_method": network_method,
            "predictor_design": predictor_design,
            "edge_mask": edge_mask,
            "score_normalization": score_normalization,
            "outcome": outcome,
            "predictor_block": diagnostic_block,
            "model_variant": diagnostic_variant,
        }
        diagnostics = cached_prediction_diagnostics(**filters)
        if outcome in {"diagnosis_binary", "diagnosis_three_class", "parkinsonism"}:
            curves = cached_prediction_curves(**filters)
            confusion = cached_prediction_confusion(**filters)
            if curves.empty:
                st.info("Classification curves are unavailable for this fitted model.")
            else:
                curve_columns = st.columns(3)
                for column, curve_name in zip(
                    curve_columns, ["ROC", "Precision-recall", "Calibration"], strict=True
                ):
                    column.plotly_chart(
                        prediction_curve_figure(
                            curves, curve=curve_name,
                            title=curve_name,
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
            if not confusion.empty:
                render_plotly_chart(
                    prediction_confusion_figure(
                        confusion, title="Held-out confusion matrix"
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            if outcome in {"diagnosis_binary", "parkinsonism"} and not diagnostics.empty:
                render_plotly_chart(
                    prediction_threshold_figure(
                        diagnostics, title="Held-out threshold diagnostics"
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
        elif diagnostics.empty:
            st.info("Held-out donor diagnostics are unavailable in the local cache.")
        else:
            render_plotly_chart(
                prediction_observed_figure(
                    diagnostics,
                    title=f"Observed versus held-out prediction: {PREDICTION_OUTCOME_LABELS[outcome]}",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            render_plotly_chart(
                prediction_error_figure(
                    diagnostics,
                    title="Held-out residual and error diagnostics",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        if not diagnostics.empty:
            filterable_dataframe(
                diagnostics,
                table_key="prediction_diagnostics",
                table_name="Held-out prediction diagnostics",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected held-out diagnostics (TSV)",
                data=dataframe_to_tsv_bytes(diagnostics),
                file_name="lioness_prediction_heldout_diagnostics.tsv",
                mime="text/tab-separated-values",
            )

    with coefficient_tab:
        coefficient_filters = {
            "reference_provenance": reference_provenance,
            "module_definition": module_definition,
            "network_method": network_method,
            "predictor_design": predictor_design,
            "edge_mask": edge_mask,
            "score_normalization": score_normalization,
            "outcome": outcome,
        }
        coefficients = cached_prediction_coefficients(**coefficient_filters)
        coefficients = coefficients.loc[
            coefficients["model_variant"].isin(
                [
                    "network_only",
                    "covariates_plus_network",
                    "transcriptomics_only",
                    "covariates_plus_transcriptomics",
                    "network_plus_transcriptomics",
                    "covariates_plus_network_plus_transcriptomics",
                ]
            )
        ]
        if coefficients.empty:
            st.info("No nonzero influential coefficients are available for this selection.")
        else:
            coefficient_block = st.selectbox(
                "Coefficient predictor block",
                options=ordered_prediction_blocks(coefficients["predictor_block"]),
                format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
            )
            coefficient_variant = st.radio(
                "Coefficient model",
                options=sorted(coefficients["model_variant"].unique()),
                format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
                horizontal=True,
                key="prediction_coefficient_variant",
            )
            shown = coefficients.loc[
                coefficients["predictor_block"].eq(coefficient_block)
                & coefficients["model_variant"].eq(coefficient_variant)
            ]
            render_plotly_chart(
                prediction_coefficient_figure(
                    shown,
                    title="Largest standardized module/component coefficients",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            filterable_dataframe(
                shown,
                table_key="prediction_coefficients",
                table_name="Prediction coefficients and KEGG annotations",
                use_container_width=True,
                hide_index=True,
            )
            module_coefficients = shown.loc[shown["module"].notna()].copy()
            if not module_coefficients.empty:
                module_coefficients["component"] = (
                    module_coefficients["feature_name"].astype(str)
                    .str.split("__", n=1).str[1]
                    .str.replace("MFBA9BA46", "DLPFC", regex=False)
                )
                component_summary = module_coefficients.groupby(
                    ["coefficient_label", "component"], observed=True
                ).agg(
                    modules_shown=("module", "nunique"),
                    coefficient_sum=("standardized_coefficient", "sum"),
                    absolute_coefficient_sum=("abs_standardized_coefficient", "sum"),
                    median_absolute_coefficient=("abs_standardized_coefficient", "median"),
                ).reset_index()
                filterable_dataframe(
                    component_summary,
                    table_key="prediction_component_coefficients",
                    table_name="Component-level coefficient summary",
                    use_container_width=True,
                    hide_index=True,
                )

    with tables_tab:
        filterable_dataframe(
            selected_performance,
            table_key="prediction_performance",
            table_name="Held-out model metrics",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download selected held-out metrics (TSV)",
            data=dataframe_to_tsv_bytes(selected_performance),
            file_name="lioness_prediction_heldout_metrics.tsv",
            mime="text/tab-separated-values",
        )

    with methods_tab:
        st.markdown(
            "Development-standard LIONESS uses all 314 development donors as its frozen "
            "reference. Development donors are leave-one-out scored and held-out donors are "
            "add-one scored. Development-Control LIONESS uses the 114 development Controls; "
            "all other donors are add-one scored without using their outcome label. Differential "
            "edge masks use only the 117 development AD and 114 development Controls. Models "
            "use elastic-net regularization with five-fold development-only cross-validation "
            "before one held-out evaluation. Logistic models use a 50/50 L1/L2 mix and tune "
            "three regularization strengths; continuous models tune both the mixing fraction "
            "and regularization strength."
        )
        st.markdown(
            "Module eigengenes are fitted on development expression only and applied to the "
            "held-out donors with frozen gene preprocessing and PCA loadings. Tissue and "
            "tissue-pair blocks use only their represented regions; pooled blocks concatenate "
            "AC, DLPFC, and PCG eigengenes. Transcriptomic-only and joint network–transcriptomic "
            "models are displayed beside the dummy, demographic/APOE, and network models."
        )
        st.caption(
            "For prediction only, 30 exact repeated full-cohort tissue–gene assignment "
            "rows were removed before scoring in both reference designs. Existing "
            "association analyses elsewhere in the app were not changed."
        )
        st.json(manifest, expanded=False)


try:
    require_data_files()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


@st.cache_resource(show_spinner=False)
def _bundle_cache_state() -> dict[str, str | None]:
    """Track the deployed bundle independently of Streamlit's data caches."""
    return {"manifest_sha256": None}


def _synchronize_bundle_cache() -> str:
    """Clear cached tables exactly once when a new deploy manifest appears."""
    manifest_path = ensure_data_path(DATA_DIR / "data_manifest.json")
    digest = hashlib.sha256(manifest_path.read_bytes())
    if prediction_data_available():
        try:
            prediction_manifest_path = ensure_data_path(DATA_DIR / "prediction/prediction_public_manifest.json")
            digest.update(prediction_manifest_path.read_bytes())
        except (FileNotFoundError, RuntimeError):
            pass
    if targeted_prediction_data_available():
        try:
            targeted_manifest_path = ensure_data_path(
                DATA_DIR / "prediction_targeted/targeted_prediction_public_manifest.json"
            )
            digest.update(targeted_manifest_path.read_bytes())
        except (FileNotFoundError, RuntimeError):
            pass
    manifest_sha256 = digest.hexdigest()
    state = _bundle_cache_state()
    if state["manifest_sha256"] != manifest_sha256:
        st.cache_data.clear()
        state["manifest_sha256"] = manifest_sha256
    return manifest_sha256


try:
    _synchronize_bundle_cache()
    data_manifest = cached_data_manifest()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Could not initialize the Google Drive data bundle: {error}")
    st.info(
        "Check the `[google_drive]` Streamlit Secrets entry, paste the complete "
        "service-account JSON without editing its `private_key`, save, and reboot the app."
    )
    st.stop()

st.title("ROSMAP Single-Sample Network Explorer")
st.markdown(
    "Explore donor-level LIONESS and BONOBO module features, including aggregate "
    "CT/TS and tissue-resolved components."
)

view_options = [
    "Associations",
    "Feature distributions",
    "Correlation heatmaps",
    "Prediction",
    "Module finder",
    "CT–TS screen",
    "Edge summaries",
    "Edge volcano",
    "MDC",
    "Statistics",
    "KEGG enrichment",
    "Module details",
    "Methods & data",
]
active_view = st.pills(
    "Analysis view",
    options=view_options,
    default="Associations",
    selection_mode="single",
    help=(
        "Only the selected view is calculated. This avoids loading unrelated large "
        "tables from Google Drive on every interaction."
    ),
)
if active_view is None:
    active_view = "Associations"
if active_view == "Prediction":
    render_prediction_view()
    st.stop()

with st.sidebar:
    st.header("Plot controls")
    st.caption(f"Data source: {data_source_label()}")
    module_set = st.selectbox(
        "Module definition",
        options=list(MODULE_SET_LABELS),
        format_func=lambda value: MODULE_SET_LABELS[value],
        index=0,
        help=(
            "Identically numbered modules in the two definitions have different gene "
            "memberships and are always loaded from separate data bundles."
        ),
    )
    module_set_label = MODULE_SET_LABELS[module_set]
    module_manifest = data_manifest.get("module_sets", {}).get(module_set, {})
    annotations = cached_annotations(module_set)
    modules = sorted(annotations["module"].astype(int).unique().tolist())
    module_count = int(module_manifest.get("modules", len(modules)))
    if module_count != len(modules):
        st.error(
            f"The {module_set_label} manifest declares {module_count} modules, but its "
            f"annotation table contains {len(modules)}."
        )
        st.stop()
    module_details = cached_module_details(module_set)
    estimator = st.radio(
        "Network estimator",
        options=list(ESTIMATOR_LABELS),
        format_func=lambda value: ESTIMATOR_LABELS[value],
        horizontal=True,
    )
    allowed_methods = (
        list(module_manifest.get("methods", MODULE_SET_METHODS[module_set]))
        if estimator == "lioness"
        else ["bonobo"]
    )
    method = st.radio(
        "Network method",
        options=allowed_methods,
        format_func=readable_method,
        index=allowed_methods.index("control_anchored")
        if "control_anchored" in allowed_methods
        else 0,
    )
    default_module = modules.index(935) if 935 in modules else 0
    module = st.selectbox(
        "Module",
        options=modules,
        index=default_module,
        format_func=module_label,
    )
    phenotype = st.selectbox(
        "Association outcome",
        options=(
            list(ASSOCIATION_OUTCOME_LABELS)
            if active_view == "Associations"
            else [*OUTCOME_LABELS, "clusters"]
        ),
        format_func=lambda value: ASSOCIATION_OUTCOME_LABELS[value],
    )
    if active_view == "Associations" and phenotype in SELECTABLE_ASSOCIATION_OUTCOMES:
        association_interpretation = st.radio(
            "Outcome interpretation",
            options=["numeric", "categorical"],
            format_func=lambda value: (
                "Numeric / ordinal correlation"
                if value == "numeric" else "Categorical comparison"
            ),
            horizontal=True,
            help=(
                "Numeric mode preserves ordered score information. Categorical mode treats "
                "the codes as labels and uses a Kruskal–Wallis omnibus comparison."
            ),
        )
    elif phenotype in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES:
        association_interpretation = "categorical"
    else:
        association_interpretation = "numeric"
    active_feature_labels = dict(
        FEATURE_LABELS if estimator == "lioness" else BONOBO_FEATURE_LABELS
    )
    feature = st.selectbox(
        "Module feature",
        options=list(active_feature_labels),
        format_func=lambda value: active_feature_labels[value],
    )
    if estimator == "bonobo":
        bonobo_edge_subset = st.radio(
            "BONOBO edge subset",
            options=["All edges", "Significant edges"],
            help=(
                "Significance tests whether a donor-specific BONOBO covariance edge "
                "differs from zero; it is not a diagnostic-group comparison."
            ),
        )
        if bonobo_edge_subset == "Significant edges":
            edge_rule = st.radio(
                "Significant-edge rule",
                options=list(BONOBO_EDGE_RULE_LABELS),
                format_func=lambda value: BONOBO_EDGE_RULE_LABELS[value],
            )
        else:
            edge_rule = "all"
    else:
        bonobo_edge_subset = "All edges"
        edge_rule = "all"
    differential_available = differential_data_available(module_set)
    differential_edge_rule = st.radio(
        "AD–Control edge subset",
        options=list(DIFFERENTIAL_EDGE_RULE_LABELS) if differential_available else ["all"],
        format_func=lambda value: DIFFERENTIAL_EDGE_RULE_LABELS[value],
        help=(
            "The differential mask is learned only in the diagnosis-and-sex-stratified "
            "discovery donors. The FDR scope below determines whether BH is corrected "
            "across the full analysis family or within the selected module."
        ),
    )
    if not differential_available:
        st.caption("AD–Control filtered scores are not present in this deployed bundle.")
        differential_fdr_scope = "global"
        differential_fdr_threshold = 0.05
    else:
        differential_fdr_scope = st.radio(
            "Differential-edge FDR scope",
            options=list(FDR_SCOPE_LABELS),
            format_func=lambda value: FDR_SCOPE_LABELS[value],
            index=0,
            help=(
                "Global BH adjusts all tested edges within the selected module definition, "
                "estimator, network method, and discovery/validation family. Per-module BH "
                "adjusts only across the edges of each module."
            ),
        )
        differential_fdr_threshold = st.radio(
            "Differential-edge FDR cutoff",
            options=list(FDR_THRESHOLD_LABELS),
            format_func=lambda value: FDR_THRESHOLD_LABELS[value],
            index=0,
            help=(
                "FDR < 0.10 is an exploratory, less restrictive mask. The selected "
                "cutoff is applied after the chosen global or per-module BH correction."
            ),
        )
    if differential_edge_rule == "all":
        score_normalization = "standard_pruned"
        analysis_subset = "all_donors"
    else:
        score_normalization = st.radio(
            "Score handling",
            options=list(SCORE_NORMALIZATION_LABELS),
            format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
            help=(
                "Standard pruning preserves the module-gene denominator. Retained-edge "
                "normalization divides edge sums by the number of retained edges."
            ),
        )
        analysis_subset = st.selectbox(
            "Evaluation cohort",
            options=list(ANALYSIS_SUBSET_LABELS),
            format_func=lambda value: ANALYSIS_SUBSET_LABELS[value],
            index=list(ANALYSIS_SUBSET_LABELS).index("validation_ad_control"),
        )
        if score_normalization == "retained_edge":
            active_feature_labels.update(
                {
                    "connectivity": "Mean retained signed edge weight",
                    "abs_sum": "Mean retained absolute edge weight",
                    "positive_abs_sum": "Positive contribution per retained edge",
                    "negative_abs_sum": "Negative contribution per retained edge",
                }
            )
    resolution = st.radio("Resolution", ["Aggregate CT / TS", "Tissue resolved"])
    scale = st.selectbox(
        "Feature scale",
        options=list(SCALE_LABELS),
        format_func=lambda value: SCALE_LABELS[value],
        index=0,
    )
    correlation_method = st.radio(
        "Association correlation",
        options=["Spearman", "Pearson"],
        index=0,
        help=(
            "Controls association scatter annotations, correlation heatmaps, the CT–TS "
            "screen, and the primary statistics table. Spearman is the default."
        ),
        disabled=association_interpretation == "categorical",
    )
    diagnoses = st.multiselect(
        "Diagnosis groups",
        options=DIAGNOSIS_ORDER,
        default=DIAGNOSIS_ORDER,
    )
    grouping_variable = "diagnosis_group"
    selected_group_levels: list[object] = list(diagnoses)
    minimum_group_n = 10
    annotation_fields: list[str] = ["n", "coefficient", "p", "fdr"]
    trend_line_rule = "all"
    association_significance_cutoff = 0.05
    show_pooled_association = False
    if active_view == "Associations":
        association_metadata = cached_sample_metadata()
        association_metadata = association_metadata.loc[
            association_metadata["diagnosis_group"].isin(diagnoses)
        ].copy()
        if differential_edge_rule != "all" and analysis_subset != "all_donors":
            selected_split = {
                "discovery_ad_control": "Discovery",
                "validation_ad_control": "Validation",
                "mci_external": "MCI_external",
            }[analysis_subset]
            association_metadata = association_metadata.loc[
                association_metadata["ad_control_split"].eq(selected_split)
            ]
        minimum_group_n = st.selectbox(
            "Minimum category size", options=[5, 10, 20], index=1,
            help=(
                "Smaller groups remain visible, but their statistics and trend lines are "
                "marked unavailable."
            ),
        )
        if association_interpretation == "numeric":
            grouping_options = [
                variable for variable in ASSOCIATION_GROUP_LABELS
                if variable != phenotype
            ]
            grouping_variable = st.selectbox(
                "Group correlations by",
                options=grouping_options,
                format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                index=grouping_options.index("diagnosis_group"),
            )
            available_group_levels = ordered_association_levels(
                association_metadata, grouping_variable
            )
            selected_group_levels = st.multiselect(
                "Group levels",
                options=available_group_levels,
                default=available_group_levels,
                format_func=lambda value: association_level_label(grouping_variable, value),
                disabled=grouping_variable == "__all__",
                help="The selection controls points, trends, annotations, legends, and downloads.",
            )
            if grouping_variable == "__all__":
                selected_group_levels = ["__all__"]
            show_pooled_association = (
                st.checkbox(
                    "Show pooled association across displayed donors",
                    value=True,
                    help=(
                        "The dashed pooled fit is recalculated after diagnosis, cohort, "
                        "component, grouping-level, and missing-value filters."
                    ),
                )
                if grouping_variable != "__all__" else False
            )
            annotation_fields = st.multiselect(
                "Plot annotation fields",
                options=["n", "coefficient", "p", "fdr"],
                default=["n", "coefficient", "p", "fdr"],
                format_func=lambda value: {
                    "n": "Sample size (n)", "coefficient": "Correlation coefficient",
                    "p": "Nominal p-value", "fdr": "Module-set FDR",
                }[value],
            )
            trend_line_rule = st.selectbox(
                "Trend-line rule",
                options=["all", "p", "fdr", "none"],
                format_func=lambda value: {
                    "all": "All eligible groups", "p": "Nominal p below cutoff",
                    "fdr": "Module-set FDR below cutoff", "none": "No trend lines",
                }[value],
            )
            association_significance_cutoff = st.radio(
                "Association significance cutoff",
                options=[0.05, 0.10], index=0, horizontal=True,
                format_func=lambda value: f"{value:.2f}" + (
                    " (exploratory)" if value == 0.10 else ""
                ),
                disabled=trend_line_rule not in {"p", "fdr"},
            )
        else:
            available_group_levels = ordered_association_levels(
                association_metadata, phenotype
            )
            selected_group_levels = st.multiselect(
                "Category levels",
                options=available_group_levels,
                default=available_group_levels,
                format_func=lambda value: association_level_label(phenotype, value),
                help="The selection controls displayed points and the omnibus test.",
            )
            annotation_fields = st.multiselect(
                "Plot annotation fields",
                options=["n", "h", "effect", "p", "fdr"],
                default=["n", "h", "effect", "p", "fdr"],
                format_func=lambda value: {
                    "n": "Sample size (n)", "h": "Kruskal–Wallis H",
                    "effect": "Epsilon-squared", "p": "Nominal p-value",
                    "fdr": "Module-set FDR",
                }[value],
            )
    color_by = st.selectbox(
        "Color points by",
        options=list(COLOR_LABELS),
        format_func=lambda value: COLOR_LABELS[value],
    )
    continuous_colorscale = st.selectbox(
        "Continuous color scale",
        options=list(CONTINUOUS_COLOR_SCALES),
        index=0,
        disabled=(
            color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
            or (
                active_view == "Associations"
                and association_interpretation == "numeric"
                and color_by == grouping_variable
            )
        ),
    )
    reverse_colorscale = st.checkbox(
        "Reverse continuous color scale",
        value=False,
        disabled=(
            color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
            or (
                active_view == "Associations"
                and association_interpretation == "numeric"
                and color_by == grouping_variable
            )
        ),
    )

differential_caption = (
    "not applied to module scores"
    if differential_edge_rule == "all"
    else (
        f"{FDR_SCOPE_LABELS[differential_fdr_scope]}, "
        f"{FDR_THRESHOLD_LABELS[differential_fdr_threshold]}"
    )
)
st.caption(
    f"Module definition: **{module_set_label}** · Estimator: "
    f"**{ESTIMATOR_LABELS[estimator]}** · Edge subset: "
    f"**{DIFFERENTIAL_EDGE_RULE_LABELS[differential_edge_rule]}** · Differential FDR: "
    f"**{differential_caption}**"
)
download_prefix = (
    f"{module_set}__{estimator}__{edge_rule}__{differential_edge_rule}__"
    f"{differential_fdr_scope}__fdr{differential_fdr_threshold:.2f}__"
    f"{score_normalization}__"
)

if not diagnoses:
    st.warning("Select at least one diagnosis group in the sidebar.")
    st.stop()

with st.spinner("Loading the selected module…"):
    if resolution == "Aggregate CT / TS":
        plot_data = cached_aggregate(
            module_set, estimator, method, module, feature, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
        )
        plot_data = aggregate_to_long(plot_data, scale)
        statistics = (
            cached_aggregate_stats(
                module_set, estimator, method, module, phenotype, feature, edge_rule,
                differential_edge_rule, differential_fdr_scope,
                differential_fdr_threshold, score_normalization, analysis_subset,
            )
            if active_view == "Statistics" and association_interpretation == "numeric"
            else pd.DataFrame()
        )
        resolved = False
    else:
        plot_data = cached_resolved(
            module_set, estimator, method, module, feature, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
        )
        plot_data = resolved_to_long(plot_data, scale)
        statistics = (
            cached_resolved_stats(
                module_set, estimator, method, module, phenotype, feature, edge_rule,
                differential_edge_rule, differential_fdr_scope,
                differential_fdr_threshold, score_normalization, analysis_subset,
            )
            if active_view == "Statistics" and association_interpretation == "numeric"
            else pd.DataFrame()
        )
        resolved = True

plot_data = attach_metadata(plot_data)
if differential_edge_rule != "all" and analysis_subset != "all_donors":
    selected_split = {
        "discovery_ad_control": "Discovery",
        "validation_ad_control": "Validation",
        "mci_external": "MCI_external",
    }[analysis_subset]
    plot_data = plot_data.loc[plot_data["ad_control_split"].eq(selected_split)].copy()
if differential_edge_rule != "all":
    st.warning(
        "The AD–Control edge mask was selected in the discovery donors. Use the held-out "
        "validation view for diagnostic-group evaluation. This validates edge selection, "
        "not a fully external network model: Control-referenced LIONESS still uses its "
        "established Control reference."
    )

available_components = (
    plot_data[["component", "component_label"]]
    .drop_duplicates()
    .assign(
        component_order=lambda frame: frame["component"].map(
            {component: index for index, component in enumerate(COMPONENT_ORDER)}
        )
    )
    .sort_values(["component_order", "component"])
)
component_labels = dict(
    available_components[["component", "component_label"]].itertuples(index=False, name=None)
)
if resolved:
    selected_components = st.sidebar.multiselect(
        "Tissue components",
        options=available_components["component"].tolist(),
        default=available_components["component"].tolist(),
        format_func=lambda value: component_labels[value],
    )
else:
    selected_components = available_components["component"].tolist()

plot_data = plot_data.loc[
    plot_data["diagnosis_group"].isin(diagnoses)
    & plot_data["component"].isin(selected_components)
].copy()
if active_view == "Associations":
    if association_interpretation == "numeric" and grouping_variable != "__all__":
        plot_data = plot_data.loc[
            plot_data[grouping_variable].notna()
            & plot_data[grouping_variable].isin(selected_group_levels)
        ].copy()
    elif association_interpretation == "categorical":
        plot_data = plot_data.loc[
            plot_data[phenotype].notna()
            & plot_data[phenotype].isin(selected_group_levels)
        ].copy()
if not statistics.empty:
    statistics = statistics.loc[
        statistics["diagnosis_group"].isin(diagnoses)
    ].copy()
    if resolved:
        statistics = statistics.loc[statistics["component"].isin(selected_components)]

if plot_data.empty or not selected_components:
    st.warning("No data remain for the current filters.")
    st.stop()

component_variation = plot_data.groupby("component", observed=True)["metric_value"].agg(
    nonmissing="count", distinct=lambda values: values.nunique(dropna=True)
)
structurally_unavailable_components = component_variation.loc[
    component_variation["nonmissing"].eq(0)
].index.tolist()
diagnosis_variation = plot_data.groupby(
    ["component", "diagnosis_group"], observed=True
)["metric_value"].agg(
    nonmissing="count", distinct=lambda values: values.nunique(dropna=True)
)
zero_variance_states = diagnosis_variation.loc[
    diagnosis_variation["nonmissing"].gt(0)
    & diagnosis_variation["distinct"].le(1)
].reset_index()
if estimator == "bonobo" and structurally_unavailable_components:
    unavailable_labels = [
        component_labels.get(value, value)
        for value in structurally_unavailable_components
    ]
    st.warning(
        "The selected module has no structurally possible edges in: "
        + ", ".join(unavailable_labels)
        + ". These scopes remain missing rather than being encoded as zero."
    )
if estimator == "bonobo" and not zero_variance_states.empty:
    zero_labels = [
        f"{component_labels.get(row.component, row.component)} ({row.diagnosis_group})"
        for row in zero_variance_states.itertuples(index=False)
    ]
    st.warning(
        "BONOBO has no donor-to-donor variation for the selected sparse-network score in: "
        + ", ".join(zero_labels)
        + ". Correlations are intentionally unavailable for constant states; for sparse "
        "features these are generally zero because no edge passed the selected rule."
    )

selected_details = module_details.loc[
    module_details["module"].astype(int).eq(int(module))
]
if len(selected_details) != 1:
    st.error(f"Expected exactly one module-details row for {module_label(module)}.")
    st.stop()
selected_details = selected_details.iloc[0]

annotation = selected_annotation(annotations, module)
module_kegg = cached_kegg(module_set, int(module))
association_subtitles = association_kegg_subtitles(
    module_kegg,
    selected_components,
    resolved=resolved,
    aggregate_annotation=annotation,
)
if annotation:
    st.markdown(f'<div class="kegg-note">{annotation}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="kegg-note">KEGG enrichment: unavailable</div>',
        unsafe_allow_html=True,
    )

summary_cols = st.columns(6)
summary_cols[0].metric("Estimator", ESTIMATOR_LABELS[estimator])
summary_cols[1].metric("Network method", readable_method(method))
summary_cols[2].metric("Module", module_label(module))
summary_cols[3].metric("Module genes", int(selected_details["module_size"]))
summary_cols[4].metric("Samples shown", plot_data["sample_id"].nunique())
summary_cols[5].metric("Components", plot_data["component"].nunique())

tissue_composition = pd.DataFrame(
    [
        {
            "Tissue": tissue,
            "Genes": int(selected_details[count_column]),
            "Module proportion": float(selected_details[proportion_column]),
        }
        for tissue, count_column, proportion_column in [
            ("AC", "n_genes_ac", "proportion_ac"),
            ("DLPFC", "n_genes_dlpfc", "proportion_dlpfc"),
            ("PCG", "n_genes_pcg", "proportion_pcg"),
        ]
    ]
)
tissue_composition["Possible TS edges"] = tissue_composition["Genes"].map(
    lambda count: count * (count - 1) // 2
)
tissue_composition["TS feature"] = tissue_composition["Genes"].map(
    lambda count: "Available" if count >= 2 else "Unavailable (<2 genes)"
)

with st.expander(
    f"{module_label(module)} module composition · {int(selected_details['module_size'])} genes · "
    f"{selected_details['tissues']}",
    expanded=True,
):
    detail_cols = st.columns(4)
    detail_cols[0].metric("Module size", f"{int(selected_details['module_size']):,} genes")
    detail_cols[1].metric("Tissues represented", f"{int(selected_details['n_tissues'])} of 3")
    detail_cols[2].metric("Module type", str(selected_details["cluster_type"]))
    detail_cols[3].metric("Dominant tissue", str(selected_details["dominant_tissue"]))
    filterable_dataframe(
        tissue_composition,
        table_key="module_composition",
        table_name="Module composition",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Genes": st.column_config.NumberColumn(format="%d"),
            "Module proportion": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
            "Possible TS edges": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Tissue proportions use total module genes as the denominator. A tissue needs at "
        "least two module genes for a within-tissue (TS) edge; one gene can still participate "
        "in cross-tissue (CT) edges."
    )

if active_view == "Associations":
    if association_interpretation == "numeric":
        module_set_associations = cached_grouped_module_set_associations(
            module_set, module_count, estimator, method, resolved, feature,
            phenotype, scale, tuple(selected_components), tuple(diagnoses),
            grouping_variable, tuple(selected_group_levels),
            show_pooled_association, minimum_group_n, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
    else:
        module_set_associations = cached_categorical_module_set_associations(
            module_set, module_count, estimator, method, resolved, feature,
            phenotype, scale, tuple(selected_components), tuple(diagnoses),
            tuple(selected_group_levels), minimum_group_n, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
    displayed_association_statistics = module_set_associations.loc[
        module_set_associations["module"].astype(int).eq(int(module))
    ].copy()
    pooled_label = (
        "All donors (pooled)"
        if set(diagnoses) == set(DIAGNOSIS_ORDER)
        and analysis_subset == "all_donors"
        else "All displayed donors (pooled)"
    )
    st.subheader("Phenotype association")
    if association_interpretation == "categorical":
        st.caption(
            (
                "Cluster labels are nominal categories, not numbers. "
                if phenotype == "clusters"
                else f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} is treated as a categorical outcome. "
            )
            +
            "Each panel shows its network-score distributions; category codes are never "
            "entered into Pearson or Spearman calculations. The Kruskal–Wallis omnibus "
            f"test includes levels with at least {minimum_group_n} displayed donors; smaller "
            "levels remain visible but are identified as excluded. Epsilon-squared is the effect "
            f"size. FDR is Benjamini–Hochberg adjusted only across the {module_count} "
            "modules in the selected definition for the fixed estimator, network method, "
            "feature, component, cohort, edge rule, and score normalization. Pearson, "
            "Spearman, and regression lines are intentionally not calculated."
        )
    else:
        st.caption(
            f"Correlations are grouped by {ASSOCIATION_GROUP_LABELS[grouping_variable]}; "
            f"annotations report {correlation_method}, while OLS lines remain visual guides. "
            "The selected levels control points, lines, annotations, legend groups, and "
            "downloads together. Diagnosis remains a donor-inclusion filter and marker shape. "
            f"Pearson and Spearman FDRs are BH-adjusted only across the {module_count} modules "
            "while holding the module definition, estimator, method, feature, component, "
            "outcome, cohort, edge rules, score handling, grouping variable, and grouping "
            "level fixed. Missing and constant tests are excluded from the BH denominator. "
            "The dashed pooled association, when enabled, is recalculated after every donor "
            "and level filter. A legend entry toggles its correlation group's points and line "
            "together. Aggregate and resolved KEGG subtitles remain component-specific."
        )
    grouping_labels = {
        str(value): association_level_label(
            phenotype if association_interpretation == "categorical" else grouping_variable,
            value,
        )
        for value in selected_group_levels
    }
    figure = (
        categorical_association_figure(
            plot_data,
            displayed_association_statistics,
            category_variable=phenotype,
            category_label=ASSOCIATION_OUTCOME_LABELS[phenotype],
            category_levels=selected_group_levels,
            category_labels=grouping_labels,
            feature_label=active_feature_labels[feature],
            scale_label=SCALE_LABELS[scale],
            module=module,
            minimum_group_n=minimum_group_n,
            annotation_fields=annotation_fields,
            module_definition=module_set_label,
            kegg_subtitles=association_subtitles,
            hover_fields=HOVER_LABELS,
        )
        if association_interpretation == "categorical"
        else grouped_association_figure(
            plot_data,
            displayed_association_statistics,
            phenotype=phenotype,
            phenotype_label=OUTCOME_LABELS[phenotype],
            feature_label=active_feature_labels[feature],
            scale_label=SCALE_LABELS[scale],
            grouping_variable=grouping_variable,
            grouping_levels=selected_group_levels,
            grouping_labels=grouping_labels,
            module=module,
            color_by=color_by,
            color_label=COLOR_LABELS[color_by],
            hover_fields=HOVER_LABELS,
            correlation_method=correlation_method.lower(),
            annotation_fields=annotation_fields,
            trend_line_rule=trend_line_rule,
            significance_cutoff=association_significance_cutoff,
            minimum_group_n=minimum_group_n,
            show_pooled=show_pooled_association,
            pooled_label=pooled_label,
            module_definition=module_set_label,
            continuous_colorscale=continuous_colorscale,
            reverse_colorscale=reverse_colorscale,
            categorical_color_fields=CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
            kegg_subtitles=association_subtitles,
        )
    )
    render_plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": (
                    f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                    f"{association_interpretation}_{correlation_method.lower()}"
                ),
                "scale": 3,
            },
        },
    )
    public_columns = list(dict.fromkeys([
        "sample_id",
        "diagnosis_group",
        "module",
        "metric_family",
        "component",
        "component_label",
        "metric_value",
        phenotype,
        "lioness_method",
        *[column for column in HOVER_LABELS if column != phenotype],
    ]))
    public_plot_data = plot_data[public_columns].rename(
        columns={
            "metric_value": f"feature_{scale}",
            "lioness_method": "network_method",
        }
    )
    public_plot_data.insert(0, "module_definition", module_set_label)
    st.download_button(
        "Download displayed plot data (TSV)",
        data=dataframe_to_tsv_bytes(public_plot_data),
        file_name=(
            f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
            f"{resolution.replace(' ', '_')}.tsv"
        ),
        mime="text/tab-separated-values",
    )
    association_table = displayed_association_statistics.copy()
    if association_interpretation == "numeric":
        association_table["grouping_label"] = association_table["grouping_label"].replace(
            {"All displayed donors (pooled)": pooled_label}
        )
        if "diagnosis_group" in association_table:
            association_table["diagnosis_group"] = association_table[
                "diagnosis_group"
            ].replace({"All donors": pooled_label})
        else:
            association_table["diagnosis_group"] = " / ".join(diagnoses)
    association_table.insert(0, "module_definition", module_set_label)
    association_columns = (
        [
            "module_definition", "module", "metric_family", "component_label",
            "category_variable", "outcome", "n", "n_tested", "k_tested",
            "level_counts", "level_medians",
            "levels_tested", "levels_excluded_small_n", "kruskal_h",
            "kruskal_df", "epsilon_squared", "categorical_p",
            "categorical_fdr_across_modules", "categorical_fdr_module_family_n",
        ]
        if association_interpretation == "categorical"
        else [
            "module_definition", "module", "metric_family", "component_label",
            "diagnosis_group", "grouping_variable", "grouping_level",
            "grouping_label", "is_pooled",
            "outcome", "n", "minimum_group_n", "eligible", "unavailable_reason",
            "pearson_r", "pearson_p",
            "pearson_fdr_across_modules", "pearson_fdr_module_family_n",
            "spearman_rho", "spearman_p", "spearman_fdr_across_modules",
            "spearman_fdr_module_family_n",
        ]
    )
    with st.expander(
        "Kruskal–Wallis statistics · BH across modules"
        if association_interpretation == "categorical"
        else "Pearson and Spearman statistics · BH across modules",
        expanded=False,
    ):
        filterable_dataframe(
            association_table[association_columns],
            table_key="association_module_set_fdr",
            table_name="Association statistics",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download displayed association statistics (TSV)",
            data=dataframe_to_tsv_bytes(association_table[association_columns]),
            file_name=(
                f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                "association_module_set_fdr.tsv"
            ),
            mime="text/tab-separated-values",
        )

if active_view == "Correlation heatmaps":
    heatmap_statistic_type = st.radio(
        "Association type",
        ["Numeric correlations", "Nominal ROSMAP clusters"],
        horizontal=True,
        key="heatmap_statistic_type",
    )
    if heatmap_statistic_type == "Nominal ROSMAP clusters":
        st.subheader("Network-score associations with nominal ROSMAP clusters")
        st.caption(
            "This is a separate nonnegative epsilon-squared heatmap. Asterisks mark "
            "Kruskal–Wallis FDR < 0.05 after BH across modules only. Cluster codes are "
            "never passed to Pearson or Spearman correlation."
        )
        cluster_heatmap_scope = st.radio(
            "Cluster heatmap scope",
            ["Selected module", f"All {module_count} modules"],
            horizontal=True,
        )
        cluster_heatmap_diagnosis = st.selectbox(
            "Diagnosis/cohort in cluster heatmap",
            ["All donors", *DIAGNOSIS_ORDER, "All diagnosis groups"],
            index=0,
        )
        cluster_features = st.multiselect(
            "Network features",
            options=list(active_feature_labels),
            default=list(active_feature_labels),
            format_func=lambda value: active_feature_labels[value],
        )
        cluster_components = tuple(selected_components)
        if cluster_heatmap_diagnosis == "All donors":
            requested_diagnoses = tuple(DIAGNOSIS_ORDER)
            include_cluster_pooled = True
            display_groups = ["All donors"]
        elif cluster_heatmap_diagnosis == "All diagnosis groups":
            requested_diagnoses = tuple(DIAGNOSIS_ORDER)
            include_cluster_pooled = True
            display_groups = ["All donors", *DIAGNOSIS_ORDER]
        else:
            requested_diagnoses = (cluster_heatmap_diagnosis,)
            include_cluster_pooled = False
            display_groups = [cluster_heatmap_diagnosis]
        cluster_frames = [
            cached_module_set_cluster_associations(
                module_set, module_count, estimator, method, resolved,
                selected_feature, cluster_components, requested_diagnoses,
                include_cluster_pooled, edge_rule, differential_edge_rule,
                differential_fdr_scope, differential_fdr_threshold,
                score_normalization, analysis_subset,
            )
            for selected_feature in cluster_features
        ]
        cluster_table = (
            pd.concat(cluster_frames, ignore_index=True)
            if cluster_frames else pd.DataFrame()
        )
        if not cluster_table.empty:
            cluster_table = cluster_table.loc[
                cluster_table["diagnosis_group"].isin(display_groups)
            ].copy()
            if cluster_heatmap_scope == "Selected module":
                cluster_table = cluster_table.loc[
                    cluster_table["module"].astype(int).eq(int(module))
                ]
            cluster_table["feature_label"] = cluster_table["metric_family"].map(
                active_feature_labels
            )
            cluster_table["base_component_label"] = cluster_table["component_label"]
            if len(display_groups) > 1:
                cluster_table["component_label"] = (
                    cluster_table["diagnosis_group"].astype(str)
                    + " · " + cluster_table["component_label"].astype(str)
                )
            cluster_table["heatmap_row"] = (
                "M" + cluster_table["module"].astype(int).astype(str)
                + " · " + cluster_table["feature_label"].astype(str)
            )
        show_cluster_significant = st.checkbox(
            "Show only rows with at least one cluster FDR < 0.05",
            value=False,
        )
        if show_cluster_significant and not cluster_table.empty:
            keep_rows = cluster_table.loc[
                cluster_table["categorical_fdr_across_modules"].lt(0.05),
                "heatmap_row",
            ].unique()
            cluster_table = cluster_table.loc[
                cluster_table["heatmap_row"].isin(keep_rows)
            ]
        cluster_ordering = st.selectbox(
            "Cluster heatmap clustering",
            ["None", "Rows", "Columns", "Rows and columns"],
        )
        if cluster_table.empty:
            st.info("No nominal cluster associations meet the selected filters.")
        else:
            render_plotly_chart(
                cluster_association_heatmap_figure(
                    cluster_table,
                    title="ROSMAP cluster association effect sizes",
                    cluster_rows=cluster_ordering in {"Rows", "Rows and columns"},
                    cluster_columns=cluster_ordering in {"Columns", "Rows and columns"},
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            filterable_dataframe(
                cluster_table,
                table_key="complete_cluster_association_table",
                table_name="Complete nominal cluster association table",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download nominal cluster association table (TSV)",
                data=dataframe_to_tsv_bytes(cluster_table),
                file_name=f"{download_prefix}cluster_association_statistics.tsv",
                mime="text/tab-separated-values",
            )
        st.stop()
    st.subheader("Network-score correlations across phenotypes and outcomes")
    st.caption(
        "Heatmaps use donor-level Z-scored module features. Nominal fields such as sex code "
        "and APOE genotype remain available in hover but are excluded from Pearson/Spearman "
        "heatmaps because numeric correlation is not appropriate for unordered categories. "
        "CogDx, Braak, CERAD, ADNC, and Parkinsonism are source-coded ordinal/binary outcomes; "
        "Spearman is generally the more appropriate descriptive coefficient for the ordinal fields."
    )
    heatmap_mode = st.radio(
        "Heatmap scope",
        [
            "Selected module: all feature scores",
            f"All {module_count} modules: selected or all feature scores",
        ],
        horizontal=True,
    )
    st.caption(f"Correlation method: **{correlation_method}** (controlled in the sidebar).")
    heatmap_diagnosis = st.selectbox(
        "Diagnosis in heatmap",
        options=["All donors", *DIAGNOSIS_ORDER, "All diagnosis groups"],
        index=3,
        key="heatmap_diagnosis",
        help=(
            "All diagnosis groups displays pooled donors, Control, MCI, and AD together; "
            "the group is included in each heatmap row and in the complete table."
        ),
    )
    clustering_options = ["None", "Rows", "Columns", "Rows and columns"]
    if heatmap_mode.startswith("All"):
        clustering_options.extend(["Modules", "Modules and columns"])
    heatmap_clustering = st.selectbox(
        "Heatmap clustering",
        options=clustering_options,
        help=(
            "Reorders the selected axes using average-linkage hierarchical clustering "
            "of Euclidean distances between correlation profiles. Module clustering "
            "clusters whole module profiles, keeps all score rows from a module together, "
            "and outlines each module block."
        ),
    )
    cluster_rows = heatmap_clustering in {"Rows", "Rows and columns"}
    cluster_columns = heatmap_clustering in {
        "Columns", "Rows and columns", "Modules and columns"
    }
    cluster_modules = heatmap_clustering in {"Modules", "Modules and columns"}
    if correlation_method == "Pearson":
        value_column = "pearson_r"
        p_column = "pearson_p"
    else:
        value_column = "spearman_rho"
        p_column = "spearman_p"

    row_group_labels: dict[str, str] | None = None

    if heatmap_mode.startswith("Selected module"):
        correlation_table = cached_module_correlations(
            module_set, estimator, method, module, resolved, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        if heatmap_diagnosis == "All diagnosis groups":
            heatmap_data = correlation_table.copy()
            heatmap_data["heatmap_row"] = (
                heatmap_data["heatmap_row"]
                + " · "
                + heatmap_data["diagnosis_group"].astype(str)
            )
        else:
            heatmap_data = correlation_table.loc[
                correlation_table["diagnosis_group"].eq(heatmap_diagnosis)
            ].copy()
        fdr_column = (
            "pearson_fdr_displayed_family"
            if correlation_method == "Pearson"
            else "spearman_fdr_displayed_family"
        )
        component_rank = {value: index for index, value in enumerate(COMPONENT_ORDER)}
        feature_rank = {value: index for index, value in enumerate(FEATURE_LABELS)}
        diagnosis_rank = {
            value: index
            for index, value in enumerate(["All donors", *DIAGNOSIS_ORDER])
        }
        row_order_frame = (
            heatmap_data[
                ["metric_family", "component", "diagnosis_group", "heatmap_row"]
            ]
            .drop_duplicates()
            .assign(
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
                diagnosis_rank=lambda frame: frame["diagnosis_group"].map(diagnosis_rank),
            )
            .sort_values(
                ["feature_rank", "component_rank", "component", "diagnosis_rank"]
            )
        )
        row_order = row_order_frame["heatmap_row"].tolist()
        heatmap_title = (
            f"{module_set_label} · Module M{module}: all {ESTIMATOR_LABELS[estimator]} "
            "feature scores vs "
            f"outcomes · {heatmap_diagnosis}"
        )
        fdr_scope = (
            "Displayed-family FDR is Benjamini–Hochberg correction across every feature, "
            "component, outcome, and diagnosis correlation calculated for this module."
        )
    else:
        all_feature = st.selectbox(
            "Features for all-module heatmap and table",
            options=["__all__", *list(active_feature_labels)],
            format_func=lambda value: (
                f"All {ESTIMATOR_LABELS[estimator]} features"
                if value == "__all__"
                else active_feature_labels[value]
            ),
            index=1 + list(active_feature_labels).index(feature),
        )
        if resolved:
            component_options = available_components["component"].tolist()
            all_component = st.selectbox(
                "Components for all-module heatmap and table",
                options=["__all__", *component_options],
                format_func=lambda value: (
                    f"All {len(component_options)} tissue-resolved components"
                    if value == "__all__"
                    else component_labels[value]
                ),
                index=1,
                help=(
                    "All components combines AC, DLPFC, PCG, AC–DLPFC, AC–PCG, "
                    "and DLPFC–PCG in the same heatmap and complete table."
                ),
            )
        else:
            all_component = st.selectbox(
                "Components for all-module heatmap and table",
                options=["__all__", "CT", "TS"],
                format_func=lambda value: (
                    "Both aggregate components (CT and TS)"
                    if value == "__all__"
                    else f"{value} aggregate"
                ),
                index=1,
                help=(
                    "Both aggregate components displays CT and TS rows together in "
                    "the heatmap and complete table."
                ),
            )
        correlation_table = cached_all_module_correlations(
            module_set,
            estimator,
            method,
            resolved,
            all_feature,
            all_component,
            heatmap_diagnosis,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
        )
        correlation_table["feature_label"] = correlation_table["metric_family"].map(
            active_feature_labels
        )
        heatmap_data = correlation_table.copy()
        include_feature_in_row = all_feature == "__all__"
        include_component_in_row = all_component == "__all__"
        include_group_in_row = heatmap_diagnosis == "All diagnosis groups"
        heatmap_data["heatmap_row"] = heatmap_data["module"].map(
            lambda value: f"M{int(value)}"
        )
        if include_feature_in_row:
            heatmap_data["heatmap_row"] += (
                " · " + heatmap_data["feature_label"].astype(str)
            )
        if include_component_in_row:
            heatmap_data["heatmap_row"] += " · " + heatmap_data[
                "component_label"
            ].astype(str)
        if include_group_in_row:
            heatmap_data["heatmap_row"] += (
                " · " + heatmap_data["diagnosis_group"].astype(str)
            )

        if cluster_modules:
            module_order = clustered_correlation_group_order(
                heatmap_data,
                value_column=value_column,
                group_column="module",
                subgroup_columns=("metric_family", "component", "diagnosis_group"),
            )
        else:
            module_order = sorted(heatmap_data["module"].astype(int).unique())
        module_rank = {int(value): index for index, value in enumerate(module_order)}
        feature_rank = {
            value: index for index, value in enumerate(active_feature_labels)
        }
        diagnosis_rank = {
            value: index
            for index, value in enumerate(["All donors", *DIAGNOSIS_ORDER])
        }
        component_order = COMPONENT_ORDER if resolved else ["CT", "TS"]
        component_rank = {
            value: index for index, value in enumerate(component_order)
        }
        row_order_frame = (
            heatmap_data[
                [
                    "module",
                    "metric_family",
                    "component",
                    "diagnosis_group",
                    "heatmap_row",
                ]
            ]
            .drop_duplicates()
            .assign(
                module_rank=lambda frame: frame["module"].map(module_rank),
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
                diagnosis_rank=lambda frame: frame["diagnosis_group"].map(diagnosis_rank),
            )
            .sort_values(
                ["module_rank", "feature_rank", "component_rank", "diagnosis_rank"]
            )
        )
        row_order = row_order_frame["heatmap_row"].tolist()
        if cluster_modules:
            row_group_labels = dict(
                row_order_frame[["heatmap_row", "module"]]
                .assign(module=lambda frame: frame["module"].map(lambda value: f"M{int(value)}"))
                .itertuples(index=False, name=None)
            )
        feature_title = (
            f"all {len(active_feature_labels)} {ESTIMATOR_LABELS[estimator]} features"
            if all_feature == "__all__"
            else active_feature_labels[all_feature]
        )
        if all_component == "__all__":
            component_title = (
                f"all {len(component_options)} tissue-resolved components"
                if resolved
                else "both aggregate components (CT and TS)"
            )
        else:
            component_title = (
                component_labels[all_component]
                if resolved
                else f"{all_component} aggregate"
            )
        heatmap_title = (
            f"{module_set_label}: {feature_title} · "
            f"{component_title} · {heatmap_diagnosis}"
        )
        fdr_column = (
            "pearson_fdr_across_modules"
            if correlation_method == "Pearson"
            else "spearman_fdr_across_modules"
        )
        fdr_scope = (
            f"FDR is Benjamini–Hochberg correction across the {module_count} modules "
            "only, separately for each fixed feature, component, outcome, diagnosis/cohort, "
            "and Pearson/Spearman method. Missing or constant tests are excluded."
        )

        heatmap_row_limit = st.selectbox(
            "Rows displayed in heatmap",
            options=["All rows", "Top 50", "Top 100", "Top 250"],
            help=(
                "Top-row views rank module/feature/component/group rows by their strongest "
                "absolute correlation across the displayed outcomes. The complete table "
                "remains exhaustive."
            ),
        )
        if heatmap_row_limit != "All rows":
            row_count = int(heatmap_row_limit.split()[-1])
            strongest_rows = (
                heatmap_data.assign(
                    _absolute_correlation=pd.to_numeric(
                        heatmap_data[value_column], errors="coerce"
                    ).abs()
                )
                .groupby("heatmap_row", observed=True)["_absolute_correlation"]
                .max()
                .nlargest(row_count)
                .index
            )
            strongest_set = set(strongest_rows)
            heatmap_data = heatmap_data.loc[
                heatmap_data["heatmap_row"].isin(strongest_set)
            ].copy()
            row_order = [row for row in row_order if row in strongest_set]
            if row_group_labels:
                row_group_labels = {
                    row: group
                    for row, group in row_group_labels.items()
                    if row in strongest_set
                }

    significance_view = st.radio(
        "Correlation rows",
        options=["All rows", "At least one FDR < 0.05"],
        horizontal=True,
        help=(
            "The significant-only heatmap retains a module/feature/group row when at least "
            "one displayed outcome has across-module FDR < 0.05. The table then contains "
            "only its significant correlation cells."
        ),
    )
    table_data = correlation_table.copy()
    if significance_view == "At least one FDR < 0.05":
        significant_rows = set(
            heatmap_data.loc[
                pd.to_numeric(heatmap_data[fdr_column], errors="coerce").lt(0.05),
                "heatmap_row",
            ]
        )
        heatmap_data = heatmap_data.loc[
            heatmap_data["heatmap_row"].isin(significant_rows)
        ].copy()
        row_order = [row for row in row_order if row in significant_rows]
        table_data = table_data.loc[
            pd.to_numeric(table_data[fdr_column], errors="coerce").lt(0.05)
        ].copy()
        if row_group_labels:
            row_group_labels = {
                row: group
                for row, group in row_group_labels.items()
                if row in significant_rows
            }

    correlation_table = correlation_table.copy()
    correlation_table.insert(0, "module_definition", module_set_label)
    table_data = table_data.copy()
    table_data.insert(0, "module_definition", module_set_label)
    table_data["correlation_method"] = correlation_method
    table_data["correlation"] = pd.to_numeric(
        table_data[value_column], errors="coerce"
    )
    table_data["absolute_correlation"] = table_data["correlation"].abs()
    table_data["p_value"] = pd.to_numeric(table_data[p_column], errors="coerce")
    table_data["fdr"] = pd.to_numeric(table_data[fdr_column], errors="coerce")
    table_data = table_data.sort_values(
        ["absolute_correlation", "module", "metric_family", "component"],
        ascending=[False, True, True, True],
        na_position="last",
    )

    if heatmap_data.empty or not row_order:
        st.info("No correlations meet the selected heatmap filters.")
    else:
        if cluster_rows and len(row_order) > 1000:
            st.warning(
                "Row clustering is disabled above 1,000 rows to avoid a large pairwise-distance "
                "matrix. Choose a Top-row view or use Modules clustering."
            )
            cluster_rows = False
        correlation_heatmap = correlation_heatmap_figure(
            heatmap_data,
            value_column=value_column,
            p_column=p_column,
            fdr_column=fdr_column,
            title=heatmap_title,
            row_order=row_order,
            cluster_rows=cluster_rows,
            cluster_columns=cluster_columns,
            row_group_labels=row_group_labels,
            significance_threshold=0.05,
        )
        render_plotly_chart(
            correlation_heatmap,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"{download_prefix}{method}_{heatmap_diagnosis}_"
                        f"{correlation_method.lower()}_"
                        "correlation_heatmap"
                    ),
                    "scale": 3,
                },
            },
        )
    if heatmap_clustering != "None":
        st.caption(
            "Display order uses average-linkage hierarchical clustering with Euclidean "
            "distance. Missing coefficients are treated as zero only when determining "
            "the order; displayed correlations, p-values, FDRs, and downloads are unchanged."
        )
    st.caption(fdr_scope)
    st.caption("* indicates selected-method FDR < 0.05 for that heatmap cell.")
    correlation_columns = [
        "module_definition",
        "module",
        "feature_label",
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "outcome",
        "outcome_label",
        "n",
        "correlation_method",
        "correlation",
        "absolute_correlation",
        "p_value",
        "fdr",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
    ]
    correlation_columns = [
        column for column in correlation_columns if column in table_data.columns
    ]
    with st.expander("Complete correlation table", expanded=False):
        filterable_dataframe(
            table_data[correlation_columns],
            table_key="complete_correlation_table",
            table_name="Complete correlation table",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download complete correlation table (TSV)",
            data=dataframe_to_tsv_bytes(table_data[correlation_columns]),
            file_name=(
                f"{download_prefix}{method}_{heatmap_diagnosis}_"
                "network_outcome_correlations.tsv"
            ),
            mime="text/tab-separated-values",
        )
if active_view == "Feature distributions":
    st.subheader("Donor-level feature distributions")
    st.caption(
        "These views use only the selected module feature; no phenotype is on an axis. "
        "Histogram heights are probability densities so diagnosis groups with different sample "
        "sizes remain comparable."
    )
    chart_col, bin_col = st.columns([1, 2])
    chart_type = chart_col.radio("Distribution view", ["Histogram", "Violin"], horizontal=True)
    bins = bin_col.slider("Histogram bins", 10, 80, 30, disabled=chart_type != "Histogram")
    distribution = distribution_figure(
        plot_data,
        feature_label=active_feature_labels[feature],
        scale_label=SCALE_LABELS[scale],
        diagnoses=diagnoses,
        module=module,
        chart_type=chart_type,
        bins=bins,
        module_definition=module_set_label,
    )
    render_plotly_chart(
        distribution,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": (
                    f"{download_prefix}M{module}_{feature}_{method}_distribution"
                ),
                "scale": 3,
            },
        },
    )
    summary = distribution_summary(plot_data)
    summary.insert(0, "module_definition", module_set_label)
    with st.expander("Distribution summary table", expanded=False):
        filterable_dataframe(
            summary,
            table_key="distribution_summary",
            table_name="Distribution summary",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download distribution summary (TSV)",
            data=dataframe_to_tsv_bytes(summary),
            file_name=(
                f"{download_prefix}M{module}_{feature}_{method}_distribution_summary.tsv"
            ),
            mime="text/tab-separated-values",
        )

if active_view == "CT–TS screen":
    st.subheader("Descriptive CT–TS pattern screen")
    st.caption(
        f"Ranks all {module_count} modules for the selected phenotype, feature, diagnosis, "
        "and method by "
        f"the absolute difference between CT and TS {correlation_method} correlations. "
        "This is an exploratory screen, not an independent validation ranking."
    )
    screen_diagnosis = st.selectbox(
        "Diagnosis for screen",
        options=diagnoses,
        index=diagnoses.index("AD") if "AD" in diagnoses else 0,
        key="screen_diagnosis",
    )
    if phenotype == "clusters":
        cluster_screen = cached_module_set_cluster_associations(
            module_set, module_count, estimator, method, False, feature,
            ("CT", "TS"), (screen_diagnosis,), False, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        wide = cluster_screen.pivot_table(
            index="module",
            columns="component",
            values=["epsilon_squared", "categorical_p", "categorical_fdr_across_modules"],
            aggfunc="first",
        )
        wide.columns = [f"{value}_{component}" for value, component in wide.columns]
        screen = wide.reset_index()
        screen["delta_epsilon_squared_CT_minus_TS"] = (
            screen["epsilon_squared_CT"] - screen["epsilon_squared_TS"]
        )
        screen["abs_delta_epsilon_squared"] = screen[
            "delta_epsilon_squared_CT_minus_TS"
        ].abs()
        screen = screen.merge(
            annotations[["module", "displayed_category", "displayed_subcategory", "displayed_pathway", "displayed_fdr"]],
            on="module", how="left", validate="one_to_one",
        ).sort_values("abs_delta_epsilon_squared", ascending=False, na_position="last")
        screen.insert(0, "screen_rank", range(1, len(screen) + 1))
        screen.insert(1, "module_definition", module_set_label)
        n_screen = st.slider("Rows to show", 10, module_count, min(30, module_count), 10)
        st.warning(
            "CT−TS epsilon-squared differences are descriptive effect-size contrasts. "
            "No permutation p-value or FDR is assigned to the difference itself; the table "
            "shows each component's omnibus p-value and across-module FDR."
        )
        filterable_dataframe(
            screen.head(n_screen), table_key="cluster_ct_ts_screen",
            table_name="Nominal cluster CT–TS screen", use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"Download complete {module_count}-module cluster screen (TSV)",
            data=dataframe_to_tsv_bytes(screen),
            file_name=f"{download_prefix}{method}_{screen_diagnosis}_clusters_{feature}_CT_TS_screen.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    screen = cached_aggregate_stats(
        module_set, estimator, method, None, phenotype, feature, edge_rule,
        differential_edge_rule, differential_fdr_scope,
        differential_fdr_threshold, score_normalization, analysis_subset,
    )
    screen = screen.loc[screen["diagnosis_group"].eq(screen_diagnosis)].copy()
    if correlation_method == "Spearman":
        screen["correlation_CT"] = screen["rho_CT"]
        screen["p_CT"] = screen["p_spearman_CT"]
        screen["fdr_CT_global"] = screen["q_spearman_CT_all12_global"]
        screen["correlation_TS"] = screen["rho_TS"]
        screen["p_TS"] = screen["p_spearman_TS"]
        screen["fdr_TS_global"] = screen["q_spearman_TS_all12_global"]
        screen["ct_ts_test_statistic"] = screen["component_t_rank"]
        screen["p_CT_vs_TS"] = screen["p_component_rank"]
        screen["fdr_CT_vs_TS_within_phenotype"] = screen[
            "q_component_rank_within_phenotype"
        ]
        screen["fdr_CT_vs_TS_global"] = screen[
            "q_component_rank_all12_global"
        ]
    else:
        screen["correlation_CT"] = screen["r_rint_CT"]
        screen["p_CT"] = screen["p_rint_CT"]
        screen["fdr_CT_global"] = screen["q_rint_CT_all12_global"]
        screen["correlation_TS"] = screen["r_rint_TS"]
        screen["p_TS"] = screen["p_rint_TS"]
        screen["fdr_TS_global"] = screen["q_rint_TS_all12_global"]
        screen["ct_ts_test_statistic"] = screen["component_t_rint"]
        screen["p_CT_vs_TS"] = screen["p_component_rint"]
        screen["fdr_CT_vs_TS_within_phenotype"] = screen[
            "q_component_rint_within_phenotype"
        ]
        screen["fdr_CT_vs_TS_global"] = screen[
            "q_component_rint_all12_global"
        ]
    screen["correlation_method"] = correlation_method
    screen["delta_correlation_CT_minus_TS"] = (
        screen["correlation_CT"] - screen["correlation_TS"]
    )
    screen["abs_delta_correlation"] = screen[
        "delta_correlation_CT_minus_TS"
    ].abs()
    paired_correlations = screen["correlation_CT"].notna() & screen[
        "correlation_TS"
    ].notna()
    screen["opposite_CT_TS_sign"] = pd.Series(
        pd.NA, index=screen.index, dtype="boolean"
    )
    screen.loc[paired_correlations, "opposite_CT_TS_sign"] = (
        np.sign(screen.loc[paired_correlations, "correlation_CT"])
        != np.sign(screen.loc[paired_correlations, "correlation_TS"])
    )
    screen["max_abs_correlation"] = screen[
        ["correlation_CT", "correlation_TS"]
    ].abs().max(axis=1)
    annotation_columns = annotations[
        [
            "module",
            "displayed_category",
            "displayed_subcategory",
            "displayed_pathway",
            "displayed_fdr",
        ]
    ].copy()
    screen = screen.merge(annotation_columns, on="module", how="left")
    screen = screen.sort_values(
        ["abs_delta_correlation", "max_abs_correlation"],
        ascending=[False, False],
        na_position="last",
    )
    screen.insert(0, "screen_rank", np.arange(1, len(screen) + 1))
    screen.insert(1, "module_definition", module_set_label)
    n_screen = st.slider("Rows to show", 10, module_count, min(30, module_count), 10)
    screen_columns = [
        "screen_rank",
        "module_definition",
        "module",
        "correlation_method",
        "correlation_CT",
        "p_CT",
        "correlation_TS",
        "p_TS",
        "delta_correlation_CT_minus_TS",
        "abs_delta_correlation",
        "opposite_CT_TS_sign",
        "ct_ts_test_statistic",
        "p_CT_vs_TS",
        "displayed_category",
        "displayed_subcategory",
        "displayed_pathway",
        "displayed_fdr",
    ]
    ct_p_index = screen_columns.index("p_CT") + 1
    screen_columns[ct_p_index:ct_p_index] = ["fdr_CT_global"]
    ts_p_index = screen_columns.index("p_TS") + 1
    screen_columns[ts_p_index:ts_p_index] = ["fdr_TS_global"]
    ct_ts_p_index = screen_columns.index("p_CT_vs_TS") + 1
    screen_columns[ct_ts_p_index:ct_ts_p_index] = [
        "fdr_CT_vs_TS_within_phenotype",
        "fdr_CT_vs_TS_global",
    ]
    filterable_dataframe(
        screen[screen_columns].head(n_screen),
        table_key="ct_ts_screen",
        table_name="CT–TS screen",
        use_container_width=True,
        hide_index=True,
        column_config={"module": st.column_config.NumberColumn(format="M%d")},
    )
    st.download_button(
        f"Download complete {module_count}-module screen (TSV)",
        data=dataframe_to_tsv_bytes(screen[screen_columns]),
        file_name=(
            f"{download_prefix}{method}_{screen_diagnosis}_{phenotype}_{feature}_"
            f"{correlation_method.lower()}_CT_TS_screen.tsv"
        ),
        mime="text/tab-separated-values",
    )
    if correlation_method == "Spearman":
        st.caption(
            "Spearman and rank-based CT-vs-TS FDR values use the expanded all-12-outcome "
            "Benjamini–Hochberg families for the selected module definition, estimator, "
            "network method, component family, and BONOBO edge rule."
        )
    st.info(fdr_text(module_manifest, module_count))

if active_view == "Module finder":
    st.subheader("Find modules with contrasting association patterns")
    st.caption(
        "Rank modules by the difference between CT and TS correlations across all donors "
        "or within one diagnosis, by the difference between Control and AD correlations, "
        "or by both. "
        "All comparisons use the selected phenotype, feature, estimator, network method, "
        "edge rule, and Spearman/Pearson setting. These are association-pattern "
        "differences; they do not test a raw mean difference in the module score."
    )
    if phenotype == "clusters":
        cluster_finder_cohort = st.selectbox(
            "Cohort for nominal CT–TS comparison",
            ["All donors", *DIAGNOSIS_ORDER],
            index=0,
        )
        cluster_finder_stats = cached_module_set_cluster_associations(
            module_set, module_count, estimator, method, False, feature,
            ("CT", "TS"), tuple(DIAGNOSIS_ORDER), True, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        effect = cluster_finder_stats.pivot_table(
            index="module", columns=["diagnosis_group", "component"],
            values="epsilon_squared", aggfunc="first",
        )
        effect.columns = [f"epsilon_squared_{group}_{component}" for group, component in effect.columns]
        fdr = cluster_finder_stats.pivot_table(
            index="module", columns=["diagnosis_group", "component"],
            values="categorical_fdr_across_modules", aggfunc="first",
        )
        fdr.columns = [f"fdr_{group}_{component}" for group, component in fdr.columns]
        finder = effect.join(fdr).reset_index()
        finder["ct_ts_delta_epsilon_squared"] = (
            finder[f"epsilon_squared_{cluster_finder_cohort}_CT"]
            - finder[f"epsilon_squared_{cluster_finder_cohort}_TS"]
        )
        finder["ct_ts_abs_delta_epsilon_squared"] = finder[
            "ct_ts_delta_epsilon_squared"
        ].abs()
        for component in ("CT", "TS"):
            finder[f"ad_control_delta_epsilon_squared_{component}"] = (
                finder[f"epsilon_squared_AD_{component}"]
                - finder[f"epsilon_squared_Control_{component}"]
            )
        finder["ad_control_max_abs_delta_epsilon_squared"] = finder[
            ["ad_control_delta_epsilon_squared_CT", "ad_control_delta_epsilon_squared_TS"]
        ].abs().max(axis=1)
        finder["combined_descriptive_score"] = (
            finder["ct_ts_abs_delta_epsilon_squared"]
            + finder["ad_control_max_abs_delta_epsilon_squared"]
        )
        cluster_finder_criterion = st.radio(
            "Ranking criterion",
            ["CT–TS difference", "Control–AD difference", "Both"],
            horizontal=True,
        )
        sort_column = {
            "CT–TS difference": "ct_ts_abs_delta_epsilon_squared",
            "Control–AD difference": "ad_control_max_abs_delta_epsilon_squared",
            "Both": "combined_descriptive_score",
        }[cluster_finder_criterion]
        finder = finder.merge(
            annotations[["module", "displayed_category", "displayed_subcategory", "displayed_pathway", "displayed_fdr"]],
            on="module", how="left", validate="one_to_one",
        ).sort_values(sort_column, ascending=False, na_position="last")
        finder.insert(0, "finder_rank", range(1, len(finder) + 1))
        finder.insert(1, "module_definition", module_set_label)
        finder_rows = st.slider("Top modules", 10, module_count, min(30, module_count), 10)
        st.warning(
            "These epsilon-squared differences are descriptive. FDR columns belong to the "
            "underlying component omnibus tests, not to a formal test of the difference."
        )
        filterable_dataframe(
            finder.head(finder_rows), table_key="cluster_module_finder",
            table_name="Nominal cluster module finder", use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"Download complete {module_count}-module cluster finder (TSV)",
            data=dataframe_to_tsv_bytes(finder),
            file_name=f"{download_prefix}clusters_{feature}_module_finder.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    finder_statistics = cached_aggregate_stats(
        module_set, estimator, method, None, phenotype, feature, edge_rule,
        differential_edge_rule, differential_fdr_scope,
        differential_fdr_threshold, score_normalization, analysis_subset,
    )
    available_finder_diagnoses = [
        cohort
        for cohort in DIAGNOSIS_ORDER
        if cohort in set(finder_statistics["diagnosis_group"].dropna().astype(str))
    ]
    if not {"Control", "AD"}.issubset(available_finder_diagnoses):
        st.warning(
            "This evaluation cohort does not contain both Control and AD association "
            "statistics. Choose All donors, Discovery AD/Control, or Held-out validation "
            "AD/Control in the sidebar."
        )
        st.stop()

    finder_controls = st.columns([1.2, 1.0, 1.0, 0.8])
    with finder_controls[0]:
        finder_criterion = st.radio(
            "Ranking criterion",
            options=list(FINDER_CRITERIA),
            format_func=lambda value: FINDER_CRITERIA[value],
            horizontal=True,
        )
    with finder_controls[1]:
        finder_ct_ts_diagnosis = st.selectbox(
            "Cohort for CT–TS comparison",
            options=["All donors", *available_finder_diagnoses],
            index=0,
            help=(
                "All donors pools Control, MCI, and AD donors for the phenotype–score "
                "correlations. Diagnosis choices calculate the same CT–TS contrast "
                "within the selected group only."
            ),
        )
    with finder_controls[2]:
        finder_fdr_filter = st.selectbox(
            "Association-difference FDR filter",
            options=[None, 0.10, 0.05],
            format_func=lambda value: (
                "No FDR filter" if value is None else f"FDR ≤ {value:.2f}"
            ),
            help=(
                "For Both criteria, the selected cutoff must be met by both the CT–TS "
                "test and at least one Control–AD component comparison."
            ),
        )
    with finder_controls[3]:
        finder_rows = st.slider(
            "Top modules",
            min_value=10,
            max_value=module_count,
            value=min(30, module_count),
            step=10,
        )

    finder_effect_controls = st.columns(2)
    with finder_effect_controls[0]:
        minimum_ct_ts_difference = st.slider(
            "Minimum absolute CT–TS Δ correlation",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            disabled=finder_criterion == "ad_control",
        )
    with finder_effect_controls[1]:
        minimum_ad_control_difference = st.slider(
            "Minimum absolute Control–AD Δ correlation",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            disabled=finder_criterion == "ct_ts",
        )

    if finder_ct_ts_diagnosis == "All donors":
        pooled_finder_statistics = cached_pooled_module_finder_stats(
            module_set,
            estimator,
            method,
            phenotype,
            feature,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
        )
        finder_statistics = pd.concat(
            [finder_statistics, pooled_finder_statistics], ignore_index=True
        )

    finder = build_module_finder_table(
        finder_statistics,
        ct_ts_diagnosis=finder_ct_ts_diagnosis,
        correlation_method=correlation_method,
        criterion=finder_criterion,
    )
    finder = finder.merge(
        annotations[
            [
                "module",
                "displayed_category",
                "displayed_subcategory",
                "displayed_pathway",
                "displayed_fdr",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    ).merge(
        module_details[
            [
                "module",
                "module_size",
                "cluster_type",
                "tissue_entropy_normalized",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    )
    finder.insert(1, "module_definition", module_set_label)
    finder.insert(2, "phenotype", phenotype)
    finder.insert(3, "metric_family", feature)
    finder_filtered = finder.copy()
    if finder_criterion in {"ct_ts", "both"}:
        finder_filtered = finder_filtered.loc[
            finder_filtered["ct_ts_abs_delta_correlation"].ge(
                minimum_ct_ts_difference
            )
        ]
    if finder_criterion in {"ad_control", "both"}:
        finder_filtered = finder_filtered.loc[
            finder_filtered["ad_control_max_abs_delta_correlation"].ge(
                minimum_ad_control_difference
            )
        ]
    if finder_fdr_filter is not None:
        ct_ts_significant = finder_filtered["ct_ts_fdr_within_phenotype"].le(
            finder_fdr_filter
        )
        ad_control_significant = finder_filtered["ad_control_min_fdr"].le(
            finder_fdr_filter
        )
        if finder_criterion == "ct_ts":
            finder_filtered = finder_filtered.loc[ct_ts_significant]
        elif finder_criterion == "ad_control":
            finder_filtered = finder_filtered.loc[ad_control_significant]
        else:
            finder_filtered = finder_filtered.loc[
                ct_ts_significant & ad_control_significant
            ]
    finder_filtered = finder_filtered.sort_values(
        "finder_rank", kind="stable"
    ).reset_index(drop=True)
    finder_filtered.insert(0, "display_rank", np.arange(1, len(finder_filtered) + 1))

    if finder_filtered.empty:
        st.warning(
            "No modules meet the current effect-size and FDR filters. Relax one or more "
            "filters to restore the exploratory ranking."
        )
    else:
        top_finder = finder_filtered.iloc[0]
        finder_summary = st.columns(4)
        finder_summary[0].metric("Modules retained", f"{len(finder_filtered):,}")
        finder_summary[1].metric("Top module", f"M{int(top_finder['module'])}")
        finder_summary[2].metric(
            "Top CT–TS |Δr|",
            f"{top_finder['ct_ts_abs_delta_correlation']:.3f}",
        )
        finder_summary[3].metric(
            "Top Control–AD |Δr|",
            f"{top_finder['ad_control_max_abs_delta_correlation']:.3f}",
        )
        finder_figure = module_finder_figure(
            finder_filtered,
            phenotype_label=OUTCOME_LABELS[phenotype],
            feature_label=active_feature_labels[feature],
            correlation_method=correlation_method,
            ct_ts_diagnosis=finder_ct_ts_diagnosis,
            criterion_label=FINDER_CRITERIA[finder_criterion],
            selected_module=module,
            label_count=min(10, finder_rows),
            minimum_ct_ts_difference=minimum_ct_ts_difference,
            minimum_ad_control_difference=minimum_ad_control_difference,
        )
        render_plotly_chart(
            finder_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}{phenotype}_{feature}_module_finder",
                    "scale": 3,
                },
            },
        )
        finder_columns = [
            "display_rank",
            "module_definition",
            "module",
            "finder_criterion",
            "finder_score",
            "correlation_method",
            "ct_ts_diagnosis",
            "ct_ts_n",
            "ct_ts_correlation_CT",
            "ct_ts_correlation_TS",
            "ct_ts_delta_correlation",
            "ct_ts_abs_delta_correlation",
            "ct_ts_test_statistic",
            "ct_ts_p_value",
            "ct_ts_fdr_within_phenotype",
            "ct_ts_fdr_all12_global",
            "n_control",
            "n_ad",
            "correlation_CT_control",
            "correlation_CT_ad",
            "ad_control_delta_correlation_CT",
            "ad_control_fisher_z_CT",
            "ad_control_p_value_CT",
            "ad_control_fdr_CT",
            "correlation_TS_control",
            "correlation_TS_ad",
            "ad_control_delta_correlation_TS",
            "ad_control_fisher_z_TS",
            "ad_control_p_value_TS",
            "ad_control_fdr_TS",
            "ad_control_best_component",
            "ad_control_max_abs_delta_correlation",
            "ad_control_min_fdr",
            "module_size",
            "cluster_type",
            "tissue_entropy_normalized",
            "displayed_category",
            "displayed_subcategory",
            "displayed_pathway",
            "displayed_fdr",
        ]
        finder_columns = [
            column for column in finder_columns if column in finder_filtered.columns
        ]
        filterable_dataframe(
            finder_filtered[finder_columns].head(finder_rows),
            table_key="module_finder",
            table_name="Module finder ranking",
            use_container_width=True,
            hide_index=True,
            height=560,
            column_config={
                "module": st.column_config.NumberColumn(format="M%d"),
                "finder_score": st.column_config.ProgressColumn(
                    "Ranking score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "tissue_entropy_normalized": st.column_config.ProgressColumn(
                    "Normalized tissue entropy", min_value=0.0, max_value=1.0,
                    format="%.3f",
                ),
            },
        )
        st.download_button(
            "Download current module-finder ranking (TSV)",
            data=dataframe_to_tsv_bytes(finder_filtered[finder_columns]),
            file_name=(
                f"{download_prefix}{phenotype}_{feature}_{correlation_method.lower()}_"
                f"{finder_criterion}_module_finder.tsv"
            ),
            mime="text/tab-separated-values",
        )
    st.info(
        "CT–TS FDR uses the existing dependent-component test corrected across modules "
        "within the selected phenotype. For All donors this test is calculated from the "
        "pooled donor rows; its legacy all-12-outcome FDR field remains unavailable. "
        "Control–AD differences use an approximate "
        "independent-groups Fisher-z test for CT and TS correlations, with BH across "
        f"both components and all {module_count} displayed modules. The Both score is "
        "the lower of the two percentile scores, so a module must rank well on both "
        "criteria rather than excelling on only one."
    )

if active_view == "Edge summaries":
    st.subheader("Donor- and diagnosis-level edge summaries")
    st.caption(
        "Each undirected edge is counted once and the diagonal is excluded. LIONESS sign "
        "counts are recovered from the stored density and weight-sum identities and validated "
        "as integers. BONOBO counts are computed directly from the selected all-edge or "
        "significant-edge mask. Structurally unavailable scopes remain missing. These rows "
        "summarize each underlying network/edge rule once, so they do not change when the "
        "derived feature dropdown changes."
    )
    edge_data = attach_metadata(
        cached_edge_summaries(
            module_set, estimator, method, module, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold,
        )
    )
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        edge_data = edge_data.loc[edge_data["ad_control_split"].eq(selected_split)]
    edge_data = edge_data.loc[edge_data["diagnosis_group"].isin(diagnoses)].copy()
    edge_scope_options = edge_data["scope"].drop_duplicates().tolist()
    selected_edge_scopes = st.multiselect(
        "Edge scopes",
        options=edge_scope_options,
        default=edge_scope_options,
        format_func=lambda value: edge_data.loc[
            edge_data["scope"].eq(value), "scope_label"
        ].iloc[0],
        key=f"edge_scopes_{module_set}_{estimator}_{method}_{edge_rule}",
    )
    edge_data = edge_data.loc[edge_data["scope"].isin(selected_edge_scopes)]
    edge_metric_labels = {
        "n_possible_edges": "Possible edges",
        "n_retained_edges": "Retained edges",
        "n_positive_edges": "Positive edges",
        "n_negative_edges": "Negative edges",
        "n_zero_edges": "Zero-weight edges",
        "n_pruned_edges": "Pruned edges",
        "signed_weight_sum": "Signed weight sum",
        "positive_weight_sum": "Positive weight sum",
        "negative_weight_magnitude": "Negative weight magnitude",
        "absolute_weight_sum": "Absolute weight sum",
    }
    edge_metric = st.selectbox(
        "Edge summary metric",
        options=list(edge_metric_labels),
        format_func=lambda value: edge_metric_labels[value],
        key=f"edge_metric_{module_set}_{estimator}",
    )
    if edge_data.empty:
        st.info("No edge scopes remain for the current filters.")
    else:
        render_plotly_chart(
            edge_summary_figure(
                edge_data,
                edge_metric,
                edge_metric_labels[edge_metric],
                module,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        edge_group_summary = (
            edge_data.groupby(
                ["scope", "scope_label", "diagnosis_group"], observed=True
            )[edge_metric]
            .agg(
                group_n="count",
                mean="mean",
                sd="std",
                median="median",
                q1=lambda values: values.quantile(0.25),
                q3=lambda values: values.quantile(0.75),
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )
        edge_group_summary["iqr"] = (
            edge_group_summary["q3"] - edge_group_summary["q1"]
        )
        edge_group_summary.insert(0, "module_definition", module_set_label)
        edge_group_summary.insert(1, "estimator", ESTIMATOR_LABELS[estimator])
        edge_group_summary.insert(2, "network_method", readable_method(method))
        edge_group_summary.insert(3, "edge_rule", edge_rule)
        filterable_dataframe(
            edge_group_summary,
            table_key="edge_group_summary",
            table_name="Diagnosis-group edge summary",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download diagnosis-group summary (TSV)",
            data=dataframe_to_tsv_bytes(edge_group_summary),
            file_name=f"{download_prefix}M{module}_{edge_metric}_group_summary.tsv",
            mime="text/tab-separated-values",
        )
        with st.expander("Donor-level edge-summary rows", expanded=False):
            filterable_dataframe(
                edge_data,
                table_key="donor_edge_summary",
                table_name="Donor-level edge summary",
                use_container_width=True,
                hide_index=True,
                height=480,
            )
            st.download_button(
                "Download donor-level edge summaries (TSV)",
                data=dataframe_to_tsv_bytes(edge_data),
                file_name=f"{download_prefix}M{module}_donor_edge_summaries.tsv",
                mime="text/tab-separated-values",
            )

if active_view == "Edge volcano":
    st.subheader("AD–Control differential edges")
    st.caption(
        "Each point is one undirected module edge. The mask is discovered with a "
        "two-sided Welch test in 117 AD and 114 Control donors; Hedges’ g and mean "
        "difference are AD minus Control. Both global and per-module BH values are "
        "stored for every edge; this view and the filtered-score mask use the "
        f"sidebar selection (**{FDR_SCOPE_LABELS[differential_fdr_scope]}**). The "
        "validation view uses the held-out 50 AD and 50 Control donors and never "
        "changes the discovery mask."
    )
    if not differential_available:
        st.info("The differential-edge volcano bundle is not available in this deployment.")
    else:
        volcano_candidates = cached_volcano_candidates(
            module_set, estimator, method, module
        )
        volcano_bins = cached_volcano_bins(
            module_set, estimator, method, module, differential_fdr_scope
        )
        volcano_left, volcano_middle, volcano_right = st.columns(3)
        with volcano_left:
            volcano_analysis_set = st.radio(
                "Edge-test cohort",
                options=["Discovery", "Validation"],
                horizontal=True,
                key=f"volcano_set_{module_set}_{estimator}_{method}_{module}",
            )
            volcano_x_metric = st.selectbox(
                "Effect axis",
                options=["hedges_g", "mean_difference"],
                format_func=lambda value: {
                    "hedges_g": "Hedges’ g",
                    "mean_difference": "Mean edge-weight difference",
                }[value],
            )
        with volcano_middle:
            volcano_y_metric = st.selectbox(
                "Significance axis",
                options=["fdr", "p_value"],
                format_func=lambda value: {
                    "fdr": "BH FDR",
                    "p_value": "Welch p-value",
                }[value],
            )
            volcano_threshold = st.selectbox(
                "Volcano display threshold",
                options=[0.05, 0.10],
                index=0 if np.isclose(differential_fdr_threshold, 0.05) else 1,
                help=(
                    "Defaults to the sidebar feature-mask cutoff. Changing this only "
                    "changes the volcano display, not the calculated module scores."
                ),
            )
        with volcano_right:
            volcano_scope_options = volcano_bins["scope"].drop_duplicates().tolist()
            volcano_scope = st.selectbox(
                "Edge scope",
                options=volcano_scope_options,
                format_func=lambda value: {
                    "total": "Total",
                    "TS": "TS pooled",
                    "CT": "CT pooled",
                    "TS_AC": "TS: AC",
                    "TS_DLPFC": "TS: DLPFC",
                    "TS_PCGBA23": "TS: PCG",
                    "CT_AC__DLPFC": "CT: AC - DLPFC",
                    "CT_AC__PCGBA23": "CT: AC - PCG",
                    "CT_DLPFC__PCGBA23": "CT: DLPFC - PCG",
                }.get(value, value),
            )
            volcano_direction = st.selectbox(
                "Effect direction", ["Either", "Higher in AD", "Higher in Control"]
            )
        significant_only = st.checkbox(
            "Hide nonsignificant edges",
            value=False,
            help=(
                "When selected, nonsignificant edges are removed from the display. Their "
                "effect sizes are never replaced by zero."
            ),
        )
        prevalence_column = None
        minimum_prevalence = 0.0
        if estimator == "bonobo":
            available_prevalence_rules = [
                rule
                for rule in ("native_p05", "bh_fdr05")
                if any(
                    f"bonobo_{rule}_prevalence_{group}" in volcano_candidates.columns
                    for group in ("all", "control", "mci", "ad")
                )
            ]
            if available_prevalence_rules:
                prevalence_cols = st.columns(3)
                with prevalence_cols[0]:
                    prevalence_rule = st.selectbox(
                        "BONOBO prevalence rule",
                        options=available_prevalence_rules,
                        format_func=lambda value: BONOBO_EDGE_RULE_LABELS[value],
                    )
                available_prevalence_groups = [
                    group
                    for group in ("all", "control", "mci", "ad")
                    if f"bonobo_{prevalence_rule}_prevalence_{group}"
                    in volcano_candidates.columns
                ]
                with prevalence_cols[1]:
                    prevalence_group = st.selectbox(
                        "Prevalence donor group",
                        options=available_prevalence_groups,
                        format_func=lambda value: {
                            "all": "All donors",
                            "control": "Control",
                            "mci": "MCI",
                            "ad": "AD",
                        }[value],
                    )
                with prevalence_cols[2]:
                    minimum_prevalence = st.slider(
                        "Minimum native significance prevalence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.0f%%",
                    )
                prevalence_column = (
                    f"bonobo_{prevalence_rule}_prevalence_{prevalence_group}"
                )
            else:
                st.caption(
                    "BONOBO native-significance prevalence is unavailable in this "
                    "volcano shard; effect and AD–Control significance filters remain active."
                )

        volcano = edge_volcano_figure(
            volcano_candidates,
            volcano_bins,
            module=module,
            scope=volcano_scope,
            analysis_set=volcano_analysis_set,
            fdr_scope=differential_fdr_scope,
            x_metric=volcano_x_metric,
            y_metric=volcano_y_metric,
            significant_only=significant_only,
            significance_threshold=volcano_threshold,
            direction=volcano_direction,
            prevalence_column=prevalence_column,
            minimum_prevalence=minimum_prevalence,
            module_definition=module_set_label,
        )
        render_plotly_chart(
            volcano,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}M{module}_edge_volcano",
                    "scale": 3,
                },
            },
        )
        st.caption(
            "Exact dots are colored by tissue component: circles are within-tissue "
            "edges and diamonds are cross-tissue pairs. The gray density background "
            "includes every edge in the selected scope. Exact hoverable rows include "
            "all discovery/validation FDR≤0.10 or p≤0.05 edges plus the top 500 "
            "remaining edges per resolved component."
        )
        prefix = volcano_analysis_set.lower()
        probability_column = (
            f"{prefix}_fdr_{differential_fdr_scope}"
            if volcano_y_metric == "fdr"
            else f"{prefix}_p_value"
        )
        displayed_edges = volcano_candidates.copy()
        if volcano_scope in {"CT", "TS"}:
            displayed_edges = displayed_edges.loc[
                displayed_edges["component_class"].eq(volcano_scope)
            ]
        elif volcano_scope != "total":
            displayed_edges = displayed_edges.loc[
                displayed_edges["component"].eq(volcano_scope)
            ]
        if volcano_direction == "Higher in AD":
            displayed_edges = displayed_edges.loc[
                displayed_edges[f"{prefix}_mean_difference"].gt(0)
            ]
        elif volcano_direction == "Higher in Control":
            displayed_edges = displayed_edges.loc[
                displayed_edges[f"{prefix}_mean_difference"].lt(0)
            ]
        if significant_only:
            displayed_edges = displayed_edges.loc[
                displayed_edges[probability_column].le(volcano_threshold)
            ]
        if prevalence_column and prevalence_column in displayed_edges.columns:
            displayed_edges = displayed_edges.loc[
                displayed_edges[prevalence_column].ge(minimum_prevalence)
            ]
        displayed_edges = displayed_edges.sort_values(
            [probability_column, f"{prefix}_hedges_g"],
            ascending=[True, False],
            kind="stable",
        )
        displayed_edges.insert(
            displayed_edges.columns.get_loc("component") + 1,
            "tissue_component",
            displayed_edges["component"].map(EDGE_COMPONENT_LABELS).fillna(
                displayed_edges["component"].astype(str)
            ),
        )
        filterable_dataframe(
            displayed_edges,
            table_key="volcano_edges",
            table_name="Displayed volcano edges",
            use_container_width=True,
            hide_index=True,
            height=460,
            column_config={
                "gene_a": "Gene A symbol",
                "gene_b": "Gene B symbol",
                "tissue_component": "Tissue component",
            },
        )
        st.download_button(
            "Download displayed exact edge rows (TSV)",
            data=dataframe_to_tsv_bytes(displayed_edges),
            file_name=f"{download_prefix}M{module}_displayed_edge_statistics.tsv",
            mime="text/tab-separated-values",
        )

if active_view == "MDC":
    st.subheader("Module differential connectivity (MDC)")
    if (
        differential_edge_rule != "all"
        and not differential_mdc_data_available(module_set)
    ):
        st.info(
            "AD–Control-filtered MDC is currently being generated for this module "
            "definition. Select **All edges** to use the existing MDC catalog now; "
            "the filtered option will activate automatically when its validated public "
            "tables are deployed."
        )
        st.stop()
    mdc_summary = cached_mdc_summary(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
    )
    if mdc_summary.empty:
        st.error("No MDC rows are available for the selected edge mask and network method.")
        st.stop()
    st.caption(
        "MDC compares the mean absolute signedAlt adjacency in AD with Control. "
        "Values above 1 indicate higher connectivity in AD; values below 1 indicate "
        "higher connectivity in Control. Total uses all edges, TS uses same-tissue edges, "
        "and CT uses cross-tissue edges."
    )
    mdc_metadata = (
        data_manifest.get("mdc_differential_edges", {})
        if differential_edge_rule != "all"
        else module_manifest.get("mdc", data_manifest.get("mdc", {}))
    )
    estimator_note = (
        "All-edge MDC is module-level context and does not change with the estimator "
        "or network-method selector."
        if differential_edge_rule == "all"
        else (
            f"This view applies the {ESTIMATOR_LABELS[estimator]} / "
            f"{readable_method(method)} discovery mask selected above. The mask is held "
            "fixed during MDC calculation; BONOBO donor-native edge significance is not "
            "an additional group-MDC filter."
        )
    )
    st.warning(
        "Cohort scope differs from the donor-complete LIONESS analysis. The MDC source "
        f"assembled {mdc_metadata.get('reference_assembled_donors', 517)} AD and "
        f"{mdc_metadata.get('target_assembled_donors', 408)} Control donors across the "
        "tissue union, including the 167 AD and 164 Control complete-three-tissue donors "
        "used here plus donors with partial tissue availability. MCI was not included. "
        + estimator_note
    )
    if differential_edge_rule != "all":
        st.warning(
            "Filtered MDC is exploratory post-selection context. The AD–Control discovery "
            "mask is fixed in the 200 sample and 200 tissue-count-preserving gene "
            "permutations, so its directional FDR does not account for adaptive edge "
            "selection and is not independent validation. Modules with no retained edge "
            "in a scope remain unavailable rather than being assigned MDC=0."
        )
    mdc_control_columns = st.columns(2)
    with mdc_control_columns[0]:
        mdc_threshold = st.radio(
            "MDC significance threshold",
            options=[0.05, 0.10],
            format_func=lambda value: f"FDR < {value:.2f}",
            horizontal=True,
            key="mdc_threshold",
        )
    with mdc_control_columns[1]:
        mdc_scale = st.radio(
            "MDC display scale",
            options=["log2", "raw"],
            format_func=lambda value: {
                "log2": "log₂ ratio (equality = 0)",
                "raw": "Raw AD/Control ratio (equality = 1)",
            }[value],
            horizontal=True,
            key="mdc_display_scale",
        )
    if mdc_scale == "raw":
        st.caption(
            "Raw heatmap cells and hovers show MDC ratios. Heatmap colors are "
            "log-symmetric around MDC=1, so reciprocal changes such as 0.5 and 2 "
            "have equal color intensity in opposite directions."
        )

    mdc_view = mdc_summary.copy()
    if "module_definition" in mdc_view:
        mdc_view.pop("module_definition")
    mdc_view.insert(0, "module_definition", module_set_label)
    mdc_view = mdc_view.merge(
        module_details[
            [
                "module",
                "module_size",
                "cluster_type",
                "tissue_entropy",
                "tissue_entropy_normalized",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    )
    if mdc_view["tissue_entropy_normalized"].isna().any():
        st.warning(
            "Some MDC modules lack module-composition entropy and will be omitted from "
            "the entropy relationship plots."
        )
    for scope in ["total", "ts", "ct"]:
        mdc_view[f"significant_{scope}"] = mdc_view[
            f"directional_fdr_{scope}"
        ].lt(mdc_threshold)
    mdc_view["any_significant"] = mdc_view[
        ["significant_total", "significant_ts", "significant_ct"]
    ].any(axis=1)
    if differential_edge_rule != "all":
        available_counts = {
            scope.upper(): int(mdc_view[f"mdc_{scope}"].notna().sum())
            for scope in ("total", "ts", "ct")
        }
        st.caption(
            "Modules with at least one retained edge and an available MDC ratio: "
            + " · ".join(
                f"{scope}={count}/{module_count}"
                for scope, count in available_counts.items()
            )
        )

    count_columns = st.columns(4)
    count_columns[0].metric(
        "Total significant", int(mdc_view["significant_total"].sum())
    )
    count_columns[1].metric("TS significant", int(mdc_view["significant_ts"].sum()))
    count_columns[2].metric("CT significant", int(mdc_view["significant_ct"].sum()))
    count_columns[3].metric("Any MDC significant", int(mdc_view["any_significant"].sum()))

    selected_mdc = mdc_view.loc[mdc_view["module"].astype(int).eq(int(module))]
    if selected_mdc.empty:
        st.error(f"No MDC row is available for {module_label(module)}.")
    else:
        selected_mdc_row = selected_mdc.iloc[0]
        st.markdown(f"#### Selected module: {module_label(module)}")
        if differential_edge_rule != "all":
            st.caption(
                "Retained discovery edges: "
                f"total={int(selected_mdc_row['n_retained_edges_total']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_total']):,}, "
                f"TS={int(selected_mdc_row['n_retained_edges_ts']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_ts']):,}, "
                f"CT={int(selected_mdc_row['n_retained_edges_ct']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_ct']):,}."
            )
        selected_columns = st.columns(3)
        for container, scope, label in zip(
            selected_columns,
            ["total", "ts", "ct"],
            ["Total MDC", "TS MDC", "CT MDC"],
            strict=True,
        ):
            ratio = selected_mdc_row[f"mdc_{scope}"]
            directional_fdr = selected_mdc_row[f"directional_fdr_{scope}"]
            direction = selected_mdc_row[f"direction_{scope}"]
            ratio_text = "NA" if pd.isna(ratio) else f"{float(ratio):.3f}"
            fdr_value_text = (
                "NA" if pd.isna(directional_fdr) else f"{float(directional_fdr):.3g}"
            )
            significance_text = (
                "Significant"
                if pd.notna(directional_fdr) and directional_fdr < mdc_threshold
                else "Not significant"
            )
            if pd.isna(ratio):
                significance_text = "Not available"
            with container:
                st.metric(label, ratio_text)
                st.caption(
                    f"{direction} · directional FDR={fdr_value_text} · {significance_text}"
                )

        selected_mdc_chart = mdc_module_figure(
            selected_mdc_row,
            mdc_threshold,
            module_definition=module_set_label,
            scale=mdc_scale,
        )
        render_plotly_chart(
            selected_mdc_chart,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}M{module}_AD_Control_MDC",
                    "scale": 3,
                },
            },
        )
        st.caption(
            (
                "Bars use log2 MDC, so equal AD/Control connectivity is centered at zero. "
                if mdc_scale == "log2"
                else "Bars use the raw AD/Control MDC ratio, so equal connectivity is marked at one. "
            )
            +
            "Orange indicates higher connectivity in AD, blue indicates higher connectivity "
            "in Control, and ★ marks significance at the selected threshold."
        )

    significance_filter = st.selectbox(
        "Modules to show in the MDC overview and entropy plots",
        options=[
            "All modules",
            "Any significant",
            "Total significant",
            "TS significant",
            "CT significant",
            "No significant MDC",
        ],
    )
    filter_masks = {
        "All modules": pd.Series(True, index=mdc_view.index),
        "Any significant": mdc_view["any_significant"],
        "Total significant": mdc_view["significant_total"],
        "TS significant": mdc_view["significant_ts"],
        "CT significant": mdc_view["significant_ct"],
        "No significant MDC": ~mdc_view["any_significant"],
    }
    displayed_mdc = mdc_view.loc[filter_masks[significance_filter]].copy()
    mdc_overview = mdc_overview_figure(
        displayed_mdc,
        module,
        mdc_threshold,
        module_definition=module_set_label,
        scale=mdc_scale,
    )
    render_plotly_chart(
        mdc_overview,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{download_prefix}AD_Control_TS_CT_MDC_overview",
                "scale": 3,
            },
        },
    )
    if differential_edge_rule == "all":
        ct_unavailable_count = len(
            module_manifest.get(
                "ct_unavailable_modules",
                mdc_summary.loc[
                    pd.to_numeric(mdc_summary["n_ct_edges"], errors="coerce")
                    .fillna(0)
                    .eq(0),
                    "module",
                ].astype(int).tolist(),
            )
        )
        unavailable_reason = "have no structural CT edges"
    else:
        ct_unavailable_count = int(
            pd.to_numeric(mdc_summary["n_ct_edges"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        unavailable_reason = "retain no CT edge under the selected discovery mask"
    st.caption(
        (
            "Dashed zero lines separate AD-higher from Control-higher MDC. "
            if mdc_scale == "log2"
            else "Dashed equality lines at one separate AD-higher from Control-higher MDC. "
        )
        + "The dotted diagonal "
        f"marks equal TS and CT effects. {ct_unavailable_count} module(s) {unavailable_reason} "
        "and therefore no CT MDC point."
    )

    st.markdown("#### MDC and tissue-mixing entropy")
    st.caption(
        "Each point is one module after applying the module filter above. Diamonds are "
        f"directionally significant at FDR < {mdc_threshold:.2f}; open circles are not. "
        "The dotted line is an OLS visual guide, while the subtitle reports the Spearman "
        "association across the displayed modules."
    )
    entropy_chart_columns = st.columns(2)
    for chart_column, entropy_scope in zip(
        entropy_chart_columns, ["ts", "ct"], strict=True
    ):
        with chart_column:
            render_plotly_chart(
                mdc_entropy_figure(
                    displayed_mdc,
                    scope=entropy_scope,
                    selected_module=module,
                    threshold=mdc_threshold,
                    scale=mdc_scale,
                    module_definition=module_set_label,
                ),
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": (
                            f"{download_prefix}{entropy_scope.upper()}_MDC_"
                            f"normalized_Shannon_entropy_{mdc_scale}"
                        ),
                        "scale": 3,
                    },
                },
            )

    displayed_mdc["minimum_directional_fdr"] = displayed_mdc[
        ["directional_fdr_total", "directional_fdr_ts", "directional_fdr_ct"]
    ].min(axis=1, skipna=True)
    displayed_mdc["maximum_abs_log2_mdc"] = displayed_mdc[
        ["log2_mdc_total", "log2_mdc_ts", "log2_mdc_ct"]
    ].abs().max(axis=1, skipna=True)
    displayed_mdc = displayed_mdc.sort_values(
        ["minimum_directional_fdr", "maximum_abs_log2_mdc"],
        ascending=[True, False],
        na_position="last",
    )
    mdc_columns = [
        "module_definition",
        "module",
        "module_size_mapped",
        "module_size",
        "cluster_type",
        "tissue_entropy",
        "tissue_entropy_normalized",
        "mdc_total",
        "direction_total",
        "directional_fdr_total",
        "significant_total",
        "mdc_ts",
        "direction_ts",
        "directional_fdr_ts",
        "significant_ts",
        "mdc_ct",
        "direction_ct",
        "directional_fdr_ct",
        "significant_ct",
        "ts_minus_ct_log2_mdc",
        "n_ts_edges",
        "n_ct_edges",
    ]
    if differential_edge_rule != "all":
        mdc_columns.extend(
            [
                "estimator",
                "network_method",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "n_possible_edges_total",
                "n_possible_edges_ts",
                "n_possible_edges_ct",
                "n_retained_edges_total",
                "n_retained_edges_ts",
                "n_retained_edges_ct",
                "retained_edge_fraction_total",
                "retained_edge_fraction_ts",
                "retained_edge_fraction_ct",
                "mean_abs_ad_total",
                "mean_abs_control_total",
                "mean_abs_ad_ts",
                "mean_abs_control_ts",
                "mean_abs_ad_ct",
                "mean_abs_control_ct",
                "inference_scope",
            ]
        )
    filterable_dataframe(
        displayed_mdc[mdc_columns],
        table_key="mdc_summary",
        table_name="MDC summary",
        use_container_width=True,
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn(format="M%d"),
            "module_size": st.column_config.NumberColumn(format="%d"),
            "tissue_entropy": st.column_config.NumberColumn(format="%.4f"),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "mdc_total": st.column_config.NumberColumn(format="%.3f"),
            "mdc_ts": st.column_config.NumberColumn(format="%.3f"),
            "mdc_ct": st.column_config.NumberColumn(format="%.3f"),
            "directional_fdr_total": st.column_config.NumberColumn(format="%.3g"),
            "directional_fdr_ts": st.column_config.NumberColumn(format="%.3g"),
            "directional_fdr_ct": st.column_config.NumberColumn(format="%.3g"),
            "ts_minus_ct_log2_mdc": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.download_button(
        "Download displayed MDC table (TSV)",
        data=dataframe_to_tsv_bytes(displayed_mdc[mdc_columns]),
        file_name=(
            f"{download_prefix}AD_Control_MDC_"
            f"{significance_filter.replace(' ', '_')}.tsv"
        ),
        mime="text/tab-separated-values",
    )
    st.info(
        "For each scope and direction, sample-permutation and gene-permutation p-values were "
        f"Benjamini–Hochberg adjusted separately across the {module_count} modules. The "
        "directional FDR is the larger of those two adjusted values for the observed MDC "
        "direction, making significance require support from both null models. The latest "
        "source used 200 sample permutations and 200 gene permutations."
    )

    region_mdc_tab, pathway_mdc_tab = st.tabs(
        ["Region-resolved MDC", "Pathway-resolved MDC"]
    )
    with region_mdc_tab:
        st.markdown("#### Tissue-resolved MDC")
        st.caption(
            "Resolved MDC separates the three within-tissue blocks and the three cross-tissue "
            "pairs. Gene permutations preserve each module's AC, DLPFC, and PCG feature counts. "
            "For each component and direction, BH correction uses only modules where that edge "
            "block structurally exists; unavailable blocks remain missing."
        )
        resolved_mdc = cached_mdc_resolved(
            module_set,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        ).copy()
        resolved_components_available = resolved_mdc[
            ["component", "component_label"]
        ].drop_duplicates()
        selected_mdc_components = st.multiselect(
            "Resolved MDC components",
            options=resolved_components_available["component"].tolist(),
            default=resolved_components_available["component"].tolist(),
            format_func=lambda value: resolved_components_available.loc[
                resolved_components_available["component"].eq(value), "component_label"
            ].iloc[0],
            key=f"resolved_mdc_components_{module_set}",
        )
        resolved_mdc = resolved_mdc.loc[
            resolved_mdc["component"].isin(selected_mdc_components)
        ].copy()
        resolved_significance = st.selectbox(
            "Resolved MDC rows",
            options=["All", "FDR significant", "Not FDR significant"],
            key=f"resolved_mdc_significance_{module_set}",
        )
        resolved_is_significant = pd.to_numeric(
            resolved_mdc["directional_fdr"], errors="coerce"
        ).lt(mdc_threshold)
        if resolved_significance == "FDR significant":
            resolved_mdc = resolved_mdc.loc[resolved_is_significant]
        elif resolved_significance == "Not FDR significant":
            resolved_mdc = resolved_mdc.loc[~resolved_is_significant]
        selected_resolved_mdc = resolved_mdc.loc[
            resolved_mdc["module"].astype(int).eq(int(module))
        ]
        if selected_resolved_mdc.empty:
            st.info("The selected module has no resolved MDC row under the current filters.")
        else:
            render_plotly_chart(
                mdc_resolved_module_figure(
                    selected_resolved_mdc,
                    mdc_threshold,
                    module_definition=module_set_label,
                    scale=mdc_scale,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        if resolved_mdc.empty:
            st.info("No resolved MDC rows remain under the current filters.")
        else:
            render_plotly_chart(
                mdc_resolved_heatmap_figure(
                    resolved_mdc,
                    mdc_threshold,
                    selected_module=module,
                    module_definition=module_set_label,
                    scale=mdc_scale,
                ),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            resolved_mdc.insert(0, "module_definition_label", module_set_label)
            filterable_dataframe(
                resolved_mdc,
                table_key="resolved_mdc",
                table_name="Resolved MDC",
                use_container_width=True,
                hide_index=True,
                height=520,
            )
            st.download_button(
                "Download displayed resolved MDC rows (TSV)",
                data=dataframe_to_tsv_bytes(resolved_mdc),
                file_name=f"{download_prefix}AD_Control_resolved_MDC.tsv",
                mime="text/tab-separated-values",
            )

    with pathway_mdc_tab:
        st.markdown("#### Pathway-annotated regional MDC")
        st.caption(
            "MDC remains a module-level statistic. This view annotates each module-component "
            "MDC with KEGG pathways enriched in the matching region, then summarizes MDC "
            "across enriched modules. For a CT tissue pair, both regions must meet the KEGG "
            "threshold; the displayed pair FDR is the larger regional FDR and is not a new "
            "combined p-value."
        )
        pathway_control_columns = st.columns(4)
        with pathway_control_columns[0]:
            pathway_mdc_resolution = st.selectbox(
                "MDC enrichment resolution",
                options=list(MDC_ENRICHMENT_RESOLUTION_LABELS),
                format_func=MDC_ENRICHMENT_RESOLUTION_LABELS.get,
                key=f"pathway_mdc_resolution_{module_set}",
            )
        with pathway_control_columns[1]:
            pathway_kegg_threshold = st.radio(
                "KEGG enrichment threshold",
                options=[0.05, 0.10],
                format_func=lambda value: (
                    f"FDR < {value:.2f}"
                    + (" (exploratory)" if np.isclose(value, 0.10) else "")
                ),
                horizontal=True,
                key=f"pathway_mdc_kegg_threshold_{module_set}",
            )
        with pathway_control_columns[2]:
            minimum_pathway_modules = st.slider(
                "Minimum enriched modules per cell",
                min_value=1,
                max_value=10,
                value=1,
                key=f"pathway_mdc_minimum_modules_{module_set}",
            )
        with pathway_control_columns[3]:
            pathway_mdc_row_scope = st.radio(
                "Module MDC rows",
                options=["All", "MDC FDR-significant only"],
                horizontal=True,
                key=f"pathway_mdc_row_scope_{module_set}",
            )

        pathway_rows = cached_pathway_mdc_rows(
            module_set,
            pathway_kegg_threshold,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        ).copy()
        component_preference = [
            "TS_AC",
            "TS_DLPFC",
            "TS_PCGBA23",
            "CT_AC__DLPFC",
            "CT_AC__PCGBA23",
            "CT_DLPFC__PCGBA23",
            "total",
            "TS",
            "CT",
        ]
        component_labels = (
            pathway_rows[["component", "component_label"]]
            .drop_duplicates("component")
            .set_index("component")["component_label"]
            .to_dict()
        )
        available_pathway_components = [
            value for value in component_preference if value in component_labels
        ]
        default_pathway_components = available_pathway_components[:6]
        selected_pathway_components = st.multiselect(
            "Regions and tissue pairs",
            options=available_pathway_components,
            default=default_pathway_components,
            format_func=lambda value: component_labels[value],
            key=f"pathway_mdc_components_{module_set}",
        )
        pathway_rows = pathway_rows.loc[
            pathway_rows["component"].isin(selected_pathway_components)
        ].copy()
        if pathway_mdc_row_scope == "MDC FDR-significant only":
            pathway_rows = pathway_rows.loc[
                pd.to_numeric(pathway_rows["directional_fdr"], errors="coerce").lt(
                    mdc_threshold
                )
            ].copy()

        category_options = sorted(
            pathway_rows["category_level1"].dropna().astype(str).unique().tolist()
        )
        filter_columns = st.columns([2, 3])
        selected_pathway_categories = filter_columns[0].multiselect(
            "KEGG categories",
            options=category_options,
            default=[],
            placeholder="All categories",
            key=f"pathway_mdc_categories_{module_set}",
        )
        pathway_search = filter_columns[1].text_input(
            "Search pathways",
            placeholder="Pathway, category, sub-category, or KEGG ID",
            key=f"pathway_mdc_search_{module_set}",
        ).strip()
        if selected_pathway_categories:
            pathway_rows = pathway_rows.loc[
                pathway_rows["category_level1"].isin(selected_pathway_categories)
            ].copy()
        if pathway_search:
            searchable = pathway_rows[
                [
                    "pathway_id",
                    "pathway_label",
                    "category_level1",
                    "category_level2",
                ]
            ].fillna("").astype(str)
            pathway_rows = pathway_rows.loc[
                searchable.agg(" ".join, axis=1).str.contains(
                    pathway_search, case=False, regex=False
                )
            ].copy()

        pathway_group_rows = collapse_pathway_mdc_rows(
            pathway_rows,
            resolution=pathway_mdc_resolution,
        )
        pathway_summary = summarize_pathway_mdc_rows(
            pathway_group_rows,
            mdc_fdr_threshold=mdc_threshold,
            minimum_modules=minimum_pathway_modules,
            resolution=pathway_mdc_resolution,
        )
        if pathway_summary.empty:
            st.info("No KEGG-annotated MDC cells remain under the current filters.")
        else:
            retained_pathway_ids = set(pathway_summary["pathway_id"].astype(str))
            pathway_group_rows = pathway_group_rows.loc[
                pathway_group_rows["pathway_id"]
                .astype(str)
                .isin(retained_pathway_ids)
            ].copy()
            pathway_ranking = (
                pathway_summary.assign(
                    absolute_mean_log2_mdc=pathway_summary["mean_log2_mdc"].abs()
                )
                .groupby(["pathway_id", "pathway_label"], observed=True)
                .agg(
                    maximum_absolute_mean_log2_mdc=(
                        "absolute_mean_log2_mdc",
                        "max",
                    ),
                    minimum_enrichment_fdr=("minimum_enrichment_fdr", "min"),
                    maximum_module_support=("n_modules", "max"),
                )
                .reset_index()
                .sort_values(
                    [
                        "maximum_absolute_mean_log2_mdc",
                        "minimum_enrichment_fdr",
                        "maximum_module_support",
                    ],
                    ascending=[False, True, False],
                    kind="stable",
                )
            )
            pathway_metric_columns = st.columns(4)
            resolution_singular = MDC_ENRICHMENT_RESOLUTION_LABELS[
                pathway_mdc_resolution
            ]
            resolution_plural = {
                "pathway": "Pathways",
                "subcategory": "KEGG sub-categories",
                "category": "KEGG categories",
            }[pathway_mdc_resolution]
            pathway_metric_columns[0].metric(
                resolution_plural,
                f"{pathway_summary['pathway_id'].nunique():,}",
            )
            pathway_metric_columns[1].metric(
                "Modules", f"{pathway_group_rows['module'].nunique():,}"
            )
            pathway_metric_columns[2].metric(
                f"Region/{resolution_singular.lower()} cells",
                f"{len(pathway_summary):,}",
            )
            pathway_metric_columns[3].metric(
                "Module-component annotations", f"{len(pathway_group_rows):,}"
            )
            pathway_options = pathway_ranking["pathway_id"].astype(str).tolist()
            pathway_label_map = pathway_ranking.set_index("pathway_id")[
                "pathway_label"
            ].to_dict()
            pathway_selection_columns = st.columns([2, 1])
            selected_pathway_id = pathway_selection_columns[0].selectbox(
                f"{resolution_singular} detail",
                options=pathway_options,
                format_func=lambda value: (
                    f"{pathway_label_map[value]} ({value})"
                    if pathway_mdc_resolution == "pathway"
                    else pathway_label_map[value]
                ),
                key=(
                    f"pathway_mdc_selected_{module_set}_"
                    f"{pathway_mdc_resolution}"
                ),
            )
            maximum_top_pathways = min(60, len(pathway_options))
            top_pathways = pathway_selection_columns[1].slider(
                f"{resolution_plural} in heatmap",
                min_value=1,
                max_value=maximum_top_pathways,
                value=min(25, maximum_top_pathways),
                key=(
                    f"pathway_mdc_top_n_{module_set}_"
                    f"{pathway_mdc_resolution}"
                ),
            )
            render_plotly_chart(
                pathway_mdc_heatmap_figure(
                    pathway_summary,
                    scale=mdc_scale,
                    top_n=top_pathways,
                    selected_pathway_id=selected_pathway_id,
                    module_definition=module_set_label,
                    resolution=pathway_mdc_resolution,
                ),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            st.caption(
                "Each heatmap cell gives equal weight to every qualifying enriched module. "
                "At category and sub-category resolution, a module with several supporting "
                "pathways is counted once and uses its smallest component-matched KEGG FDR. "
                "On log2 scale it is the arithmetic mean log2 MDC; on raw scale it is the "
                "equivalent geometric mean MDC ratio. Cell n is the number of enriched modules; "
                "hover also reports the number of distinct supporting pathways."
            )
            selected_pathway_rows = pathway_group_rows.loc[
                pathway_group_rows["pathway_id"]
                .astype(str)
                .eq(selected_pathway_id)
            ].copy()
            render_plotly_chart(
                pathway_mdc_detail_figure(
                    selected_pathway_rows,
                    pathway_id=selected_pathway_id,
                    selected_module=module,
                    threshold=mdc_threshold,
                    scale=mdc_scale,
                    module_definition=module_set_label,
                    resolution=pathway_mdc_resolution,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            pathway_detail_columns = [
                "enrichment_resolution_label",
                "pathway_label",
                "category_level1",
                "category_level2",
                "supporting_pathway_count",
                "supporting_pathway_ids",
                "supporting_pathway_names",
                "supporting_subcategories",
                "best_supporting_pathway_id",
                "best_supporting_pathway_name",
                "module",
                "component_label",
                "enrichment_scope",
                "enrichment_p",
                "enrichment_fdr",
                "enrichment_region_a",
                "enrichment_fdr_region_a",
                "enrichment_region_b",
                "enrichment_fdr_region_b",
                "mdc",
                "log2_mdc",
                "direction",
                "directional_fdr",
                "n_edges",
            ]
            if pathway_mdc_resolution == "pathway":
                pathway_detail_columns[1:1] = ["pathway_id"]
                pathway_detail_columns.extend(["overlap_k", "overlap_genes"])
            selected_pathway_rows = selected_pathway_rows.sort_values(
                ["component_label", "directional_fdr", "enrichment_fdr"],
                kind="stable",
            )
            filterable_dataframe(
                selected_pathway_rows[pathway_detail_columns],
                table_key=f"pathway_mdc_detail_{pathway_mdc_resolution}",
                table_name=f"{resolution_singular}-resolved MDC detail",
                use_container_width=True,
                hide_index=True,
                height=480,
                column_config={
                    **(
                        {"overlap_genes": "Overlap gene symbols"}
                        if pathway_mdc_resolution == "pathway"
                        else {}
                    ),
                    "supporting_pathway_names": "Supporting pathway names",
                    "best_supporting_pathway_name": "Best-FDR supporting pathway",
                },
            )
            download_columns = [
                "enrichment_resolution",
                "enrichment_resolution_label",
                "pathway_id",
                "pathway_label",
                "category_level1",
                "category_level2",
                "component",
                "component_label",
                "enrichment_scope",
                "n_modules",
                "n_pathways",
                "mean_log2_mdc",
                "geometric_mean_mdc",
                "median_mdc",
                "minimum_enrichment_fdr",
                "median_enrichment_fdr",
                "n_mdc_significant",
                "proportion_mdc_significant",
                "minimum_mdc_fdr",
                "total_edges",
            ]
            download_columns = [
                column for column in download_columns if column in pathway_summary
            ]
            pathway_download_columns = st.columns(2)
            pathway_download_columns[0].download_button(
                f"Download {resolution_singular.lower()}-component summary (TSV)",
                data=dataframe_to_tsv_bytes(pathway_summary[download_columns]),
                file_name=(
                    f"{download_prefix}{pathway_mdc_resolution}_resolved_"
                    "MDC_summary.tsv"
                ),
                mime="text/tab-separated-values",
            )
            pathway_download_columns[1].download_button(
                f"Download selected {resolution_singular.lower()} module rows (TSV)",
                data=dataframe_to_tsv_bytes(
                    selected_pathway_rows[pathway_detail_columns]
                ),
                file_name=(
                    f"{download_prefix}selected_{pathway_mdc_resolution}_"
                    "regional_MDC_modules.tsv"
                ),
                mime="text/tab-separated-values",
            )

if active_view == "Statistics":
    st.subheader("Robust association statistics")
    if phenotype == "clusters":
        cluster_statistics = cached_module_set_cluster_associations(
            module_set, module_count, estimator, method, resolved, feature,
            tuple(selected_components), tuple(diagnoses), True, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        cluster_statistics = cluster_statistics.loc[
            cluster_statistics["module"].astype(int).eq(int(module))
        ].copy()
        cluster_statistics.insert(0, "module_definition", module_set_label)
        st.caption(
            "Nominal Cluster 1–4 statistics use Kruskal–Wallis and epsilon-squared. "
            "Clusters with fewer than five non-missing donors are excluded from the test "
            "but reported. FDR is BH across modules only; no Pearson/Spearman statistic is "
            "calculated from cluster codes."
        )
        filterable_dataframe(
            cluster_statistics,
            table_key="cluster_robust_statistics",
            table_name="Nominal cluster statistics",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download nominal cluster statistics (TSV)",
            data=dataframe_to_tsv_bytes(cluster_statistics),
            file_name=f"{download_prefix}M{module}_clusters_{feature}_{method}_statistics.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    st.caption(
        f"The primary table follows the sidebar selection: {correlation_method}. The "
        "download retains raw, asinh, Winsorized, Spearman, RINT, leave-one-out "
        "sensitivity, dependent CT-vs-TS tests, and all stored FDR fields. All-12 "
        "global FDR is the current primary correction; primary-five global FDR is "
        "retained for backward comparison and is unavailable for the seven added outcomes."
    )
    if resolved and correlation_method == "Spearman":
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "rho",
            "p_spearman",
            "q_spearman_all12_global",
            "q_spearman_primary5_global",
        ]
    elif resolved:
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "r_rint",
            "p_rint",
            "q_rint_all12_global",
            "q_rint_primary5_global",
            "q_rint_within_phenotype",
            "rho",
            "p_spearman",
            "loo_rint_max_delta",
            "loo_rint_sign_flip",
        ]
    elif correlation_method == "Spearman":
        display_columns = [
            "diagnosis_group",
            "n",
            "rho_CT",
            "p_spearman_CT",
            "q_spearman_CT_all12_global",
            "q_spearman_CT_primary5_global",
            "rho_TS",
            "p_spearman_TS",
            "q_spearman_TS_all12_global",
            "q_spearman_TS_primary5_global",
            "component_t_rank",
            "p_component_rank",
            "q_component_rank_within_phenotype",
            "q_component_rank_all12_global",
            "q_component_rank_primary5_global",
        ]
    else:
        display_columns = [
            "diagnosis_group",
            "n",
            "r_rint_CT",
            "p_rint_CT",
            "q_rint_CT_all12_global",
            "q_rint_CT_primary5_global",
            "r_rint_TS",
            "p_rint_TS",
            "q_rint_TS_all12_global",
            "q_rint_TS_primary5_global",
            "rho_CT",
            "rho_TS",
            "p_component_rint",
            "q_component_rint_within_phenotype",
            "q_component_rint_all12_global",
            "q_component_rint_primary5_global",
        ]
    statistics = statistics.copy()
    statistics.insert(0, "module_definition", module_set_label)
    filterable_dataframe(
        statistics[display_columns],
        table_key="robust_statistics",
        table_name="Robust statistics",
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download all selected robust-statistics columns (TSV)",
        data=dataframe_to_tsv_bytes(statistics),
        file_name=(
            f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
            "robust_statistics.tsv"
        ),
        mime="text/tab-separated-values",
    )
    st.info(fdr_text(module_manifest, module_count))

if active_view == "KEGG enrichment":
    st.subheader("Tissue-expanded KEGG enrichments")
    kegg_scope = st.radio(
        "Enrichment table scope",
        options=["Selected module", "All modules"],
        horizontal=True,
        key=f"kegg_scope_{module_set}",
        help=(
            "Selected module follows the module chosen in the sidebar. All modules "
            "loads every reported enrichment row for the selected module definition."
        ),
    )
    kegg_module = module if kegg_scope == "Selected module" else None
    source_kegg = cached_kegg(module_set, kegg_module).copy()
    if not source_kegg.empty:
        source_kegg.insert(0, "module_definition", module_set_label)

    if source_kegg.empty:
        st.info(
            f"{module_label(module)} has no row in the supplied tissue-expanded KEGG "
            "table. The module itself is still available in every plot and "
            "statistics view. Choose All modules to browse the reported enrichments."
        )
    else:
        st.markdown("#### Filters")
        filter_row_one = st.columns([1.5, 1.25, 1.1, 0.9])
        if kegg_scope == "All modules":
            selected_kegg_modules = filter_row_one[0].multiselect(
                "Modules",
                options=sorted(module_details["module"].astype(int).unique()),
                default=[],
                format_func=module_label,
                placeholder="All modules",
                key=f"kegg_modules_{module_set}",
                help="Leave empty to include all modules in this definition.",
            )
        else:
            selected_kegg_modules = [int(module)]
            filter_row_one[0].text_input(
                "Module",
                value=module_label(module),
                disabled=True,
                key=f"kegg_selected_module_{module_set}",
            )
        statistical_scope = filter_row_one[1].selectbox(
            "Statistical scope",
            options=[
                "All regions (expanded)",
                "Any individual region",
                "AC",
                "DLPFC",
                "PCG",
            ],
            key=f"kegg_statistical_scope_{module_set}_{kegg_scope}",
            help=(
                "Choose which FDR and significance columns control the significance "
                "and maximum-FDR filters."
            ),
        )
        significance_label = filter_row_one[2].selectbox(
            "Significance",
            options=["All rows", "FDR-significant only", "Not FDR-significant"],
            key=f"kegg_significance_{module_set}_{kegg_scope}",
        )
        maximum_fdr = filter_row_one[3].number_input(
            "Maximum FDR",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.3f",
            key=f"kegg_maximum_fdr_{module_set}_{kegg_scope}",
        )

        filter_row_two = st.columns([1.1, 1.3, 1.6])
        category_options = sorted(
            source_kegg["category_level1"].dropna().astype(str).unique()
        )
        selected_categories = filter_row_two[0].multiselect(
            "KEGG categories",
            options=category_options,
            default=[],
            placeholder="All categories",
            key=f"kegg_categories_{module_set}_{kegg_scope}",
        )
        subcategory_source = source_kegg
        if selected_categories:
            subcategory_source = source_kegg.loc[
                source_kegg["category_level1"].isin(selected_categories)
            ]
        subcategory_options = sorted(
            subcategory_source["category_level2"].dropna().astype(str).unique()
        )
        selected_subcategories = filter_row_two[1].multiselect(
            "KEGG sub-categories",
            options=subcategory_options,
            default=[],
            placeholder="All sub-categories",
            key=f"kegg_subcategories_{module_set}_{kegg_scope}",
        )
        search = filter_row_two[2].text_input(
            "Search pathways or genes",
            placeholder="Alzheimer, lipid, infection, APOE…",
            key=f"kegg_search_{module_set}_{kegg_scope}",
        )

        significance_value = {
            "All rows": "all",
            "FDR-significant only": "significant",
            "Not FDR-significant": "not_significant",
        }[significance_label]
        statistical_scope_columns = {
            "All regions (expanded)": (["significant"], ["fdr"]),
            "Any individual region": (
                ["significant_AC", "significant_DLPFC", "significant_PCGBA23"],
                ["fdr_AC", "fdr_DLPFC", "fdr_PCGBA23"],
            ),
            "AC": (["significant_AC"], ["fdr_AC"]),
            "DLPFC": (["significant_DLPFC"], ["fdr_DLPFC"]),
            "PCG": (["significant_PCGBA23"], ["fdr_PCGBA23"]),
        }
        significance_columns, fdr_columns = statistical_scope_columns[statistical_scope]
        shown_kegg = filter_kegg_enrichments(
            source_kegg,
            modules=selected_kegg_modules if kegg_scope == "All modules" else None,
            categories=selected_categories,
            subcategories=selected_subcategories,
            significance=significance_value,
            maximum_fdr=maximum_fdr,
            search=search,
            significance_columns=significance_columns,
            fdr_columns=fdr_columns,
        )

        significant_count = int(
            shown_kegg[significance_columns]
            .fillna(False)
            .astype(bool)
            .any(axis=1)
            .sum()
        )
        shown_module_count = int(shown_kegg["cluster_id"].nunique())
        source_module_count = int(source_kegg["cluster_id"].nunique())
        best_fdr = (
            shown_kegg[fdr_columns]
            .apply(pd.to_numeric, errors="coerce")
            .min(axis=1)
            .min()
            if not shown_kegg.empty
            else np.nan
        )
        kegg_metrics = st.columns(4)
        kegg_metrics[0].metric(
            "Rows shown", f"{len(shown_kegg):,} / {len(source_kegg):,}"
        )
        kegg_metrics[1].metric(
            "Modules shown",
            f"{shown_module_count:,} / "
            f"{module_count if kegg_scope == 'All modules' else 1:,}",
        )
        kegg_metrics[2].metric(
            f"Significant: {statistical_scope}", f"{significant_count:,}"
        )
        kegg_metrics[3].metric(
            f"Best FDR: {statistical_scope}",
            "NA" if pd.isna(best_fdr) else f"{best_fdr:.3e}",
        )

        if kegg_scope == "All modules":
            st.caption(
                f"The supplied KEGG table contains reported rows for "
                f"{source_module_count} of {module_count} modules in this definition. "
                "Modules without a pathway meeting the enrichment input thresholds have no "
                "table row."
            )
        if shown_kegg.empty:
            st.info("No KEGG enrichment rows match the current filters.")
        kegg_priority_columns = [
            "module_definition",
            "cluster_id",
            "pathway_name",
            "category_level1",
            "category_level2",
            "p",
            "fdr",
            "significant",
            "p_AC",
            "fdr_AC",
            "significant_AC",
            "p_DLPFC",
            "fdr_DLPFC",
            "significant_DLPFC",
            "p_PCGBA23",
            "fdr_PCGBA23",
            "significant_PCGBA23",
        ]
        kegg_priority_columns = [
            column for column in kegg_priority_columns if column in shown_kegg
        ]
        displayed_kegg = shown_kegg[
            kegg_priority_columns
            + [
                column
                for column in shown_kegg.columns
                if column not in kegg_priority_columns
            ]
        ]
        filterable_dataframe(
            displayed_kegg,
            table_key="kegg_enrichments",
            table_name="KEGG enrichments",
            use_container_width=True,
            hide_index=True,
            height=650,
            column_config={
                "cluster_id": st.column_config.NumberColumn("Module", format="M%d"),
                "p": st.column_config.NumberColumn(
                    "All regions p (expanded)", format="%.3e"
                ),
                "fdr": st.column_config.NumberColumn(
                    "All regions FDR (expanded)", format="%.3e"
                ),
                "significant": "All regions significant (expanded)",
                "p_AC": st.column_config.NumberColumn("AC p-value", format="%.3e"),
                "fdr_AC": st.column_config.NumberColumn("AC FDR", format="%.3e"),
                "significant_AC": "AC significant",
                "p_DLPFC": st.column_config.NumberColumn(
                    "DLPFC p-value", format="%.3e"
                ),
                "fdr_DLPFC": st.column_config.NumberColumn(
                    "DLPFC FDR", format="%.3e"
                ),
                "significant_DLPFC": "DLPFC significant",
                "p_PCGBA23": st.column_config.NumberColumn(
                    "PCG p-value", format="%.3e"
                ),
                "fdr_PCGBA23": st.column_config.NumberColumn(
                    "PCG FDR", format="%.3e"
                ),
                "significant_PCGBA23": "PCG significant",
                "overlap_genes": "Overlap gene symbols",
            },
        )
        st.download_button(
            "Download currently shown KEGG rows (TSV)",
            data=dataframe_to_tsv_bytes(shown_kegg),
            file_name=(
                f"{download_prefix}"
                + (
                    f"M{module}_filtered_tissue_expanded_KEGG.tsv"
                    if kegg_scope == "Selected module"
                    else "allmodules_filtered_tissue_expanded_KEGG.tsv"
                )
            ),
            mime="text/tab-separated-values",
        )
    st.download_button(
        "Download the complete tissue-expanded KEGG table (all modules)",
        data=cached_kegg_tsv(module_set),
        file_name=f"{download_prefix}method4_tissue_expanded_kegg_annotated.tsv",
        mime="text/tab-separated-values",
    )
    st.caption(
        "All-regions p/FDR uses the tissue-expanded enrichment test. AC, DLPFC, and "
        "PCG p-values use separate region-specific ORA tests against their corresponding "
        "expression backgrounds. Each regional FDR is Benjamini-Hochberg adjusted within "
        "that module and region across 350 KEGG pathways; pathways below the minimum "
        "overlap receive p=1. Values are supplied by the enrichment analyses and are not "
        "recalculated by this app."
    )

if active_view == "Module details":
    st.subheader("Level-4 module composition")
    st.caption(
        "Module size, represented tissues, per-tissue gene counts, and proportions from "
        "the level-4 SE2 module-details file. Tissue proportions use module size as the denominator."
    )
    st.markdown("#### Module landscape")
    module_type_scope = st.radio(
        "Module types to show",
        options=["Both", "CT only", "TS only"],
        horizontal=True,
        key=f"module_details_scope_{module_set}",
        help=(
            "This filter uses the CT/TS module classification in the selected level-4 "
            "SE2 module-details file and controls both charts below."
        ),
    )
    entropy_range = st.slider(
        "Normalized tissue-entropy range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        key=f"module_entropy_range_{module_set}",
        help=(
            "Zero is a single-tissue module; one is equal AC/DLPFC/PCG "
            "composition. This filter combines with the CT/TS filter."
        ),
    )
    if module_type_scope == "Both":
        chart_module_details = module_details.copy()
    else:
        selected_module_type = module_type_scope.split()[0]
        chart_module_details = module_details.loc[
            module_details["cluster_type"].astype(str).str.upper().eq(selected_module_type)
        ].copy()
    chart_module_details = chart_module_details.loc[
        chart_module_details["tissue_entropy_normalized"].between(
            entropy_range[0], entropy_range[1], inclusive="both"
        )
    ].copy()

    if chart_module_details.empty:
        st.warning(f"No {module_type_scope} modules are available in this definition.")
    else:
        landscape_metrics = st.columns(5)
        landscape_metrics[0].metric("Modules shown", f"{len(chart_module_details):,}")
        landscape_metrics[1].metric(
            "Median size", f"{chart_module_details['module_size'].median():,.0f} genes"
        )
        largest_module = chart_module_details.sort_values(
            ["module_size", "module"], ascending=[False, True]
        ).iloc[0]
        landscape_metrics[2].metric(
            "Largest module", f"M{int(largest_module['module'])}"
        )
        landscape_metrics[3].metric(
            "Largest size", f"{int(largest_module['module_size']):,} genes"
        )
        landscape_metrics[4].metric(
            "Median normalized entropy",
            f"{chart_module_details['tissue_entropy_normalized'].median():.3f}",
        )
        render_plotly_chart(
            module_size_distribution_figure(
                chart_module_details,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
            key=f"module_size_distribution_{module_set}_{module_type_scope}",
        )
        render_plotly_chart(
            module_region_composition_figure(
                chart_module_details,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
            },
            key=f"module_region_composition_{module_set}_{module_type_scope}",
        )
        render_plotly_chart(
            module_entropy_figure(
                chart_module_details,
                selected_module=module,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
            key=f"module_entropy_{module_set}_{module_type_scope}",
        )
        st.caption(
            "Total bar width is module size; colored segments are AC, DLPFC, and PCG "
            "gene counts. Modules are ordered largest to smallest. Hover for exact "
            "counts and proportions; drag or scroll to zoom. Entropy is the continuous "
            "tissue-mixing complement to the discrete CT/TS module classification: zero "
            "means one tissue only and one means equal representation of all three tissues."
        )
        landscape_download = chart_module_details.copy()
        landscape_download.insert(0, "module_definition", module_set_label)
        st.download_button(
            "Download modules shown in the landscape (TSV)",
            data=dataframe_to_tsv_bytes(landscape_download),
            file_name=(
                f"{download_prefix}module_landscape_"
                f"{module_type_scope.replace(' ', '_')}_entropy_filtered.tsv"
            ),
            mime="text/tab-separated-values",
        )

    detail_columns = [
        "module_definition",
        "module",
        "module_size",
        "cluster_type",
        "tissues",
        "n_tissues",
        "dominant_tissue",
        "tissue_entropy",
        "tissue_entropy_normalized",
        "n_genes_ac",
        "proportion_ac",
        "n_genes_dlpfc",
        "proportion_dlpfc",
        "n_genes_pcg",
        "proportion_pcg",
    ]
    module_details = module_details.copy()
    module_details.insert(0, "module_definition", module_set_label)
    st.markdown(f"#### Selected module: {module_label(module)}")
    filterable_dataframe(
        module_details.loc[module_details["module"].astype(int).eq(int(module)), detail_columns],
        table_key="selected_module_details",
        table_name="Selected module details",
        use_container_width=True,
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "tissue_entropy": st.column_config.NumberColumn(
                "Tissue entropy (bits)", format="%.4f"
            ),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                "Normalized tissue entropy", format="%.3f", min_value=0.0, max_value=1.0
            ),
            "n_genes_ac": st.column_config.NumberColumn("AC genes", format="%d"),
            "proportion_ac": st.column_config.NumberColumn("AC proportion", format="percent"),
            "n_genes_dlpfc": st.column_config.NumberColumn("DLPFC genes", format="%d"),
            "proportion_dlpfc": st.column_config.NumberColumn(
                "DLPFC proportion", format="percent"
            ),
            "n_genes_pcg": st.column_config.NumberColumn("PCG genes", format="%d"),
            "proportion_pcg": st.column_config.NumberColumn("PCG proportion", format="percent"),
        },
    )
    st.markdown(f"#### All {module_count} modules")
    filterable_dataframe(
        module_details[detail_columns],
        table_key="all_module_details",
        table_name="All module details",
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "tissue_entropy": st.column_config.NumberColumn(
                "Tissue entropy (bits)", format="%.4f"
            ),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                "Normalized tissue entropy", format="%.3f", min_value=0.0, max_value=1.0
            ),
            "n_genes_ac": st.column_config.NumberColumn("AC genes", format="%d"),
            "proportion_ac": st.column_config.NumberColumn("AC proportion", format="percent"),
            "n_genes_dlpfc": st.column_config.NumberColumn("DLPFC genes", format="%d"),
            "proportion_dlpfc": st.column_config.NumberColumn(
                "DLPFC proportion", format="percent"
            ),
            "n_genes_pcg": st.column_config.NumberColumn("PCG genes", format="%d"),
            "proportion_pcg": st.column_config.NumberColumn("PCG proportion", format="percent"),
        },
    )
    st.download_button(
        "Download complete module-details table (TSV)",
        data=dataframe_to_tsv_bytes(module_details[detail_columns]),
        file_name=f"{download_prefix}se2_level4_module_details.tsv",
        mime="text/tab-separated-values",
    )
    st.info(
        "A tissue with one module gene can contribute CT edges but has no within-tissue "
        "gene pair, so its tissue-specific feature is unavailable."
    )

if active_view == "Methods & data":
    st.subheader("Analysis methods and deploy data")
    st.markdown(
        f"**Selected module definition:** {module_set_label}. Module IDs are scoped to "
        "this definition; the same number in the other definition can contain a different "
        "set of tissue-gene members."
    )
    st.markdown(
        "**Standard LIONESS** scores each donor relative to the complete 450-donor "
        "reference network. **Control-referenced LIONESS** uses the 164 Controls as the "
        "reference and scores Controls by leave-one-out and MCI/AD by donor addition."
    )
    st.markdown(
        "**BONOBO** estimates one empirical-Bayes covariance network for each of the same "
        "450 donors. The donor/module delta is estimated automatically, correlations are "
        "converted with the same signedAlt β=3 TS and β=2 CT weighting, and the app can "
        "summarize all edges or edges passing the native posterior p<0.05 or within-"
        "donor-module BH FDR<0.05 rule. BONOBO edge significance tests nonzero covariance, "
        "not AD/MCI/Control differences."
    )
    formula_col1, formula_col2 = st.columns(2)
    with formula_col1:
        st.markdown("Standard donor network")
        st.latex(r"A_i = N A_{all} - (N-1)A_{all\\setminus i},\\quad N=450")
        st.markdown("Control donor")
        st.latex(r"A_i = n_C A_C - (n_C-1)A_{C\\setminus i},\\quad n_C=164")
    with formula_col2:
        st.markdown("MCI or AD donor added to the Control reference")
        st.latex(r"A_i = (n_C+1)A_{C+i} - n_C A_C")
        st.markdown(
            "Networks use signedAlt adjacency with β=3 for tissue-specific edges and "
            "β=2 for cross-tissue edges. Each undirected edge is counted once and the "
            "diagonal is excluded."
        )
    st.markdown(
        "For each estimator/method, module, feature, and CT/TS or resolved component, "
        "the app stores raw values, a robust-scale asinh transformation, and a rank "
        "inverse-normal Z-score calculated across all 450 donors."
    )
    st.markdown(
        "**Grouped associations** use diagnosis by default and may instead stratify "
        "correlations by clusters, CogDx, Braak, CERAD, ADNC, Parkinsonism, sex code, "
        "or APOE genotype. Pearson and Spearman are calculated separately within each "
        "eligible displayed level. Their BH FDR families contain only the 154 or 186 "
        "modules in the selected definition, while holding every other analysis and "
        "grouping field fixed; missing and constant tests are excluded from the denominator. "
        "OLS trend lines are descriptive guides. Categorical comparisons use Kruskal–Wallis "
        "and epsilon-squared and never correlate nominal codes."
    )
    st.markdown(
        "**ROSMAP clusters** are an unordered four-class donor partition available for "
        "313 of the 450 donors (Cluster 1/2/3/4: 168/71/44/30; 137 unavailable). "
        "Cluster association views use Kruskal–Wallis tests and epsilon-squared, require "
        "at least five donors per tested category, and apply BH across modules only within "
        "fixed analysis strata. The numeric class codes are never used for Pearson or "
        "Spearman correlation. Cluster prediction is exploratory because it reuses the "
        "diagnosis-derived tissue-neutral AD panel and cluster membership is strongly "
        "associated with diagnosis in this cohort."
    )
    st.markdown("#### AD–Control differential-edge filtering")
    st.markdown(
        "A diagnosis-and-sex-stratified discovery cohort (117 AD, 114 Control) defines "
        "the optional edge mask with two-sided Welch tests on raw signedAlt donor-edge "
        "weights. Every edge stores AD and Control means, AD-minus-Control difference, "
        "Hedges’ g, p-value, **global BH FDR**, and **per-module BH FDR**. Global BH is "
        "calculated across all tested edges within one module-definition, estimator, "
        "network-method, and discovery/validation family; per-module BH is calculated "
        "across all edges of one module. The sidebar FDR-scope and 0.05/0.10 cutoff "
        "choices control filtered features, associations, distributions, heatmaps, and "
        "edge summaries. The same scope and cutoff initialize the volcano view, where "
        "the display-only threshold can be changed without recalculating module scores. "
        "Selecting All edges continues to use the original unfiltered data."
    )
    st.markdown(
        "The default filtered evaluation uses 50 held-out AD and 50 held-out Control "
        "donors. MCI (n=119) is external to edge selection. Validation labels do not "
        "alter the discovery mask, although the underlying unsupervised donor-network "
        "estimation is not a completely external network validation."
    )

    st.markdown("#### Module differential connectivity")
    st.markdown(
        "All-edge MDC is an independent group-network comparison supplied as module-level context: "
        "mean absolute AD adjacency divided by mean absolute Control adjacency. The app "
        "shows total, pooled TS/CT, and six tissue-resolved ratios and their conservative "
        "directional permutation FDR on raw-ratio or log2 scales. Separate TS and CT "
        "scatter plots relate MDC to normalized Shannon tissue-mixing entropy. The "
        "pathway-resolved tab annotates module MDC with component-matched KEGG enrichment; "
        "it is not a newly recomputed edge-level pathway MDC. Its broader tissue-union "
        "AD/Control cohort and absence of MCI are stated in the MDC tab. When the sidebar "
        "selects AD–Control differential edges, MDC is recomputed on the matching estimator/"
        "method discovery mask for Global or Per-module BH at 0.05 or 0.10. That fixed-mask "
        "permutation analysis is post-selection and exploratory, not independent validation."
    )

    st.markdown("#### Module feature definitions")
    filterable_dataframe(
        cached_feature_definitions(),
        table_key="feature_definitions",
        table_name="Module feature definitions",
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Tissue labels")
    filterable_dataframe(
        cached_tissue_mapping(),
        table_key="tissue_labels",
        table_name="Tissue labels",
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Public deploy safeguards")
    st.markdown(
        "The deploy Parquet files do not contain `projid`, the source donor identifier, "
        "or a donor-to-pseudonym key. Each point has a random-salted pseudonymous sample "
        "label that is stable within this bundle only. Diagnosis, the five primary phenotypes, "
        "and selected age, education, APOE, cognitive diagnosis, neuropathology, Parkinsonism, "
        "and nominal cluster membership "
        "fields are included for color, hover, and correlation views. Public release still "
        "requires confirmation that the governing ROSMAP data-use agreement permits sharing "
        "these donor-level derived and metadata values."
    )
    st.caption(
        "Sex is displayed as source Code 0/1 because this deploy bundle does not infer a label "
        "without an accompanying reviewed data dictionary."
    )
    manifest_path = ensure_data_path(DATA_DIR / "data_manifest.json")
    if manifest_path.is_file():
        with st.expander("Deploy data manifest"):
            st.json(manifest_path.read_text(encoding="utf-8"))

st.markdown(
    '<p class="app-note">Exploratory research interface. Associations are not causal, and '
    "nominal p-values should be interpreted alongside FDR and leave-one-out sensitivity.</p>",
    unsafe_allow_html=True,
)
