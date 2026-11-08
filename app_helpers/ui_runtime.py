"""Public explorer for ROSMAP LIONESS and BONOBO module networks."""

from __future__ import annotations

from app_helpers.figure_export import render_exportable_plot as render_plotly_chart


import hashlib
import importlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from app_helpers import data as _data_helpers

if (
    getattr(_data_helpers, "PATHOLOGY_OUTCOME_API_VERSION", 0) < 1
    or getattr(_data_helpers, "MANIFEST_CONTEXT_API_VERSION", 0) < 1
):
    _data_helpers = importlib.reload(_data_helpers)
from app_helpers.pathology import PATHOLOGY_OUTCOMES
from app_helpers.camel import score_framework, CAMEL_EXPANSION, VALIDATION_SCOPE
from app_helpers import camel_validation as _camel_validation

# A deployed entrypoint can be refreshed while Streamlit still holds the old
# validation reader. Refresh its complete API before routing any dataset;
# otherwise even the original-cohort default can fail before rendering a page.
if (
    getattr(_camel_validation, "VALIDATION_DATASET_API_VERSION", 0) < 1
    or not hasattr(_camel_validation, "DATASETS")
    or not all(
        callable(getattr(_camel_validation, name, None))
        for name in ("supports_dataset", "for_dataset", "available", "manifest")
    )
):
    _camel_validation = importlib.reload(_camel_validation)

# Streamlit Community Cloud may hot-reload this entrypoint while retaining an older
# app_helpers.charts module in memory. Reload the helper when its current public API
# is absent so a repository update cannot leave the two modules out of sync.
from app_helpers import charts as _chart_helpers

if (
    getattr(_chart_helpers, "DISTRIBUTION_GROUPING_API_VERSION", 0) < 3
    or getattr(_chart_helpers, "PUBLICATION_EXPORT_API_VERSION", 0) < 1
    or getattr(_chart_helpers, "ASSOCIATION_STYLE_API_VERSION", 0) < 1
    or getattr(_chart_helpers, "MODULE_COMPOSITION_API_VERSION", 0) < 2
    or not all(
        hasattr(_chart_helpers, name)
        for name in (
            "CONTINUOUS_COLOR_SCALES",
            "categorical_association_figure",
            "EDGE_COMPONENT_LABELS",
            "edge_volcano_figure",
            "module_finder_figure",
            "mdc_entropy_figure",
            "mdc_normalization_comparison_figure",
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
            "DISTRIBUTION_GROUPING_API_VERSION",
            "distribution_omnibus_component_figure",
            "distribution_pairwise_forest_figure",
            "distribution_pairwise_heatmap_figure",
            "distribution_module_ranking_figure",
            "distribution_feature_heatmap_figure",
            "COEFFICIENT_ANNOTATION_API_VERSION",
            "KEGG_REGIONAL_SHARING_API_VERSION",
        )
    )
):
    _chart_helpers = importlib.reload(_chart_helpers)

from app_helpers.charts import (
    donor_edge_composition_figure,
    donor_edge_difference_figure,
    donor_edge_driver_figure,
    donor_edge_scatter_figure,
    endpoint_expression_2d_figure,
    endpoint_expression_3d_figure,
    expression_residual_figure,
    partition_association_concordance_figure,
    partition_entropy_figure,
    partition_mdc_concordance_figure,
    partition_overlap_figure,
    partition_proportion_figure,
    partition_size_figure,
    CONTINUOUS_COLOR_SCALES,
    EDGE_COMPONENT_LABELS,
    aggregate_to_long,
    categorical_association_figure,
    cluster_association_heatmap_figure,
    classification_diagnostic_rows,
    correlation_heatmap_figure,
    clustered_correlation_group_order,
    distribution_figure,
    distribution_feature_heatmap_figure,
    distribution_module_ranking_figure,
    distribution_omnibus_component_figure,
    distribution_pairwise_forest_figure,
    distribution_pairwise_heatmap_figure,
    distribution_summary,
    edge_volcano_figure,
    edge_summary_figure,
    mdc_entropy_figure,
    mdc_module_figure,
    mdc_overview_figure,
    mdc_resolved_heatmap_figure,
    mdc_resolved_module_figure,
    grouped_association_figure,
    kegg_region_combination_figure,
    kegg_region_heatmap_figure,
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

if getattr(_correlation_helpers, "GROUPED_ASSOCIATION_API_VERSION", 0) < 2:
    _correlation_helpers = importlib.reload(_correlation_helpers)

from app_helpers.correlations import (
    add_across_module_fdr,
    add_categorical_across_module_fdr,
    benjamini_hochberg,
    calculate_categorical_associations,
    calculate_correlations,
)
from app_helpers.distributions import (
    calculate_pairwise_distribution_statistics,
    default_reference_level,
    distribution_contrasts,
)
from app_helpers.module_finder import (
    FINDER_CRITERIA,
    build_module_finder_table,
    build_pooled_ct_ts_statistics,
)
from app_helpers.table_controls import filterable_dataframe
from app_helpers.streamlit_compat import (
    image as render_streamlit_image,
    stateful_tabs,
    tab_is_open,
)
from app_helpers import regional_modules as _regional_helpers

if getattr(_regional_helpers, "REGIONAL_API_VERSION", 0) < 4:
    _regional_helpers = importlib.reload(_regional_helpers)

from app_helpers.regional_modules import (
    REGIONAL_ANALYSIS_VIEWS,
    render_regional_analysis_view,
)
from app_helpers.protocol_v11 import (
    protocol_v11_available,
    render_protocol_v11_view,
)
from app_helpers import streaming_associations as _streaming_helpers

if getattr(
    _streaming_helpers, "STREAMING_ASSOCIATION_API_VERSION", 0
) < 2 or not hasattr(_streaming_helpers, "stream_pairwise_distribution_associations"):
    _streaming_helpers = importlib.reload(_streaming_helpers)

from app_helpers.streaming_associations import (
    stream_categorical_associations,
    stream_diagnosis_categorical_associations,
    stream_grouped_correlation_matrix,
    stream_grouped_correlations,
    stream_pairwise_distribution_associations,
    stream_pooled_correlations,
)
from app_helpers.drive_data import data_source_label, ensure_data_path
from app_helpers.edge_expression import (
    find_edge_triangles,
    fit_ols,
    gene_pathway_membership,
    rank_edges,
    scope_edges,
)
from app_helpers import data as _data_helpers

if (
    getattr(_data_helpers, "EFFECT_ENDPOINT_KEGG_API_VERSION", 0) < 1
    or getattr(_data_helpers, "MDC_NORMALIZATION_API_VERSION", 0) < 1
    or not all(
        hasattr(_data_helpers, name)
        for name in (
            "collapse_pathway_mdc_rows",
            "SCORE_TRANSFORM_LABELS",
            "EIGENGENE_SOURCE_LABELS",
            "ASSOCIATION_GROUP_LABELS",
            "association_level_label",
            "PREDICTION_BLOCK_ORDER",
            "EIGENGENE_FEATURE_LABELS",
            "descriptive_eigengene_data_available",
            "effect_size_data_available",
            "effect_rule_key",
            "effect_rule_label",
            "effect_mask_column",
            "load_effect_drivers",
            "load_effect_size_manifest",
            "EFFECT_STATISTIC_LABELS",
            "edge_expression_data_available",
            "load_edge_expression",
            "load_edge_expression_nodes",
            "annotate_prediction_coefficients",
            "coefficient_kegg_scope_lookup",
            "TARGETED_HEDGES_MASKS",
            "load_effect_endpoint_summary",
            "load_mdc_split_heatmap",
            "mdc_split_heatmap_available",
            "load_mdc_normalization_manifest",
            "load_mdc_normalization_summary",
        )
    )
):
    _data_helpers = importlib.reload(_data_helpers)

from app_helpers.data import (
    MUSCLE_MODULE_SETS,
    ANALYSIS_SUBSET_LABELS,
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_LEVEL_ORDERS,
    ASSOCIATION_OUTCOME_LABELS,
    BONOBO_EDGE_RULE_LABELS,
    BONOBO_FEATURE_LABELS,
    COLOR_LABELS,
    COHORT_SCOPE_LABELS,
    COMPONENT_ORDER,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    DATA_DIR,
    DIFFERENTIAL_EDGE_RULE_LABELS,
    EFFECT_DIRECTION_LABELS,
    EFFECT_STATISTIC_LABELS,
    EIGENGENE_FEATURE_LABELS,
    DIAGNOSIS_ORDER,
    EDGE_SCOPE_LABELS,
    FEATURE_LABELS,
    HOVER_LABELS,
    METHOD_LABELS,
    MDC_ENRICHMENT_RESOLUTION_LABELS,
    ESTIMATOR_LABELS,
    EIGENGENE_SOURCE_LABELS,
    FDR_SCOPE_LABELS,
    FDR_THRESHOLD_LABELS,
    HEDGES_FIXED_THRESHOLDS,
    MODULE_SET_LABELS,
    MODULE_SET_METHODS,
    MODULE_TOP_FRACTIONS,
    PREDICTION_MODULE_SETS,
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
    TARGETED_HEDGES_MASKS,
    TARGETED_PREDICTION_MODE_LABELS,
    TARGETED_TIER_LABELS,
    SCALE_LABELS,
    SCORE_TRANSFORM_LABELS,
    SCORE_NORMALIZATION_LABELS,
    SELECTABLE_ASSOCIATION_OUTCOMES,
    association_level_label,
    available_module_sets,
    build_pathway_mdc_rows,
    collapse_pathway_mdc_rows,
    association_kegg_subtitles,
    add_kegg_region_sharing,
    annotate_prediction_coefficients,
    coefficient_kegg_scope_lookup,
    dataframe_to_tsv_bytes,
    differential_data_available,
    effect_rule_key,
    effect_rule_label,
    effect_mask_column,
    effect_size_data_available,
    effect_endpoint_kegg_available,
    edge_expression_data_available,
    descriptive_eigengene_data_available,
    differential_mdc_data_available,
    filter_kegg_enrichments,
    filter_kegg_region_sharing,
    load_aggregate,
    load_aggregate_scope,
    load_aggregate_statistics,
    load_data_manifest,
    load_feature_definitions,
    load_edge_summaries,
    load_effect_drivers,
    load_effect_size_manifest,
    load_edge_expression,
    load_edge_expression_nodes,
    load_kegg,
    load_effect_endpoint_kegg,
    load_effect_endpoint_summary,
    load_kegg_tsv_bytes,
    load_cluster_association_statistics,
    load_mdc_summary,
    load_mdc_resolved,
    load_mdc_normalization_manifest,
    load_mdc_normalization_summary,
    load_mdc_split_heatmap,
    mdc_split_heatmap_available,
    load_module_annotations,
    load_module_details,
    load_partition_comparison_summary,
    load_partition_comparison_table,
    load_partition_module_mapping,
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
    module_set_components,
    module_set_tissue_detail_columns,
    module_set_tissues,
    partition_comparison_available,
    prediction_data_available,
    targeted_prediction_data_available,
    require_data_files,
    selected_annotation,
    selected_kegg_table_annotation,
    summarize_pathway_mdc_rows,
)

from app_helpers.ui_cache import cache_data


@cache_data(show_spinner=False)
def cached_annotations(module_set: str) -> pd.DataFrame:
    return load_module_annotations(module_set)


@cache_data(show_spinner=False)
def cached_module_details(module_set: str) -> pd.DataFrame:
    return load_module_details(module_set=module_set)


@cache_data(show_spinner=False)
def cached_partition_comparison_summary() -> dict[str, object]:
    return load_partition_comparison_summary()


@cache_data(show_spinner=False)
def cached_partition_module_mapping() -> pd.DataFrame:
    return load_partition_module_mapping()


@cache_data(show_spinner=False, max_entries=3)
def cached_partition_comparison_table(name: str) -> pd.DataFrame:
    return load_partition_comparison_table(name)


@cache_data(show_spinner=False, max_entries=6, ttl=1800)
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
        method,
        module,
        feature,
        module_set=module_set,
        estimator=estimator,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    )


@cache_data(show_spinner=False, max_entries=4, ttl=1800)
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    return load_resolved(
        method,
        module,
        feature,
        module_set=module_set,
        estimator=estimator,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        cohort_scope=cohort_scope,
    )


@cache_data(show_spinner=False, max_entries=3)
def cached_sample_metadata(
    module_set: str = "full_cohort",
    cohort_scope: str = "complete_450",
    metadata_schema_version: int = 2,
) -> pd.DataFrame:
    """Metadata schema includes pathology and nominal ROS/MAP study membership."""
    return load_sample_metadata(module_set=module_set, cohort_scope=cohort_scope)


@cache_data(show_spinner=False, max_entries=3, ttl=1800)
def cached_mdc_summary(
    module_set: str,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    normalization_variant: str = "original",
    data_revision: str = "",
) -> pd.DataFrame:
    return load_mdc_summary(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
        normalization_variant,
    )


@cache_data(show_spinner=False, max_entries=3, ttl=1800)
def cached_mdc_resolved(
    module_set: str,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    normalization_variant: str = "original",
    data_revision: str = "",
) -> pd.DataFrame:
    return load_mdc_resolved(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
        normalization_variant,
    )


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_pathway_mdc_rows(
    module_set: str,
    enrichment_fdr_threshold: float,
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    normalization_variant: str = "original",
    data_revision: str = "",
) -> pd.DataFrame:
    return build_pathway_mdc_rows(
        load_mdc_summary(
            module_set,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            normalization_variant,
        ),
        load_mdc_resolved(
            module_set,
            estimator,
            method,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            normalization_variant,
        ),
        load_kegg(module_set=module_set),
        enrichment_fdr_threshold=enrichment_fdr_threshold,
    )


@cache_data(show_spinner=False, max_entries=3, ttl=1800)
def cached_edge_summaries(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    return load_edge_summaries(
        estimator,
        method,
        module,
        module_set=module_set,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        cohort_scope=cohort_scope,
    )


@cache_data(show_spinner=False, max_entries=4)
def cached_effect_drivers(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    edge_rule: str,
    differential_edge_rule: str,
    sample_ids: tuple[str, ...],
) -> pd.DataFrame:
    return load_effect_drivers(
        module_set,
        estimator,
        method,
        module,
        edge_rule,
        differential_edge_rule,
        sample_ids,
    )


@cache_data(show_spinner=False, max_entries=3)
def cached_effect_size_manifest(module_set: str) -> dict[str, object]:
    return load_effect_size_manifest(module_set)


@cache_data(show_spinner=False)
def cached_data_manifest() -> dict[str, object]:
    """Read the deploy registry including the additive pathology catalog."""
    return load_data_manifest()


@cache_data(show_spinner=False, max_entries=4, ttl=1800)
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
        method,
        module,
        phenotype,
        feature,
        module_set=module_set,
        estimator=estimator,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
    )


@cache_data(
    show_spinner="Calculating pooled-donor CT–TS module statistics…",
    max_entries=2,
    ttl=1800,
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
    source = attach_metadata(source, module_set=module_set)
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        source = source.loc[source["ad_control_split"].eq(split_value)].copy()
    return build_pooled_ct_ts_statistics(source, phenotype=phenotype)


@cache_data(show_spinner=False, max_entries=4, ttl=1800)
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    return load_resolved_statistics(
        method,
        module,
        phenotype,
        feature,
        module_set=module_set,
        estimator=estimator,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
    )


@cache_data(show_spinner=False, max_entries=8)
def cached_volcano_candidates(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    differential_edge_rule: str = "all",
) -> pd.DataFrame:
    return load_volcano_candidates(
        module_set, estimator, method, module, differential_edge_rule
    )


@cache_data(show_spinner=False, max_entries=8)
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


@cache_data(show_spinner=False, max_entries=3)
def cached_edge_expression_candidates(
    module_set: str, estimator: str, method: str, module: int
) -> pd.DataFrame:
    """Union the bounded significance- and effect-oriented exact edge catalogs."""

    parts = [load_volcano_candidates(module_set, estimator, method, module, "all")]
    if effect_size_data_available(module_set, estimator, method):
        effect_rule = effect_rule_key("hedges_g", "fixed", 0.20, "either")
        parts.insert(
            0,
            load_volcano_candidates(module_set, estimator, method, module, effect_rule),
        )
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values("edge_index", kind="stable")
        .drop_duplicates("edge_index", keep="first")
        .reset_index(drop=True)
    )


@cache_data(show_spinner=False, max_entries=4)
def cached_edge_expression_nodes(module_set: str, module: int) -> pd.DataFrame:
    return load_edge_expression_nodes(module_set, module)


@cache_data(show_spinner=False, max_entries=4)
def cached_edge_expression(
    module_set: str, module: int, node_indices: tuple[int, ...]
) -> pd.DataFrame:
    return load_edge_expression(module_set, module, node_indices)


@cache_data(show_spinner=False, max_entries=8)
def cached_kegg(module_set: str, module: int | None) -> pd.DataFrame:
    return load_kegg(module, module_set=module_set)


@cache_data(show_spinner=False, max_entries=4, ttl=1800)
def cached_effect_endpoint_kegg(
    module_set: str,
    method: str,
    direction: str,
    module: int | None,
) -> pd.DataFrame:
    return load_effect_endpoint_kegg(
        module_set=module_set,
        method=method,
        direction=direction,
        module=module,
    )


@cache_data(show_spinner=False, max_entries=4, ttl=1800)
def cached_effect_endpoint_summary(
    module_set: str,
    method: str,
    direction: str,
    module: int | None,
) -> pd.DataFrame:
    return load_effect_endpoint_summary(
        module_set=module_set,
        method=method,
        direction=direction,
        module=module,
    )


@cache_data(show_spinner=False, max_entries=3)
def cached_coefficient_kegg_scopes(module_set: str) -> pd.DataFrame:
    return coefficient_kegg_scope_lookup(load_kegg(module_set=module_set))


@cache_data(show_spinner=False)
def cached_kegg_tsv(module_set: str) -> bytes:
    return load_kegg_tsv_bytes(module_set)


@cache_data(show_spinner=False, max_entries=8)
def cached_effect_endpoint_kegg_tsv(
    module_set: str,
    method: str,
    direction: str,
) -> bytes:
    return load_kegg_tsv_bytes(
        module_set,
        source="effect_endpoint",
        method=method,
        direction=direction,
    )


@cache_data(show_spinner=False)
def cached_feature_definitions() -> pd.DataFrame:
    return load_feature_definitions()


@cache_data(show_spinner=False, max_entries=3, ttl=1800)
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
        reference_provenance,
        module_definition,
        network_method,
        predictor_design,
        edge_mask,
        score_normalization,
        outcome,
    )


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
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
        reference_provenance,
        module_definition,
        network_method,
        predictor_design,
        edge_mask,
        score_normalization,
        outcome,
    )


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_prediction_curves(**filters: str | None) -> pd.DataFrame:
    return load_prediction_curves(**filters)


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_prediction_confusion(**filters: str | None) -> pd.DataFrame:
    return load_prediction_confusion(**filters)


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_prediction_coefficients(**filters: str | None) -> pd.DataFrame:
    return load_prediction_coefficients(**filters)


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_prediction_diagnostics(**filters: str | None) -> pd.DataFrame:
    return load_prediction_diagnostics(**filters)


@cache_data(show_spinner=False, max_entries=2, ttl=1800)
def cached_prediction_whole_network(**filters: str | None) -> pd.DataFrame:
    return load_prediction_whole_network_features(**filters)


@cache_data(show_spinner=False, max_entries=3, ttl=1800)
def cached_targeted_prediction_table(
    table: str,
    filters: tuple[tuple[str, object], ...] = (),
) -> pd.DataFrame:
    return load_targeted_prediction_table(table, **dict(filters))


@cache_data(show_spinner=False)
def cached_tissue_mapping() -> pd.DataFrame:
    return load_tissue_mapping()


def attach_metadata(
    frame: pd.DataFrame,
    *,
    module_set: str = "full_cohort",
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    metadata = cached_sample_metadata(module_set, cohort_scope)
    additional = [
        column
        for column in metadata.columns
        if column == "sample_id" or column not in frame.columns
    ]
    return frame.merge(
        metadata[additional], on="sample_id", how="left", validate="many_to_one"
    )


def _edge_expression_mask(
    frame: pd.DataFrame,
    differential_edge_rule: str,
    fdr_scope: str,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Apply the sidebar edge rule to exact rows used by expression plots."""

    selected = frame.copy()
    if differential_edge_rule == "all":
        return selected
    if differential_edge_rule.startswith("ad_control_discovery_effect__"):
        membership = effect_mask_column(differential_edge_rule)
        if membership not in selected:
            return selected.iloc[0:0].copy()
        selected = selected.loc[selected[membership].fillna(False)].copy()
        direction = differential_edge_rule.rsplit("__", 1)[-1]
        if direction == "ad_higher":
            selected = selected.loc[selected["discovery_mean_difference"].gt(0)]
        elif direction == "control_higher":
            selected = selected.loc[selected["discovery_mean_difference"].lt(0)]
        return selected
    probability = f"discovery_fdr_{fdr_scope}"
    if probability not in selected:
        return selected.iloc[0:0].copy()
    return selected.loc[
        pd.to_numeric(selected[probability], errors="coerce").lt(float(fdr_threshold))
    ].copy()


def _edge_expression_color_kwargs(
    frame: pd.DataFrame,
    color_by: str,
    continuous_colorscale: str,
    reverse_colorscale: bool,
) -> dict[str, object]:
    """Return consistent Plotly Express color arguments for endpoint plots."""

    if color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES:
        if color_by == "diagnosis_group":
            return {
                "color_discrete_map": {
                    "Control": "#2C7FB8",
                    "MCI": "#D8A500",
                    "AD": "#E66101",
                }
            }
        if color_by == "clusters":
            return {
                "color_discrete_map": {
                    "Cluster 1": "#3B4CC0",
                    "Cluster 2": "#20A486",
                    "Cluster 3": "#F6C141",
                    "Cluster 4": "#D1495B",
                }
            }
        return {}
    colorscale = CONTINUOUS_COLOR_SCALES.get(
        continuous_colorscale, CONTINUOUS_COLOR_SCALES["Blue–white–orange"]
    )
    if reverse_colorscale:
        if isinstance(colorscale, str):
            colorscale = f"{colorscale}_r"
        else:
            colorscale = [
                [1.0 - float(position), color]
                for position, color in reversed(colorscale)
            ]
    return {"color_continuous_scale": colorscale}


def _expression_ols_fits(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    fit_scope: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit pooled or diagnosis-stratified endpoint-expression OLS models."""

    groups: list[tuple[str, pd.DataFrame]]
    if fit_scope == "pooled":
        groups = [("All displayed donors", frame)]
    else:
        groups = [
            (diagnosis, frame.loc[frame["diagnosis_group"].eq(diagnosis)].copy())
            for diagnosis in DIAGNOSIS_ORDER
            if frame["diagnosis_group"].eq(diagnosis).any()
        ]
    fits: dict[str, object] = {}
    summaries: list[dict[str, object]] = []
    coefficients: list[pd.DataFrame] = []
    for label, group in groups:
        fit = fit_ols(group, response, predictors, minimum_n=len(predictors) + 4)
        if fit is None:
            continue
        fits[label] = fit
        summaries.append(
            {
                "fit_group": label,
                "n": fit.n,
                "r_squared": fit.r_squared,
                "adjusted_r_squared": fit.adjusted_r_squared,
                "rmse": fit.rmse,
                "residual_sd": fit.residual_sd,
            }
        )
        coefficient = fit.coefficients.copy()
        coefficient.insert(0, "fit_group", label)
        coefficients.append(coefficient)
    return (
        fits,
        pd.DataFrame(summaries),
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
    )


def _render_expression_residuals(
    fits: dict[str, object],
    *,
    key: str,
) -> None:
    """Render residual diagnostics and compact OLS tables."""

    diagnostics = []
    for label, fit in fits.items():
        part = fit.diagnostics.copy()
        part["fit_group"] = label
        diagnostics.append(part)
    if not diagnostics:
        st.info(
            "OLS is unavailable because too few complete points or a constant axis remain."
        )
        return
    residuals = pd.concat(diagnostics, ignore_index=True)
    residual_figure = expression_residual_figure(residuals)
    render_plotly_chart(residual_figure, key=f"{key}_residuals")


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
        result["spearman_rho"] = result.get(f"rho_{scale}", result["rho"])
        result["spearman_p"] = result.get(f"p_spearman_{scale}", result["p_spearman"])
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
            if f"n_{component}" in part:
                part["n"] = part[f"n_{component}"].fillna(part["n"])
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


@cache_data(
    show_spinner="Calculating categorical associations across modules…",
    max_entries=2,
    ttl=1800,
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
    cohort_scope: str = "complete_450",
    category_variable: str = "clusters",
) -> pd.DataFrame:
    """Return Kruskal/epsilon-squared rows; never correlate cluster codes."""

    selected_groups = list(diagnoses)
    if include_pooled:
        selected_groups.append("All donors")
    if (
        category_variable == "clusters"
        and feature != "eigengene"
        and differential_edge_rule == "all"
        and cohort_scope == "complete_450"
    ):
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
            module_ids,
            cached_sample_metadata(module_set, cohort_scope),
            module_set=module_set,
            estimator=estimator,
            method=method,
            resolved=resolved,
            feature=feature,
            category_variable=category_variable,
            scale="raw",
            components=components,
            diagnoses=diagnoses,
            include_pooled=include_pooled,
            min_group_n=5,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
            analysis_subset=analysis_subset,
            cohort_scope=cohort_scope,
        )
        result = add_categorical_across_module_fdr(
            result,
            family_columns=["metric_family", "component", "diagnosis_group", "outcome"],
        )
    families = result.groupby(
        ["component", "diagnosis_group"], observed=True, sort=False
    )["module"].nunique()
    if (
        feature != "eigengene"
        and not families.empty
        and not families.eq(int(module_count)).all()
    ):
        raise ValueError(
            "Nominal association families do not contain every module in the selected "
            f"definition ({module_count} expected): {families.to_dict()}"
        )
    return result


@cache_data(
    show_spinner="Calculating association FDR across the selected module set…",
    max_entries=2,
    ttl=1800,
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Return Pearson/Spearman statistics with BH applied across modules only."""

    if phenotype in {"clusters", "study"}:
        return cached_module_set_cluster_associations(
            module_set,
            module_count,
            estimator,
            method,
            resolved,
            feature,
            components,
            diagnoses,
            include_pooled,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
            category_variable=phenotype,
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
        statistic_arguments["cohort_scope"] = cohort_scope
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

    if cohort_scope == VALIDATION_SCOPE and set(diagnoses) == set(
        cached_sample_metadata(module_set, cohort_scope).diagnosis_group.dropna()
    ):
        pooled = standardize_stored_associations(
            stored,
            resolved=resolved,
            scale=scale,
            components=components,
            phenotype=phenotype,
        )
        pooled = pooled.loc[pooled.diagnosis_group.eq("All donors")].copy()
        pooled = add_across_module_fdr(
            pooled, family_columns=["component", "diagnosis_group", "outcome"]
        )
        return pd.concat([group_statistics, pooled], ignore_index=True)

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    pooled_statistics = stream_pooled_correlations(
        module_ids,
        cached_sample_metadata(module_set, cohort_scope),
        module_set=module_set,
        estimator=estimator,
        method=method,
        resolved=resolved,
        feature=feature,
        component=None,
        outcomes=(phenotype,),
        scale=scale,
        diagnoses=diagnoses,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
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


@cache_data(
    show_spinner="Calculating grouped associations across the selected module set…",
    max_entries=1,
    ttl=1800,
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Return grouped Pearson/Spearman statistics with module-set BH FDR.

    Diagnosis grouping reuses the compact stored correlation catalog for network
    features. Eigengenes and other grouping variables are calculated lazily from
    one feature/resolution scope. Every cache-key field fixes a statistical family
    or donor filter.
    """

    min_group_n = int(min_group_n)
    if (
        grouping_variable == "diagnosis_group"
        and feature != "eigengene"
        and (cohort_scope != VALIDATION_SCOPE or min_group_n == 10)
    ):
        result = cached_module_set_associations(
            module_set,
            module_count,
            estimator,
            method,
            resolved,
            feature,
            phenotype,
            scale,
            components,
            tuple(str(value) for value in grouping_levels),
            include_pooled,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
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
            result["n"].lt(min_group_n),
            f"n < {min_group_n}",
            np.where(result["eligible"], "", "constant or unavailable values"),
        )
        return result

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    result = stream_grouped_correlations(
        module_ids,
        cached_sample_metadata(module_set, cohort_scope),
        module_set=module_set,
        estimator=estimator,
        method=method,
        resolved=resolved,
        feature=feature,
        phenotype=phenotype,
        scale=scale,
        components=components,
        diagnoses=diagnoses,
        grouping_variable=grouping_variable,
        grouping_levels=grouping_levels,
        include_pooled=include_pooled,
        min_group_n=min_group_n,
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
    )
    if not result.empty:
        result = add_across_module_fdr(
            result,
            family_columns=[
                "component",
                "outcome",
                "grouping_variable",
                "grouping_level",
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
        family_sizes = (
            result.loc[~result["is_pooled"]]
            .groupby(
                ["component", "outcome", "grouping_variable", "grouping_level"],
                observed=True,
                dropna=False,
            )["module"]
            .nunique()
        )
        if (
            feature != "eigengene"
            and not family_sizes.empty
            and not family_sizes.eq(int(module_count)).all()
        ):
            raise ValueError(
                "Grouped association families do not contain every module in the "
                f"selected definition ({module_count} expected): {family_sizes.to_dict()}"
            )
    return result


@cache_data(
    show_spinner="Calculating categorical associations across the selected module set…",
    max_entries=1,
    ttl=1800,
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Return a generic Kruskal–Wallis/epsilon-squared module-set catalog."""

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    feature_filter = None if feature == "__all__" else feature
    result = stream_categorical_associations(
        module_ids,
        cached_sample_metadata(module_set, cohort_scope),
        module_set=module_set,
        estimator=estimator,
        method=method,
        resolved=resolved,
        feature=feature_filter,
        category_variable=category_variable,
        scale=scale,
        components=components,
        diagnoses=diagnoses,
        category_levels=category_levels,
        min_group_n=int(min_group_n),
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
    )
    if result.empty:
        return result
    result["category_variable"] = category_variable
    result = add_categorical_across_module_fdr(
        result,
        family_columns=["metric_family", "component", "outcome", "category_variable"],
    )
    family_sizes = result.groupby(
        ["metric_family", "component", "outcome", "category_variable"],
        observed=True,
        sort=False,
    )["module"].nunique()
    network_family_sizes = family_sizes.loc[
        family_sizes.index.get_level_values("metric_family") != "eigengene"
    ]
    if (
        feature != "eigengene"
        and not network_family_sizes.empty
        and not network_family_sizes.eq(int(module_count)).all()
    ):
        raise ValueError(
            "Categorical association families do not contain every module in the "
            f"selected definition ({module_count} expected): "
            f"{network_family_sizes.to_dict()}"
        )
    return result


@cache_data(
    show_spinner="Calculating pairwise distribution effects across modules…",
    max_entries=1,
    ttl=1800,
)
def cached_distribution_pairwise_associations(
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    category_variable: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    category_levels: tuple[object, ...],
    contrasts: tuple[tuple[object, object], ...],
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Return pairwise effects with BH applied separately per module contrast."""

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    return stream_pairwise_distribution_associations(
        module_ids,
        cached_sample_metadata(module_set, cohort_scope),
        module_set=module_set,
        estimator=estimator,
        method=method,
        resolved=resolved,
        feature=feature,
        category_variable=category_variable,
        scale=scale,
        components=components,
        diagnoses=diagnoses,
        category_levels=category_levels,
        contrasts=contrasts,
        min_group_n=int(min_group_n),
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
    )


@cache_data(
    show_spinner="Bootstrapping selected-module effect intervals…",
    max_entries=2,
    ttl=1800,
)
def cached_selected_distribution_pairwise(
    frame: pd.DataFrame,
    category_variable: str,
    contrasts: tuple[tuple[object, object], ...],
    min_group_n: int,
    include_ks: bool,
) -> pd.DataFrame:
    """Return deterministic selected-module pairwise effects and bootstrap CIs."""

    return calculate_pairwise_distribution_statistics(
        frame,
        ["module", "metric_family", "component", "component_label"],
        category_column=category_variable,
        contrasts=contrasts,
        minimum_group_n=int(min_group_n),
        bootstrap_resamples=1000,
        seed=42,
        include_ks=include_ks,
    )


def add_correlation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["outcome_label"] = result["outcome"].map(OUTCOME_LABELS)
    result["feature_label"] = result["metric_family"].map(
        {**FEATURE_LABELS, **EIGENGENE_FEATURE_LABELS}
    )
    result["heatmap_row"] = result["feature_label"] + " · " + result["component_label"]
    return result


@cache_data(
    show_spinner="Calculating the selected module correlation matrix…",
    max_entries=2,
    ttl=1800,
)
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Selected-module matrix including the three verified pathology outcomes."""
    if resolved:
        source = load_resolved_scope(
            method,
            module=module,
            module_set=module_set,
            estimator=estimator,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
            cohort_scope=cohort_scope,
        )
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(
            method,
            module=module,
            module_set=module_set,
            estimator=estimator,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = aggregate_to_long(source, "rint")
    long = attach_metadata(long, module_set=module_set, cohort_scope=cohort_scope)
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


@cache_data(
    show_spinner="Calculating the selected module correlations by category…",
    max_entries=2,
    ttl=1800,
)
def cached_grouped_module_correlations(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    resolved: bool,
    diagnoses: tuple[str, ...],
    grouping_variable: str,
    grouping_levels: tuple[object, ...],
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Return all-outcome correlations, including pathology, within selected levels."""

    if resolved:
        source = load_resolved_scope(
            method,
            module=module,
            module_set=module_set,
            estimator=estimator,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
            cohort_scope=cohort_scope,
        )
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(
            method,
            module=module,
            module_set=module_set,
            estimator=estimator,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = aggregate_to_long(source, "rint")
    long = attach_metadata(long, module_set=module_set, cohort_scope=cohort_scope)
    long = long.loc[long["diagnosis_group"].isin(diagnoses)].copy()
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        long = long.loc[long["ad_control_split"].eq(split_value)].copy()
    if grouping_variable == "__all__":
        long["grouping_variable"] = "__all__"
        long["grouping_level"] = "__all__"
    else:
        long = long.loc[
            long[grouping_variable].notna()
            & long[grouping_variable].isin(grouping_levels)
        ].copy()
        long["grouping_variable"] = grouping_variable
        long["grouping_level"] = long[grouping_variable]
    summary = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "grouping_variable",
            "grouping_level",
        ],
        outcomes=NUMERIC_OUTCOMES,
        min_group_n=int(min_group_n),
    )
    if summary.empty:
        return summary
    # The selected-module displayed family stays independent for each level.
    for method_name in ("pearson", "spearman"):
        summary[f"{method_name}_fdr_displayed_family"] = summary.groupby(
            ["grouping_variable", "grouping_level"],
            observed=True,
            dropna=False,
        )[f"{method_name}_p"].transform(benjamini_hochberg)
    summary["grouping_label"] = summary["grouping_level"].map(
        lambda value: association_level_label(grouping_variable, value)
    )
    summary["diagnosis_group"] = summary["grouping_label"]
    return add_correlation_labels(summary)


@cache_data(
    show_spinner="Calculating grouped correlations across all modules…",
    max_entries=1,
    ttl=1800,
)
def cached_grouped_all_module_correlations(
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    component: str,
    diagnoses: tuple[str, ...],
    grouping_variable: str,
    grouping_levels: tuple[object, ...],
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """Memory-bounded grouped matrix including the three pathology outcomes."""

    module_ids = tuple(
        sorted(cached_annotations(module_set)["module"].astype(int).unique())
    )
    result = stream_grouped_correlation_matrix(
        module_ids,
        cached_sample_metadata(module_set, cohort_scope),
        module_set=module_set,
        estimator=estimator,
        method=method,
        resolved=resolved,
        feature=None if feature == "__all__" else feature,
        component=None if component == "__all__" else component,
        outcomes=tuple(NUMERIC_OUTCOMES),
        scale="rint",
        diagnoses=diagnoses,
        grouping_variable=grouping_variable,
        grouping_levels=grouping_levels,
        min_group_n=int(min_group_n),
        edge_rule=edge_rule,
        differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
        analysis_subset=analysis_subset,
        cohort_scope=cohort_scope,
    )
    if result.empty:
        return result
    result["grouping_label"] = result["grouping_level"].map(
        lambda value: association_level_label(grouping_variable, value)
    )
    result["diagnosis_group"] = result["grouping_label"]
    return add_correlation_labels(result)


@cache_data(
    show_spinner="Calculating correlations across all modules…",
    max_entries=1,
    ttl=1800,
)
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
    cohort_scope: str = "complete_450",
) -> pd.DataFrame:
    """All-module matrix including the additive pathology statistics catalog."""
    feature_filter = None if feature == "__all__" else feature
    component_filter = None if component == "__all__" else component
    summaries: list[pd.DataFrame] = []

    # The diagnosis-stratified statistics are already stored. Reading them avoids
    # loading all donor-level values just to reconstruct Control/MCI/AD summaries.
    include_eigengenes = resolved and feature_filter in {None, "eigengene"}
    include_stored_network = feature_filter != "eigengene"
    if diagnosis != "All donors" and include_stored_network:
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
            "diagnosis_group": (
                None if diagnosis == "All diagnosis groups" else diagnosis
            ),
        }
        if resolved:
            stored = load_resolved_statistics(
                **statistic_arguments,
                component=component_filter,
                cohort_scope=cohort_scope,
            )
            selected_components = tuple(
                sorted(
                    stored["component"].dropna().astype(str).unique(),
                    key=lambda value: (
                        (
                            COMPONENT_ORDER.index(value)
                            if value in COMPONENT_ORDER
                            else len(COMPONENT_ORDER)
                        ),
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
            stored_diagnoses = [*DIAGNOSIS_ORDER]
            if cohort_scope != "complete_450":
                stored_diagnoses.append("Unclassified")
            group_summary = group_summary.loc[
                group_summary["diagnosis_group"].isin(stored_diagnoses)
            ]
        summaries.append(group_summary)

    if diagnosis != "All donors" and include_eigengenes:
        module_ids = tuple(
            sorted(cached_annotations(module_set)["module"].astype(int).unique())
        )
        requested_diagnoses = (
            tuple(DIAGNOSIS_ORDER)
            if diagnosis == "All diagnosis groups"
            else (diagnosis,)
        )
        eigengene_summary = stream_grouped_correlation_matrix(
            module_ids,
            cached_sample_metadata(module_set, cohort_scope),
            module_set=module_set,
            estimator=estimator,
            method=method,
            resolved=True,
            feature="eigengene",
            component=component_filter,
            outcomes=tuple(NUMERIC_OUTCOMES),
            scale="rint",
            diagnoses=requested_diagnoses,
            grouping_variable="diagnosis_group",
            grouping_levels=requested_diagnoses,
            min_group_n=3,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
            analysis_subset=analysis_subset,
            cohort_scope=cohort_scope,
        )
        if not eigengene_summary.empty:
            eigengene_summary["diagnosis_group"] = eigengene_summary[
                "grouping_level"
            ].astype(str)
            summaries.append(eigengene_summary)

    # Pooled-donor correlations are intentionally calculated from the anonymous
    # donor rows because the original robustness tables are diagnosis-stratified.
    if diagnosis in {"All donors", "All diagnosis groups"}:
        module_ids = tuple(
            sorted(cached_annotations(module_set)["module"].astype(int).unique())
        )
        summaries.append(
            stream_pooled_correlations(
                module_ids,
                cached_sample_metadata(module_set, cohort_scope),
                module_set=module_set,
                estimator=estimator,
                method=method,
                resolved=resolved,
                feature=feature_filter,
                component=component_filter,
                outcomes=tuple(NUMERIC_OUTCOMES),
                scale="rint",
                diagnoses=tuple(
                    [*DIAGNOSIS_ORDER]
                    + (["Unclassified"] if cohort_scope != "complete_450" else [])
                ),
                edge_rule=edge_rule,
                differential_edge_rule=differential_edge_rule,
                differential_fdr_scope=differential_fdr_scope,
                differential_fdr_threshold=differential_fdr_threshold,
                score_normalization=score_normalization,
                analysis_subset=analysis_subset,
                cohort_scope=cohort_scope,
            )
        )

    if not summaries:
        return pd.DataFrame()
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
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "outcome",
        "outcome_label",
        "feature_label",
        "heatmap_row",
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


def refresh_helper_apis():
    """Keep entrypoint/helper hot reload compatible without clearing data caches."""
    checks = (
        (
            _chart_helpers,
            "COEFFICIENT_ANNOTATION_API_VERSION",
            1,
            ("CONTINUOUS_COLOR_SCALES",),
        ),
        (_correlation_helpers, "GROUPED_ASSOCIATION_API_VERSION", 2, ()),
        (
            _streaming_helpers,
            "STREAMING_ASSOCIATION_API_VERSION",
            2,
            ("stream_pairwise_distribution_associations",),
        ),
        (
            _camel_validation,
            "VALIDATION_DATASET_API_VERSION",
            1,
            ("DATASETS", "for_dataset", "supports_dataset"),
        ),
        (_data_helpers, "MANIFEST_CONTEXT_API_VERSION", 1, ()),
        (_regional_helpers, "REGIONAL_API_VERSION", 4, ()),
    )
    for module, version, minimum, exports in checks:
        if getattr(module, version, 0) < minimum or any(
            not hasattr(module, name) for name in exports
        ):
            importlib.reload(module)
            for name, value in tuple(globals().items()):
                if getattr(value, "__module__", None) == module.__name__ and hasattr(
                    module, name
                ):
                    globals()[name] = getattr(module, name)
            for name in exports:
                if name in globals():
                    globals()[name] = getattr(module, name)
