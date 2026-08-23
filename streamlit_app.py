"""Public explorer for ROSMAP LIONESS and BONOBO module networks."""

from __future__ import annotations

import hashlib
import importlib

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
        "edge_volcano_figure",
        "mdc_entropy_figure",
        "pathway_mdc_heatmap_figure",
    )
):
    _chart_helpers = importlib.reload(_chart_helpers)

from app_helpers.charts import (
    CONTINUOUS_COLOR_SCALES,
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
    distribution_summary,
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
    resolved_to_long,
)
from app_helpers.correlations import calculate_correlations
from app_helpers.drive_data import data_source_label, ensure_data_path
from app_helpers.data import (
    ANALYSIS_SUBSET_LABELS,
    BONOBO_EDGE_RULE_LABELS,
    BONOBO_FEATURE_LABELS,
    COLOR_LABELS,
    COMPONENT_ORDER,
    DATA_DIR,
    DIFFERENTIAL_EDGE_RULE_LABELS,
    DIAGNOSIS_ORDER,
    FEATURE_LABELS,
    HOVER_LABELS,
    METHOD_LABELS,
    ESTIMATOR_LABELS,
    FDR_SCOPE_LABELS,
    FDR_THRESHOLD_LABELS,
    MODULE_SET_LABELS,
    MODULE_SET_METHODS,
    NUMERIC_OUTCOMES,
    OUTCOME_LABELS,
    PHENOTYPE_LABELS,
    SCALE_LABELS,
    SCORE_NORMALIZATION_LABELS,
    build_pathway_mdc_rows,
    association_kegg_subtitles,
    dataframe_to_tsv_bytes,
    differential_data_available,
    filter_kegg_enrichments,
    load_aggregate,
    load_aggregate_scope,
    load_aggregate_statistics,
    load_data_manifest,
    load_feature_definitions,
    load_edge_summaries,
    load_kegg,
    load_kegg_tsv_bytes,
    load_mdc_summary,
    load_mdc_resolved,
    load_module_annotations,
    load_module_details,
    load_resolved,
    load_resolved_scope,
    load_resolved_statistics,
    load_sample_metadata,
    load_tissue_mapping,
    load_volcano_bins,
    load_volcano_candidates,
    module_label,
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


@st.cache_data(show_spinner=False, max_entries=48)
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


@st.cache_data(show_spinner=False, max_entries=48)
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


@st.cache_data(show_spinner=False)
def cached_mdc_summary(module_set: str) -> pd.DataFrame:
    return load_mdc_summary(module_set)


@st.cache_data(show_spinner=False)
def cached_mdc_resolved(module_set: str) -> pd.DataFrame:
    return load_mdc_resolved(module_set)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_pathway_mdc_rows(
    module_set: str, enrichment_fdr_threshold: float
) -> pd.DataFrame:
    return build_pathway_mdc_rows(
        load_mdc_summary(module_set),
        load_mdc_resolved(module_set),
        load_kegg(module_set=module_set),
        enrichment_fdr_threshold=enrichment_fdr_threshold,
    )


@st.cache_data(show_spinner=False, max_entries=32)
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


@st.cache_data(show_spinner=False, max_entries=96)
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


@st.cache_data(show_spinner=False, max_entries=48)
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


@st.cache_data(show_spinner=False, max_entries=48)
def cached_volcano_candidates(
    module_set: str, estimator: str, method: str, module: int
) -> pd.DataFrame:
    return load_volcano_candidates(module_set, estimator, method, module)


@st.cache_data(show_spinner=False, max_entries=48)
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


@st.cache_data(show_spinner=False, max_entries=48)
def cached_kegg(module_set: str, module: int | None) -> pd.DataFrame:
    return load_kegg(module, module_set=module_set)


@st.cache_data(show_spinner=False)
def cached_kegg_tsv(module_set: str) -> bytes:
    return load_kegg_tsv_bytes(module_set)


@st.cache_data(show_spinner=False)
def cached_feature_definitions() -> pd.DataFrame:
    return load_feature_definitions()


@st.cache_data(show_spinner=False)
def cached_tissue_mapping() -> pd.DataFrame:
    return load_tissue_mapping()


def attach_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = cached_sample_metadata()
    additional = [
        column for column in metadata.columns if column == "sample_id" or column not in frame.columns
    ]
    return frame.merge(metadata[additional], on="sample_id", how="left", validate="many_to_one")


def add_correlation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["outcome_label"] = result["outcome"].map(OUTCOME_LABELS)
    result["feature_label"] = result["metric_family"].map(FEATURE_LABELS)
    result["heatmap_row"] = result["feature_label"] + " · " + result["component_label"]
    return result


@st.cache_data(show_spinner="Calculating the selected module correlation matrix…", max_entries=24)
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


@st.cache_data(show_spinner="Calculating correlations across all modules…", max_entries=24)
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
    if resolved:
        source = load_resolved_scope(
            method,
            metric_family=feature,
            component=component,
            module_set=module_set,
            estimator=estimator,
            edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(
            method, metric_family=feature, module_set=module_set,
            estimator=estimator, edge_rule=edge_rule,
            differential_edge_rule=differential_edge_rule,
            differential_fdr_scope=differential_fdr_scope,
            differential_fdr_threshold=differential_fdr_threshold,
            score_normalization=score_normalization,
        )
        long = aggregate_to_long(source, "rint")
        long = long.loc[long["component"].eq(component)]
    long = attach_metadata(long)
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        long = long.loc[long["ad_control_split"].eq(split_value)]
    if diagnosis != "All donors":
        long = long.loc[long["diagnosis_group"].eq(diagnosis)]
    else:
        long = long.copy()
        long["diagnosis_group"] = "All donors"
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
    summary = add_correlation_labels(summary)
    summary["heatmap_row"] = summary["module"].map(lambda value: f"M{int(value)}")
    return summary


def readable_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fdr_text(module_manifest: dict[str, object], module_count: int) -> str:
    return (
        "Association FDR values use Benjamini–Hochberg correction separately by module "
        "definition, estimator/network method, aggregate versus resolved component family, "
        "correlation test, and BONOBO edge rule. Expanded global columns cover all 12 numeric "
        "outcomes; primary-five columns are retained for backward comparison, and "
        "within-outcome columns correct the selected outcome family. KEGG FDR is independent "
        "and comes from the supplied enrichment analysis."
    )


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
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
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
    mdc_summary = cached_mdc_summary(module_set)
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
        options=list(OUTCOME_LABELS),
        format_func=lambda value: OUTCOME_LABELS[value],
    )
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
    )
    diagnoses = st.multiselect(
        "Diagnosis groups",
        options=DIAGNOSIS_ORDER,
        default=DIAGNOSIS_ORDER,
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
        disabled=color_by == "diagnosis_group",
    )
    reverse_colorscale = st.checkbox(
        "Reverse continuous color scale",
        value=False,
        disabled=color_by == "diagnosis_group",
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
        statistics = cached_aggregate_stats(
            module_set, estimator, method, module, phenotype, feature, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        resolved = False
    else:
        plot_data = cached_resolved(
            module_set, estimator, method, module, feature, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
        )
        plot_data = resolved_to_long(plot_data, scale)
        statistics = cached_resolved_stats(
            module_set, estimator, method, module, phenotype, feature, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
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
    st.dataframe(
        tissue_composition,
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

view_options = [
    "Associations",
    "Feature distributions",
    "Correlation heatmaps",
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

if active_view == "Associations":
    st.subheader("Phenotype association")
    st.caption(
        f"Diagnosis-specific points with {correlation_method} correlation annotations and "
        "ordinary least-squares trend lines. Click a diagnosis "
        "in the legend to hide or show its points and trend together. Point shape identifies "
        "diagnosis; point color follows the selected color variable. Gray points have a missing "
        "value for a continuous color variable. The OLS lines are visual guides and do not "
        "change when Spearman is selected. Aggregate CT/TS panel subtitles use the "
        "tissue-expanded KEGG result. Tissue-resolved TS subtitles use the corresponding "
        "regional enrichment; CT-pair subtitles report both regions' FDRs for the pathway "
        "with the strongest conservative joint regional support. Those two FDRs are regional "
        "tests, not a newly combined pair-level p-value."
    )
    figure = association_figure(
        plot_data,
        statistics,
        phenotype=phenotype,
        phenotype_label=OUTCOME_LABELS[phenotype],
        feature_label=active_feature_labels[feature],
        scale=scale,
        scale_label=SCALE_LABELS[scale],
        diagnoses=diagnoses,
        module=module,
        resolved=resolved,
        color_by=color_by,
        color_label=COLOR_LABELS[color_by],
        hover_fields=HOVER_LABELS,
        correlation_method=correlation_method.lower(),
        module_definition=module_set_label,
        continuous_colorscale=continuous_colorscale,
        reverse_colorscale=reverse_colorscale,
        kegg_subtitles=association_subtitles,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": (
                    f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                    f"{correlation_method.lower()}"
                ),
                "scale": 3,
            },
        },
    )
    public_columns = [
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
    ]
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

if active_view == "Correlation heatmaps":
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
            f"All {module_count} modules: selected score",
        ],
        horizontal=True,
    )
    st.caption(f"Correlation method: **{correlation_method}** (controlled in the sidebar).")
    heatmap_diagnosis = st.selectbox(
        "Diagnosis in heatmap",
        options=["All donors", *DIAGNOSIS_ORDER],
        index=3,
        key="heatmap_diagnosis",
    )
    heatmap_clustering = st.selectbox(
        "Heatmap clustering",
        options=["None", "Rows", "Columns", "Rows and columns"],
        help=(
            "Reorders the selected axes using average-linkage hierarchical clustering "
            "of Euclidean distances between correlation profiles."
        ),
    )
    cluster_rows = heatmap_clustering in {"Rows", "Rows and columns"}
    cluster_columns = heatmap_clustering in {"Columns", "Rows and columns"}
    if correlation_method == "Pearson":
        value_column = "pearson_r"
        p_column = "pearson_p"
        fdr_column = "pearson_fdr_displayed_family"
    else:
        value_column = "spearman_rho"
        p_column = "spearman_p"
        fdr_column = "spearman_fdr_displayed_family"

    if heatmap_mode.startswith("Selected module"):
        correlation_table = cached_module_correlations(
            module_set, estimator, method, module, resolved, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization, analysis_subset,
        )
        heatmap_data = correlation_table.loc[
            correlation_table["diagnosis_group"].eq(heatmap_diagnosis)
        ].copy()
        component_rank = {value: index for index, value in enumerate(COMPONENT_ORDER)}
        feature_rank = {value: index for index, value in enumerate(FEATURE_LABELS)}
        row_order_frame = (
            heatmap_data[["metric_family", "component", "heatmap_row"]]
            .drop_duplicates()
            .assign(
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
            )
            .sort_values(["feature_rank", "component_rank", "component"])
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
            "Feature for all-module heatmap",
            options=list(active_feature_labels),
            format_func=lambda value: active_feature_labels[value],
            index=list(active_feature_labels).index(feature),
        )
        if resolved:
            component_options = available_components["component"].tolist()
            all_component = st.selectbox(
                "Component for all-module heatmap",
                options=component_options,
                format_func=lambda value: component_labels[value],
            )
        else:
            all_component = st.selectbox(
                "Component for all-module heatmap",
                options=["CT", "TS"],
                format_func=lambda value: f"{value} aggregate",
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
        heatmap_data = correlation_table.copy()
        module_order = sorted(heatmap_data["module"].astype(int).unique())
        row_order = [f"M{value}" for value in module_order]
        heatmap_title = (
            f"{module_set_label}: {active_feature_labels[all_feature]} · "
            f"{heatmap_data['component_label'].iloc[0]} · {heatmap_diagnosis}"
        )
        fdr_scope = (
            f"Displayed-family FDR is Benjamini–Hochberg correction across all "
            f"{module_count} modules "
            "and all numeric outcomes in this selected feature/component/diagnosis map."
        )

    correlation_table = correlation_table.copy()
    correlation_table.insert(0, "module_definition", module_set_label)

    correlation_heatmap = correlation_heatmap_figure(
        heatmap_data,
        value_column=value_column,
        p_column=p_column,
        fdr_column=fdr_column,
        title=heatmap_title,
        row_order=row_order,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
    )
    st.plotly_chart(
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
    correlation_columns = [
        "module_definition",
        "module",
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "outcome",
        "outcome_label",
        "n",
        "pearson_r",
        "pearson_p",
        "pearson_fdr_displayed_family",
        "spearman_rho",
        "spearman_p",
        "spearman_fdr_displayed_family",
    ]
    with st.expander("Complete correlation table", expanded=False):
        st.dataframe(
            correlation_table[correlation_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download complete correlation table (TSV)",
            data=dataframe_to_tsv_bytes(correlation_table[correlation_columns]),
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
    st.plotly_chart(
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
        st.dataframe(summary, use_container_width=True, hide_index=True)
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
    st.dataframe(
        screen[screen_columns].head(n_screen),
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
        st.plotly_chart(
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
        st.dataframe(edge_group_summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download diagnosis-group summary (TSV)",
            data=dataframe_to_tsv_bytes(edge_group_summary),
            file_name=f"{download_prefix}M{module}_{edge_metric}_group_summary.tsv",
            mime="text/tab-separated-values",
        )
        with st.expander("Donor-level edge-summary rows", expanded=False):
            st.dataframe(edge_data, use_container_width=True, hide_index=True, height=480)
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
        st.plotly_chart(
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
            "The density background includes every edge in the selected scope. Exact "
            "hoverable rows include all discovery/validation FDR≤0.10 or p≤0.05 edges "
            "plus the top 500 remaining edges per resolved component."
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
        st.dataframe(
            displayed_edges,
            use_container_width=True,
            hide_index=True,
            height=460,
            column_config={
                "gene_a": "Gene A symbol",
                "gene_b": "Gene B symbol",
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
    st.caption(
        "MDC compares the mean absolute signedAlt adjacency in AD with Control. "
        "Values above 1 indicate higher connectivity in AD; values below 1 indicate "
        "higher connectivity in Control. Total uses all edges, TS uses same-tissue edges, "
        "and CT uses cross-tissue edges."
    )
    mdc_metadata = module_manifest.get("mdc", data_manifest.get("mdc", {}))
    st.warning(
        "Cohort scope differs from the donor-complete LIONESS analysis. The MDC source "
        f"assembled {mdc_metadata.get('reference_assembled_donors', 517)} AD and "
        f"{mdc_metadata.get('target_assembled_donors', 408)} Control donors across the "
        "tissue union, including the 167 AD and 164 Control complete-three-tissue donors "
        "used here plus donors with partial tissue availability. MCI was not included. "
        "MDC is module-level context and does not change with the estimator or network-method selector."
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

    mdc_view = mdc_summary.copy()
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
        st.plotly_chart(
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
    st.plotly_chart(
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
    st.caption(
        (
            "Dashed zero lines separate AD-higher from Control-higher MDC. "
            if mdc_scale == "log2"
            else "Dashed equality lines at one separate AD-higher from Control-higher MDC. "
        )
        + "The dotted diagonal "
        f"marks equal TS and CT effects. {ct_unavailable_count} module(s) have no CT edges "
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
            st.plotly_chart(
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
    st.dataframe(
        displayed_mdc[mdc_columns],
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
        resolved_mdc = cached_mdc_resolved(module_set).copy()
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
            st.plotly_chart(
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
            st.plotly_chart(
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
            st.dataframe(
                resolved_mdc, use_container_width=True, hide_index=True, height=520
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
        pathway_control_columns = st.columns(3)
        with pathway_control_columns[0]:
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
        with pathway_control_columns[1]:
            minimum_pathway_modules = st.slider(
                "Minimum enriched modules per cell",
                min_value=1,
                max_value=10,
                value=1,
                key=f"pathway_mdc_minimum_modules_{module_set}",
            )
        with pathway_control_columns[2]:
            pathway_mdc_row_scope = st.radio(
                "Module MDC rows",
                options=["All", "MDC FDR-significant only"],
                horizontal=True,
                key=f"pathway_mdc_row_scope_{module_set}",
            )

        pathway_rows = cached_pathway_mdc_rows(
            module_set, pathway_kegg_threshold
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

        pathway_summary = summarize_pathway_mdc_rows(
            pathway_rows,
            mdc_fdr_threshold=mdc_threshold,
            minimum_modules=minimum_pathway_modules,
        )
        if pathway_summary.empty:
            st.info("No pathway-MDC cells remain under the current filters.")
        else:
            retained_pathway_ids = set(pathway_summary["pathway_id"].astype(str))
            pathway_rows = pathway_rows.loc[
                pathway_rows["pathway_id"].astype(str).isin(retained_pathway_ids)
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
            pathway_metric_columns[0].metric(
                "Pathways", f"{pathway_summary['pathway_id'].nunique():,}"
            )
            pathway_metric_columns[1].metric(
                "Modules", f"{pathway_rows['module'].nunique():,}"
            )
            pathway_metric_columns[2].metric(
                "Region/pathway cells", f"{len(pathway_summary):,}"
            )
            pathway_metric_columns[3].metric(
                "Module-component annotations", f"{len(pathway_rows):,}"
            )
            pathway_options = pathway_ranking["pathway_id"].astype(str).tolist()
            pathway_label_map = pathway_ranking.set_index("pathway_id")[
                "pathway_label"
            ].to_dict()
            pathway_selection_columns = st.columns([2, 1])
            selected_pathway_id = pathway_selection_columns[0].selectbox(
                "Pathway detail",
                options=pathway_options,
                format_func=lambda value: f"{pathway_label_map[value]} ({value})",
                key=f"pathway_mdc_selected_{module_set}",
            )
            maximum_top_pathways = min(60, len(pathway_options))
            top_pathways = pathway_selection_columns[1].slider(
                "Pathways in heatmap",
                min_value=1,
                max_value=maximum_top_pathways,
                value=min(25, maximum_top_pathways),
                key=f"pathway_mdc_top_n_{module_set}",
            )
            st.plotly_chart(
                pathway_mdc_heatmap_figure(
                    pathway_summary,
                    scale=mdc_scale,
                    top_n=top_pathways,
                    selected_pathway_id=selected_pathway_id,
                    module_definition=module_set_label,
                ),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            st.caption(
                "Each heatmap cell gives equal weight to every qualifying enriched module. "
                "On log2 scale it is the arithmetic mean log2 MDC; on raw scale it is the "
                "equivalent geometric mean MDC ratio. Cell n is the number of enriched modules."
            )
            selected_pathway_rows = pathway_rows.loc[
                pathway_rows["pathway_id"].astype(str).eq(selected_pathway_id)
            ].copy()
            st.plotly_chart(
                pathway_mdc_detail_figure(
                    selected_pathway_rows,
                    pathway_id=selected_pathway_id,
                    selected_module=module,
                    threshold=mdc_threshold,
                    scale=mdc_scale,
                    module_definition=module_set_label,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            pathway_detail_columns = [
                "pathway_id",
                "pathway_label",
                "category_level1",
                "category_level2",
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
                "overlap_k",
                "overlap_genes",
            ]
            selected_pathway_rows = selected_pathway_rows.sort_values(
                ["component_label", "directional_fdr", "enrichment_fdr"],
                kind="stable",
            )
            st.dataframe(
                selected_pathway_rows[pathway_detail_columns],
                use_container_width=True,
                hide_index=True,
                height=480,
                column_config={"overlap_genes": "Overlap gene symbols"},
            )
            download_columns = [
                "pathway_id",
                "pathway_label",
                "category_level1",
                "category_level2",
                "component",
                "component_label",
                "enrichment_scope",
                "n_modules",
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
                "Download pathway-component summary (TSV)",
                data=dataframe_to_tsv_bytes(pathway_summary[download_columns]),
                file_name=f"{download_prefix}pathway_resolved_MDC_summary.tsv",
                mime="text/tab-separated-values",
            )
            pathway_download_columns[1].download_button(
                "Download selected pathway module rows (TSV)",
                data=dataframe_to_tsv_bytes(
                    selected_pathway_rows[pathway_detail_columns]
                ),
                file_name=(
                    f"{download_prefix}{selected_pathway_id}_regional_MDC_modules.tsv"
                ),
                mime="text/tab-separated-values",
            )

if active_view == "Statistics":
    st.subheader("Robust association statistics")
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
    st.dataframe(
        statistics[display_columns], use_container_width=True, hide_index=True
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
        st.dataframe(
            displayed_kegg,
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
        st.plotly_chart(
            module_size_distribution_figure(
                chart_module_details,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
            key=f"module_size_distribution_{module_set}_{module_type_scope}",
        )
        st.plotly_chart(
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
        st.plotly_chart(
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
    st.dataframe(
        module_details.loc[module_details["module"].astype(int).eq(int(module)), detail_columns],
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
    st.dataframe(
        module_details[detail_columns],
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
        "MDC is an independent group-network comparison supplied as module-level context: "
        "mean absolute AD adjacency divided by mean absolute Control adjacency. The app "
        "shows total, pooled TS/CT, and six tissue-resolved ratios and their conservative "
        "directional permutation FDR on raw-ratio or log2 scales. Separate TS and CT "
        "scatter plots relate MDC to normalized Shannon tissue-mixing entropy. The "
        "pathway-resolved tab annotates module MDC with component-matched KEGG enrichment; "
        "it is not a newly recomputed edge-level pathway MDC. Its broader tissue-union "
        "AD/Control cohort and absence of MCI are stated in the MDC tab."
    )

    st.markdown("#### Module feature definitions")
    st.dataframe(cached_feature_definitions(), use_container_width=True, hide_index=True)
    st.markdown("#### Tissue labels")
    st.dataframe(cached_tissue_mapping(), use_container_width=True, hide_index=True)
    st.markdown("#### Public deploy safeguards")
    st.markdown(
        "The deploy Parquet files do not contain `projid`, the source donor identifier, "
        "or a donor-to-pseudonym key. Each point has a random-salted pseudonymous sample "
        "label that is stable within this bundle only. Diagnosis, the five primary phenotypes, "
        "and selected age, education, APOE, cognitive diagnosis, neuropathology, and Parkinsonism "
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
