"""Independent single-region SE2 module data and Streamlit views.

Regional modules are intentionally isolated from the multi-tissue module-set
router.  Numeric module IDs are only meaningful within one partition source
and tissue, so every public row carries a namespaced ``module_key``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st

from app_helpers.charts import (
    CONTINUOUS_COLOR_SCALES,
    categorical_association_figure,
    correlation_heatmap_figure,
    distribution_feature_heatmap_figure,
    distribution_figure,
    distribution_summary,
    grouped_association_figure,
)
from app_helpers.correlations import (
    add_across_module_fdr,
    add_categorical_across_module_fdr,
    calculate_categorical_associations,
    calculate_correlations,
)
from app_helpers.data import (
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_LEVEL_ORDERS,
    ASSOCIATION_OUTCOME_LABELS,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    COLOR_LABELS,
    DIAGNOSIS_ORDER,
    HOVER_LABELS,
    NUMERIC_OUTCOMES,
    OUTCOME_LABELS,
    SCALE_LABELS,
    SELECTABLE_ASSOCIATION_OUTCOMES,
    association_level_label,
    dataframe_to_tsv_bytes,
    load_sample_metadata,
)
from app_helpers.distributions import (
    add_pairwise_across_module_fdr,
    calculate_pairwise_distribution_statistics,
    default_reference_level,
    distribution_contrasts,
)
from app_helpers.drive_data import DATA_DIR, data_path_available, ensure_data_path
from app_helpers.streamlit_compat import plotly_chart as render_plotly_chart
from app_helpers.table_controls import filterable_dataframe


REGIONAL_ANALYSIS_VIEWS = {
    "Associations",
    "Feature distributions",
    "Correlation heatmaps",
}
REGIONAL_ROOT = DATA_DIR / "regional_modules"
REGIONAL_MANIFEST = REGIONAL_ROOT / "manifest.json"
REGIONAL_DETAILS = REGIONAL_ROOT / "module_details.parquet"
REGIONAL_KEGG = REGIONAL_ROOT / "kegg_enrichment.parquet"
REGIONAL_SOURCE_ORDER = (
    "single_region_complete_case_l3",
    "single_region_full_tissue_l3",
)
REGION_ORDER = ("AC", "DLPFC", "PCG")
REGION_COMPONENTS = {
    "AC": "REGION_AC",
    "DLPFC": "REGION_DLPFC",
    "PCG": "REGION_PCG",
}
COHORT_LABELS = {
    "common_450": "Common 450 donors",
    "maximum_tissue": "Maximum tissue cohort",
}
SCORE_LABELS = {
    ("eigengene", "not_applicable"): "Module eigengene (PCA1 expression)",
    ("connectivity", "control_anchored"): "Control-referenced LIONESS connectivity",
    ("connectivity", "standard"): "Standard LIONESS connectivity",
}
KEGG_PRIORITY_PATTERN = re.compile(
    r"lipid|fatty acid|cholesterol|glycerolipid|glycerophospholipid|"
    r"sphingolipid|metaboli|alzheimer|infection|infectious|viral|virus",
    flags=re.IGNORECASE,
)
REGIONAL_API_VERSION = 1


def regional_data_available() -> bool:
    return all(
        data_path_available(path)
        for path in (REGIONAL_MANIFEST, REGIONAL_DETAILS, REGIONAL_KEGG)
    )


@st.cache_data(show_spinner=False)
def load_regional_manifest() -> dict[str, object]:
    path = ensure_data_path(REGIONAL_MANIFEST)
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_regional_details() -> pd.DataFrame:
    return pd.read_parquet(ensure_data_path(REGIONAL_DETAILS))


@st.cache_data(show_spinner=False)
def load_regional_kegg() -> pd.DataFrame:
    return pd.read_parquet(ensure_data_path(REGIONAL_KEGG))


def regional_score_path(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
) -> Path:
    if score_variant == "eigengene":
        return REGIONAL_ROOT / source / cohort_scope / "eigengene" / f"{tissue}.parquet"
    return (
        REGIONAL_ROOT
        / source
        / cohort_scope
        / "connectivity"
        / network_method
        / f"{tissue}.parquet"
    )


def _read_parquet(
    path: Path,
    filters: list[tuple[str, str, object]] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    materialized = ensure_data_path(path, filters or ())
    return pq.read_table(
        materialized,
        filters=filters or None,
        columns=list(columns) if columns is not None else None,
    ).to_pandas()


@st.cache_data(show_spinner=False, max_entries=96)
def load_regional_scores(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
    module: int | None = None,
) -> pd.DataFrame:
    path = regional_score_path(
        source, cohort_scope, score_variant, network_method, tissue
    )
    filters = [("module", "=", int(module))] if module is not None else []
    result = _read_parquet(path, filters=filters)
    if cohort_scope == "common_450":
        metadata = load_sample_metadata()
    else:
        metadata_path = REGIONAL_ROOT / "metadata" / cohort_scope / f"{tissue}.parquet"
        metadata = pd.read_parquet(ensure_data_path(metadata_path))
    duplicate = metadata["sample_id"].duplicated()
    if duplicate.any():
        raise ValueError("Regional sample metadata contain duplicate pseudonyms")
    metadata_columns = [
        column for column in metadata.columns
        if column != "sample_id" and column not in result.columns
    ]
    result = result.merge(
        metadata[["sample_id", *metadata_columns]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    return result


def completed_combinations(manifest: dict[str, object]) -> pd.DataFrame:
    result = pd.DataFrame(manifest.get("completed_combinations", []))
    if result.empty:
        return result
    return result.loc[result["status"].eq("complete")].copy()


def score_label(score_variant: str, network_method: str) -> str:
    return SCORE_LABELS.get((score_variant, network_method), score_variant)


def _ordered_levels(frame: pd.DataFrame, variable: str) -> list[object]:
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
    ordered.extend(
        sorted(
            (value for value in observed if str(value) not in included),
            key=str,
        )
    )
    return ordered


def _selected_level_mask(values: pd.Series, selected: Iterable[object]) -> pd.Series:
    selected_text = {str(value) for value in selected}
    return values.astype(str).isin(selected_text)


def _prepare_score_frame(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
    scale: str,
    diagnoses: tuple[str, ...],
    module: int | None = None,
) -> pd.DataFrame:
    result = load_regional_scores(
        source, cohort_scope, score_variant, network_method, tissue, module
    ).copy()
    result["metric_value"] = pd.to_numeric(
        result[f"metric_{scale}"], errors="coerce"
    )
    result["component"] = REGION_COMPONENTS[tissue]
    result["component_label"] = (
        tissue + " regional M" + result["module"].astype(int).astype(str)
    )
    result["metric_family"] = score_variant
    if diagnoses and "diagnosis_group" in result:
        result = result.loc[result["diagnosis_group"].isin(diagnoses)].copy()
    return result


@st.cache_data(show_spinner=False, max_entries=128)
def regional_correlation_catalog(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
    scale: str,
    diagnoses: tuple[str, ...],
    grouping_variable: str,
    grouping_levels: tuple[object, ...],
    minimum_group_n: int,
    outcomes: tuple[str, ...],
    include_pooled: bool,
) -> pd.DataFrame:
    frame = _prepare_score_frame(
        source, cohort_scope, score_variant, network_method,
        tissue, scale, diagnoses,
    )
    if grouping_variable == "__all__":
        frame["grouping_variable"] = "__all__"
        frame["grouping_level"] = "__all__"
    else:
        frame = frame.loc[
            _selected_level_mask(frame[grouping_variable], grouping_levels)
        ].copy()
        frame["grouping_variable"] = grouping_variable
        frame["grouping_level"] = frame[grouping_variable]
    group_columns = [
        "partition_source", "tissue", "module", "component", "component_label",
        "grouping_variable", "grouping_level",
    ]
    result = calculate_correlations(
        frame, group_columns=group_columns, outcomes=outcomes,
        min_group_n=minimum_group_n,
    )
    if include_pooled and grouping_variable != "__all__":
        pooled = frame.copy()
        pooled["grouping_level"] = "__pooled__"
        pooled_result = calculate_correlations(
            pooled, group_columns=group_columns, outcomes=outcomes,
            min_group_n=minimum_group_n,
        )
        result = pd.concat([result, pooled_result], ignore_index=True)
    if result.empty:
        return result
    result["cohort_scope"] = cohort_scope
    result["score_variant"] = score_variant
    result["network_method"] = network_method
    result = add_across_module_fdr(
        result,
        family_columns=[
            "partition_source", "tissue", "cohort_scope", "score_variant",
            "network_method", "outcome", "grouping_variable", "grouping_level",
        ],
    )
    result["grouping_label"] = result["grouping_level"].map(
        lambda value: (
            "All displayed donors (pooled)"
            if str(value) == "__pooled__"
            else association_level_label(grouping_variable, value)
        )
    )
    result["outcome_label"] = result["outcome"].map(OUTCOME_LABELS)
    result["feature_label"] = score_label(score_variant, network_method)
    return result


@st.cache_data(show_spinner=False, max_entries=128)
def regional_categorical_catalog(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
    scale: str,
    diagnoses: tuple[str, ...],
    category_variable: str,
    category_levels: tuple[object, ...],
    minimum_group_n: int,
) -> pd.DataFrame:
    frame = _prepare_score_frame(
        source, cohort_scope, score_variant, network_method,
        tissue, scale, diagnoses,
    )
    frame = frame.loc[
        _selected_level_mask(frame[category_variable], category_levels)
    ].copy()
    result = calculate_categorical_associations(
        frame,
        group_columns=[
            "partition_source", "tissue", "module", "component", "component_label"
        ],
        category_column=category_variable,
        min_group_n=minimum_group_n,
    )
    if result.empty:
        return result
    result["cohort_scope"] = cohort_scope
    result["score_variant"] = score_variant
    result["network_method"] = network_method
    result["grouping_variable"] = category_variable
    result = add_categorical_across_module_fdr(
        result,
        family_columns=[
            "partition_source", "tissue", "cohort_scope", "score_variant",
            "network_method", "outcome", "grouping_variable",
        ],
    )
    result["feature_label"] = score_label(score_variant, network_method)
    return result


@st.cache_data(show_spinner=False, max_entries=96)
def regional_pairwise_catalog(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    tissue: str,
    scale: str,
    diagnoses: tuple[str, ...],
    category_variable: str,
    category_levels: tuple[object, ...],
    contrasts: tuple[tuple[object, object], ...],
    minimum_group_n: int,
) -> pd.DataFrame:
    frame = _prepare_score_frame(
        source, cohort_scope, score_variant, network_method,
        tissue, scale, diagnoses,
    )
    frame = frame.loc[
        _selected_level_mask(frame[category_variable], category_levels)
    ].copy()
    result = calculate_pairwise_distribution_statistics(
        frame,
        group_columns=[
            "partition_source", "tissue", "module", "component", "component_label"
        ],
        category_column=category_variable,
        contrasts=contrasts,
        minimum_group_n=minimum_group_n,
        bootstrap_resamples=0,
        seed=42,
        include_ks=True,
    )
    if result.empty:
        return result
    result["cohort_scope"] = cohort_scope
    result["score_variant"] = score_variant
    result["network_method"] = network_method
    result = add_pairwise_across_module_fdr(
        result,
        family_columns=[
            "partition_source", "tissue", "cohort_scope", "score_variant",
            "network_method", "grouping_variable",
        ],
    )
    return result


def _format_fdr(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.2e}" if float(numeric) < 0.001 else f"{float(numeric):.3f}"


def _regional_kegg_subtitle(
    kegg: pd.DataFrame,
    source: str,
    tissue: str,
    module: int,
) -> str:
    selected = kegg.loc[
        kegg["partition_source"].eq(source)
        & kegg["tissue"].eq(tissue)
        & kegg["module"].astype(int).eq(int(module))
    ].copy()
    if selected.empty:
        return f"KEGG enrichment ({tissue}): unavailable"
    searchable = selected[
        ["category_level1", "category_level2", "pathway_name"]
    ].fillna("").astype(str).agg(" ".join, axis=1)
    priority = selected.loc[
        searchable.map(lambda value: bool(KEGG_PRIORITY_PATTERN.search(value)))
        & pd.to_numeric(selected["fdr"], errors="coerce").lt(0.05)
    ]
    candidates = priority if not priority.empty else selected
    row = candidates.sort_values(["fdr", "p", "pathway_name"]).iloc[0]
    pathway = str(row["pathway_name"]).replace(" - Homo sapiens (human)", "")
    return (
        f"KEGG enrichment ({tissue}): {row['category_level1']} / "
        f"{row['category_level2']} / {pathway} | FDR={_format_fdr(row['fdr'])}"
    )


def _module_options(
    details: pd.DataFrame,
    source: str,
    tissue: str,
) -> tuple[list[int], dict[int, str], int]:
    selected = details.loc[
        details["partition_source"].eq(source) & details["tissue"].eq(tissue)
    ].copy()
    selected["module"] = selected["module"].astype(int)
    modules = sorted(selected["module"].tolist())
    sizes = selected.set_index("module")["n_genes"].astype(int).to_dict()
    labels = {
        module: f"{tissue} regional M{module} ({sizes[module]:,} genes)"
        for module in modules
    }
    largest = int(selected.sort_values(["n_genes", "module"], ascending=[False, True]).iloc[0]["module"])
    return modules, labels, modules.index(largest)


def _selected_frames(
    source: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    scale: str,
    diagnoses: tuple[str, ...],
    selected_modules: dict[str, int],
) -> pd.DataFrame:
    parts = [
        _prepare_score_frame(
            source, cohort_scope, score_variant, network_method,
            tissue, scale, diagnoses, module,
        )
        for tissue, module in selected_modules.items()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _download_table(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=dataframe_to_tsv_bytes(frame),
        file_name=filename,
        mime="text/tab-separated-values",
        key=key,
    )


def _render_associations(
    *,
    source: str,
    source_label: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    scale: str,
    tissues: tuple[str, ...],
    selected_modules: dict[str, int],
    diagnoses: tuple[str, ...],
    metadata: pd.DataFrame,
    kegg: pd.DataFrame,
) -> None:
    with st.sidebar:
        phenotype = st.selectbox(
            "Association outcome",
            options=list(ASSOCIATION_OUTCOME_LABELS),
            format_func=lambda value: ASSOCIATION_OUTCOME_LABELS[value],
            key="regional_association_outcome",
        )
        if phenotype in SELECTABLE_ASSOCIATION_OUTCOMES:
            interpretation = st.radio(
                "Outcome interpretation",
                ["numeric", "categorical"],
                format_func=lambda value: (
                    "Numeric / ordinal correlation"
                    if value == "numeric" else "Categorical comparison"
                ),
                horizontal=True,
                key="regional_association_interpretation",
            )
        elif phenotype in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES:
            interpretation = "categorical"
        else:
            interpretation = "numeric"
        minimum_group_n = st.selectbox(
            "Minimum category size", [5, 10, 20], index=1,
            key="regional_association_minimum_n",
        )
        grouping_variable = "diagnosis_group"
        selected_levels: list[object] = list(diagnoses)
        show_pooled = True
        correlation_method = "Spearman"
        annotation_fields = ["n", "coefficient", "p", "fdr"]
        trend_line_rule = "all"
        significance_cutoff = 0.05
        show_group_trends = True
        if interpretation == "numeric":
            grouping_options = [
                value for value in ASSOCIATION_GROUP_LABELS if value != phenotype
            ]
            grouping_variable = st.selectbox(
                "Group correlations by",
                grouping_options,
                format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                key="regional_association_grouping",
            )
            levels = _ordered_levels(metadata, grouping_variable)
            selected_levels = st.multiselect(
                "Group levels",
                levels,
                default=levels,
                format_func=lambda value: association_level_label(grouping_variable, value),
                disabled=grouping_variable == "__all__",
                key="regional_association_levels",
            )
            if grouping_variable == "__all__":
                selected_levels = ["__all__"]
            show_pooled = (
                st.checkbox(
                    "Show pooled association across displayed donors",
                    value=True,
                    key="regional_association_pooled",
                )
                if grouping_variable != "__all__" else False
            )
            correlation_method = st.radio(
                "Association correlation", ["Spearman", "Pearson"],
                horizontal=True, key="regional_association_method",
            )
            annotation_fields = st.multiselect(
                "Plot annotation fields",
                ["n", "coefficient", "p", "fdr"],
                default=["n", "coefficient", "p", "fdr"],
                key="regional_association_annotations",
            )
            trend_line_rule = st.selectbox(
                "Trend-line rule", ["all", "p", "fdr", "none"],
                format_func=lambda value: {
                    "all": "All eligible groups",
                    "p": "Nominal p below cutoff",
                    "fdr": "Module-set FDR below cutoff",
                    "none": "No trend lines",
                }[value],
                key="regional_association_line_rule",
            )
            show_group_trends = st.checkbox(
                "Show group-specific OLS trend lines",
                value=True,
                key="regional_association_group_lines",
            )
            significance_cutoff = st.radio(
                "Significance cutoff", [0.05, 0.10], horizontal=True,
                key="regional_association_cutoff",
            )
        color_by = st.selectbox(
            "Color points by",
            options=list(COLOR_LABELS),
            format_func=lambda value: COLOR_LABELS[value],
            index=list(COLOR_LABELS).index(
                grouping_variable
                if grouping_variable in COLOR_LABELS else "diagnosis_group"
            ),
            key="regional_association_color",
        )
        palette = "Blue–white–orange"
        reverse_palette = False
        if color_by not in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES:
            palette = st.selectbox(
                "Continuous color scale",
                list(CONTINUOUS_COLOR_SCALES),
                key="regional_association_palette",
            )
            reverse_palette = st.checkbox(
                "Reverse color scale", key="regional_association_reverse_palette"
            )

    if not selected_levels:
        st.warning("Select at least one category level.")
        return
    frame = _selected_frames(
        source, cohort_scope, score_variant, network_method,
        scale, diagnoses, selected_modules,
    )
    subtitles = {
        REGION_COMPONENTS[tissue]: _regional_kegg_subtitle(
            kegg, source, tissue, selected_modules[tissue]
        )
        for tissue in tissues
    }
    feature_label = score_label(score_variant, network_method)
    title = f"Independent regional modules: {ASSOCIATION_OUTCOME_LABELS[phenotype]} vs {feature_label}"
    definition_label = f"{source_label} · {COHORT_LABELS.get(cohort_scope, cohort_scope)}"

    if interpretation == "numeric":
        catalogs = [
            regional_correlation_catalog(
                source, cohort_scope, score_variant, network_method, tissue,
                scale, diagnoses, grouping_variable, tuple(selected_levels),
                int(minimum_group_n), (phenotype,), bool(show_pooled),
            )
            for tissue in tissues
        ]
        statistics = pd.concat(catalogs, ignore_index=True)
        statistics = statistics.loc[
            statistics.apply(
                lambda row: int(row["module"]) == int(
                    selected_modules[str(row["tissue"])]
                ),
                axis=1,
            )
        ].copy()
        level_labels = {
            str(value): association_level_label(grouping_variable, value)
            for value in selected_levels
        }
        figure = grouped_association_figure(
            frame,
            statistics,
            phenotype=phenotype,
            phenotype_label=ASSOCIATION_OUTCOME_LABELS[phenotype],
            feature_label=feature_label,
            scale_label=SCALE_LABELS[scale],
            grouping_variable=grouping_variable,
            grouping_levels=selected_levels,
            grouping_labels=level_labels,
            module=0,
            color_by=color_by,
            color_label=COLOR_LABELS[color_by],
            hover_fields=HOVER_LABELS,
            correlation_method=correlation_method.lower(),
            annotation_fields=annotation_fields,
            trend_line_rule=trend_line_rule,
            significance_cutoff=float(significance_cutoff),
            minimum_group_n=int(minimum_group_n),
            show_group_trends=show_group_trends,
            show_pooled=show_pooled,
            module_definition=definition_label,
            continuous_colorscale=palette,
            reverse_colorscale=reverse_palette,
            categorical_color_fields=CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
            kegg_subtitles=subtitles,
            title_override=title,
        )
    else:
        levels = _ordered_levels(metadata, phenotype)
        catalogs = [
            regional_categorical_catalog(
                source, cohort_scope, score_variant, network_method, tissue,
                scale, diagnoses, phenotype, tuple(levels), int(minimum_group_n),
            )
            for tissue in tissues
        ]
        statistics = pd.concat(catalogs, ignore_index=True)
        statistics = statistics.loc[
            statistics.apply(
                lambda row: int(row["module"]) == int(
                    selected_modules[str(row["tissue"])]
                ),
                axis=1,
            )
        ].copy()
        figure = categorical_association_figure(
            frame,
            statistics,
            category_variable=phenotype,
            category_label=ASSOCIATION_OUTCOME_LABELS[phenotype],
            category_levels=levels,
            category_labels={
                str(value): association_level_label(phenotype, value)
                for value in levels
            },
            feature_label=feature_label,
            scale_label=SCALE_LABELS[scale],
            module=0,
            minimum_group_n=int(minimum_group_n),
            module_definition=definition_label,
            kegg_subtitles=subtitles,
            hover_fields=HOVER_LABELS,
            title_override=title,
        )
    render_plotly_chart(figure, key="regional_association_figure")
    table = statistics.copy()
    table.insert(0, "partition_source_label", source_label)
    filterable_dataframe(
        table,
        table_key="regional_association_statistics",
        table_name="Regional association statistics",
    )
    _download_table(
        "Download regional association statistics (TSV)",
        table,
        "regional_module_associations.tsv",
        "regional_association_download",
    )
    st.caption(
        "Association FDR is BH across valid modules from the same fixed tissue "
        "partition, with source, cohort, score, method, outcome, grouping variable, "
        "grouping level, and correlation method held fixed. Tissues and category "
        "levels are never combined into one FDR family."
    )


def _render_distributions(
    *,
    source: str,
    source_label: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    scale: str,
    tissues: tuple[str, ...],
    selected_modules: dict[str, int],
    diagnoses: tuple[str, ...],
    metadata: pd.DataFrame,
) -> None:
    with st.sidebar:
        grouping_variable = st.selectbox(
            "Group distributions by",
            options=list(ASSOCIATION_GROUP_LABELS),
            format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
            index=list(ASSOCIATION_GROUP_LABELS).index("diagnosis_group"),
            key="regional_distribution_grouping",
        )
        levels = _ordered_levels(metadata, grouping_variable)
        selected_levels = st.multiselect(
            "Distribution group levels",
            levels,
            default=levels,
            format_func=lambda value: association_level_label(grouping_variable, value),
            disabled=grouping_variable == "__all__",
            key="regional_distribution_levels",
        )
        if grouping_variable == "__all__":
            selected_levels = ["__all__"]
        minimum_group_n = st.selectbox(
            "Minimum group size", [5, 10, 20], index=1,
            key="regional_distribution_minimum_n",
        )
        analysis = st.radio(
            "Distribution analysis",
            [
                "Distribution shape",
                "Overall differentiation",
                "Pairwise differentiation",
                "Feature comparison heatmaps",
                "Find differentiated modules",
            ],
            key="regional_distribution_analysis",
        )
    if not selected_levels:
        st.warning("Select at least one distribution group level.")
        return
    frame = _selected_frames(
        source, cohort_scope, score_variant, network_method,
        scale, diagnoses, selected_modules,
    )
    if grouping_variable == "__all__":
        frame["distribution_group"] = "All displayed donors"
        display_groups = ["All displayed donors"]
    else:
        frame = frame.loc[
            _selected_level_mask(frame[grouping_variable], selected_levels)
        ].copy()
        frame["distribution_group"] = frame[grouping_variable].map(
            lambda value: association_level_label(grouping_variable, value)
        )
        display_groups = [
            association_level_label(grouping_variable, value)
            for value in selected_levels
        ]
    feature_label = score_label(score_variant, network_method)
    definition_label = f"{source_label} · {COHORT_LABELS.get(cohort_scope, cohort_scope)}"

    if analysis == "Distribution shape":
        chart_type = st.radio(
            "Distribution view", ["Histogram", "Violin", "Raincloud", "ECDF"],
            horizontal=True, key="regional_distribution_chart_type",
        )
        figure = distribution_figure(
            frame,
            feature_label=feature_label,
            scale_label=SCALE_LABELS[scale],
            diagnoses=display_groups,
            module=0,
            chart_type=chart_type,
            module_definition=definition_label,
            group_column="distribution_group",
            group_label=ASSOCIATION_GROUP_LABELS[grouping_variable],
            title_override="Independent regional module score distributions",
        )
        render_plotly_chart(figure, key="regional_distribution_shape")
        summary = distribution_summary(
            frame,
            group_column="distribution_group",
            group_label="distribution_group",
        )
        summary.insert(0, "partition_source_label", source_label)
        filterable_dataframe(
            summary,
            table_key="regional_distribution_summary",
            table_name="Regional distribution summary",
        )
        _download_table(
            "Download distribution summary (TSV)", summary,
            "regional_distribution_summary.tsv", "regional_distribution_summary_download",
        )
        return

    if grouping_variable == "__all__":
        st.info("Choose a categorical grouping variable for differentiation tests.")
        return

    catalogs = [
        regional_categorical_catalog(
            source, cohort_scope, score_variant, network_method, tissue,
            scale, diagnoses, grouping_variable, tuple(selected_levels),
            int(minimum_group_n),
        )
        for tissue in tissues
    ]
    omnibus = pd.concat(catalogs, ignore_index=True)
    selected_omnibus = omnibus.loc[
        omnibus.apply(
            lambda row: int(row["module"]) == int(
                selected_modules[str(row["tissue"])]
            ),
            axis=1,
        )
    ].copy()

    if analysis == "Overall differentiation":
        selected_omnibus["display_label"] = selected_omnibus["component_label"]
        figure = px.bar(
            selected_omnibus.sort_values("epsilon_squared"),
            x="epsilon_squared", y="display_label", orientation="h",
            hover_data={
                "categorical_p": ":.3g",
                "categorical_fdr_across_modules": ":.3g",
                "categorical_fdr_module_family_n": True,
                "n_tested": True,
            },
            title="Overall regional-module group differentiation",
            labels={"epsilon_squared": "Kruskal–Wallis epsilon-squared", "display_label": ""},
        )
        figure.update_traces(marker_color="#2C7FB8")
        figure.update_layout(template="plotly_white", height=430)
        render_plotly_chart(figure, key="regional_distribution_omnibus")
        filterable_dataframe(
            selected_omnibus,
            table_key="regional_distribution_omnibus_table",
            table_name="Regional omnibus differentiation",
        )
        return

    if analysis == "Pairwise differentiation":
        counts = frame.groupby(grouping_variable, observed=True).size()
        reference = default_reference_level(
            grouping_variable, selected_levels, counts,
            minimum_group_n=int(minimum_group_n),
        )
        if reference is None:
            st.info("No selected group is large enough to serve as a reference.")
            return
        reference = st.selectbox(
            "Reference group",
            selected_levels,
            index=list(selected_levels).index(reference),
            format_func=lambda value: association_level_label(grouping_variable, value),
            key="regional_distribution_reference",
        )
        contrast_mode = st.radio(
            "Pairwise contrasts",
            ["Reference vs other groups", "All selected pairs"],
            horizontal=True,
            key="regional_distribution_contrast_mode",
        )
        contrasts = tuple(
            distribution_contrasts(
                selected_levels,
                mode="reference" if contrast_mode.startswith("Reference") else "all_pairs",
                reference_level=reference,
            )
        )
        catalog_parts = [
            regional_pairwise_catalog(
                source, cohort_scope, score_variant, network_method, tissue,
                scale, diagnoses, grouping_variable, tuple(selected_levels),
                contrasts, int(minimum_group_n),
            )
            for tissue in tissues
        ]
        catalog = pd.concat(catalog_parts, ignore_index=True)
        selected_catalog = catalog.loc[
            catalog.apply(
                lambda row: int(row["module"]) == int(
                    selected_modules[str(row["tissue"])]
                ),
                axis=1,
            )
        ].copy()
        # Bootstrap only the displayed modules. Across-module FDR comes from the
        # zero-bootstrap catalog above, so the expensive resampling stays bounded.
        bootstrap_parts: list[pd.DataFrame] = []
        for tissue, module in selected_modules.items():
            selected_frame = frame.loc[
                frame["tissue"].eq(tissue) & frame["module"].astype(int).eq(int(module))
            ]
            boot = calculate_pairwise_distribution_statistics(
                selected_frame,
                group_columns=[
                    "partition_source", "tissue", "module", "component", "component_label"
                ],
                category_column=grouping_variable,
                contrasts=contrasts,
                minimum_group_n=int(minimum_group_n),
                bootstrap_resamples=1000,
                seed=42,
                include_ks=True,
            )
            bootstrap_parts.append(boot)
        bootstrapped = pd.concat(bootstrap_parts, ignore_index=True)
        merge_keys = [
            "partition_source", "tissue", "module", "component",
            "reference_level_key", "comparison_level_key",
        ]
        fdr_columns = [
            *merge_keys,
            "mann_whitney_fdr_across_modules", "mann_whitney_fdr_module_family_n",
            "ks_fdr_across_modules", "ks_fdr_module_family_n",
        ]
        pairwise = bootstrapped.merge(
            selected_catalog[fdr_columns], on=merge_keys,
            how="left", validate="one_to_one",
        )
        pairwise["reference_label"] = pairwise["reference_level"].map(
            lambda value: association_level_label(grouping_variable, value)
        )
        pairwise["comparison_label"] = pairwise["comparison_level"].map(
            lambda value: association_level_label(grouping_variable, value)
        )
        pairwise["contrast"] = (
            pairwise["comparison_label"] + " vs " + pairwise["reference_label"]
        )
        pairwise["display_label"] = pairwise["component_label"] + " · " + pairwise["contrast"]
        figure = px.scatter(
            pairwise,
            x="cliffs_delta", y="display_label",
            error_x=pairwise["cliffs_delta_ci_high"] - pairwise["cliffs_delta"],
            error_x_minus=pairwise["cliffs_delta"] - pairwise["cliffs_delta_ci_low"],
            color="tissue",
            hover_data={
                "mann_whitney_p": ":.3g",
                "mann_whitney_fdr_across_modules": ":.3g",
                "probability_superiority": ":.3f",
                "median_difference": ":.3f",
            },
            title="Pairwise regional-module distribution effects",
            labels={"cliffs_delta": "Cliff's delta (comparison − reference)", "display_label": ""},
        )
        figure.add_vline(x=0, line_dash="dash", line_color="#65727E")
        figure.update_layout(template="plotly_white", height=max(440, 120 + 38 * len(pairwise)))
        render_plotly_chart(figure, key="regional_distribution_pairwise")
        filterable_dataframe(
            pairwise,
            table_key="regional_distribution_pairwise_table",
            table_name="Regional pairwise differentiation",
        )
        return

    if analysis == "Feature comparison heatmaps":
        scope = st.radio(
            "Differentiation heatmap scope",
            ["Selected regional modules", "All regional modules"],
            horizontal=True,
            key="regional_distribution_heatmap_scope",
        )
        heatmap = omnibus.copy()
        if scope.startswith("Selected"):
            heatmap = selected_omnibus
        heatmap["heatmap_row"] = heatmap.apply(
            lambda row: f"{row['tissue']} regional M{int(row['module'])}", axis=1
        )
        heatmap["heatmap_column"] = heatmap["feature_label"]
        heatmap["component_label"] = heatmap["tissue"]
        top_n = len(heatmap["heatmap_row"].unique())
        if scope.startswith("All"):
            top_n = st.selectbox(
                "Rows in all-module heatmap", [20, 50, 100, top_n],
                key="regional_distribution_heatmap_top_n",
            )
            strongest = (
                heatmap.groupby("heatmap_row", observed=True)["epsilon_squared"]
                .max().nlargest(int(top_n)).index
            )
            heatmap = heatmap.loc[heatmap["heatmap_row"].isin(strongest)]
        row_order = (
            heatmap.groupby("heatmap_row", observed=True)["epsilon_squared"]
            .max().sort_values(ascending=False).index.tolist()
        )
        figure = distribution_feature_heatmap_figure(
            heatmap,
            title=(
                f"Regional-module differentiation by {ASSOCIATION_GROUP_LABELS[grouping_variable]}"
                "<br><sup>* tissue-specific module-family FDR &lt; 0.05</sup>"
            ),
            row_order=row_order,
            column_order=heatmap["heatmap_column"].drop_duplicates().tolist(),
        )
        render_plotly_chart(figure, key="regional_distribution_feature_heatmap")
        filterable_dataframe(
            heatmap,
            table_key="regional_distribution_feature_heatmap_table",
            table_name="Regional feature-differentiation table",
        )
        return

    # Find differentiated modules.
    significance = st.radio(
        "Significance filter",
        ["No FDR filter", "FDR < 0.05", "FDR < 0.10 (exploratory)"],
        horizontal=True,
        key="regional_distribution_ranking_fdr",
    )
    top_n = st.selectbox(
        "Top modules", [10, 20, 50], index=1,
        key="regional_distribution_ranking_top_n",
    )
    ranked = omnibus.copy()
    cutoff = {"No FDR filter": None, "FDR < 0.05": 0.05, "FDR < 0.10 (exploratory)": 0.10}[significance]
    if cutoff is not None:
        ranked = ranked.loc[
            pd.to_numeric(ranked["categorical_fdr_across_modules"], errors="coerce").lt(cutoff)
        ]
    ranked = ranked.sort_values("epsilon_squared", ascending=False).head(int(top_n))
    ranked["ranking_label"] = ranked.apply(
        lambda row: f"{row['tissue']} regional M{int(row['module'])}", axis=1
    )
    figure = px.bar(
        ranked.iloc[::-1], x="epsilon_squared", y="ranking_label",
        orientation="h", color="tissue",
        hover_data={
            "categorical_p": ":.3g", "categorical_fdr_across_modules": ":.3g",
            "categorical_fdr_module_family_n": True,
        },
        title="Most differentiated independent regional modules",
        labels={"epsilon_squared": "Kruskal–Wallis epsilon-squared", "ranking_label": ""},
    )
    figure.update_layout(template="plotly_white", height=max(480, 170 + 28 * len(ranked)))
    render_plotly_chart(figure, key="regional_distribution_ranking")
    filterable_dataframe(
        ranked,
        table_key="regional_distribution_ranking_table",
        table_name="Regional differentiated-module ranking",
    )


def _render_correlation_heatmaps(
    *,
    source: str,
    source_label: str,
    cohort_scope: str,
    score_variant: str,
    network_method: str,
    scale: str,
    tissues: tuple[str, ...],
    selected_modules: dict[str, int],
    diagnoses: tuple[str, ...],
    metadata: pd.DataFrame,
) -> None:
    with st.sidebar:
        association_type = st.radio(
            "Association type",
            ["Numeric correlations", "Nominal category comparison"],
            horizontal=True,
            key="regional_heatmap_association_type",
        )
        minimum_group_n = st.selectbox(
            "Minimum category size", [5, 10, 20], index=1,
            key="regional_heatmap_minimum_n",
        )
        heatmap_scope = st.radio(
            "Correlation heatmap scope",
            ["Selected regional modules", "All regional modules"],
            horizontal=True,
            key="regional_heatmap_scope",
        )
        clustering = st.multiselect(
            "Cluster heatmap",
            ["Rows", "Columns"],
            default=[],
            key="regional_heatmap_clustering",
        )

    feature_label = score_label(score_variant, network_method)
    if association_type == "Numeric correlations":
        with st.sidebar:
            grouping_variable = st.selectbox(
                "Group correlations by",
                options=list(ASSOCIATION_GROUP_LABELS),
                format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                key="regional_heatmap_grouping",
            )
            levels = _ordered_levels(metadata, grouping_variable)
            selected_levels = st.multiselect(
                "Heatmap group levels", levels, default=levels,
                format_func=lambda value: association_level_label(grouping_variable, value),
                disabled=grouping_variable == "__all__",
                key="regional_heatmap_levels",
            )
            if grouping_variable == "__all__":
                selected_levels = ["__all__"]
            correlation_method = st.radio(
                "Heatmap correlation", ["Spearman", "Pearson"],
                horizontal=True, key="regional_heatmap_correlation",
            )
            significant_only = st.checkbox(
                "Show only rows with FDR < 0.05",
                key="regional_heatmap_significant_only",
            )
        if not selected_levels:
            st.warning("Select at least one heatmap group level.")
            return
        catalogs = [
            regional_correlation_catalog(
                source, cohort_scope, score_variant, network_method, tissue,
                scale, diagnoses, grouping_variable, tuple(selected_levels),
                int(minimum_group_n), tuple(NUMERIC_OUTCOMES), False,
            )
            for tissue in tissues
        ]
        table = pd.concat(catalogs, ignore_index=True)
        if heatmap_scope.startswith("Selected"):
            table = table.loc[
                table.apply(
                    lambda row: int(row["module"]) == int(
                        selected_modules[str(row["tissue"])]
                    ),
                    axis=1,
                )
            ].copy()
        table["heatmap_row"] = table.apply(
            lambda row: (
                f"{row['tissue']} regional M{int(row['module'])} · "
                f"{row['grouping_label']}"
            ),
            axis=1,
        )
        value_column = "spearman_rho" if correlation_method == "Spearman" else "pearson_r"
        p_column = "spearman_p" if correlation_method == "Spearman" else "pearson_p"
        fdr_column = (
            "spearman_fdr_across_modules"
            if correlation_method == "Spearman" else "pearson_fdr_across_modules"
        )
        if significant_only:
            significant_rows = table.loc[
                pd.to_numeric(table[fdr_column], errors="coerce").lt(0.05),
                "heatmap_row",
            ].unique()
            table = table.loc[table["heatmap_row"].isin(significant_rows)].copy()
        if heatmap_scope.startswith("All") and not table.empty:
            top_n = st.selectbox(
                "Rows displayed in heatmap",
                [50, 100, 250, table["heatmap_row"].nunique()],
                key="regional_heatmap_top_n",
            )
            strongest = (
                table.assign(_abs=pd.to_numeric(table[value_column], errors="coerce").abs())
                .groupby("heatmap_row", observed=True)["_abs"].max()
                .nlargest(int(top_n)).index
            )
            display = table.loc[table["heatmap_row"].isin(strongest)].copy()
        else:
            display = table
        if display.empty:
            st.info("No regional correlations meet the selected filters.")
        else:
            row_order = display["heatmap_row"].drop_duplicates().tolist()
            figure = correlation_heatmap_figure(
                display,
                value_column=value_column,
                p_column=p_column,
                fdr_column=fdr_column,
                title=(
                    f"{source_label}: {feature_label} versus numeric outcomes"
                    "<br><sup>* tissue-specific module-family FDR &lt; 0.05</sup>"
                ),
                row_order=row_order,
                cluster_rows="Rows" in clustering,
                cluster_columns="Columns" in clustering,
            )
            render_plotly_chart(figure, key="regional_correlation_heatmap")
        table = table.copy()
        table["correlation_method"] = correlation_method
        table["correlation"] = pd.to_numeric(table[value_column], errors="coerce")
        table["p_value"] = pd.to_numeric(table[p_column], errors="coerce")
        table["fdr"] = pd.to_numeric(table[fdr_column], errors="coerce")
        filterable_dataframe(
            table,
            table_key="regional_complete_correlation_table",
            table_name="Complete regional correlation table",
        )
        _download_table(
            "Download complete regional correlation table (TSV)",
            table,
            "regional_module_correlations.tsv",
            "regional_correlation_download",
        )
        st.caption(
            "Each FDR is adjusted across modules within one tissue-specific partition. "
            "AC, DLPFC, PCG, and grouping levels are separate FDR families."
        )
        return

    with st.sidebar:
        category_variable = st.selectbox(
            "Category",
            [value for value in ASSOCIATION_GROUP_LABELS if value != "__all__"],
            format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
            key="regional_heatmap_category",
        )
        levels = _ordered_levels(metadata, category_variable)
        selected_levels = st.multiselect(
            "Category levels", levels, default=levels,
            format_func=lambda value: association_level_label(category_variable, value),
            key="regional_heatmap_category_levels",
        )
    if not selected_levels:
        st.warning("Select at least two category levels.")
        return
    catalogs = [
        regional_categorical_catalog(
            source, cohort_scope, score_variant, network_method, tissue,
            scale, diagnoses, category_variable, tuple(selected_levels),
            int(minimum_group_n),
        )
        for tissue in tissues
    ]
    table = pd.concat(catalogs, ignore_index=True)
    if heatmap_scope.startswith("Selected"):
        table = table.loc[
            table.apply(
                lambda row: int(row["module"]) == int(
                    selected_modules[str(row["tissue"])]
                ),
                axis=1,
            )
        ].copy()
    table["heatmap_row"] = table.apply(
        lambda row: f"{row['tissue']} regional M{int(row['module'])}", axis=1
    )
    table["heatmap_column"] = table["feature_label"]
    table["component_label"] = table["tissue"]
    if heatmap_scope.startswith("All") and not table.empty:
        top_n = st.selectbox(
            "Rows displayed in heatmap", [20, 50, 100, table["heatmap_row"].nunique()],
            key="regional_nominal_heatmap_top_n",
        )
        strongest = (
            table.groupby("heatmap_row", observed=True)["epsilon_squared"]
            .max().nlargest(int(top_n)).index
        )
        display = table.loc[table["heatmap_row"].isin(strongest)].copy()
    else:
        display = table
    if display.empty:
        st.info("No categorical regional associations meet the selected filters.")
    else:
        figure = distribution_feature_heatmap_figure(
            display,
            title=(
                f"{source_label}: {ASSOCIATION_GROUP_LABELS[category_variable]} differentiation"
                "<br><sup>* tissue-specific module-family FDR &lt; 0.05</sup>"
            ),
            row_order=display["heatmap_row"].drop_duplicates().tolist(),
            column_order=display["heatmap_column"].drop_duplicates().tolist(),
        )
        render_plotly_chart(figure, key="regional_nominal_heatmap")
    filterable_dataframe(
        table,
        table_key="regional_nominal_heatmap_table",
        table_name="Regional nominal-association table",
    )


def render_regional_analysis_view(active_view: str) -> None:
    """Render one of the three regional-module exploratory views."""

    if active_view not in REGIONAL_ANALYSIS_VIEWS:
        raise ValueError(f"Regional modules are unavailable for {active_view}")
    if not regional_data_available():
        st.info(
            "Independent regional-module data are not present in this deployment yet. "
            "The existing multi-tissue analyses remain available."
        )
        return

    manifest = load_regional_manifest()
    completed = completed_combinations(manifest)
    if completed.empty:
        st.info("No validated regional-module result combination is available yet.")
        return
    details = load_regional_details()
    kegg = load_regional_kegg()
    with st.sidebar:
        st.subheader("Independent regional modules")
        source_options = [
            value for value in REGIONAL_SOURCE_ORDER
            if value in set(completed["partition_source"])
        ]
        source = st.selectbox(
            "Regional partition source",
            source_options,
            format_func=lambda value: manifest.get("source_labels", {}).get(value, value),
            key="regional_partition_source",
        )
        source_label = manifest.get("source_labels", {}).get(source, source)
        source_rows = completed.loc[completed["partition_source"].eq(source)]
        cohort_options = [
            value for value in ("common_450", "maximum_tissue")
            if value in set(source_rows["cohort_scope"])
        ]
        cohort_scope = st.selectbox(
            "Regional donor cohort",
            cohort_options,
            format_func=lambda value: COHORT_LABELS.get(value, value),
            key="regional_cohort_scope",
        )
        cohort_rows = source_rows.loc[source_rows["cohort_scope"].eq(cohort_scope)]
        variants = cohort_rows[
            ["score_variant", "network_method"]
        ].drop_duplicates().itertuples(index=False, name=None)
        variant_options = list(variants)
        variant_options.sort(
            key=lambda value: list(SCORE_LABELS).index(value)
            if value in SCORE_LABELS else len(SCORE_LABELS)
        )
        score_variant, network_method = st.selectbox(
            "Regional module score",
            variant_options,
            format_func=lambda value: score_label(*value),
            key="regional_score_variant",
        )
        score_rows = cohort_rows.loc[
            cohort_rows["score_variant"].eq(score_variant)
            & cohort_rows["network_method"].eq(network_method)
        ]
        available_tissues = [
            tissue for tissue in REGION_ORDER if tissue in set(score_rows["tissue"])
        ]
        tissues = tuple(
            st.multiselect(
                "Regions shown side by side",
                available_tissues,
                default=available_tissues,
                key="regional_tissues",
            )
        )
        if not tissues:
            st.warning("Select at least one region.")
            return
        selected_modules: dict[str, int] = {}
        for tissue in tissues:
            modules, labels, default_index = _module_options(details, source, tissue)
            selected_modules[tissue] = int(
                st.selectbox(
                    f"{tissue} module",
                    modules,
                    index=default_index,
                    format_func=lambda value, lookup=labels: lookup[int(value)],
                    key=f"regional_module_{tissue}",
                )
            )
        scale = st.selectbox(
            "Regional score scale",
            list(SCALE_LABELS),
            format_func=lambda value: SCALE_LABELS[value],
            key="regional_score_scale",
        )
        metadata = load_sample_metadata() if cohort_scope == "common_450" else pd.concat(
            [
                pd.read_parquet(
                    ensure_data_path(
                        REGIONAL_ROOT / "metadata" / cohort_scope / f"{tissue}.parquet"
                    )
                )
                for tissue in tissues
            ],
            ignore_index=True,
        ).drop_duplicates("sample_id")
        diagnosis_options = [
            value for value in [*DIAGNOSIS_ORDER, "Unclassified"]
            if value in set(metadata["diagnosis_group"].dropna())
        ]
        diagnoses = tuple(
            st.multiselect(
                "Diagnosis groups",
                diagnosis_options,
                default=[value for value in DIAGNOSIS_ORDER if value in diagnosis_options],
                key="regional_diagnoses",
            )
        )
    if not diagnoses:
        st.warning("Select at least one diagnosis group.")
        return

    st.subheader("Independent single-region SE2 modules")
    st.caption(
        f"{source_label} · {COHORT_LABELS.get(cohort_scope, cohort_scope)} · "
        f"{score_label(score_variant, network_method)}. Module IDs are tissue-scoped; "
        "the three selected modules do not imply cross-tissue correspondence."
    )
    if manifest.get("status") != "complete":
        st.info(
            "Incremental regional release: only validated combinations are selectable. "
            "Additional cohort and LIONESS combinations will appear as they finish."
        )

    common = dict(
        source=source,
        source_label=str(source_label),
        cohort_scope=cohort_scope,
        score_variant=score_variant,
        network_method=network_method,
        scale=scale,
        tissues=tissues,
        selected_modules=selected_modules,
        diagnoses=diagnoses,
        metadata=metadata,
    )
    if active_view == "Associations":
        _render_associations(**common, kegg=kegg)
    elif active_view == "Feature distributions":
        _render_distributions(**common)
    else:
        _render_correlation_heatmaps(**common)

