"""Bounded readers and display helpers for the packaged public data."""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LIONESS_APP_DATA_DIR", APP_ROOT / "data")).resolve()

MODULE_SET_LABELS = {
    "full_cohort": "Full-cohort L4 modules (154)",
    "control_derived": "Control-derived L4 modules (186)",
}
MODULE_SET_DIRS = {
    "full_cohort": DATA_DIR,
    "control_derived": DATA_DIR / "control_derived",
}
MODULE_SET_METHODS = {
    "full_cohort": ("standard", "control_anchored"),
    "control_derived": ("control_anchored",),
}
ESTIMATOR_LABELS = {
    "lioness": "LIONESS",
    "bonobo": "BONOBO",
}
BONOBO_METHOD = "bonobo"
BONOBO_EDGE_RULE_LABELS = {
    "native_p05": "Native posterior p < 0.05",
    "bh_fdr05": "Within donor-module BH FDR < 0.05",
}

KEGG_REGION_FDR_COLUMNS = {
    "AC": "fdr_AC",
    "DLPFC": "fdr_DLPFC",
    "PCG": "fdr_PCGBA23",
}
KEGG_COMPONENT_REGIONS = {
    "TS_AC": ("AC",),
    "TS_DLPFC": ("DLPFC",),
    "TS_PCGBA23": ("PCG",),
    "CT_AC__DLPFC": ("AC", "DLPFC"),
    "CT_AC__PCGBA23": ("AC", "PCG"),
    "CT_DLPFC__PCGBA23": ("DLPFC", "PCG"),
}
KEGG_PRIORITY_PATTERN = (
    r"lipid|fatty acid|cholesterol|glycerolipid|glycerophospholipid|"
    r"sphingolipid|metaboli|alzheimer|infection|infectious|viral|virus"
)

AGGREGATE_DATA = DATA_DIR / "aggregate_plot_data.parquet"
RESOLVED_DATA = DATA_DIR / "resolved_plot_data.parquet"
AGGREGATE_STATS = DATA_DIR / "aggregate_statistics.parquet"
RESOLVED_STATS = DATA_DIR / "resolved_statistics.parquet"
KEGG_PARQUET = DATA_DIR / "kegg_tissue_expanded_full.parquet"
KEGG_TSV = DATA_DIR / "kegg_tissue_expanded_full.tsv"
MODULE_ANNOTATIONS = DATA_DIR / "module_kegg_annotations.tsv"
MODULE_DETAILS = DATA_DIR / "module_details.tsv"
FEATURE_DEFINITIONS = DATA_DIR / "feature_definitions.tsv"
TISSUE_MAPPING = DATA_DIR / "tissue_mapping.tsv"
SAMPLE_METADATA = DATA_DIR / "sample_metadata.parquet"
MDC_SUMMARY = DATA_DIR / "mdc_ad_vs_control_summary.tsv"
MDC_RESOLVED = DATA_DIR / "mdc_resolved_ad_vs_control.tsv"
DATA_MANIFEST = DATA_DIR / "data_manifest.json"

MODULE_SET_FILENAMES = (
    "aggregate_plot_data.parquet",
    "resolved_plot_data.parquet",
    "aggregate_statistics.parquet",
    "resolved_statistics.parquet",
    "kegg_tissue_expanded_full.parquet",
    "kegg_tissue_expanded_full.tsv",
    "module_kegg_annotations.tsv",
    "module_details.tsv",
    "mdc_ad_vs_control_summary.tsv",
    "mdc_resolved_ad_vs_control.tsv",
)

METHOD_LABELS = {
    "standard": "Standard LIONESS (all-donor reference)",
    "control_anchored": "Control-referenced LIONESS",
    "bonobo": "All-donor empirical-Bayes BONOBO",
}

PHENOTYPE_LABELS = {
    "cogn_global": "Global cognition",
    "cogng_demog_slope": "Demographic-adjusted cognitive slope",
    "cogng_path_slope": "Pathology-adjusted cognitive slope",
    "motor10_demog_slope": "Demographic-adjusted motor slope",
    "sqrt_parksc_demog_slope": "Demographic-adjusted Parkinsonian score",
}

OUTCOME_LABELS = {
    **PHENOTYPE_LABELS,
    "age_at_death": "Age at death",
    "education_years": "Years of education",
    "cogdx": "Final cognitive diagnosis code (CogDx)",
    "braak_stage": "Braak stage",
    "cerad_score": "CERAD score",
    "adnc": "AD neuropathologic change (ADNC)",
    "parkinsonism": "Parkinsonism",
}

HOVER_LABELS = {
    **OUTCOME_LABELS,
    "sex_code": "Sex code",
    "apoe_genotype": "APOE genotype",
    "parkinsonism_label": "Parkinsonism status",
}

NUMERIC_OUTCOMES = list(OUTCOME_LABELS)
COLOR_LABELS = {"diagnosis_group": "Diagnosis group", **OUTCOME_LABELS}

FEATURE_LABELS = {
    "connectivity": "Connectivity",
    "positive_density": "Positive density",
    "negative_density": "Negative density",
    "abs_sum": "Absolute edge-weight sum",
    "positive_abs_sum": "Positive absolute edge-weight sum",
    "negative_abs_sum": "Negative absolute edge-weight sum",
}
BONOBO_FEATURE_LABELS = {
    "connectivity": "Connectivity",
    "positive_density": "Positive density",
    "negative_density": "Negative density",
}

SCALE_LABELS = {
    "rint": "Z-score",
    "asinh": "Asinh-transformed",
    "raw": "Raw",
}

DIAGNOSIS_ORDER = ["Control", "MCI", "AD"]
COMPONENT_ORDER = [
    "CT_AC__DLPFC",
    "CT_AC__PCGBA23",
    "CT_DLPFC__PCGBA23",
    "TS_AC",
    "TS_DLPFC",
    "TS_PCGBA23",
]
EDGE_SCOPE_LABELS = {
    "total": "Total",
    "TS": "TS pooled",
    "CT": "CT pooled",
    "TS_AC": "TS: AC",
    "TS_DLPFC": "TS: DLPFC",
    "TS_PCGBA23": "TS: PCG",
    "CT_AC__DLPFC": "CT: AC - DLPFC",
    "CT_AC__PCGBA23": "CT: AC - PCG",
    "CT_DLPFC__PCGBA23": "CT: DLPFC - PCG",
}


def module_set_data_dir(module_set: str = "full_cohort") -> Path:
    """Return the isolated deploy directory for a module definition."""
    if module_set not in MODULE_SET_DIRS:
        raise ValueError(
            f"Unknown module definition {module_set!r}; expected one of "
            f"{sorted(MODULE_SET_DIRS)}"
        )
    return MODULE_SET_DIRS[module_set]


def module_set_path(filename: str, module_set: str = "full_cohort") -> Path:
    return module_set_data_dir(module_set) / filename


def estimator_path(
    filename: str,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> Path:
    if estimator == "lioness":
        return module_set_path(filename, module_set)
    if estimator != "bonobo":
        raise ValueError(f"Unknown estimator: {estimator}")
    if edge_rule not in {"all", *BONOBO_EDGE_RULE_LABELS}:
        raise ValueError(f"Unknown BONOBO edge rule: {edge_rule}")
    return module_set_data_dir(module_set) / "bonobo" / edge_rule / filename


def require_data_files() -> None:
    """Raise a useful error if the GitHub data bundle was not built."""
    required = [
        FEATURE_DEFINITIONS,
        TISSUE_MAPPING,
        SAMPLE_METADATA,
        DATA_MANIFEST,
    ]
    for module_set in MODULE_SET_DIRS:
        required.extend(
            module_set_path(filename, module_set) for filename in MODULE_SET_FILENAMES
        )
        for edge_rule in ("all", *BONOBO_EDGE_RULE_LABELS):
            required.extend(
                estimator_path(filename, module_set, "bonobo", edge_rule)
                for filename in (
                    "aggregate_plot_data.parquet",
                    "resolved_plot_data.parquet",
                    "aggregate_statistics.parquet",
                    "resolved_statistics.parquet",
                )
            )
            required.append(
                module_set_data_dir(module_set)
                / "edge_summaries"
                / f"bonobo__{edge_rule}.parquet"
            )
        required.extend(
            module_set_data_dir(module_set)
            / "edge_summaries"
            / f"lioness__{method}.parquet"
            for method in MODULE_SET_METHODS[module_set]
        )
    missing = [
        str(path.relative_to(APP_ROOT)) if path.is_relative_to(APP_ROOT) else str(path)
        for path in required
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The deploy data bundle is incomplete. Run "
            "`python scripts/build_public_data.py` before deployment. Missing: "
            + ", ".join(missing)
        )


def _read_filtered(
    path: Path,
    filters: list[tuple[str, str, object]],
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    table = pq.read_table(
        path,
        filters=filters or None,
        columns=list(columns) if columns else None,
    )
    return table.to_pandas()


def load_aggregate(
    method: str,
    module: int,
    metric_family: str,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "module",
        "metric_family",
        "CT_raw",
        "TS_raw",
        "CT_asinh",
        "TS_asinh",
        "CT_rint",
        "TS_rint",
        "diagnosis_group",
        *PHENOTYPE_LABELS,
        "lioness_method",
    ]
    return _read_filtered(
        estimator_path(
            "aggregate_plot_data.parquet", module_set, estimator, edge_rule
        ),
        [
            ("lioness_method", "=", method),
            ("module", "=", int(module)),
            ("metric_family", "=", metric_family),
        ],
        columns,
    )


def load_aggregate_scope(
    method: str,
    module: int | None = None,
    metric_family: str | None = None,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    columns = [
        "sample_id",
        "module",
        "metric_family",
        "CT_rint",
        "TS_rint",
        "diagnosis_group",
        *PHENOTYPE_LABELS,
        "lioness_method",
    ]
    return _read_filtered(
        estimator_path(
            "aggregate_plot_data.parquet", module_set, estimator, edge_rule
        ), filters, columns
    )


def load_resolved(
    method: str,
    module: int,
    metric_family: str,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "module",
        "metric_family",
        "component",
        "component_class",
        "component_label",
        "metric_raw",
        "metric_asinh",
        "metric_rint",
        "diagnosis_group",
        *PHENOTYPE_LABELS,
        "lioness_method",
    ]
    return _read_filtered(
        estimator_path(
            "resolved_plot_data.parquet", module_set, estimator, edge_rule
        ),
        [
            ("lioness_method", "=", method),
            ("module", "=", int(module)),
            ("metric_family", "=", metric_family),
        ],
        columns,
    )


def load_resolved_scope(
    method: str,
    module: int | None = None,
    metric_family: str | None = None,
    component: str | None = None,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    if component is not None:
        filters.append(("component", "=", component))
    columns = [
        "sample_id",
        "module",
        "metric_family",
        "component",
        "component_class",
        "component_label",
        "metric_rint",
        "diagnosis_group",
        *PHENOTYPE_LABELS,
        "lioness_method",
    ]
    return _read_filtered(
        estimator_path(
            "resolved_plot_data.parquet", module_set, estimator, edge_rule
        ), filters, columns
    )


def load_aggregate_statistics(
    method: str,
    module: int | None = None,
    phenotype: str | None = None,
    metric_family: str | None = None,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    return _read_filtered(
        estimator_path(
            "aggregate_statistics.parquet", module_set, estimator, edge_rule
        ), filters
    )


def load_resolved_statistics(
    method: str,
    module: int | None = None,
    phenotype: str | None = None,
    metric_family: str | None = None,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    return _read_filtered(
        estimator_path(
            "resolved_statistics.parquet", module_set, estimator, edge_rule
        ), filters
    )


def load_module_annotations(module_set: str = "full_cohort") -> pd.DataFrame:
    return pd.read_csv(module_set_path("module_kegg_annotations.tsv", module_set), sep="\t")


def load_module_details(
    module: int | None = None, module_set: str = "full_cohort"
) -> pd.DataFrame:
    details = pd.read_csv(module_set_path("module_details.tsv", module_set), sep="\t")
    details = ensure_tissue_entropy(details)
    if module is not None:
        details = details.loc[details["module"].astype(int).eq(int(module))]
    return details.reset_index(drop=True)


def ensure_tissue_entropy(details: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct tissue entropy when a hot deploy briefly exposes legacy details."""
    required = ("tissue_entropy", "tissue_entropy_normalized")
    if all(column in details.columns for column in required):
        return details
    proportion_columns = ("proportion_ac", "proportion_dlpfc", "proportion_pcg")
    missing = set(proportion_columns).difference(details.columns)
    if missing:
        raise ValueError(
            "Module details lack tissue entropy and the tissue proportions needed to "
            f"reconstruct it: {sorted(missing)}"
        )
    result = details.copy()
    proportions = (
        result[list(proportion_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    log_proportions = np.zeros_like(proportions)
    positive = proportions > 0
    np.log2(proportions, out=log_proportions, where=positive)
    entropy = -(proportions * log_proportions).sum(axis=1)
    result["tissue_entropy"] = entropy
    result["tissue_entropy_normalized"] = entropy / np.log2(3.0)
    return result


def load_feature_definitions() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DEFINITIONS, sep="\t")


def load_tissue_mapping() -> pd.DataFrame:
    return pd.read_csv(TISSUE_MAPPING, sep="\t")


def load_sample_metadata() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE_METADATA)


def load_mdc_summary(module_set: str = "full_cohort") -> pd.DataFrame:
    return pd.read_csv(
        module_set_path("mdc_ad_vs_control_summary.tsv", module_set), sep="\t"
    )


def load_mdc_resolved(module_set: str = "full_cohort") -> pd.DataFrame:
    return pd.read_csv(
        module_set_path("mdc_resolved_ad_vs_control.tsv", module_set), sep="\t"
    )


def load_edge_summaries(
    estimator: str,
    method: str,
    module: int,
    module_set: str = "full_cohort",
    edge_rule: str = "all",
) -> pd.DataFrame:
    if estimator == "lioness":
        path = module_set_data_dir(module_set) / "edge_summaries" / f"lioness__{method}.parquet"
    elif estimator == "bonobo":
        path = module_set_data_dir(module_set) / "edge_summaries" / f"bonobo__{edge_rule}.parquet"
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    frame = _read_filtered(path, [("module", "=", int(module))])
    if "scope" in frame:
        frame["scope"] = (
            frame["scope"].astype("string").str.replace("MFBA9BA46", "DLPFC", regex=False)
        )
        frame["scope_label"] = frame["scope"].map(EDGE_SCOPE_LABELS)
    return frame


def load_data_manifest() -> dict[str, object]:
    return json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))


def load_kegg(
    module: int | None = None, module_set: str = "full_cohort"
) -> pd.DataFrame:
    filters = [("cluster_id", "=", int(module))] if module is not None else []
    return _read_filtered(
        module_set_path("kegg_tissue_expanded_full.parquet", module_set), filters
    )


def filter_kegg_enrichments(
    frame: pd.DataFrame,
    modules: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
    subcategories: Iterable[str] | None = None,
    significance: str = "all",
    maximum_fdr: float | None = None,
    search: str = "",
    significance_columns: Iterable[str] | None = None,
    fdr_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter and consistently order selected- or all-module KEGG rows."""
    if significance not in {"all", "significant", "not_significant"}:
        raise ValueError(
            "significance must be 'all', 'significant', or 'not_significant'"
        )

    result = frame.copy()
    selected_modules = [int(value) for value in modules or []]
    if selected_modules:
        result = result.loc[
            pd.to_numeric(result["cluster_id"], errors="coerce").isin(selected_modules)
        ]

    selected_categories = [str(value) for value in categories or []]
    if selected_categories:
        result = result.loc[result["category_level1"].isin(selected_categories)]

    selected_subcategories = [str(value) for value in subcategories or []]
    if selected_subcategories:
        result = result.loc[result["category_level2"].isin(selected_subcategories)]

    selected_significance_columns = list(significance_columns or ["significant"])
    selected_fdr_columns = list(fdr_columns or ["fdr"])
    missing_scope_columns = set(
        selected_significance_columns + selected_fdr_columns
    ).difference(result.columns)
    if missing_scope_columns:
        raise ValueError(
            "KEGG significance scope columns are missing: "
            f"{sorted(missing_scope_columns)}"
        )
    significant = (
        result[selected_significance_columns].fillna(False).astype(bool).any(axis=1)
    )
    if significance == "significant":
        result = result.loc[significant]
    elif significance == "not_significant":
        result = result.loc[~significant]

    scope_fdr = result[selected_fdr_columns].apply(pd.to_numeric, errors="coerce").min(
        axis=1
    )
    if maximum_fdr is not None:
        maximum_fdr = float(maximum_fdr)
        if not 0.0 <= maximum_fdr <= 1.0:
            raise ValueError("maximum_fdr must be between 0 and 1")
        result = result.loc[scope_fdr.le(maximum_fdr)]
        scope_fdr = scope_fdr.loc[result.index]

    search = search.strip()
    if search:
        search_columns = [
            column
            for column in [
                "cluster_id",
                "term",
                "pathway_id",
                "pathway_name",
                "category_level1",
                "category_level2",
                "overlap_genes",
            ]
            if column in result.columns
        ]
        searchable = result[search_columns].fillna("").astype(str)
        mask = searchable.agg(" ".join, axis=1).str.contains(
            search, case=False, regex=False
        )
        result = result.loc[mask]

    result = result.assign(_selected_scope_fdr=scope_fdr.reindex(result.index))
    sort_columns = ["_selected_scope_fdr"] + [
        column for column in ["p", "cluster_id", "pathway_name"] if column in result
    ]
    return (
        result.sort_values(sort_columns, na_position="last")
        .drop(columns="_selected_scope_fdr")
        .reset_index(drop=True)
    )


def module_label(module: int | str) -> str:
    return f"M{int(module)}"


def selected_annotation(annotations: pd.DataFrame, module: int) -> str | None:
    match = annotations.loc[annotations["module"].astype(int).eq(int(module))]
    if match.empty or not bool(match.iloc[0].get("annotation_available", False)):
        return None
    text = match.iloc[0].get("subtitle_text")
    return str(text) if pd.notna(text) and str(text).strip() else None


def _format_kegg_fdr(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(numeric) else f"{float(numeric):.3g}"


def _select_resolved_kegg_row(
    enrichments: pd.DataFrame, regions: tuple[str, ...]
) -> pd.Series | None:
    fdr_columns = [KEGG_REGION_FDR_COLUMNS[region] for region in regions]
    if enrichments.empty or any(column not in enrichments for column in fdr_columns):
        return None
    candidates = enrichments.copy()
    regional_fdr = candidates[fdr_columns].apply(pd.to_numeric, errors="coerce")
    valid = regional_fdr.notna().all(axis=1)
    if not valid.any():
        return None
    candidates = candidates.loc[valid].copy()
    regional_fdr = regional_fdr.loc[valid]
    candidates["_joint_regional_score"] = regional_fdr.max(axis=1)
    searchable = candidates[
        [
            column
            for column in ("category_level1", "category_level2", "pathway_name")
            if column in candidates
        ]
    ].fillna("").astype(str).agg(" ".join, axis=1)
    priority_significant = searchable.str.contains(
        KEGG_PRIORITY_PATTERN, case=False, regex=True
    ) & regional_fdr.le(0.05).all(axis=1)
    selection = candidates.loc[priority_significant]
    if selection.empty:
        selection = candidates
    sort_columns = ["_joint_regional_score"] + [
        column for column in ("p", "pathway_name") if column in selection
    ]
    return selection.sort_values(sort_columns, na_position="last").iloc[0]


def association_kegg_subtitles(
    enrichments: pd.DataFrame,
    components: Iterable[str],
    *,
    resolved: bool,
    aggregate_annotation: str | None = None,
) -> dict[str, str]:
    """Return one scope-matched KEGG subtitle for every association panel.

    Aggregate CT/TS panels use the existing tissue-expanded annotation. Within-tissue
    panels use that region's enrichment FDR. Cross-tissue panels select the pathway
    with the strongest conservative joint regional support and report both regional
    FDR values; these values are not represented as a separate pair-level test.
    """
    component_list = [str(component) for component in components]
    if not resolved:
        if aggregate_annotation:
            subtitle = str(aggregate_annotation).replace(
                "KEGG enrichment:", "KEGG enrichment (tissue-expanded):", 1
            )
        else:
            subtitle = "KEGG enrichment (tissue-expanded): unavailable"
        return {component: subtitle for component in component_list}

    subtitles: dict[str, str] = {}
    for component in component_list:
        regions = KEGG_COMPONENT_REGIONS.get(component)
        if not regions:
            subtitles[component] = "KEGG enrichment: unavailable"
            continue
        scope_label = "–".join(regions)
        row = _select_resolved_kegg_row(enrichments, regions)
        if row is None:
            subtitles[component] = f"KEGG enrichment ({scope_label}): unavailable"
            continue
        category = str(row.get("category_level1", "Unavailable"))
        subcategory = str(row.get("category_level2", "Unavailable"))
        pathway = str(row.get("pathway_name", "Unavailable")).replace(
            " - Homo sapiens (human)", ""
        )
        if len(regions) == 1:
            region = regions[0]
            fdr_text = f"FDR={_format_kegg_fdr(row[KEGG_REGION_FDR_COLUMNS[region]])}"
        else:
            fdr_text = "; ".join(
                f"{region} FDR={_format_kegg_fdr(row[KEGG_REGION_FDR_COLUMNS[region]])}"
                for region in regions
            )
            scope_label += " regional support"
        subtitles[component] = (
            f"KEGG enrichment ({scope_label}): {category} / {subcategory} / "
            f"{pathway} | {fdr_text}"
        )
    return subtitles


def dataframe_to_tsv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, sep="\t", index=False)
    return buffer.getvalue().encode("utf-8")
