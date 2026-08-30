"""Bounded readers and display helpers for the packaged public data."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app_helpers.drive_data import DATA_DIR, data_path_available, ensure_data_path
from app_helpers.gene_symbols import public_gene_labels


APP_ROOT = Path(__file__).resolve().parents[1]

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
DIFFERENTIAL_EDGE_RULE_LABELS = {
    "all": "All edges",
    "ad_control_discovery_fdr05": "AD–Control differential-filtered edges",
}
FDR_SCOPE_LABELS = {
    "global": "Global BH",
    "per_module": "Per-module BH",
}
FDR_THRESHOLD_LABELS = {
    0.05: "FDR < 0.05",
    0.10: "FDR < 0.10 (exploratory)",
}
SCORE_NORMALIZATION_LABELS = {
    "standard_pruned": "Standard pruned network",
    "retained_edge": "Retained-edge normalized",
}
ANALYSIS_SUBSET_LABELS = {
    "all_donors": "All donors (exploratory)",
    "discovery_ad_control": "Discovery AD + Control",
    "validation_ad_control": "Held-out validation AD + Control",
    "mci_external": "MCI (external to edge selection)",
}

KEGG_REGION_FDR_COLUMNS = {
    "AC": "fdr_AC",
    "DLPFC": "fdr_DLPFC",
    "PCG": "fdr_PCGBA23",
}
KEGG_REGION_P_COLUMNS = {
    "AC": "p_AC",
    "DLPFC": "p_DLPFC",
    "PCG": "p_PCGBA23",
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
PREDICTION_DIR = DATA_DIR / "prediction"
PREDICTION_MANIFEST = PREDICTION_DIR / "prediction_public_manifest.json"
PREDICTION_PERFORMANCE = PREDICTION_DIR / "prediction_performance.parquet"
PREDICTION_BOOTSTRAP = PREDICTION_DIR / "prediction_ct_ts_bootstrap.parquet"
PREDICTION_CURVES = PREDICTION_DIR / "prediction_curves.parquet"
PREDICTION_CONFUSION = PREDICTION_DIR / "prediction_confusion.parquet"
PREDICTION_COEFFICIENTS = PREDICTION_DIR / "prediction_top_coefficients.parquet"
PREDICTION_PREDICTIONS = PREDICTION_DIR / "prediction_diagnostics.parquet"
PREDICTION_WHOLE_NETWORK = PREDICTION_DIR / "prediction_whole_network_features.parquet"
TARGETED_PREDICTION_DIR = DATA_DIR / "prediction_targeted"
TARGETED_PREDICTION_MANIFEST = (
    TARGETED_PREDICTION_DIR / "targeted_prediction_public_manifest.json"
)
TARGETED_PREDICTION_FILES = {
    "fold_performance": TARGETED_PREDICTION_DIR / "targeted_fold_performance.parquet",
    "oof_performance": TARGETED_PREDICTION_DIR / "targeted_oof_performance.parquet",
    "primary_comparisons": TARGETED_PREDICTION_DIR / "targeted_primary_comparisons.parquet",
    "tier_comparisons": TARGETED_PREDICTION_DIR / "targeted_tier_comparisons.parquet",
    "panel_selection": TARGETED_PREDICTION_DIR / "targeted_panel_selection.parquet",
    "consensus_panels": TARGETED_PREDICTION_DIR / "targeted_consensus_panels.parquet",
    "inner_k_scores": TARGETED_PREDICTION_DIR / "targeted_inner_k_scores.parquet",
    "coefficients": TARGETED_PREDICTION_DIR / "targeted_top_coefficients.parquet",
    "transformation_status": TARGETED_PREDICTION_DIR / "targeted_transformation_status.parquet",
    "transformation_comparisons": TARGETED_PREDICTION_DIR / "targeted_transformation_comparisons.parquet",
    "panel_transform_overlap": TARGETED_PREDICTION_DIR / "targeted_panel_transform_overlap.parquet",
    "consensus_transform_overlap": TARGETED_PREDICTION_DIR / "targeted_consensus_transform_overlap.parquet",
    "oof_predictions": TARGETED_PREDICTION_DIR / "targeted_oof_predictions.parquet",
    "masked_fold_performance": TARGETED_PREDICTION_DIR / "targeted_masked_fold_performance.parquet",
    "masked_oof_performance": TARGETED_PREDICTION_DIR / "targeted_masked_oof_performance.parquet",
    "masked_coefficients": TARGETED_PREDICTION_DIR / "targeted_masked_top_coefficients.parquet",
    "masked_oof_predictions": TARGETED_PREDICTION_DIR / "targeted_masked_oof_predictions.parquet",
}
CLUSTER_ASSOCIATION_STATS = DATA_DIR / "cluster_association_statistics.parquet"

PREDICTION_OUTCOME_LABELS = {
    "diagnosis_binary": "Diagnosis: AD versus Control",
    "diagnosis_three_class": "Diagnosis: Control / MCI / AD",
    "cogdx": "Final cognitive diagnosis code (CogDx 1–5)",
    "cogn_global": "Global cognition",
    "cogng_demog_slope": "Demographic-adjusted cognitive slope",
    "cogng_path_slope": "Pathology-adjusted cognitive slope",
    "motor10_demog_slope": "Demographic-adjusted motor slope",
    "sqrt_parksc_demog_slope": "Demographic-adjusted Parkinsonian-score slope",
    "parkinsonism": "Parkinsonism",
    "clusters": "ROSMAP donor cluster (nominal 1–4)",
}
PREDICTION_DESIGN_LABELS = {
    "edge_sums": "Whole-network retained-edge sums",
    "module_connectivity": "Module connectivity",
}
PREDICTION_MASK_LABELS = {
    "all": "All edges",
    "global_fdr05": "Global BH FDR < 0.05",
    "global_fdr10": "Global BH FDR < 0.10",
    "per_module_fdr05": "Per-module BH FDR < 0.05",
    "per_module_fdr10": "Per-module BH FDR < 0.10 (exploratory)",
}
SCORE_TRANSFORM_LABELS = {
    "raw": "Raw (primary)",
    "asinh": "asinh (robustness)",
    "rint": "RINT (robustness)",
}
PREDICTION_MODEL_LABELS = {
    "dummy": "Dummy / intercept baseline",
    "covariates": "Demographics + APOE baseline",
    "network_only": "Network only",
    "covariates_plus_network": "Demographics + APOE + network",
    "transcriptomics_only": "Transcriptomics only (module eigengenes)",
    "covariates_plus_transcriptomics": "Demographics + APOE + transcriptomics",
    "network_plus_transcriptomics": "Network + transcriptomics",
    "covariates_plus_network_plus_transcriptomics": (
        "Demographics + APOE + network + transcriptomics"
    ),
}
PREDICTION_REFERENCE_LABELS = {
    "development_frozen": "Leakage-reduced frozen development reference",
    "existing_sensitivity": "Exploratory existing all-donor/all-Control reference",
}
PREDICTION_BLOCK_ORDER = (
    "AC",
    "PCG",
    "DLPFC",
    "AC_PCG",
    "DLPFC_PCG",
    "AC_DLPFC",
    "TS_pooled",
    "CT_pooled",
    "CT_TS_pooled",
    "TS_resolved",
    "CT_resolved",
    "all_resolved",
)
PREDICTION_BLOCK_LABELS = {
    "AC": "AC",
    "PCG": "PCG",
    "DLPFC": "DLPFC",
    "AC_PCG": "AC–PCG",
    "DLPFC_PCG": "DLPFC–PCG",
    "AC_DLPFC": "AC–DLPFC",
    "TS_pooled": "TS pooled",
    "CT_pooled": "CT pooled",
    "CT_TS_pooled": "CT + TS pooled",
    "TS_resolved": "All three TS tissues",
    "CT_resolved": "All three CT tissue pairs",
    "all_resolved": "All six resolved components",
}
TARGETED_PREDICTION_MODE_LABELS = {
    "benchmark": "All-module benchmark",
    "targeted": "Targeted modules",
}
TARGETED_PANEL_LABELS = {
    "tissue_neutral_ad": "Tissue-neutral stable AD panel",
    "ct_specific_ad": "CT-specific stable AD panel",
    "outcome_specific": "Outcome-specific stable panel",
    "all_modules": "All modules",
}
TARGETED_TIER_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "exploratory": "Exploratory",
}

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

CATEGORICAL_OUTCOME_LABELS = {
    "clusters": "ROSMAP donor cluster (nominal 1–4)",
    "diagnosis_group": "Diagnosis group",
    "sex_code": "Sex code",
    "apoe_genotype": "APOE genotype",
}
ASSOCIATION_OUTCOME_LABELS = {**OUTCOME_LABELS, **CATEGORICAL_OUTCOME_LABELS}

# Association fields that can be used as correlation strata.  The ``__all__``
# sentinel represents one unstratified association after all donor filters.
ASSOCIATION_GROUP_LABELS = {
    "diagnosis_group": "Diagnosis group",
    "__all__": "All displayed donors",
    "clusters": "ROSMAP clusters",
    "cogdx": "CogDx",
    "braak_stage": "Braak stage",
    "cerad_score": "CERAD score",
    "adnc": "ADNC",
    "parkinsonism": "Parkinsonism",
    "sex_code": "Sex code",
    "apoe_genotype": "APOE genotype",
}

CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES = {
    "clusters",
    "diagnosis_group",
    "sex_code",
    "apoe_genotype",
}
SELECTABLE_ASSOCIATION_OUTCOMES = {
    "cogdx",
    "braak_stage",
    "cerad_score",
    "adnc",
    "parkinsonism",
}

ASSOCIATION_LEVEL_ORDERS = {
    "diagnosis_group": ["Control", "MCI", "AD"],
    "clusters": [1, 2, 3, 4],
    "cogdx": [1, 2, 3, 4, 5],
    "braak_stage": [0, 1, 2, 3, 4, 5, 6],
    "cerad_score": [1, 2, 3, 4],
    "adnc": [0, 1, 2, 3],
    "parkinsonism": [0, 1],
    "sex_code": ["Code 0", "Code 1"],
    "apoe_genotype": ["ε2/ε2", "ε2/ε3", "ε2/ε4", "ε3/ε3", "ε3/ε4", "ε4/ε4"],
}


def association_level_label(variable: str, value: object) -> str:
    """Return a stable public label for an association category level."""

    if variable == "__all__":
        return "All displayed donors"
    if pd.isna(value):
        return "Missing"
    if variable == "diagnosis_group":
        return str(value)
    if variable == "clusters":
        return f"Cluster {int(float(value))}"
    if variable == "cogdx":
        return f"CogDx {int(float(value))}"
    if variable == "braak_stage":
        return f"Braak {int(float(value))}"
    if variable == "cerad_score":
        return f"CERAD {int(float(value))}"
    if variable == "adnc":
        return f"ADNC {int(float(value))}"
    if variable == "parkinsonism":
        return "Parkinsonism" if int(float(value)) == 1 else "No Parkinsonism"
    if variable == "sex_code":
        text = str(value)
        return text if text.lower().startswith("sex ") else f"Sex {text.lower()}"
    return str(value)

HOVER_LABELS = {
    **OUTCOME_LABELS,
    **CATEGORICAL_OUTCOME_LABELS,
    "sex_code": "Sex code",
    "apoe_genotype": "APOE genotype",
    "parkinsonism_label": "Parkinsonism status",
}

NUMERIC_OUTCOMES = list(OUTCOME_LABELS)
COLOR_LABELS = {
    "diagnosis_group": "Diagnosis group",
    **OUTCOME_LABELS,
    **CATEGORICAL_OUTCOME_LABELS,
}

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


def differential_estimator_path(
    filename: str,
    module_set: str,
    estimator: str,
    method: str,
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
) -> Path:
    """Return a filtered-score path while preserving legacy all-edge paths."""

    if differential_edge_rule == "all":
        return estimator_path(filename, module_set, estimator, edge_rule)
    if differential_edge_rule not in DIFFERENTIAL_EDGE_RULE_LABELS:
        raise ValueError(f"Unknown differential edge rule: {differential_edge_rule}")
    if differential_fdr_scope not in FDR_SCOPE_LABELS:
        raise ValueError(f"Unknown differential FDR scope: {differential_fdr_scope}")
    threshold = float(differential_fdr_threshold)
    if not any(np.isclose(threshold, value) for value in FDR_THRESHOLD_LABELS):
        raise ValueError(f"Unknown differential FDR threshold: {threshold}")
    if score_normalization not in SCORE_NORMALIZATION_LABELS:
        raise ValueError(f"Unknown score normalization: {score_normalization}")
    return (
        module_set_data_dir(module_set)
        / "differential"
        / estimator
        / method
        / differential_edge_rule
        / differential_fdr_scope
        / f"fdr_{threshold:.2f}"
        / edge_rule
        / score_normalization
        / filename
    )


def differential_data_available(module_set: str = "full_cohort") -> bool:
    return data_path_available(
        module_set_data_dir(module_set)
        / "differential"
        / "volcano_candidates.parquet"
    )


def differential_mdc_data_available(module_set: str = "full_cohort") -> bool:
    """Return whether both generated differential-edge MDC tables are deployed."""

    directory = module_set_data_dir(module_set) / "differential"
    return all(
        data_path_available(directory / filename)
        for filename in (
            "mdc_filtered_summary.parquet",
            "mdc_filtered_resolved.parquet",
        )
    )


def require_data_files() -> None:
    """Raise a useful error if the GitHub data bundle was not built."""
    required = [
        FEATURE_DEFINITIONS,
        TISSUE_MAPPING,
        SAMPLE_METADATA,
        DATA_MANIFEST,
        CLUSTER_ASSOCIATION_STATS,
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
        if not data_path_available(path)
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
    materialized = ensure_data_path(path, filters)
    table = pq.read_table(
        materialized,
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
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
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
    if differential_edge_rule != "all":
        columns.extend(
            [
                "estimator",
                "network_method",
                "edge_rule",
                "differential_edge_rule",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "score_normalization",
                "ad_control_split",
            ]
        )
    return _read_filtered(
        differential_estimator_path(
            "aggregate_plot_data.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
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
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    metric_scale: str | None = None,
    include_embedded_metadata: bool = True,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    metric_columns = (
        [f"CT_{metric_scale}", f"TS_{metric_scale}"]
        if metric_scale is not None
        else ["CT_raw", "TS_raw", "CT_asinh", "TS_asinh", "CT_rint", "TS_rint"]
    )
    columns = [
        "sample_id",
        "module",
        "metric_family",
        *metric_columns,
        "lioness_method",
    ]
    if include_embedded_metadata:
        columns.extend(["diagnosis_group", *PHENOTYPE_LABELS])
    if differential_edge_rule != "all" and include_embedded_metadata:
        columns.extend(
            [
                "estimator",
                "network_method",
                "edge_rule",
                "differential_edge_rule",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "score_normalization",
                "ad_control_split",
            ]
        )
    return _read_filtered(
        differential_estimator_path(
            "aggregate_plot_data.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
        ), filters, columns
    )


def load_resolved(
    method: str,
    module: int,
    metric_family: str,
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    edge_rule: str = "all",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
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
    if differential_edge_rule != "all":
        columns.extend(
            [
                "estimator",
                "network_method",
                "edge_rule",
                "differential_edge_rule",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "score_normalization",
                "ad_control_split",
            ]
        )
    return _read_filtered(
        differential_estimator_path(
            "resolved_plot_data.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
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
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    metric_scale: str | None = None,
    include_embedded_metadata: bool = True,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    if component is not None:
        filters.append(("component", "=", component))
    metric_columns = (
        [f"metric_{metric_scale}"]
        if metric_scale is not None
        else ["metric_raw", "metric_asinh", "metric_rint"]
    )
    columns = [
        "sample_id",
        "module",
        "metric_family",
        "component",
        "component_class",
        "component_label",
        *metric_columns,
        "lioness_method",
    ]
    if include_embedded_metadata:
        columns.extend(["diagnosis_group", *PHENOTYPE_LABELS])
    if differential_edge_rule != "all" and include_embedded_metadata:
        columns.extend(
            [
                "estimator",
                "network_method",
                "edge_rule",
                "differential_edge_rule",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "score_normalization",
                "ad_control_split",
            ]
        )
    return _read_filtered(
        differential_estimator_path(
            "resolved_plot_data.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
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
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
    diagnosis_group: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    if diagnosis_group is not None:
        filters.append(("diagnosis_group", "=", diagnosis_group))
    if differential_edge_rule != "all":
        filters.append(("analysis_subset", "=", analysis_subset))
    return _read_filtered(
        differential_estimator_path(
            "aggregate_statistics.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
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
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
    score_normalization: str = "standard_pruned",
    analysis_subset: str = "all_donors",
    component: str | None = None,
    diagnosis_group: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    if component is not None:
        filters.append(("component", "=", component))
    if diagnosis_group is not None:
        filters.append(("diagnosis_group", "=", diagnosis_group))
    if differential_edge_rule != "all":
        filters.append(("analysis_subset", "=", analysis_subset))
    return _read_filtered(
        differential_estimator_path(
            "resolved_statistics.parquet", module_set, estimator, method, edge_rule,
            differential_edge_rule, differential_fdr_scope,
            differential_fdr_threshold, score_normalization,
        ), filters
    )


def load_module_annotations(module_set: str = "full_cohort") -> pd.DataFrame:
    path = ensure_data_path(
        module_set_path("module_kegg_annotations.tsv", module_set)
    )
    return pd.read_csv(path, sep="\t")


def load_module_details(
    module: int | None = None, module_set: str = "full_cohort"
) -> pd.DataFrame:
    path = ensure_data_path(module_set_path("module_details.tsv", module_set))
    details = pd.read_csv(path, sep="\t")
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
    return pd.read_csv(ensure_data_path(FEATURE_DEFINITIONS), sep="\t")


def load_tissue_mapping() -> pd.DataFrame:
    return pd.read_csv(ensure_data_path(TISSUE_MAPPING), sep="\t")


def load_sample_metadata() -> pd.DataFrame:
    return pd.read_parquet(ensure_data_path(SAMPLE_METADATA))


def load_cluster_association_statistics(
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolution: str,
    metric_family: str | None = None,
    component: str | None = None,
    diagnosis_group: str | None = None,
    edge_rule: str = "all",
) -> pd.DataFrame:
    """Read compact all-edge nominal association statistics."""

    filters: list[tuple[str, str, object]] = [
        ("module_set", "=", module_set),
        ("estimator", "=", estimator),
        ("network_method", "=", method),
        ("resolution", "=", resolution),
        ("edge_rule", "=", edge_rule),
    ]
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    if component is not None:
        filters.append(("component", "=", component))
    if diagnosis_group is not None:
        filters.append(("diagnosis_group", "=", diagnosis_group))
    return _read_filtered(CLUSTER_ASSOCIATION_STATS, filters)


def load_mdc_summary(
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Load all-edge MDC or the selected discovery-filtered MDC summary."""

    if differential_edge_rule == "all":
        path = ensure_data_path(
            module_set_path("mdc_ad_vs_control_summary.tsv", module_set)
        )
        return pd.read_csv(path, sep="\t")
    if differential_edge_rule not in DIFFERENTIAL_EDGE_RULE_LABELS:
        raise ValueError(f"Unknown differential edge rule: {differential_edge_rule}")
    path = module_set_data_dir(module_set) / "differential/mdc_filtered_summary.parquet"
    return _read_filtered(
        path,
        [
            ("estimator", "=", estimator),
            ("network_method", "=", method),
            ("differential_edge_rule", "=", differential_edge_rule),
            ("differential_fdr_scope", "=", differential_fdr_scope),
            ("differential_fdr_threshold", "=", float(differential_fdr_threshold)),
        ],
    )


def load_mdc_resolved(
    module_set: str = "full_cohort",
    estimator: str = "lioness",
    method: str = "control_anchored",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Load all-edge MDC or the selected discovery-filtered resolved MDC rows."""

    if differential_edge_rule == "all":
        path = ensure_data_path(
            module_set_path("mdc_resolved_ad_vs_control.tsv", module_set)
        )
        return pd.read_csv(path, sep="\t")
    if differential_edge_rule not in DIFFERENTIAL_EDGE_RULE_LABELS:
        raise ValueError(f"Unknown differential edge rule: {differential_edge_rule}")
    path = module_set_data_dir(module_set) / "differential/mdc_filtered_resolved.parquet"
    return _read_filtered(
        path,
        [
            ("estimator", "=", estimator),
            ("network_method", "=", method),
            ("differential_edge_rule", "=", differential_edge_rule),
            ("differential_fdr_scope", "=", differential_fdr_scope),
            ("differential_fdr_threshold", "=", float(differential_fdr_threshold)),
        ],
    )


def load_edge_summaries(
    estimator: str,
    method: str,
    module: int,
    module_set: str = "full_cohort",
    edge_rule: str = "all",
    differential_edge_rule: str = "all",
    differential_fdr_scope: str = "global",
    differential_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    if differential_edge_rule != "all":
        path = (
            module_set_data_dir(module_set)
            / "differential"
            / "edge_summaries"
            / f"{estimator}__{method}__{edge_rule}.parquet"
        )
    elif estimator == "lioness":
        path = module_set_data_dir(module_set) / "edge_summaries" / f"lioness__{method}.parquet"
    elif estimator == "bonobo":
        path = module_set_data_dir(module_set) / "edge_summaries" / f"bonobo__{edge_rule}.parquet"
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    filters: list[tuple[str, str, object]] = [("module", "=", int(module))]
    if differential_edge_rule != "all":
        filters.append(("differential_fdr_scope", "=", differential_fdr_scope))
        filters.append(
            ("differential_fdr_threshold", "=", float(differential_fdr_threshold))
        )
    frame = _read_filtered(path, filters)
    if "scope" in frame:
        frame["scope"] = (
            frame["scope"].astype("string").str.replace("MFBA9BA46", "DLPFC", regex=False)
        )
        frame["scope_label"] = frame["scope"].map(EDGE_SCOPE_LABELS)
    return frame


def load_volcano_candidates(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
) -> pd.DataFrame:
    path = module_set_data_dir(module_set) / "differential" / "volcano_candidates.parquet"
    frame = _read_filtered(
        path,
        [
            ("estimator", "=", estimator),
            ("network_method", "=", method),
            ("module", "=", int(module)),
        ],
    )
    return public_gene_labels(frame, gene_columns=("gene_a", "gene_b"))


def load_volcano_bins(
    module_set: str,
    estimator: str,
    method: str,
    module: int,
    differential_fdr_scope: str = "global",
) -> pd.DataFrame:
    path = module_set_data_dir(module_set) / "differential" / "volcano_bins.parquet"
    return _read_filtered(
        path,
        [
            ("estimator", "=", estimator),
            ("network_method", "=", method),
            ("module", "=", int(module)),
            ("fdr_scope", "=", differential_fdr_scope),
        ],
    )


def load_data_manifest() -> dict[str, object]:
    path = ensure_data_path(DATA_MANIFEST)
    return json.loads(path.read_text(encoding="utf-8"))


def load_kegg(
    module: int | None = None, module_set: str = "full_cohort"
) -> pd.DataFrame:
    filters = [("cluster_id", "=", int(module))] if module is not None else []
    frame = _read_filtered(
        module_set_path("kegg_tissue_expanded_full.parquet", module_set), filters
    )
    return public_gene_labels(frame, text_columns=("overlap_genes",))


def load_kegg_tsv_bytes(module_set: str = "full_cohort") -> bytes:
    """Return the complete KEGG table with official symbols in every gene list."""
    return dataframe_to_tsv_bytes(load_kegg(module_set=module_set))


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


def build_pathway_mdc_rows(
    mdc_summary: pd.DataFrame,
    mdc_resolved: pd.DataFrame,
    kegg: pd.DataFrame,
    *,
    enrichment_fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Join module MDC to KEGG pathways using component-matched region FDRs.

    This is an annotation-level join: MDC remains a module statistic. For a CT
    tissue pair, both regions must support the pathway and the displayed pair FDR
    is the larger of the two region-specific KEGG FDRs.
    """

    enrichment_fdr_threshold = float(enrichment_fdr_threshold)
    if not 0.0 <= enrichment_fdr_threshold <= 1.0:
        raise ValueError("enrichment_fdr_threshold must be between 0 and 1")

    required_summary = {
        "module",
        "mdc_total",
        "log2_mdc_total",
        "direction_total",
        "directional_fdr_total",
        "n_ts_edges",
        "n_ct_edges",
        "mdc_ts",
        "log2_mdc_ts",
        "direction_ts",
        "directional_fdr_ts",
        "mdc_ct",
        "log2_mdc_ct",
        "direction_ct",
        "directional_fdr_ct",
    }
    required_resolved = {
        "module",
        "component",
        "component_label",
        "mdc",
        "log2_mdc",
        "direction",
        "directional_fdr",
        "n_edges",
    }
    required_kegg = {
        "cluster_id",
        "pathway_id",
        "pathway_name",
        "category_level1",
        "category_level2",
        "p",
        "fdr",
        *KEGG_REGION_P_COLUMNS.values(),
        *KEGG_REGION_FDR_COLUMNS.values(),
    }
    for label, frame, required in (
        ("MDC summary", mdc_summary, required_summary),
        ("resolved MDC", mdc_resolved, required_resolved),
        ("KEGG", kegg, required_kegg),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    pooled_parts = []
    pooled_config = [
        ("total", "Total", "total", "n_ts_edges", "n_ct_edges"),
        ("TS", "TS pooled", "ts", "n_ts_edges", None),
        ("CT", "CT pooled", "ct", "n_ct_edges", None),
    ]
    for component, component_label, suffix, edge_column, extra_edge_column in pooled_config:
        columns = [
            "module",
            f"mdc_{suffix}",
            f"log2_mdc_{suffix}",
            f"direction_{suffix}",
            f"directional_fdr_{suffix}",
            edge_column,
        ]
        if extra_edge_column:
            columns.append(extra_edge_column)
        part = mdc_summary[columns].copy()
        part = part.rename(
            columns={
                f"mdc_{suffix}": "mdc",
                f"log2_mdc_{suffix}": "log2_mdc",
                f"direction_{suffix}": "direction",
                f"directional_fdr_{suffix}": "directional_fdr",
                edge_column: "n_edges",
            }
        )
        if component == "total":
            part["n_edges"] = (
                pd.to_numeric(part["n_edges"], errors="coerce").fillna(0)
                + pd.to_numeric(part[extra_edge_column], errors="coerce").fillna(0)
            )
            part = part.drop(columns=extra_edge_column)
        part["component"] = component
        part["component_label"] = component_label
        part["component_class"] = component if component in {"TS", "CT"} else "total"
        pooled_parts.append(part)

    resolved = mdc_resolved[
        [
            "module",
            "component",
            "component_label",
            "mdc",
            "log2_mdc",
            "direction",
            "directional_fdr",
            "n_edges",
        ]
    ].copy()
    resolved["component_class"] = resolved["component"].astype(str).str[:2]
    mdc_long = pd.concat([*pooled_parts, resolved], ignore_index=True)
    mdc_long["module"] = pd.to_numeric(mdc_long["module"], errors="raise").astype(int)

    pathways = kegg.copy().rename(columns={"cluster_id": "module"})
    pathways["module"] = pd.to_numeric(pathways["module"], errors="raise").astype(int)
    rows = mdc_long.merge(pathways, on="module", how="inner", validate="many_to_many")
    rows["enrichment_scope"] = "All regions (expanded)"
    rows["enrichment_p"] = pd.to_numeric(rows["p"], errors="coerce")
    rows["enrichment_fdr"] = pd.to_numeric(rows["fdr"], errors="coerce")
    rows["enrichment_region_a"] = pd.NA
    rows["enrichment_region_b"] = pd.NA
    rows["enrichment_fdr_region_a"] = np.nan
    rows["enrichment_fdr_region_b"] = np.nan

    for component, regions in KEGG_COMPONENT_REGIONS.items():
        component_mask = rows["component"].astype(str).eq(component)
        if not component_mask.any():
            continue
        region_a = regions[0]
        region_b = regions[1] if len(regions) == 2 else None
        rows.loc[component_mask, "enrichment_region_a"] = region_a
        rows.loc[component_mask, "enrichment_fdr_region_a"] = pd.to_numeric(
            rows.loc[component_mask, KEGG_REGION_FDR_COLUMNS[region_a]],
            errors="coerce",
        )
        if region_b is None:
            rows.loc[component_mask, "enrichment_scope"] = region_a
            rows.loc[component_mask, "enrichment_p"] = pd.to_numeric(
                rows.loc[component_mask, KEGG_REGION_P_COLUMNS[region_a]],
                errors="coerce",
            )
            rows.loc[component_mask, "enrichment_fdr"] = rows.loc[
                component_mask, "enrichment_fdr_region_a"
            ]
        else:
            rows.loc[component_mask, "enrichment_region_b"] = region_b
            rows.loc[component_mask, "enrichment_fdr_region_b"] = pd.to_numeric(
                rows.loc[component_mask, KEGG_REGION_FDR_COLUMNS[region_b]],
                errors="coerce",
            )
            rows.loc[component_mask, "enrichment_scope"] = (
                f"{region_a} + {region_b} (both regions)"
            )
            pair_p_columns = [
                KEGG_REGION_P_COLUMNS[region_a],
                KEGG_REGION_P_COLUMNS[region_b],
            ]
            rows.loc[component_mask, "enrichment_p"] = rows.loc[
                component_mask, pair_p_columns
            ].apply(pd.to_numeric, errors="coerce").max(axis=1)
            rows.loc[component_mask, "enrichment_fdr"] = rows.loc[
                component_mask,
                ["enrichment_fdr_region_a", "enrichment_fdr_region_b"],
            ].max(axis=1)

    rows = rows.loc[
        pd.to_numeric(rows["enrichment_fdr"], errors="coerce").le(
            enrichment_fdr_threshold
        )
    ].copy()
    rows["pathway_label"] = (
        rows["pathway_name"]
        .astype("string")
        .str.replace(" - Homo sapiens (human)", "", regex=False)
    )
    rows["enrichment_fdr_threshold"] = enrichment_fdr_threshold
    sort_columns = ["enrichment_fdr", "pathway_label", "module", "component"]
    return rows.sort_values(sort_columns, kind="stable").reset_index(drop=True)


MDC_ENRICHMENT_RESOLUTION_LABELS = {
    "pathway": "Pathway",
    "subcategory": "KEGG sub-category",
    "category": "KEGG category",
}


def collapse_pathway_mdc_rows(
    rows: pd.DataFrame,
    *,
    resolution: str = "pathway",
) -> pd.DataFrame:
    """Collapse pathway annotations to one module-component row per KEGG group.

    At category and sub-category resolution, a module can support the group through
    several enriched pathways. Such a module is counted once; its group-level KEGG
    FDR is the smallest component-matched FDR among its supporting pathways.
    """

    if resolution not in MDC_ENRICHMENT_RESOLUTION_LABELS:
        raise ValueError(
            "resolution must be one of "
            f"{sorted(MDC_ENRICHMENT_RESOLUTION_LABELS)}"
        )
    if rows.empty:
        return rows.copy()

    required = {
        "module",
        "component",
        "pathway_id",
        "pathway_label",
        "category_level1",
        "category_level2",
        "enrichment_fdr",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Pathway MDC rows are missing columns: {sorted(missing)}")

    data = rows.copy()
    data["source_pathway_id"] = data["pathway_id"].astype("string")
    data["source_pathway_label"] = data["pathway_label"].astype("string")
    category = data["category_level1"].fillna("Unclassified").astype(str)
    subcategory = data["category_level2"].fillna("Unclassified").astype(str)
    if resolution == "pathway":
        data["enrichment_group_id"] = data["source_pathway_id"].astype(str)
        data["enrichment_group_label"] = data["source_pathway_label"].astype(str)
    elif resolution == "subcategory":
        data["enrichment_group_id"] = "subcategory::" + category + "::" + subcategory
        data["enrichment_group_label"] = subcategory
    else:
        data["enrichment_group_id"] = "category::" + category
        data["enrichment_group_label"] = category

    data["enrichment_resolution"] = resolution
    data["enrichment_resolution_label"] = MDC_ENRICHMENT_RESOLUTION_LABELS[
        resolution
    ]
    group_keys = ["enrichment_group_id", "module", "component"]

    def _joined_unique(values: pd.Series) -> str:
        cleaned = sorted(
            {
                str(value).strip()
                for value in values.dropna()
                if str(value).strip()
            }
        )
        return "; ".join(cleaned)

    support = (
        data.groupby(group_keys, observed=True, dropna=False)
        .agg(
            supporting_pathway_count=("source_pathway_id", "nunique"),
            supporting_pathway_ids=("source_pathway_id", _joined_unique),
            supporting_pathway_names=("source_pathway_label", _joined_unique),
            supporting_subcategories=("category_level2", _joined_unique),
        )
        .reset_index()
    )
    representative = (
        data.sort_values(
            ["enrichment_fdr", "source_pathway_id"],
            na_position="last",
            kind="stable",
        )
        .drop_duplicates(group_keys, keep="first")
        .copy()
    )
    representative = representative.drop(
        columns=[
            "supporting_pathway_count",
            "supporting_pathway_ids",
            "supporting_pathway_names",
            "supporting_subcategories",
        ],
        errors="ignore",
    ).merge(support, on=group_keys, how="left", validate="one_to_one")
    representative["best_supporting_pathway_id"] = representative[
        "source_pathway_id"
    ]
    representative["best_supporting_pathway_name"] = representative[
        "source_pathway_label"
    ]
    representative["pathway_id"] = representative["enrichment_group_id"]
    representative["pathway_label"] = representative["enrichment_group_label"]
    representative["pathway_name"] = representative["enrichment_group_label"]
    if resolution == "category":
        representative["category_level2"] = "All sub-categories"
    return representative.sort_values(
        ["enrichment_fdr", "pathway_label", "module", "component"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def summarize_pathway_mdc_rows(
    rows: pd.DataFrame,
    *,
    mdc_fdr_threshold: float = 0.05,
    minimum_modules: int = 1,
    resolution: str = "pathway",
) -> pd.DataFrame:
    """Summarize KEGG-annotated module MDC within each edge component."""

    if rows.empty:
        return pd.DataFrame()
    minimum_modules = int(minimum_modules)
    if minimum_modules < 1:
        raise ValueError("minimum_modules must be at least 1")
    if "enrichment_resolution" in rows.columns:
        existing_resolutions = set(
            rows["enrichment_resolution"].dropna().astype(str).unique()
        )
        if existing_resolutions != {resolution}:
            raise ValueError(
                "Collapsed pathway MDC rows do not match the requested resolution"
            )
        data = rows.copy()
    else:
        data = collapse_pathway_mdc_rows(rows, resolution=resolution)
    data["mdc_significant"] = pd.to_numeric(
        data["directional_fdr"], errors="coerce"
    ).lt(float(mdc_fdr_threshold))
    group_columns = [
        "pathway_id",
        "pathway_label",
        "pathway_name",
        "category_level1",
        "category_level2",
        "component",
        "component_label",
        "component_class",
        "enrichment_scope",
        "enrichment_resolution",
        "enrichment_resolution_label",
    ]
    summary = (
        data.groupby(group_columns, observed=True, dropna=False)
        .agg(
            n_modules=("module", "nunique"),
            mean_log2_mdc=("log2_mdc", "mean"),
            median_log2_mdc=("log2_mdc", "median"),
            median_mdc=("mdc", "median"),
            minimum_enrichment_fdr=("enrichment_fdr", "min"),
            median_enrichment_fdr=("enrichment_fdr", "median"),
            n_mdc_significant=("mdc_significant", "sum"),
            minimum_mdc_fdr=("directional_fdr", "min"),
            total_edges=("n_edges", "sum"),
            n_pathways=("supporting_pathway_count", "sum"),
        )
        .reset_index()
    )
    pathway_counts = (
        data.groupby(group_columns, observed=True, dropna=False)[
            "supporting_pathway_ids"
        ]
        .apply(
            lambda values: len(
                {
                    pathway_id
                    for value in values.dropna().astype(str)
                    for pathway_id in value.split("; ")
                    if pathway_id
                }
            )
        )
        .rename("n_distinct_pathways")
        .reset_index()
    )
    summary = summary.drop(columns="n_pathways").merge(
        pathway_counts,
        on=group_columns,
        how="left",
        validate="one_to_one",
    ).rename(columns={"n_distinct_pathways": "n_pathways"})
    summary["geometric_mean_mdc"] = np.exp2(summary["mean_log2_mdc"])
    summary["proportion_mdc_significant"] = (
        summary["n_mdc_significant"] / summary["n_modules"]
    )
    summary = summary.loc[summary["n_modules"].ge(minimum_modules)].copy()
    return summary.sort_values(
        ["minimum_enrichment_fdr", "mean_log2_mdc"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


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


def prediction_data_available() -> bool:
    """Return whether the compact prediction catalog is local or Drive-indexed."""

    return data_path_available(PREDICTION_MANIFEST) and data_path_available(
        PREDICTION_PERFORMANCE
    )


def targeted_prediction_data_available() -> bool:
    """Return whether the repeated nested-CV targeted catalog is available."""

    return data_path_available(TARGETED_PREDICTION_MANIFEST) and data_path_available(
        TARGETED_PREDICTION_FILES["oof_performance"]
    )


def load_targeted_prediction_manifest() -> dict[str, object]:
    return json.loads(
        ensure_data_path(TARGETED_PREDICTION_MANIFEST).read_text(encoding="utf-8")
    )


def load_targeted_prediction_table(
    table: str,
    **filters: object | None,
) -> pd.DataFrame:
    """Load one targeted-prediction table with Parquet predicate pushdown."""

    if table not in TARGETED_PREDICTION_FILES:
        raise KeyError(f"Unknown targeted-prediction table: {table}")
    manifest = load_targeted_prediction_manifest()
    legacy_transform_schema = int(manifest.get("schema_version", 2)) < 3
    deferred = {
        column: value
        for column, value in filters.items()
        if legacy_transform_schema and column in {"score_transform", "transformation_role"}
        and value is not None
    }
    predicates = [
        (column, "=", value)
        for column, value in filters.items()
        if value is not None and column not in deferred
    ]
    result = _read_filtered(TARGETED_PREDICTION_FILES[table], predicates)
    if "score_transform" not in result:
        result["score_transform"] = "asinh"
    if "transformation_role" not in result:
        result["transformation_role"] = (
            "existing_masked_sensitivity"
            if table.startswith("masked_")
            else "legacy_asinh_provenance"
        )
    for column, value in deferred.items():
        result = result.loc[result[column].eq(value)]
    return result.reset_index(drop=True)


def load_prediction_manifest() -> dict[str, object]:
    return json.loads(ensure_data_path(PREDICTION_MANIFEST).read_text(encoding="utf-8"))


def load_prediction_performance(
    reference_provenance: str | None = None,
    module_definition: str | None = None,
    network_method: str | None = None,
    predictor_design: str | None = None,
    edge_mask: str | None = None,
    score_normalization: str | None = None,
    outcome: str | None = None,
) -> pd.DataFrame:
    filters = [
        (column, "=", value)
        for column, value in (
            ("reference_provenance", reference_provenance),
            ("module_definition", module_definition),
            ("network_method", network_method),
            ("predictor_design", predictor_design),
            ("edge_mask", edge_mask),
            ("score_normalization", score_normalization),
            ("outcome", outcome),
        )
        if value is not None
    ]
    return _read_filtered(PREDICTION_PERFORMANCE, filters or None)


def load_prediction_bootstrap(
    reference_provenance: str | None = None,
    module_definition: str | None = None,
    network_method: str | None = None,
    predictor_design: str | None = None,
    edge_mask: str | None = None,
    score_normalization: str | None = None,
    outcome: str | None = None,
) -> pd.DataFrame:
    filters = [
        (column, "=", value)
        for column, value in (
            ("reference_provenance", reference_provenance),
            ("module_definition", module_definition),
            ("network_method", network_method),
            ("predictor_design", predictor_design),
            ("edge_mask", edge_mask),
            ("score_normalization", score_normalization),
            ("outcome", outcome),
        )
        if value is not None
    ]
    return _read_filtered(PREDICTION_BOOTSTRAP, filters or None)


def load_prediction_curves(**filters: str | None) -> pd.DataFrame:
    predicates = [(column, "=", value) for column, value in filters.items() if value is not None]
    return _read_filtered(PREDICTION_CURVES, predicates or None)


def load_prediction_confusion(**filters: str | None) -> pd.DataFrame:
    predicates = [(column, "=", value) for column, value in filters.items() if value is not None]
    return _read_filtered(PREDICTION_CONFUSION, predicates or None)


def load_prediction_coefficients(**filters: str | None) -> pd.DataFrame:
    predicates = [(column, "=", value) for column, value in filters.items() if value is not None]
    return _read_filtered(PREDICTION_COEFFICIENTS, predicates or None)


def load_prediction_diagnostics(**filters: str | None) -> pd.DataFrame:
    predicates = [(column, "=", value) for column, value in filters.items() if value is not None]
    return _read_filtered(PREDICTION_PREDICTIONS, predicates or None)


def load_prediction_whole_network_features(**filters: str | None) -> pd.DataFrame:
    predicates = [(column, "=", value) for column, value in filters.items() if value is not None]
    return _read_filtered(PREDICTION_WHOLE_NETWORK, predicates or None)
