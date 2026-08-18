"""Bounded readers and display helpers for the packaged public data."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"

AGGREGATE_DATA = DATA_DIR / "aggregate_plot_data.parquet"
RESOLVED_DATA = DATA_DIR / "resolved_plot_data.parquet"
AGGREGATE_STATS = DATA_DIR / "aggregate_statistics.parquet"
RESOLVED_STATS = DATA_DIR / "resolved_statistics.parquet"
KEGG_PARQUET = DATA_DIR / "kegg_tissue_expanded_full.parquet"
KEGG_TSV = DATA_DIR / "kegg_tissue_expanded_full.tsv"
MODULE_ANNOTATIONS = DATA_DIR / "module_kegg_annotations.tsv"
FEATURE_DEFINITIONS = DATA_DIR / "feature_definitions.tsv"
TISSUE_MAPPING = DATA_DIR / "tissue_mapping.tsv"
SAMPLE_METADATA = DATA_DIR / "sample_metadata.parquet"
MDC_SUMMARY = DATA_DIR / "mdc_ad_vs_control_summary.tsv"
DATA_MANIFEST = DATA_DIR / "data_manifest.json"

METHOD_LABELS = {
    "standard": "Standard LIONESS (all-donor reference)",
    "control_anchored": "Control-referenced LIONESS",
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


def require_data_files() -> None:
    """Raise a useful error if the GitHub data bundle was not built."""
    required = [
        AGGREGATE_DATA,
        RESOLVED_DATA,
        AGGREGATE_STATS,
        RESOLVED_STATS,
        KEGG_PARQUET,
        KEGG_TSV,
        MODULE_ANNOTATIONS,
        FEATURE_DEFINITIONS,
        TISSUE_MAPPING,
        SAMPLE_METADATA,
        MDC_SUMMARY,
        DATA_MANIFEST,
    ]
    missing = [str(path.relative_to(APP_ROOT)) for path in required if not path.exists()]
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
        AGGREGATE_DATA,
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
    return _read_filtered(AGGREGATE_DATA, filters, columns)


def load_resolved(
    method: str,
    module: int,
    metric_family: str,
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
        RESOLVED_DATA,
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
    return _read_filtered(RESOLVED_DATA, filters, columns)


def load_aggregate_statistics(
    method: str,
    module: int | None = None,
    phenotype: str | None = None,
    metric_family: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    return _read_filtered(AGGREGATE_STATS, filters)


def load_resolved_statistics(
    method: str,
    module: int | None = None,
    phenotype: str | None = None,
    metric_family: str | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [("lioness_method", "=", method)]
    if module is not None:
        filters.append(("module", "=", int(module)))
    if phenotype is not None:
        filters.append(("phenotype", "=", phenotype))
    if metric_family is not None:
        filters.append(("metric_family", "=", metric_family))
    return _read_filtered(RESOLVED_STATS, filters)


def load_module_annotations() -> pd.DataFrame:
    return pd.read_csv(MODULE_ANNOTATIONS, sep="\t")


def load_feature_definitions() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DEFINITIONS, sep="\t")


def load_tissue_mapping() -> pd.DataFrame:
    return pd.read_csv(TISSUE_MAPPING, sep="\t")


def load_sample_metadata() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE_METADATA)


def load_mdc_summary() -> pd.DataFrame:
    return pd.read_csv(MDC_SUMMARY, sep="\t")


def load_data_manifest() -> dict[str, object]:
    return json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))


def load_kegg(module: int | None = None) -> pd.DataFrame:
    filters = [("cluster_id", "=", int(module))] if module is not None else []
    return _read_filtered(KEGG_PARQUET, filters)


def module_label(module: int | str) -> str:
    return f"M{int(module)}"


def selected_annotation(annotations: pd.DataFrame, module: int) -> str | None:
    match = annotations.loc[annotations["module"].astype(int).eq(int(module))]
    if match.empty or not bool(match.iloc[0].get("annotation_available", False)):
        return None
    text = match.iloc[0].get("subtitle_text")
    return str(text) if pd.notna(text) and str(text).strip() else None


def dataframe_to_tsv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, sep="\t", index=False)
    return buffer.getvalue().encode("utf-8")
