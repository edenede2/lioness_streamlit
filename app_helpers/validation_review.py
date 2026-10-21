"""Compact, read-only evidence for technical validation and cohort replication.

This page never reads expression or donor scores and never fits a statistical
test. Corrections are copied from the full module families, before UI filtering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app_helpers import camel_validation as V
from app_helpers.drive_data import data_path_available, ensure_data_path

MATCH_KEYS = ["module", "metric_family", "component", "phenotype", "diagnosis_group"]


def review_root() -> Path:
    return V.root() / "review"


def available(path: Path) -> bool:
    if path.is_file():
        return True
    try:
        return data_path_available(path)
    except ValueError:
        return False


def manifest() -> dict:
    path = review_root() / "manifest.json"
    return json.loads(ensure_data_path(path).read_text()) if available(path) else {}


def table(name: str) -> pd.DataFrame:
    if name not in {"technical_checks", "formula_checks", "score_reproduction", "cohorts", "completion", "mdc_comparison"}:
        raise ValueError("Unknown validation evidence table")
    path = review_root() / f"{name}.parquet"
    return pd.read_parquet(ensure_data_path(path)) if available(path) else pd.DataFrame()


def association_table(rule: str, normalization: str, feature: str, outcome: str) -> pd.DataFrame:
    entry = next((e for e in manifest().get("association_shards", [])
                  if (e["rule"], e["normalization"], e["feature"], e["outcome"]) ==
                  (rule, normalization, feature, outcome)), None)
    if entry is None:
        return pd.DataFrame()
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid validation-review shard path")
    return pd.read_parquet(ensure_data_path(review_root() / relative))


def association_effects(frame: pd.DataFrame, method="Spearman", scale="raw") -> pd.DataFrame:
    """Choose scale-specific statistics; in particular, RINT may introduce ties."""
    if method not in {"Spearman", "Pearson"} or scale not in {"raw", "asinh", "rint"}:
        raise ValueError("Unsupported association method or scale")
    f = frame.copy()
    if f.empty:
        return f
    if method == "Spearman":
        mapping = {
            "original_effect": "original_rho", "original_p": "original_p_spearman",
            "original_fdr": "original_q_spearman",
            "validation_effect": f"validation_rho_{scale}",
            "validation_p": f"validation_p_spearman_{scale}",
            "validation_fdr": f"validation_q_spearman_{scale}_across_modules",
        }
    else:
        mapping = {
            "original_effect": f"original_r_{scale}", "original_p": f"original_p_{scale}",
            "original_fdr": f"original_q_{scale}", "validation_effect": f"validation_r_{scale}",
            "validation_p": f"validation_p_{scale}",
            "validation_fdr": f"validation_q_{scale}_across_modules",
        }
    for dest, source in mapping.items():
        f[dest] = pd.to_numeric(f[source], errors="coerce")
    f["original_family_n"] = f[f"original_family_n_{'spearman' if method == 'Spearman' else scale}"]
    f["validation_family_n"] = f[f"validation_family_n_{'spearman_' + scale if method == 'Spearman' else scale}"]
    f["comparison_available"] = np.isfinite(f.original_effect) & np.isfinite(f.validation_effect)
    f["same_direction"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    finite = f.comparison_available
    f.loc[finite, "same_direction"] = (np.sign(f.loc[finite, "original_effect"]) ==
                                       np.sign(f.loc[finite, "validation_effect"]))
    f["effect_difference"] = f.validation_effect - f.original_effect
    if "unavailable_reason" not in f:
        f["unavailable_reason"] = ""
    missing = ~f.comparison_available
    for index in f.index[missing]:
        reasons = []
        if not np.isfinite(f.at[index, "original_effect"]):
            reasons.append("Original correlation unavailable in the exact saved stratum/scale")
        if not np.isfinite(f.at[index, "validation_effect"]):
            detail = f.at[index, "validation_unavailable_reason"] if "validation_unavailable_reason" in f else ""
            reasons.append("Validation: " + (str(detail) if pd.notna(detail) and str(detail) else
                           "insufficient pairs, constant score/outcome, or unavailable frozen transform"))
        f.at[index, "unavailable_reason"] = "; ".join(reasons)
    return f


def replication_labels(frame: pd.DataFrame, alpha=.05) -> pd.DataFrame:
    if alpha not in {.05, .10}:
        raise ValueError("Unsupported replication cutoff")
    f = frame.copy()
    f["original_signal"] = f.original_fdr.lt(alpha)
    f["validation_nominal_signal"] = f.validation_p.lt(alpha)
    f["validation_fdr_signal"] = f.validation_fdr.lt(alpha)
    same = f.same_direction.fillna(False).astype(bool)
    f["replication_status"] = "Unavailable comparison"
    f.loc[f.comparison_available, "replication_status"] = "No original FDR signal"
    original = f.comparison_available & f.original_signal
    f.loc[original, "replication_status"] = "Original signal; opposite/zero direction"
    f.loc[original & same, "replication_status"] = "Same direction only"
    f.loc[original & same & f.validation_nominal_signal, "replication_status"] = "Same direction + validation nominal p"
    f.loc[original & same & f.validation_fdr_signal, "replication_status"] = "Same direction + validation FDR"
    return f


def replication_counts(frame: pd.DataFrame) -> dict:
    valid = frame.comparison_available
    original = valid & frame.original_signal
    same = frame.same_direction.fillna(False).astype(bool)
    return {
        "comparable_modules": int(valid.sum()),
        "original_signals": int(frame.original_signal.sum()),
        "original_signals_evaluable": int(original.sum()),
        "direction_agreement_all": int((valid & same).sum()),
        "same_direction": int((original & same).sum()),
        "nominal_replication": int((original & same & frame.validation_nominal_signal).sum()),
        "fdr_replication": int((original & same & frame.validation_fdr_signal).sum()),
    }
