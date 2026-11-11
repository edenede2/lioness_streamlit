"""Module-only association families; archived scan corrections stay untouched."""
from __future__ import annotations

import numpy as np
import pandas as pd
from app_helpers.correlations import benjamini_hochberg

FAMILY_COLUMNS = (
    "family", "regional_tissue", "beta", "cohort_scope", "allocation_id",
    "transform", "mask", "score_normalization", "component", "metric_family",
    "outcome", "method", "model", "grouping_variable", "grouping_level",
    "cohort", "study_stratum", "analysis_subset", "subgroup", "edge_rule",
    "differential_edge_rule", "score_scale",
    "differential_fdr_scope", "differential_fdr_threshold", "minimum_group_n",
)


def add_module_fdr(frame):
    """Call on the COMPLETE partition search, before module/display filtering.

    One p-value per module and fixed context. Group levels, components, measures,
    outcomes, methods and beta values NEVER enter one shared BH denominator.
    """
    result = frame.copy()
    if result.empty:
        return result
    numeric_p = pd.to_numeric(result.p, errors="coerce")
    if ((numeric_p < 0) | (numeric_p > 1)).any():
        raise ValueError("Association p-values must lie between zero and one")
    result["p"] = numeric_p.where(np.isfinite(numeric_p))
    family = [c for c in FAMILY_COLUMNS if c in result]
    if "regional_tissue" not in result and "module_key" in result:
        # Regional numeric IDs can repeat across tissues. Correct within tissue.
        result["regional_tissue"] = result.module_key.str.split("::").map(
            lambda parts: parts[1] if len(parts) == 3 else ""
        )
        family.append("regional_tissue")
    identity = "module_key" if "module_key" in result else "module"
    if not family or identity not in result:
        raise ValueError("A complete module family and module identity are required")
    if result.duplicated([*family, identity]).any():
        raise ValueError("Duplicate module tests in a module-only BH family")
    grouped = result.groupby(family, observed=True, dropna=False, sort=False)
    result["q_across_modules"] = grouped.p.transform(benjamini_hochberg)
    result["module_family_n"] = grouped.p.transform(lambda p: p.notna().sum()).astype(int)
    result["module_fdr_scope"] = "BH across valid modules only; fixed " + ", ".join(family)
    return result


def from_baseline(frame):
    """Preserve baseline p/inference and q_scan; add the module-only alternative."""
    return add_module_fdr(frame)
