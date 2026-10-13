"""Fixed-module diagnosis-label MDC tests and descriptive balanced sensitivity.

These tests do not account for using the same Controls to discover the modules.
Legacy coexpression/gene-null results are deliberately not overwritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

VARIANTS = {"pearson": "Unpowered Pearson (comparable CT/TS weights)",
            "m5_high_no_q90": "M5 without q90 (weighting sensitivity)"}
SCOPES = ("total", "TS", "CT", "TS_VH", "TS_SMA", "TS_QUAD",
          "CT_VH__SMA", "CT_VH__QUAD", "CT_SMA__QUAD", "CT_minus_TS")


def permute_within_strata(labels, strata, seed):
    """Preserve each study×sex stratum's exact class counts and donor profiles."""
    labels, strata = np.asarray(labels), np.asarray(strata)
    if labels.shape != strata.shape or labels.ndim != 1:
        raise ValueError("Labels and strata must be aligned vectors")
    rng = np.random.default_rng(seed)
    out = labels.copy()
    for key in sorted(set(strata)):
        selected = np.flatnonzero(strata == key)
        out[selected] = rng.permutation(labels[selected])
    return out


def proportional_subset(indices, strata, size, seed):
    """Sample without replacement, preserving AD stratum proportions by quotas."""
    indices, strata = np.asarray(indices), np.asarray(strata)
    if not 0 < size <= len(indices):
        raise ValueError("Invalid balanced sample size")
    keys = sorted(set(strata[indices]))
    groups = [indices[strata[indices] == key] for key in keys]
    ideal = np.asarray([len(g) for g in groups]) * size / len(indices)
    quotas = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal - quotas), kind="stable")
    quotas[order[:size - quotas.sum()]] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(g, int(n), replace=False)
                                  for g, n in zip(groups, quotas) if n]))


def add_contrasts(observed):
    """A within-module paired contrast; never substitute zero for missing scopes."""
    rows = []
    for (variant, module), group in observed.groupby(["normalization_variant", "module"]):
        ct, ts = group.set_index("scope").loc[["CT", "TS"]].to_dict("index").values()
        eligible = bool(ct["test_eligible"] and ts["test_eligible"])
        value = ct["statistic"] - ts["statistic"] if eligible else np.nan
        rows.append({"normalization_variant": variant, "module": module,
                     "scope": "CT_minus_TS", "statistic": value,
                     "mdc_raw": np.nan, "mean_abs_ad": np.nan,
                     "mean_abs_control": np.nan, "n_edges": ct["n_edges"] + ts["n_edges"],
                     "test_eligible": eligible,
                     "unavailable_reason": "" if eligible else "CT or TS unavailable/unstable"})
    return pd.concat([observed, pd.DataFrame(rows)], ignore_index=True)


def summarize_validation(observed, label_null, balanced, expected_label_n, expected_balance_n):
    """Two-sided equal-tail permutation p; BH within variant×scope across modules.

    A complete null family is required. Balanced 2.5/97.5 percentiles describe
    AD-subsampling variability, not confidence intervals or significance tests.
    """
    result = observed.copy()
    result["label_test_complete"] = len(label_null) == expected_label_n
    result["balanced_complete"] = len(balanced) == expected_balance_n
    result["label_p"] = np.nan
    result["label_fdr"] = np.nan
    result["tested_module_count"] = 0
    result["null_finite_n"] = 0
    keys = list(zip(result.normalization_variant, result.module, result.scope))
    for i, key in zip(result.index, keys):
        v, m, s = key
        null_values = np.asarray([rep[key] for rep in label_null], float)
        null_values = null_values[np.isfinite(null_values)]
        result.loc[i, "null_finite_n"] = len(null_values)
        for prefix, reps in [("null", label_null), ("balanced", balanced)]:
            values = np.asarray([rep[key] for rep in reps], float)
            values = values[np.isfinite(values)]
            result.loc[i, prefix + "_finite_n"] = len(values)
            for name, q in [("low", .025), ("median", .5), ("high", .975)]:
                result.loc[i, prefix + "_" + name] = np.quantile(values, q) if len(values) else np.nan
        if not result.loc[i, "test_eligible"] or len(label_null) != expected_label_n:
            continue
        if len(null_values) < np.ceil(.9 * expected_label_n):
            result.loc[i, "test_eligible"] = False
            result.loc[i, "unavailable_reason"] = "Fewer than 90% finite label permutations"
            continue
        if np.ptp(null_values) == 0:
            result.loc[i, "test_eligible"] = False
            result.loc[i, "unavailable_reason"] = "Constant label-permutation statistic"
            continue
        value = result.loc[i, "statistic"]
        lower = (1 + np.count_nonzero(null_values <= value)) / (1 + len(null_values))
        upper = (1 + np.count_nonzero(null_values >= value)) / (1 + len(null_values))
        result.loc[i, "label_p"] = min(1., 2 * min(lower, upper))
    for _, family in result.groupby(["normalization_variant", "scope"]):
        valid = family.label_p.notna()
        ids = family.index[valid]
        result.loc[family.index, "tested_module_count"] = len(ids)
        if len(ids):
            result.loc[ids, "label_fdr"] = false_discovery_control(result.loc[ids, "label_p"], method="bh")
    result["significant_fdr05"] = result.label_fdr.lt(.05)
    result["significant_fdr10"] = result.label_fdr.lt(.1)
    result["validation_level"] = "Fixed Control-derived modules; discovery bias not corrected"
    return result
