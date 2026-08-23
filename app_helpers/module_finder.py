"""Rank modules by CT–TS and diagnosis-specific association differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from app_helpers.correlations import benjamini_hochberg


FINDER_CRITERIA = {
    "ct_ts": "CT vs TS",
    "ad_control": "Control vs AD",
    "both": "Both criteria",
}


def _correlation_columns(method: str) -> dict[str, str]:
    if method == "Spearman":
        return {
            "CT": "rho_CT",
            "TS": "rho_TS",
            "component_test": "component_t_rank",
            "component_p": "p_component_rank",
            "component_fdr": "q_component_rank_within_phenotype",
            "component_fdr_global": "q_component_rank_all12_global",
        }
    if method == "Pearson":
        return {
            "CT": "r_rint_CT",
            "TS": "r_rint_TS",
            "component_test": "component_t_rint",
            "component_p": "p_component_rint",
            "component_fdr": "q_component_rint_within_phenotype",
            "component_fdr_global": "q_component_rint_all12_global",
        }
    raise ValueError(f"Unknown correlation method: {method}")


def _fisher_difference(
    correlation_a: pd.Series,
    n_a: pd.Series,
    correlation_b: pd.Series,
    n_b: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Approximate an independent-groups correlation difference with Fisher z."""

    r_a = pd.to_numeric(correlation_a, errors="coerce").to_numpy(dtype=float)
    r_b = pd.to_numeric(correlation_b, errors="coerce").to_numpy(dtype=float)
    count_a = pd.to_numeric(n_a, errors="coerce").to_numpy(dtype=float)
    count_b = pd.to_numeric(n_b, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(r_a)
        & np.isfinite(r_b)
        & np.isfinite(count_a)
        & np.isfinite(count_b)
        & (count_a > 3)
        & (count_b > 3)
    )
    statistic = np.full(len(r_a), np.nan, dtype=float)
    p_value = np.full(len(r_a), np.nan, dtype=float)
    if valid.any():
        epsilon = np.finfo(float).eps
        clipped_a = np.clip(r_a[valid], -1 + epsilon, 1 - epsilon)
        clipped_b = np.clip(r_b[valid], -1 + epsilon, 1 - epsilon)
        standard_error = np.sqrt(
            1.0 / (count_a[valid] - 3.0) + 1.0 / (count_b[valid] - 3.0)
        )
        statistic[valid] = (np.arctanh(clipped_a) - np.arctanh(clipped_b)) / standard_error
        p_value[valid] = 2.0 * norm.sf(np.abs(statistic[valid]))
    return (
        pd.Series(statistic, index=correlation_a.index, dtype=float),
        pd.Series(p_value, index=correlation_a.index, dtype=float),
    )


def build_module_finder_table(
    statistics: pd.DataFrame,
    *,
    ct_ts_diagnosis: str,
    correlation_method: str,
    criterion: str,
) -> pd.DataFrame:
    """Build one module-level table for the two association-difference criteria.

    The CT–TS criterion uses the existing dependent-component test within the chosen
    diagnosis. The Control–AD criterion compares independent diagnosis-specific
    correlations with the Fisher-z approximation separately for CT and TS, followed
    by BH across both components and every displayed module.
    """

    if criterion not in FINDER_CRITERIA:
        raise ValueError(f"Unknown module-finder criterion: {criterion}")
    columns = _correlation_columns(correlation_method)
    required = {
        "module",
        "diagnosis_group",
        "n",
        *columns.values(),
    }
    missing = required.difference(statistics.columns)
    if missing:
        raise ValueError(f"Module-finder statistics are missing: {sorted(missing)}")

    selected = statistics.loc[
        statistics["diagnosis_group"].eq(ct_ts_diagnosis)
    ].copy()
    if selected.duplicated("module").any():
        raise ValueError("CT–TS module-finder rows are not unique by module")
    selected = selected[
        [
            "module",
            "n",
            columns["CT"],
            columns["TS"],
            columns["component_test"],
            columns["component_p"],
            columns["component_fdr"],
            columns["component_fdr_global"],
        ]
    ].rename(
        columns={
            "n": "ct_ts_n",
            columns["CT"]: "ct_ts_correlation_CT",
            columns["TS"]: "ct_ts_correlation_TS",
            columns["component_test"]: "ct_ts_test_statistic",
            columns["component_p"]: "ct_ts_p_value",
            columns["component_fdr"]: "ct_ts_fdr_within_phenotype",
            columns["component_fdr_global"]: "ct_ts_fdr_all12_global",
        }
    )
    selected["ct_ts_delta_correlation"] = (
        selected["ct_ts_correlation_CT"] - selected["ct_ts_correlation_TS"]
    )
    selected["ct_ts_abs_delta_correlation"] = selected[
        "ct_ts_delta_correlation"
    ].abs()
    selected["ct_ts_diagnosis"] = ct_ts_diagnosis

    diagnosis_rows: dict[str, pd.DataFrame] = {}
    for diagnosis in ("Control", "AD"):
        frame = statistics.loc[statistics["diagnosis_group"].eq(diagnosis)].copy()
        if frame.duplicated("module").any():
            raise ValueError(f"{diagnosis} module-finder rows are not unique by module")
        diagnosis_rows[diagnosis] = frame[
            ["module", "n", columns["CT"], columns["TS"]]
        ].rename(
            columns={
                "n": f"n_{diagnosis.lower()}",
                columns["CT"]: f"correlation_CT_{diagnosis.lower()}",
                columns["TS"]: f"correlation_TS_{diagnosis.lower()}",
            }
        )

    comparison = diagnosis_rows["Control"].merge(
        diagnosis_rows["AD"], on="module", how="outer", validate="one_to_one"
    )
    for component in ("CT", "TS"):
        control_column = f"correlation_{component}_control"
        ad_column = f"correlation_{component}_ad"
        comparison[f"ad_control_delta_correlation_{component}"] = (
            comparison[ad_column] - comparison[control_column]
        )
        statistic, p_value = _fisher_difference(
            comparison[ad_column],
            comparison["n_ad"],
            comparison[control_column],
            comparison["n_control"],
        )
        comparison[f"ad_control_fisher_z_{component}"] = statistic
        comparison[f"ad_control_p_value_{component}"] = p_value

    family_p_values = pd.concat(
        [
            comparison["ad_control_p_value_CT"].reset_index(drop=True),
            comparison["ad_control_p_value_TS"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    family_fdr = benjamini_hochberg(family_p_values)
    row_count = len(comparison)
    comparison["ad_control_fdr_CT"] = family_fdr.iloc[:row_count].to_numpy()
    comparison["ad_control_fdr_TS"] = family_fdr.iloc[row_count:].to_numpy()
    absolute_differences = comparison[
        ["ad_control_delta_correlation_CT", "ad_control_delta_correlation_TS"]
    ].abs()
    comparison["ad_control_max_abs_delta_correlation"] = absolute_differences.max(
        axis=1, skipna=True
    )
    all_components_missing = absolute_differences.isna().all(axis=1)
    comparison["ad_control_best_component"] = (
        absolute_differences.fillna(-np.inf)
        .idxmax(axis=1)
        .str.rsplit("_", n=1)
        .str[-1]
    )
    comparison.loc[all_components_missing, "ad_control_best_component"] = pd.NA
    comparison["ad_control_min_fdr"] = comparison[
        ["ad_control_fdr_CT", "ad_control_fdr_TS"]
    ].min(axis=1, skipna=True)

    result = selected.merge(comparison, on="module", how="outer", validate="one_to_one")
    result["ct_ts_percentile_score"] = result[
        "ct_ts_abs_delta_correlation"
    ].rank(method="average", pct=True)
    result["ad_control_percentile_score"] = result[
        "ad_control_max_abs_delta_correlation"
    ].rank(method="average", pct=True)
    if criterion == "ct_ts":
        result["finder_score"] = result["ct_ts_percentile_score"]
    elif criterion == "ad_control":
        result["finder_score"] = result["ad_control_percentile_score"]
    else:
        result["finder_score"] = result[
            ["ct_ts_percentile_score", "ad_control_percentile_score"]
        ].min(axis=1, skipna=False)
    result["finder_criterion"] = FINDER_CRITERIA[criterion]
    result["correlation_method"] = correlation_method
    result = result.sort_values(
        [
            "finder_score",
            "ct_ts_abs_delta_correlation",
            "ad_control_max_abs_delta_correlation",
        ],
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "finder_rank", np.arange(1, len(result) + 1))
    return result
