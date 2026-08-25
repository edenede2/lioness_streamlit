"""Rank modules by CT–TS cohort and diagnosis-specific association differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

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
        p_value[valid] = 2.0 * stats.norm.sf(np.abs(statistic[valid]))
    return (
        pd.Series(statistic, index=correlation_a.index, dtype=float),
        pd.Series(p_value, index=correlation_a.index, dtype=float),
    )


def _safe_correlation(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    method: str = "pearson",
) -> tuple[float, float]:
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[finite]
    values_b = values_b[finite]
    if (
        len(values_a) < 4
        or len(np.unique(values_a)) < 2
        or len(np.unique(values_b)) < 2
    ):
        return np.nan, np.nan
    if method == "spearman":
        result = stats.spearmanr(values_a, values_b)
    else:
        result = stats.pearsonr(values_a, values_b)
    return float(result.statistic), float(result.pvalue)


def _williams_test(
    correlation_y_ct: float,
    correlation_y_ts: float,
    correlation_ct_ts: float,
    n: int,
) -> tuple[float, float]:
    """Compare two overlapping correlations that share the phenotype variable."""

    values = np.array(
        [correlation_y_ct, correlation_y_ts, correlation_ct_ts], dtype=float
    )
    if n < 6 or not np.isfinite(values).all():
        return np.nan, np.nan
    determinant = (
        1.0
        - correlation_y_ct**2
        - correlation_y_ts**2
        - correlation_ct_ts**2
        + 2.0 * correlation_y_ct * correlation_y_ts * correlation_ct_ts
    )
    denominator_squared = (
        2.0 * determinant * (n - 1) / (n - 3)
        + ((correlation_y_ct + correlation_y_ts) ** 2 / 4.0)
        * (1.0 - correlation_ct_ts) ** 3
    )
    if not np.isfinite(denominator_squared) or denominator_squared <= 0:
        return np.nan, np.nan
    statistic = (
        (correlation_y_ct - correlation_y_ts)
        * np.sqrt((n - 1) * (1.0 + correlation_ct_ts))
        / np.sqrt(denominator_squared)
    )
    return float(statistic), float(2.0 * stats.t.sf(abs(statistic), df=n - 3))


def build_pooled_ct_ts_statistics(
    donor_scores: pd.DataFrame,
    *,
    phenotype: str,
) -> pd.DataFrame:
    """Calculate all-donor CT/TS correlations and dependent-component tests.

    This mirrors the stored diagnosis-specific robustness statistics: all three
    variables must be observed, Pearson uses stored RINT scores, Spearman uses raw
    scores, and Williams' test compares the two overlapping correlations. BH is
    applied across the displayed modules for this one phenotype.
    """

    required = {"module", "CT_raw", "TS_raw", "CT_rint", "TS_rint", phenotype}
    missing = required.difference(donor_scores.columns)
    if missing:
        raise ValueError(f"Pooled CT–TS donor scores are missing: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for module, group in donor_scores.groupby("module", observed=True, sort=True):
        y = pd.to_numeric(group[phenotype], errors="coerce").to_numpy(dtype=float)
        ct_raw = pd.to_numeric(group["CT_raw"], errors="coerce").to_numpy(dtype=float)
        ts_raw = pd.to_numeric(group["TS_raw"], errors="coerce").to_numpy(dtype=float)
        ct_rint = pd.to_numeric(group["CT_rint"], errors="coerce").to_numpy(dtype=float)
        ts_rint = pd.to_numeric(group["TS_rint"], errors="coerce").to_numpy(dtype=float)
        complete = (
            np.isfinite(y)
            & np.isfinite(ct_raw)
            & np.isfinite(ts_raw)
            & np.isfinite(ct_rint)
            & np.isfinite(ts_rint)
        )
        y = y[complete]
        ct_raw = ct_raw[complete]
        ts_raw = ts_raw[complete]
        ct_rint = ct_rint[complete]
        ts_rint = ts_rint[complete]

        rho_ct, p_spearman_ct = _safe_correlation(ct_raw, y, method="spearman")
        rho_ts, p_spearman_ts = _safe_correlation(ts_raw, y, method="spearman")
        r_rint_ct, p_rint_ct = _safe_correlation(ct_rint, y)
        r_rint_ts, p_rint_ts = _safe_correlation(ts_rint, y)

        rank_y = stats.rankdata(y, method="average")
        rank_ct = stats.rankdata(ct_raw, method="average")
        rank_ts = stats.rankdata(ts_raw, method="average")
        rank_ct_ts, _ = _safe_correlation(rank_ct, rank_ts)
        component_rank = _williams_test(rho_ct, rho_ts, rank_ct_ts, len(y))
        rint_ct_ts, _ = _safe_correlation(ct_rint, ts_rint)
        component_rint = _williams_test(
            r_rint_ct, r_rint_ts, rint_ct_ts, len(y)
        )
        rows.append(
            {
                "module": int(module),
                "diagnosis_group": "All donors",
                "n": int(len(y)),
                "rho_CT": rho_ct,
                "p_spearman_CT": p_spearman_ct,
                "rho_TS": rho_ts,
                "p_spearman_TS": p_spearman_ts,
                "r_rint_CT": r_rint_ct,
                "p_rint_CT": p_rint_ct,
                "r_rint_TS": r_rint_ts,
                "p_rint_TS": p_rint_ts,
                "component_t_rank": component_rank[0],
                "p_component_rank": component_rank[1],
                "component_t_rint": component_rint[0],
                "p_component_rint": component_rint[1],
            }
        )

    result = pd.DataFrame(rows)
    for suffix in ("rank", "rint"):
        fdr = benjamini_hochberg(result[f"p_component_{suffix}"])
        result[f"q_component_{suffix}_within_phenotype"] = fdr
        # The existing finder download retains this legacy field, but no pooled
        # all-outcome dependent-test family was stored or dynamically calculated.
        result[f"q_component_{suffix}_all12_global"] = np.nan
    return result


def build_module_finder_table(
    statistics: pd.DataFrame,
    *,
    ct_ts_diagnosis: str,
    correlation_method: str,
    criterion: str,
) -> pd.DataFrame:
    """Build one module-level table for the two association-difference criteria.

    The CT–TS criterion uses the existing dependent-component test within the chosen
    cohort, which may be all donors or one diagnosis. The Control–AD criterion compares
    independent diagnosis-specific
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
