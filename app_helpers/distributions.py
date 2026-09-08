"""Robust group-distribution comparisons for module-level donor scores."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from app_helpers.correlations import benjamini_hochberg


def distribution_contrasts(
    levels: Sequence[object],
    *,
    mode: str = "reference",
    reference_level: object | None = None,
) -> list[tuple[object, object]]:
    """Return ordered ``(reference, comparison)`` distribution contrasts."""

    ordered = list(dict.fromkeys(levels))
    if mode == "reference":
        if reference_level not in ordered:
            raise ValueError("The reference level must be one of the selected levels")
        return [(reference_level, value) for value in ordered if value != reference_level]
    if mode == "all_pairs":
        return list(itertools.combinations(ordered, 2))
    raise ValueError(f"Unknown distribution contrast mode: {mode}")


def default_reference_level(
    variable: str,
    levels: Sequence[object],
    counts: dict[object, int] | pd.Series,
    *,
    minimum_group_n: int,
) -> object | None:
    """Choose a stable biological reference, falling back to the largest group."""

    selected = list(levels)
    preferred = {
        "diagnosis_group": "Control",
        "clusters": 1,
        "cogdx": 1,
        "cerad_score": 4,
        "adnc": 0,
        "parkinsonism": 0,
        "sex_code": "Code 0",
        "apoe_genotype": "ε3/ε3",
    }.get(variable)

    def count(value: object) -> int:
        try:
            return int(counts.get(value, 0))
        except AttributeError:
            return 0

    eligible = [value for value in selected if count(value) >= int(minimum_group_n)]
    if variable == "braak_stage" and eligible:
        return eligible[0]
    if preferred in eligible:
        return preferred
    if eligible:
        order = {str(value): index for index, value in enumerate(selected)}
        return max(eligible, key=lambda value: (count(value), -order[str(value)]))
    return None


def _finite(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def cliffs_delta(comparison: np.ndarray, reference: np.ndarray) -> float:
    """Return Cliff's delta for comparison minus reference, including ties."""

    comparison = np.asarray(comparison, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if comparison.size == 0 or reference.size == 0:
        return float("nan")
    ordered_reference = np.sort(reference)
    wins = np.searchsorted(ordered_reference, comparison, side="left").sum()
    losses = (
        reference.size
        - np.searchsorted(ordered_reference, comparison, side="right")
    ).sum()
    return float((wins - losses) / (comparison.size * reference.size))


def _bootstrap_seed(
    seed: int,
    base: dict[str, object],
    reference_level: object,
    comparison_level: object,
) -> int:
    payload = json.dumps(
        [seed, sorted((str(key), str(value)) for key, value in base.items()),
         str(reference_level), str(comparison_level)],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _bootstrap_cliffs_delta_ci(
    comparison: np.ndarray,
    reference: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        sampled_comparison = comparison[
            rng.integers(0, comparison.size, size=comparison.size)
        ]
        sampled_reference = reference[
            rng.integers(0, reference.size, size=reference.size)
        ]
        estimates[index] = cliffs_delta(sampled_comparison, sampled_reference)
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def calculate_pairwise_distribution_statistics(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    category_column: str,
    contrasts: Iterable[tuple[object, object]],
    minimum_group_n: int = 10,
    bootstrap_resamples: int = 0,
    seed: int = 42,
    include_ks: bool = True,
) -> pd.DataFrame:
    """Calculate robust pairwise comparisons for fixed category contrasts."""

    requested = list(contrasts)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, observed=True, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        for reference_level, comparison_level in requested:
            reference = _finite(
                group.loc[group[category_column].eq(reference_level), "metric_value"]
            )
            comparison = _finite(
                group.loc[group[category_column].eq(comparison_level), "metric_value"]
            )
            n_reference = int(reference.size)
            n_comparison = int(comparison.size)
            eligible = True
            unavailable_reason = ""
            if n_reference < int(minimum_group_n):
                eligible = False
                unavailable_reason = f"reference n < {int(minimum_group_n)}"
            elif n_comparison < int(minimum_group_n):
                eligible = False
                unavailable_reason = f"comparison n < {int(minimum_group_n)}"
            elif np.unique(np.concatenate([reference, comparison])).size <= 1:
                eligible = False
                unavailable_reason = "constant scores"

            mann_u = mann_p = delta = superiority = np.nan
            ci_low = ci_high = ks_d = ks_p = np.nan
            if eligible:
                mann_result = mannwhitneyu(
                    comparison, reference, alternative="two-sided", method="auto"
                )
                mann_u = float(mann_result.statistic)
                mann_p = float(mann_result.pvalue)
                delta = cliffs_delta(comparison, reference)
                superiority = float((delta + 1.0) / 2.0)
                ci_low, ci_high = _bootstrap_cliffs_delta_ci(
                    comparison,
                    reference,
                    resamples=int(bootstrap_resamples),
                    seed=_bootstrap_seed(
                        seed, base, reference_level, comparison_level
                    ),
                )
                if include_ks:
                    ks_result = ks_2samp(
                        comparison, reference, alternative="two-sided", method="auto"
                    )
                    ks_d = float(ks_result.statistic)
                    ks_p = float(ks_result.pvalue)

            def quantile(values: np.ndarray, probability: float) -> float:
                return float(np.quantile(values, probability)) if values.size else np.nan

            rows.append(
                {
                    **base,
                    "grouping_variable": category_column,
                    "reference_level": reference_level,
                    "comparison_level": comparison_level,
                    "reference_level_key": str(reference_level),
                    "comparison_level_key": str(comparison_level),
                    "n_reference": n_reference,
                    "n_comparison": n_comparison,
                    "n_total": n_reference + n_comparison,
                    "median_reference": quantile(reference, 0.5),
                    "q1_reference": quantile(reference, 0.25),
                    "q3_reference": quantile(reference, 0.75),
                    "median_comparison": quantile(comparison, 0.5),
                    "q1_comparison": quantile(comparison, 0.25),
                    "q3_comparison": quantile(comparison, 0.75),
                    "median_difference": (
                        quantile(comparison, 0.5) - quantile(reference, 0.5)
                    ),
                    "eligible": eligible,
                    "unavailable_reason": unavailable_reason,
                    "mann_whitney_u": mann_u,
                    "mann_whitney_p": mann_p,
                    "cliffs_delta": delta,
                    "probability_superiority": superiority,
                    "cliffs_delta_ci_low": ci_low,
                    "cliffs_delta_ci_high": ci_high,
                    "ks_d": ks_d,
                    "ks_p": ks_p,
                    "minimum_group_n": int(minimum_group_n),
                    "bootstrap_resamples": int(bootstrap_resamples),
                }
            )
    return pd.DataFrame(rows)


def add_pairwise_across_module_fdr(
    frame: pd.DataFrame,
    *,
    family_columns: Iterable[str],
    module_column: str = "module",
) -> pd.DataFrame:
    """Apply BH across modules separately for every fixed group contrast."""

    result = frame.copy()
    families = [
        *family_columns,
        "reference_level_key",
        "comparison_level_key",
    ]
    duplicate = result.duplicated([*families, module_column], keep=False)
    if duplicate.any():
        raise ValueError("Pairwise FDR family contains duplicate module rows")
    grouped = result.groupby(families, observed=True, dropna=False, sort=False)
    for prefix in ("mann_whitney", "ks"):
        p_column = f"{prefix}_p"
        fdr_column = f"{prefix}_fdr_across_modules"
        n_column = f"{prefix}_fdr_module_family_n"
        result[fdr_column] = grouped[p_column].transform(benjamini_hochberg)
        result[n_column] = grouped[p_column].transform(
            lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())
        ).astype(int)
    return result
