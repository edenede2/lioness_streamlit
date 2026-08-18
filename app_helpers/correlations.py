"""Correlation summaries used by the module and all-module heatmaps."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment that preserves missing-value positions."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return result
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    ordered_index = valid.index.to_numpy()[order]
    result.loc[ordered_index] = adjusted
    return result


def calculate_correlations(
    frame: pd.DataFrame,
    group_columns: list[str],
    outcomes: Iterable[str],
) -> pd.DataFrame:
    """Calculate Pearson/Spearman statistics for metric_value and each outcome."""
    rows: list[dict[str, object]] = []
    outcomes = list(outcomes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for keys, group in frame.groupby(group_columns, observed=True, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(group_columns, keys, strict=True))
            x_all = pd.to_numeric(group["metric_value"], errors="coerce")
            for outcome in outcomes:
                y_all = pd.to_numeric(group[outcome], errors="coerce")
                valid = x_all.notna() & y_all.notna() & np.isfinite(x_all) & np.isfinite(y_all)
                x = x_all.loc[valid]
                y = y_all.loc[valid]
                n = int(valid.sum())
                pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
                if n >= 3 and x.nunique() > 1 and y.nunique() > 1:
                    pearson_r, pearson_p = pearsonr(x, y)
                    spearman_rho, spearman_p = spearmanr(x, y)
                rows.append(
                    {
                        **base,
                        "outcome": outcome,
                        "n": n,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["pearson_fdr_displayed_family"] = benjamini_hochberg(result["pearson_p"])
    result["spearman_fdr_displayed_family"] = benjamini_hochberg(result["spearman_p"])
    return result

