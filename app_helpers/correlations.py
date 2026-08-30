"""Correlation summaries used by the module and all-module heatmaps."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, kruskal, pearsonr, spearmanr


# Increment when the callable contract changes.  Streamlit Community Cloud can
# retain this module across an entrypoint hot reload; the app uses this sentinel
# to force a reload before importing helpers with a newer signature.
GROUPED_ASSOCIATION_API_VERSION = 1


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


def add_across_module_fdr(
    frame: pd.DataFrame,
    *,
    family_columns: Iterable[str],
    module_column: str = "module",
) -> pd.DataFrame:
    """Adjust Pearson and Spearman p-values across modules within fixed strata.

    Each family must contain at most one row per module. Missing p-values are retained
    as missing and are excluded from the BH denominator, as no correlation was tested.
    """

    family_columns = list(family_columns)
    required = {
        module_column,
        *family_columns,
        "pearson_p",
        "spearman_p",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Across-module FDR input is missing columns: " + ", ".join(sorted(missing))
        )
    result = frame.copy()
    duplicate = result.duplicated([*family_columns, module_column], keep=False)
    if duplicate.any():
        example = result.loc[
            duplicate, [*family_columns, module_column]
        ].head(5)
        raise ValueError(
            "Across-module FDR requires one correlation per module and family; "
            f"duplicate keys include {example.to_dict(orient='records')}"
        )
    grouped = result.groupby(family_columns, observed=True, dropna=False, sort=False)
    for method in ("pearson", "spearman"):
        p_column = f"{method}_p"
        result[f"{method}_fdr_across_modules"] = grouped[p_column].transform(
            benjamini_hochberg
        )
        result[f"{method}_fdr_module_family_n"] = grouped[p_column].transform(
            lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())
        ).astype(int)
    return result


def calculate_correlations(
    frame: pd.DataFrame,
    group_columns: list[str],
    outcomes: Iterable[str],
    *,
    min_group_n: int = 3,
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
                x_constant = x.nunique() <= 1
                y_constant = y.nunique() <= 1
                eligible = n >= int(min_group_n) and not x_constant and not y_constant
                if eligible:
                    pearson_r, pearson_p = pearsonr(x, y)
                    spearman_rho, spearman_p = spearmanr(x, y)
                if n < int(min_group_n):
                    unavailable_reason = f"n < {int(min_group_n)}"
                elif x_constant:
                    unavailable_reason = "constant network score"
                elif y_constant:
                    unavailable_reason = "constant outcome"
                else:
                    unavailable_reason = ""
                rows.append(
                    {
                        **base,
                        "outcome": outcome,
                        "n": n,
                        "eligible": bool(eligible),
                        "minimum_group_n": int(min_group_n),
                        "unavailable_reason": unavailable_reason,
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


def calculate_categorical_associations(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    category_column: str = "clusters",
    min_group_n: int = 5,
) -> pd.DataFrame:
    """Kruskal–Wallis and epsilon-squared for an unordered donor category.

    Categories with fewer than ``min_group_n`` non-missing score values remain
    visible in plots but are excluded from the omnibus test.  No numeric
    correlation is ever calculated from the category codes.
    """

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, observed=True, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        work = pd.DataFrame(
            {
                "score": pd.to_numeric(group["metric_value"], errors="coerce"),
                "category": group[category_column],
            }
        ).dropna()
        work = work.loc[np.isfinite(work["score"])]
        counts = work.groupby("category", observed=True)["score"].size()
        eligible = counts.index[counts.ge(int(min_group_n))].tolist()
        excluded = counts.index[counts.lt(int(min_group_n))].tolist()
        samples = [
            work.loc[work["category"].eq(value), "score"].to_numpy(dtype=float)
            for value in eligible
        ]
        h_statistic = p_value = epsilon_squared = np.nan
        n_tested = int(sum(len(values) for values in samples))
        k_tested = int(len(samples))
        if k_tested >= 2 and n_tested > k_tested and any(
            np.unique(values).size > 1 for values in samples
        ):
            try:
                h_statistic, p_value = kruskal(*samples, nan_policy="omit")
                epsilon_squared = max(
                    0.0,
                    float((h_statistic - k_tested + 1) / (n_tested - k_tested)),
                )
            except ValueError:
                pass
        row: dict[str, object] = {
            **base,
            "outcome": category_column,
            "n": int(len(work)),
            "n_tested": n_tested,
            "levels_tested": ",".join(map(str, eligible)),
            "levels_excluded_small_n": ",".join(map(str, excluded)),
            "level_counts": "; ".join(
                f"{value}: {int(counts.loc[value])}" for value in counts.index
            ),
            "level_medians": "; ".join(
                f"{value}: {float(work.loc[work['category'].eq(value), 'score'].median()):.6g}"
                for value in counts.index
            ),
            # Retain these aliases for the already-deployed cluster hot cache and
            # backward-compatible downloads.
            "clusters_tested": ",".join(map(str, eligible)),
            "clusters_excluded_small_n": ",".join(map(str, excluded)),
            "kruskal_h": h_statistic,
            "kruskal_df": float(k_tested - 1) if k_tested >= 2 else np.nan,
            "k_tested": k_tested,
            "epsilon_squared": epsilon_squared,
            "categorical_p": p_value,
            "min_group_n": int(min_group_n),
        }
        # Keep the fixed Cluster 1–4 columns for old consumers, while generic
        # category summaries are carried in a separate long table in the app.
        for label in (1, 2, 3, 4):
            values = work.loc[
                work["category"].astype(str).eq(str(label)), "score"
            ]
            row[f"n_cluster_{label}"] = int(len(values))
            row[f"median_cluster_{label}"] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_categorical_across_module_fdr(
    frame: pd.DataFrame,
    *,
    family_columns: Iterable[str],
    module_column: str = "module",
) -> pd.DataFrame:
    """Apply BH to nominal omnibus tests across modules only."""

    result = frame.copy()
    family_columns = list(family_columns)
    duplicate = result.duplicated([*family_columns, module_column], keep=False)
    if duplicate.any():
        raise ValueError("Categorical FDR family contains duplicate module rows")
    grouped = result.groupby(family_columns, observed=True, dropna=False, sort=False)
    result["categorical_fdr_across_modules"] = grouped["categorical_p"].transform(
        benjamini_hochberg
    )
    result["categorical_fdr_module_family_n"] = grouped["categorical_p"].transform(
        lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())
    ).astype(int)
    return result
