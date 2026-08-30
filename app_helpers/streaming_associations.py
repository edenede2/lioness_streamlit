"""Memory-bounded all-module association calculations for the Streamlit app."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import wraps
from threading import Lock
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

from app_helpers.correlations import (
    benjamini_hochberg,
    calculate_categorical_associations,
    calculate_correlations,
)
from app_helpers.data import load_aggregate_scope, load_resolved_scope


_STREAMING_ANALYSIS_LOCK = Lock()
_FrameFunction = TypeVar("_FrameFunction", bound=Callable[..., pd.DataFrame])


def _serialized_heavy_read(function: _FrameFunction) -> _FrameFunction:
    """Prevent two sessions from multiplying the same large-memory workload."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with _STREAMING_ANALYSIS_LOCK:
            return function(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def _module_score_frames(
    modules: Iterable[int],
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str | None,
    scale: str,
    components: tuple[str, ...] | None,
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
) -> Iterator[pd.DataFrame]:
    """Yield one narrow donor-score frame at a time.

    Reading one module per iteration trades a small amount of repeated Parquet
    metadata work for a large reduction in peak memory. Parquet row-group
    statistics and Drive part predicates still prune unrelated modules.
    """

    common = {
        "method": method,
        "metric_family": feature,
        "module_set": module_set,
        "estimator": estimator,
        "edge_rule": edge_rule,
        "differential_edge_rule": differential_edge_rule,
        "differential_fdr_scope": differential_fdr_scope,
        "differential_fdr_threshold": differential_fdr_threshold,
        "score_normalization": score_normalization,
        "metric_scale": scale,
        "include_embedded_metadata": False,
    }
    selected_components = set(components or ())
    for module in modules:
        if resolved:
            source = load_resolved_scope(module=int(module), **common)
            if selected_components:
                source = source.loc[source["component"].isin(selected_components)]
            if source.empty:
                continue
            source = source.rename(columns={f"metric_{scale}": "metric_value"})
        else:
            source = load_aggregate_scope(module=int(module), **common)
            if source.empty:
                continue
            base_columns = [
                column
                for column in [
                    "sample_id", "module", "metric_family", "lioness_method",
                ]
                if column in source
            ]
            pieces = []
            for component in ("CT", "TS"):
                if selected_components and component not in selected_components:
                    continue
                part = source[base_columns].copy()
                part["component"] = component
                part["component_class"] = component
                part["component_label"] = f"{component} aggregate"
                part["metric_value"] = source[f"{component}_{scale}"].to_numpy()
                pieces.append(part)
            if not pieces:
                continue
            source = pd.concat(pieces, ignore_index=True)
        yield source


def _attach_metadata(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    selected = ["sample_id", *dict.fromkeys(columns)]
    selected = [column for column in selected if column in metadata]
    return scores.merge(
        metadata[selected], on="sample_id", how="left", validate="many_to_one"
    )


def _filter_analysis_rows(
    frame: pd.DataFrame,
    *,
    diagnoses: tuple[str, ...],
    analysis_subset: str,
) -> pd.DataFrame:
    result = frame.loc[frame["diagnosis_group"].isin(diagnoses)].copy()
    if analysis_subset != "all_donors":
        split_value = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        result = result.loc[result["ad_control_split"].eq(split_value)].copy()
    return result


@_serialized_heavy_read
def stream_grouped_correlations(
    modules: Iterable[int],
    metadata: pd.DataFrame,
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    phenotype: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    grouping_variable: str,
    grouping_levels: tuple[object, ...],
    include_pooled: bool,
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
    analysis_subset: str,
) -> pd.DataFrame:
    """Calculate grouped correlations while retaining one module in memory."""

    rows: list[pd.DataFrame] = []
    metadata_columns = ["diagnosis_group", "ad_control_split", phenotype]
    if grouping_variable != "__all__":
        metadata_columns.append(grouping_variable)
    for scores in _module_score_frames(
        modules, module_set=module_set, estimator=estimator, method=method,
        resolved=resolved, feature=feature, scale=scale, components=components,
        edge_rule=edge_rule, differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    ):
        long = _attach_metadata(scores, metadata, metadata_columns)
        long = _filter_analysis_rows(
            long, diagnoses=diagnoses, analysis_subset=analysis_subset
        )
        if grouping_variable == "__all__":
            long["grouping_variable"] = "__all__"
            long["grouping_level"] = "__all__"
        else:
            long = long.loc[
                long[grouping_variable].notna()
                & long[grouping_variable].isin(grouping_levels)
            ].copy()
            long["grouping_variable"] = grouping_variable
            long["grouping_level"] = long[grouping_variable]
        pieces = [long]
        if include_pooled and grouping_variable != "__all__" and not long.empty:
            pooled = long.copy()
            pooled["grouping_level"] = "__pooled__"
            pieces.append(pooled)
        if not pieces or all(piece.empty for piece in pieces):
            continue
        rows.append(
            calculate_correlations(
                pd.concat(pieces, ignore_index=True),
                group_columns=[
                    "module", "metric_family", "component", "component_label",
                    "grouping_variable", "grouping_level",
                ],
                outcomes=[phenotype], min_group_n=int(min_group_n),
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@_serialized_heavy_read
def stream_categorical_associations(
    modules: Iterable[int],
    metadata: pd.DataFrame,
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    category_variable: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    category_levels: tuple[object, ...],
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
    analysis_subset: str,
) -> pd.DataFrame:
    """Calculate Kruskal–Wallis rows while retaining one module in memory."""

    rows: list[pd.DataFrame] = []
    for scores in _module_score_frames(
        modules, module_set=module_set, estimator=estimator, method=method,
        resolved=resolved, feature=feature, scale=scale, components=components,
        edge_rule=edge_rule, differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    ):
        long = _attach_metadata(
            scores, metadata,
            ["diagnosis_group", "ad_control_split", category_variable],
        )
        long = _filter_analysis_rows(
            long, diagnoses=diagnoses, analysis_subset=analysis_subset
        )
        long = long.loc[
            long[category_variable].notna()
            & long[category_variable].isin(category_levels)
        ].copy()
        if long.empty:
            continue
        rows.append(
            calculate_categorical_associations(
                long,
                ["module", "metric_family", "component", "component_label"],
                category_column=category_variable,
                min_group_n=int(min_group_n),
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@_serialized_heavy_read
def stream_diagnosis_categorical_associations(
    modules: Iterable[int],
    metadata: pd.DataFrame,
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str,
    category_variable: str,
    scale: str,
    components: tuple[str, ...],
    diagnoses: tuple[str, ...],
    include_pooled: bool,
    min_group_n: int,
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
    analysis_subset: str,
) -> pd.DataFrame:
    """Calculate category effects per diagnosis and optionally all donors."""

    rows: list[pd.DataFrame] = []
    for scores in _module_score_frames(
        modules, module_set=module_set, estimator=estimator, method=method,
        resolved=resolved, feature=feature, scale=scale, components=components,
        edge_rule=edge_rule, differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    ):
        long = _attach_metadata(
            scores, metadata,
            ["diagnosis_group", "ad_control_split", category_variable],
        )
        long = _filter_analysis_rows(
            long, diagnoses=diagnoses, analysis_subset=analysis_subset
        )
        pieces = [long]
        if include_pooled and not long.empty:
            pooled = long.copy()
            pooled["diagnosis_group"] = "All donors"
            pieces.append(pooled)
        work = pd.concat(pieces, ignore_index=True)
        if work.empty:
            continue
        rows.append(
            calculate_categorical_associations(
                work,
                [
                    "module", "metric_family", "component", "component_label",
                    "diagnosis_group",
                ],
                category_column=category_variable,
                min_group_n=int(min_group_n),
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@_serialized_heavy_read
def stream_pooled_correlations(
    modules: Iterable[int],
    metadata: pd.DataFrame,
    *,
    module_set: str,
    estimator: str,
    method: str,
    resolved: bool,
    feature: str | None,
    component: str | None,
    outcomes: tuple[str, ...],
    scale: str,
    diagnoses: tuple[str, ...],
    edge_rule: str,
    differential_edge_rule: str,
    differential_fdr_scope: str,
    differential_fdr_threshold: float,
    score_normalization: str,
    analysis_subset: str,
) -> pd.DataFrame:
    """Calculate pooled all-donor heatmap statistics one module at a time."""

    rows: list[pd.DataFrame] = []
    components = (component,) if component is not None else ()
    for scores in _module_score_frames(
        modules, module_set=module_set, estimator=estimator, method=method,
        resolved=resolved, feature=feature, scale=scale, components=components,
        edge_rule=edge_rule, differential_edge_rule=differential_edge_rule,
        differential_fdr_scope=differential_fdr_scope,
        differential_fdr_threshold=differential_fdr_threshold,
        score_normalization=score_normalization,
    ):
        long = _attach_metadata(
            scores, metadata,
            ["diagnosis_group", "ad_control_split", *outcomes],
        )
        long = _filter_analysis_rows(
            long, diagnoses=diagnoses, analysis_subset=analysis_subset
        )
        long["diagnosis_group"] = "All donors"
        rows.append(_pooled_module_correlations(long, outcomes))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pooled_module_correlations(
    frame: pd.DataFrame,
    outcomes: tuple[str, ...],
    *,
    min_group_n: int = 3,
) -> pd.DataFrame:
    """Vectorized Pearson/Spearman tests for one module's pooled donors.

    The complete heatmap can request every feature, component, and outcome. A
    Python/SciPy call for every cell is unnecessarily expensive, so this helper
    ranks and correlates all score columns for one outcome together. Pairwise
    missingness, SciPy's Pearson beta-test p-value, and Spearman t approximation
    match ``calculate_correlations`` to numerical tolerance.
    """

    if frame.empty:
        return pd.DataFrame()
    column_keys = ["metric_family", "component", "component_label"]
    wide = frame.pivot(
        index="sample_id", columns=column_keys, values="metric_value"
    )
    donor_outcomes = (
        frame[["sample_id", *outcomes]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .reindex(wide.index)
    )
    x_matrix = wide.to_numpy(dtype=float)
    modules = frame["module"].dropna().unique()
    if len(modules) != 1:
        raise ValueError("Pooled vectorized correlations require exactly one module")
    module = int(modules[0])
    rows: list[dict[str, object]] = []

    for outcome in outcomes:
        y = pd.to_numeric(donor_outcomes[outcome], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_matrix) & np.isfinite(y)[:, None]
        n = valid.sum(axis=0).astype(int)
        x = np.where(valid, x_matrix, 0.0)
        y_matrix = np.where(valid, y[:, None], 0.0)
        sum_x = x.sum(axis=0)
        sum_y = y_matrix.sum(axis=0)
        sum_xx = np.square(x).sum(axis=0)
        sum_yy = np.square(y_matrix).sum(axis=0)
        sum_xy = (x * y_matrix).sum(axis=0)
        variance_x = n * sum_xx - np.square(sum_x)
        variance_y = n * sum_yy - np.square(sum_y)
        denominator = np.sqrt(np.maximum(0.0, variance_x * variance_y))
        pearson_r = np.divide(
            n * sum_xy - sum_x * sum_y,
            denominator,
            out=np.full(wide.shape[1], np.nan),
            where=denominator > 0,
        )
        pearson_r = np.clip(pearson_r, -1.0, 1.0)
        pearson_p = np.full(wide.shape[1], np.nan)
        pearson_tested = (n >= int(min_group_n)) & np.isfinite(pearson_r)
        shape = n[pearson_tested] / 2.0 - 1.0
        pearson_p[pearson_tested] = 2.0 * stats.beta.cdf(
            -np.abs(pearson_r[pearson_tested]),
            shape,
            shape,
            loc=-1.0,
            scale=2.0,
        )

        rank_x = stats.rankdata(
            np.where(valid, x_matrix, np.nan),
            method="average",
            axis=0,
            nan_policy="omit",
        )
        rank_y = stats.rankdata(
            np.where(valid, y[:, None], np.nan),
            method="average",
            axis=0,
            nan_policy="omit",
        )
        rank_x = np.where(valid, rank_x, 0.0)
        rank_y = np.where(valid, rank_y, 0.0)
        rank_sum_x = rank_x.sum(axis=0)
        rank_sum_y = rank_y.sum(axis=0)
        rank_variance_x = n * np.square(rank_x).sum(axis=0) - np.square(rank_sum_x)
        rank_variance_y = n * np.square(rank_y).sum(axis=0) - np.square(rank_sum_y)
        rank_denominator = np.sqrt(
            np.maximum(0.0, rank_variance_x * rank_variance_y)
        )
        spearman_rho = np.divide(
            n * (rank_x * rank_y).sum(axis=0) - rank_sum_x * rank_sum_y,
            rank_denominator,
            out=np.full(wide.shape[1], np.nan),
            where=rank_denominator > 0,
        )
        spearman_rho = np.clip(spearman_rho, -1.0, 1.0)
        spearman_p = np.full(wide.shape[1], np.nan)
        spearman_tested = (n >= int(min_group_n)) & np.isfinite(spearman_rho)
        degrees_freedom = n[spearman_tested] - 2
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = spearman_rho[spearman_tested] * np.sqrt(
                degrees_freedom
                / (
                    (1.0 + spearman_rho[spearman_tested])
                    * (1.0 - spearman_rho[spearman_tested])
                )
            )
        spearman_p[spearman_tested] = 2.0 * stats.t.sf(
            np.abs(statistic), degrees_freedom
        )

        for index, keys in enumerate(wide.columns):
            eligible = bool(pearson_tested[index] or spearman_tested[index])
            if n[index] < int(min_group_n):
                reason = f"n < {int(min_group_n)}"
            elif variance_x[index] <= 0:
                reason = "constant network score"
            elif variance_y[index] <= 0:
                reason = "constant outcome"
            else:
                reason = ""
            rows.append(
                {
                    "module": module,
                    **dict(zip(column_keys, keys, strict=True)),
                    "diagnosis_group": "All donors",
                    "outcome": outcome,
                    "n": int(n[index]),
                    "eligible": eligible,
                    "minimum_group_n": int(min_group_n),
                    "unavailable_reason": reason,
                    "pearson_r": pearson_r[index],
                    "pearson_p": pearson_p[index],
                    "spearman_rho": spearman_rho[index],
                    "spearman_p": spearman_p[index],
                }
            )

    result = pd.DataFrame(rows)
    result["pearson_fdr_displayed_family"] = benjamini_hochberg(result["pearson_p"])
    result["spearman_fdr_displayed_family"] = benjamini_hochberg(result["spearman_p"])
    return result
