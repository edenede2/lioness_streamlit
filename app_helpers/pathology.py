"""Additive, module-only-corrected pathology statistics from saved donor scores.

Never constructs networks or changes existing outcome/correction catalogs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import rankdata

from app_helpers.correlations import benjamini_hochberg
from app_helpers.score_robustness import (
    _fast_corr, _fast_spearman, _fast_slope, _leave_one_out, _robust_skew,
    _williams_test, _winsorize_asinh,
)

OUTCOME_COLUMNS = {
    "gpath": "gpath", "amyloid": "amylsqrt_est_8reg", "tangles": "tangsqrt_est_8reg",
}
PATHOLOGY_OUTCOMES = tuple(OUTCOME_COLUMNS)
SCHEMA_VERSION = 1


def component_statistics(raw, asinh, rint, outcome) -> dict:
    """Independent finite-pair masks; structurally absent blocks stay unavailable."""
    raw, asinh, rint, outcome = [np.asarray(v, dtype=float) for v in (raw, asinh, rint, outcome)]
    n, _, _ = _fast_corr(raw, outcome)
    result = {"n": n, "metric_zero_variance": np.unique(raw[np.isfinite(raw)]).size < 2}
    for name, values in (("raw", raw), ("asinh", asinh), ("rint", rint), ("winsor", _winsorize_asinh(asinh))):
        _, result[f"r_{name}"], result[f"p_{name}"] = _fast_corr(values, outcome)
        if name != "winsor":
            result[f"skew_{name}"] = _robust_skew(values)
    _, result["rho"], result["p_spearman"] = _fast_spearman(raw, outcome)
    for name, values in (("asinh", asinh), ("rint", rint)):
        delta, minimum, flip = _leave_one_out(values, outcome)
        result.update({f"loo_{name}_max_delta": delta, f"loo_{name}_min_abs": minimum,
                       f"loo_{name}_sign_flip": flip})
    for name, value in zip(("intercept_rint", "slope_rint", "x_min_rint", "x_max_rint"), _fast_slope(rint, outcome)):
        result[name] = value
    result["unavailable_reason"] = (
        "n < 4" if n < 4 else "constant or unavailable values" if not np.isfinite(result["rho"]) else ""
    )
    return result


def statistics_from_scores(scores: pd.DataFrame, *, resolved: bool, outcomes=PATHOLOGY_OUTCOMES) -> pd.DataFrame:
    """Calculate raw/asinh/Winsor/Spearman/RINT/LOO and paired Williams tests."""
    groups = ["module", "metric_family", "diagnosis_group", "lioness_method"]
    if resolved:
        groups += ["component", "component_class", "component_label"]
    rows = []
    for keys, group in scores.groupby(groups, observed=True, sort=False, dropna=False):
        base = dict(zip(groups, keys, strict=True))
        for outcome in outcomes:
            y = pd.to_numeric(group[outcome], errors="coerce").to_numpy(float)
            row = {**base, "phenotype": outcome, "fdr_scope": "across_modules"}
            if resolved:
                row.update(component_statistics(group.metric_raw, group.metric_asinh, group.metric_rint, y))
            else:
                for component in ("CT", "TS"):
                    statistics = component_statistics(*(group[f"{component}_{scale}"] for scale in ("raw", "asinh", "rint")), y)
                    row.update({f"{key}_{component}": value for key, value in statistics.items()})
                ct, ts = group.CT_raw.to_numpy(float), group.TS_raw.to_numpy(float)
                valid = np.isfinite(ct) & np.isfinite(ts) & np.isfinite(y)
                row["n"] = int(valid.sum())
                for name in ("asinh", "rint", "rank"):
                    if name == "rank":
                        a, b, target = (rankdata(v[valid]) for v in (ct, ts, y))
                    else:
                        a, b, target = group[f"CT_{name}"].to_numpy(float)[valid], group[f"TS_{name}"].to_numpy(float)[valid], y[valid]
                    statistic, p = _williams_test(_fast_corr(a, target)[1], _fast_corr(b, target)[1], _fast_corr(a, b)[1], len(target))
                    row[f"component_t_{name}"], row[f"p_component_{name}"] = statistic, p
            rows.append(row)
    return pd.DataFrame(rows)


def adjust_module_fdr(frame: pd.DataFrame, *, resolved: bool) -> pd.DataFrame:
    """Every p-column corrected over modules, never over outcomes or groups."""
    if frame.empty:
        return frame
    result = frame.copy()
    family = ["lioness_method", "metric_family", "diagnosis_group", "phenotype"]
    if resolved:
        family += ["component"]
    if result.duplicated([*family, "module"]).any():
        raise ValueError("Pathology statistics must have one row per module and family")
    grouped = result.groupby(family, observed=True, dropna=False, sort=False)
    for column in [v for v in result if v.startswith("p_")]:
        prefix = "q_" + column[2:]
        result[f"{prefix}_across_modules"] = grouped[column].transform(benjamini_hochberg)
        result[f"{prefix}_across_modules_n"] = grouped[column].transform("count")
        result[f"{prefix}_within_phenotype"] = result[f"{prefix}_across_modules"]
        # Old consumer schemas still request these columns. Do not falsely
        # label a module-only correction as an all-12/primary-five correction.
        result[f"{prefix}_all12_global"] = np.nan
        result[f"{prefix}_primary5_global"] = np.nan
    return result


def statistics_path(module_set: str, estimator: str, method: str, edge_rule: str, resolved: bool, cohort_scope: str = "complete_450"):
    from app_helpers.data import DATA_DIR
    return DATA_DIR / "pathology" / module_set / estimator / method / edge_rule / cohort_scope / ("resolved_statistics.parquet" if resolved else "aggregate_statistics.parquet")


@st.cache_data(show_spinner="Calculating pathology statistics from saved scores…", max_entries=2, ttl=1800)
def _calculate_statistics(module_set: str, estimator: str, method: str, edge_rule: str,
                          resolved: bool, metric_family: str | None, differential_edge_rule: str,
                          differential_fdr_scope: str, differential_fdr_threshold: float,
                          score_normalization: str, analysis_subset: str, cohort_scope: str) -> pd.DataFrame:
    from app_helpers import data as D
    from app_helpers.streaming_associations import _STREAMING_ANALYSIS_LOCK
    with _STREAMING_ANALYSIS_LOCK:
        metadata = D.load_sample_metadata(module_set, cohort_scope)
        wanted = ["sample_id", "diagnosis_group", *PATHOLOGY_OUTCOMES]
        if analysis_subset != "all_donors":
            split = {"discovery_ad_control": "Discovery", "validation_ad_control": "Validation", "mci_external": "MCI_external"}[analysis_subset]
            metadata = metadata.loc[metadata.ad_control_split.eq(split)]
        metadata = metadata[wanted]
        rows = []
        for module in D.load_module_annotations(module_set).module.astype(int).unique():
            kwargs = dict(method=method, module=int(module), metric_family=metric_family,
                          module_set=module_set, estimator=estimator, edge_rule=edge_rule,
                          differential_edge_rule=differential_edge_rule,
                          differential_fdr_scope=differential_fdr_scope,
                          differential_fdr_threshold=differential_fdr_threshold,
                          score_normalization=score_normalization, include_embedded_metadata=False)
            if resolved:
                kwargs["cohort_scope"] = cohort_scope
            source = (D.load_resolved_scope if resolved else D.load_aggregate_scope)(**kwargs)
            if source.empty:
                continue
            source = source.merge(metadata, on="sample_id", how="inner", validate="many_to_one")
            rows.append(statistics_from_scores(source, resolved=resolved))
            # Only the compact summary survives each donor-score read.
            del source
        return adjust_module_fdr(pd.concat(rows, ignore_index=True), resolved=resolved) if rows else pd.DataFrame()


def load_statistics(*, resolved: bool, method: str, module: int | None = None,
                    phenotype: str | None = None, metric_family: str | None = None,
                    module_set: str = "full_cohort", estimator: str = "lioness", edge_rule: str = "all",
                    differential_edge_rule: str = "all", differential_fdr_scope: str = "global",
                    differential_fdr_threshold: float = .05, score_normalization: str = "standard_pruned",
                    analysis_subset: str = "all_donors", component: str | None = None,
                    diagnosis_group: str | None = None, cohort_scope: str = "complete_450") -> pd.DataFrame:
    from app_helpers import data as D
    filters = [(key, "=", value) for key, value in (("module", module), ("phenotype", phenotype),
        ("metric_family", metric_family), ("diagnosis_group", diagnosis_group), ("component", component)) if value is not None]
    path = statistics_path(module_set, estimator, method, edge_rule, resolved, cohort_scope)
    if differential_edge_rule == "all" and analysis_subset == "all_donors" and D.data_path_available(path):
        return D._read_filtered(path, filters)
    result = _calculate_statistics(module_set, estimator, method, edge_rule, resolved, metric_family,
        differential_edge_rule, differential_fdr_scope, float(differential_fdr_threshold),
        score_normalization, analysis_subset, cohort_scope)
    for key, _, value in filters:
        result = result.loc[result[key].eq(value)]
    return result.reset_index(drop=True)
