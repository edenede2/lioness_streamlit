"""CAMEL identities and leakage-safe, component-wise frozen-reference arithmetic.

This module has no application I/O or private identifiers. Missing regions must
be removed before scoring; they are never filled with expression from a donor's
other region. The numerical estimator remains Control-referenced LIONESS.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

CAMEL_NAME = "CAMEL"
CAMEL_EXPANSION = "Contrastive Anchored Modular Edge-type LIONESS"
VALIDATION_SCOPE = "incomplete_region_validation"
VALIDATION_LABEL = "Incomplete-region validation donors"
CAMEL_MODULE_SETS = frozenset({"control_derived", "muscle_control_l4"})
COMPONENT_TISSUES = {
    "TS_AC": ("AC",), "TS_DLPFC": ("DLPFC",), "TS_PCGBA23": ("PCGBA23",),
    "CT_AC__DLPFC": ("AC", "DLPFC"),
    "CT_AC__PCGBA23": ("AC", "PCGBA23"),
    "CT_DLPFC__PCGBA23": ("DLPFC", "PCGBA23"),
}
COMPONENT_LABELS = {
    "TS_AC": "AC", "TS_DLPFC": "DLPFC", "TS_PCGBA23": "PCG",
    "CT_AC__DLPFC": "AC–DLPFC", "CT_AC__PCGBA23": "AC–PCG",
    "CT_DLPFC__PCGBA23": "DLPFC–PCG",
}
FEATURES = ("connectivity", "positive_density", "negative_density", "abs_sum", "positive_abs_sum", "negative_abs_sum")


def robust_asinh_scale(values: np.ndarray) -> float:
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return 1.
    median=np.median(x)
    candidates=(np.median(np.abs(x)),np.median(np.abs(x-median)),
                (np.quantile(x,.75)-np.quantile(x,.25))/1.349,
                np.std(x,ddof=1) if len(x)>1 else np.nan)
    return float(next((v for v in candidates if np.isfinite(v) and v>np.sqrt(np.finfo(float).eps)),1.))


def score_framework(module_set: str, method: str = "control_anchored", estimator: str = "lioness") -> str:
    return "CAMEL" if module_set in CAMEL_MODULE_SETS and method == "control_anchored" and estimator == "lioness" else ("BONOBO" if estimator == "bonobo" else "LIONESS")


@dataclass(frozen=True)
class FrozenRINT:
    x: np.ndarray
    z: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray) -> "FrozenRINT":
        values = np.asarray(training, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2 or np.unique(values).size < 2:
            return cls(np.array([]), np.array([]))
        scores = norm.ppf((rankdata(values, method="average") - .5) / len(values))
        order = np.argsort(values, kind="stable")
        x, first = np.unique(values[order], return_index=True)
        return cls(x, scores[order][first])

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        result = np.full(values.shape, np.nan)
        valid = np.isfinite(values)
        if len(self.x):
            result[valid] = np.interp(values[valid], self.x, self.z)
        return result


def correlation_edges(reference: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson edge sufficient statistics; reference is donors × genes."""
    x = np.asarray(reference, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("A finite reference of at least three donors is required")
    mean = x.mean(axis=0)
    centered = x - mean
    ss = centered.T @ centered
    return mean, np.diag(ss).copy(), ss[rows, columns]


def frozen_edge_weights(reference: np.ndarray, targets: np.ndarray, rows: np.ndarray,
                        columns: np.ndarray, powers: np.ndarray, *,
                        reference_positions: np.ndarray | None = None,
                        reference_statistics: tuple | None = None) -> np.ndarray:
    """Exact add-one weights; LOO is permitted only for original-reference QA.

    Target diagnosis and the membership/order of other targets are not inputs.
    The result is targets × undirected edges. Call with bounded edge batches.
    """
    ref, targets = np.asarray(reference, float), np.asarray(targets, float)
    if not np.isfinite(targets).all():
        raise ValueError("Do not impute absent regions before CAMEL scoring")
    mean, variance, cross = (correlation_edges(ref, rows, columns)
                             if reference_statistics is None else reference_statistics)
    n = len(ref)
    positions = np.full(len(targets), -1, dtype=int) if reference_positions is None else np.asarray(reference_positions)
    if positions.shape != (len(targets),):
        raise ValueError("Reference role shape mismatch")
    loo = positions >= 0
    if np.any(positions[loo] >= n) or (loo.any() and not np.array_equal(targets[loo], ref[positions[loo]])):
        raise ValueError("LOO targets must exactly equal their reference row")
    denominator = np.sqrt(variance[rows] * variance[columns])
    corr = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    adjacency = np.sign(corr) * np.abs(np.clip(corr, -1, 1)) ** powers
    delta = targets - mean
    factor = np.where(loo, -n/(n-1), n/(n+1))
    vx = np.maximum(variance[rows] + factor[:, None] * delta[:, rows] ** 2, 0)
    vy = np.maximum(variance[columns] + factor[:, None] * delta[:, columns] ** 2, 0)
    cov = cross + factor[:, None] * delta[:, rows] * delta[:, columns]
    den = np.sqrt(vx * vy)
    changed = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    changed = np.sign(changed) * np.abs(np.clip(changed, -1, 1)) ** powers
    result = (n+1)*changed - n*adjacency
    result[loo] = n*adjacency - (n-1)*changed[loo]
    return result


def feature_values(summary: pd.DataFrame, family: str, normalization: str) -> np.ndarray:
    """Reconstruct existing six feature conventions from compact summaries."""
    n = summary.n_retained_edges.to_numpy(float)
    p, m = summary.positive_weight_sum.to_numpy(float), summary.negative_weight_magnitude.to_numpy(float)
    if family == "positive_density":
        count = summary.n_positive_edges.to_numpy(float)
        result = np.divide(p, count, out=np.zeros_like(p), where=count > 0)
    elif family == "negative_density":
        count = summary.n_negative_edges.to_numpy(float)
        result = np.divide(-m, count, out=np.zeros_like(m), where=count > 0)
    else:
        numerator = {"connectivity": p-m, "abs_sum": p+m,
                     "positive_abs_sum": p, "negative_abs_sum": m}[family]
        result = numerator.copy()
        if family == "connectivity" and normalization == "standard_pruned":
            result = 2*result/summary.n_features.to_numpy(float)
        elif normalization == "retained_edge":
            result = np.divide(result, n, out=np.full_like(result, np.nan), where=n > 0)
    if normalization == "retained_edge":
        result[n == 0] = np.nan
    elif normalization != "standard_pruned":
        raise ValueError("Unknown score handling")
    result[summary.n_possible_edges.to_numpy(float) == 0] = np.nan
    return result
