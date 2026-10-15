"""Pure, bounded robustness kernels shared with the pathology score extension."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import rankdata, t as student_t


def _fast_corr(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 4:
        return n, np.nan, np.nan
    selected_x = x[valid]
    selected_y = y[valid]
    centered_x = selected_x - selected_x.mean()
    centered_y = selected_y - selected_y.mean()
    denominator = float(
        np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return n, np.nan, np.nan
    coefficient = float(np.clip(np.dot(centered_x, centered_y) / denominator, -1, 1))
    residual = max(0.0, 1.0 - coefficient * coefficient)
    if residual <= np.finfo(float).eps:
        p_value = 0.0
    else:
        statistic = abs(coefficient) * math.sqrt((n - 2) / residual)
        p_value = float(2.0 * student_t.sf(statistic, n - 2))
    return n, coefficient, p_value


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 4:
        return n, np.nan, np.nan
    return _fast_corr(
        rankdata(x[valid], method="average"),
        rankdata(y[valid], method="average"),
    )


def _fast_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 4:
        return np.nan, np.nan, np.nan, np.nan
    selected_x = x[valid]
    selected_y = y[valid]
    centered_x = selected_x - selected_x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0 or np.unique(selected_y).size < 2:
        return np.nan, np.nan, np.nan, np.nan
    slope = float(np.dot(centered_x, selected_y - selected_y.mean()) / denominator)
    intercept = float(selected_y.mean() - slope * selected_x.mean())
    return intercept, slope, float(selected_x.min()), float(selected_x.max())


def _robust_skew(values: np.ndarray) -> float:
    selected = values[np.isfinite(values)]
    if len(selected) < 3:
        return np.nan
    standard_deviation = float(np.std(selected, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(np.mean(np.power(selected - selected.mean(), 3)) / standard_deviation**3)


def _winsorize_asinh(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        lower, upper = np.quantile(
            values[valid], [0.025, 0.975], method="median_unbiased"
        )
        result[valid] = np.clip(values[valid], lower, upper)
    return result


def _leave_one_out(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    selected_x = x[valid]
    selected_y = y[valid]
    n = len(selected_x)
    if n < 6 or np.unique(selected_x).size < 2 or np.unique(selected_y).size < 2:
        return np.nan, np.nan, np.nan
    sum_x, sum_y = selected_x.sum(), selected_y.sum()
    sum_x2, sum_y2 = np.square(selected_x).sum(), np.square(selected_y).sum()
    sum_xy = (selected_x * selected_y).sum()
    full_covariance = sum_xy - sum_x * sum_y / n
    full_variance_x = sum_x2 - sum_x**2 / n
    full_variance_y = sum_y2 - sum_y**2 / n
    full_denominator_sq = max(full_variance_x, 0.0) * max(full_variance_y, 0.0)
    if not np.isfinite(full_denominator_sq) or full_denominator_sq <= 0:
        return np.nan, np.nan, np.nan
    full = full_covariance / math.sqrt(full_denominator_sq)
    n_minus = n - 1
    sum_x_minus = sum_x - selected_x
    sum_y_minus = sum_y - selected_y
    covariance = (
        sum_xy - selected_x * selected_y
        - sum_x_minus * sum_y_minus / n_minus
    )
    variance_x = sum_x2 - selected_x**2 - sum_x_minus**2 / n_minus
    variance_y = sum_y2 - selected_y**2 - sum_y_minus**2 / n_minus
    denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
    estimates = np.divide(
        covariance,
        denominator,
        out=np.full(n, np.nan),
        where=denominator > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return np.nan, np.nan, np.nan
    return (
        float(np.max(np.abs(estimates - full))),
        float(np.min(np.abs(estimates))),
        float(np.any(np.sign(estimates) != np.sign(full))),
    )


def _williams_test(
    r_y_ct: float, r_y_ts: float, r_ct_ts: float, n: int
) -> tuple[float, float]:
    values = np.asarray([r_y_ct, r_y_ts, r_ct_ts], dtype=float)
    if n < 6 or not np.isfinite(values).all():
        return np.nan, np.nan
    determinant = (
        1.0 - r_y_ct**2 - r_y_ts**2 - r_ct_ts**2
        + 2.0 * r_y_ct * r_y_ts * r_ct_ts
    )
    denominator_sq = (
        2.0 * determinant * (n - 1) / (n - 3)
        + ((r_y_ct + r_y_ts) ** 2 / 4.0) * (1.0 - r_ct_ts) ** 3
    )
    if not np.isfinite(denominator_sq) or denominator_sq <= 0:
        return np.nan, np.nan
    statistic = (
        (r_y_ct - r_y_ts)
        * math.sqrt((n - 1) * (1.0 + r_ct_ts))
        / math.sqrt(denominator_sq)
    )
    return float(statistic), float(2.0 * student_t.sf(abs(statistic), n - 3))
