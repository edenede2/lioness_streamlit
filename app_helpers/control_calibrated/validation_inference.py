"""Donor-level rank inference; never fits or changes connectivity scores.

Freedman--Lane uses fixed initial ranks and refits the same nuisance span for
each pseudo-outcome. Shared matrix products are algebraically identical to
those refits. Bootstrap replicates instead rerank and refit sampled donors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats

COVARIATES = ("age_at_death", "sex_code", "education_years", "study")
CATEGORIES = {"sex_code": ("Code 0", "Code 1"), "study": ("MAP", "ROS")}
TOL = 64 * np.finfo(float).eps


def stable_seed(seed, *keys):
    value = json.dumps([int(seed), *keys], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def membership_hash(ids):
    return hashlib.sha256(
        json.dumps(sorted(map(str, ids)), separators=(",", ":")).encode()
    ).hexdigest()


def finite_numeric(values):
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )


def complete_covariates(meta, covariates=COVARIATES):
    keep = np.ones(len(meta), bool)
    for name in covariates:
        if name not in meta:
            return np.zeros(len(meta), bool)
        keep &= (
            meta[name].notna().to_numpy()
            if name in CATEGORIES
            else np.isfinite(finite_numeric(meta[name]))
        )
        if name in CATEGORIES:
            keep &= meta[name].isin(CATEGORIES[name]).to_numpy()
    return keep


@dataclass
class Design:
    matrix: np.ndarray
    names: tuple[str, ...]
    removed: tuple[str, ...]
    rank: int
    reason: str = ""
    age_knots: tuple[float, ...] = ()


def nuisance_design(meta, covariates=COVARIATES, *, spline_age=False, age_knots=None):
    n = len(meta)
    if not n:
        return Design(np.ones((0, 1)), ("Intercept",), (), 0, "no_complete_cases")
    columns, names, removed = [np.ones(n)], ["Intercept"], []
    knots = ()
    for name in covariates:
        if name not in meta:
            return Design(
                np.ones((n, 1)),
                ("Intercept",),
                (),
                1 if n else 0,
                "missing_covariate:" + name,
            )
        if name in CATEGORIES:
            values = meta[name]
            if values.isna().any() or not values.isin(CATEGORIES[name]).all():
                return Design(
                    np.ones((n, 1)),
                    ("Intercept",),
                    (),
                    1 if n else 0,
                    "missing_or_unknown_category:" + name,
                )
            for level in CATEGORIES[name][1:]:
                value = values.eq(level).to_numpy(float)
                label = name + "=" + level + " (reference " + CATEGORIES[name][0] + ")"
                if np.ptp(value) == 0:
                    removed.append(label + ":constant")
                else:
                    columns.append(value)
                    names.append(label)
            continue
        value = finite_numeric(meta[name])
        if not np.isfinite(value).all():
            return Design(
                np.ones((n, 1)),
                ("Intercept",),
                (),
                1 if n else 0,
                "missing_covariate:" + name,
            )
        if not len(value) or np.ptp(value) == 0:
            removed.append(name + ":constant")
            continue
        if name == "age_at_death" and spline_age:
            knots = tuple(
                np.quantile(value, [0.05, 0.35, 0.65, 0.95])
                if age_knots is None
                else age_knots
            )
            if len(knots) != 4 or len(set(knots)) != 4:
                return Design(
                    np.column_stack(columns),
                    tuple(names),
                    tuple(removed),
                    0,
                    "spline_age_knots_not_distinct",
                    knots,
                )
            a, b = knots[-2:]

            def d(k):
                return (
                    np.maximum(value - k, 0) ** 3 - np.maximum(value - b, 0) ** 3
                ) / (b - k)

            basis = [value, d(knots[0]) - d(a), d(knots[1]) - d(a)]
            labels = [
                "natural_age_linear",
                "natural_age_nonlinear_1",
                "natural_age_nonlinear_2",
            ]
        else:
            basis = [stats.rankdata(value, method="average")]
            labels = ["midrank(" + name + ")"]
        for vector, label in zip(basis, labels):
            if np.std(vector) <= np.finfo(float).eps:
                removed.append(label + ":constant")
            else:
                columns.append((vector - vector.mean()) / vector.std())
                names.append(label)
    z = np.column_stack(columns)
    rank = np.linalg.matrix_rank(z) if n else 0
    reason = "rank_deficient_nuisance_design" if rank != z.shape[1] else ""
    return Design(z, tuple(names), tuple(removed), int(rank), reason, knots)


def permutation_indices(n, draws, seed, blocks=None, batch=256):
    """Independent RNG per block makes results invariant to draw chunking."""
    labels = np.zeros(n, int) if blocks is None else np.asarray(blocks).astype(str)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    rngs = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(len(groups))
    ]
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        result = np.broadcast_to(np.arange(n), (count, n)).copy()
        for positions, rng in zip(groups, rngs):
            result[:, positions] = rng.permuted(
                np.broadcast_to(positions, (count, len(positions))), axis=1
            )
        yield result


def permutation_space(ranks, cap=100_000):
    counts = np.unique(ranks, return_counts=True)[1]
    total, left = 1, len(ranks)
    for count in counts:
        total *= math.comb(left, int(count))
        if total > cap:
            return cap + 1
        left -= int(count)
    return total


def unique_pairings(values):
    levels, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts, kind="stable")
    levels, counts = levels[order], counts[order]
    buffer = np.empty(len(values))

    def visit(level, free):
        if level == len(levels) - 1:
            buffer[list(free)] = levels[level]
            yield buffer.copy()
            return
        for positions in combinations(free, int(counts[level])):
            buffer[list(positions)] = levels[level]
            chosen = set(positions)
            yield from visit(level + 1, tuple(p for p in free if p not in chosen))

    yield from visit(0, tuple(range(len(values))))


@lru_cache(maxsize=512)
def _exact_null(x_sorted, y_sorted):
    x, y = np.asarray(x_sorted), np.asarray(y_sorted)
    cap = min(100_000, 8_000_000 // len(x))
    sx, sy = permutation_space(x, cap), permutation_space(y, cap)
    if min(sx, sy) > cap:
        return None
    if sx < sy:
        x, y = y, x
    x = (x - x.mean()) / np.linalg.norm(x - x.mean())
    permutations = np.asarray(list(unique_pairings(y)))
    permutations = (permutations - y.mean()) / np.linalg.norm(y - y.mean())
    return np.sort(np.abs(permutations @ x))


def _normal_rows(values):
    norm = np.linalg.norm(values, axis=1)
    ok = norm > np.maximum(1, np.linalg.norm(values, axis=1)) * 1e-12
    return (
        np.divide(values, norm[:, None], out=np.zeros_like(values), where=ok[:, None]),
        norm,
        ok,
    )


def infer_group(
    x,
    y,
    meta,
    *,
    adjusted=False,
    draws=9999,
    seed=611104,
    blocks=None,
    minimum_n=30,
    minimum_df=20,
    covariates=COVARIATES,
    spline_age=False,
    batch=256,
):
    """x is score rows by donors, with one shared exact complete-case subset."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, n = x.shape
    if y.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inference requires an explicit common finite donor subset")
    design = nuisance_design(
        meta, covariates if adjusted else (), spline_age=spline_age
    )
    result = dict(
        effect=np.full(m, np.nan),
        p=np.full(m, np.nan),
        reason=np.full(m, "", dtype=object),
        draws=np.zeros(m, int),
        exceedances=np.zeros(m, int),
        method=np.full(m, "not_estimable", dtype=object),
        design=design,
        residual_df=n - design.rank - 1,
        rank_beta=np.full(m, np.nan),
        hc3_se=np.full(m, np.nan),
        hc3_p=np.full(m, np.nan),
        residual_variance_ratio=np.full(m, np.nan),
        degenerate_permutations=0,
    )
    if n < 3:
        result["reason"][:] = "insufficient_paired_donors"
        return result
    if design.reason:
        result["reason"][:] = design.reason
        return result
    rx = stats.rankdata(x, axis=1, method="average")
    ry = stats.rankdata(y, method="average")
    q = np.linalg.qr(design.matrix, mode="reduced")[0]
    residual_x = rx - (rx @ q) @ q.T
    residual_y = ry - q @ (q.T @ ry)
    unit_x, nx, valid_x = _normal_rows(residual_x)
    valid_x &= nx > 1e-10 * np.maximum(1, np.linalg.norm(rx, axis=1))
    ny = np.linalg.norm(residual_y)
    valid_x &= np.ptp(x, axis=1) > 0
    result["reason"][~valid_x] = "constant_score_or_score_explained_by_nuisance"
    if np.ptp(y) == 0 or ny <= 1e-10 * max(1, np.linalg.norm(ry)):
        result["reason"][:] = "constant_outcome_or_outcome_explained_by_nuisance"
        return result
    rho = np.clip(unit_x @ residual_y / ny, -1, 1)
    result["effect"][valid_x] = rho[valid_x]
    df = result["residual_df"]
    if adjusted:
        beta = np.divide(
            residual_x @ residual_y, nx**2, out=np.full(m, np.nan), where=valid_x
        )
        error = residual_y[None, :] - beta[:, None] * residual_x
        h = np.sum(q * q, axis=1)[None, :] + unit_x**2
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(np.sum((residual_x * error / (1 - h)) ** 2, axis=1)) / nx**2
            result["hc3_p"] = 2 * stats.t.sf(np.abs(beta / se), df)
        result["rank_beta"], result["hc3_se"] = beta, se
        if "study" in meta and set(meta.study.dropna()) >= {"MAP", "ROS"}:
            group = meta.study.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                result["residual_variance_ratio"] = np.var(
                    error[:, group == "ROS"], axis=1, ddof=1
                ) / np.var(error[:, group == "MAP"], axis=1, ddof=1)
        if n < minimum_n or df < minimum_df:
            result["reason"][valid_x] = (
                "primary_ineligible_n_below_30"
                if n < minimum_n
                else "primary_ineligible_residual_df_below_20"
            )
            return result
    active = valid_x.copy()
    if not adjusted:
        y_sorted = tuple(sorted(ry))
        cap = min(100_000, 8_000_000 // n)
        yspace = permutation_space(ry, cap)
        for i in np.flatnonzero(valid_x):
            if yspace <= cap or permutation_space(rx[i], cap) <= cap:
                null = _exact_null(tuple(sorted(rx[i])), y_sorted)
                if null is not None:
                    b = len(null) - np.searchsorted(
                        null, abs(rho[i]) - TOL, side="left"
                    )
                    result["p"][i] = b / len(null)
                    result["exceedances"][i] = b
                    result["draws"][i] = len(null)
                    result["method"][i] = "exact_tie_aware_pairings"
                    active[i] = False
    positions = np.flatnonzero(active)
    if len(positions):
        counts = np.zeros(len(positions), np.int64)
        unit = np.ascontiguousarray(unit_x[positions].T)
        target = np.abs(rho[positions]) - TOL
        vector = residual_y if adjusted else ry - ry.mean()
        norm = np.linalg.norm(vector)
        for indices in permutation_indices(
            n, draws, seed, blocks if adjusted else None, batch
        ):
            permuted = vector[indices]
            if adjusted:
                denominator = np.sqrt(
                    np.maximum(0, norm**2 - np.sum((permuted @ q) ** 2, axis=1))
                )
            else:
                denominator = np.full(len(indices), norm)
            degenerate = denominator <= 1e-10 * max(1, norm)
            result["degenerate_permutations"] += int(degenerate.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                simulated = np.abs(permuted @ unit) / denominator[:, None]
            counts += np.sum((simulated >= target) | degenerate[:, None], axis=0)
        result["p"][positions] = (counts + 1) / (draws + 1)
        result["draws"][positions] = draws
        result["exceedances"][positions] = counts
        result["method"][positions] = (
            (
                "freedman_lane_within_study"
                if blocks is not None
                else "freedman_lane_unrestricted"
            )
            if adjusted
            else "monte_carlo_pairings"
        )
    return result


def bootstrap_group(
    x,
    y,
    meta,
    *,
    draws=199,
    seed=611105,
    covariates=COVARIATES,
    adjusted=True,
    spline_age=False,
):
    """Stratified donor percentile intervals; B/C share every donor resample."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, n = x.shape
    unadjusted = np.full((draws, m), np.nan)
    conditional = unadjusted.copy()
    if n < 3:
        return dict(
            unadjusted_low=np.full(m, np.nan),
            unadjusted_high=np.full(m, np.nan),
            adjusted_low=np.full(m, np.nan),
            adjusted_high=np.full(m, np.nan),
            unadjusted_valid=np.zeros(m, int),
            adjusted_valid=np.zeros(m, int),
        )
    original = (
        nuisance_design(meta, covariates, spline_age=spline_age) if adjusted else None
    )
    labels = (
        meta.study.fillna("Missing study").astype(str).to_numpy()
        if "study" in meta
        else np.repeat("all", n)
    )
    groups = [np.flatnonzero(labels == v) for v in np.unique(labels)]
    rng = np.random.default_rng(seed)
    for b in range(draws):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        rx = stats.rankdata(x[:, ids], axis=1, method="average")
        ry = stats.rankdata(y[ids], method="average")
        ux, _, ok = _normal_rows(rx - rx.mean(axis=1, keepdims=True))
        uy = ry - ry.mean()
        norm = np.linalg.norm(uy)
        if norm <= 1e-12:
            continue
        unadjusted[b, ok] = np.clip(ux[ok] @ uy / norm, -1, 1)
        if adjusted and not original.reason:
            design = nuisance_design(
                meta.iloc[ids],
                covariates,
                spline_age=spline_age,
                age_knots=original.age_knots or None,
            )
            if design.reason or design.names != original.names or n <= design.rank + 1:
                continue
            q = np.linalg.qr(design.matrix, mode="reduced")[0]
            ry = ry - q @ (q.T @ ry)
            norm = np.linalg.norm(ry)
            residual = rx - (rx @ q) @ q.T
            ux, norm_x, ok = _normal_rows(residual)
            ok &= norm_x > 1e-10 * np.maximum(1, np.linalg.norm(rx, axis=1))
            if norm > 1e-10:
                conditional[b, ok] = np.clip(ux[ok] @ ry / norm, -1, 1)
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for label, values in [("unadjusted", unadjusted), ("adjusted", conditional)]:
            count = np.isfinite(values).sum(axis=0)
            bounds = np.nanquantile(values, [0.025, 0.975], axis=0)
            bounds[:, count < math.ceil(0.8 * draws)] = np.nan
            result[label + "_low"], result[label + "_high"] = bounds
            result[label + "_valid"] = count
    return result
