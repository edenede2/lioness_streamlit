"""Numerical kernels for offline gene/edge screening; no application I/O.

Each Spearman test ranks precisely its pairwise-complete observations. Original
search FDR is computed before selection; replication FDR has a different,
explicitly prespecified selected-hit denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, false_discovery_control, rankdata, t

CORRELATIONS = ("pearson", "spearman")
SCHEMA_VERSION = 1


def correlations(values, outcome, *, min_n=10):
    """Vectorized donors × tests, exact pairwise handling (including missing X)."""
    x, y = np.asarray(values, float), np.asarray(outcome, float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("Expected donors × tests and one outcome per donor")
    if min_n < 3:
        raise ValueError("min_n must be at least three")
    size = x.shape[1]
    n = np.zeros(size, np.int32)
    result = {k: np.full(size, np.nan) for k in ("pearson_r", "pearson_p", "spearman_r", "spearman_p")}
    reason = np.full(size, "too_few_complete_pairs", dtype=object)
    masks = np.isfinite(x) & np.isfinite(y[:, None])
    # Usually one mask for an entire edge batch. Group exceptional missingness
    # without incorrect global ranking or one Python call for every edge.
    packed = np.packbits(masks, axis=0).T
    _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    for group, column in enumerate(first):
        columns = np.flatnonzero(inverse == group)
        mask = masks[:, column]
        count = int(mask.sum()); n[columns] = count
        if count < min_n:
            continue
        a, b = x[mask][:, columns], y[mask]
        if np.ptp(b) == 0:
            reason[columns] = "constant_outcome"; continue
        nonconstant = np.ptp(a, axis=0) > 0
        reason[columns] = "constant_predictor"
        indices = columns[nonconstant]
        if not len(indices):
            continue
        a = a[:, nonconstant]
        for method in CORRELATIONS:
            aa = rankdata(a, axis=0, method="average") if method == "spearman" else a
            bb = rankdata(b, method="average") if method == "spearman" else b
            aa = aa - aa.mean(axis=0); bb = bb - bb.mean()
            den = np.sqrt(np.sum(aa*aa, axis=0)*np.sum(bb*bb))
            r = np.clip(np.sum(aa*bb[:, None], axis=0)/den, -1., 1.)
            if method == "pearson":
                p = 2*beta.cdf(-np.abs(r), count/2-1, count/2-1, loc=-1, scale=2)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    statistic = r*np.sqrt((count-2)/((1-r)*(1+r)))
                p = 2*t.sf(np.abs(statistic), count-2)
            result[f"{method}_r"][indices] = r
            result[f"{method}_p"][indices] = np.clip(p, 0., 1.)
        reason[indices] = ""
    return pd.DataFrame({"n": n, **result, "unavailable_reason": reason})


def bh(values):
    """Exact ordinary BH; missing tests are not part of the denominator."""
    p = np.asarray(values, float)
    valid = np.isfinite(p)
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError("P-values must lie in [0, 1]")
    q = np.full(p.shape, np.nan)
    if valid.any():
        q[valid] = false_discovery_control(p[valid], method="bh")
    return q, int(valid.sum())


def top_hits(frame, k, method, *, identity="identity"):
    """Top by |r|; canonical identity breaks ties, never p or input order."""
    if k < 1 or method not in CORRELATIONS:
        raise ValueError("Invalid K or correlation method")
    valid = frame.loc[np.isfinite(frame[f"{method}_r"])].copy()
    valid["absolute_correlation"] = valid[f"{method}_r"].abs()
    valid = valid.sort_values(["absolute_correlation", identity], ascending=[False, True], kind="stable").head(k)
    valid["selection_method"] = method
    valid["selection_rank"] = np.arange(1, len(valid)+1)
    return valid


def selected_union(frame, k, *, identity="identity"):
    """One physical hit with both ranks when selected by either search."""
    ranks = []
    for method in CORRELATIONS:
        ranks.append(top_hits(frame, k, method, identity=identity)[[identity, "selection_rank"]].rename(columns={"selection_rank": f"{method}_rank"}).set_index(identity))
    selected = ranks[0].join(ranks[1], how="outer").reset_index()
    return selected.merge(frame, on=identity, validate="one_to_one").sort_values(identity).reset_index(drop=True)


def replication_statistics(values, outcomes, selected, *, min_n=10):
    """Test the frozen union once per outcome; neither new ranking nor full FDR."""
    results = []
    for outcome, hits in selected.groupby("outcome", sort=False):
        indices = hits["value_index"].to_numpy(int)
        stats = correlations(values[:, indices], outcomes[outcome], min_n=min_n)
        stats.insert(0, "identity", hits.identity.to_numpy())
        stats.insert(1, "outcome", outcome)
        for method in CORRELATIONS:
            stats[f"{method}_fdr"], family_n = bh(stats[f"{method}_p"])
            stats[f"{method}_tested_n"] = family_n
        results.append(stats)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def signed_alt_edges(reference, targets, rows, columns, powers, *, reference_positions=None, batch_statistics=None):
    """Exact LOO/add-one without constructing a genes × genes covariance matrix."""
    ref, target = np.asarray(reference, float), np.asarray(targets, float)
    rows, columns = np.asarray(rows, int), np.asarray(columns, int)
    if not np.isfinite(ref).all() or not np.isfinite(target).all() or len(ref) < 3:
        raise ValueError("Finite expression and >=3 reference donors required")
    positions = np.full(len(target), -1, int) if reference_positions is None else np.asarray(reference_positions, int)
    if positions.shape != (len(target),):
        raise ValueError("Reference role dimensions disagree")
    loo = positions >= 0
    if np.any(positions[loo] >= len(ref)) or not np.array_equal(target[loo], ref[positions[loo]]):
        raise ValueError("LOO target must be its exact reference row")
    if batch_statistics is None:
        mean = ref.mean(axis=0); centered = ref-mean
        variance = np.sum(centered*centered, axis=0)
        cross = np.sum(centered[:, rows]*centered[:, columns], axis=0)
    else:
        mean, variance, cross = batch_statistics
    den = np.sqrt(variance[rows]*variance[columns])
    r = np.divide(cross, den, out=np.zeros_like(cross), where=den>0)
    base = np.sign(r)*np.abs(np.clip(r, -1, 1))**powers
    delta = target-mean; n = len(ref)
    factor = np.where(loo, -n/(n-1), n/(n+1))[:, None]
    v1 = np.maximum(variance[rows]+factor*delta[:, rows]**2, 0)
    v2 = np.maximum(variance[columns]+factor*delta[:, columns]**2, 0)
    den = np.sqrt(v1*v2)
    cross = cross+factor*delta[:, rows]*delta[:, columns]
    changed = np.divide(cross, den, out=np.zeros_like(cross), where=den>0)
    changed = np.sign(changed)*np.abs(np.clip(changed, -1, 1))**powers
    weights = (n+1)*changed-n*base
    weights[loo] = n*base-(n-1)*changed[loo]
    return weights


def edge_batches(n, batch_size=2048):
    """Canonical upper triangle in bounded arrays (no global triu_indices)."""
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    left, right = [], []
    for i in range(n-1):
        start = i+1
        while start < n:
            take = min(batch_size-len(left), n-start)
            left.extend([i]*take); right.extend(range(start, start+take)); start += take
            if len(left) == batch_size:
                yield np.asarray(left, np.int32), np.asarray(right, np.int32)
                left, right = [], []
    if left:
        yield np.asarray(left, np.int32), np.asarray(right, np.int32)
