"""Bounded, deterministic brain-wide signedAlt screening kernels.

Physical nodes are sorted (tissue, internal gene ID) tuples. Upper-triangle
indices therefore provide a fixed canonical tie break without billions of
temporary endpoint strings. Internal IDs never enter public catalogs.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata, t

from scripts.screening_bh import DUCKDB_VERSION


def component_indices(start, stop, left_n, right_n=None):
    """Random-access edges; no preceding batches or full triu allocation."""
    k = np.arange(start, stop, dtype=np.int64)
    total = left_n*(left_n-1)//2 if right_n is None else left_n*right_n
    if not 0 <= start <= stop <= total:
        raise ValueError("Edge interval outside component")
    if right_n is not None:
        return k//right_n, k % right_n
    rows = np.arange(left_n, dtype=np.int64)
    offsets = rows*(2*left_n-rows-1)//2
    left = np.searchsorted(offsets, k, side="right")-1
    return left, left+1+k-offsets[left]


def edge_ids(left, right, node_count):
    left, right = np.asarray(left, np.int64), np.asarray(right, np.int64)
    if np.any(left >= right) or np.any(left < 0) or np.any(right >= node_count):
        raise ValueError("Expected canonical upper-triangle nodes")
    return left*(2*node_count-left-1)//2+right-left-1


def edge_nodes(ids, node_count):
    ids = np.asarray(ids, np.int64)
    if np.any(ids < 0) or np.any(ids >= node_count*(node_count-1)//2):
        raise ValueError("Invalid physical edge ID")
    rows = np.arange(node_count, dtype=np.int64)
    offsets = rows*(2*node_count-rows-1)//2
    left = np.searchsorted(offsets, ids, side="right")-1
    return left, left+1+ids-offsets[left]


class FrozenSignedAlt:
    """Frozen reference, exact LOO roles or independent add-one target scoring."""
    def __init__(self, reference, targets, positions=None):
        reference, targets = np.asarray(reference, float), np.asarray(targets, float)
        if reference.ndim != 2 or targets.ndim != 2 or reference.shape[1] != targets.shape[1]:
            raise ValueError("Reference/target dimensions disagree")
        if len(reference) < 3 or not np.isfinite(reference).all() or not np.isfinite(targets).all():
            raise ValueError("Finite expression and >=3 reference donors required")
        self.n = len(reference)
        self.mean = reference.mean(axis=0)
        self.centered = reference-self.mean
        self.variance = np.sum(self.centered*self.centered, axis=0)
        self.delta = targets-self.mean
        self.positions = np.full(len(targets), -1, int) if positions is None else np.asarray(positions, int)
        if self.positions.shape != (len(targets),) or np.any(self.positions < -1):
            raise ValueError("Invalid reference roles")
        self.loo = self.positions >= 0
        if (np.any(self.positions[self.loo] >= self.n) or
                not np.array_equal(targets[self.loo], reference[self.positions[self.loo]])):
            raise ValueError("LOO donor must be its exact reference row")
        self.factor = np.where(self.loo, -self.n/(self.n-1), self.n/(self.n+1))[:, None]

    def weights(self, left, right, power):
        left, right = np.asarray(left, int), np.asarray(right, int)
        cross = np.sum(self.centered[:, left]*self.centered[:, right], axis=0)
        denominator = np.sqrt(self.variance[left]*self.variance[right])
        raw = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
        base = np.sign(raw)*np.abs(np.clip(raw, -1, 1))**power
        a, b = self.delta[:, left], self.delta[:, right]
        v1 = np.maximum(self.variance[left]+self.factor*a*a, 0)
        v2 = np.maximum(self.variance[right]+self.factor*b*b, 0)
        denominator = np.sqrt(v1*v2)
        raw = np.divide(cross+self.factor*a*b, denominator,
                        out=np.zeros_like(denominator), where=denominator > 0)
        changed = np.sign(raw)*np.abs(np.clip(raw, -1, 1))**power
        result = (self.n+1)*changed-self.n*base
        result[self.loo] = self.n*base-(self.n-1)*changed[self.loo]
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite donor weights")
        return result


class FiniteOutcomeCorrelator:
    """Share exact ranks only across outcomes with identical observed donors."""
    def __init__(self, outcomes, min_n=10):
        self.outcomes = {name:np.asarray(y, float) for name,y in outcomes.items()}
        sizes = {len(y) for y in self.outcomes.values()}
        if len(sizes) != 1 or min_n < 3:
            raise ValueError("Outcome dimensions/minimum N disagree")
        self.donors = sizes.pop()
        self.groups = {}
        for name,y in self.outcomes.items():
            mask = np.isfinite(y)
            self.groups.setdefault(mask.tobytes(), (mask, []))[1].append(name)
        self.min_n = min_n

    def calculate(self, values):
        x = np.asarray(values, float)
        if x.ndim != 2 or len(x) != self.donors or not np.isfinite(x).all():
            raise ValueError("Fast kernel requires finite donors × edges")
        result = {name:dict(n=np.full(x.shape[1], np.isfinite(y).sum(), np.int32),
            **{f"{r}_{field}":np.full(x.shape[1], np.nan)
               for r in ("pearson", "spearman") for field in ("r", "p")})
            for name,y in self.outcomes.items()}
        for mask,names in self.groups.values():
            n = int(mask.sum())
            if n < self.min_n:
                continue
            a = x[mask]
            variable = np.ptp(a, axis=0) > 0
            if not variable.any():
                continue
            a = a[:, variable]
            for method in ("pearson", "spearman"):
                aa = rankdata(a, axis=0, method="average") if method == "spearman" else a.copy()
                aa -= aa.mean(axis=0)
                norm = np.sqrt(np.sum(aa*aa, axis=0))
                for name in names:
                    y = self.outcomes[name][mask]
                    if np.ptp(y) == 0:
                        continue
                    b = rankdata(y, method="average") if method == "spearman" else y.copy()
                    b -= b.mean()
                    r = np.clip(np.sum(aa*b[:, None], axis=0)/(norm*np.sqrt(np.sum(b*b))), -1, 1)
                    if method == "pearson":
                        p = 2*beta.cdf(-np.abs(r), n/2-1, n/2-1, loc=-1, scale=2)
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            statistic = r*np.sqrt((n-2)/((1-r)*(1+r)))
                        p = 2*t.sf(np.abs(statistic), n-2)
                    result[name][f"{method}_r"][variable] = r
                    result[name][f"{method}_p"][variable] = np.clip(p, 0, 1)
        return result


def top_indices(ids, values, k):
    """Exact K by |r|; include deterministic boundary ties without full sorting."""
    finite = np.flatnonzero(np.isfinite(values))
    magnitude = np.abs(np.asarray(values)[finite])
    if len(finite) > k:
        boundary = np.partition(magnitude, len(magnitude)-k)[len(magnitude)-k]
        above = finite[magnitude > boundary]
        tied = finite[magnitude == boundary]
        tied = tied[np.argsort(np.asarray(ids)[tied], kind="stable")[:k-len(above)]]
        finite = np.r_[above, tied]
    return finite[np.lexsort((np.asarray(ids)[finite], -np.abs(np.asarray(values)[finite])))]


class TopUnion:
    def __init__(self, outcomes, k=1000):
        if k < 1:
            raise ValueError("Positive top K required")
        self.k = k
        self.rows = {name:{} for name in outcomes}

    def update(self, ids, results):
        for name,stats in results.items():
            incoming = dict(test_id=np.asarray(ids, np.int64), **stats)
            old = self.rows[name]
            joined = {column:np.r_[old[column],value] if old else value for column,value in incoming.items()}
            selected = np.union1d(top_indices(joined["test_id"], joined["pearson_r"], self.k),
                                  top_indices(joined["test_id"], joined["spearman_r"], self.k))
            self.rows[name] = {column:np.asarray(value)[selected] for column,value in joined.items()}

    def frame(self):
        frames = []
        for name,arrays in self.rows.items():
            if not arrays or not len(arrays["test_id"]):
                continue
            frame = pd.DataFrame(arrays)
            for r in ("pearson", "spearman"):
                positions = top_indices(frame.test_id.to_numpy(), frame[f"{r}_r"].to_numpy(), self.k)
                frame[f"{r}_rank"] = np.nan
                frame.loc[positions, f"{r}_rank"] = np.arange(1, len(positions)+1)
            frame["outcome"] = name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scoped_external_bh(paths, p_column, selected_ids, *, temporary_directory,
                       memory_gib=4, batch_size=65536):
    """One disk-backed sort, exact network AND component BH cumulative minima."""
    import duckdb
    if duckdb.__version__ != DUCKDB_VERSION or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", p_column):
        raise ValueError("Pinned DuckDB and safe p-value column required")
    if not 0.0625 <= memory_gib <= 4:
        raise ValueError("BH memory outside acceptance budget")
    directory = Path(temporary_directory); directory.mkdir(parents=True, exist_ok=True)
    wanted = np.unique(np.asarray(selected_ids, np.int64))
    found = []
    db = duckdb.connect()
    try:
        db.execute(f"SET memory_limit='{int(memory_gib*1024)}MB'")
        db.execute("SET threads=1")
        db.execute("SET preserve_insertion_order=false")
        db.execute("SET max_temp_directory_size='95GiB'")
        db.execute("SET temp_directory=?", [str(directory)])
        db.read_parquet([str(p) for p in paths]).project(f"test_id,scope_id,{p_column} AS p").create_view("tests")
        if db.execute("SELECT count(*) FROM tests WHERE isfinite(p) AND (p<0 OR p>1)").fetchone()[0]:
            raise ValueError("Invalid original p-values")
        counts = {int(s):int(n) for s,n in db.execute(
            "SELECT scope_id,count(*) FROM tests WHERE isfinite(p) GROUP BY scope_id").fetchall()}
        total = sum(counts.values()); remaining = dict(counts)
        rank, running = total, 1.
        component_min = {s:1. for s in counts}
        reader = db.execute("SELECT test_id,scope_id,p FROM tests WHERE isfinite(p) ORDER BY p DESC,test_id DESC").fetch_record_batch(batch_size)
        for batch in reader:
            ids, scopes, p = (batch.column(i).to_numpy() for i in range(3))
            network_q = np.minimum.accumulate(np.minimum(running, p*total/np.arange(rank,rank-len(p),-1)))
            running = float(network_q[-1]); rank -= len(p)
            component_q = np.empty(len(p))
            for scope in np.unique(scopes):
                positions = np.flatnonzero(scopes == scope)
                q = np.minimum.accumulate(np.minimum(component_min[int(scope)],
                    p[positions]*counts[int(scope)]/np.arange(remaining[int(scope)],remaining[int(scope)]-len(positions),-1)))
                component_q[positions] = q
                component_min[int(scope)] = float(q[-1]); remaining[int(scope)] -= len(positions)
            if len(wanted):
                indices = np.searchsorted(wanted, ids)
                keep = (indices < len(wanted)) & (wanted[np.minimum(indices,len(wanted)-1)] == ids)
                found.extend(zip(ids[keep].tolist(), scopes[keep].tolist(), network_q[keep].tolist(), component_q[keep].tolist()))
        if rank != 0 or any(remaining.values()):
            raise RuntimeError("BH family/stream mismatch")
        result = pd.DataFrame(found, columns=["test_id","scope_id","network_fdr","component_fdr"])
        return result, dict(network_tested_n=total, component_tested_n=counts)
    finally:
        db.close()
