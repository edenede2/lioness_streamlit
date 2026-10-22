"""Private full no-q90 M5 estimator and direct LIONESS role arithmetic.

This is NOT the app's historical beta-only motor donor estimator. Every changed
reference rebuilds bootstrap frequency, conditional mean and eligible CT null.
Random priority keys are donor/edge identities, never diagnosis, module ID,
array position or chunk boundaries. Float64 avoids LIONESS amplification of
float32 adjacency rounding. Module-free execution requires a resource review.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

PROFILES = {
    "control_fitted": {"VH":5., "SMA":7., "QUAD":13., "VH-SMA":8., "VH-QUAD":10., "SMA-QUAD":10.},
    "all_donor_fitted": {"VH":6., "SMA":6., "QUAD":10., "VH-SMA":6., "VH-QUAD":6., "SMA-QUAD":7.},
}
TISSUES = ("VH", "SMA", "QUAD")


@dataclass(frozen=True)
class M5Settings:
    seed: int = 42
    subsamples: int = 200
    donor_fraction: float = .80
    threshold: float = .10
    null_permutations: int = 50
    null_edge_cap: int = 100_000
    null_epsilon: float = 1e-8


def priority(seed, namespace, identity):
    value = f"{seed}|{namespace}|{identity}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def canonical_input(matrix, donor_ids, node_ids):
    x = np.asarray(matrix, float)
    donors, nodes = list(donor_ids), list(node_ids)
    if x.shape != (len(donors), len(nodes)) or len(set(donors)) != len(donors) or len(set(nodes)) != len(nodes):
        raise ValueError("Unique donor/node identities and matched finite expression required")
    if not np.isfinite(x).all() or len(donors) < 13:
        raise ValueError("M5 requires finite expression and >=13 reference donors")
    if any(t not in TISSUES for t, g in nodes):
        raise ValueError("Unknown motor tissue")
    di = sorted(range(len(donors)), key=lambda i: donors[i])
    ni = sorted(range(len(nodes)), key=lambda i: (TISSUES.index(nodes[i][0]), nodes[i][1]))
    return x[np.ix_(di, ni)], [donors[i] for i in di], [nodes[i] for i in ni], np.argsort(ni)


def subsample_plan(donors, settings):
    count = max(10, int(len(donors)*settings.donor_fraction))
    if count > len(donors):
        raise ValueError("Insufficient M5 donors")
    return [np.asarray(sorted(range(len(donors)), key=lambda i: (priority(settings.seed, f"bootstrap:{b}", donors[i]), donors[i]))[:count], int)
            for b in range(settings.subsamples)]


def paired_correlations(left, right):
    a = left-left.mean(axis=0); b = right-right.mean(axis=0)
    den = np.sqrt(np.sum(a*a, axis=0)*np.sum(b*b, axis=0))
    return np.divide(np.sum(a*b, axis=0), den, out=np.zeros(a.shape[1]), where=den>0)


def full_m5(matrix, donor_ids, node_ids, profile, *, settings=M5Settings()):
    """Complete estimator on a bounded module; returned adjacency is original order.

    All eligible CT edges participate in uniform hash-priority sampling. The
    null is recalculated for this exact reference. No frozen factors, q90 or
    clipping. A separate streamed implementation is required for all-gene scale.
    """
    if profile not in PROFILES:
        raise ValueError("Unknown fitted motor beta profile")
    x, donors, nodes, inverse = canonical_input(matrix, donor_ids, node_ids)
    n = len(nodes)
    if n > 3000:
        raise ValueError("Dense M5 is module/pilot only; genome-wide execution is review-gated")
    count = np.zeros((n, n), np.uint16)
    sums = np.zeros((n, n), float)
    for subset in subsample_plan(donors, settings):
        a = x[subset]-x[subset].mean(axis=0)
        norms = np.sqrt(np.sum(a*a, axis=0))
        a = np.divide(a, norms, out=np.zeros_like(a), where=norms>0)
        r = np.clip(a.T@a, -1., 1.); np.fill_diagonal(r, 0)
        keep = np.abs(r)>settings.threshold
        count += keep; sums += np.where(keep, r, 0.)
    freq = count.astype(float)/settings.subsamples
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count>0)
    adjacency = np.zeros((n, n), float)
    diagnostics = {}
    for ti, first in enumerate(TISSUES):
        i = np.asarray([k for k, (t,g) in enumerate(nodes) if t == first], int)
        if not len(i):
            continue
        for second in TISSUES[ti:]:
            j = np.asarray([k for k, (t,g) in enumerate(nodes) if t == second], int)
            if not len(j):
                continue
            block = np.ix_(i, j)
            f, m = freq[block], means[block]
            beta = PROFILES[profile][first if first == second else f"{first}-{second}"]
            w = np.sign(m)*f*np.abs(m)**beta
            diag = dict(beta=beta, eligible_edges=int((f>0).sum()))
            if first != second:
                li, lj = np.where(f>0)
                eligible = len(li)
                # Uniform random priorities with canonical tie-breaks are a
                # sample without replacement, stable under array reordering.
                if eligible > settings.null_edge_cap:
                    keys = np.fromiter((priority(settings.seed, f"null_edges:{first}-{second}", f"{nodes[i[a]][1]}|{nodes[j[b]][1]}")
                                        for a,b in zip(li, lj, strict=True)), np.uint64, eligible)
                    keep = np.argsort(keys, kind="stable")[:settings.null_edge_cap]
                    li, lj = li[keep], lj[keep]
                total, square, size = 0., 0., 0
                for rep in range(settings.null_permutations):
                    perm = np.asarray(sorted(range(len(donors)), key=lambda k: (priority(settings.seed, f"null_donors:{first}-{second}:{rep}", donors[k]), donors[k])), int)
                    # Bounded batches; never retain 50 full null vectors.
                    for start in range(0, len(li), 2048):
                        aa, bb = i[li[start:start+2048]], j[lj[start:start+2048]]
                        r = np.abs(paired_correlations(x[:, aa], x[perm][:, bb]))
                        total += float(r.sum()); square += float(np.dot(r, r)); size += len(r)
                mu = total/size if size else 0.
                sigma = math.sqrt(max(square/size-mu*mu, 0.)) if size else 1.
                w *= np.maximum((np.abs(m)-mu)/(sigma+settings.null_epsilon), 0.)
                diag.update(null_mean=mu, null_sd=sigma, null_edges=len(li), null_values=size)
            adjacency[block] = w
            if first != second:
                adjacency[np.ix_(j,i)] = w.T
            diagnostics[f"{first}-{second}"] = diag
    np.fill_diagonal(adjacency, 0.)
    if not np.isfinite(adjacency).all() or not np.allclose(adjacency, adjacency.T, rtol=0, atol=1e-14):
        raise RuntimeError("Full M5 adjacency symmetry/finite gate failed")
    return adjacency[np.ix_(inverse, inverse)], diagnostics


def m5_donor_network(reference, reference_ids, target, target_id, node_ids, profile, *, settings=M5Settings(), base=None):
    """Correct inclusion-direction LIONESS; recompute the complete estimator."""
    ref = np.asarray(reference, float); target = np.asarray(target, float)
    ids = list(reference_ids); n = len(ids)
    if target.shape != (ref.shape[1],):
        raise ValueError("Target must have every required observed node")
    if base is None:
        base, _ = full_m5(ref, ids, node_ids, profile, settings=settings)
    if target_id in ids:
        pos = ids.index(target_id)
        if not np.array_equal(target, ref[pos]):
            raise ValueError("LOO donor expression does not match reference")
        changed, diagnostics = full_m5(np.delete(ref, pos, axis=0), ids[:pos]+ids[pos+1:], node_ids, profile, settings=settings)
        weights = n*base-(n-1)*changed
        role = "leave_one_out"
    else:
        changed, diagnostics = full_m5(np.vstack((ref, target)), ids+[target_id], node_ids, profile, settings=settings)
        weights = (n+1)*changed-n*base
        role = "add_one"
    return weights, dict(role=role, reference_n=n, changed=diagnostics)
