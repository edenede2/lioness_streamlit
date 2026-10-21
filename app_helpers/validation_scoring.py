"""Bounded, diagnosis-independent scoring against a frozen reference.

Module rows are positional nodes: repeated tissue-gene assignments must not be
expanded by a duplicate-labelled DataFrame join or silently deduplicated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app_helpers.camel import (FEATURES, FrozenRINT, correlation_edges,
                               feature_values, frozen_edge_weights, robust_asinh_scale)


def formula_checks(reference, targets, rows, columns, power, positions, original_n):
    """Independent sampled correlation recalculation and target invariance."""
    if not len(rows):
        return []
    chosen = np.linspace(0, len(rows) - 1, min(64, len(rows)), dtype=int)
    nodes, inverse = np.unique(np.r_[rows[chosen], columns[chosen]], return_inverse=True)
    ri, ci = inverse[:len(chosen)], inverse[len(chosen):]
    ref = reference[:, nodes]
    sample = list(dict.fromkeys([0, original_n,
        *np.flatnonzero(positions[:original_n] < 0)[:1].tolist()]))
    sample = [i for i in sample if i < len(targets)]
    records = []
    def adjacency(x):
        corr = np.corrcoef(x, rowvar=False)[ri, ci]
        # Zero-variance endpoints follow the existing zero-correlation policy.
        corr = np.nan_to_num(corr, nan=0.)
        return np.sign(corr) * np.abs(corr) ** power
    baseline = adjacency(ref)
    for index in sample:
        position = positions[index]
        target = targets[index, nodes]
        changed = np.delete(ref, position, axis=0) if position >= 0 else np.vstack([ref, target])
        direct = (len(ref)*baseline - (len(ref)-1)*adjacency(changed) if position >= 0
                  else (len(ref)+1)*adjacency(changed) - len(ref)*baseline)
        got = frozen_edge_weights(ref, target[None, :], ri, ci,
            np.full(len(ri), power), reference_positions=np.array([position]))[0]
        error = float(np.max(np.abs(got-direct)))
        if error > 1e-8:
            raise ValueError(f"Direct LIONESS formula gate failed: {error}")
        records.append(dict(role="loo" if position >= 0 else "add_one", max_abs_edge_error=error))
    if original_n < len(targets):
        new = targets[original_n:original_n+2, nodes]
        a = frozen_edge_weights(ref, new, ri, ci, np.full(len(ri), power))
        b = frozen_edge_weights(ref, new[:1], ri, ci, np.full(len(ri), power))
        c = frozen_edge_weights(ref, new[::-1], ri, ci, np.full(len(ri), power))
        np.testing.assert_allclose(a[0], b[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a[0], c[-1], atol=1e-12, rtol=0)
    return records


def score_component(*, reference, original, validation, reference_positions,
                    rows, columns, masks, power, module, component, n_features,
                    original_ids, validation_ids, edge_batch=2048,
                    possible_edges=None, node_conventions=None, source_n_features=None):
    if edge_batch < 1:
        raise ValueError("edge_batch must be positive")
    targets = np.vstack([original, validation])
    positions = np.r_[reference_positions, np.full(len(validation), -1)]
    rules = list(masks)
    chosen = np.vstack([masks[r] for r in rules])
    if chosen.shape[1] != len(rows):
        raise ValueError("Masks and positional edge arrays differ")
    retained = chosen.sum(axis=1)
    acc = np.zeros((len(targets), len(rules), 5))
    if len(rows):
        mean, variance, cross = correlation_edges(reference, rows, columns)
        for start in range(0, len(rows), edge_batch):
            sl = slice(start, start+edge_batch)
            w = frozen_edge_weights(reference, targets, rows[sl], columns[sl],
                np.full(len(rows[sl]), power), reference_positions=positions,
                reference_statistics=(mean, variance, cross[sl]))
            selection = chosen[:, sl].astype(float).T
            acc[:, :, 0] += np.maximum(w, 0) @ selection
            acc[:, :, 1] += np.maximum(-w, 0) @ selection
            acc[:, :, 2] += (w > 0).astype(float) @ selection
            acc[:, :, 3] += (w < 0).astype(float) @ selection
            acc[:, :, 4] += (w*w) @ selection
    checks = formula_checks(reference, targets, rows, columns, power, positions, len(original))
    original_parts, validation_parts, anchors = [], [], []
    for k, rule in enumerate(rules):
        pos, neg, pc, nc, ss = acc[:, k].T
        count = int(retained[k])
        possible=int(possible_edges[rule]) if possible_edges is not None else len(rows)
        effective=int(n_features[rule]) if isinstance(n_features,dict) else n_features
        if count>possible:
            raise ValueError("Retained edges exceed this analysis's possible edges")
        frame = pd.DataFrame(dict(sample_id=[*original_ids, *validation_ids], module=int(module),
            component=component, differential_edge_rule=rule, n_features=effective,
            n_possible_edges=possible, n_retained_edges=count,
            n_positive_edges=np.rint(pc).astype(int), n_negative_edges=np.rint(nc).astype(int),
            n_zero_edges=count-np.rint(pc+nc).astype(int), n_pruned_edges=possible-count,
            positive_weight_sum=pos, negative_weight_magnitude=neg, signed_weight_sum=pos-neg,
            absolute_weight_sum=pos+neg, weight_squared_sum=ss))
        if node_conventions is not None:
            frame["node_convention"]=node_conventions[rule]
            frame["n_source_assignment_rows"]=int(source_n_features)
            frame["cross_filter_node_convention_warning"]=int(source_n_features)!=effective
            # Filtered rows also need the warning when their module's all-edge
            # comparator uses a different node convention.
            if isinstance(n_features,dict) and len(set(n_features.values()))>1:
                frame["cross_filter_node_convention_warning"]=True
        if not possible:
            frame.loc[:, ["positive_weight_sum", "negative_weight_magnitude", "signed_weight_sum",
                          "absolute_weight_sum", "weight_squared_sum"]] = np.nan
        if not (frame.n_positive_edges+frame.n_negative_edges+frame.n_zero_edges == count).all():
            raise ValueError("Sign-count identity failed")
        for normalization in ("standard_pruned", "retained_edge"):
            for family in FEATURES:
                raw = feature_values(frame, family, normalization)[:len(original)]
                fitted = FrozenRINT.fit(raw)
                anchors.append(dict(module=int(module), component=component,
                    differential_edge_rule=rule, score_normalization=normalization, metric_family=family,
                    x=fitted.x.tolist(), z=fitted.z.tolist(), asinh_scale=robust_asinh_scale(raw),
                    training_n=int(np.isfinite(raw).sum())))
        original_parts.append(frame.iloc[:len(original)].copy())
        validation_parts.append(frame.iloc[len(original):].copy())
    return (pd.concat(original_parts, ignore_index=True),
            pd.concat(validation_parts, ignore_index=True), pd.DataFrame(anchors), checks)
