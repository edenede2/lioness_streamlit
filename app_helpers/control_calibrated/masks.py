"""Verified frozen discovery-mask mappings; never select edges from targets."""

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from .registry import ROOT
from .inputs import REPO, gene, tissue, atomic_json, atomic_parquet, sha
from .adapter import module_filename


def key(rule, scope="global", threshold=0.05):
    return (
        rule
        if rule.startswith("ad_control_discovery_effect__")
        else f"{rule}__{scope}__q{threshold:.2f}"
    )


def mask_artifact_path(
    root, beta, family, cohort, module_key, rule, scope="global", threshold=0.05
):
    return (
        Path(root)
        / "masks"
        / key(rule, scope, threshold)
        / "scores"
        / f"beta{beta}"
        / family
        / cohort
        / module_filename(module_key)
    )


def options(family, module, tissue=None, root=ROOT):
    result = {"all": dict(rule="all", label="All physical edges (unmasked baseline)")}
    path = Path(root) / "masks/manifest.json"
    if not path.exists():
        return result
    m = json.loads(path.read_text())
    module_key = f"{family}::{tissue}::M{module}" if tissue else f"{family}::M{module}"
    for k, v in m.get("modules", {}).get(module_key, {}).items():
        if v.get("status") == "complete":
            result[k] = dict(m["specs"][family][k])
    return result


def family_options(family, root=ROOT):
    result = {"all": dict(rule="all", label="All physical edges (unmasked)")}
    root = Path(root)
    path = root / "masks/manifest.json"
    if not path.exists():
        return result
    manifest = json.loads(path.read_text())
    catalog = json.loads((root / "catalog.json").read_text())
    modules = [j["module_key"] for j in catalog["modules"] if j["family"] == family]
    for k, spec in manifest.get("specs", {}).get(family, {}).items():
        if modules and all(
            manifest.get("modules", {}).get(m, {}).get(k, {}).get("status")
            == "complete"
            for m in modules
        ):
            result[k] = spec
    return result


def source_path(family, module):
    if family in ("control_derived", "full_cohort"):
        return (
            REPO
            / f"out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks/{family}/checkpoints/lioness__control_anchored/M{module}__edge_statistics.parquet"
        )
    base = (
        "out/lioness_app_expansion/20261012_muscle_effect_size_edge_filtering"
        if family == "muscle_control_l4"
        else "out/muscle_all_donors_l4/20261013_nonprediction/effect_edges"
    )
    return (
        REPO
        / base
        / family
        / "checkpoints/lioness__control_anchored"
        / f"M{module}__edge_statistics.parquet"
    )


def specs_for(family):
    from app_helpers import data as D

    # The canonical legacy policy is a production research module, not an
    # audit script. Copy its exact tie-breaking and zero-direction semantics.
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from models.differential_edges import (
        effect_mask_specs,
        effect_rule_key,
        EffectMaskSpec,
    )

    manifest = D.load_effect_size_manifest(family)
    specs = effect_mask_specs()
    if family.startswith("muscle"):
        specs = (EffectMaskSpec("hedges_g", "fixed", 0.4),)
    out = {}
    for spec in specs:
        for direction in ("either", "ad_higher", "control_higher"):
            rule = effect_rule_key(spec, direction)
            out[key(rule)] = dict(
                rule=rule,
                label=D.effect_rule_label(rule),
                statistic=spec.statistic,
                mode=spec.cutoff_mode,
                value=spec.cutoff_value,
                direction=direction,
                provenance="Frozen historical discovery AD-Control statistic; exploratory reuse; targets never relearn the mask",
            )
    for scope in ("global", "per_module"):
        if not D.differential_data_available(family):
            continue
        for threshold in (0.05, 0.10):
            rule = "ad_control_discovery_fdr05"
            out[key(rule, scope, threshold)] = dict(
                rule=rule,
                label=f"Historical FDR < {threshold:g}, "
                + ("across partition" if scope == "global" else "within module"),
                fdr_scope=scope,
                threshold=threshold,
                provenance="Frozen discovery Welch edge test; legacy motor global aliases require truthful within-module compatibility label",
            )
    return out


def resolve_module(job, info, specs):
    from models.differential_edges import effect_size_mask, EffectMaskSpec

    path = source_path(job["family"], job["module"])
    names = [tuple(n) for n in info["nodes"]]
    lookup = {n: i for i, n in enumerate(names)}
    n_nodes = len(names)
    if not path.exists():
        return (
            None,
            {},
            dict(
                status="unavailable",
                reason="Missing frozen legacy edge-statistics input: " + str(path),
            ),
        )
    f = pd.read_parquet(path)
    r = np.array(
        [lookup.get((tissue(t), gene(g)), -1) for t, g in zip(f.tissue_a, f.gene_a)]
    )
    c = np.array(
        [lookup.get((tissue(t), gene(g)), -1) for t, g in zip(f.tissue_b, f.gene_b)]
    )
    if (r < 0).any() or (c < 0).any():
        return (
            None,
            {},
            dict(
                status="unavailable",
                reason="Legacy endpoint identity absent from verified expression universe",
            ),
        )
    code = np.minimum(r, c) * n_nodes + np.maximum(r, c)
    nodes = np.asarray(job["node_indices"])
    a, b = np.triu_indices(len(nodes), 1)
    expected = np.sort(
        np.unique(
            np.minimum(nodes[a], nodes[b]) * n_nodes + np.maximum(nodes[a], nodes[b])
        )
    )
    valid = r != c
    physical = np.unique(code[valid])
    if not np.array_equal(physical, expected):
        return (
            None,
            {},
            dict(
                status="unavailable",
                reason="Frozen legacy endpoints do not cover the exact deduplicated module physical-edge universe",
            ),
        )
    unique, first, inverse = np.unique(
        code[valid], return_index=True, return_inverse=True
    )
    mapped = {}
    reasons = {}
    for k, spec in specs.items():
        if spec["rule"].startswith("ad_control_discovery_effect__"):
            selected = effect_size_mask(
                f["discovery_" + spec["statistic"]].to_numpy(float),
                f.edge_index.to_numpy(),
                EffectMaskSpec(spec["statistic"], spec["mode"], spec["value"]),
                direction=spec["direction"],
            )
        else:
            col = "discovery_fdr_" + spec["fdr_scope"]
            if col not in f:
                reasons[k] = dict(
                    status="unavailable", reason="Frozen source lacks " + col
                )
                continue
            selected = (
                pd.to_numeric(f[col], errors="coerce").lt(spec["threshold"]).to_numpy()
            )
            if job["dataset"] == "motor" and spec["fdr_scope"] == "global":
                spec["label"] = (
                    f"Historical FDR < {spec['threshold']:g}, within-module legacy compatibility alias"
                )
        q = selected[valid]
        representative = q[first]
        if np.any(q != representative[inverse]):
            reasons[k] = dict(
                status="unavailable",
                reason="Duplicated legacy positional nodes disagree on membership of the same physical edge; mapping is ambiguous",
            )
            continue
        mapped[k] = representative
    return (
        unique,
        mapped,
        dict(
            status="mapped",
            source=str(path),
            source_sha256=sha(path),
            source_rows=len(f),
            deduplicated_physical_edges=len(unique),
            self_edges_dropped=int((~valid).sum()),
            unavailable=reasons,
        ),
    )


def prepare_mappings(root=ROOT):
    root = Path(root)
    catalog = json.loads((root / "catalog.json").read_text())
    families = [k for k, v in catalog["families"].items() if not v["regional"]]
    all_specs = {f: specs_for(f) for f in families}
    records = {}
    for j in catalog["modules"]:
        if j["family"] not in families:
            continue
        path = (
            root
            / "private/mask_mappings"
            / module_filename(j["module_key"]).replace(".parquet", ".npz")
        )
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            receipt = json.loads(meta.read_text())
            receipt["mapped_keys"] = [
                k for k in receipt["mapped_keys"] if k in all_specs[j["family"]]
            ]
            receipt["unavailable"] = {
                k: v
                for k, v in receipt.get("unavailable", {}).items()
                if k in all_specs[j["family"]]
            }
            atomic_json(meta, receipt)
            records[j["module_key"]] = receipt
            continue
        codes, masks, receipt = resolve_module(
            j, catalog["datasets"][j["input_variant"]], all_specs[j["family"]]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        if codes is not None:
            tmp = path.with_suffix(".partial.npz")
            np.savez_compressed(tmp, codes=codes, **masks)
            tmp.replace(path)
            receipt["mapped_keys"] = list(masks)
            receipt["array_sha256"] = sha(path)
        atomic_json(meta, receipt)
        records[j["module_key"]] = receipt
    atomic_json(
        root / "masks/mapping_manifest.json",
        dict(
            specs=all_specs,
            modules=records,
            policy="Endpoint identities only; deduplicate exact physical edges; reject conflicting duplicate membership; masks frozen from historical discovery, no target learning",
        ),
    )
    return records
