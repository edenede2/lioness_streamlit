"""Offline masked aggregation from cached physical-edge calibrations.

Weights are reconstructed in bounded blocks; masks never refit calibration.
All supported masks for a module share the same reconstructed block.
"""

from pathlib import Path
import json, time, resource
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .registry import ROOT, METRICS
from .core import canonical_hash, edge_attributes
from .inputs import atomic_json, atomic_parquet, sha
from . import adapter as A
from .masks import mask_artifact_path, prepare_mappings
from .inspection import reference_cache, prepare_reference_caches
from .generate import (
    pearson_block,
    weights,
    unpack,
    finish_values,
    r_transforms,
    NSTATS,
)

_STATE = {}


def public_mask_records(report):
    # File hashes stay in the per-module receipt. The app's selector manifest
    # remains compact even when tens of thousands of masked files are ready.
    return {
        k: {name: value for name, value in record.items() if name != "files"}
        for k, record in report.get("masks", {}).items()
    }


def channels(w, cal):
    """Edge×donor×sufficient-contribution channels for sparse mask sums."""
    a, z = edge_attributes(w, cal)
    good = np.isfinite(a) & cal["valid"]
    zg = np.isfinite(z) & cal["signed_valid"]
    positive = np.where(good, np.maximum(a, 0), 0)
    negative = np.where(good, np.maximum(-a, 0), 0)
    out = np.zeros((w.shape[1], len(w), NSTATS))
    out[:, :, 0] = positive.T
    out[:, :, 1] = negative.T
    out[:, :, 2] = np.where(good, abs(w), 0).T
    out[:, :, 3] = np.where(good, abs(w) - cal["nu"], 0).T
    out[:, :, 4] = np.where(zg, z * z, 0).T
    out[:, :, 5] = np.where(zg, abs(z), 0).T
    zero = w == 0
    near = (abs(w) <= 0.01 * cal["nu"]) & ~zero
    uncertain = ~cal["mean_sign_supported"] & ~zero & ~near
    rev = (
        cal["mean_sign_supported"] & (np.sign(w) != np.sign(cal["mu"])) & ~zero & ~near
    )
    pres = ~zero & ~near & ~uncertain & ~rev
    for k, mask in enumerate((zero, near, uncertain, pres, rev)):
        out[:, :, 6 + k] = np.where(mask, positive, 0).T
        out[:, :, 11 + k] = np.where(mask, negative, 0).T
    out[:, :, 16] = good.T
    out[:, :, 17] = zg.T
    bound = np.divide(
        cal["nu_scaled"],
        cal["tau_scaled"],
        out=np.zeros(w.shape[1]),
        where=cal["valid"],
    )
    out[:, :, 18] = np.where(good, bound, 0).T
    return out


def module_task(job):
    started = time.perf_counter()
    root = _STATE["root"]
    catalog = _STATE["catalog"]
    variant = job["input_variant"]
    info = catalog["datasets"][variant]
    data = _STATE["variants"][variant]
    mapping = (
        root
        / "private/mask_mappings"
        / A.module_filename(job["module_key"]).replace(".parquet", ".npz")
    )
    receipt = json.loads(mapping.with_suffix(".json").read_text())
    done = root / "masks/module_completion" / mapping.with_suffix(".json").name
    signature = canonical_hash(
        dict(
            module=job,
            mask_array_sha256=receipt.get("array_sha256"),
            mask_keys=receipt.get("mapped_keys", []),
            calibration_sha256=data["cal_hash"],
            expression_sha256=info["expression_sha256"],
            presence_sha256=info["presence_sha256"],
            annotation="sign711010:711109/0.8/0.01nu",
            aggregation="G-L=S/G+L=A/common_physical_denominator.v2",
        )
    )
    if done.exists():
        result = json.loads(done.read_text())
        if result.get("signature", signature) != signature:
            raise ValueError("Masked aggregation checkpoint inputs changed")
        if all(
            (root / item["path"]).exists()
            and sha(root / item["path"]) == item["sha256"]
            for record in result["masks"].values()
            for item in record["files"]
        ):
            result["signature"] = signature
            atomic_json(done, result)
            return result
    if receipt["status"] != "mapped":
        return dict(
            module_key=job["module_key"],
            status="unavailable",
            reason=receipt["reason"],
            masks={},
        )
    if sha(mapping) != receipt["array_sha256"]:
        raise ValueError("Mask mapping content hash mismatch")
    mp = np.load(mapping)
    keys = receipt["mapped_keys"]
    codes = mp["codes"]
    r = codes // info["n_nodes"]
    c = codes % info["n_nodes"]
    tissues = np.array([n[0] for n in info["nodes"]])
    tr = tissues[r]
    tc = tissues[c]
    parts = []
    for t in info["tissues"]:
        keep = (tr == t) & (tc == t)
        if keep.any():
            parts.append(("TS_" + t, [t], keep))
    for ai, a in enumerate(info["tissues"]):
        for b in info["tissues"][ai + 1 :]:
            keep = ((tr == a) & (tc == b)) | ((tr == b) & (tc == a))
            if keep.any():
                parts.append(("CT_" + a + "__" + b, [a, b], keep))
    cells = [(k, p, mp[k] & keep) for k in keys for p, (_, _, keep) in enumerate(parts)]
    K = len(cells)
    N = info["n_donors"]
    possible = np.array([sel.sum() for _, _, sel in cells])
    eligible = {b: np.zeros(K, int) for b in (3, 1)}
    signed = {b: np.zeros(K, int) for b in (3, 1)}
    sums = {b: np.zeros((K, N, NSTATS)) for b in (3, 1)}
    ix = np.searchsorted(data["edges"], codes)
    if not np.array_equal(data["edges"][ix], codes):
        raise ValueError("Mask codes differ from frozen physical universe")
    checkpoint = root / "private/mask_checkpoints" / mapping.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.chmod(0o700)
    processed = np.zeros(len(codes), bool)
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved["signature"]) != signature:
                raise ValueError("Masked checkpoint signature changed")
            if "processed" in saved:
                processed = saved["processed"]
            else:
                processed[: int(saved["next"])] = True
            for b in (3, 1):
                sums[b] = saved[f"sums_{b}"]
                eligible[b] = saved[f"eligible_{b}"]
                signed[b] = saved[f"signed_{b}"]
    # Grouping by resolved tissue component avoids allocating mostly-zero
    # mask/component outputs, and never reconstructs donors lacking that tissue.
    # Old prefix checkpoints map exactly to this physical-edge processed mask.
    selected_any = np.logical_or.reduce([mp[k] for k in keys])
    processed[~selected_any] = True
    for pi, (_, required, part_edges) in enumerate(parts):
        positions = np.flatnonzero(part_edges & ~processed)
        donors = np.flatnonzero(
            data["presence"][:, [info["tissues"].index(t) for t in required]].all(1)
        )
        local_R = np.searchsorted(donors, data["R"])
        if not np.array_equal(donors[local_R], data["R"]):
            raise ValueError("Reference tissue absent during masked reconstruction")
        cell_ids = np.array([ki * len(parts) + pi for ki in range(len(keys))])
        for start in range(0, len(positions), 1024):
            pos = positions[start : start + 1024]
            rr = r[pos]
            cc = c[pos]
            r0, r1, valid, _ = pearson_block(
                data["X"],
                data["reference"],
                data["presence"][donors],
                data["reference"]["node_tissues"],
                rr,
                cc,
                local_R,
                donor_indices=donors,
            )
            membership = csr_matrix(np.stack([mp[k][pos] for k in keys]).astype(float))
            for beta in (3, 1):
                cal = unpack(np.asarray(data["cal"][beta][ix[pos]]))
                w = weights(r0, r1, beta, local_R, len(data["R"]))
                w[~valid] = np.nan
                ch = channels(w, cal)
                increment = (membership @ ch.reshape(len(pos), -1)).reshape(
                    len(keys), len(donors), NSTATS
                )
                sums[beta][cell_ids[:, None], donors[None, :], :] += increment
                eligible[beta][cell_ids] += (
                    membership @ cal["valid"].astype(int)
                ).astype(int)
                signed[beta][cell_ids] += (
                    membership @ cal["signed_valid"].astype(int)
                ).astype(int)
            processed[pos] = True
            if (start // 1024) % 64 == 0 or start + 1024 >= len(positions):
                tmp = checkpoint.with_suffix(".partial.npz")
                np.savez(
                    tmp,
                    signature=np.array(signature),
                    next=np.array(processed.sum()),
                    processed=processed,
                    **{f"sums_{b}": sums[b] for b in (3, 1)},
                    **{f"eligible_{b}": eligible[b] for b in (3, 1)},
                    **{f"signed_{b}": signed[b] for b in (3, 1)},
                )
                tmp.replace(checkpoint)
    meta = data["meta"]
    R = data["R"]
    presence = data["presence"]
    target = meta.role.eq("target").to_numpy()
    incomplete = ~presence.all(1)
    scopes = {
        "complete_450": target & meta.in_original_analysis.to_numpy(),
        "incomplete_region_validation": target & incomplete,
        "maximum_component": target,
    }
    completed = {}
    coverage = []
    for ki, k in enumerate(keys):
        spec = _STATE["specs"][job["family"]][k]
        pathroot = root / "masks" / k
        ids = list(range(ki * len(parts), (ki + 1) * len(parts)))
        groups = [(parts[pi][0], [cid]) for pi, cid in enumerate(ids)] + [
            ("all", ids),
            (
                "TS",
                [cid for pi, cid in enumerate(ids) if parts[pi][0].startswith("TS")],
            ),
            (
                "CT",
                [cid for pi, cid in enumerate(ids) if parts[pi][0].startswith("CT")],
            ),
        ]
        universe_hash = canonical_hash(codes[mp[k]].tolist())
        files = []
        for beta in (3, 1):
            frames = {scope: [] for scope in scopes}
            fits = {}
            for component, cs in groups:
                ss = sums[beta][cs].sum(0) if cs else np.zeros((N, NSTATS))
                poss = int(possible[cs].sum())
                ne = int(eligible[beta][cs].sum())
                ns = int(signed[beta][cs].sum())
                required = {
                    t
                    for cid in cs
                    if possible[cid] > 0
                    for t in parts[cells[cid][1]][1]
                }
                available = (
                    presence[:, [info["tissues"].index(t) for t in required]].all(1)
                    if required
                    else np.ones(N, bool)
                )
                values = finish_values(ss, ne, ns, poss, available)
                frame = pd.DataFrame(values)
                transforms = {}
                for metric in METRICS:
                    aa, rr, fit = r_transforms(
                        frame[metric].to_numpy(), frame[metric].to_numpy()[R]
                    )
                    frame[metric + "_asinh"] = aa
                    frame[metric + "_rint"] = rr
                    transforms[metric] = fit
                selected = (
                    np.logical_or.reduce([cells[cid][2] for cid in cs])
                    if cs
                    else np.zeros(len(codes), bool)
                )
                endpoints = np.unique(np.r_[r[selected], c[selected]])
                node_presence = presence[
                    :, data["reference"]["node_tissues"][endpoints]
                ]
                frame["sparse_endpoint_cells_imputed"] = (
                    (~np.isfinite(data["X"][:, endpoints])) & node_presence
                ).sum(1)
                frame["absent_tissue_endpoint_cells"] = (~node_presence).sum(1)
                fits[component] = transforms
                frame["sample_id"] = meta.sample_id.to_numpy()
                frame["diagnosis_group"] = meta.diagnosis_group.to_numpy()
                frame["role"] = meta.role.to_numpy()
                frame["component"] = component
                frame["component_class"] = (
                    "CT"
                    if component.startswith("CT")
                    else "TS" if component.startswith("TS") else "all"
                )
                frame["module"] = job["module"]
                frame["module_key"] = job["module_key"]
                frame["family"] = job["family"]
                frame["beta"] = beta
                frame["allocation_id"] = info["allocation_id"]
                frame["mask"] = k
                frame["edge_universe_sha256"] = universe_hash
                frame["artifact_hash"] = canonical_hash(
                    dict(
                        module_key=job["module_key"],
                        beta=beta,
                        component=component,
                        mask=k,
                        universe=universe_hash,
                        allocation=info["allocation_id"],
                        calibration_sha256=data["cal_hash"][beta],
                    )
                )
                frame["mask_provenance"] = spec["provenance"]
                for scope, keep in scopes.items():
                    q = frame.loc[keep].copy()
                    q["cohort_scope"] = scope
                    frames[scope].append(q)
                    coverage.append(
                        dict(
                            module_key=job["module_key"],
                            family=job["family"],
                            module=job["module"],
                            component=component,
                            beta=beta,
                            cohort_scope=scope,
                            mask=k,
                            possible_edges=poss,
                            eligible_edges=ne,
                            target_candidates=int(keep.sum()),
                            scored_targets=int(np.isfinite(q.gain).sum()),
                            structural=poss > 0,
                            failed=False,
                            unavailable_reasons=json.dumps(
                                q.unavailable_reason.value_counts().to_dict()
                            ),
                        )
                    )
            for scope, ps in frames.items():
                path = mask_artifact_path(
                    root,
                    beta,
                    job["family"],
                    scope,
                    job["module_key"],
                    spec["rule"],
                    spec.get("fdr_scope", "global"),
                    spec.get("threshold", 0.05),
                )
                atomic_parquet(path, pd.concat(ps, ignore_index=True))
                files.append(
                    dict(
                        path=str(path.relative_to(root)),
                        sha256=sha(path),
                        beta=beta,
                        cohort_scope=scope,
                    )
                )
            atomic_json(
                pathroot
                / "transforms"
                / f"beta{beta}"
                / job["family"]
                / A.module_filename(job["module_key"]).replace(".parquet", ".json"),
                dict(module_key=job["module_key"], components=fits),
            )
        completed[k] = dict(
            status="complete",
            physical_edges=int(mp[k].sum()),
            edge_universe_sha256=universe_hash,
            files=files,
        )
    report = dict(
        module_key=job["module_key"],
        status="complete",
        signature=signature,
        masks=completed,
        unavailable=receipt.get("unavailable", {}),
        seconds=time.perf_counter() - started,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_parquet(
        root / "masks/coverage" / A.module_filename(job["module_key"]),
        pd.DataFrame(coverage),
    )
    atomic_json(done, report)
    checkpoint.unlink(missing_ok=True)
    return report


def generate_masks(root=ROOT, workers=4, families=None):
    import multiprocessing as mp

    root = Path(root)
    catalog = A.catalog(root)
    manifest = json.loads((root / "manifest.json").read_text())
    selected = families or [
        k for k, v in catalog["families"].items() if not v["regional"]
    ]
    reports = manifest.get(
        "generation",
        (
            json.loads((root / "generation_progress.json").read_text())
            if (root / "generation_progress.json").exists()
            else []
        ),
    )
    required_variants = {
        j["input_variant"] for j in catalog["modules"] if j["family"] in selected
    }
    if not required_variants <= set(v["variant"] for v in reports):
        raise ValueError(
            "Complete calibrations for every requested variant are required before masked aggregation"
        )
    started = time.perf_counter()
    prepare_reference_caches(root)
    mapping = prepare_mappings(root)
    specs = json.loads((root / "masks/mapping_manifest.json").read_text())["specs"]
    variants = {}
    for variant in sorted(required_variants):
        info = catalog["datasets"][variant]
        p = root / "private" / f"{variant}_reference.npz"
        reference = reference_cache(str(root), variant, p.stat().st_mtime_ns)
        work = root / "private" / variant
        report = next(v for v in reports if v["variant"] == variant)
        cal = {}
        for suffix, expected in [
            ("expression", info["expression_sha256"]),
            ("presence", info["presence_sha256"]),
        ]:
            if sha(root / "private" / f"{variant}_{suffix}.npy") != expected:
                raise ValueError("Frozen expression/availability content hash changed")
        for beta in (3, 1):
            path = work / f"calibration_beta{beta}.npy"
            if sha(path) != report["parameter_files"][str(beta)]["sha256"]:
                raise ValueError("Calibration content hash changed")
            cal[beta] = np.load(path, mmap_mode="r")
        variants[variant] = dict(
            X=np.load(root / "private" / f"{variant}_expression.npy", mmap_mode="r"),
            presence=np.load(root / "private" / f"{variant}_presence.npy"),
            reference=reference,
            R=np.asarray(info["role_indices"]["R"]),
            meta=pd.read_parquet(root / info["metadata_file"]),
            edges=np.load(work / "edges.npy", mmap_mode="r"),
            cal=cal,
            cal_hash={b: report["parameter_files"][str(b)]["sha256"] for b in (3, 1)},
        )
    _STATE.update(root=root, catalog=catalog, specs=specs, variants=variants)
    jobs = [j for j in catalog["modules"] if j["family"] in selected]
    reports = []
    public = dict(
        status="running", specs=specs, modules={}, failures={}, unavailable={}
    )
    jobs.sort(key=lambda j: j["physical_nodes"], reverse=True)
    for p in (root / "masks/module_completion").glob("*.json"):
        record = json.loads(p.read_text())
        public["modules"][record["module_key"]] = public_mask_records(record)
        public["unavailable"][record["module_key"]] = record.get("unavailable", {})
    with mp.get_context("fork").Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(module_task, jobs, chunksize=1)):
            reports.append(result)
            public["modules"][result["module_key"]] = public_mask_records(result)
            public["unavailable"][result["module_key"]] = result.get("unavailable", {})
            if result["status"] == "unavailable":
                public["unavailable"][result["module_key"]] = {
                    "all": result.get("reason", "Unavailable source")
                }
            atomic_json(root / "masks/manifest.json", public)
            if i % 10 == 0 or i + 1 == len(jobs):
                print(
                    json.dumps(
                        dict(
                            stage="masked_aggregation",
                            completed=i + 1,
                            total=len(jobs),
                            seconds=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
    expected = {
        j["module_key"]
        for j in catalog["modules"]
        if not catalog["families"][j["family"]]["regional"]
    }
    done = set(public["modules"])
    public.update(
        status="complete" if done == expected else "partial",
        completed_modules=len(done),
        planned_modules=len(expected),
        failed_modules=0,
        seconds=time.perf_counter() - started,
        workers=workers,
        peak_worker_RSS_MiB=max(r.get("peak_RSS_MiB", 0) for r in reports),
    )
    atomic_json(root / "masks/manifest.json", public)
    return public
