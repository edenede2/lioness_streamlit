"""Deduplicated physical-edge calibration/scoring with transactional resume.

Only C×edge and donor×edge batches exist in memory. Persistent arrays contain
edge calibration parameters or donor×component sums, never a catalog-wide
donor×edge cube. A write-ahead backup makes component accumulation resumable.
"""

from pathlib import Path
from dataclasses import replace
import hashlib, json, time, resource, os
import numpy as np
import pandas as pd
from .inputs import atomic_json, atomic_parquet, sha
from .core import Settings, fit_reference, calibrate, canonical_hash
from .registry import SCHEMA_VERSION, METRICS

PARAMS = (
    "scale",
    "mu_scaled",
    "nu_scaled",
    "sd_scaled",
    "tau_scaled",
    "influence_share",
    "mean_sign_interval_low",
    "mean_sign_interval_high",
    "sign_agreement",
    "n",
    "valid",
    "signed_valid",
    "mean_sign_supported",
)
NSTATS = 19
BETAS = (3, 1)
_WORKER = {}


def compute_block(bounds):
    """Read-only worker; the parent alone commits checkpointed accumulations."""
    start, end = bounds
    g = _WORKER
    block = np.asarray(g["edges"][start:end])
    rows = block // g["n_nodes"]
    cols = block % g["n_nodes"]
    active = []
    owners = []
    for cell, indices in zip(g["cells"], g["postings"]):
        lo, hi = np.searchsorted(indices, [start, end])
        if hi > lo:
            active.append(cell["id"])
            owners.append(np.asarray(indices[lo:hi]) - start)
    r0, r1, valid, _ = pearson_block(
        g["X"], g["reference"], g["presence"], g["node_tissues"], rows, cols, g["R"]
    )
    results = {}
    for b in BETAS:
        w = weights(r0, r1, b, g["R"], len(g["R"]))
        w[~valid] = np.nan
        cal = fast_calibration(w[g["C"]], g["settings"], g["bootstrap"])
        increments = (
            np.stack([reduce_edges(w, cal, ix) for ix in owners])
            if owners
            else np.zeros((0, len(w), NSTATS))
        )
        results[b] = (pack(cal), increments)
    return (
        start,
        end,
        np.asarray(active, int),
        results,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


def block_results(bounds, workers):
    """At most 2×workers pending blocks; no unbounded executor result queue."""
    if workers == 1:
        for pair in bounds:
            yield compute_block(pair)
        return
    import multiprocessing as mp
    from collections import deque

    with mp.get_context("fork").Pool(workers) as pool:
        pending = deque()
        iterator = iter(bounds)
        for _ in range(2 * workers):
            try:
                pending.append(pool.apply_async(compute_block, (next(iterator),)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().get()
            try:
                pending.append(pool.apply_async(compute_block, (next(iterator),)))
            except StopIteration:
                pass
            yield result


def pack(cal):
    return np.column_stack([cal[k] for k in PARAMS])


def unpack(array):
    d = {k: array[:, i] for i, k in enumerate(PARAMS)}
    for k in ("valid", "signed_valid", "mean_sign_supported"):
        d[k] = d[k].astype(bool)
    d.update(
        mu=d["scale"] * d["mu_scaled"],
        nu=d["scale"] * d["nu_scaled"],
        sigma=d["scale"] * d["sd_scaled"],
        tau=d["scale"] * d["tau_scaled"],
    )
    return d


def fit_engine_fingerprint(path):
    """Hash only core functions actually used to produce these edge fits.

    Pure aggregation API repairs do not change frozen per-edge calibration.
    The original producer snapshot remains recorded and content-verified.
    """
    import ast

    parsed = ast.parse(Path(path).read_text())
    names = {"Settings", "fit_reference", "calibrate", "canonical_hash"}
    nodes = [
        n
        for n in parsed.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    return canonical_hash([ast.dump(n, include_attributes=False) for n in nodes])


def fast_calibration(w, settings, bootstrap):
    cal = calibrate(w, replace(settings, sign_seeds=()))
    scaled = np.divide(w, cal["scale"], out=np.zeros_like(w), where=cal["scale"] > 0)
    means = bootstrap @ scaled
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    cal["mean_sign_interval_low"] = low * cal["scale"]
    cal["mean_sign_interval_high"] = high * cal["scale"]
    cal["mean_sign_supported"] = (
        (low * high > 0)
        & (cal["sign_agreement"] >= settings.sign_agreement)
        & (cal["n"] == settings.expected_C)
    )
    return cal


def pearson_block(
    X, ref, presence, node_tissues, rows, cols, Rindices, donor_indices=None
):
    n = len(ref["values"])
    mean = ref["mean"]
    xx = np.asarray(
        X[:, rows] if donor_indices is None else X[np.ix_(donor_indices, rows)]
    )
    yy = np.asarray(
        X[:, cols] if donor_indices is None else X[np.ix_(donor_indices, cols)]
    )
    present = presence[:, node_tissues[rows]] & presence[:, node_tissues[cols]]
    imputed = ((~np.isfinite(xx)) & presence[:, node_tissues[rows]]).astype(
        np.uint8
    ) + ((~np.isfinite(yy)) & presence[:, node_tissues[cols]]).astype(np.uint8)
    xx = np.where(np.isfinite(xx), xx, mean[rows])
    yy = np.where(np.isfinite(yy), yy, mean[cols])
    dx = xx - mean[rows]
    dy = yy - mean[cols]
    cross = (
        (ref["values"][:, rows] - mean[rows]) * (ref["values"][:, cols] - mean[cols])
    ).sum(0)
    ssx = ref["ss"][rows]
    ssy = ref["ss"][cols]
    valid = ref["endpoint_valid"][rows] & ref["endpoint_valid"][cols]
    r0 = np.divide(cross, np.sqrt(ssx * ssy), out=np.zeros(len(rows)), where=valid)
    factor = np.full(len(xx), n / (n + 1))
    factor[Rindices] = -n / (n - 1)
    cov = cross + factor[:, None] * dx * dy
    vx = np.maximum(ssx + factor[:, None] * dx * dx, 0)
    vy = np.maximum(ssy + factor[:, None] * dy * dy, 0)
    den = np.sqrt(vx * vy)
    r1 = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    np.clip(r0, -1, 1, out=r0)
    np.clip(r1, -1, 1, out=r1)
    return r0, r1, present & valid, imputed


def weights(r0, r1, beta, Rindices, n):
    a = r0 if beta == 1 else r0 * r0 * r0
    b = r1 if beta == 1 else r1 * r1 * r1
    w = (n + 1) * b - n * a
    w[Rindices] = n * a - (n - 1) * b[Rindices]
    return w


def reduce_edges(w, cal, ix):
    v = np.divide(
        w[:, ix],
        cal["scale"][ix],
        out=np.zeros((len(w), len(ix))),
        where=cal["scale"][ix] > 0,
    )
    m = cal["valid"][ix]
    sm = cal["signed_valid"][ix]
    a = np.divide(
        abs(v) - cal["nu_scaled"][ix],
        cal["tau_scaled"][ix],
        out=np.full_like(v, np.nan),
        where=m,
    )
    z = np.divide(
        v - cal["mu_scaled"][ix],
        cal["sd_scaled"][ix],
        out=np.full_like(v, np.nan),
        where=sm,
    )
    good = np.isfinite(a) & m
    zg = np.isfinite(z) & sm
    pos = np.where(good, np.maximum(a, 0), 0)
    neg = np.where(good, np.maximum(-a, 0), 0)
    ww = w[:, ix]
    out = np.zeros((len(w), NSTATS))
    out[:, 0] = pos.sum(1)
    out[:, 1] = neg.sum(1)
    out[:, 2] = np.where(good, abs(ww), 0).sum(1)
    out[:, 3] = np.where(good, abs(ww) - cal["nu"][ix], 0).sum(1)
    out[:, 4] = np.where(zg, z * z, 0).sum(1)
    out[:, 5] = np.where(zg, abs(z), 0).sum(1)
    zero = ww == 0
    near = (abs(ww) <= 0.01 * cal["nu"][ix]) & ~zero
    uncertain = ~cal["mean_sign_supported"][ix] & ~zero & ~near
    rev = (
        cal["mean_sign_supported"][ix]
        & (np.sign(ww) != np.sign(cal["mu_scaled"][ix]))
        & ~zero
        & ~near
    )
    pres = ~zero & ~near & ~uncertain & ~rev
    for k, mask in enumerate((zero, near, uncertain, pres, rev)):
        out[:, 6 + k] = np.where(mask, pos, 0).sum(1)
        out[:, 11 + k] = np.where(mask, neg, 0).sum(1)
    out[:, 16] = good.sum(1)
    out[:, 17] = zg.sum(1)
    bound = np.divide(
        cal["nu_scaled"][ix], cal["tau_scaled"][ix], out=np.zeros(len(ix)), where=m
    )
    out[:, 18] = np.where(good, bound, 0).sum(1)
    return out


def build_edges(modules, n_nodes, work, tissue_order=None):
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    local = []
    for j in modules:
        nodes = np.array(j["node_indices"], np.int64)
        r, c = np.triu_indices(len(nodes), 1)
        ids = np.unique(
            np.minimum(nodes[r], nodes[c]) * n_nodes + np.maximum(nodes[r], nodes[c])
        )
        parts.append(ids)
        local.append(ids)
    edges = np.unique(np.concatenate(parts))
    np.save(work / "edges.npy", edges)
    cells = []
    module_cells = {}
    # Resolved components partition the physical edge set, so unions can sum
    # sufficient contributions/counts without duplicating any physical edge.
    for ji, (j, ids) in enumerate(zip(modules, local)):
        indices = np.searchsorted(edges, ids)
        node_t = np.full(n_nodes, "", dtype="<U8")
        for ni, node in zip(j["node_indices"], j["nodes"]):
            node_t[ni] = node[0]
        r = ids // n_nodes
        c = ids % n_nodes
        tr = node_t[r]
        tc = node_t[c]
        present = {n[0] for n in j["nodes"]}
        ts = [t for t in (tissue_order or sorted(present)) if t in present]
        components = [("TS_" + t, (t,)) for t in ts] + [
            ("CT_" + a + "__" + b, (a, b))
            for ai, a in enumerate(ts)
            for b in ts[ai + 1 :]
        ]
        module_cells[j["module_key"]] = []
        for component, required in components:
            keep = (
                (tr == required[0]) & (tc == required[0])
                if len(required) == 1
                else ((tr == required[0]) & (tc == required[1]))
                | ((tr == required[1]) & (tc == required[0]))
            )
            chosen = indices[keep]
            if not len(chosen):
                continue
            cell = dict(
                id=len(cells),
                module_index=ji,
                module_key=j["module_key"],
                component=component,
                required_tissues=list(required),
                possible_edges=len(chosen),
            )
            np.save(work / f"cell_{len(cells)}.npy", chosen)
            cells.append(cell)
            module_cells[j["module_key"]].append(cell["id"])
    atomic_json(work / "cells.json", dict(cells=cells, module_cells=module_cells))
    return edges, cells, module_cells


def finish_values(sums, eligible, signed, possible, available):
    n = len(sums)
    valid = (eligible > 0) & (sums[:, 16] == eligible) & available
    sv = (signed > 0) & (sums[:, 17] == signed) & available
    divide = lambda v: np.divide(v, eligible, out=np.full(n, np.nan), where=valid)
    d = dict(
        gain=divide(sums[:, 0]),
        loss=divide(sums[:, 1]),
        mean_abs_w=divide(sums[:, 2]),
        mean_delta_abs=divide(sums[:, 3]),
        signed_distance=np.divide(sums[:, 4], signed, out=np.full(n, np.nan), where=sv),
        signed_abs_z=np.divide(sums[:, 5], signed, out=np.full(n, np.nan), where=sv),
    )
    d["net_change"] = d["gain"] - d["loss"]
    d["magnitude_atypicality"] = d["gain"] + d["loss"]
    d["eligible_edges"] = np.full(n, eligible)
    d["possible_edges"] = np.full(n, possible)
    d["scored_edges"] = sums[:, 16].astype(int)
    d["signed_eligible_edges"] = np.full(n, signed)
    d["loss_bound"] = divide(sums[:, 18])
    reason = np.full(n, "ok", dtype="<U56")
    reason[~available] = "missing_required_tissue"
    reason[available & (possible == 0)] = "no_physical_edges"
    reason[available & (possible > 0) & (eligible == 0)] = "no_eligible_magnitude_edges"
    reason[available & (eligible > 0) & ~valid] = "incomplete_target_edge_coverage"
    d["unavailable_reason"] = reason
    for i, c in enumerate(("zero", "near_zero", "uncertain", "preserved", "reversed")):
        d["gain_" + c] = divide(sums[:, 6 + i])
        d["loss_" + c] = divide(sums[:, 11 + i])
    return d


def r_transforms(values, training):
    from scipy.stats import norm, rankdata

    x = np.asarray(training, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (
            np.full(len(values), np.nan),
            np.full(len(values), np.nan),
            dict(reason="No finite R training diagnostics"),
        )
    med = np.median(x)
    scales = (np.median(abs(x)), np.median(abs(x - med)), np.std(x, ddof=1))
    scale = next(
        (
            float(v)
            for v in scales
            if np.isfinite(v) and v > np.sqrt(np.finfo(float).eps)
        ),
        1.0,
    )
    asinh = np.arcsinh(np.asarray(values) / scale)
    out = np.full(len(values), np.nan)
    fit = dict(
        fit_population="R-only LOO training diagnostics; not independent targets",
        asinh_scale=scale,
    )
    if len(np.unique(x)) > 1:
        ranks = norm.ppf((rankdata(x, method="average") - 0.5) / len(x))
        order = np.argsort(x, kind="stable")
        xx, first = np.unique(x[order], return_index=True)
        zz = ranks[order][first]
        good = np.isfinite(values)
        out[good] = np.interp(np.asarray(values)[good], xx, zz)
        fit.update(rint_x=xx.tolist(), rint_z=zz.tolist())
    else:
        fit["rint_reason"] = "Constant R training diagnostic scores"
    return asinh, out, fit


def materialize(
    root,
    variant,
    info,
    modules,
    cells,
    module_cells,
    accum,
    cal_maps,
    work,
    signature,
    source_root=None,
):
    source_root = Path(source_root or root)
    expression = np.load(
        source_root / "private" / f"{variant}_expression.npy", mmap_mode="r"
    )
    node_tissues = np.array([info["tissues"].index(n[0]) for n in info["nodes"]])
    meta = pd.read_parquet(source_root / info["metadata_file"])
    presence = np.load(source_root / "private" / f"{variant}_presence.npy")
    R = np.array(info["role_indices"]["R"])
    roles = meta.role.to_numpy()
    target = roles == "target"
    incomplete = (
        meta[[f"has_{t}" for t in info["tissues"]]]
        .sum(1)
        .lt(len(info["tissues"]))
        .to_numpy()
    )
    original = meta.in_original_analysis.to_numpy()
    scopes = {
        "complete_450": target & original,
        "incomplete_region_validation": target & incomplete,
        "maximum_component": target,
    }
    coverage = []
    files = []
    for beta in BETAS:
        cp = cal_maps[beta]
        cell_counts = {}
        for cell in cells:
            idx = np.load(work / f"cell_{cell['id']}.npy")
            cell_counts[cell["id"]] = (int(cp[idx, 10].sum()), int(cp[idx, 11].sum()))
        for j in modules:
            name = hashlib.sha256(j["module_key"].encode()).hexdigest()[:20]
            receipt_path = (
                root
                / "private/materialization"
                / variant
                / f"beta{beta}"
                / f"{name}.json"
            )
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["signature"] != signature:
                    raise ValueError("Materialization checkpoint protocol changed")
                if all(
                    (root / item["path"]).exists()
                    and sha(root / item["path"]) == item["sha256"]
                    for item in receipt["files"]
                ):
                    coverage.extend(receipt["coverage"])
                    files.extend(receipt["files"])
                    continue
            coverage_start = len(coverage)
            files_start = len(files)
            mid = j["module_key"]
            ids = module_cells[mid]
            resolved = [cells[k] for k in ids]
            unions = [
                ("all", ids),
                ("TS", [k for k in ids if cells[k]["component"].startswith("TS_")]),
                ("CT", [k for k in ids if cells[k]["component"].startswith("CT_")]),
            ]
            groups = [(c["component"], [c["id"]]) for c in resolved] + unions
            frames = {scope: [] for scope in scopes}
            fits = {}
            for component, cs in groups:
                sums = (
                    np.asarray(accum[beta][cs]).sum(0)
                    if cs
                    else np.zeros((len(meta), NSTATS))
                )
                possible = sum(cells[k]["possible_edges"] for k in cs)
                eligible = sum(cell_counts[k][0] for k in cs)
                signed = sum(cell_counts[k][1] for k in cs)
                required = {t for k in cs for t in cells[k]["required_tissues"]}
                available = (
                    presence[:, [info["tissues"].index(t) for t in required]].all(1)
                    if required
                    else np.ones(len(meta), bool)
                )
                vals = finish_values(sums, eligible, signed, possible, available)
                df = pd.DataFrame(vals)
                transforms = {}
                for metric in METRICS:
                    aa, rr, fit = r_transforms(
                        df[metric].to_numpy(), df[metric].to_numpy()[R]
                    )
                    df[metric + "_asinh"] = aa
                    df[metric + "_rint"] = rr
                    transforms[metric] = fit
                indices = (
                    np.sort(
                        np.concatenate([np.load(work / f"cell_{k}.npy") for k in cs])
                    )
                    if cs
                    else np.array([], np.int64)
                )
                global_edges = np.load(work / "edges.npy", mmap_mode="r")
                universe_hash = hashlib.sha256(
                    np.asarray(global_edges[indices], dtype="<i8").tobytes()
                ).hexdigest()
                codes = np.asarray(global_edges[indices])
                endpoints = np.unique(
                    np.r_[codes // info["n_nodes"], codes % info["n_nodes"]]
                )
                node_presence = presence[:, node_tissues[endpoints]]
                df["sparse_endpoint_cells_imputed"] = (
                    (~np.isfinite(expression[:, endpoints])) & node_presence
                ).sum(1)
                df["absent_tissue_endpoint_cells"] = (~node_presence).sum(1)
                fits[component] = transforms
                df["sample_id"] = meta.sample_id.to_numpy()
                df["role"] = roles
                df["component"] = component
                df["component_class"] = (
                    "CT"
                    if component.startswith("CT")
                    else "TS" if component.startswith("TS") else "all"
                )
                df["module"] = j["module"]
                df["module_key"] = mid
                df["family"] = j["family"]
                df["beta"] = beta
                df["allocation_id"] = info["allocation_id"]
                df["edge_universe_sha256"] = universe_hash
                df["artifact_hash"] = canonical_hash(
                    dict(
                        signature=signature,
                        module_key=mid,
                        beta=beta,
                        component=component,
                        edge_universe=universe_hash,
                    )
                )
                df["diagnosis_group"] = meta.diagnosis_group.to_numpy()
                for scope, keep in scopes.items():
                    q = df.loc[keep].copy()
                    q["cohort_scope"] = scope
                    frames[scope].append(q)
                    reason_counts = q.unavailable_reason.value_counts().to_dict()
                    coverage.append(
                        dict(
                            family=j["family"],
                            module=j["module"],
                            module_key=mid,
                            component=component,
                            cohort_scope=scope,
                            beta=beta,
                            possible_edges=possible,
                            eligible_edges=eligible,
                            target_candidates=int(keep.sum()),
                            scored_targets=int(np.isfinite(q.gain).sum()),
                            structural=possible > 0,
                            status=(
                                "structural_unavailable"
                                if possible == 0
                                else (
                                    "eligible_unavailable"
                                    if eligible == 0
                                    else (
                                        "scored"
                                        if np.isfinite(q.gain).any()
                                        else "donor_scope_unavailable"
                                    )
                                )
                            ),
                            unavailable_reasons=json.dumps(reason_counts),
                            failed=False,
                        )
                    )
            name = hashlib.sha256(mid.encode()).hexdigest()[:20]
            for scope, parts in frames.items():
                path = (
                    root
                    / "scores"
                    / f"beta{beta}"
                    / j["family"]
                    / scope
                    / f"{name}.parquet"
                )
                atomic_parquet(path, pd.concat(parts, ignore_index=True))
                files.append(
                    dict(
                        path=str(path.relative_to(root)),
                        sha256=sha(path),
                        module_key=mid,
                        beta=beta,
                        scope=scope,
                        rows=sum(len(x) for x in parts),
                    )
                )
            atomic_json(
                root / "transforms" / f"beta{beta}" / j["family"] / f"{name}.json",
                dict(
                    module_key=mid,
                    fit_population="R training diagnostics, no target fitting",
                    components=fits,
                ),
            )
            atomic_json(
                receipt_path,
                dict(
                    signature=signature,
                    coverage=coverage[coverage_start:],
                    files=files[files_start:],
                ),
            )
            if len(files) % 300 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="materialize",
                            variant=variant,
                            beta=beta,
                            completed_files=len(files),
                        )
                    ),
                    flush=True,
                )
    atomic_parquet(root / "coverage" / f"{variant}.parquet", pd.DataFrame(coverage))
    atomic_json(root / "file_indexes" / f"{variant}.json", files)
    return coverage


def generate(root, edge_batch=2048, benchmark=False, resume=True, workers=4):
    root = Path(root)
    catalog = json.loads((root / "catalog.json").read_text())
    out = root / "benchmark" if benchmark else root
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(
        out / "manifest.json",
        dict(
            schema_version=2,
            status="running",
            betas=list(BETAS),
            prediction_supported=False,
            allocation_seed=611010,
        ),
    )
    all_coverage = []
    variant_reports = []
    if benchmark:
        subset = {
            ("full_cohort", 188),
            ("control_derived", 1116),
            ("muscle_control_l4", 312),
        }
    for variant, info in catalog["datasets"].items():
        modules = [
            j
            for j in catalog["modules"]
            if j["input_variant"] == variant
            and (not benchmark or (j["family"], j["module"]) in subset)
        ]
        if not modules:
            continue
        tick = time.perf_counter()
        work = out / "private" / variant
        work.mkdir(parents=True, exist_ok=True)
        work.chmod(0o700)
        n_nodes = info["shape"][1]
        catalog_receipt = work / "catalog_receipt.json"
        catalog_hash = sha(root / "catalog.json")
        if (
            catalog_receipt.exists()
            and json.loads(catalog_receipt.read_text())["sha256"] != catalog_hash
        ):
            raise ValueError(
                "Checkpoint catalog membership changed; use a new versioned root"
            )
        atomic_json(catalog_receipt, dict(sha256=catalog_hash))
        if (
            sha(root / "private" / f"{variant}_expression.npy")
            != info["expression_sha256"]
            or sha(root / "private" / f"{variant}_presence.npy")
            != info["presence_sha256"]
        ):
            raise ValueError("Prepared expression/availability content hash changed")
        if (work / "cells.json").exists():
            edges = np.load(work / "edges.npy", mmap_mode="r")
            mapping = json.loads((work / "cells.json").read_text())
            cells = mapping["cells"]
            module_cells = mapping["module_cells"]
        else:
            edges, cells, module_cells = build_edges(
                modules, n_nodes, work, info["tissues"]
            )
        E = len(edges)
        N = info["n_donors"]
        R = np.array(info["role_indices"]["R"])
        C = np.array(info["role_indices"]["C"])
        settings = Settings(len(R), len(C))
        X = np.load(root / "private" / f"{variant}_expression.npy", mmap_mode="r")
        presence = np.load(root / "private" / f"{variant}_presence.npy")
        node_tissues = np.array([info["tissues"].index(n[0]) for n in info["nodes"]])
        reference = fit_reference(
            np.asarray(X[R]), settings, presence[R][:, node_tissues]
        )
        bootstrap = np.stack(
            [
                np.bincount(
                    np.random.default_rng(seed).integers(len(C), size=len(C)),
                    minlength=len(C),
                )
                / len(C)
                for seed in settings.sign_seeds
            ]
        )
        signature = canonical_hash(
            dict(
                version=SCHEMA_VERSION,
                batch=edge_batch,
                variant=variant,
                source=info["source_sha256"],
                availability=info["presence_sha256"],
                R=R.tolist(),
                C=C.tolist(),
                edge_universe=sha(work / "edges.npy"),
                guard=64,
                sign_agreement=0.8,
                near_zero=0.01,
                sign_seeds=list(settings.sign_seeds),
                core_sha256=sha(Path(__file__).with_name("core.py")),
            )
        )
        progress_path = work / "progress.json"
        state = (
            json.loads(progress_path.read_text())
            if progress_path.exists()
            else {"next": 0, "signature": signature}
        )
        if state["signature"] != signature:
            current_core = Path(__file__).with_name("core.py")
            snapshot = root / "private/code_snapshots/core.py"
            receipt_path = root / "private/code_snapshots/receipts.json"
            if (
                snapshot.exists()
                and receipt_path.exists()
                and sha(snapshot) == json.loads(receipt_path.read_text())["core.py"]
                and fit_engine_fingerprint(snapshot)
                == fit_engine_fingerprint(current_core)
            ):
                legacy = canonical_hash(
                    dict(
                        version=SCHEMA_VERSION,
                        batch=edge_batch,
                        variant=variant,
                        source=info["source_sha256"],
                        availability=info["presence_sha256"],
                        R=R.tolist(),
                        C=C.tolist(),
                        edge_universe=sha(work / "edges.npy"),
                        guard=64,
                        sign_agreement=0.8,
                        near_zero=0.01,
                        sign_seeds=list(settings.sign_seeds),
                        core_sha256=sha(snapshot),
                    )
                )
                if state["signature"] == legacy:
                    signature = legacy
                    atomic_json(
                        work / "core_api_compatibility.json",
                        dict(
                            producer_core_sha256=sha(snapshot),
                            current_core_sha256=sha(current_core),
                            fit_engine_fingerprint=fit_engine_fingerprint(current_core),
                            reason="Only aggregation API changed; reference fit and edge calibration functions are AST-identical to the content-verified producer snapshot",
                        ),
                    )
            if state["signature"] != signature:
                raise ValueError(
                    "Checkpoint inputs/configuration changed; use a new versioned root"
                )
        mode = "r+" if state["next"] else "w+"
        accum = {
            b: np.lib.format.open_memmap(
                work / f"sums_beta{b}.npy",
                mode=mode,
                dtype=np.float64,
                shape=(len(cells), N, NSTATS),
            )
            for b in BETAS
        }
        params = {
            b: np.lib.format.open_memmap(
                work / f"calibration_beta{b}.npy",
                mode=mode,
                dtype=np.float64,
                shape=(E, len(PARAMS)),
            )
            for b in BETAS
        }
        if not state["next"]:
            for a in accum.values():
                a[:] = 0
                a.flush()
        wal = work / "update_wal.npz"
        if wal.exists():
            z = np.load(wal)
            start = int(z["start"])
            active = z["active"]
            if state["next"] <= start:
                for b in BETAS:
                    accum[b][active] = z[f"beta{b}"]
                    accum[b].flush()
            wal.unlink()
        postings = [np.load(work / f"cell_{c['id']}.npy", mmap_mode="r") for c in cells]
        _WORKER.clear()
        _WORKER.update(
            edges=edges,
            n_nodes=n_nodes,
            cells=cells,
            postings=postings,
            X=X,
            reference=reference,
            presence=presence,
            node_tissues=node_tissues,
            R=R,
            C=C,
            settings=settings,
            bootstrap=bootstrap,
        )
        worker_peak = 0
        bounds = (
            (s, min(E, s + edge_batch)) for s in range(state["next"], E, edge_batch)
        )
        for start, end, active, results, worker_rss in block_results(bounds, workers):
            worker_peak = max(worker_peak, worker_rss)
            tmp = wal.with_suffix(".partial.npz")
            np.savez(
                tmp,
                start=np.array(start),
                active=active,
                **{f"beta{b}": np.asarray(accum[b][active]) for b in BETAS},
            )
            tmp.replace(wal)
            for b in BETAS:
                packed, increments = results[b]
                params[b][start:end] = packed
                accum[b][active] += increments
                params[b].flush()
                accum[b].flush()
            atomic_json(progress_path, dict(next=end, edges=E, signature=signature))
            wal.unlink()
            if start == 0 or end == E or (start // edge_batch) % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="physical_edges",
                            benchmark=benchmark,
                            variant=variant,
                            completed=end,
                            total=E,
                            seconds=time.perf_counter() - tick,
                            peak_RSS_MiB=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / 1024,
                        )
                    ),
                    flush=True,
                )
        cov = materialize(
            out,
            variant,
            info,
            modules,
            cells,
            module_cells,
            accum,
            params,
            work,
            signature,
            source_root=root,
        )
        # Benchmark references public metadata from the prepared source root.
        all_coverage.extend(cov)
        report = dict(
            variant=variant,
            physical_edges=E,
            resolved_cells=len(cells),
            modules=len(modules),
            seconds=time.perf_counter() - tick,
            signature=signature,
            workers=workers,
            max_worker_RSS_MiB=worker_peak,
            parameter_files={
                str(b): dict(
                    path=str((work / f"calibration_beta{b}.npy").relative_to(out)),
                    sha256=sha(work / f"calibration_beta{b}.npy"),
                )
                for b in BETAS
            },
        )
        variant_reports.append(report)
        atomic_json(out / "generation_progress.json", variant_reports)
    atomic_json(
        out / "manifest.json",
        dict(
            schema_version=SCHEMA_VERSION,
            status="benchmark_complete" if benchmark else "complete",
            allocation_seed=611010,
            betas=list(BETAS),
            default_beta=3,
            default_units="native standardized magnitude deviations",
            families=catalog["families"],
            datasets={
                k: {
                    q: v[q]
                    for q in ("dataset", "n_donors", "allocation_id", "metadata_file")
                }
                for k, v in catalog["datasets"].items()
            },
            generation=variant_reports,
            structural_cells=sum(x["structural"] for x in all_coverage),
            unavailable_cells=sum(x["status"] != "scored" for x in all_coverage),
            failed_cells=0,
            prediction_supported=False,
            annotation=dict(
                mean_sign_bootstrap="100 conditional C resamples; per-edge, not simultaneous confidence",
                agreement=0.8,
                near_zero_nu=0.01,
            ),
            resources=dict(
                seconds=time.perf_counter() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        ),
    )
