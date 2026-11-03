"""Bounded reconstruction from frozen cached fits for edge/contribution views."""

from pathlib import Path
from functools import lru_cache
import json, hashlib
import numpy as np
import pandas as pd
from .registry import ROOT, ESTIMATORS
from .inputs import atomic_json, sha
from .core import Settings, fit_reference, edge_attributes
from .generate import pearson_block, weights, unpack, PARAMS
from . import adapter as A


def prepare_reference_caches(root=ROOT):
    """Offline preparation only. No app rerender fits preprocessing or controls."""
    root = Path(root)
    from .hosted import is_hosted

    if is_hosted(root):
        raise A.ScoreUnavailable(
            "Edge reconstruction remains local. The hosted release contains donor-score summaries; private expression/reference arrays are not published."
        )
    catalog = A.catalog(root)
    for variant, info in catalog["datasets"].items():
        path = root / "private" / f"{variant}_reference.npz"
        if path.exists():
            continue
        x_path = root / "private" / f"{variant}_expression.npy"
        p_path = root / "private" / f"{variant}_presence.npy"
        if (
            sha(x_path) != info["expression_sha256"]
            or sha(p_path) != info["presence_sha256"]
        ):
            raise ValueError(
                "Prepared expression/presence content hash differs from catalog"
            )
        X = np.load(x_path, mmap_mode="r")
        presence = np.load(p_path)
        R = np.asarray(info["role_indices"]["R"])
        nt = np.array([info["tissues"].index(n[0]) for n in info["nodes"]])
        ref = fit_reference(
            X[R], Settings(len(R), len(info["role_indices"]["C"])), presence[R][:, nt]
        )
        tmp = path.with_suffix(".partial.npz")
        np.savez(tmp, **ref, node_tissues=nt)
        tmp.replace(path)
        atomic_json(
            path.with_suffix(".json"),
            dict(
                array_sha256=sha(path),
                expression_sha256=info["expression_sha256"],
                presence_sha256=info["presence_sha256"],
                R=R.tolist(),
                C=info["role_indices"]["C"],
                sparse_cell_policy="R_mean",
                fit_population="R only; frozen preprocessing used for bounded edge reconstruction",
            ),
        )


@lru_cache(maxsize=3)
def reference_cache(root, variant, revision):
    root = Path(root)
    path = root / "private" / f"{variant}_reference.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != receipt["array_sha256"]:
        raise ValueError("Frozen reference array content hash mismatch")
    return dict(np.load(path))


@lru_cache(maxsize=6)
def calibration_cache(root, variant, beta, revision):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    report = next(v for v in m["generation"] if v["variant"] == variant)
    receipt = report["parameter_files"][str(beta)]
    path = root / receipt["path"]
    if sha(path) != receipt["sha256"]:
        raise ValueError("Calibration parameter array content hash mismatch")
    return np.load(path, mmap_mode="r")


@lru_cache(maxsize=6)
def verify_source(path, revision, expected):
    if sha(path) != expected:
        raise ValueError("Frozen expression/availability content hash mismatch")
    return True


def iter_edges(
    module_set,
    module,
    sample_ids,
    estimator="calibrated_beta3",
    component="all",
    cohort_scope="complete_450",
    regional_tissue=None,
    mask="all",
    fdr_scope="global",
    threshold=0.05,
    edge_batch=2048,
    root=ROOT,
):
    root = Path(root)
    from .hosted import is_hosted

    if is_hosted(root):
        raise A.ScoreUnavailable(
            "Edge reconstruction remains local. The hosted release contains donor-score summaries; private expression/reference arrays are not published."
        )
    catalog = A.catalog(root)
    jobs = [
        j
        for j in catalog["modules"]
        if j["family"] == module_set
        and j["module"] == int(module)
        and (regional_tissue is None or j["regional_tissue"] == regional_tissue)
    ]
    if len(jobs) != 1:
        raise A.ScoreUnavailable(
            "An edge request must resolve one partition/tissue/module namespace"
        )
    job = jobs[0]
    variant = job["input_variant"]
    info = catalog["datasets"][variant]
    metadata = A.metadata(module_set, cohort_scope, root)
    selected = metadata[metadata.sample_id.isin(sample_ids)]
    if len(selected) != len(set(sample_ids)):
        raise A.ScoreUnavailable(
            "R/C or unaligned donor requested for an independent-target edge view"
        )
    allmeta = pd.read_parquet(root / info["metadata_file"])
    indices = allmeta.index[allmeta.sample_id.isin(sample_ids)].to_numpy()
    public = allmeta.iloc[indices].sample_id.to_numpy()
    path = root / "private" / f"{variant}_reference.npz"
    if set(indices) & set(info["role_indices"]["R"] + info["role_indices"]["C"]):
        raise A.ScoreUnavailable(
            "Training-role donor cannot enter independent-target edge inspection"
        )
    if not path.exists():
        raise A.ScoreUnavailable(
            "Frozen reference reconstruction cache has not been prepared offline"
        )
    ref = reference_cache(str(root), variant, path.stat().st_mtime_ns)
    for suffix, expected in [
        ("expression", info["expression_sha256"]),
        ("presence", info["presence_sha256"]),
    ]:
        source = root / "private" / f"{variant}_{suffix}.npy"
        verify_source(str(source), source.stat().st_mtime_ns, expected)
    X = np.load(root / "private" / f"{variant}_expression.npy", mmap_mode="r")
    presence = np.load(root / "private" / f"{variant}_presence.npy")
    nt = ref["node_tissues"]
    nodes = np.asarray(job["node_indices"])
    r, c = np.triu_indices(len(nodes), 1)
    codes = np.unique(
        np.minimum(nodes[r], nodes[c]) * info["n_nodes"]
        + np.maximum(nodes[r], nodes[c])
    )
    rows = codes // info["n_nodes"]
    cols = codes % info["n_nodes"]
    tissues = np.array([n[0] for n in info["nodes"]])
    comp = A.component_public(component)
    if comp == "TS":
        keep = tissues[rows] == tissues[cols]
    elif comp == "CT":
        keep = tissues[rows] != tissues[cols]
    elif comp.startswith("TS_"):
        keep = (tissues[rows] == comp[3:]) & (tissues[cols] == comp[3:])
    elif comp.startswith("CT_"):
        a, b = comp[3:].split("__")
        keep = ((tissues[rows] == a) & (tissues[cols] == b)) | (
            (tissues[rows] == b) & (tissues[cols] == a)
        )
    elif comp == "all":
        keep = np.ones(len(rows), bool)
    else:
        raise A.ScoreUnavailable("Unknown physical-edge component: " + str(component))
    codes = codes[keep]
    if mask != "all":
        from .masks import key

        mp = (
            root
            / "private/mask_mappings"
            / A.module_filename(job["module_key"]).replace(".parquet", ".npz")
        )
        receipt = json.loads(mp.with_suffix(".json").read_text())
        if sha(mp) != receipt["array_sha256"]:
            raise ValueError("Frozen mask array content hash mismatch")
        arrays = np.load(mp)
        mask_key = key(mask, fdr_scope, threshold)
        if mask_key not in arrays:
            raise A.ScoreUnavailable(
                receipt.get("unavailable", {})
                .get(mask_key, {})
                .get("reason", "Physical mask mapping is unavailable")
            )
        codes = np.intersect1d(codes, arrays["codes"][arrays[mask_key]])
    work = root / "private" / variant
    global_codes = np.load(work / "edges.npy", mmap_mode="r")
    beta = ESTIMATORS[estimator]
    parameter_path = work / f"calibration_beta{beta}.npy"
    parameters = calibration_cache(
        str(root), variant, beta, parameter_path.stat().st_mtime_ns
    )
    position = np.searchsorted(global_codes, codes)
    if not np.array_equal(global_codes[position], codes):
        raise ValueError("Physical-edge cache universe mismatch")
    denominator = int(parameters[position, 10].sum())
    signed_denominator = int(parameters[position, 11].sum())
    R = np.asarray(info["role_indices"]["R"])
    n = len(R)
    X_selected = np.asarray(X[indices])
    P = presence[indices]
    names = info["nodes"]
    endpoints = np.unique(np.r_[codes // info["n_nodes"], codes % info["n_nodes"]])
    required = np.unique(nt[endpoints])
    aggregate_available = (
        P[:, required].all(1) if len(required) else np.ones(len(indices), bool)
    )
    for start in range(0, len(codes), edge_batch):
        code = codes[start : start + edge_batch]
        r = code // info["n_nodes"]
        c = code % info["n_nodes"]
        r0, r1, valid, imputed = pearson_block(
            X_selected, ref, P, nt, r, c, np.array([], int)
        )
        w = weights(r0, r1, beta, np.array([], int), n)
        w[~valid] = np.nan
        cal = unpack(np.asarray(parameters[position[start : start + edge_batch]]))
        a, z = edge_attributes(w, cal)
        zero = w == 0
        near = (abs(w) <= 0.01 * cal["nu"]) & ~zero
        uncertain = ~cal["mean_sign_supported"] & ~zero & ~near
        reversed_ = (
            cal["mean_sign_supported"]
            & (np.sign(w) != np.sign(cal["mu"]))
            & ~zero
            & ~near
        )
        sign_class = np.full(w.shape, "preserved", dtype="<U12")
        sign_class[reversed_] = "reversed"
        sign_class[uncertain] = "uncertain"
        sign_class[near] = "near_zero"
        sign_class[zero] = "zero"
        reason = np.where(
            cal["valid"],
            "eligible",
            np.where(
                cal["n"] < len(info["role_indices"]["C"]),
                "incomplete_declared_calibration_count",
                "constant_magnitude_or_relative_guard",
            ),
        )
        for di, public_id in enumerate(public):
            yield pd.DataFrame(
                dict(
                    sample_id=public_id,
                    physical_edge_code=code,
                    tissue_a=[names[v][0] for v in r],
                    gene_a=[names[v][1] for v in r],
                    tissue_b=[names[v][0] for v in c],
                    gene_b=[names[v][1] for v in c],
                    w=w[di],
                    abs_w=abs(w[di]),
                    delta_abs=abs(w[di]) - cal["nu"],
                    a=a[di],
                    nu=cal["nu"],
                    tau=cal["tau"],
                    calibration_n=cal["n"].astype(int),
                    eligible=cal["valid"],
                    eligibility_reason=reason,
                    signed_eligible=cal["signed_valid"],
                    z=z[di],
                    sign_class=sign_class[di],
                    mean_sign_supported=cal["mean_sign_supported"],
                    sign_agreement=cal["sign_agreement"],
                    gain_contribution=(
                        np.maximum(a[di], 0) / denominator
                        if denominator and aggregate_available[di]
                        else np.nan
                    ),
                    loss_contribution=(
                        np.maximum(-a[di], 0) / denominator
                        if denominator and aggregate_available[di]
                        else np.nan
                    ),
                    aggregate_denominator=denominator,
                    signed_denominator=signed_denominator,
                    aggregate_available=aggregate_available[di],
                    aggregate_unavailable_reason=(
                        "missing_required_tissue"
                        if not aggregate_available[di]
                        else "no_eligible_magnitude_edges" if not denominator else "ok"
                    ),
                    endpoint_imputed=imputed[di],
                    beta=beta,
                    module_key=job["module_key"],
                    allocation_id=info["allocation_id"],
                    component=component,
                    cohort_scope=cohort_scope,
                    mask=mask,
                    role="independent_target",
                    score_version=f"control_calibrated_connectivity.v2.beta{beta}",
                )
            )


def ranked_edges(*args, rank_by="gain_contribution", top=100, **kwargs):
    pieces = []
    for frame in iter_edges(*args, **kwargs):
        pieces.append(frame.nlargest(top, rank_by))
    return (
        pd.concat(pieces, ignore_index=True).nlargest(top, rank_by)
        if pieces
        else pd.DataFrame()
    )
