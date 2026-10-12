#!/usr/bin/env python3
"""Build full Method-5 AD/Control networks and summarize muscle-module MDC.

The network stage deliberately uses the complete Method-5 extended-SFN
functional (bootstrap stability, CT null excess, and block q90 scaling), not a
beta-only signed adjacency.  Network files are reusable by the MDC summary and
split-triangle heatmap stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from models.se2.adjacency import make_block_info  # noqa: E402
from models.se2.data import load_expression, prepare_multitissue  # noqa: E402
from models.se2.methods.null_calibrated import compute_block_null  # noqa: E402
from models.se2.methods.soft_weighted import (  # noqa: E402
    _full_bootstrap_stability,
    build_m5_adjacency,
)


TISSUES = ("VH", "SMA", "QUAD")
GROUP_COUNTS = {"Control": 120, "AD": 155}
BETAS_TS = {"VH": 5, "SMA": 7, "QUAD": 13}
BETAS_CT = {"VH-SMA": 8, "VH-QUAD": 10, "SMA-QUAD": 10}
M5_PARAMETERS = {
    "beta_ts": BETAS_TS,
    "beta_ct": BETAS_CT,
    "corr_method": "pearson",
    "B": 200,
    "donor_frac": 0.80,
    "tau_r": 0.10,
    "num_permutations": 50,
    "null_sample_edges": 100_000,
    "null_eps": 1e-8,
    "final_epsilon": 0.0,
    "seed": 42,
}
DEFAULT_EXPRESSION = REPO_ROOT / "data/raw/ROSMAP_muscle_split"
DEFAULT_ASSIGNMENTS = (
    REPO_ROOT
    / "results/se2_all_genes/20260319_090116/M5_all_genes/"
    "M5_all_genes_table_3.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "out/mdc_muscle_control_l4/20261012_m5_ad_vs_control"
SCOPES = (
    "total",
    "TS",
    "CT",
    "TS_VH",
    "TS_SMA",
    "TS_QUAD",
    "CT_VH__SMA",
    "CT_VH__QUAD",
    "CT_SMA__QUAD",
)
BLOCKS = (
    ("VH", "VH"),
    ("SMA", "SMA"),
    ("QUAD", "QUAD"),
    ("VH", "SMA"),
    ("VH", "QUAD"),
    ("SMA", "QUAD"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    """Return a deterministic uint32 seed independent of Python hash state."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Could not determine available RAM")


def expression_path(root: Path, tissue: str, group: str) -> Path:
    return root / f"ROSMAP_tissue_{tissue.lower()}_{group}.csv"


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def load_group(root: Path, group: str) -> tuple[dict[str, pd.DataFrame], list[str], list[dict[str, object]]]:
    if group not in GROUP_COUNTS:
        raise ValueError(group)
    raw: dict[str, pd.DataFrame] = {}
    sources: list[dict[str, object]] = []
    for tissue in TISSUES:
        path = expression_path(root, tissue, group)
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))[1:]
        if len(header) != len(set(header)) or any(not value for value in header):
            raise ValueError(f"Duplicate or empty gene header in {path}")
        expression = load_expression(str(path))
        if expression.index.has_duplicates or expression.columns.has_duplicates:
            raise ValueError(f"Duplicate genes or donors in {path}")
        if not np.isfinite(expression.to_numpy()).all():
            raise ValueError(f"Non-finite expression in {path}")
        raw[tissue] = expression
        sources.append(file_identity(path))
    tissues, donors = prepare_multitissue(raw, n_genes=None, aggregate_donors=False)
    if len(donors) != GROUP_COUNTS[group]:
        raise ValueError(f"{group}: expected {GROUP_COUNTS[group]} donors, found {len(donors)}")
    for tissue, expression in tissues.items():
        if list(expression.columns) != donors:
            raise ValueError(f"{group}/{tissue}: donor order differs")
        if np.any(np.ptp(expression.to_numpy(), axis=1) == 0):
            raise ValueError(f"{group}/{tissue}: constant genes after donor intersection")
    return tissues, donors, sources


def node_frame(tissues: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for tissue, expression in tissues.items():
        for gene in expression.index:
            rows.append({"node_index": index, "tissue": tissue, "gene_id": str(gene)})
            index += 1
    return pd.DataFrame(rows)


def build_network(expression_root: Path, output: Path, group: str, force: bool) -> Path:
    group_dir = output / "networks" / group.lower()
    adjacency_path = group_dir / "m5_adjacency_upper_float32.npy"
    diagnostics_path = group_dir / "m5_diagnostics.json"
    manifest_path = group_dir / "manifest.json"
    if adjacency_path.exists() and diagnostics_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete existing {group} network manifest")
        if sha256_file(adjacency_path) != manifest["adjacency_sha256"]:
            raise RuntimeError(f"{group} adjacency checksum mismatch")
        return adjacency_path
    if available_ram_gib() < 100:
        raise MemoryError("M5 adjacency preflight requires at least 100 GiB available RAM")

    group_dir.mkdir(parents=True, exist_ok=True)
    tissues, donors, sources = load_group(expression_root, group)
    nodes = node_frame(tissues)
    if len(nodes) != 37_324:
        raise ValueError(f"Expected 37,324 nodes, found {len(nodes)}")
    nodes.to_parquet(group_dir / "node_index.parquet", index=False, compression="zstd")
    running = {
        "status": "running",
        "group": group,
        "started_at_utc": utc_now(),
        "donor_count": len(donors),
        "node_count": len(nodes),
        "m5_parameters": M5_PARAMETERS,
        "input_files": sources,
    }
    atomic_json(running, manifest_path)
    adjacency, diagnostics = build_m5_adjacency(
        tissues, make_block_info(tissues), **M5_PARAMETERS
    )
    if adjacency.shape != (len(nodes), len(nodes)) or adjacency.dtype != np.float32:
        raise ValueError("M5 returned an unexpected adjacency shape or dtype")
    for lower in range(0, len(adjacency), 256):
        upper = min(lower + 256, len(adjacency))
        rows = adjacency[lower:upper]
        if not np.isfinite(rows).all():
            raise ValueError("M5 adjacency contains invalid values")
        if np.any(rows[:, :lower]) or np.any(np.tril(rows[:, lower:upper])):
            raise ValueError("M5 checkpoint must be strict upper triangular")
    temporary = adjacency_path.with_name(adjacency_path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, adjacency, allow_pickle=False)
    os.replace(temporary, adjacency_path)
    atomic_json(diagnostics, diagnostics_path)
    running.update(
        {
            "status": "complete",
            "finished_at_utc": utc_now(),
            "adjacency_sha256": sha256_file(adjacency_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "node_index_sha256": sha256_file(group_dir / "node_index.parquet"),
        }
    )
    atomic_json(running, manifest_path)
    return adjacency_path


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna() & np.isfinite(values)
    if not valid.any():
        return result
    p = values.loc[valid].to_numpy(dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid] = restored
    return result


def scope_edges(tissues: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    first, second = np.triu_indices(len(tissues), k=1)
    same = tissues[first] == tissues[second]
    masks: dict[str, np.ndarray] = {
        "total": np.ones(len(first), dtype=bool),
        "TS": same,
        "CT": ~same,
    }
    for tissue in TISSUES:
        masks[f"TS_{tissue}"] = same & (tissues[first] == tissue)
    for left, right in (("VH", "SMA"), ("VH", "QUAD"), ("SMA", "QUAD")):
        masks[f"CT_{left}__{right}"] = (
            ((tissues[first] == left) & (tissues[second] == right))
            | ((tissues[first] == right) & (tissues[second] == left))
        )
    return {scope: (first[mask], second[mask]) for scope, mask in masks.items()}


def read_filtered_assignments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Tissue": str, "Gene ID": str, "Gene Symbol": str})
    sizes = frame.groupby("Cluster ID", observed=True).size()
    frame = frame.loc[frame["Cluster ID"].isin(sizes.loc[sizes.ge(30)].index)].copy()
    if frame.duplicated(["Tissue", "Gene ID"]).any():
        raise ValueError("Duplicate tissue-gene rows in assignments")
    return frame


def _resolve_beta(left: str, right: str) -> float:
    if left == right:
        return float(BETAS_TS[left])
    direct = f"{left}-{right}"
    reverse = f"{right}-{left}"
    return float(BETAS_CT[direct] if direct in BETAS_CT else BETAS_CT[reverse])


def _group_null_statistics(
    tissues: dict[str, pd.DataFrame], group: str
) -> dict[str, dict[str, float | int]]:
    """Estimate the M5 inter-tissue donor-permutation nulls.

    The original full M5 builder samples at most 100,000 candidate edges from
    each cross-tissue block.  At tau=0.1 and B=200 essentially every edge is a
    candidate, so sampling directly from the structurally possible block is
    equivalent while avoiding allocation of a full 15k x 15k mask.
    """
    results: dict[str, dict[str, float | int]] = {}
    for left, right in BLOCKS:
        if left == right:
            continue
        Xi = tissues[left].to_numpy(dtype=np.float32, copy=False)
        Xj = tissues[right].to_numpy(dtype=np.float32, copy=False)
        possible = Xi.shape[0] * Xj.shape[0]
        sample_n = min(M5_PARAMETERS["null_sample_edges"], possible)
        rng = np.random.default_rng(stable_seed("muscle-m5-null", group, left, right, 42))
        flat = rng.choice(possible, size=sample_n, replace=False)
        src = (flat // Xj.shape[0]).astype(np.int32, copy=False)
        dst = (flat % Xj.shape[0]).astype(np.int32, copy=False)
        mu, sigma, _ = compute_block_null(
            Xi,
            Xj,
            src,
            dst,
            n_perms=M5_PARAMETERS["num_permutations"],
            max_edges=sample_n,
            rng=rng,
            use_spearman=False,
        )
        key = f"{left}-{right}"
        results[key] = {
            "mu_null": float(mu),
            "sigma_null": float(sigma),
            "sampled_edges": int(sample_n),
            "possible_edges": int(possible),
            "permutations": int(M5_PARAMETERS["num_permutations"]),
            "seed": stable_seed("muscle-m5-null", group, left, right, 42),
        }
        print(
            f"  [M5 recovery] {group} {key} null: mu={mu:.6f}, "
            f"sigma={sigma:.6f}, sampled={sample_n:,}",
            flush=True,
        )
    return results


def _raw_module_m5(
    module: int,
    rows: pd.DataFrame,
    tissues: dict[str, pd.DataFrame],
    group: str,
    nulls: dict[str, dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the complete unnormalised M5 subnetwork for one module."""
    node_tissues = rows["Tissue"].astype(str).to_numpy()
    gene_ids = rows["Gene ID"].astype(str).to_numpy()
    count = len(rows)
    weights = np.zeros((count, count), dtype=np.float32)
    for left, right in BLOCKS:
        left_positions = np.flatnonzero(node_tissues == left)
        right_positions = np.flatnonzero(node_tissues == right)
        if left == right and len(left_positions) < 2:
            continue
        if left != right and (not len(left_positions) or not len(right_positions)):
            continue
        left_lookup = {str(gene): idx for idx, gene in enumerate(tissues[left].index)}
        right_lookup = (
            left_lookup
            if left == right
            else {str(gene): idx for idx, gene in enumerate(tissues[right].index)}
        )
        try:
            left_rows = np.asarray(
                [left_lookup[gene_ids[position]] for position in left_positions], dtype=np.int64
            )
            right_rows = np.asarray(
                [right_lookup[gene_ids[position]] for position in right_positions], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(f"M{module}: expression is missing {error.args[0]}") from error
        X = tissues[left].to_numpy(dtype=np.float32, copy=False)[left_rows]
        Y = tissues[right].to_numpy(dtype=np.float32, copy=False)[right_rows]
        rng = np.random.default_rng(
            stable_seed("muscle-m5-bootstrap", group, module, left, right, 42)
        )
        freq, mean_r = _full_bootstrap_stability(
            X,
            Y,
            B=M5_PARAMETERS["B"],
            donor_frac=M5_PARAMETERS["donor_frac"],
            corr_method=M5_PARAMETERS["corr_method"],
            tau_r=M5_PARAMETERS["tau_r"],
            is_intra=left == right,
            rng=rng,
        )
        beta = _resolve_beta(left, right)
        raw = np.sign(mean_r) * freq * np.abs(mean_r) ** beta
        if left != right:
            stats = nulls[f"{left}-{right}"]
            z = np.maximum(
                (np.abs(mean_r) - float(stats["mu_null"]))
                / (float(stats["sigma_null"]) + M5_PARAMETERS["null_eps"]),
                0.0,
            )
            raw = np.sign(mean_r) * freq * np.abs(mean_r) ** beta * z
            weights[np.ix_(left_positions, right_positions)] = raw
            weights[np.ix_(right_positions, left_positions)] = raw.T
        else:
            upper_i, upper_j = np.triu_indices(len(left_positions), k=1)
            values = raw[upper_i, upper_j]
            weights[left_positions[upper_i], left_positions[upper_j]] = values
            weights[left_positions[upper_j], left_positions[upper_i]] = values
    return weights, node_tissues, gene_ids


def _block_values(weights: np.ndarray, node_tissues: np.ndarray, left: str, right: str) -> np.ndarray:
    left_positions = np.flatnonzero(node_tissues == left)
    right_positions = np.flatnonzero(node_tissues == right)
    if left == right:
        if len(left_positions) < 2:
            return np.empty(0, dtype=np.float32)
        first, second = np.triu_indices(len(left_positions), k=1)
        return weights[left_positions[first], left_positions[second]]
    if not len(left_positions) or not len(right_positions):
        return np.empty(0, dtype=np.float32)
    return weights[np.ix_(left_positions, right_positions)].ravel()


def build_recovered_module_networks(
    expression_root: Path,
    assignments_path: Path,
    output: Path,
    group: str,
    force: bool,
) -> Path:
    """Build M5 only for edges recoverable inside the filtered L4 modules.

    This is the relevant edge universe for module MDC.  It retains every term
    of Method 5 and derives each block q90 over that same recovered universe,
    avoiding the >600 million unused between-module edges in a dense supra
    adjacency.  Raw module checkpoints make the run resumable.
    """
    group_dir = output / "recovered_modules" / group.lower()
    raw_dir = group_dir / "raw"
    module_dir = group_dir / "modules"
    manifest_path = group_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            return module_dir
    group_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    module_dir.mkdir(parents=True, exist_ok=True)
    tissues, donors, sources = load_group(expression_root, group)
    assignments = read_filtered_assignments(assignments_path)
    if assignments["Cluster ID"].nunique() != 155:
        raise ValueError("Expected 155 filtered L4 modules")
    null_path = group_dir / "ct_null_statistics.json"
    if null_path.exists() and not force:
        nulls = json.loads(null_path.read_text())
    else:
        nulls = _group_null_statistics(tissues, group)
        atomic_json(nulls, null_path)
    running = {
        "status": "running",
        "group": group,
        "started_at_utc": utc_now(),
        "donor_count": len(donors),
        "module_count": 155,
        "assignment_count": int(len(assignments)),
        "m5_parameters": M5_PARAMETERS,
        "normalization_edge_universe": (
            "All undirected edges whose two tissue-gene endpoints occur in the same "
            "filtered biological-L4 module; block q90 is estimated across this complete "
            "recovered-module edge universe separately in AD and Control."
        ),
        "input_files": sources + [file_identity(assignments_path)],
    }
    atomic_json(running, manifest_path)

    block_parts: dict[str, list[np.ndarray]] = {f"{a}-{b}": [] for a, b in BLOCKS}
    grouped = list(assignments.groupby("Cluster ID", sort=True, observed=True))
    for position, (module_value, rows) in enumerate(grouped, start=1):
        module = int(module_value)
        raw_path = raw_dir / f"M{module}.npz"
        if raw_path.exists() and not force:
            # These are private checkpoints produced by this script.  Early
            # checkpoints stored pandas-derived tissue labels as an object
            # array; permit that representation when resuming, then rewrite
            # the public-facing normalized file with plain Unicode strings.
            checkpoint = np.load(raw_path, allow_pickle=True)
            weights = checkpoint["weights"]
            node_tissues = checkpoint["tissues"].astype(str)
        else:
            weights, node_tissues, gene_ids = _raw_module_m5(
                module, rows, tissues, group, nulls
            )
            temporary = raw_path.with_name(raw_path.name + ".partial")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    weights=weights,
                    tissues=node_tissues,
                    gene_ids=gene_ids,
                )
            os.replace(temporary, raw_path)
        for left, right in BLOCKS:
            values = _block_values(weights, node_tissues, left, right)
            if len(values):
                block_parts[f"{left}-{right}"].append(np.abs(values[values != 0]))
        if position % 5 == 0 or position == len(grouped):
            print(
                f"  [M5 recovery] {group}: raw checkpoints {position}/{len(grouped)}",
                flush=True,
            )

    normalizers: dict[str, dict[str, float | int]] = {}
    for left, right in BLOCKS:
        key = f"{left}-{right}"
        values = np.concatenate(block_parts[key]) if block_parts[key] else np.empty(0)
        q90 = float(np.quantile(values, 0.90)) if len(values) else 0.0
        normalizers[key] = {"q90": q90, "nonzero_recovered_edges": int(len(values))}
    atomic_json(normalizers, group_dir / "block_q90_normalizers.json")

    for position, (module_value, _) in enumerate(grouped, start=1):
        module = int(module_value)
        raw = np.load(raw_dir / f"M{module}.npz", allow_pickle=True)
        weights = raw["weights"].astype(np.float32, copy=True)
        node_tissues = raw["tissues"].astype(str)
        for left, right in BLOCKS:
            key = f"{left}-{right}"
            q90 = float(normalizers[key]["q90"])
            left_positions = np.flatnonzero(node_tissues == left)
            right_positions = np.flatnonzero(node_tissues == right)
            if q90 <= 0 or not len(left_positions) or not len(right_positions):
                continue
            if left == right:
                index = np.ix_(left_positions, left_positions)
                block = np.clip(weights[index] / q90, -10.0, 10.0)
                weights[index] = block
            else:
                block = weights[np.ix_(left_positions, right_positions)] / q90
                block = np.clip(block, -10.0, 10.0)
                weights[np.ix_(left_positions, right_positions)] = block
                weights[np.ix_(right_positions, left_positions)] = block.T
        destination = module_dir / f"M{module}.npz"
        temporary = destination.with_name(destination.name + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                weights=weights,
                tissues=node_tissues,
                gene_ids=raw["gene_ids"].astype(str),
            )
        os.replace(temporary, destination)
        if position % 25 == 0 or position == len(grouped):
            print(
                f"  [M5 recovery] {group}: normalized {position}/{len(grouped)}",
                flush=True,
            )
    running.update(
        {
            "status": "complete",
            "finished_at_utc": utc_now(),
            "ct_null_statistics_sha256": sha256_file(null_path),
            "normalizers_sha256": sha256_file(group_dir / "block_q90_normalizers.json"),
            "completed_module_files": 155,
        }
    )
    atomic_json(running, manifest_path)
    return module_dir


def summarize_recovered_modules(assignments_path: Path, output: Path, force: bool) -> Path:
    destination = output / "mdc_observed.parquet"
    if destination.exists() and not force:
        return destination
    assignments = read_filtered_assignments(assignments_path)
    records: list[dict[str, object]] = []
    for module in sorted(assignments["Cluster ID"].unique()):
        group_data = {}
        for group in ("ad", "control"):
            path = output / "recovered_modules" / group / "modules" / f"M{int(module)}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            group_data[group] = np.load(path, allow_pickle=False)
        ad = group_data["ad"]["weights"]
        control = group_data["control"]["weights"]
        ad_tissues = group_data["ad"]["tissues"].astype(str)
        control_tissues = group_data["control"]["tissues"].astype(str)
        if not np.array_equal(ad_tissues, control_tissues):
            raise ValueError(f"M{module}: AD/Control node order differs")
        records.extend(
            summarize_module(
                int(module), np.arange(len(ad_tissues)), ad_tissues, ad, control
            )
        )
    frame = pd.DataFrame(records)
    if frame["module"].nunique() != 155 or len(frame) != 155 * len(SCOPES):
        raise AssertionError("Observed MDC table has unexpected dimensions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")
    return destination


def summarize_module(
    module: int,
    indices: np.ndarray,
    node_tissues: np.ndarray,
    ad: np.ndarray,
    control: np.ndarray,
) -> list[dict[str, object]]:
    tissues = node_tissues[indices]
    rows: list[dict[str, object]] = []
    for scope, (local_i, local_j) in scope_edges(tissues).items():
        if len(local_i) == 0:
            rows.append({"module": module, "scope": scope, "n_edges": 0,
                         "mdc_raw": np.nan, "mdc_log2": np.nan,
                         "mean_abs_ad": np.nan, "mean_abs_control": np.nan})
            continue
        global_i, global_j = indices[local_i], indices[local_j]
        # Stored matrices are strict upper triangular in global node order.
        lo = np.minimum(global_i, global_j)
        hi = np.maximum(global_i, global_j)
        ad_values = np.abs(ad[lo, hi])
        control_values = np.abs(control[lo, hi])
        ad_mean = float(ad_values.mean())
        control_mean = float(control_values.mean())
        ratio = ad_mean / control_mean if control_mean > 1e-12 else np.nan
        rows.append({
            "module": module,
            "scope": scope,
            "n_edges": len(local_i),
            "mdc_raw": ratio,
            "mdc_log2": math.log2(ratio) if np.isfinite(ratio) and ratio > 0 else np.nan,
            "mean_abs_ad": ad_mean,
            "mean_abs_control": control_mean,
        })
    return rows


def summarize_networks(assignments_path: Path, output: Path, force: bool) -> Path:
    destination = output / "mdc_observed.parquet"
    if destination.exists() and not force:
        return destination
    ad_path = output / "networks/ad/m5_adjacency_upper_float32.npy"
    control_path = output / "networks/control/m5_adjacency_upper_float32.npy"
    if not ad_path.exists() or not control_path.exists():
        raise FileNotFoundError("Both completed M5 group adjacencies are required")
    ad = np.load(ad_path, mmap_mode="r")
    control = np.load(control_path, mmap_mode="r")
    node_path = output / "networks/control/node_index.parquet"
    nodes = pd.read_parquet(node_path)
    other_nodes = pd.read_parquet(output / "networks/ad/node_index.parquet")
    if not nodes.equals(other_nodes):
        raise ValueError("AD and Control M5 node order differs")
    assignments = read_filtered_assignments(assignments_path)
    lookup = dict(zip(zip(nodes["tissue"], nodes["gene_id"]), nodes["node_index"], strict=True))
    modules: dict[int, np.ndarray] = {}
    for module, rows in assignments.groupby("Cluster ID", observed=True):
        keys = list(zip(rows["Tissue"].astype(str), rows["Gene ID"].astype(str), strict=True))
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ValueError(f"M{module}: {len(missing)} nodes absent from M5 network")
        modules[int(module)] = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    records = [
        row
        for module, indices in sorted(modules.items())
        for row in summarize_module(
            module, indices, nodes["tissue"].to_numpy(), ad, control
        )
    ]
    frame = pd.DataFrame(records)
    if frame["module"].nunique() != 155 or len(frame) != 155 * len(SCOPES):
        raise AssertionError("Observed MDC table has unexpected dimensions")
    frame.to_parquet(destination, index=False, compression="zstd")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("network", "summarize", "recovered-network", "recovered-summarize", "recovered-all", "all"),
        default="recovered-all",
    )
    parser.add_argument("--group", choices=tuple(GROUP_COUNTS), action="append", default=[])
    parser.add_argument("--expression-root", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = args.group or list(GROUP_COUNTS)
    if args.stage in {"network", "all"}:
        for group in groups:
            path = build_network(args.expression_root, args.output, group, args.force)
            print(f"Completed {group} M5 network: {path}", flush=True)
    if args.stage in {"summarize", "all"}:
        path = summarize_networks(args.assignments, args.output, args.force)
        print(f"Completed observed MDC summary: {path}", flush=True)
    if args.stage in {"recovered-network", "recovered-all"}:
        for group in groups:
            path = build_recovered_module_networks(
                args.expression_root, args.assignments, args.output, group, args.force
            )
            print(f"Completed {group} recovered-module M5 network: {path}", flush=True)
    if args.stage in {"recovered-summarize", "recovered-all"}:
        path = summarize_recovered_modules(args.assignments, args.output, args.force)
        print(f"Completed recovered-module observed MDC summary: {path}", flush=True)


if __name__ == "__main__":
    main()
