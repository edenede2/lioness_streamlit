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
from models.se2.methods.soft_weighted import build_m5_adjacency  # noqa: E402


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--stage", choices=("network", "summarize", "all"), default="all")
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


if __name__ == "__main__":
    main()
