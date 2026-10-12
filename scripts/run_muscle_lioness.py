#!/usr/bin/env python3
"""Compute resumable LIONESS module features for VH/SMA/QUAD.

The fixed module definition is the Control-only SE2 ``table_3`` partition.
Modules smaller than 30 tissue-gene nodes are excluded.  This program writes
only donor-by-module summaries; donor-specific edge matrices are never stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
TISSUES = ("VH", "SMA", "QUAD")
GROUPS = ("Control", "MCI", "AD")
METHODS = ("standard", "control_anchored")
BETAS_TS = {"VH": 5.0, "SMA": 7.0, "QUAD": 13.0}
BETAS_CT = {"VH-SMA": 8.0, "VH-QUAD": 10.0, "SMA-QUAD": 10.0}
DEFAULT_ASSIGNMENTS = (
    REPO_ROOT
    / "results/se2_all_genes/20260319_090116/M5_all_genes/"
    "M5_all_genes_table_3.csv"
)
DEFAULT_EXPRESSION = REPO_ROOT / "data/raw/ROSMAP_muscle_split"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "out/lioness_reference_comparison_rosmap/"
    "20261012_muscle_control_l4_nonprediction"
)
FEATURE_FAMILIES = (
    "connectivity",
    "positive_density",
    "negative_density",
    "abs_sum",
    "positive_abs_sum",
    "negative_abs_sum",
)
RESOLVED_COMPONENTS = (
    "TS_VH",
    "TS_SMA",
    "TS_QUAD",
    "CT_VH__SMA",
    "CT_VH__QUAD",
    "CT_SMA__QUAD",
)
ALL_COMPONENTS = ("all", "TS", "CT", *RESOLVED_COMPONENTS)


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
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary,
                       compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expression_path(root: Path, tissue: str, group: str) -> Path:
    return root / f"ROSMAP_tissue_{tissue.lower()}_{group}.csv"


def donor_to_projid(value: str) -> str:
    return str(int(str(value).split("-", 1)[-1]))


def pair_key(first: str, second: str) -> str:
    positions = {tissue: index for index, tissue in enumerate(TISSUES)}
    left, right = sorted((first, second), key=positions.__getitem__)
    return f"{left}-{right}"


@dataclass(frozen=True)
class Cohort:
    donors: tuple[str, ...]
    projids: tuple[str, ...]
    diagnoses: tuple[str, ...]
    matrices: dict[str, np.ndarray]
    genes: dict[str, tuple[str, ...]]
    gene_indices: dict[str, dict[str, int]]


def read_assignments(path: Path, minimum_size: int) -> pd.DataFrame:
    assignments = pd.read_csv(path, dtype={"Gene Symbol": "string", "Tissue": "string"})
    required = {"Cluster ID", "Tissue", "Gene Symbol"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignment table is missing columns: {sorted(missing)}")
    assignments = assignments.loc[
        assignments["Tissue"].isin(TISSUES),
        ["Cluster ID", "Tissue", "Gene Symbol"],
    ].copy()
    assignments["Cluster ID"] = pd.to_numeric(
        assignments["Cluster ID"], errors="raise"
    ).astype("int64")
    if assignments.duplicated(["Tissue", "Gene Symbol"]).any():
        duplicate = assignments.loc[
            assignments.duplicated(["Tissue", "Gene Symbol"], keep=False)
        ].head()
        raise ValueError(f"Duplicate tissue-gene assignments detected:\n{duplicate}")
    sizes = assignments.groupby("Cluster ID", observed=True).size()
    keep = sizes.loc[sizes.ge(minimum_size)].index
    result = assignments.loc[assignments["Cluster ID"].isin(keep)].copy()
    result = result.sort_values(["Cluster ID", "Tissue", "Gene Symbol"])
    if result["Cluster ID"].nunique() != len(keep):
        raise AssertionError("Filtered module count changed unexpectedly")
    return result.reset_index(drop=True)


def load_cohort(expression_root: Path, assignments: pd.DataFrame) -> Cohort:
    needed = {
        tissue: set(
            assignments.loc[assignments["Tissue"].eq(tissue), "Gene Symbol"].astype(str)
        )
        for tissue in TISSUES
    }
    tissue_frames: dict[str, pd.DataFrame] = {}
    donor_diagnosis: dict[str, str] = {}
    for tissue in TISSUES:
        parts: list[pd.DataFrame] = []
        for group in GROUPS:
            path = expression_path(expression_root, tissue, group)
            header = pd.read_csv(path, nrows=0).columns.tolist()
            donor_column = header[0]
            missing_genes = needed[tissue].difference(header[1:])
            if missing_genes:
                raise ValueError(
                    f"{tissue}/{group} lacks {len(missing_genes)} assigned genes"
                )
            selected = {donor_column, *needed[tissue]}
            frame = pd.read_csv(path, usecols=lambda column: column in selected)
            frame = frame.rename(columns={donor_column: "donor"})
            frame["donor"] = frame["donor"].astype(str)
            if frame["donor"].duplicated().any():
                raise ValueError(f"Duplicate donor rows in {path}")
            for donor in frame["donor"]:
                previous = donor_diagnosis.setdefault(donor, group)
                if previous != group:
                    raise ValueError(f"Donor {donor} occurs in {previous} and {group}")
            parts.append(frame.set_index("donor"))
        tissue_frame = pd.concat(parts, axis=0, verify_integrity=True)
        tissue_frames[tissue] = tissue_frame

    common = sorted(set.intersection(*(set(frame.index) for frame in tissue_frames.values())))
    if len(common) != 357:
        raise ValueError(f"Expected 357 complete-tissue donors, found {len(common)}")
    diagnoses = tuple(donor_diagnosis[donor] for donor in common)
    expected_groups = {"Control": 120, "MCI": 82, "AD": 155}
    observed_groups = pd.Series(diagnoses).value_counts().to_dict()
    if observed_groups != expected_groups:
        raise ValueError(f"Unexpected diagnosis counts: {observed_groups}")

    matrices: dict[str, np.ndarray] = {}
    genes: dict[str, tuple[str, ...]] = {}
    gene_indices: dict[str, dict[str, int]] = {}
    for tissue, frame in tissue_frames.items():
        ordered_genes = tuple(sorted(needed[tissue]))
        matrix = frame.loc[common, ordered_genes].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite expression values in {tissue}")
        matrices[tissue] = matrix
        genes[tissue] = ordered_genes
        gene_indices[tissue] = {gene: index for index, gene in enumerate(ordered_genes)}
    return Cohort(
        donors=tuple(common),
        projids=tuple(donor_to_projid(donor) for donor in common),
        diagnoses=diagnoses,
        matrices=matrices,
        genes=genes,
        gene_indices=gene_indices,
    )


def module_matrix(
    cohort: Cohort, assignments: pd.DataFrame, module: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = assignments.loc[assignments["Cluster ID"].eq(module)]
    arrays: list[np.ndarray] = []
    tissues: list[str] = []
    genes: list[str] = []
    for tissue in TISSUES:
        selected = rows.loc[rows["Tissue"].eq(tissue), "Gene Symbol"].astype(str).tolist()
        if not selected:
            continue
        indices = [cohort.gene_indices[tissue][gene] for gene in selected]
        arrays.append(cohort.matrices[tissue][:, indices])
        tissues.extend([tissue] * len(indices))
        genes.extend(selected)
    if not arrays:
        raise ValueError(f"Module {module} has no mapped expression genes")
    return np.concatenate(arrays, axis=1), np.asarray(tissues), tuple(genes)


def reference_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    centered = matrix - means
    return means, centered.T @ centered


def edge_metadata(tissues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    upper_i, upper_j = np.triu_indices(len(tissues), k=1)
    powers = np.empty(len(upper_i), dtype=np.float64)
    components: dict[str, np.ndarray] = {}
    component_codes = np.empty(len(upper_i), dtype=np.int8)
    resolved_to_code = {name: index for index, name in enumerate(RESOLVED_COMPONENTS)}
    for index, (left, right) in enumerate(zip(tissues[upper_i], tissues[upper_j], strict=True)):
        if left == right:
            name = f"TS_{left}"
            powers[index] = BETAS_TS[str(left)]
        else:
            pair = pair_key(str(left), str(right))
            name = f"CT_{pair.replace('-', '__')}"
            powers[index] = BETAS_CT[pair]
        component_codes[index] = resolved_to_code[name]
    for name, code in resolved_to_code.items():
        components[name] = np.flatnonzero(component_codes == code)
    return upper_i, upper_j, powers, components


def correlation_edges(
    centered_ss: np.ndarray, upper_i: np.ndarray, upper_j: np.ndarray
) -> np.ndarray:
    diagonal = np.maximum(np.diag(centered_ss), 0.0)
    denominator = np.sqrt(diagonal[upper_i] * diagonal[upper_j])
    result = np.divide(
        centered_ss[upper_i, upper_j],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return np.clip(result, -1.0, 1.0)


def signed_power(correlation: np.ndarray, powers: np.ndarray) -> np.ndarray:
    return np.sign(correlation) * np.power(np.abs(correlation), powers)


def updated_correlation_edges(
    centered_ss: np.ndarray,
    mean: np.ndarray,
    donor_values: np.ndarray,
    reference_n: int,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    operation: str,
) -> np.ndarray:
    delta = donor_values - mean
    if operation == "remove":
        factor = reference_n / (reference_n - 1.0)
        sign = -1.0
    elif operation == "add":
        factor = reference_n / (reference_n + 1.0)
        sign = 1.0
    else:
        raise ValueError(operation)
    diagonal = np.diag(centered_ss)[None, :] + sign * factor * delta * delta
    numerator = (
        centered_ss[upper_i, upper_j][None, :]
        + sign * factor * delta[:, upper_i] * delta[:, upper_j]
    )
    denominator = np.sqrt(
        np.maximum(diagonal[:, upper_i], 0.0)
        * np.maximum(diagonal[:, upper_j], 0.0)
    )
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.clip(result, -1.0, 1.0)


def empty_summary(n_donors: int) -> dict[str, np.ndarray]:
    return {
        "signed": np.zeros(n_donors),
        "absolute": np.zeros(n_donors),
        "positive": np.zeros(n_donors),
        "negative": np.zeros(n_donors),
        "positive_n": np.zeros(n_donors, dtype=np.int64),
        "negative_n": np.zeros(n_donors, dtype=np.int64),
        "possible_n": np.zeros(n_donors, dtype=np.int64),
    }


def add_summary(target: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> None:
    for field in target:
        target[field] += source[field]


def summarize_edges(weights: np.ndarray) -> dict[str, np.ndarray]:
    positive = np.where(weights > 0, weights, 0.0)
    negative = np.where(weights < 0, -weights, 0.0)
    n_donors, n_edges = weights.shape
    return {
        "signed": weights.sum(axis=1),
        "absolute": np.abs(weights).sum(axis=1),
        "positive": positive.sum(axis=1),
        "negative": negative.sum(axis=1),
        "positive_n": (weights > 0).sum(axis=1),
        "negative_n": (weights < 0).sum(axis=1),
        "possible_n": np.full(n_donors, n_edges, dtype=np.int64),
    }


def feature_columns(summary: dict[str, np.ndarray], module_size: int) -> dict[str, np.ndarray]:
    possible = summary["possible_n"]
    valid = possible > 0
    positive_density = np.divide(
        summary["positive"], summary["positive_n"],
        out=np.zeros_like(summary["positive"]), where=summary["positive_n"] > 0,
    )
    negative_density = np.divide(
        -summary["negative"], summary["negative_n"],
        out=np.zeros_like(summary["negative"]), where=summary["negative_n"] > 0,
    )
    result = {
        "connectivity": 2.0 * summary["signed"] / float(module_size),
        "positive_density": positive_density,
        "negative_density": negative_density,
        "abs_sum": summary["absolute"],
        "positive_abs_sum": summary["positive"],
        "negative_abs_sum": summary["negative"],
    }
    for values in result.values():
        values[~valid] = np.nan
    return result


def method_weights(
    matrix: np.ndarray,
    diagnoses: np.ndarray,
    method: str,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    powers: np.ndarray,
    chunk_size: int,
):
    n_donors = len(matrix)
    if method == "standard":
        reference_rows = np.arange(n_donors)
    elif method == "control_anchored":
        reference_rows = np.flatnonzero(diagnoses == "Control")
    else:
        raise ValueError(method)
    reference = matrix[reference_rows]
    reference_n = len(reference)
    mean, centered_ss = reference_statistics(reference)
    reference_correlation = correlation_edges(centered_ss, upper_i, upper_j)
    reference_adjacency = signed_power(reference_correlation, powers)
    is_reference = np.zeros(n_donors, dtype=bool)
    is_reference[reference_rows] = True

    for start in range(0, n_donors, chunk_size):
        stop = min(start + chunk_size, n_donors)
        donor_indices = np.arange(start, stop)
        weights = np.empty((len(donor_indices), len(upper_i)), dtype=np.float64)
        remove_mask = is_reference[donor_indices]
        if remove_mask.any():
            correlations = updated_correlation_edges(
                centered_ss, mean, matrix[donor_indices[remove_mask]], reference_n,
                upper_i, upper_j, "remove",
            )
            adjacency = signed_power(correlations, powers)
            weights[remove_mask] = (
                reference_n * reference_adjacency
                - (reference_n - 1.0) * adjacency
            )
        if (~remove_mask).any():
            correlations = updated_correlation_edges(
                centered_ss, mean, matrix[donor_indices[~remove_mask]], reference_n,
                upper_i, upper_j, "add",
            )
            adjacency = signed_power(correlations, powers)
            weights[~remove_mask] = (
                (reference_n + 1.0) * adjacency
                - reference_n * reference_adjacency
            )
        yield donor_indices, weights


def compute_module_method(
    cohort: Cohort,
    assignments: pd.DataFrame,
    module: int,
    method: str,
    chunk_size: int,
) -> pd.DataFrame:
    matrix, tissues, _ = module_matrix(cohort, assignments, module)
    upper_i, upper_j, powers, component_edges = edge_metadata(tissues)
    summaries = {name: empty_summary(len(matrix)) for name in RESOLVED_COMPONENTS}
    diagnoses = np.asarray(cohort.diagnoses)
    for donor_indices, weights in method_weights(
        matrix, diagnoses, method, upper_i, upper_j, powers, chunk_size
    ):
        for component, edges in component_edges.items():
            block = summarize_edges(weights[:, edges])
            for field, values in block.items():
                summaries[component][field][donor_indices] = values

    pooled = {
        "TS": empty_summary(len(matrix)),
        "CT": empty_summary(len(matrix)),
        "all": empty_summary(len(matrix)),
    }
    for component, summary in summaries.items():
        add_summary(pooled["TS" if component.startswith("TS_") else "CT"], summary)
        add_summary(pooled["all"], summary)

    frame = pd.DataFrame({
        "lioness_method": method,
        "donor": cohort.donors,
        "projid": cohort.projids,
        "diagnosis_group": cohort.diagnoses,
        "module": module,
    })
    for component in ALL_COMPONENTS:
        summary = pooled[component] if component in pooled else summaries[component]
        features = feature_columns(summary, matrix.shape[1])
        for family, values in features.items():
            frame[f"{family}_{component}"] = values
    return frame


def validate_feature_identities(frame: pd.DataFrame, tolerance: float = 1e-10) -> None:
    for component in ALL_COMPONENTS:
        absolute = frame[f"abs_sum_{component}"].to_numpy()
        signed_parts = (
            frame[f"positive_abs_sum_{component}"].to_numpy()
            + frame[f"negative_abs_sum_{component}"].to_numpy()
        )
        valid = np.isfinite(absolute) & np.isfinite(signed_parts)
        if valid.any() and not np.allclose(
            absolute[valid], signed_parts[valid], rtol=tolerance, atol=tolerance
        ):
            raise AssertionError(f"Absolute-sum identity failed for {component}")
    for family in ("connectivity", "abs_sum", "positive_abs_sum", "negative_abs_sum"):
        for pooled, members in {
            "TS": ("TS_VH", "TS_SMA", "TS_QUAD"),
            "CT": ("CT_VH__SMA", "CT_VH__QUAD", "CT_SMA__QUAD"),
        }.items():
            observed = frame[f"{family}_{pooled}"].to_numpy()
            parts = frame[[f"{family}_{member}" for member in members]].to_numpy()
            if family == "connectivity":
                expected = np.nansum(parts, axis=1)
            else:
                expected = np.nansum(parts, axis=1)
            valid = np.isfinite(observed)
            if valid.any() and not np.allclose(
                observed[valid], expected[valid], rtol=tolerance, atol=tolerance
            ):
                raise AssertionError(f"Pooled identity failed for {family}/{pooled}")


def expected_columns() -> list[str]:
    return [
        "lioness_method", "donor", "projid", "diagnosis_group", "module",
        *[
            f"{family}_{component}"
            for component in ALL_COMPONENTS
            for family in FEATURE_FAMILIES
        ],
    ]


def checkpoint_valid(path: Path, module: int, method: str, cohort: Cohort) -> bool:
    try:
        frame = pq.read_table(path).to_pandas()
    except Exception:
        return False
    return (
        frame.columns.tolist() == expected_columns()
        and len(frame) == len(cohort.donors)
        and frame["module"].eq(module).all()
        and frame["lioness_method"].eq(method).all()
        and frame["donor"].astype(str).tolist() == list(cohort.donors)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--expression-root", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-module-size", type=int, default=30)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--module", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments = read_assignments(args.assignments, args.minimum_module_size)
    available_modules = sorted(assignments["Cluster ID"].unique().tolist())
    modules = args.module or available_modules
    unknown = sorted(set(modules).difference(available_modules))
    if unknown:
        raise ValueError(f"Unknown/filtered module IDs: {unknown}")
    cohort = load_cohort(args.expression_root, assignments)
    output = args.output.resolve()
    checkpoint_root = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    input_paths = [args.assignments] + [
        expression_path(args.expression_root, tissue, group)
        for tissue in TISSUES for group in GROUPS
    ]
    manifest = {
        "analysis": "muscle_control_derived_l4_lioness_nonprediction",
        "module_partition_network": (
            "Control-only Method 5 extended SFN (bootstrap stability, CT null "
            "calibration, and per-block q90 scaling)"
        ),
        "lioness_adjacency_functional": (
            "Pearson signedAlt with frozen Control-derived block powers; the "
            "Method-5 bootstrap/null/q90 functional is used for MDC group "
            "networks, not inside donor leave-one-out/add-one scoring"
        ),
        "status": "running",
        "started_at_utc": utc_now(),
        "assignment_level": "table_3 (zero-indexed; biological subcluster 4)",
        "minimum_module_size": args.minimum_module_size,
        "module_count": len(available_modules),
        "assignment_rows": len(assignments),
        "donor_count": len(cohort.donors),
        "diagnosis_counts": pd.Series(cohort.diagnoses).value_counts().to_dict(),
        "tissues": list(TISSUES),
        "beta_ts": BETAS_TS,
        "beta_ct": BETAS_CT,
        "methods": args.methods,
        "feature_families": list(FEATURE_FAMILIES),
        "components": list(ALL_COMPONENTS),
        "input_files": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "completed": {},
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text())
        manifest["started_at_utc"] = previous.get("started_at_utc", utc_now())
    atomic_json(manifest, manifest_path)

    tasks: list[tuple[str, int, Path]] = []
    for method in args.methods:
        for module in modules:
            path = checkpoint_root / method / f"M{module}.parquet"
            if not args.force and path.exists() and checkpoint_valid(
                path, module, method, cohort
            ):
                manifest["completed"][f"{method}:M{module}"] = "checkpoint"
            else:
                tasks.append((method, module, path))

    def run_task(task: tuple[str, int, Path]) -> tuple[str, int, Path]:
        method, module, path = task
        frame = compute_module_method(
            cohort, assignments, module, method, args.chunk_size
        )
        frame = frame[expected_columns()]
        validate_feature_identities(frame)
        atomic_parquet(frame, path)
        return method, module, path

    completed_now = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            method, module, _ = future.result()
            completed_now += 1
            manifest["completed"][f"{method}:M{module}"] = "computed"
            if completed_now % 5 == 0 or completed_now == len(tasks):
                manifest["last_update_utc"] = utc_now()
                atomic_json(manifest, manifest_path)
                print(
                    f"Completed {completed_now}/{len(tasks)} new checkpoints "
                    f"({method} M{module})",
                    flush=True,
                )

    full_run = set(modules) == set(available_modules)
    combined_paths: dict[str, str] = {}
    if full_run:
        for method in args.methods:
            frames = [
                pq.read_table(checkpoint_root / method / f"M{module}.parquet").to_pandas()
                for module in available_modules
            ]
            combined = pd.concat(frames, ignore_index=True)
            expected_rows = len(available_modules) * len(cohort.donors)
            if len(combined) != expected_rows:
                raise AssertionError(f"{method}: {len(combined)} != {expected_rows}")
            destination = output / method / "donor_module_metrics.parquet"
            atomic_parquet(combined, destination)
            combined_paths[method] = str(destination)

    manifest.update({
        "status": "complete" if full_run else "partial",
        "finished_at_utc": utc_now(),
        "new_checkpoints": completed_now,
        "expected_checkpoint_count": len(available_modules) * len(args.methods),
        "combined_outputs": combined_paths,
    })
    atomic_json(manifest, manifest_path)
    print(json.dumps({
        "status": manifest["status"],
        "modules": len(modules),
        "donors": len(cohort.donors),
        "new_checkpoints": completed_now,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
