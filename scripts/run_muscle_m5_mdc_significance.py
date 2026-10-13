#!/usr/bin/env python3
"""Calculate resumable sample/gene permutation significance for muscle M5 MDC.

The observed muscle MDC is based on the full Method-5 recovered-module
functional.  This runner preserves that functional in both nulls:

* sample null: independently permute donor values within every gene and group,
  then rebuild CT null excess, bootstrap stability, signed powers, and the
  recovered-edge block-q90 normalizers;
* gene null: sample tissue-specific genes while preserving every module's
  VH/SMA/QUAD counts, then rebuild bootstrap stability, signed powers, and
  recovered-edge block-q90 normalizers using the observed group CT nulls.

Each permutation is checkpointed independently.  Directional BH is calculated
separately by component and direction across stable, structurally available
modules.  Components with fewer than ten edges or either group mean absolute
weight at or below 1e-8 remain descriptive but are excluded from inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from .run_muscle_m5_mdc import (
        BLOCKS,
        DEFAULT_ASSIGNMENTS,
        DEFAULT_EXPRESSION,
        DEFAULT_OUTPUT,
        SCOPES,
        TISSUES,
        _block_values,
        _group_null_statistics,
        _raw_module_m5,
        atomic_json,
        load_group,
        read_filtered_assignments,
        stable_seed,
        summarize_module,
    )
except ImportError:  # Direct script execution.
    from run_muscle_m5_mdc import (
        BLOCKS,
        DEFAULT_ASSIGNMENTS,
        DEFAULT_EXPRESSION,
        DEFAULT_OUTPUT,
        SCOPES,
        TISSUES,
        _block_values,
        _group_null_statistics,
        _raw_module_m5,
        atomic_json,
        load_group,
        read_filtered_assignments,
        stable_seed,
        summarize_module,
    )


DEFAULT_SIGNIFICANCE = DEFAULT_OUTPUT / "significance_200perms"
MIN_INFERENCE_EDGES = 10
MIN_GROUP_MEAN_ABS_WEIGHT = 1e-8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna() & np.isfinite(values)
    if not valid.any():
        return result
    p = values.loc[valid].to_numpy(float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid] = restored
    return result


def empirical_upper(observed: float, null: np.ndarray) -> float:
    finite = null[np.isfinite(null)]
    return float((1 + np.sum(finite >= observed)) / (1 + len(finite)))


def empirical_lower(observed: float, null: np.ndarray) -> float:
    finite = null[np.isfinite(null)]
    return float((1 + np.sum(finite <= observed)) / (1 + len(finite)))


def independently_permute_genes(
    tissues: dict[str, pd.DataFrame], kind: str, permutation: int, seed: int
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for tissue, frame in tissues.items():
        values = frame.to_numpy(dtype=np.float32, copy=True)
        rng = np.random.default_rng(
            stable_seed("muscle-m5-mdc-sample-null", kind, tissue, permutation, seed)
        )
        ordering = np.argsort(rng.random(values.shape, dtype=np.float32), axis=1)
        values = np.take_along_axis(values, ordering, axis=1)
        result[tissue] = pd.DataFrame(
            values, index=frame.index.copy(), columns=frame.columns.copy()
        )
    return result


def sampled_assignments(
    assignments: pd.DataFrame,
    common_genes: dict[str, np.ndarray],
    permutation: int,
    seed: int,
) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(
        stable_seed("muscle-m5-mdc-gene-null", permutation, seed)
    )
    result: dict[int, pd.DataFrame] = {}
    for module, rows in assignments.groupby("Cluster ID", sort=True, observed=True):
        pieces: list[pd.DataFrame] = []
        for tissue in TISSUES:
            count = int(rows["Tissue"].eq(tissue).sum())
            if not count:
                continue
            pool = common_genes[tissue]
            if count > len(pool):
                raise ValueError(f"M{module}/{tissue}: requested more genes than available")
            chosen = rng.choice(pool, size=count, replace=False)
            pieces.append(
                pd.DataFrame(
                    {
                        "Cluster ID": int(module),
                        "Tissue": tissue,
                        "Gene ID": chosen.astype(str),
                        "Gene Symbol": chosen.astype(str),
                    }
                )
            )
        result[int(module)] = pd.concat(pieces, ignore_index=True)
    return result


def original_assignments(assignments: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(module): rows.copy()
        for module, rows in assignments.groupby(
            "Cluster ID", sort=True, observed=True
        )
    }


def observed_ct_nulls(output_root: Path, group: str) -> dict[str, dict[str, float | int]]:
    path = output_root / "recovered_modules" / group / "ct_null_statistics.json"
    if not path.exists():
        raise FileNotFoundError(f"Observed M5 CT-null checkpoint is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_group_modules(
    rows_by_module: dict[int, pd.DataFrame],
    tissues: dict[str, pd.DataFrame],
    group_tag: str,
    nulls: dict[str, dict[str, float | int]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    raw: dict[int, np.ndarray] = {}
    node_tissues: dict[int, np.ndarray] = {}
    block_parts: dict[str, list[np.ndarray]] = {
        f"{left}-{right}": [] for left, right in BLOCKS
    }
    for module, rows in rows_by_module.items():
        weights, labels, _ = _raw_module_m5(
            module, rows, tissues, group_tag, nulls
        )
        raw[module] = weights
        node_tissues[module] = labels
        for left, right in BLOCKS:
            values = _block_values(weights, labels, left, right)
            nonzero = np.abs(values[values != 0])
            if len(nonzero):
                block_parts[f"{left}-{right}"].append(nonzero)
    q90: dict[str, float] = {}
    for left, right in BLOCKS:
        key = f"{left}-{right}"
        values = np.concatenate(block_parts[key]) if block_parts[key] else np.empty(0)
        q90[key] = float(np.quantile(values, 0.90)) if len(values) else 0.0
    normalized: dict[int, np.ndarray] = {}
    for module, weights in raw.items():
        weights = weights.astype(np.float32, copy=True)
        labels = node_tissues[module]
        for left, right in BLOCKS:
            scale = q90[f"{left}-{right}"]
            left_positions = np.flatnonzero(labels == left)
            right_positions = np.flatnonzero(labels == right)
            if scale <= 0 or not len(left_positions) or not len(right_positions):
                continue
            if left == right:
                index = np.ix_(left_positions, left_positions)
                weights[index] = np.clip(weights[index] / scale, -10.0, 10.0)
            else:
                block = np.clip(
                    weights[np.ix_(left_positions, right_positions)] / scale,
                    -10.0,
                    10.0,
                )
                weights[np.ix_(left_positions, right_positions)] = block
                weights[np.ix_(right_positions, left_positions)] = block.T
        normalized[module] = weights
    return normalized, node_tissues


def calculate_null_permutation(
    kind: str,
    permutation: int,
    seed: int,
    expression_root: Path,
    assignments_path: Path,
    observed_output: Path,
    checkpoint: Path,
) -> dict[str, object]:
    if checkpoint.exists():
        return {"kind": kind, "permutation": permutation, "status": "reused"}
    started = time.monotonic()
    assignments = read_filtered_assignments(assignments_path)
    control, _, _ = load_group(expression_root, "Control")
    ad, _, _ = load_group(expression_root, "AD")
    common_genes = {
        tissue: np.asarray(
            sorted(set(control[tissue].index).intersection(ad[tissue].index)),
            dtype=str,
        )
        for tissue in TISSUES
    }
    if kind == "sample":
        rows_by_module = original_assignments(assignments)
        control_input = independently_permute_genes(
            control, "control", permutation, seed
        )
        ad_input = independently_permute_genes(ad, "ad", permutation, seed)
        control_tag = f"control_sample_{permutation}_{seed}"
        ad_tag = f"ad_sample_{permutation}_{seed}"
        control_nulls = _group_null_statistics(control_input, control_tag)
        ad_nulls = _group_null_statistics(ad_input, ad_tag)
    elif kind == "gene":
        rows_by_module = sampled_assignments(
            assignments, common_genes, permutation, seed
        )
        control_input = control
        ad_input = ad
        control_tag = f"control_gene_{permutation}_{seed}"
        ad_tag = f"ad_gene_{permutation}_{seed}"
        control_nulls = observed_ct_nulls(observed_output, "control")
        ad_nulls = observed_ct_nulls(observed_output, "ad")
    else:
        raise ValueError(kind)
    control_networks, control_tissues = build_group_modules(
        rows_by_module, control_input, control_tag, control_nulls
    )
    ad_networks, ad_tissues = build_group_modules(
        rows_by_module, ad_input, ad_tag, ad_nulls
    )
    modules = np.asarray(sorted(rows_by_module), dtype=np.int64)
    values = np.full((len(modules), len(SCOPES)), np.nan, dtype=np.float64)
    for row_index, module in enumerate(modules):
        if not np.array_equal(ad_tissues[int(module)], control_tissues[int(module)]):
            raise ValueError(f"M{module}: null group node order differs")
        records = summarize_module(
            int(module),
            np.arange(len(ad_tissues[int(module)])),
            ad_tissues[int(module)],
            ad_networks[int(module)],
            control_networks[int(module)],
        )
        by_scope = {record["scope"]: record["mdc_raw"] for record in records}
        values[row_index] = [by_scope[scope] for scope in SCOPES]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            modules=modules,
            scopes=np.asarray(SCOPES),
            mdc=values,
            kind=kind,
            permutation=permutation,
            seed=seed,
        )
    os.replace(temporary, checkpoint)
    return {
        "kind": kind,
        "permutation": permutation,
        "status": "computed",
        "seconds": round(time.monotonic() - started, 3),
    }


def load_null(checkpoint_dir: Path, kind: str, permutations: int) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    modules: np.ndarray | None = None
    for permutation in range(1, permutations + 1):
        path = checkpoint_dir / kind / f"perm_{permutation:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        if modules is None:
            modules = data["modules"]
        elif not np.array_equal(modules, data["modules"]):
            raise ValueError(f"{kind}: module order differs between checkpoints")
        if tuple(data["scopes"].astype(str)) != SCOPES:
            raise ValueError(f"{kind}: scope order differs")
        arrays.append(data["mdc"])
    if modules is None:
        raise ValueError(f"{kind}: no checkpoints")
    return modules, np.stack(arrays, axis=2)


def build_significance(
    observed_path: Path,
    checkpoint_dir: Path,
    sample_permutations: int,
    gene_permutations: int,
) -> pd.DataFrame:
    observed = pd.read_parquet(observed_path).copy()
    sample_modules, sample_null = load_null(
        checkpoint_dir, "sample", sample_permutations
    )
    gene_modules, gene_null = load_null(
        checkpoint_dir, "gene", gene_permutations
    )
    if not np.array_equal(sample_modules, gene_modules):
        raise ValueError("Sample/gene null module order differs")
    return build_significance_from_arrays(
        observed, sample_modules, sample_null, gene_null,
        sample_permutations, gene_permutations,
    )


def build_significance_from_arrays(
    observed: pd.DataFrame,
    modules: np.ndarray,
    sample_null: np.ndarray,
    gene_null: np.ndarray,
    sample_permutations: int,
    gene_permutations: int,
) -> pd.DataFrame:
    """Use the identical inference rule for each separately computed functional."""
    observed = observed.copy()
    if sample_null.shape != (len(modules), len(SCOPES), sample_permutations):
        raise ValueError("Sample-null dimensions differ from the requested family")
    if gene_null.shape != (len(modules), len(SCOPES), gene_permutations):
        raise ValueError("Gene-null dimensions differ from the requested family")
    if set(observed["module"].astype(int)) != set(modules.astype(int)):
        raise ValueError("Observed/null module membership differs")
    lookup = {int(module): index for index, module in enumerate(modules)}
    scope_lookup = {scope: index for index, scope in enumerate(SCOPES)}
    stable = (
        observed["n_edges"].ge(MIN_INFERENCE_EDGES)
        & observed["mean_abs_ad"].gt(MIN_GROUP_MEAN_ABS_WEIGHT)
        & observed["mean_abs_control"].gt(MIN_GROUP_MEAN_ABS_WEIGHT)
        & observed["mdc_raw"].notna()
        & np.isfinite(observed["mdc_raw"])
    )
    observed["inference_available"] = stable
    observed["inference_unavailable_reason"] = ""
    observed.loc[observed["n_edges"].lt(MIN_INFERENCE_EDGES), "inference_unavailable_reason"] = (
        f"fewer than {MIN_INFERENCE_EDGES} edges"
    )
    near_zero = (
        observed["mean_abs_ad"].le(MIN_GROUP_MEAN_ABS_WEIGHT)
        | observed["mean_abs_control"].le(MIN_GROUP_MEAN_ABS_WEIGHT)
    )
    observed.loc[near_zero, "inference_unavailable_reason"] = (
        "near-zero AD or Control mean absolute weight"
    )
    observed.loc[observed["mdc_raw"].isna(), "inference_unavailable_reason"] = (
        "structurally unavailable or zero denominator"
    )
    for column in (
        "p_loss_sample", "p_loss_gene", "p_gain_sample", "p_gain_gene"
    ):
        observed[column] = np.nan
    observed["sample_null_finite_n"] = 0
    observed["gene_null_finite_n"] = 0
    for index, row in observed.loc[stable].iterrows():
        module_index = lookup[int(row["module"])]
        scope_index = scope_lookup[str(row["scope"])]
        value = float(row["mdc_raw"])
        sample_values = sample_null[module_index, scope_index]
        gene_values = gene_null[module_index, scope_index]
        observed.loc[index, "sample_null_finite_n"] = int(
            np.isfinite(sample_values).sum()
        )
        observed.loc[index, "gene_null_finite_n"] = int(
            np.isfinite(gene_values).sum()
        )
        observed.loc[index, "p_loss_sample"] = empirical_upper(value, sample_values)
        observed.loc[index, "p_loss_gene"] = empirical_upper(value, gene_values)
        observed.loc[index, "p_gain_sample"] = empirical_lower(value, sample_values)
        observed.loc[index, "p_gain_gene"] = empirical_lower(value, gene_values)
    insufficient_null = observed["inference_available"] & (
        observed["sample_null_finite_n"].lt(math.ceil(0.90 * sample_permutations))
        | observed["gene_null_finite_n"].lt(math.ceil(0.90 * gene_permutations))
    )
    observed.loc[insufficient_null, "inference_available"] = False
    observed.loc[insufficient_null, "inference_unavailable_reason"] = (
        "fewer than 90% finite sample-null or gene-null permutations"
    )
    observed.loc[insufficient_null, [
        "p_loss_sample", "p_loss_gene", "p_gain_sample", "p_gain_gene"
    ]] = np.nan
    for scope in SCOPES:
        scope_rows = observed["scope"].eq(scope) & observed["inference_available"]
        for direction in ("loss", "gain"):
            for null in ("sample", "gene"):
                p_column = f"p_{direction}_{null}"
                q_column = f"q_{direction}_{null}"
                observed.loc[scope_rows, q_column] = bh(
                    observed.loc[scope_rows, p_column]
                )
    observed["q_loss_max"] = observed[["q_loss_sample", "q_loss_gene"]].max(
        axis=1, skipna=False
    )
    observed["q_gain_max"] = observed[["q_gain_sample", "q_gain_gene"]].max(
        axis=1, skipna=False
    )
    observed["direction"] = np.select(
        [observed["mdc_raw"].gt(1), observed["mdc_raw"].lt(1)],
        ["Higher in AD", "Higher in Control"],
        default="Equal",
    )
    observed["directional_p_sample"] = np.where(
        observed["mdc_raw"].ge(1),
        observed["p_loss_sample"], observed["p_gain_sample"],
    )
    observed["directional_p_gene"] = np.where(
        observed["mdc_raw"].ge(1),
        observed["p_loss_gene"], observed["p_gain_gene"],
    )
    observed["directional_fdr"] = np.where(
        observed["mdc_raw"].ge(1), observed["q_loss_max"], observed["q_gain_max"]
    )
    observed.loc[~observed["inference_available"], [
        "directional_p_sample", "directional_p_gene", "directional_fdr"
    ]] = np.nan
    observed["significant_fdr_05"] = observed["directional_fdr"].lt(0.05)
    observed["significant_fdr_10"] = observed["directional_fdr"].lt(0.10)
    observed["sample_permutations"] = int(sample_permutations)
    observed["gene_permutations"] = int(gene_permutations)
    return observed.sort_values(
        ["module", "scope"],
        key=lambda values: (
            values.map(scope_lookup) if values.name == "scope" else values
        ),
        kind="stable",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-root", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--observed-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SIGNIFICANCE)
    parser.add_argument("--sample-permutations", type=int, default=200)
    parser.add_argument("--gene-permutations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kind", choices=("all", "sample", "gene", "summarize"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_permutations < 1 or args.gene_permutations < 1:
        raise ValueError("Permutation counts must be positive")
    output = args.output.resolve()
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    kinds = (
        ("sample", args.sample_permutations),
        ("gene", args.gene_permutations),
    )
    if args.kind in {"sample", "gene"}:
        kinds = tuple(item for item in kinds if item[0] == args.kind)
    manifest = {
        "status": "running",
        "updated_utc": utc_now(),
        "sample_permutations": args.sample_permutations,
        "gene_permutations": args.gene_permutations,
        "seed": args.seed,
        "workers": args.workers,
        "minimum_inference_edges": MIN_INFERENCE_EDGES,
        "minimum_group_mean_absolute_weight": MIN_GROUP_MEAN_ABS_WEIGHT,
        "sample_null": (
            "independent within-gene donor permutations; rebuilds M5 CT nulls, "
            "bootstrap stability, signed powers, and recovered-edge block q90"
        ),
        "gene_null": (
            "tissue-count-preserving gene sampling; rebuilds M5 bootstrap stability, "
            "signed powers, and recovered-edge block q90"
        ),
        "directional_fdr": (
            "maximum of sample-null and gene-null BH values, corrected separately "
            "by component and direction across inference-available modules"
        ),
    }
    atomic_json(manifest, manifest_path)
    results: list[dict[str, object]] = []
    if args.kind != "summarize":
        tasks = []
        for kind, permutations in kinds:
            for permutation in range(1, permutations + 1):
                checkpoint = checkpoint_dir / kind / f"perm_{permutation:03d}.npz"
                if not checkpoint.exists():
                    tasks.append((kind, permutation, checkpoint))
        if tasks:
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        calculate_null_permutation,
                        kind,
                        permutation,
                        args.seed,
                        args.expression_root.resolve(),
                        args.assignments.resolve(),
                        args.observed_output.resolve(),
                        checkpoint,
                    ): (kind, permutation)
                    for kind, permutation, checkpoint in tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
    complete = all(
        (checkpoint_dir / kind / f"perm_{permutation:03d}.npz").exists()
        for kind, permutations in (
            ("sample", args.sample_permutations),
            ("gene", args.gene_permutations),
        )
        for permutation in range(1, permutations + 1)
    )
    if complete:
        significance = build_significance(
            args.observed_output / "mdc_observed.parquet",
            checkpoint_dir,
            args.sample_permutations,
            args.gene_permutations,
        )
        destination = output / "mdc_significance.parquet"
        temporary = destination.with_name(destination.name + ".partial")
        significance.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, destination)
        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "rows": len(significance),
                "modules": int(significance["module"].nunique()),
                "scopes": list(SCOPES),
                "inference_available_rows": int(
                    significance["inference_available"].sum()
                ),
                "significant_fdr_05_rows": int(
                    significance["significant_fdr_05"].sum()
                ),
                "significant_fdr_10_rows": int(
                    significance["significant_fdr_10"].sum()
                ),
            }
        )
    else:
        manifest["status"] = "partial"
    manifest["updated_utc"] = utc_now()
    manifest["completed_sample_checkpoints"] = len(
        list((checkpoint_dir / "sample").glob("perm_*.npz"))
    )
    manifest["completed_gene_checkpoints"] = len(
        list((checkpoint_dir / "gene").glob("perm_*.npz"))
    )
    manifest["new_results"] = results
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
