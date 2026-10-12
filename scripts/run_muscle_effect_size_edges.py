#!/usr/bin/env python3
"""Build Hedges-g-filtered LIONESS edge summaries for VH/SMA/QUAD.

The edge mask is learned only from a deterministic diagnosis-by-sex stratified
discovery split.  MCI and held-out validation donors do not participate in mask
construction.  Donor-specific edge matrices are streamed and never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = APP_ROOT.parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.bonobo.optimized import EdgeScopeContext  # noqa: E402
from models.differential_edges import (  # noqa: E402
    EFFECT_DIRECTIONS,
    EffectMaskSpec,
    StreamingMoments,
    effect_rule_key,
    effect_size_mask,
    summarize_edge_network,
    welch_hedges_from_moments,
)
from run_muscle_lioness import (  # noqa: E402
    DEFAULT_ASSIGNMENTS,
    DEFAULT_EXPRESSION,
    METHODS,
    RESOLVED_COMPONENTS,
    TISSUES,
    edge_metadata,
    load_cohort,
    method_weights,
    module_matrix,
    read_assignments,
    sha256_file,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "out/lioness_app_expansion"
    / "20261012_muscle_effect_size_edge_filtering"
)
MUSCLE_METADATA = (
    REPO_ROOT
    / "data/external/muscle_samples_with_clinical_groups_by_IID_cogdx_RADC.csv"
)
MODULE_DEFINITION = "muscle_control_l4"
SPEC = EffectMaskSpec("hedges_g", "fixed", 0.40)
TOP_DRIVER_COUNT = 20
DRIVER_DIRECTIONS = ("ad_higher", "control_higher")
DRIVER_COLUMNS = (
    "donor", "module", "edge_rule", "effect_statistic", "effect_cutoff_mode",
    "effect_cutoff_value", "effect_direction", "effect_rule_key", "driver_rank",
    "edge_index", "component", "tissue_a", "gene_a", "tissue_b", "gene_b",
    "donor_edge_weight", "control_reference_weight", "driver_absolute_deviation",
    "discovery_mean_ad", "discovery_mean_control", "discovery_mean_difference",
    "discovery_hedges_g",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), temporary,
            compression="zstd", row_group_size=50_000,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_split(cohort, metadata_path: Path, seed: int) -> pd.DataFrame:
    """Create a reproducible 70/30 AD-Control split within diagnosis and sex."""

    source = pd.read_csv(metadata_path, low_memory=False)
    source["ProjID"] = source["ProjID"].astype(str).str.replace(r"\.0$", "", regex=True)
    conflicts = source.groupby("ProjID", observed=True)["msex"].nunique(dropna=True)
    if conflicts.gt(1).any():
        raise ValueError("Conflicting sex values in muscle metadata")
    sex = (
        source.dropna(subset=["msex"]).drop_duplicates("ProjID")
        .set_index("ProjID")["msex"]
    )
    frame = pd.DataFrame(
        {
            "donor": cohort.donors,
            "projid": cohort.projids,
            "diagnosis_group": cohort.diagnoses,
        }
    )
    frame["sex_code"] = frame["projid"].map(sex)
    eligible = frame["diagnosis_group"].isin(["AD", "Control"])
    if frame.loc[eligible, "sex_code"].isna().any():
        raise ValueError("AD/Control sex is required for stratification")
    frame["ad_control_split"] = "MCI_external"
    rng = np.random.default_rng(seed)
    for _, indices in frame.loc[eligible].groupby(
        ["diagnosis_group", "sex_code"], sort=True, observed=True
    ).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        shuffled = rng.permutation(positions)
        validation_n = int(np.floor(len(shuffled) * 0.30 + 0.5))
        frame.loc[shuffled[:validation_n], "ad_control_split"] = "Validation"
        frame.loc[shuffled[validation_n:], "ad_control_split"] = "Discovery"
    counts = (
        frame.groupby(["ad_control_split", "diagnosis_group"], observed=True)
        .size().rename("n").reset_index()
    )
    if frame.loc[eligible, "ad_control_split"].eq("MCI_external").any():
        raise AssertionError("An AD/Control donor was not assigned")
    if not frame.loc[frame["diagnosis_group"].eq("MCI"), "ad_control_split"].eq(
        "MCI_external"
    ).all():
        raise AssertionError("MCI donors entered edge-mask discovery")
    frame.attrs["counts"] = counts.to_dict(orient="records")
    return frame


def scope_context(
    tissues: np.ndarray, upper_i: np.ndarray, upper_j: np.ndarray,
    component_edges: dict[str, np.ndarray],
) -> EdgeScopeContext:
    n_edges = len(upper_i)
    resolved: dict[str, np.ndarray] = {}
    for component in RESOLVED_COMPONENTS:
        mask = np.zeros(n_edges, dtype=bool)
        mask[component_edges[component]] = True
        resolved[component] = mask
    is_ts = tissues[upper_i] == tissues[upper_j]
    scopes = {
        "total": np.ones(n_edges, dtype=bool),
        "TS": is_ts,
        "CT": ~is_ts,
        **resolved,
    }
    if sum(int(mask.sum()) for key, mask in scopes.items() if key.startswith("TS_")) != int(is_ts.sum()):
        raise AssertionError("Resolved TS masks do not reproduce pooled TS")
    if sum(int(mask.sum()) for key, mask in scopes.items() if key.startswith("CT_")) != int((~is_ts).sum()):
        raise AssertionError("Resolved CT masks do not reproduce pooled CT")
    return EdgeScopeContext(
        row_index=upper_i, column_index=upper_j, is_ts=is_ts,
        scopes=scopes, n_features=len(tissues),
    )


def moments_for_split(
    matrix: np.ndarray,
    diagnoses: np.ndarray,
    split: np.ndarray,
    method: str,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    powers: np.ndarray,
    chunk_size: int,
) -> dict[str, dict[str, StreamingMoments]]:
    groups = {
        label: {
            diagnosis: StreamingMoments.empty(len(upper_i))
            for diagnosis in ("AD", "Control")
        }
        for label in ("Discovery", "Validation")
    }
    for donor_indices, weights in method_weights(
        matrix, diagnoses, method, upper_i, upper_j, powers, chunk_size
    ):
        for local, donor_index in enumerate(donor_indices):
            diagnosis = str(diagnoses[donor_index])
            split_name = str(split[donor_index])
            if diagnosis in {"AD", "Control"} and split_name in groups:
                groups[split_name][diagnosis].update(weights[local])
    return groups


def edge_statistics(
    module: int,
    method: str,
    genes: tuple[str, ...],
    tissues: np.ndarray,
    context: EdgeScopeContext,
    moments: dict[str, dict[str, StreamingMoments]],
) -> pd.DataFrame:
    component = np.full(len(context.row_index), "", dtype=object)
    for name, mask in context.scopes.items():
        if name.startswith(("TS_", "CT_")):
            component[mask] = name
    rows = {
        "module_definition": MODULE_DEFINITION,
        "estimator": "lioness",
        "network_method": method,
        "module": np.full(len(context.row_index), module, dtype=np.int64),
        "edge_index": np.arange(len(context.row_index), dtype=np.int64),
        "row_index": context.row_index.astype(np.int32),
        "column_index": context.column_index.astype(np.int32),
        "tissue_a": tissues[context.row_index].astype(str),
        "gene_a": np.asarray(genes, dtype=object)[context.row_index].astype(str),
        "tissue_b": tissues[context.column_index].astype(str),
        "gene_b": np.asarray(genes, dtype=object)[context.column_index].astype(str),
        "component": component.astype(str),
        "component_class": np.where(context.is_ts, "TS", "CT"),
    }
    frame = pd.DataFrame(rows)
    for label in ("Discovery", "Validation"):
        stats = welch_hedges_from_moments(
            moments[label]["AD"], moments[label]["Control"]
        )
        prefix = label.lower()
        for source, destination in (
            ("mean_ad", "mean_ad"), ("mean_control", "mean_control"),
            ("mean_difference", "mean_difference"), ("hedges_g", "hedges_g"),
            ("welch_t", "welch_t"), ("welch_df", "welch_df"),
            ("p_value", "p_value"), ("fdr", "fdr_per_module"),
        ):
            frame[f"{prefix}_{destination}"] = np.asarray(stats[source])
        frame[f"{prefix}_n_ad"] = int(stats["n_ad"])
        frame[f"{prefix}_n_control"] = int(stats["n_control"])
        frame[f"{prefix}_significant_fdr05_per_module"] = frame[
            f"{prefix}_fdr_per_module"
        ].lt(0.05)
    frame["validation_direction_concordant"] = np.sign(
        frame["discovery_mean_difference"]
    ).eq(np.sign(frame["validation_mean_difference"]))
    return frame


def top_driver_rows(
    donor: str,
    module: int,
    weights: np.ndarray,
    baseline: np.ndarray,
    edges: pd.DataFrame,
    mask: np.ndarray,
    direction: str,
) -> list[dict[str, object]]:
    selected = np.flatnonzero(mask)
    if not selected.size:
        return []
    deviation = np.abs(weights[selected] - baseline[selected])
    keep = min(TOP_DRIVER_COUNT, selected.size)
    if keep < selected.size:
        chosen = selected[np.argpartition(deviation, -keep)[-keep:]]
    else:
        chosen = selected
    order = np.lexsort((chosen, -np.abs(weights[chosen] - baseline[chosen])))
    rows: list[dict[str, object]] = []
    for rank, position in enumerate(chosen[order], start=1):
        edge = edges.iloc[int(position)]
        rows.append(
            {
                "donor": donor,
                "module": module,
                "edge_rule": "all",
                "effect_statistic": "hedges_g",
                "effect_cutoff_mode": "fixed",
                "effect_cutoff_value": 0.40,
                "effect_direction": direction,
                "effect_rule_key": effect_rule_key(SPEC, direction),
                "driver_rank": rank,
                "edge_index": int(edge["edge_index"]),
                "component": edge["component"],
                "tissue_a": edge["tissue_a"],
                "gene_a": edge["gene_a"],
                "tissue_b": edge["tissue_b"],
                "gene_b": edge["gene_b"],
                "donor_edge_weight": float(weights[position]),
                "control_reference_weight": float(baseline[position]),
                "driver_absolute_deviation": float(abs(weights[position] - baseline[position])),
                "discovery_mean_ad": float(edge["discovery_mean_ad"]),
                "discovery_mean_control": float(edge["discovery_mean_control"]),
                "discovery_mean_difference": float(edge["discovery_mean_difference"]),
                "discovery_hedges_g": float(edge["discovery_hedges_g"]),
            }
        )
    return rows


def output_paths(root: Path, method: str, module: int) -> dict[str, Path]:
    prefix = root / MODULE_DEFINITION / "checkpoints" / f"lioness__{method}" / f"M{module}"
    return {
        "statistics": Path(f"{prefix}__edge_statistics.parquet"),
        "summaries": Path(f"{prefix}__effect_edge_summaries.parquet"),
        "drivers": Path(f"{prefix}__effect_top_drivers.parquet"),
        "qa": Path(f"{prefix}__effect_qa.json"),
    }


def checkpoint_valid(root: Path, method: str, module: int, donors: int) -> bool:
    paths = output_paths(root, method, module)
    if not all(path.exists() for path in paths.values()):
        return False
    expected_summary_rows = donors * len(EFFECT_DIRECTIONS) * 9
    try:
        if pq.ParquetFile(paths["summaries"]).metadata.num_rows != expected_summary_rows:
            return False
        driver_schema = set(pq.ParquetFile(paths["drivers"]).schema_arrow.names)
        if not {"donor", "module", "effect_direction"}.issubset(driver_schema):
            return False
        qa = json.loads(paths["qa"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return qa.get("criterion") == SPEC.key and qa.get("status") == "complete"


def compute_task(
    cohort, assignments: pd.DataFrame, split_frame: pd.DataFrame,
    root: Path, method: str, module: int, chunk_size: int,
) -> dict[str, object]:
    started = time.monotonic()
    if checkpoint_valid(root, method, module, len(cohort.donors)):
        return {"method": method, "module": module, "status": "reused", "seconds": 0.0}
    matrix, tissues, genes = module_matrix(cohort, assignments, module)
    upper_i, upper_j, powers, component_edges = edge_metadata(tissues)
    context = scope_context(tissues, upper_i, upper_j, component_edges)
    diagnoses = np.asarray(cohort.diagnoses, dtype=object)
    split = split_frame.set_index("donor").loc[list(cohort.donors), "ad_control_split"].to_numpy()
    moments = moments_for_split(
        matrix, diagnoses, split, method, upper_i, upper_j, powers, chunk_size
    )
    edges = edge_statistics(module, method, genes, tissues, context, moments)
    edge_index = edges["edge_index"].to_numpy(dtype=np.int64)
    effect = edges["discovery_hedges_g"].to_numpy(dtype=float)
    masks = {
        direction: effect_size_mask(effect, edge_index, SPEC, direction=direction)
        for direction in EFFECT_DIRECTIONS
    }
    if not np.array_equal(masks["either"], masks["ad_higher"] | masks["control_higher"]):
        raise AssertionError("Either-direction mask is not the directional union")
    control_mean = edges["discovery_mean_control"].to_numpy(dtype=float)
    n_discovery_control = moments["Discovery"]["Control"].count
    summary_rows: list[dict[str, object]] = []
    driver_rows: list[dict[str, object]] = []
    for donor_indices, weights in method_weights(
        matrix, diagnoses, method, upper_i, upper_j, powers, chunk_size
    ):
        for local, donor_index in enumerate(donor_indices):
            donor = str(cohort.donors[donor_index])
            donor_weights = weights[local]
            for direction in EFFECT_DIRECTIONS:
                rule = effect_rule_key(SPEC, direction)
                rows = summarize_edge_network(
                    donor_weights, context, donor=donor, module=module,
                    differential_mask=masks[direction],
                    edge_rule_masks={"all": np.ones(len(donor_weights), dtype=bool)},
                    differential_edge_rule=rule,
                )
                for row in rows:
                    row.update(
                        {
                            "effect_statistic": "hedges_g",
                            "effect_cutoff_mode": "fixed",
                            "effect_cutoff_value": 0.40,
                            "effect_direction": direction,
                        }
                    )
                summary_rows.extend(rows)
            baseline = control_mean
            if diagnoses[donor_index] == "Control" and split[donor_index] == "Discovery":
                baseline = (
                    n_discovery_control * control_mean - donor_weights
                ) / (n_discovery_control - 1)
            for direction in DRIVER_DIRECTIONS:
                driver_rows.extend(
                    top_driver_rows(
                        donor, module, donor_weights, baseline, edges,
                        masks[direction], direction,
                    )
                )

    split_map = split_frame.set_index("donor")["ad_control_split"]
    summaries = pd.DataFrame(summary_rows)
    summaries.insert(0, "module_definition", MODULE_DEFINITION)
    summaries["estimator"] = "lioness"
    summaries["network_method"] = method
    summaries["ad_control_split"] = summaries["donor"].map(split_map)
    drivers = pd.DataFrame(driver_rows, columns=DRIVER_COLUMNS)
    drivers.insert(0, "module_definition", MODULE_DEFINITION)
    drivers["estimator"] = "lioness"
    drivers["network_method"] = method
    drivers["ad_control_split"] = drivers["donor"].map(split_map)

    paths = output_paths(root, method, module)
    atomic_parquet(edges, paths["statistics"])
    atomic_parquet(summaries, paths["summaries"])
    atomic_parquet(drivers, paths["drivers"])
    qa = {
        "status": "complete",
        "module_definition": MODULE_DEFINITION,
        "estimator": "lioness",
        "network_method": method,
        "module": int(module),
        "module_size": int(matrix.shape[1]),
        "tested_edges": int(len(edges)),
        "criterion": SPEC.key,
        "directions": list(EFFECT_DIRECTIONS),
        "summary_rows": int(len(summaries)),
        "driver_rows": int(len(drivers)),
        "mask_counts": {
            effect_rule_key(SPEC, direction): int(mask.sum())
            for direction, mask in masks.items()
        },
    }
    atomic_json(qa, paths["qa"])
    return {
        "method": method, "module": module, "status": "computed",
        "tested_edges": len(edges), "driver_rows": len(drivers),
        "seconds": round(time.monotonic() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--expression-root", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--metadata", type=Path, default=MUSCLE_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["control_anchored"],
        help=(
            "Network methods to calculate. This Control-derived module definition "
            "defaults to Control-referenced LIONESS only."
        ),
    )
    parser.add_argument("--module", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments = read_assignments(args.assignments, 30)
    cohort = load_cohort(args.expression_root, assignments)
    split = build_split(cohort, args.metadata, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "ad_control_discovery_validation_split.tsv"
    split.to_csv(split_path, sep="\t", index=False)
    available = sorted(assignments["Cluster ID"].astype(int).unique())
    modules = args.module or available
    unknown = sorted(set(modules).difference(available))
    if unknown:
        raise ValueError(f"Unknown module IDs: {unknown}")
    tasks = [(method, int(module)) for method in args.methods for module in modules]
    manifest_path = output / "run_manifest.json"
    manifest = {
        "analysis": "muscle_control_l4_hedges_g_040_effect_filter",
        "status": "running",
        "created_utc": utc_now(),
        "module_definition": MODULE_DEFINITION,
        "module_count": len(available),
        "donor_count": len(cohort.donors),
        "diagnosis_counts": pd.Series(cohort.diagnoses).value_counts().to_dict(),
        "split_counts": split.attrs["counts"],
        "split_seed": args.seed,
        "split_design": "70/30 within diagnosis x sex; MCI external",
        "effect_source": "discovery AD minus Control raw signedAlt LIONESS edge weights",
        "criterion": {"statistic": "hedges_g", "cutoff": 0.40},
        "directions": list(EFFECT_DIRECTIONS),
        "methods": args.methods,
        "source_hashes": {
            "assignments": sha256_file(args.assignments),
            "metadata": sha256_file(args.metadata),
            "split": sha256_file(split_path),
        },
        "completed_tasks": 0,
        "expected_tasks": len(tasks),
    }
    atomic_json(manifest, manifest_path)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                compute_task, cohort, assignments, split, output,
                method, module, args.chunk_size,
            ): (method, module)
            for method, module in tasks
        }
        for position, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            manifest["completed_tasks"] = len(results)
            manifest["last_completed"] = result
            atomic_json(manifest, manifest_path)
            print(f"[{position}/{len(tasks)}] {json.dumps(result, sort_keys=True)}", flush=True)
    manifest["status"] = "complete"
    manifest["completed_utc"] = utc_now()
    manifest["task_status_counts"] = (
        pd.Series([row["status"] for row in results]).value_counts().to_dict()
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
