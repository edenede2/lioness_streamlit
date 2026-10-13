#!/usr/bin/env python3
"""Prepare and publish VH/SMA/QUAD Hedges-g edge-filtered app data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = APP_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.differential_edges import (  # noqa: E402
    EFFECT_DIRECTIONS,
    EffectMaskSpec,
    effect_rule_key,
    effect_size_mask,
)
from scripts.python import prepare_effect_size_edge_app_data as prepare  # noqa: E402
from build_muscle_public_data import (  # noqa: E402
    ASSIGNMENTS,
    COMPONENT_LABELS,
    OUTCOMES,
    PRIVATE_MAP,
    RUN_ROOT,
    atomic_parquet,
    refresh_existing_manifest,
    update_app_manifest,
    gene_symbol_mapping,
)


RUN_NAME = "20261012_muscle_effect_size_edge_filtering"
MODULE_SET = "muscle_control_l4"
MODULE_COUNT = 155
# This module definition was learned in Controls. Standard-LIONESS checkpoints
# may exist in the private analysis root from early development, but they are
# intentionally never packaged or exposed by the public application.
METHODS = ("control_anchored",)
SPEC = EffectMaskSpec("hedges_g", "fixed", 0.40)
MAX_PUBLIC_FILE_BYTES = 95 * 1024 * 1024
DEFAULT_ANALYSIS = REPO_ROOT / "out/lioness_app_expansion" / RUN_NAME
DEFAULT_OUTPUT = APP_ROOT / "data/muscle_control_l4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int | None:
    files = [path] if path.is_file() else sorted(path.glob("*.parquet")) if path.is_dir() else []
    if not files:
        return None
    return sum(pq.ParquetFile(file).metadata.num_rows for file in files)


def write_shards(frame: pd.DataFrame, path: Path, modules_per_part: int = 20) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    modules = sorted(frame["module"].astype(int).unique())
    part_number = 0
    for start in range(0, len(modules), modules_per_part):
        selected = modules[start : start + modules_per_part]
        pending = [frame.loc[frame["module"].isin(selected)].copy()]
        while pending:
            part = pending.pop(0)
            destination = path / f"part-{part_number:05d}.parquet"
            part.to_parquet(
                destination, index=False, compression="zstd", row_group_size=50_000
            )
            if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                destination.unlink()
                if len(part) < 2:
                    raise ValueError(f"Cannot split oversized public shard: {path}")
                midpoint = len(part) // 2
                pending = [part.iloc[:midpoint], part.iloc[midpoint:], *pending]
                continue
            part_number += 1


def private_metadata(analysis_root: Path, output: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    split = pd.read_csv(
        analysis_root / "ad_control_discovery_validation_split.tsv",
        sep="\t", dtype={"donor": str},
    )
    mapping = pd.read_csv(PRIVATE_MAP, sep="\t", dtype=str)
    mapping = mapping.loc[mapping["donor"].isin(set(split["donor"]))].copy()
    if len(mapping) != len(split) or mapping["sample_id"].nunique() != len(split):
        raise ValueError("Muscle pseudonym map does not cover the effect cohort")
    public = pd.read_parquet(output / "sample_metadata.parquet")
    metadata = split[["donor", "diagnosis_group", "ad_control_split"]].merge(
        mapping, on="donor", how="left", validate="one_to_one"
    ).merge(
        public[["sample_id", *OUTCOMES]], on="sample_id", how="left",
        validate="one_to_one",
    )
    metadata = metadata.drop(columns="sample_id")
    if metadata["ad_control_split"].isna().any():
        raise ValueError("Effect split metadata are incomplete")
    return metadata, dict(zip(mapping["donor"], mapping["sample_id"], strict=True))


def sanitize(frame: pd.DataFrame, sample_map: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    if "donor" in result:
        result.insert(0, "sample_id", result.pop("donor").astype(str).map(sample_map))
        if result["sample_id"].isna().any():
            raise ValueError("A private donor lacks a public pseudonym")
    forbidden = {"donor", "projid"}.intersection(result.columns)
    if forbidden:
        raise ValueError(f"Private identifiers remain: {sorted(forbidden)}")
    gene_columns = [column for column in ("gene_a", "gene_b") if column in result]
    if gene_columns:
        symbols = gene_symbol_mapping()
        for column in gene_columns:
            result[column] = result[column].map(
                lambda value: symbols.get(str(value).split(".")[0], "Unmapped gene")
                if str(value).startswith("ENSG") else value
            )
    return result


def prepare_private_variants(
    analysis_root: Path,
    app_data_root: Path,
    metadata: pd.DataFrame,
    workers: int,
    overwrite: bool,
) -> dict[str, object]:
    # The generic transformation/statistics engine is intentionally reused so
    # brain and muscle effect-filtered scores have identical definitions.
    prepare.COMPONENT_LABELS = {
        component: (component.split("_", 1)[0], label)
        for component, label in COMPONENT_LABELS.items()
    }
    prepare.OUTCOMES = OUTCOMES
    variants: dict[str, object] = {}
    for method in METHODS:
        for direction in EFFECT_DIRECTIONS:
            for normalization in ("standard_pruned", "retained_edge"):
                key = "/".join(
                    (
                        MODULE_SET, "lioness", method,
                        effect_rule_key(SPEC, direction), "all", normalization,
                    )
                )
                variants[key] = prepare.prepare_variant(
                    input_root=analysis_root,
                    output_root=app_data_root,
                    module_set=MODULE_SET,
                    estimator="lioness",
                    method=method,
                    spec=SPEC,
                    direction=direction,
                    edge_rule="all",
                    normalization=normalization,
                    metadata=metadata,
                    overwrite=overwrite,
                    statistics_workers=workers,
                )
                print(f"prepared {key}", flush=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "analysis_subsets": {
            "all_donors": None,
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        },
        "variants": variants,
    }
    app_data_root.mkdir(parents=True, exist_ok=True)
    (app_data_root / "app_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def publish_variants(
    app_data_root: Path, output: Path, sample_map: dict[str, str]
) -> dict[str, dict[str, int]]:
    published: dict[str, dict[str, int]] = {}
    for method in METHODS:
        for direction in EFFECT_DIRECTIONS:
            rule = effect_rule_key(SPEC, direction)
            for normalization in ("standard_pruned", "retained_edge"):
                source = app_data_root.joinpath(
                    MODULE_SET, "lioness", method, rule, "all", normalization
                )
                destination = output.joinpath(
                    "effect_size", "lioness", method, rule, "all", normalization
                )
                rows: dict[str, int] = {}
                for filename in (
                    "aggregate_plot_data.parquet", "resolved_plot_data.parquet",
                    "aggregate_statistics.parquet", "resolved_statistics.parquet",
                ):
                    frame = pd.read_parquet(source / filename)
                    if "plot_data" in filename:
                        frame = sanitize(frame, sample_map)
                    target = destination / filename
                    if "plot_data" in filename:
                        write_shards(frame, target)
                    else:
                        atomic_parquet(frame, target)
                        if target.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                            write_shards(frame, target)
                    rows[filename.removesuffix(".parquet")] = len(frame)
                published[f"lioness/{method}/{rule}/all/{normalization}"] = rows
    return published


def publish_edge_data(
    analysis_root: Path, output: Path, sample_map: dict[str, str]
) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    retention_rows: list[dict[str, object]] = []
    volcano_rows = 0
    mask_column = "effect_mask__hedges_g__fixed_0p40"
    for method in METHODS:
        checkpoint = analysis_root / MODULE_SET / "checkpoints" / f"lioness__{method}"
        for path in sorted(checkpoint.glob("M*__effect_qa.json")):
            qa = json.loads(path.read_text(encoding="utf-8"))
            for rule, retained in qa["mask_counts"].items():
                direction = rule.rsplit("__", 1)[-1]
                retention_rows.append(
                    {
                        "module_definition": MODULE_SET,
                        "estimator": "lioness",
                        "network_method": method,
                        "module": int(qa["module"]),
                        "effect_rule_key": rule,
                        "effect_statistic": "hedges_g",
                        "effect_cutoff_mode": "fixed",
                        "effect_cutoff_value": 0.40,
                        "effect_direction": direction,
                        "tested_edges": int(qa["tested_edges"]),
                        "retained_edges": int(retained),
                        "retained_edge_fraction": (
                            float(retained) / int(qa["tested_edges"])
                            if int(qa["tested_edges"]) else np.nan
                        ),
                    }
                )
        for direction in EFFECT_DIRECTIONS:
            rule = effect_rule_key(SPEC, direction)
            parts = []
            for path in sorted(checkpoint.glob("M*__effect_edge_summaries.parquet")):
                selected = pd.read_parquet(
                    path,
                    filters=[
                        ("differential_edge_rule", "=", rule),
                        ("edge_rule", "=", "all"),
                    ],
                )
                if not selected.empty:
                    parts.append(selected)
            frame = sanitize(pd.concat(parts, ignore_index=True), sample_map)
            destination = output.joinpath(
                "effect_size", "edge_summaries", "lioness", method, rule,
                "all.parquet",
            )
            write_shards(frame, destination)
            counts[f"summaries/lioness/{method}/{rule}/all"] = len(frame)
        driver_destination = output / "effect_size/drivers/lioness" / method / SPEC.key / "all"
        driver_destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(checkpoint.glob("M*__effect_top_drivers.parquet")):
            module = int(path.name.split("__", 1)[0].removeprefix("M"))
            frame = sanitize(pd.read_parquet(path), sample_map)
            target = driver_destination / f"M{module}.parquet"
            atomic_parquet(frame, target)
            if target.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                raise ValueError(f"Driver shard exceeds public file limit: {target}")
        volcano_destination = output / "effect_size/volcano_candidates/lioness" / method
        volcano_destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(checkpoint.glob("M*__edge_statistics.parquet")):
            frame = pd.read_parquet(path)
            values = frame["discovery_hedges_g"].to_numpy(float)
            mask = effect_size_mask(
                values, frame["edge_index"].to_numpy(np.int64), SPEC, direction="either"
            )
            selected = frame.loc[mask].copy()
            selected[mask_column] = True
            # The compact muscle release preserves exact per-module BH values.
            # Global-BH columns remain explicit NA rather than being mislabeled.
            selected["discovery_fdr_global"] = np.nan
            selected["validation_fdr_global"] = np.nan
            module = int(selected["module"].iloc[0]) if not selected.empty else int(
                path.name.split("__", 1)[0].removeprefix("M")
            )
            target = volcano_destination / f"M{module}.parquet"
            atomic_parquet(selected, target)
            if target.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                raise ValueError(f"Volcano shard exceeds public file limit: {target}")
            volcano_rows += len(selected)
    retention = pd.DataFrame(retention_rows)
    atomic_parquet(retention, output / "effect_size/mask_retention_summary.parquet")
    return counts, volcano_rows


def _recover_count(weight_sum: pd.Series, density: pd.Series, label: str) -> pd.Series:
    weight = pd.to_numeric(weight_sum, errors="coerce").to_numpy(dtype=float)
    mean = np.abs(pd.to_numeric(density, errors="coerce").to_numpy(dtype=float))
    recovered = np.divide(weight, mean, out=np.zeros_like(weight), where=mean > 0)
    rounded = np.rint(recovered)
    finite = np.isfinite(recovered)
    if finite.any() and np.max(np.abs(recovered[finite] - rounded[finite])) > 2e-5:
        raise ValueError(f"{label}: edge-count recovery is not integer-valued")
    if np.any((mean == 0) & (np.abs(weight) > 1e-8)):
        raise ValueError(f"{label}: zero density has a nonzero weight sum")
    return pd.Series(rounded, index=weight_sum.index, dtype="Int64")


def publish_all_edge_summaries(output: Path, sample_map: dict[str, str]) -> int:
    """Recover the all-edge summary required by the Edge summaries view."""

    assignments = pd.read_csv(ASSIGNMENTS, dtype={"Tissue": str})
    assignments["module"] = pd.to_numeric(
        assignments["Cluster ID"], errors="raise"
    ).astype(int)
    module_sizes = assignments.groupby("module", observed=True).size()
    kept = set(module_sizes.loc[module_sizes.ge(30)].index.astype(int))
    assignments = assignments.loc[assignments["module"].isin(kept)].copy()
    counts = (
        assignments.groupby(["module", "Tissue"], observed=True)
        .size().unstack(fill_value=0)
        .reindex(columns=["VH", "SMA", "QUAD"], fill_value=0)
    )
    if len(counts) != MODULE_COUNT:
        raise ValueError(f"Expected {MODULE_COUNT} retained muscle modules, found {len(counts)}")
    possible_rows: list[dict[str, int | str]] = []
    for module, row in counts.iterrows():
        n = {tissue: int(row[tissue]) for tissue in ("VH", "SMA", "QUAD")}
        scopes = {
            "total": sum(n.values()) * (sum(n.values()) - 1) // 2,
            "TS": sum(value * (value - 1) // 2 for value in n.values()),
            "CT": n["VH"] * n["SMA"] + n["VH"] * n["QUAD"] + n["SMA"] * n["QUAD"],
            "TS_VH": n["VH"] * (n["VH"] - 1) // 2,
            "TS_SMA": n["SMA"] * (n["SMA"] - 1) // 2,
            "TS_QUAD": n["QUAD"] * (n["QUAD"] - 1) // 2,
            "CT_VH__SMA": n["VH"] * n["SMA"],
            "CT_VH__QUAD": n["VH"] * n["QUAD"],
            "CT_SMA__QUAD": n["SMA"] * n["QUAD"],
        }
        if scopes["total"] != scopes["TS"] + scopes["CT"]:
            raise ValueError(f"M{module}: total edge-count identity failed")
        possible_rows.extend(
            {"module": int(module), "scope": scope, "n_possible_edges": value}
            for scope, value in scopes.items()
        )
    possible = pd.DataFrame(possible_rows)
    metrics = pd.read_parquet(
        RUN_ROOT / "control_anchored/donor_module_metrics.parquet"
    )
    if len(metrics) != MODULE_COUNT * 357:
        raise ValueError("Control-referenced all-edge metrics are not rectangular")
    scopes = (
        "total", "TS", "CT", "TS_VH", "TS_SMA", "TS_QUAD",
        "CT_VH__SMA", "CT_VH__QUAD", "CT_SMA__QUAD",
    )
    rows: list[pd.DataFrame] = []
    for scope in scopes:
        suffix = "all" if scope == "total" else scope
        positive_density = f"positive_density_{suffix}"
        negative_density = f"negative_density_{suffix}"
        positive_sum = f"positive_abs_sum_{suffix}"
        negative_sum = f"negative_abs_sum_{suffix}"
        part = metrics[
            ["donor", "diagnosis_group", "module", positive_density,
             negative_density, positive_sum, negative_sum]
        ].copy()
        part = part.rename(
            columns={
                positive_sum: "positive_weight_sum",
                negative_sum: "negative_weight_magnitude",
            }
        )
        part["n_positive_edges"] = _recover_count(
            part["positive_weight_sum"], part[positive_density],
            f"control_anchored/{scope}/positive",
        )
        part["n_negative_edges"] = _recover_count(
            part["negative_weight_magnitude"], part[negative_density],
            f"control_anchored/{scope}/negative",
        )
        part = part.merge(
            possible.loc[possible["scope"].eq(scope)],
            on="module", how="left", validate="many_to_one",
        )
        part["scope"] = scope
        part["edge_rule"] = "all"
        part["n_retained_edges"] = part["n_possible_edges"]
        part["n_zero_edges"] = (
            part["n_possible_edges"]
            - part["n_positive_edges"].fillna(0)
            - part["n_negative_edges"].fillna(0)
        ).astype("Int64")
        part["n_pruned_edges"] = 0
        if part["n_zero_edges"].dropna().lt(0).any():
            raise ValueError(f"{scope}: recovered sign counts exceed possible edges")
        part["signed_weight_sum"] = (
            part["positive_weight_sum"] - part["negative_weight_magnitude"]
        )
        part["absolute_weight_sum"] = (
            part["positive_weight_sum"] + part["negative_weight_magnitude"]
        )
        part["module_definition"] = MODULE_SET
        part["estimator"] = "lioness"
        part["network_method"] = "control_anchored"
        rows.append(
            part[
                [
                    "module_definition", "estimator", "network_method", "donor",
                    "diagnosis_group", "module", "scope", "edge_rule",
                    "n_possible_edges", "n_retained_edges", "n_positive_edges",
                    "n_negative_edges", "n_zero_edges", "n_pruned_edges",
                    "signed_weight_sum", "positive_weight_sum",
                    "negative_weight_magnitude", "absolute_weight_sum",
                ]
            ]
        )
    frame = pd.concat(rows, ignore_index=True)
    for metric in (
        "n_possible_edges", "n_retained_edges", "n_positive_edges",
        "n_negative_edges", "n_zero_edges", "n_pruned_edges",
        "signed_weight_sum", "positive_weight_sum",
        "negative_weight_magnitude", "absolute_weight_sum",
    ):
        pivot = frame.pivot(
            index=["donor", "module", "edge_rule"], columns="scope", values=metric
        ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        for parent, children in {
            "total": ["TS", "CT"],
            "TS": ["TS_VH", "TS_SMA", "TS_QUAD"],
            "CT": ["CT_VH__SMA", "CT_VH__QUAD", "CT_SMA__QUAD"],
        }.items():
            if not np.allclose(
                pivot[parent].to_numpy(float),
                pivot[children].sum(axis=1).to_numpy(float),
                atol=1e-8, rtol=1e-9,
            ):
                raise ValueError(f"{metric}: {parent} scope identity failed")
    frame = sanitize(frame, sample_map)
    destination = output / "edge_summaries/lioness__control_anchored.parquet"
    atomic_parquet(frame, destination)
    if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
        write_shards(frame, destination)
    return len(frame)


def update_manifests(
    analysis_root: Path,
    output: Path,
    variants: dict[str, dict[str, int]],
    edge_counts: dict[str, int],
    volcano_rows: int,
) -> None:
    split = pd.read_csv(
        analysis_root / "ad_control_discovery_validation_split.tsv", sep="\t"
    )
    metadata_path = output / "sample_metadata.parquet"
    metadata = pd.read_parquet(metadata_path).drop(columns=["ad_control_split"], errors="ignore")
    mapping = pd.read_csv(PRIVATE_MAP, sep="\t", dtype=str)
    public_split = split[["donor", "ad_control_split"]].merge(
        mapping, on="donor", how="left", validate="one_to_one"
    )[["sample_id", "ad_control_split"]]
    metadata = metadata.merge(public_split, on="sample_id", how="left", validate="one_to_one")
    if metadata["ad_control_split"].isna().any():
        raise ValueError("Published sample split is incomplete")
    atomic_parquet(metadata, metadata_path)

    effect_root = output / "effect_size"
    public_files = {
        path.relative_to(effect_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(effect_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "module_definition": MODULE_SET,
        "completed_networks": [f"lioness/{method}" for method in METHODS],
        "available_effect_statistics": ["hedges_g"],
        "available_cutoff_modes": {"hedges_g": ["fixed"]},
        "available_fixed_hedges_g": [0.40],
        "available_directions": list(EFFECT_DIRECTIONS),
        "available_score_normalizations": ["standard_pruned", "retained_edge"],
        "default": {
            "statistic": "hedges_g", "cutoff_mode": "fixed",
            "cutoff_value": 0.40, "direction": "either",
        },
        "split_counts": (
            split.groupby(["ad_control_split", "diagnosis_group"], observed=True)
            .size().rename("n").reset_index().to_dict(orient="records")
        ),
        "variants": variants,
        "edge_summary_rows": edge_counts,
        "volcano_candidate_rows": volcano_rows,
        "privacy": "pseudonymous sample_id only; donor and projid excluded",
        "public_files": public_files,
    }
    (effect_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    module_manifest_path = output / "module_set_manifest.json"
    module_manifest = json.loads(module_manifest_path.read_text(encoding="utf-8"))
    module_manifest["methods"] = list(METHODS)
    capabilities = module_manifest.setdefault("capabilities", {})
    capabilities["effect_size_edges"] = True
    capabilities["edge_summaries"] = True
    capabilities["differential_edges"] = True
    module_manifest["effect_size_edges"] = {
        "status": "complete",
        "statistic": "Hedges g (AD minus Control)",
        "cutoff": 0.40,
        "directions": list(EFFECT_DIRECTIONS),
        "mask_learning": "diagnosis-by-sex stratified discovery AD/Control donors only",
    }
    module_manifest_path.write_text(
        json.dumps(module_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    refreshed = refresh_existing_manifest(output)
    update_app_manifest(refreshed, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--app-data-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--statistics-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    output = args.output.resolve()
    app_data_root = (
        args.app_data_root.resolve()
        if args.app_data_root
        else analysis_root / "app_data"
    )
    run_manifest = json.loads(
        (analysis_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = (
        analysis_root / MODULE_SET / "checkpoints/lioness__control_anchored"
    )
    completed_modules = {
        int(path.name.split("__", 1)[0].removeprefix("M"))
        for path in checkpoint.glob("M*__effect_qa.json")
    }
    if run_manifest.get("status") != "complete" or len(completed_modules) != MODULE_COUNT:
        raise ValueError(
            f"The {MODULE_COUNT}-module Control-referenced effect analysis is not complete"
        )
    metadata, sample_map = private_metadata(analysis_root, output)
    if args.publish_only:
        app_manifest = json.loads(
            (app_data_root / "app_data_manifest.json").read_text(encoding="utf-8")
        )
    else:
        app_manifest = prepare_private_variants(
            analysis_root, app_data_root, metadata,
            args.statistics_workers, args.overwrite,
        )
    if args.prepare_only:
        return
    variants = publish_variants(app_data_root, output, sample_map)
    all_edge_summary_rows = publish_all_edge_summaries(output, sample_map)
    edge_counts, volcano_rows = publish_edge_data(analysis_root, output, sample_map)
    update_manifests(analysis_root, output, variants, edge_counts, volcano_rows)
    print(
        json.dumps(
            {
                "status": "complete", "variants": len(variants),
                "edge_summary_catalogs": len(edge_counts),
                "all_edge_summary_rows": all_edge_summary_rows,
                "volcano_candidate_rows": volcano_rows,
                "private_variants": len(app_manifest.get("variants", {})),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
