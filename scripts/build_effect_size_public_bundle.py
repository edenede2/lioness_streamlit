#!/usr/bin/env python3
"""Publish completed effect-size app variants without rebuilding legacy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = APP_ROOT.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_public_data import (  # noqa: E402
    normalize_public_tissue_labels,
    write_combined_statistics,
    write_sanitized_plot_data,
)
from app_helpers.gene_symbols import public_gene_labels  # noqa: E402
from models.differential_edges import (  # noqa: E402
    effect_mask_specs,
    effect_size_mask,
    parse_effect_rule_key,
)


RUN_NAME = "20260906_effect_size_edge_filtering_and_donor_explorer"
MAX_PUBLIC_FILE_BYTES = 95 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_rows(path: Path) -> int | None:
    """Return dataset rows, or None when a file/dataset is incomplete."""

    files = [path] if path.is_file() else sorted(path.glob("*.parquet")) if path.is_dir() else []
    if not files:
        return None
    try:
        return sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    except (OSError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root", type=Path,
        default=REPO_ROOT / "out/lioness_app_expansion" / RUN_NAME,
    )
    parser.add_argument(
        "--app-data-root", type=Path,
        default=REPO_ROOT / "out/lioness_app_expansion" / RUN_NAME / "app_data",
    )
    parser.add_argument("--output", type=Path, default=APP_ROOT / "data")
    parser.add_argument(
        "--identity-source", type=Path,
        default=(
            REPO_ROOT / "out/lioness_reference_comparison_rosmap"
            / "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
        ),
    )
    parser.add_argument("--module-set", choices=("all", "full_cohort", "control_derived"), default="all")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an incomplete bundle for local smoke tests only.",
    )
    return parser.parse_args()


def _private_identity_file(root: Path) -> Path:
    matches = sorted((root / "standard/data").glob("*_transformed_component_data.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one standard transformed identity source: {matches}")
    return matches[0]


def derive_current_sample_map(private_source: Path, public_source: Path) -> dict[str, str]:
    """Recover the active bundle mapping by validated row/value correspondence."""

    columns_private = ["donor", "module", "metric_family", "CT_raw", "TS_raw", "lioness_method"]
    columns_public = ["sample_id", "module", "metric_family", "CT_raw", "TS_raw", "lioness_method"]
    private = pd.read_parquet(private_source, columns=columns_private)
    public = pd.read_parquet(public_source, columns=columns_public)
    private = private.loc[private["lioness_method"].eq("standard")]
    public = public.loc[public["lioness_method"].eq("standard")]
    first_module = int(private["module"].min())
    first_feature = str(private.loc[private["module"].eq(first_module), "metric_family"].iloc[0])
    private = private.loc[
        private["module"].eq(first_module) & private["metric_family"].eq(first_feature)
    ].reset_index(drop=True)
    public = public.loc[
        public["module"].eq(first_module) & public["metric_family"].eq(first_feature)
    ].reset_index(drop=True)
    if len(private) != 450 or len(public) != 450:
        raise ValueError("Identity fingerprint must contain one row for each of 450 donors")
    if not (
        np.allclose(private["CT_raw"], public["CT_raw"], equal_nan=True, rtol=0, atol=1e-12)
        and np.allclose(private["TS_raw"], public["TS_raw"], equal_nan=True, rtol=0, atol=1e-12)
    ):
        raise ValueError("Private and public identity fingerprints are not row-aligned")
    mapping = dict(zip(private["donor"].astype(str), public["sample_id"].astype(str), strict=True))
    if len(mapping) != 450 or len(set(mapping.values())) != 450:
        raise ValueError("Recovered sample mapping is not one-to-one")
    return mapping


def _sanitize_donor_frame(frame: pd.DataFrame, sample_map: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "sample_id", result.pop("donor").astype(str).map(sample_map))
    if result["sample_id"].isna().any():
        raise ValueError("Effect data contain a donor absent from the active pseudonym map")
    forbidden = {"donor", "projid"}.intersection(result.columns)
    if forbidden:
        raise ValueError(f"Private identifiers remain in public effect data: {sorted(forbidden)}")
    result = normalize_public_tissue_labels(result)
    for column in ("gene_a", "gene_b"):
        if column in result and result[column].astype(str).str.contains(r"ENSG\d+", regex=True).any():
            raise ValueError(f"Unconverted Ensembl identifiers remain in {column}")
    return result


def _shard_parquet_by_module(path: Path, modules_per_part: int = 40) -> None:
    """Replace a generated Parquet with module-range parts for lazy Drive reads."""

    if not path.is_file():
        return
    frame = pd.read_parquet(path)
    if "module" not in frame:
        raise ValueError(f"Effect app data lack a module field: {path}")
    path.unlink()
    path.mkdir(parents=True)
    modules = sorted(frame["module"].astype(int).unique())
    part_number = 0
    for start in range(0, len(modules), modules_per_part):
        selected_modules = modules[start : start + modules_per_part]
        group = frame.loc[frame["module"].isin(selected_modules)]
        # Retain a defensive split so a future schema expansion cannot violate
        # the file-size cap even within one module-range shard.
        pending = [group]
        while pending:
            part = pending.pop(0)
            destination = path / f"part-{part_number:05d}.parquet"
            part.to_parquet(
                destination, index=False, compression="zstd", row_group_size=50_000
            )
            if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                destination.unlink()
                if len(part) < 2:
                    raise ValueError(f"Cannot shard oversized single-row Parquet: {path}")
                midpoint = len(part) // 2
                pending = [part.iloc[:midpoint], part.iloc[midpoint:], *pending]
                continue
            part_number += 1


def _module_output_root(output: Path, module_set: str) -> Path:
    return output if module_set == "full_cohort" else output / module_set


def _publish_variant(
    variant_dir: Path,
    destination: Path,
    sample_map: dict[str, str],
) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "aggregate": variant_dir / "aggregate_plot_data.parquet",
        "resolved": variant_dir / "resolved_plot_data.parquet",
        "aggregate_statistics": variant_dir / "aggregate_statistics.parquet",
        "resolved_statistics": variant_dir / "resolved_statistics.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete effect app-data variant: {missing}")
    output_names = {
        "aggregate": "aggregate_plot_data.parquet",
        "resolved": "resolved_plot_data.parquet",
        "aggregate_statistics": "aggregate_statistics.parquet",
        "resolved_statistics": "resolved_statistics.parquet",
    }
    expected_rows = {
        key: int(pq.ParquetFile(path).metadata.num_rows)
        for key, path in sources.items()
    }
    existing_rows = {
        key: _parquet_rows(destination / filename)
        for key, filename in output_names.items()
    }
    if existing_rows == expected_rows:
        for filename in output_names.values():
            _shard_parquet_by_module(destination / filename)
        oversized = [
            str(path)
            for path in destination.rglob("*.parquet")
            if path.is_file() and path.stat().st_size >= MAX_PUBLIC_FILE_BYTES
        ]
        if oversized:
            raise ValueError(f"Effect app-data file exceeds 95 MiB: {oversized}")
        return expected_rows
    rows = {
        "aggregate": write_sanitized_plot_data(
            [sources["aggregate"]], destination / "aggregate_plot_data.parquet",
            sample_map, resolved=False,
        ),
        "resolved": write_sanitized_plot_data(
            [sources["resolved"]], destination / "resolved_plot_data.parquet",
            sample_map, resolved=True,
        ),
        "aggregate_statistics": write_combined_statistics(
            [sources["aggregate_statistics"]], destination / "aggregate_statistics.parquet"
        ),
        "resolved_statistics": write_combined_statistics(
            [sources["resolved_statistics"]], destination / "resolved_statistics.parquet",
            resolved=True,
        ),
    }
    for filename in output_names.values():
        _shard_parquet_by_module(destination / filename)
    oversized = [
        str(path)
        for path in destination.rglob("*.parquet")
        if path.is_file() and path.stat().st_size >= MAX_PUBLIC_FILE_BYTES
    ]
    if oversized:
        raise ValueError(f"Effect app-data file exceeds 95 MiB: {oversized}")
    return rows


def _effect_mask_column(spec) -> str:
    cutoff = (
        f"fixed_{spec.cutoff_value:.2f}"
        if spec.cutoff_mode == "fixed"
        else f"module_top_{int(round(100 * spec.cutoff_value)):02d}pct"
    )
    return f"effect_mask__{spec.statistic}__{cutoff.replace('.', 'p')}"


def _publish_effect_volcano_candidates(
    source_root: Path,
    destination_root: Path,
    module_set: str,
    estimator: str,
    method: str,
    modules: set[int],
) -> tuple[int, int]:
    source_dir = source_root / module_set / "checkpoints" / f"{estimator}__{method}"
    destination_dir = destination_root / "effect_size/volcano_candidates" / estimator / method
    destination_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    files = 0
    for source in sorted(source_dir.glob("M*__edge_statistics.parquet")):
        source_module = int(source.name.split("__", 1)[0].removeprefix("M"))
        if source_module not in modules:
            continue
        destination = destination_dir / f"M{source_module}.parquet"
        existing_rows = _parquet_rows(destination)
        if existing_rows is not None:
            if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                raise ValueError(f"Effect volcano shard exceeds 95 MiB: {destination}")
            rows += existing_rows
            files += 1
            continue
        frame = pd.read_parquet(source)
        edge_index = frame["edge_index"].to_numpy(dtype=np.int64)
        union = np.zeros(len(frame), dtype=bool)
        masks = {}
        for spec in effect_mask_specs():
            values = frame[f"discovery_{spec.statistic}"].to_numpy(dtype=float)
            mask = effect_size_mask(values, edge_index, spec, direction="either")
            masks[_effect_mask_column(spec)] = mask
            union |= mask
        selected = frame.loc[union].copy()
        for column, mask in masks.items():
            selected[column] = mask[union]
        selected = public_gene_labels(selected, gene_columns=("gene_a", "gene_b"))
        selected = normalize_public_tissue_labels(selected)
        if selected[["gene_a", "gene_b"]].astype(str).apply(
            lambda values: values.str.contains(r"ENSG\d+", regex=True)
        ).to_numpy().any():
            raise ValueError(f"Unconverted Ensembl identifiers in {source}")
        module = int(selected["module"].iloc[0])
        if module != source_module:
            raise ValueError(f"Volcano source module mismatch: {source}")
        if _parquet_rows(destination) != len(selected):
            selected.to_parquet(
                destination, index=False, compression="zstd", row_group_size=50_000
            )
        if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
            raise ValueError(f"Effect volcano shard exceeds 95 MiB: {destination}")
        rows += len(selected)
        files += 1
    return rows, files


def _publish_mask_retention_summary(
    analysis_root: Path,
    output_root: Path,
    module_set: str,
    networks: set[tuple[str, str]],
) -> tuple[Path, int]:
    rows: list[dict[str, object]] = []
    for qa_path in sorted(
        (analysis_root / module_set / "checkpoints").glob("*/M*__effect_qa.json")
    ):
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if (str(qa["estimator"]), str(qa["network_method"])) not in networks:
            continue
        tested_edges = int(qa["tested_edges"])
        for rule, count in qa["mask_counts"].items():
            spec, direction = parse_effect_rule_key(rule)
            rows.append(
                {
                    "module_definition": module_set,
                    "estimator": qa["estimator"],
                    "network_method": qa["network_method"],
                    "module": int(qa["module"]),
                    "effect_rule_key": rule,
                    "effect_statistic": spec.statistic,
                    "effect_cutoff_mode": spec.cutoff_mode,
                    "effect_cutoff_value": float(spec.cutoff_value),
                    "effect_direction": direction,
                    "tested_edges": tested_edges,
                    "retained_edges": int(count),
                    "retained_edge_fraction": (
                        float(count) / tested_edges if tested_edges else np.nan
                    ),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No effect QA checkpoints for {module_set}")
    destination = output_root / "effect_size/mask_retention_summary.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False, compression="zstd")
    if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
        raise ValueError(f"Effect mask summary exceeds 95 MiB: {destination}")
    return destination, len(rows)


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    app_data_root = args.app_data_root.resolve()
    output = args.output.resolve()
    run_manifest_path = analysis_root / "run_manifest.json"
    app_manifest_path = app_data_root / "app_data_manifest.json"
    if not run_manifest_path.exists() or not app_manifest_path.exists():
        raise FileNotFoundError("Effect analysis and app-data manifests must both be complete")
    sample_map = derive_current_sample_map(
        _private_identity_file(args.identity_source.resolve()),
        output / "aggregate_plot_data.parquet",
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    source_edge_root = Path(run_manifest["source_run"]).resolve()
    app_manifest = json.loads(app_manifest_path.read_text(encoding="utf-8"))
    selected_sets = {"full_cohort", "control_derived"}
    if args.module_set != "all":
        selected_sets = {args.module_set}
    expected_network_variants = {
        "full_cohort": {
            ("lioness", "standard"): 60,
            ("lioness", "control_anchored"): 60,
            ("bonobo", "bonobo"): 180,
        },
        "control_derived": {
            ("lioness", "control_anchored"): 60,
            ("bonobo", "bonobo"): 180,
        },
    }
    observed_network_variants = {
        module_set: {network: 0 for network in expected_network_variants[module_set]}
        for module_set in selected_sets
    }
    for key in app_manifest.get("variants", {}):
        parts = key.split("/")
        if len(parts) == 6 and parts[0] in observed_network_variants:
            network = (parts[1], parts[2])
            if network in observed_network_variants[parts[0]]:
                observed_network_variants[parts[0]][network] += 1
    completed_networks = {
        module_set: {
            network
            for network, expected in expected_network_variants[module_set].items()
            if observed_network_variants[module_set][network] == expected
        }
        for module_set in selected_sets
    }
    required_lioness = {
        module_set: {
            network
            for network in expected_network_variants[module_set]
            if network[0] == "lioness"
        }
        for module_set in selected_sets
    }
    incomplete = {
        module_set: {
            "/".join(network): {
                "observed": observed_network_variants[module_set][network],
                "expected": expected_network_variants[module_set][network],
            }
            for network in sorted(required_lioness[module_set] - completed_networks[module_set])
        }
        for module_set in selected_sets
        if required_lioness[module_set] - completed_networks[module_set]
    }
    if incomplete and not args.allow_partial:
        raise ValueError(f"Required LIONESS effect catalog is incomplete: {incomplete}")
    published: dict[str, dict[str, object]] = {key: {"variants": {}, "drivers": {}} for key in selected_sets}
    driver_jobs: set[tuple[str, str, str, str, str]] = set()
    summary_jobs: set[tuple[str, str, str, str, str]] = set()
    for key in sorted(app_manifest.get("variants", {})):
        parts = key.split("/")
        if len(parts) != 6:
            raise ValueError(f"Malformed effect app-data key: {key}")
        module_set, estimator, method, rule, edge_rule, normalization = parts
        if module_set not in selected_sets:
            continue
        if not args.allow_partial and (estimator, method) not in completed_networks[module_set]:
            # Optional networks (currently BONOBO) may remain safely resumable
            # without exposing a partial selector in Streamlit.
            continue
        source = app_data_root.joinpath(*parts)
        checkpoint_dir = analysis_root / module_set / "checkpoints" / f"{estimator}__{method}"
        checkpoint_modules = {
            int(path.name.split("__", 1)[0].removeprefix("M"))
            for path in checkpoint_dir.glob("M*__effect_edge_summaries.parquet")
        }
        variant_modules = set(
            pd.read_parquet(
                source / "aggregate_plot_data.parquet", columns=["module"]
            )["module"].astype(int).unique()
        )
        if variant_modules != checkpoint_modules:
            raise ValueError(
                f"Effect variant/checkpoint module mismatch for {key}: "
                f"variant={len(variant_modules)}, checkpoints={len(checkpoint_modules)}"
            )
        destination = _module_output_root(output, module_set).joinpath(
            "effect_size", estimator, method, rule, edge_rule, normalization
        )
        rows = _publish_variant(source, destination, sample_map)
        published[module_set]["variants"]["/".join(parts[1:])] = rows
        statistic, cutoff, _direction = rule.removeprefix(
            "ad_control_discovery_effect__"
        ).split("__")
        driver_jobs.add((module_set, estimator, method, f"{statistic}__{cutoff}", edge_rule))
        summary_jobs.add((module_set, estimator, method, rule, edge_rule))

    for module_set, estimator, method, rule, edge_rule in sorted(summary_jobs):
        checkpoint_dir = analysis_root / module_set / "checkpoints" / f"{estimator}__{method}"
        consolidated = (
            analysis_root / module_set / "consolidated" / f"{estimator}__{method}"
            / rule / f"{edge_rule}.parquet"
        )
        if consolidated.exists():
            summary = pd.read_parquet(consolidated)
        else:
            summary_parts = []
            for source_file in sorted(checkpoint_dir.glob("M*__effect_edge_summaries.parquet")):
                frame = pd.read_parquet(
                    source_file,
                    filters=[
                        ("differential_edge_rule", "=", rule),
                        ("edge_rule", "=", edge_rule),
                    ],
                )
                if not frame.empty:
                    summary_parts.append(frame)
            summary = pd.concat(summary_parts, ignore_index=True)
        summary = _sanitize_donor_frame(summary, sample_map)
        summary_destination = _module_output_root(output, module_set).joinpath(
            "effect_size", "edge_summaries", estimator, method, rule, f"{edge_rule}.parquet"
        )
        summary_destination.parent.mkdir(parents=True, exist_ok=True)
        existing_summary_rows = _parquet_rows(summary_destination)
        if existing_summary_rows not in {None, len(summary)}:
            raise ValueError(
                f"Incomplete effect edge-summary dataset at {summary_destination}: "
                f"{existing_summary_rows} rows, expected {len(summary)}"
            )
        if existing_summary_rows is None:
            summary.to_parquet(
                summary_destination, index=False, compression="zstd",
                row_group_size=50_000,
            )
        if summary_destination.is_file() and summary_destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
            raise ValueError(f"Effect edge-summary file exceeds 95 MiB: {summary_destination}")
        # A single summary is below GitHub's hard file limit, but loading all
        # modules would still cost roughly 80 MiB per sidebar change.  Partition
        # it by module so the Drive index can fetch only the shard containing the
        # selected module.
        _shard_parquet_by_module(summary_destination)

    jobs_by_method: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for module_set, estimator, method, criterion, edge_rule in driver_jobs:
        jobs_by_method.setdefault((module_set, estimator, method), set()).add(
            (criterion, edge_rule)
        )
    for (module_set, estimator, method), selected_jobs in sorted(jobs_by_method.items()):
        checkpoint_dir = analysis_root / module_set / "checkpoints" / f"{estimator}__{method}"
        driver_row_counts = {job: 0 for job in selected_jobs}
        for source_file in sorted(checkpoint_dir.glob("M*__effect_top_drivers.parquet")):
            module = int(source_file.name.split("__", 1)[0].removeprefix("M"))
            complete = pd.read_parquet(source_file)
            groups = complete.groupby(
                ["effect_statistic", "effect_cutoff_mode", "effect_cutoff_value", "edge_rule"],
                observed=True,
                sort=False,
            )
            for (statistic, cutoff_mode, cutoff_value, edge_rule), frame in groups:
                cutoff = (
                    f"fixed_{float(cutoff_value):.2f}"
                    if cutoff_mode == "fixed"
                    else f"module_top_{int(round(100 * float(cutoff_value))):02d}pct"
                )
                criterion = f"{statistic}__{cutoff}"
                job = (criterion, str(edge_rule))
                if job not in selected_jobs:
                    continue
                destination = _module_output_root(output, module_set).joinpath(
                    "effect_size", "drivers", estimator, method, criterion,
                    str(edge_rule), f"M{module}.parquet"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _parquet_rows(destination) != len(frame):
                    public_frame = _sanitize_donor_frame(
                        frame.reset_index(drop=True), sample_map
                    )
                    public_frame.to_parquet(
                        destination, index=False, compression="zstd",
                        row_group_size=20_000,
                    )
                if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                    raise ValueError(f"Driver shard exceeds 95 MiB: {destination}")
                driver_row_counts[job] += len(frame)
        for (criterion, edge_rule), driver_rows in driver_row_counts.items():
            published[module_set]["drivers"][
                "/".join((estimator, method, criterion, edge_rule))
            ] = driver_rows

    method_jobs = sorted({job[:3] for job in driver_jobs})
    for module_set, estimator, method in method_jobs:
        effect_checkpoint_dir = (
            analysis_root / module_set / "checkpoints" / f"{estimator}__{method}"
        )
        completed_modules = {
            int(path.name.split("__", 1)[0].removeprefix("M"))
            for path in effect_checkpoint_dir.glob("M*__effect_edge_summaries.parquet")
        }
        rows, files = _publish_effect_volcano_candidates(
            source_edge_root,
            _module_output_root(output, module_set),
            module_set,
            estimator,
            method,
            completed_modules,
        )
        published[module_set].setdefault("volcano_candidates", {})[
            f"{estimator}/{method}"
        ] = {"rows": rows, "files": files}

    for module_set in selected_sets:
        _path, rows = _publish_mask_retention_summary(
            analysis_root, _module_output_root(output, module_set), module_set,
            completed_networks[module_set],
        )
        published[module_set]["mask_retention_rows"] = rows

    created = datetime.now(timezone.utc).isoformat()
    for module_set, contents in published.items():
        publication_status = "incomplete" if module_set in incomplete else "complete"
        effect_root = _module_output_root(output, module_set) / "effect_size"
        public_files = {
            path.relative_to(effect_root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(effect_root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "created_utc": created,
            "status": publication_status,
            "module_definition": module_set,
            "completed_networks": [
                "/".join(network) for network in sorted(completed_networks[module_set])
            ],
            "default": {
                "statistic": "hedges_g", "cutoff_mode": "fixed",
                "cutoff_value": 0.40, "direction": "either",
            },
            "variants": contents["variants"],
            "driver_rows": contents["drivers"],
            "privacy": "pseudonymous sample_id only; donor and projid excluded",
            "source_hashes": run_manifest.get("source_hashes", {}),
            "public_files": public_files,
        }
        destination = effect_root / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    data_manifest_path = output / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    for module_set, contents in published.items():
        module_manifest = data_manifest["module_sets"][module_set]
        publication_status = "incomplete" if module_set in incomplete else "complete"
        module_manifest.setdefault("capabilities", {})["effect_size_edges"] = (
            publication_status == "complete"
        )
        module_manifest["effect_size_edges"] = {
            "status": publication_status,
            "effect_source": "discovery AD-Control edge tests",
            "fixed_hedges_g": [0.20, 0.30, 0.40, 0.50],
            "module_top_fractions": [0.01, 0.05, 0.10],
            "directions": ["either", "ad_higher", "control_higher"],
            "top_driver_count": 20,
            "completed_networks": [
                "/".join(network) for network in sorted(completed_networks[module_set])
            ],
            "variants": len(contents["variants"]),
            "manifest_sha256": _sha256(
                _module_output_root(output, module_set) / "effect_size/manifest.json"
            ),
        }
    temporary = Path(f"{data_manifest_path}.tmp")
    temporary.write_text(json.dumps(data_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(data_manifest_path)
    print(json.dumps({"created_utc": created, "published": published}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
