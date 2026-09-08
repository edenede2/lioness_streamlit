#!/usr/bin/env python3
"""Build pseudonymous, module-sharded expression data for the edge explorer.

The public files contain within-gene Z-scores and opaque node-column names.  Raw
donor IDs, projid, Ensembl IDs, and the source expression column names are never
written to the deploy bundle.
"""

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
for location in (SCRIPT_DIR, APP_ROOT, REPO_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from app_helpers.gene_symbols import gene_symbol  # noqa: E402
from build_effect_size_public_bundle import (  # noqa: E402
    _private_identity_file,
    derive_current_sample_map,
)
from scripts.python.run_bonobo_app_rosmap import (  # noqa: E402
    configs,
    load_assignment_table,
    load_expression,
)


MAX_PUBLIC_FILE_BYTES = 95 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=APP_ROOT / "data/edge_expression")
    parser.add_argument(
        "--identity-source",
        type=Path,
        default=(
            REPO_ROOT
            / "out/lioness_reference_comparison_rosmap"
            / "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_module_file(path: Path, expected_rows: int, expected_nodes: int) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = pq.ParquetFile(path).metadata
    except (OSError, ValueError):
        return False
    return (
        metadata.num_rows == expected_rows
        and metadata.num_columns == expected_nodes + 1
        and path.stat().st_size < MAX_PUBLIC_FILE_BYTES
    )


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Return cohort-standardized float32 values while preserving missingness."""

    numeric = np.asarray(values, dtype=np.float64)
    means = np.nanmean(numeric, axis=0)
    standard_deviations = np.nanstd(numeric, axis=0, ddof=0)
    finite_scale = np.isfinite(standard_deviations) & (standard_deviations > 0)
    result = np.zeros(numeric.shape, dtype=np.float32)
    result[:] = np.nan
    if finite_scale.any():
        result[:, finite_scale] = (
            (numeric[:, finite_scale] - means[finite_scale])
            / standard_deviations[finite_scale]
        ).astype(np.float32)
    constant = ~finite_scale & np.isfinite(means)
    if constant.any():
        observed = np.isfinite(numeric[:, constant])
        constant_values = np.zeros(observed.shape, dtype=np.float32)
        constant_values[~observed] = np.nan
        result[:, constant] = constant_values
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app_data = APP_ROOT / "data"
    sample_map = derive_current_sample_map(
        _private_identity_file(args.identity_source.resolve()),
        app_data / "aggregate_plot_data.parquet",
    )
    donors = list(sample_map)
    sample_ids = [sample_map[donor] for donor in donors]
    if len(donors) != 450 or len(set(sample_ids)) != 450:
        raise ValueError("The endpoint-expression bundle requires 450 unique pseudonyms")

    module_configs = configs(REPO_ROOT)
    tables = {config.key: load_assignment_table(config) for config in module_configs}
    expression = load_expression(REPO_ROOT, tables, donors)
    if expression.shape[0] != 450:
        raise ValueError("Expression matrix did not retain exactly 450 complete-tissue donors")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": {
            "sample_identifier": "existing public pseudonymous sample_id",
            "excluded": ["donor", "projid", "Ensembl identifiers"],
            "expression_values": (
                "within-tissue-gene cohort Z-scores; source normalized-log values "
                "and source gene identifiers are not exported"
            ),
            "bulk_download_in_app": False,
        },
        "standardization": "(x - cohort mean) / population SD across 450 donors",
        "module_sets": {},
        "source_files": {},
    }
    raw_root = REPO_ROOT / "data/raw/ROSMAP_brain"
    source_paths = [
        raw_root / "ROSMAP_fixed_AC.csv",
        raw_root / "ROSMAP_fixed_DLPFC.csv",
        raw_root / "ROSMAP_fixed_PCGBA23.csv",
        *[config.assignments for config in module_configs],
    ]
    for path in source_paths:
        manifest["source_files"][str(path.relative_to(REPO_ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    for config in module_configs:
        table = tables[config.key].copy()
        module_directory = output / config.key / "modules"
        module_directory.mkdir(parents=True, exist_ok=True)
        node_frames: list[pd.DataFrame] = []
        module_files: dict[str, dict[str, object]] = {}
        for module, assignments in table.groupby("module", sort=True, observed=True):
            assignments = assignments.copy()
            features = assignments["feature_id"].astype(str).tolist()
            values = expression.loc[donors, features].to_numpy(dtype=np.float64)
            standardized = standardize_columns(values)
            columns = {"sample_id": sample_ids}
            columns.update(
                {
                    f"node_{node_index}": standardized[:, node_index]
                    for node_index in range(standardized.shape[1])
                }
            )
            destination = module_directory / f"M{int(module)}.parquet"
            if args.overwrite or not valid_module_file(
                destination, len(donors), len(features)
            ):
                temporary = destination.with_suffix(".parquet.tmp")
                pd.DataFrame(columns).to_parquet(
                    temporary,
                    index=False,
                    compression="zstd",
                    row_group_size=450,
                )
                temporary.replace(destination)
            if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
                raise ValueError(f"Expression module shard exceeds 95 MiB: {destination}")
            module_files[destination.name] = {
                "rows": 450,
                "nodes": len(features),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
            symbols = assignments["Gene Symbol"].map(gene_symbol).astype(str)
            if symbols.str.contains(r"ENSG\d+", regex=True).any():
                raise ValueError(f"Unconverted gene identifier in {config.key}/M{module}")
            node_frames.append(
                pd.DataFrame(
                    {
                        "module_definition": config.key,
                        "module": np.int32(module),
                        "node_index": np.arange(len(assignments), dtype=np.int32),
                        "expression_column": [
                            f"node_{index}" for index in range(len(assignments))
                        ],
                        "tissue": (
                            assignments["Tissue"].astype(str)
                            .replace({"MFBA9BA46": "DLPFC", "PCGBA23": "PCG"})
                            .to_numpy()
                        ),
                        "gene_symbol": symbols.to_numpy(),
                    }
                )
            )
        nodes = pd.concat(node_frames, ignore_index=True)
        nodes_path = output / config.key / "module_nodes.parquet"
        nodes.to_parquet(nodes_path, index=False, compression="zstd")
        if {"donor", "projid"}.intersection(nodes.columns):
            raise ValueError("Private identifier leaked into module-node metadata")
        manifest["module_sets"][config.key] = {
            "modules": int(nodes["module"].nunique()),
            "node_rows": len(nodes),
            "donors_per_module": 450,
            "module_nodes_file": {
                "bytes": nodes_path.stat().st_size,
                "sha256": sha256(nodes_path),
            },
            "module_files": module_files,
        }

    data_manifest_path = app_data / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_manifest["endpoint_expression_explorer"] = {
        "status": "complete",
        "module_sets": {
            key: {
                "modules": value["modules"],
                "node_rows": value["node_rows"],
                "donors_per_module": value["donors_per_module"],
            }
            for key, value in manifest["module_sets"].items()
        },
        "scale": manifest["standardization"],
        "privacy": manifest["privacy"],
    }
    for key in manifest["module_sets"]:
        data_manifest["module_sets"][key].setdefault("capabilities", {})[
            "edge_expression"
        ] = True
    data_manifest_path.write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "module_sets": {
                    key: {
                        "modules": value["modules"],
                        "node_rows": value["node_rows"],
                    }
                    for key, value in manifest["module_sets"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
