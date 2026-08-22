#!/usr/bin/env python3
"""Rewrite public volcano bins as Arrow lists and refresh manifest hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from build_public_data import (
    _write_public_edge_dataset,
    deploy_file_manifest_entry,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Completed staged public data directory")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=repo_root
        / "out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data.resolve()
    analysis_root = args.analysis_root.resolve()
    catalogs = {
        "full_cohort": {
            "module_root": data_root,
            "sources": [
                analysis_root / "full_cohort/lioness/standard/volcano_bins.parquet",
                analysis_root
                / "full_cohort/lioness/control_anchored/volcano_bins.parquet",
                analysis_root / "full_cohort/bonobo/bonobo/volcano_bins.parquet",
            ],
            "expected_rows": 66_528,
        },
        "control_derived": {
            "module_root": data_root / "control_derived",
            "sources": [
                analysis_root
                / "control_derived/lioness/control_anchored/volcano_bins.parquet",
                analysis_root / "control_derived/bonobo/bonobo/volcano_bins.parquet",
            ],
            "expected_rows": 53_568,
        },
    }
    manifest_path = data_root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: dict[str, object] = {}
    for module_set, catalog in catalogs.items():
        module_root = Path(catalog["module_root"])
        output = module_root / "differential/volcano_bins.parquet"
        rows = _write_public_edge_dataset(
            [Path(path) for path in catalog["sources"]],
            output,
            rows_per_file=20_000,
        )
        if rows != int(catalog["expected_rows"]):
            raise ValueError(
                f"{module_set}: expected {catalog['expected_rows']} bins, found {rows}"
            )
        dataset = ds.dataset(output, format="parquet")
        for column in ("x_edges", "y_edges", "counts"):
            if not pa.types.is_list(dataset.schema.field(column).type):
                raise ValueError(f"{module_set}: {column} was not preserved as an Arrow list")
        table = dataset.to_table(columns=["counts", "n_edges"])
        frame = table.to_pandas()
        represented = frame["counts"].map(lambda values: int(np.asarray(values).sum()))
        if not np.array_equal(represented.to_numpy(), frame["n_edges"].to_numpy()):
            raise ValueError(f"{module_set}: rewritten bins do not reconcile to edge totals")

        overall_files = manifest["files"]
        module_files = manifest["module_sets"][module_set]["files"]
        for part in sorted(output.glob("part-*.parquet")):
            entry = deploy_file_manifest_entry(part)
            overall_files[str(part.relative_to(data_root))] = entry
            module_files[str(part.relative_to(module_root))] = entry
        results[module_set] = {"rows": rows, "parts": len(list(output.glob("part-*.parquet")))}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "repaired", "module_sets": results}, indent=2))


if __name__ == "__main__":
    main()
