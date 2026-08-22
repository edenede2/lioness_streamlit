#!/usr/bin/env python3
"""Rewrite public volcano candidates with one union schema across methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from build_public_data import _write_public_edge_dataset, deploy_file_manifest_entry


PREVALENCE_COLUMNS = [
    f"bonobo_{rule}_prevalence_{group}"
    for rule in ("native_p05", "bh_fdr05")
    for group in ("all", "control", "mci", "ad")
]


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


def source_catalog(analysis_root: Path) -> dict[str, list[Path]]:
    return {
        "full_cohort": [
            analysis_root / "full_cohort/lioness/standard/volcano_candidates.parquet",
            analysis_root
            / "full_cohort/lioness/control_anchored/volcano_candidates.parquet",
            analysis_root / "full_cohort/bonobo/bonobo/volcano_candidates.parquet",
        ],
        "control_derived": [
            analysis_root
            / "control_derived/lioness/control_anchored/volcano_candidates.parquet",
            analysis_root
            / "control_derived/bonobo/bonobo/volcano_candidates.parquet",
        ],
    }


def main() -> None:
    args = parse_args()
    data_root = args.data.resolve()
    analysis_root = args.analysis_root.resolve()
    catalogs = source_catalog(analysis_root)
    repair_root = data_root.parent / f".{data_root.name}_candidate_union_repair"
    backup_root = data_root.parent / f"{data_root.name}_pre_union_candidate_backup"
    if repair_root.exists() or backup_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite repair/backup directories: {repair_root}, {backup_root}"
        )

    results: dict[str, dict[str, int]] = {}
    for module_set, sources in catalogs.items():
        expected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in sources)
        temporary = repair_root / module_set
        rows = _write_public_edge_dataset(sources, temporary)
        if rows != expected_rows:
            raise ValueError(f"{module_set}: expected {expected_rows} rows, found {rows}")
        dataset = ds.dataset(temporary, format="parquet")
        missing = set(PREVALENCE_COLUMNS).difference(dataset.schema.names)
        if missing:
            raise ValueError(f"{module_set}: missing prevalence fields {sorted(missing)}")
        for required in (
            "discovery_fdr_global",
            "discovery_fdr_per_module",
            "validation_fdr_global",
            "validation_fdr_per_module",
        ):
            if required not in dataset.schema.names:
                raise ValueError(f"{module_set}: missing {required}")
        results[module_set] = {
            "rows": rows,
            "parts": len(list(temporary.glob("part-*.parquet"))),
        }

    for module_set in catalogs:
        module_root = data_root if module_set == "full_cohort" else data_root / "control_derived"
        current = module_root / "differential/volcano_candidates.parquet"
        backup = backup_root / module_set
        backup.parent.mkdir(parents=True, exist_ok=True)
        current.rename(backup)
        (repair_root / module_set).rename(current)

    manifest_path = data_root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for module_set in catalogs:
        module_root = data_root if module_set == "full_cohort" else data_root / "control_derived"
        output = module_root / "differential/volcano_candidates.parquet"
        overall_prefix = str(output.relative_to(data_root)) + "/"
        module_prefix = str(output.relative_to(module_root)) + "/"
        manifest["files"] = {
            key: value
            for key, value in manifest["files"].items()
            if not key.startswith(overall_prefix)
        }
        module_files = manifest["module_sets"][module_set]["files"]
        manifest["module_sets"][module_set]["files"] = {
            key: value
            for key, value in module_files.items()
            if not key.startswith(module_prefix)
        }
        for part in sorted(output.glob("part-*.parquet")):
            entry = deploy_file_manifest_entry(part)
            manifest["files"][str(part.relative_to(data_root))] = entry
            manifest["module_sets"][module_set]["files"][
                str(part.relative_to(module_root))
            ] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "repaired",
                "module_sets": results,
                "backup": str(backup_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
