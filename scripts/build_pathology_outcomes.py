#!/usr/bin/env python3
"""Add verified postmortem pathology to existing app pseudonyms and saved scores.

No networks, eigengenes, masks or prediction artifacts are recomputed. Only the
existing private mappings are used, never clinical-value fingerprint recovery.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path.insert(0, str(APP))
from app_helpers import data as D
from app_helpers.pathology import (
    OUTCOME_COLUMNS, PATHOLOGY_OUTCOMES, _calculate_statistics, statistics_path,
)

OUTPUT = REPO / "out/pathology_app_20261015"
SOURCE = REPO / "data/external/rosmapPhenos/dataset_1593_cross-sectional_08-13-2025 (1).xlsx"
METADATA_PATHS = [
    "sample_metadata.parquet", "muscle_control_l4/sample_metadata.parquet",
    "muscle_all_donors_l4/sample_metadata.parquet",
    *[f"regional_modules/metadata/maximum_tissue/{t}.parquet" for t in ("AC", "DLPFC", "PCG")],
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path):
    file = pq.ParquetFile(path)
    return dict(bytes=path.stat().st_size, rows=file.metadata.num_rows,
                schema={f.name: str(f.type) for f in file.schema_arrow}, sha256=sha256(path))


def atomic_copy(path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copy2(path, temporary)
    temporary.replace(target)


@lru_cache(maxsize=1)
def load_source():
    audit = pd.read_csv(REPO / "out/experiment0210_protocol_v1_1/source_audit/outcome_dictionary.tsv", sep="\t")
    expected = audit.loc[audit.key.isin(PATHOLOGY_OUTCOMES), "source_sha256"].unique()
    if len(expected) != 1 or expected[0] != sha256(SOURCE):
        raise ValueError("Pathology release differs from the verified source audit")
    frame = pd.read_excel(SOURCE, usecols=["projid", *OUTCOME_COLUMNS.values()])
    frame = frame.dropna(subset=["projid"])
    frame["donor"] = frame.projid.map(lambda v: "ROSMAP-" + str(int(v)).zfill(8))
    columns = list(OUTCOME_COLUMNS.values())
    if frame.groupby("donor")[columns].nunique(dropna=True).gt(1).any().any():
        raise ValueError("Conflicting source pathology values")
    frame = frame.groupby("donor")[columns].first().rename(columns={v:k for k,v in OUTCOME_COLUMNS.items()})
    return frame.apply(pd.to_numeric, errors="raise").astype(float)


def augment_metadata(frame: pd.DataFrame, donors) -> pd.DataFrame:
    """Shared offline-builder hook; missing pathology is never filled/imputed."""
    values = load_source().reindex(list(map(str, donors)))
    result = frame.copy()
    for outcome in PATHOLOGY_OUTCOMES:
        result[outcome] = values[outcome].to_numpy(dtype=float)
    return result


def metadata_bundle(output):
    source = load_source()
    brain = json.loads((REPO / "out/lioness_regional_modules_rosmap/20261002_independent_region_app/private_regional_public_id_mapping.json").read_text())
    motor = pd.read_csv(REPO / "out/muscle_control_l4/private_public_sample_mapping.tsv", sep="\t", dtype=str)
    mappings = {"brain": pd.Series(list(brain.keys()), index=list(brain.values())),
                "motor": motor.set_index("sample_id").donor}
    stage = output / "staging"
    coverage = {}
    for relative in METADATA_PATHS:
        public = APP / "data" / relative
        old = pd.read_parquet(public)
        mapping = mappings["motor" if relative.startswith("muscle_") else "brain"]
        if mapping.index.duplicated().any():
            raise ValueError("Non-unique existing pseudonym mapping")
        donors = old.sample_id.map(mapping)
        if donors.isna().any():
            raise ValueError("An existing sample is missing its private mapping")
        values = source.reindex(donors.to_list()).reset_index(drop=True)
        fresh = old.drop(columns=list(PATHOLOGY_OUTCOMES), errors="ignore").copy()
        for outcome in PATHOLOGY_OUTCOMES:
            fresh[outcome] = values[outcome].to_numpy()
        original_columns = [column for column in old if column not in PATHOLOGY_OUTCOMES]
        pd.testing.assert_frame_equal(old[original_columns], fresh[original_columns])
        if {"donor", "projid"}.intersection(fresh.columns) or fresh.sample_id.duplicated().any():
            raise ValueError("Invalid public metadata identifiers/grain")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_parquet(target, index=False, compression="zstd")
        coverage[relative] = dict(donors=len(fresh), observed={v:int(fresh[v].notna().sum()) for v in PATHOLOGY_OUTCOMES}, **describe(target))
    expected = {"sample_metadata.parquet": (450,449,448), "muscle_control_l4/sample_metadata.parquet": (354,357,357), "muscle_all_donors_l4/sample_metadata.parquet": (354,357,357)}
    for key, counts in expected.items():
        if tuple(coverage[key]["observed"].values()) != counts:
            raise AssertionError(f"Pathology cohort counts differ: {key}")
    return coverage


def write_manifest(output, coverage, statistics):
    stage = output / "staging"
    target = stage / "pathology/manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, status="complete", source_file=SOURCE.name, source_sha256=sha256(SOURCE),
        source_release="2025-08-13", assessment="Postmortem", source_columns=OUTCOME_COLUMNS,
        outcome_transform="None added; amyloid/tangles retain supplied square-root 8-region scales",
        fdr="BH across valid modules separately within each fixed outcome, component, method, feature and diagnosis/group level; no group/outcome pooling",
        legacy_fdr="Existing all-12 and primary-five catalogs unchanged; not applicable to the three additional outcomes",
        privacy="Existing sample_id only; no donor/projid/new samples", metadata=coverage,
        statistics=statistics, lazy_contexts="Custom strata and existing edge subsets use bounded saved-score reads",
        prediction_changed=False, generated_utc=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def publish_local(output, manifest):
    stage, backup = output / "staging", output / "rollback"
    paths = [*METADATA_PATHS, "pathology/manifest.json", *manifest["statistics"]]
    for key in paths:
        target = APP / "data" / key
        if target.exists() and not (backup / key).exists():
            (backup / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / key)
        atomic_copy(stage / key, target)
    root_path = APP / "data/data_manifest.json"
    root = json.loads(root_path.read_text())
    if not (backup / "data_manifest.json").exists():
        shutil.copy2(root_path, backup / "data_manifest.json")
    root["pathology_outcomes"] = {k:manifest[k] for k in ("schema_version", "status", "source_columns", "source_sha256", "fdr", "assessment")}
    root["files"]["sample_metadata.parquet"] = describe(APP / "data/sample_metadata.parquet")
    for key, module in root.get("module_sets", {}).items():
        if key in ("full_cohort", "control_derived", "muscle_control_l4", "muscle_all_donors_l4"):
            relative = "sample_metadata.parquet" if key in ("full_cohort", "control_derived") else f"{key}/sample_metadata.parquet"
            if "sample_metadata.parquet" in module.get("files", {}):
                module["files"]["sample_metadata.parquet"] = describe(APP / "data" / relative)
            module["additional_outcomes"] = list(PATHOLOGY_OUTCOMES)
    regional_path = APP / "data/regional_modules/manifest.json"
    regional = json.loads(regional_path.read_text())
    for tissue in ("AC", "DLPFC", "PCG"):
        entry = regional["metadata_files"][tissue]
        entry.update(describe(APP / "data/regional_modules" / entry["relative_path"]))
    if not (backup / "regional_manifest.json").exists():
        shutil.copy2(regional_path, backup / "regional_manifest.json")
    for path, payload in ((root_path, root), (regional_path, regional)):
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n" if path == regional_path
                             else json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        temporary.replace(path)


def build_context(task):
    module_set, estimator, method, rule, resolved, output = task
    path = statistics_path(module_set, estimator, method, rule, resolved)
    relative = path.relative_to(D.DATA_DIR).as_posix()
    target = output / "staging" / relative
    started = time.monotonic()
    if not target.exists():
        table = _calculate_statistics(module_set, estimator, method, rule, resolved, None,
            "all", "global", .05, "standard_pruned", "all_donors", "complete_450")
        table["estimator"], table["network_method"] = estimator, method
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.partial")
        table.to_parquet(temporary, index=False, compression="zstd", row_group_size=2000)
        temporary.replace(target)
    entry = describe(target)
    if entry["bytes"] >= 95 * 1024**2:
        raise ValueError("Pathology shard exceeds public file limit")
    atomic_copy(target, path)
    print(json.dumps({"path":relative,"rows":entry["rows"],"seconds":round(time.monotonic()-started,2)}), flush=True)
    return relative, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--publish-local", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1,2,3), default=3)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    coverage = metadata_bundle(args.output_root)
    manifest = write_manifest(args.output_root, coverage, {})
    if args.publish_local:
        publish_local(args.output_root, manifest)
    if args.metadata_only:
        print(json.dumps({"status":"metadata_validated", "coverage":{k:v["observed"] for k,v in coverage.items()}}))
        return
    if not args.publish_local:
        raise ValueError("Statistics require --publish-local so readers use the validated staged metadata")
    statistics, tasks = {}, []
    registry = json.loads((APP / "data/data_manifest.json").read_text())["module_sets"]
    for module_set in D.available_module_sets():
        specification = registry[module_set]
        contexts = [("lioness", method, "all") for method in specification["methods"]]
        if "bonobo" in specification.get("estimators", []):
            contexts += [("bonobo", "bonobo", rule) for rule in ("all", "native_p05", "bh_fdr05")]
        for estimator, method, rule in contexts:
            for resolved in (False, True):
                tasks.append((module_set, estimator, method, rule, resolved, args.output_root))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(build_context, task) for task in tasks]):
            relative, entry = task.result()
            statistics[relative] = entry
    manifest = write_manifest(args.output_root, coverage, statistics)
    publish_local(args.output_root, manifest)
    print(json.dumps({"status":"complete","statistics_files":len(statistics),"prediction_changed":False}))


if __name__ == "__main__":
    main()
