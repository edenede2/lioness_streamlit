#!/usr/bin/env python3
"""Stage ROS/MAP membership using existing private mappings, without predictions.

Run in the offline analysis environment (requires openpyxl for the source release).
The application does not need openpyxl or access to private donor identifiers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pyarrow.parquet as pq

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path.insert(0, str(APP))
from app_helpers.study import normalize_study

SOURCE = REPO / "data/external/rosmapPhenos/dataset_1593_cross-sectional_08-13-2025 (1).xlsx"
OUTPUT = REPO / "out/study_metadata_app_20261021"
METADATA_PATHS = (
    "sample_metadata.parquet", "muscle_control_l4/sample_metadata.parquet",
    "muscle_all_donors_l4/sample_metadata.parquet",
    *[f"regional_modules/metadata/maximum_tissue/{t}.parquet" for t in ("AC", "DLPFC", "PCG")],
)


def describe(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, rows=pq.ParquetFile(path).metadata.num_rows,
                schema={f.name: str(f.type) for f in pq.read_schema(path)},
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@lru_cache(maxsize=2)
def source_labels(path=SOURCE):
    frame = pd.read_excel(path, usecols=["projid", "study"]).dropna(subset=["projid"])
    identifiers = pd.to_numeric(frame.projid, errors="raise")
    if identifiers.mod(1).ne(0).any():
        raise ValueError("Non-integer source identifier")
    frame["donor"] = identifiers.astype("int64").map(lambda v: "ROSMAP-" + str(v).zfill(8))
    frame["study"] = normalize_study(frame.study)
    if frame.groupby("donor").study.nunique(dropna=True).gt(1).any():
        raise ValueError("Conflicting study labels for a donor")
    return frame.groupby("donor").study.first()


def augment_metadata(frame, donors, labels=None):
    labels = source_labels() if labels is None else labels
    result = frame.copy()
    result["study"] = pd.array(labels.reindex(list(map(str, donors))).to_numpy(), dtype="string")
    return result


def stage(output=OUTPUT):
    labels = source_labels()
    brain_path = REPO / "out/lioness_regional_modules_rosmap/20261002_independent_region_app/private_regional_public_id_mapping.json"
    motor_path = REPO / "out/muscle_control_l4/private_public_sample_mapping.tsv"
    brain = json.loads(brain_path.read_text())
    motor = pd.read_csv(motor_path, sep="\t", dtype=str)
    mappings = {"brain": pd.Series(list(brain.keys()), index=list(brain.values())),
                "motor": motor.set_index("sample_id").donor}
    coverage = {}
    overlays = []
    for relative in METADATA_PATHS:
        old = pd.read_parquet(APP / "data" / relative)
        mapping = mappings["motor" if relative.startswith("muscle_") else "brain"]
        if mapping.index.duplicated().any() or old.sample_id.duplicated().any():
            raise ValueError("Non-unique mapping or public samples")
        donors = old.sample_id.map(mapping)
        if donors.isna().any():
            raise ValueError("Existing pseudonym is absent from its private mapping")
        fresh = augment_metadata(old, donors, labels)
        overlay = fresh[["sample_id", "study"]].copy()
        overlay["metadata_namespace"] = "motor" if relative.startswith("muscle_") else "brain"
        overlays.append(overlay)
        unchanged = [column for column in old if column != "study"]
        pd.testing.assert_frame_equal(old[unchanged], fresh[unchanged])
        if {"donor", "projid"}.intersection(fresh):
            raise ValueError("Private identifiers in public metadata")
        target = output / "staging" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_parquet(target, index=False, compression="zstd")
        coverage[relative] = {**describe(target), "study_counts": {"Missing" if pd.isna(k) else str(k): int(v) for k,v in fresh.study.value_counts(dropna=False).items()}}
    overlay = pd.concat(overlays, ignore_index=True)
    if overlay.groupby(["metadata_namespace", "sample_id"]).study.nunique(dropna=True).gt(1).any():
        raise ValueError("Study cache contains conflicting source labels")
    overlay = overlay.groupby(["metadata_namespace", "sample_id"], as_index=False).study.first()
    overlay_path = output / "staging/study/sample_study.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_parquet(overlay_path, index=False, compression="zstd")
    manifest = dict(schema_version=1, status="complete", source_file=SOURCE.name,
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    source_release="2025-08-13", source_column="study", source_donors=len(labels),
                    normalization="Trim whitespace and uppercase only; missing values remain missing",
                    interpretation="Nominal study membership, not a numerical phenotype or causal batch effect",
                    metadata=coverage, prediction_changed=False,
                    repository_overlay={"path":"study/sample_study.parquet", **describe(overlay_path)},
                    privacy="Existing pseudonyms only; no donor/projid/new samples",
                    generated_utc=datetime.now(timezone.utc).isoformat())
    path = output / "staging/study/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def install_local(output=OUTPUT):
    stage_root, backup = output / "staging", output / "rollback"
    manifest = json.loads((stage_root / "study/manifest.json").read_text())
    for relative in (*METADATA_PATHS, "study/sample_study.parquet", "study/manifest.json"):
        target = APP / "data" / relative
        if target.exists() and not (backup / relative).exists():
            (backup / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(stage_root / relative, temporary)
        temporary.replace(target)
    root_path = APP / "data/data_manifest.json"
    regional_path = APP / "data/regional_modules/manifest.json"
    for path in (root_path, regional_path):
        relative = path.relative_to(APP / "data")
        if not (backup / relative).exists():
            (backup / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / relative)
    root = json.loads(root_path.read_text())
    root["study_metadata"] = {k: manifest[k] for k in ("schema_version", "status", "source_sha256", "source_column", "source_release")}
    root["study_metadata"]["repository_overlay"] = manifest["repository_overlay"]
    root["files"]["sample_metadata.parquet"] = describe(APP / "data/sample_metadata.parquet")
    for key, entry in root.get("module_sets", {}).items():
        relative = "sample_metadata.parquet" if key in ("full_cohort", "control_derived") else f"{key}/sample_metadata.parquet"
        if relative in METADATA_PATHS and "sample_metadata.parquet" in entry.get("files", {}):
            entry["files"]["sample_metadata.parquet"] = describe(APP / "data" / relative)
    regional = json.loads(regional_path.read_text())
    for tissue in ("AC", "DLPFC", "PCG"):
        entry = regional["metadata_files"][tissue]
        entry.update(describe(APP / "data/regional_modules" / entry["relative_path"]))
    for path, payload in ((root_path, root), (regional_path, regional)):
        temporary = path.with_suffix(".json.partial")
        temporary.write_text((json.dumps(payload, indent=2, sort_keys=True) if path == regional_path else json.dumps(payload, separators=(",", ":"), sort_keys=True)) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--install-local", action="store_true")
    args = parser.parse_args()
    result = stage(args.output)
    if args.install_local:
        install_local(args.output)
    print(json.dumps({"status": "complete", "metadata": {k:v["study_counts"] for k,v in result["metadata"].items()}}))
