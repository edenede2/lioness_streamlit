#!/usr/bin/env python3
"""Stage completed aggregate catalogs only. No Drive calls, commit, or push."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP/"scripts")]
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.gene_symbols import ENSEMBL_PATTERN
from scripts.run_gene_edge_screen import RUN, atomic_json
from scripts.screening_sources import sha


def stage(root=RUN, destination=None):
    root = Path(root); source = root/"public_bundle"
    destination = Path(destination) if destination else APP/"data/outcome_screening"
    manifest = json.loads((source/"manifest.json").read_text())
    qa_path = root/"qa/independent_direct_checks.json"
    if not qa_path.is_file():
        raise ValueError("Independent direct-correlation QA is required before staging")
    qa = json.loads(qa_path.read_text())
    if qa.get("status") != "passed" or qa.get("audit_identity") != manifest.get("audit_identity"):
        raise ValueError("QA/source identity mismatch")
    for entry in manifest["files"]:
        path = source/entry["path"]
        if entry["status"] != "complete" or sha(path) != entry["sha256"] or path.stat().st_size >= 95*1024**2:
            raise ValueError("Catalog completion/hash/size gate failed")
        f = pd.read_parquet(path)
        if len(f) != entry["rows"] or {"sample_id", "donor", "projid", "weight"} & set(f):
            raise ValueError("Aggregate-only publication scope violation")
        for col in f.select_dtypes(include=["object", "string"]):
            if f[col].dropna().astype(str).str.contains(ENSEMBL_PATTERN).any():
                raise ValueError("Ensembl identifier in public catalog")
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copytree(destination, root/"rollback"/stamp)
    # Immutable content-hash paths: writing data cannot corrupt the currently
    # active manifest. The final atomic manifest swap alone enables new files.
    completed = []
    for entry in manifest["files"]:
        source_path = source/entry["path"]
        relative = Path(entry["path"])
        new_relative = relative.with_name(relative.stem+"__"+entry["sha256"][:12]+relative.suffix)
        target = destination/new_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            temporary = target.with_suffix(".partial")
            shutil.copy2(source_path, temporary); temporary.replace(target)
        # Compact selector metadata avoids loading all modules/outcomes merely
        # to discover widget choices in Community Cloud.
        frame = pd.read_parquet(source_path, columns=[c for c in ("outcome", "module") if c in pq.ParquetFile(source_path).schema_arrow.names])
        selectors = dict(outcomes=sorted(frame.outcome.dropna().unique().tolist()))
        if "module" in frame:
            selectors["modules"] = sorted(map(int, frame.module.dropna().unique()))
        completed.append(dict(entry, catalog_id=entry["path"], path=new_relative.as_posix(), **selectors))
    atomic_json(dict(manifest, files=completed), destination/"manifest.json")
    return dict(status="staged_locally", files=len(completed), rows=sum(e["rows"] for e in completed), donor_level_published=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.destination)))
