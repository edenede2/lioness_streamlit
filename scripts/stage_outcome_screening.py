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
    modules = [entry for entry in manifest["files"] if entry["path"].startswith("module_edges/")]
    if modules:
        reconciliation = json.loads((root/"qa/catalog_reconciliation.json").read_text())
        if reconciliation.get("status") != "passed" or reconciliation.get("audit_identity") != manifest.get("audit_identity"):
            raise ValueError("Completed-module catalog reconciliation is required")
        checked = {e["path"]:e["sha256"] for e in reconciliation["files"] if e.get("status") == "passed"}
        if any(checked.get(e["path"]) != e["sha256"] for e in modules):
            raise ValueError("A completed module catalog does not match its FDR reconciliation")
    for entry in manifest["files"]:
        if not entry["path"].startswith("whole_network/"):
            continue
        method = Path(entry["path"]).stem.split("__",1)[0]
        receipt = json.loads((root/"qa"/f"whole_network_brain_{method}.json").read_text())
        checked = {e["path"]:e["sha256"] for e in receipt["files"]}
        if (receipt.get("status") != "passed" or receipt.get("audit_identity") != manifest.get("audit_identity")
                or checked.get(entry["path"]) != entry["sha256"]):
            raise ValueError("Whole-network catalog lacks matching numerical/FDR reconciliation")
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
    updated = dict(manifest, files=completed)
    if "pilot_review_file" in manifest:
        review = manifest["pilot_review_file"]
        path = source/review["path"]
        if not path.resolve().is_relative_to(source.resolve()) or sha(path) != review["sha256"] or path.stat().st_size > 1024**2:
            raise ValueError("Pilot review hash/path/size gate failed")
        payload = json.loads(path.read_text())
        def private_key(value):
            if isinstance(value, dict):
                return any(k in {"donor", "projid", "sample_id"} or private_key(v) for k,v in value.items())
            if isinstance(value,list):
                return any(private_key(v) for v in value)
            return False
        if private_key(payload) or payload.get("audit_identity") != manifest.get("audit_identity") or payload.get("genome_launch_authorized") is not False:
            raise ValueError("Only aggregate, non-authorizing pilot reviews may be staged")
        relative = Path(path.stem+"__"+review["sha256"][:12]+".json")
        atomic_json(payload,destination/relative)
        updated["pilot_review_file"] = dict(review,path=relative.as_posix())
    atomic_json(updated, destination/"manifest.json")
    return dict(status="staged_locally", files=len(completed), rows=sum(e["rows"] for e in completed), donor_level_published=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.destination)))
