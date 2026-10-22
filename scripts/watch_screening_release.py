#!/usr/bin/env python3
"""Local-only incremental release and subsequent single-worker M5 pilot.

Run alongside run_gene_edge_screen.py --stage first-release in a persistent
terminal. There is no Git operation, Drive upload, or full-genome launch here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP/"scripts")]
from scripts.run_gene_edge_screen import RUN, atomic_json
from scripts.stage_outcome_screening import stage
from scripts.validate_outcome_screening import validate


def watch(root, max_hours=24, poll_seconds=30, pilot=False):
    root = Path(root); deadline = time.monotonic()+max_hours*3600
    previous = None
    status_path = root/"release_queue_status.json"
    while time.monotonic() < deadline:
        manifest_path = root/"public_bundle/manifest.json"
        if not manifest_path.is_file():
            time.sleep(poll_seconds); continue
        manifest = json.loads(manifest_path.read_text())
        identity = tuple((e["path"],e["sha256"]) for e in manifest["files"])
        if identity != previous:
            validate(root)
            report = stage(root)
            previous = identity
            atomic_json(dict(report, queue_status="incrementally_staged"), status_path)
            print(json.dumps(report), flush=True)
        if manifest.get("brain_module_edges_status") == "complete":
            from scripts.consolidate_screening_weights import consolidate
            print(json.dumps(consolidate(root)), flush=True)
            if pilot:
                # Screening numerical workers have exited before all three
                # complete catalogs are enabled, so the pilot uses one worker.
                atomic_json(dict(status="full_M5_pilot_running", genome_launch_authorized=False), status_path)
                command = [sys.executable, str(APP/"scripts/run_screening_m5_pilot.py"), "--root", str(root)]
                result = subprocess.run(command, cwd=APP, check=False)
                if result.returncode:
                    atomic_json(dict(status="full_M5_pilot_failed", exit_code=result.returncode,
                        genome_launch_authorized=False), status_path)
                    return result.returncode
            atomic_json(dict(status="brain_staged_M5_review_required" if pilot else "brain_staged",
                donor_values_published=False, git_pushed=False, genome_launch_authorized=False), status_path)
            return 0
        time.sleep(poll_seconds)
    atomic_json(dict(status="watch_timeout_resume_required", genome_launch_authorized=False), status_path)
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--pilot-after-brain", action="store_true")
    args = parser.parse_args()
    raise SystemExit(watch(args.root, args.max_hours, pilot=args.pilot_after_brain))
