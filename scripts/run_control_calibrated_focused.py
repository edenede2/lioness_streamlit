#!/usr/bin/env python3
"""Run the prespecified four-candidate explanatory models, without publication."""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from pathlib import Path
import sys, argparse, json

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers.control_calibrated.validation_spec import workspace
from app_helpers.control_calibrated.validation_focused import run_focused

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    root, spec = workspace()
    if (root / "complete.json").exists() or (root / "focused_complete.json").exists():
        print(
            "Focused results already completed; use recorded outputs or freeze a new version."
        )
    else:
        print(json.dumps(run_focused(root, spec, a.workers)), flush=True)
