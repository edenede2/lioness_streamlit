#!/usr/bin/env python3
"""Summarize, independently verify, or activate an unpublished local bundle."""
from pathlib import Path
import sys, argparse, json

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers.control_calibrated.validation_spec import workspace
from app_helpers.control_calibrated import validation_finish as F

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["summarize", "verify", "activate"])
    a = p.parse_args()
    root, spec = workspace()
    if (root / "complete.json").exists():
        print(
            "Completed immutable bundle already exists; use its recorded outputs or freeze a new analysis version."
        )
        raise SystemExit(0)
    result = getattr(F, a.stage)(root, spec)
    print(
        json.dumps(
            {
                "stage": a.stage,
                "status": "complete",
                "result": result if a.stage == "verify" else None,
            }
        ),
        flush=True,
    )
