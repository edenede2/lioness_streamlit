#!/usr/bin/env python3
"""Resumable full-catalog control-calibrated connectivity; no prediction/deploy."""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys, argparse, json, time, resource, logging

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1])]
logging.getLogger("streamlit").setLevel(logging.ERROR)
from app_helpers.control_calibrated.registry import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--stage",
        choices=["prepare", "benchmark", "score", "masks", "analyze", "all"],
        default="all",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=APP / "artifacts/private_calibration_followup_20261101",
    )
    p.add_argument("--edge-batch", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--family",
        action="append",
        help="Limit offline mask generation to completed requested partitions",
    )
    a = p.parse_args()
    start = time.perf_counter()
    if a.stage in ("prepare", "all"):
        from app_helpers.control_calibrated.inputs import prepare

        if not (a.root / "catalog.json").exists():
            prepare(a.root, a.baseline)
    if a.stage in ("benchmark", "score", "all"):
        from app_helpers.control_calibrated.generate import generate

        generate(
            a.root,
            edge_batch=a.edge_batch,
            benchmark=a.stage == "benchmark",
            resume=a.resume,
            workers=a.workers,
        )
    if a.stage in ("masks", "all"):
        from app_helpers.control_calibrated.generate_masks import generate_masks

        generate_masks(a.root, workers=a.workers, families=a.family)
    if a.stage in ("analyze", "all"):
        from app_helpers.control_calibrated.analysis import analyze_catalog

        analyze_catalog(a.root)
    print(
        json.dumps(
            {
                "stage": a.stage,
                "seconds": time.perf_counter() - start,
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
