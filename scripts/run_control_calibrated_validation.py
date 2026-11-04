#!/usr/bin/env python3
"""Run the five prespecified local-only ROSMAP covariate validation analyses."""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
from pathlib import Path
import sys, argparse, json

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers.control_calibrated import validation_spec as S, validation_run as R


def benchmark(root, spec):
    """Time one real bounded shard before launching the full catalog."""
    from app_helpers.control_calibrated.inputs import atomic_json

    tasks = R.plan_tasks(root, spec, "repair")
    task = next(t for t in tasks if len(t["rows"]) >= 512 and len(t["donors"]) >= 100)
    receipt = R.execute_task(task)
    receipt.update(
        score_vectors=len(task["rows"]), donors=len(task["donors"]), draws=task["draws"]
    )
    atomic_json(root / "benchmark.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "freeze",
            "prepare",
            "benchmark",
            "repair",
            "scan",
            "focused",
            "summarize",
            "verify",
            "report",
            "activate",
            "finalize",
            "all",
        ],
    )
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    root, spec = S.freeze() if a.stage in ("freeze", "all") else S.workspace()
    print(
        json.dumps(
            dict(stage=a.stage, analysis_id=spec["analysis_id"], root=str(root))
        ),
        flush=True,
    )
    if (root / "complete.json").exists():
        print(
            "Completed immutable bundle; using recorded results. Scientific changes require a new frozen version."
        )
        return
    if (
        a.stage in ("prepare", "all")
        and not (root / "preparation_complete.json").exists()
    ):
        print(json.dumps(R.prepare(root, spec)), flush=True)
    if a.stage in ("benchmark", "all") and not (root / "benchmark.json").exists():
        print(json.dumps(benchmark(root, spec)), flush=True)
    if a.stage in ("repair", "all") and not (root / "repair_complete.json").exists():
        R.run_tasks(root, spec, "repair", a.workers)
        print(json.dumps(R.assemble_repair(root, spec)), flush=True)
    if a.stage in ("scan", "all") and not (root / "scan_complete.json").exists():
        R.run_tasks(root, spec, "scan", a.workers)
        print(json.dumps(R.assemble_scan(root, spec)), flush=True)
    if (
        a.stage in ("focused", "finalize", "all")
        and not (root / "focused_complete.json").exists()
    ):
        from app_helpers.control_calibrated.validation_focused import run_focused

        print(json.dumps(run_focused(root, spec, a.workers)), flush=True)
    from app_helpers.control_calibrated import validation_finish as F

    for stage, receipt in [
        ("summarize", "summary_complete"),
        ("verify", "verification"),
        ("report", "report_complete"),
        ("activate", "complete"),
    ]:
        if (
            a.stage in (stage, "finalize", "all")
            and not (root / (receipt + ".json")).exists()
        ):
            if stage == "report":
                from app_helpers.control_calibrated.validation_report import build

                result = build(root, spec)
            else:
                result = getattr(F, stage)(root, spec)
            print(json.dumps(dict(stage=stage, result=result)), flush=True)


if __name__ == "__main__":
    main()
