#!/usr/bin/env python3
"""Aggregate catalog coverage for review; never export donor rows."""
from pathlib import Path
import os, sys, json, time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pandas as pd
from app_helpers.control_calibrated.registry import ROOT
from app_helpers.control_calibrated.inputs import atomic_json


def main():
    started = time.perf_counter()
    out = APP / "artifacts/control_calibrated_implementation_20261102"
    coverage = pd.concat(
        [pd.read_parquet(p) for p in (ROOT / "coverage").glob("*.parquet")],
        ignore_index=True,
    )
    coverage.to_csv(out / "coverage_by_module_component.csv", index=False)
    keys = ["family", "cohort_scope", "component", "beta"]
    summary = (
        coverage.groupby(keys, dropna=False)
        .agg(
            planned_modules=("module_key", "nunique"),
            structural_cells=("structural", "sum"),
            eligible_cells=("eligible_edges", lambda x: (x > 0).sum()),
            scored_cells=("scored_targets", lambda x: (x > 0).sum()),
            scored_donor_component_rows=("scored_targets", "sum"),
            target_candidate_rows=("target_candidates", "sum"),
            failed_cells=("failed", "sum"),
        )
        .reset_index()
    )
    summary["unavailable_cells"] = summary.structural_cells - summary.scored_cells
    summary["mask"] = "all"
    summary.to_csv(out / "coverage_by_family_cohort_component_beta.csv", index=False)
    cells = {}
    files = 0
    for i, p in enumerate(sorted((ROOT / "masks/module_completion").glob("*.json"))):
        receipt = json.loads(p.read_text())
        family = receipt["module_key"].split("::")[0]
        for mask, record in receipt["masks"].items():
            for item in record["files"]:
                frame = pd.read_parquet(
                    ROOT / item["path"],
                    columns=["component", "possible_edges", "eligible_edges", "gain"],
                )
                grouped = frame.groupby("component", sort=False).agg(
                    possible_edges=("possible_edges", "max"),
                    eligible_edges=("eligible_edges", "max"),
                    scored_targets=("gain", "count"),
                    target_candidates=("gain", "size"),
                )
                for component, row in grouped.iterrows():
                    key = (family, item["cohort_scope"], component, item["beta"], mask)
                    value = cells.setdefault(
                        key,
                        dict(
                            planned_modules=0,
                            structural_cells=0,
                            eligible_cells=0,
                            scored_cells=0,
                            scored_donor_component_rows=0,
                            target_candidate_rows=0,
                            failed_cells=0,
                        ),
                    )
                    value["planned_modules"] += 1
                    value["structural_cells"] += int(row.possible_edges > 0)
                    value["eligible_cells"] += int(row.eligible_edges > 0)
                    value["scored_cells"] += int(row.scored_targets > 0)
                    value["scored_donor_component_rows"] += int(row.scored_targets)
                    value["target_candidate_rows"] += int(row.target_candidates)
                files += 1
        if i % 100 == 0:
            print(
                json.dumps(
                    dict(
                        modules=i + 1,
                        files=files,
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    masked = pd.DataFrame(
        [dict(zip(keys + ["mask"], key), **value) for key, value in cells.items()]
    )
    masked["unavailable_cells"] = masked.structural_cells - masked.scored_cells
    masked.to_csv(
        out / "coverage_by_family_cohort_component_beta_mask.csv", index=False
    )
    completion = json.loads((ROOT / "associations/completion.json").read_text())
    pd.DataFrame(completion["contexts"]).to_csv(
        out / "baseline_correction_contexts.csv", index=False
    )
    atomic_json(
        out / "coverage_receipt.json",
        dict(
            status="complete",
            unmasked_cells=len(coverage),
            masked_files=files,
            masked_summary_rows=len(masked),
            seconds=time.perf_counter() - started,
            privacy="Aggregate module/component coverage only; no donor identifiers or donor-level rows",
        ),
    )


if __name__ == "__main__":
    main()
