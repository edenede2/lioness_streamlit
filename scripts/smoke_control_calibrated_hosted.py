#!/usr/bin/env python3
"""Exercise real hosted reads and compare them with the frozen research bundle.

Run after installing the release index, with LIONESS_FORCE_DRIVE=1 and the
existing app read-only credentials. No remote writes or predictions run.
"""
import json
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pandas as pd
from app_helpers import drive_data as D
from app_helpers.control_calibrated import adapter as A, hosted as H, registry as C
from app_helpers.control_calibrated import inspection as I
from app_helpers.control_calibrated.inputs import atomic_json

OUT = APP / "artifacts/control_calibrated_hosted_release_20261103"


def canonical(frame):
    keys = [
        k
        for k in ("sample_id", "component", "metric_family", "outcome", "method")
        if k in frame
    ]
    return frame.sort_values(keys).reset_index(drop=True).sort_index(axis=1)


def main():
    if (
        os.environ.get("LIONESS_FORCE_DRIVE") != "1"
        or D.DATA_DIR == D.LOCAL_DATA_DIR.resolve()
    ):
        raise ValueError(
            "This check requires an isolated hosted cache and forced Drive reads"
        )
    assert H.is_hosted(C.ROOT)
    catalog = A.catalog()
    masks = H.read_json(C.ROOT / "masks/manifest.json")
    checks = []
    for family in catalog["families"]:
        jobs = [j for j in catalog["modules"] if j["family"] == family]
        masked_jobs = [
            j
            for j in jobs
            if any(
                r.get("status") == "complete"
                for r in masks["modules"].get(j["module_key"], {}).values()
            )
        ]
        job = min(masked_jobs or jobs, key=lambda j: j["physical_nodes"])
        selected_masks = ["all"]
        available_masks = [
            k
            for k, r in masks["modules"].get(job["module_key"], {}).items()
            if r.get("status") == "complete"
        ]
        if available_masks:
            selected_masks.append(available_masks[0])
        calls = 0
        for beta in (3, 1):
            for scope in (
                "complete_450",
                "incomplete_region_validation",
                "maximum_component",
            ):
                for mask in selected_masks:
                    kwargs = dict(
                        module_set=family,
                        module=job["module"],
                        regional_tissue=job["regional_tissue"],
                        estimator=f"calibrated_beta{beta}",
                        cohort_scope=scope,
                        differential_edge_rule=mask,
                        metric_families=("gain", "loss"),
                    )
                    hosted = A.scores(**kwargs)
                    local = A.scores(root=C.RESEARCH_ROOT, **kwargs)
                    pd.testing.assert_frame_equal(
                        canonical(hosted),
                        canonical(local),
                        check_exact=True,
                        check_dtype=False,
                    )
                    assert hosted.empty or hosted.role.eq("target").all()
                    calls += 1
                kwargs.pop("differential_edge_rule")
                kwargs.pop("metric_families")
                kwargs.update(metric_family="gain", phenotype="cogn_global")
                hosted_baseline = A.baseline_associations(**kwargs)
                local_baseline = A.baseline_associations(root=C.RESEARCH_ROOT, **kwargs)
                pd.testing.assert_frame_equal(
                    canonical(hosted_baseline),
                    canonical(local_baseline),
                    check_exact=True,
                )
        check = dict(
            family=family,
            module_key=job["module_key"],
            score_contexts=calls,
            betas=[3, 1],
            baseline_contexts=6,
            exact_source_parity=True,
        )
        checks.append(check)
        print(json.dumps(check), flush=True)
    try:
        I.prepare_reference_caches()
    except ValueError as error:
        assert "local" in str(error)
    else:
        raise AssertionError("Hosted edge reconstruction must remain local")
    assert C.source_spec("calibrated_beta3")["capabilities"]["prediction"] is False
    assert C.source_spec("calibrated_beta3")["capabilities"]["edge_inspection"] is False
    assert not list(C.ROOT.rglob("*.npy")) and not list(C.ROOT.rglob("*.npz"))
    result = dict(
        status="passed",
        release_id=H.read_json(C.ROOT / "release_manifest.json")["release_id"],
        real_read_only_backend=True,
        local_fallback_used=False,
        families=checks,
        score_contexts=sum(r["score_contexts"] for r in checks),
        baseline_contexts=6 * len(checks),
        edge_reconstruction="local only",
        prediction=False,
    )
    atomic_json(OUT / "hosted_backend_smoke.json", result)
    print(json.dumps(dict(status="passed", families=len(checks))), flush=True)


if __name__ == "__main__":
    main()
