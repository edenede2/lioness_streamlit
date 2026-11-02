#!/usr/bin/env python3
"""Verify unchanged historical loaders/fixtures without running prediction."""
from pathlib import Path
import os, sys, types, subprocess, json

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pandas as pd
from app_helpers import data as current
from app_helpers.control_calibrated.inputs import atomic_json


def main():
    name = "app_helpers._calibrated_legacy_baseline"
    baseline = types.ModuleType(name)
    baseline.__file__ = str(APP / "app_helpers/data.py")
    baseline.__package__ = "app_helpers"
    sys.modules[name] = baseline
    source = subprocess.check_output(
        ["git", "show", "HEAD:app_helpers/data.py"], cwd=APP, text=True
    )
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
    checks = []
    for family, method in [
        ("full_cohort", "standard"),
        ("full_cohort", "control_anchored"),
        ("control_derived", "control_anchored"),
    ]:
        for loader in ("load_aggregate_statistics", "load_resolved_statistics"):
            a = getattr(current, loader)(method, module_set=family)
            b = getattr(baseline, loader)(method, module_set=family)
            pd.testing.assert_frame_equal(a, b)
            checks.append(
                dict(
                    family=family,
                    method=method,
                    loader=loader,
                    rows=len(a),
                    phenotypes=a.phenotype.nunique(),
                    identical_to_git_HEAD=True,
                )
            )
    for family in current.available_module_sets():
        a = current.load_kegg(module_set=family)
        b = baseline.load_kegg(module_set=family)
        pd.testing.assert_frame_equal(a, b)
        checks.append(
            dict(
                family=family,
                loader="load_kegg",
                rows=len(a),
                identical_to_git_HEAD=True,
                missing_region_p={
                    r: int(a[current.KEGG_REGION_P_COLUMNS[r]].isna().sum())
                    for r in current.module_set_tissues(family)
                },
            )
        )
    for family in ("full_cohort", "control_derived"):
        for rule in ("all", "native_p05", "bh_fdr05"):
            for loader in ("load_aggregate_statistics", "load_resolved_statistics"):
                a = getattr(current, loader)(
                    "bonobo", module_set=family, estimator="bonobo", edge_rule=rule
                )
                b = getattr(baseline, loader)(
                    "bonobo", module_set=family, estimator="bonobo", edge_rule=rule
                )
                pd.testing.assert_frame_equal(a, b)
                checks.append(
                    dict(
                        family=family,
                        method="bonobo",
                        rule=rule,
                        loader=loader,
                        rows=len(a),
                        phenotypes=a.phenotype.nunique(),
                        identical_to_git_HEAD=True,
                    )
                )
    from app_helpers.control_calibrated.inputs import sha
    import hashlib

    baseline_manifest = subprocess.check_output(
        ["git", "show", "HEAD:data/data_manifest.json"], cwd=APP
    )
    assert hashlib.sha256(baseline_manifest).hexdigest() == sha(current.DATA_MANIFEST)
    atomic_json(
        APP
        / "artifacts/control_calibrated_implementation_20261102/legacy_fixture_parity.json",
        dict(
            status="passed",
            checks=checks,
            historical_data_manifest_identical_to_git_HEAD=True,
            legacy_test_limitations=[
                dict(
                    test="test_kegg_contains_valid_whole_and_per_region_statistics",
                    reason="Existing muscle_control_l4 KEGG has 39 unavailable regional p/q rows for each tissue. Values and NA patterns are identical with Git HEAD loaders; historical outputs are preserved.",
                ),
                dict(
                    test="test_lioness_statistics_cover_all_twelve_outcomes",
                    reason="Installed historical score-statistics tables contain 15 numeric outcomes, while this test expects 12 total. Historical all12 FDR aliases still retain their recorded twelve-outcome membership; numbers are preserved.",
                ),
                dict(
                    test="test_bonobo_rules_are_complete_private_and_explicit_about_zero_states",
                    reason="The same obsolete twelve-outcome total-row assertion is applied to installed fifteen-outcome BONOBO tables; all rule-specific tables match Git HEAD loaders.",
                ),
                dict(
                    test="test_manifest_file_hashes_and_github_size_limit",
                    reason="Some entries in the unchanged historical data manifest have content hashes/bytes but omit the optional Parquet rows field required by this test.",
                ),
            ],
            no_prediction_training=True,
        ),
    )
    print(json.dumps(dict(status="passed", identical_tables=len(checks))), flush=True)


if __name__ == "__main__":
    main()
