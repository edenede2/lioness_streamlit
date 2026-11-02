#!/usr/bin/env python3
"""Audit every masked artifact and reconstruct selected donor contributions."""
from pathlib import Path
import os, sys, json, time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.control_calibrated import adapter as A
from app_helpers.control_calibrated.registry import ROOT
from app_helpers.control_calibrated.inputs import sha, atomic_json, atomic_parquet
from app_helpers.control_calibrated.inspection import iter_edges


def main():
    started = time.perf_counter()
    out = APP / "artifacts/control_calibrated_implementation_20261102"
    manifest = json.loads((ROOT / "masks/manifest.json").read_text())
    catalog = A.catalog()
    expected_modules = sum(
        e["modules"] for e in catalog["families"].values() if not e["regional"]
    )
    assert (
        manifest["status"] == "complete"
        and manifest["completed_modules"] == expected_modules
        and manifest["failed_modules"] == 0
    )
    file_count = 0
    row_count = 0
    receipts = {}
    scopes = {}
    for family, entry in catalog["families"].items():
        if entry["regional"]:
            continue
        for scope in (
            "complete_450",
            "incomplete_region_validation",
            "maximum_component",
        ):
            scopes[family, scope] = len(A.metadata(family, scope))
    for index, p in enumerate(
        sorted((ROOT / "masks/module_completion").glob("*.json"))
    ):
        receipt = json.loads(p.read_text())
        key = receipt["module_key"]
        receipts[key] = receipt
        family = key.split("::")[0]
        assert receipt["status"] == "complete"
        assert not receipt.get("unavailable")
        assert set(receipt["masks"]) == set(manifest["specs"][family])
        for mask, record in receipt["masks"].items():
            assert record["status"] == "complete" and len(record["files"]) == 6
            assert {(f["beta"], f["cohort_scope"]) for f in record["files"]} == {
                (b, s)
                for b in (3, 1)
                for s in (
                    "complete_450",
                    "incomplete_region_validation",
                    "maximum_component",
                )
            }
            for f in record["files"]:
                path = ROOT / f["path"]
                assert sha(path) == f["sha256"]
                table = pq.ParquetFile(path)
                assert {
                    "gain",
                    "loss",
                    "net_change",
                    "magnitude_atypicality",
                    "role",
                    "beta",
                    "allocation_id",
                    "edge_universe_sha256",
                    "sparse_endpoint_cells_imputed",
                } <= set(table.schema_arrow.names)
                rows = table.metadata.num_rows
                assert rows % scopes[family, f["cohort_scope"]] == 0
                file_count += 1
                row_count += rows
        if index % 100 == 0:
            print(
                json.dumps(
                    dict(
                        stage="mask_integrity",
                        modules=index + 1,
                        files=file_count,
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    expected = sum(
        catalog["families"][family]["modules"] * len(specs) * 6
        for family, specs in manifest["specs"].items()
    )
    assert file_count == expected
    checks = []
    rule = "ad_control_discovery_effect__hedges_g__fixed_0.40__either"
    for family, entry in catalog["families"].items():
        if entry["regional"]:
            continue
        jobs = [
            j
            for j in catalog["modules"]
            if j["family"] == family
            and receipts[j["module_key"]]["masks"][rule]["physical_edges"] > 0
        ]
        job = min(jobs, key=lambda j: j["physical_nodes"])
        sample = A.metadata(family).sample_id.iloc[0]
        for beta in (3, 1):
            estimator = f"calibrated_beta{beta}"
            edges = pd.concat(
                list(
                    iter_edges(
                        family, job["module"], [sample], estimator=estimator, mask=rule
                    )
                ),
                ignore_index=True,
            )
            stored = A.scores(
                module_set=family,
                module=job["module"],
                estimator=estimator,
                differential_edge_rule=rule,
                component="all",
            )
            stored = (
                stored[stored.sample_id.eq(sample)]
                .set_index("metric_family")
                .metric_raw
            )
            gain = edges.gain_contribution.sum()
            loss = edges.loss_contribution.sum()
            den = int(edges.aggregate_denominator.iloc[0])
            assert int(edges.eligible.sum()) == den
            for metric, actual in [
                ("gain", gain),
                ("loss", loss),
                ("net_change", gain - loss),
                ("magnitude_atypicality", gain + loss),
            ]:
                np.testing.assert_allclose(
                    actual, stored[metric], rtol=2e-10, atol=2e-11
                )
            checks.append(
                dict(
                    family=family,
                    module=job["module"],
                    beta=beta,
                    mask=rule,
                    physical_edges=len(edges),
                    eligible_edges=den,
                    max_absolute_error=max(
                        abs(gain - stored.gain), abs(loss - stored.loss)
                    ),
                )
            )
    atomic_parquet(out / "masked_contribution_checks.parquet", pd.DataFrame(checks))
    atomic_json(
        out / "mask_verification.json",
        dict(
            status="passed",
            modules=len(receipts),
            mask_contexts=sum(len(r["masks"]) for r in receipts.values()),
            files=file_count,
            donor_component_rows=row_count,
            reconstruction_checks=len(checks),
            seconds=time.perf_counter() - started,
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed",
                modules=len(receipts),
                files=file_count,
                seconds=time.perf_counter() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
