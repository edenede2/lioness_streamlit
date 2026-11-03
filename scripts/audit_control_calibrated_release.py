#!/usr/bin/env python3
"""Audit every hosted numeric row, source receipt, baseline file and privacy gate."""
from pathlib import Path
import os, sys, json, hashlib, argparse, time

for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.control_calibrated.inputs import sha, atomic_json
from app_helpers.control_calibrated import hosted as H


def audit_module(task):
    root, job, record = task
    root = Path(root)
    path = H.module_path(root, job["module_key"])
    if sha(path) != record["sha256"]:
        raise ValueError("Hosted module changed")
    if sha(root / record["range_path"]) != record["range_sha256"]:
        raise ValueError("Range receipt changed")
    contexts = {}
    groups = 0
    rows = 0
    with pq.ParquetFile(path) as f:
        header = json.loads(f.metadata.metadata[b"calibrated_header"])
        assert header["module_key"] == job["module_key"]
        for beta, masks in header["entries"].items():
            for mask, components in masks.items():
                for item in components.values():
                    contexts[item["aggregation"]] = (
                        beta,
                        item["context"]["edge_universe_sha256"],
                    )
                    assert item["context"]["module_key"] == job["module_key"]
        for i in range(f.num_row_groups):
            frame = f.read_row_group(i).to_pandas()
            aggregation = frame.pop("aggregation_id").iloc[0]
            beta, universe = contexts[aggregation]
            h = hashlib.sha256()
            h.update(beta.encode())
            h.update(universe.encode())
            h.update(
                pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
            )
            assert h.hexdigest() == aggregation
            assert frame.role.eq("target").all()
            assert frame.target_row.ge(0).all()
            groups += 1
            rows += len(frame)
        assert groups == header["unique_aggregations"] and rows == header["rows"]
        source_files = len(header["source_files"])
    return dict(
        module_key=job["module_key"],
        groups=groups,
        numeric_rows=rows,
        source_files=source_files,
        bytes=path.stat().st_size,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=APP
        / "artifacts/control_calibrated_hosted_release_20261103/public_bundle",
    )
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.root
    release = json.loads((root / "release_manifest.json").read_text())
    catalog = json.loads((root / "catalog.json").read_text())
    assert (
        release["status"] == "complete"
        and len(catalog["modules"]) == release["module_count"] == 1697
    )
    for d in catalog["datasets"].values():
        assert not {
            "nodes",
            "role_indices",
            "source_sha256",
            "expression_sha256",
            "presence_sha256",
        } & set(d)
    for j in catalog["modules"]:
        assert not {"nodes", "node_indices"} & set(j)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(
            pool.map(
                audit_module,
                [
                    (str(root), j, release["modules"][j["module_key"]])
                    for j in catalog["modules"]
                ],
                chunksize=1,
            )
        ):
            rows.append(row)
            if i % 50 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="numeric_integrity",
                            modules=i + 1,
                            seconds=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
    source = APP / "data/control_calibrated_v2"
    clinical_checks = []
    for variant, info in catalog["datasets"].items():
        public = pd.read_parquet(root / info["metadata_file"])
        original = pd.read_parquet(source / info["metadata_file"])
        assert public.sample_id.equals(original.sample_id)
        assert not public.sample_id.astype(str).str.startswith("ROSMAP-").any()
        target = public.role.eq("target")
        pd.testing.assert_frame_equal(
            public.loc[target].reset_index(drop=True),
            original.loc[target].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        clinical = [
            c
            for c in public
            if c
            not in {
                "sample_id",
                "role",
                "in_original_analysis",
                "dataset_namespace",
                "discovery_caveat",
            }
            and not c.startswith("has_")
        ]
        assert public.loc[~target, clinical].isna().all().all()
        clinical_checks.append(
            dict(
                variant=variant,
                target_rows=int(target.sum()),
                R_C_clinical_values_published=False,
            )
        )
    completion = json.loads((root / "associations/completion.json").read_text())
    for record in completion["files"]:
        assert sha(root / record["path"]) == record["sha256"]
    assert sum(r["source_files"] for r in rows) == 28482
    out = APP / "artifacts/control_calibrated_hosted_release_20261103"
    pd.DataFrame(rows).to_csv(out / "hosted_module_integrity.csv", index=False)
    atomic_json(
        out / "release_audit.json",
        dict(
            status="passed",
            modules=len(rows),
            source_maximum_scope_frames=sum(r["source_files"] for r in rows),
            all_scope_equivalent_artifacts=3 * sum(r["source_files"] for r in rows),
            unique_numeric_aggregations=sum(r["groups"] for r in rows),
            numeric_rows=sum(r["numeric_rows"] for r in rows),
            bytes=sum(r["bytes"] for r in rows),
            clinical_checks=clinical_checks,
            baseline_files=len(completion["files"]),
            seconds=time.perf_counter() - started,
            edge_reconstruction="local only by user instruction",
            private_arrays_published=False,
            prediction=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed",
                modules=len(rows),
                seconds=time.perf_counter() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
