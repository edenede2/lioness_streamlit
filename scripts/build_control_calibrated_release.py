#!/usr/bin/env python3
"""Build an immutable, lossless hosted summary release; never upload or push."""
from pathlib import Path
import os, sys, json, hashlib, shutil, argparse, time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from app_helpers.control_calibrated.inputs import sha, atomic_json
from app_helpers.control_calibrated.registry import APP as APP_ROOT
from app_helpers.control_calibrated.hosted import pack_module, range_receipt, LAYOUT


def module_task(task):
    source, dest, job, index, mask_receipt = task
    source = Path(source)
    dest = Path(dest)
    output = (
        dest
        / "modules"
        / (hashlib.sha256(job["module_key"].encode()).hexdigest()[:20] + ".parquet")
    )
    receipt = output.with_suffix(".receipt.json")
    if output.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if sha(output) == record["sha256"]:
            if not record.get("range_path"):
                sidecar = output.with_suffix(".ranges.json")
                atomic_json(sidecar, range_receipt(output))
                record.update(
                    range_path=str(sidecar.relative_to(dest)), range_sha256=sha(sidecar)
                )
                atomic_json(receipt, record)
            return job["module_key"], record
    sources = []
    for beta in (3, 1):
        key = f"scores/beta{beta}/{job['family']}/maximum_component/{output.name}"
        sources.append((beta, "all", source / key, index[key]))
    for mask, record in mask_receipt.get("masks", {}).items():
        for f in record["files"]:
            if f["cohort_scope"] == "maximum_component":
                sources.append((f["beta"], mask, source / f["path"], f["sha256"]))
    metadata = pd.read_parquet(
        source / "metadata" / (job["input_variant"] + ".parquet")
    )
    record = pack_module(output, job, sources, metadata)
    sidecar = output.with_suffix(".ranges.json")
    atomic_json(sidecar, range_receipt(output))
    record.update(range_path=str(sidecar.relative_to(dest)), range_sha256=sha(sidecar))
    atomic_json(receipt, record)
    return job["module_key"], record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=APP / "data/control_calibrated_v2")
    p.add_argument(
        "--output",
        type=Path,
        default=APP
        / "artifacts/control_calibrated_hosted_release_20261103/public_bundle",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    start = time.perf_counter()
    source = a.source
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "manifest.json").read_text())
    catalog = json.loads((source / "catalog.json").read_text())
    masks = json.loads((source / "masks/manifest.json").read_text())
    completion = json.loads((source / "associations/completion.json").read_text())
    if not (
        original["status"] == masks["status"] == completion["status"] == "complete"
    ):
        raise ValueError("Offline scoring, masks and baseline scan must be complete")
    identity = {
        k: sha(source / k)
        for k in (
            "manifest.json",
            "catalog.json",
            "masks/manifest.json",
            "associations/completion.json",
        )
    }
    release_id = (
        "20261103_"
        + hashlib.sha256(
            json.dumps(dict(layout=LAYOUT, source=identity), sort_keys=True).encode()
        ).hexdigest()[:12]
    )
    index = {}
    for path in (source / "file_indexes").glob("*.json"):
        index.update({f["path"]: f["sha256"] for f in json.loads(path.read_text())})
    jobs = catalog["modules"]
    if a.pilot:
        jobs = [
            min(
                [j for j in jobs if j["family"] == family],
                key=lambda j: j["physical_nodes"],
            )
            for family in catalog["families"]
        ]

    def tasks():
        for job in jobs:
            receipt = (
                source
                / "masks/module_completion"
                / (
                    hashlib.sha256(job["module_key"].encode()).hexdigest()[:20]
                    + ".json"
                )
            )
            name = (
                hashlib.sha256(job["module_key"].encode()).hexdigest()[:20] + ".parquet"
            )
            selected = {
                key: index[key]
                for beta in (3, 1)
                for key in [
                    f"scores/beta{beta}/{job['family']}/maximum_component/{name}"
                ]
            }
            yield str(source), str(out), job, selected, (
                json.loads(receipt.read_text()) if receipt.exists() else {}
            )

    records = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, (key, record) in enumerate(pool.map(module_task, tasks(), chunksize=1)):
            records[key] = record
            if i % 25 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="pack",
                            modules=i + 1,
                            total=len(jobs),
                            bytes=sum(r["bytes"] for r in records.values()),
                            seconds=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
    # Catalog is reduced to application metadata; source node lists, R/C indices
    # and all source/preprocessing paths remain in the private research root.
    public_catalog = dict(
        families=catalog["families"],
        modules=[
            {k: v for k, v in j.items() if k not in ("nodes", "node_indices")}
            for j in jobs
        ],
        datasets={
            k: {
                field: v[field]
                for field in (
                    "dataset",
                    "tissues",
                    "n_donors",
                    "n_nodes",
                    "metadata_file",
                    "allocation_id",
                    "availability_policy",
                )
            }
            for k, v in catalog["datasets"].items()
        },
    )
    atomic_json(out / "catalog.json", public_catalog)
    for variant, info in catalog["datasets"].items():
        frame = pd.read_parquet(source / info["metadata_file"])
        forbidden = {
            "projid",
            "donor",
            "canonical_id",
            "private_id",
            "source_id",
        } & set(frame)
        if forbidden or frame.sample_id.astype(str).str.startswith("ROSMAP-").any():
            raise ValueError("Private source identity in public metadata")
        # Publish R/C role conditions without their individual clinical values.
        keep = {
            "sample_id",
            "role",
            "in_original_analysis",
            "dataset_namespace",
            "discovery_caveat",
        }
        columns = [c for c in frame if c not in keep and not c.startswith("has_")]
        frame.loc[~frame.role.eq("target"), columns] = None
        target = out / info["metadata_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False, compression="zstd")
    for directory in ("coverage",):
        for path in (source / directory).glob("*.parquet"):
            target = out / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in [
        source / "associations/testing_manifest.json",
        source / "associations/completion.json",
        source / "associations/planned_cells.parquet",
        *[source / f["path"] for f in completion["files"]],
    ]:
        target = out / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_json(
        out / "masks/manifest.json",
        {
            k: masks[k]
            for k in (
                "status",
                "specs",
                "modules",
                "unavailable",
                "failed_modules",
                "completed_modules",
                "planned_modules",
            )
        },
    )
    release = dict(
        layout=LAYOUT,
        status="pilot" if a.pilot else "complete",
        release_id=release_id,
        source_hashes=identity,
        modules=records,
        module_count=len(jobs),
        summary_bytes=sum(r["bytes"] for r in records.values()),
        mask_contexts=sum(len(masks["modules"].get(j["module_key"], {})) for j in jobs),
        baseline_testing_manifest_id=completion["analysis_id"],
        capabilities=dict(
            donor_scores=True,
            associations=True,
            frozen_masks=True,
            edge_reconstruction=False,
            edge_search=False,
            prediction=False,
        ),
        privacy="Target-only numeric score summaries and existing public pseudonyms; R/C clinical values removed. No expression/reference/calibration arrays, identity crosswalks or physical source node lists.",
    )
    atomic_json(out / "release_manifest.json", release)
    public_manifest = {
        k: original[k]
        for k in (
            "schema_version",
            "allocation_seed",
            "betas",
            "default_beta",
            "default_units",
            "families",
            "structural_cells",
            "unavailable_cells",
            "failed_cells",
            "prediction_supported",
            "annotation",
        )
    }
    public_manifest.update(
        status="complete" if not a.pilot else "pilot",
        storage_layout=LAYOUT,
        release_id=release_id,
        capabilities=release["capabilities"],
    )
    atomic_json(out / "manifest.json", public_manifest)
    print(
        json.dumps(
            dict(
                status=release["status"],
                release_id=release_id,
                modules=len(jobs),
                bytes=release["summary_bytes"],
                seconds=time.perf_counter() - start,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
