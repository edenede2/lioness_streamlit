#!/usr/bin/env python3
"""Batch lossless module blobs into range-addressable immutable containers."""
from pathlib import Path
import sys, json, shutil, argparse, hashlib, time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers.control_calibrated.inputs import atomic_json, sha


def pack(root, limit=99 * 1024**2):
    root = Path(root)
    release = json.loads((root / "release_manifest.json").read_text())
    rows = list(release["modules"].items())
    bins = []
    # First fit by decreasing module size makes few bounded transport objects.
    for key, r in sorted(rows, key=lambda item: item[1]["bytes"], reverse=True):
        unit = r["bytes"] + (root / r["range_path"]).stat().st_size
        found = next((b for b in bins if b["bytes"] + unit <= limit), None)
        if unit > limit:
            raise ValueError("Module exceeds the bounded transport size: " + key)
        if found is None:
            found = dict(bytes=0, modules=[])
            bins.append(found)
        found["modules"].append(key)
        found["bytes"] += unit
    containers = {}
    started = time.perf_counter()
    (root / "containers").mkdir(exist_ok=True)
    for i, b in enumerate(bins):
        relative = f"containers/part_{i:04d}.bin"
        path = root / relative
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as output:
            for key in b["modules"]:
                r = release["modules"][key]
                module = root / "modules" / r["path"]
                sidecar = root / r["range_path"]
                r["container_path"] = relative
                r["module_offset"] = output.tell()
                with module.open("rb") as f:
                    shutil.copyfileobj(f, output, 8 * 1024**2)
                r["range_offset"] = output.tell()
                r["range_length"] = sidecar.stat().st_size
                with sidecar.open("rb") as f:
                    shutil.copyfileobj(f, output, 8 * 1024**2)
        temporary.replace(path)
        containers[relative] = dict(
            sha256=sha(path), bytes=path.stat().st_size, modules=len(b["modules"])
        )
        if i % 10 == 0:
            print(
                json.dumps(
                    dict(
                        stage="transport",
                        containers=i + 1,
                        total=len(bins),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    release.update(transport_layout="range_containers_v1", containers=containers)
    atomic_json(root / "release_manifest.json", release)
    return release


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=APP
        / "artifacts/control_calibrated_hosted_release_20261103/public_bundle",
    )
    p.add_argument("--container-mib", type=int, default=99)
    a = p.parse_args()
    r = pack(a.root, a.container_mib * 1024**2)
    print(
        json.dumps(
            dict(
                status="complete",
                containers=len(r["containers"]),
                bytes=sum(x["bytes"] for x in r["containers"].values()),
            )
        ),
        flush=True,
    )
