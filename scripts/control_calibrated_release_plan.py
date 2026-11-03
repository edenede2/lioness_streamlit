#!/usr/bin/env python3
"""Grounded immutable upload plan for the approved calibrated app release.

This helper performs no Drive writes. The connector uploader saves one receipt
per completed file. Existing public data, permissions and predictions are never
replaced. Every item belongs to the explicit summary-only release allowlist.
"""
from pathlib import Path
import json, hashlib, argparse

APP = Path(__file__).resolve().parents[1]
OUT = APP / "artifacts/control_calibrated_hosted_release_20261103"
BUNDLE = OUT / "public_bundle"
RECEIPTS = OUT / "upload_receipts"


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt_path(key):
    return RECEIPTS / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def plan(limit=50, complete=False):
    candidates = []
    release_path = BUNDLE / "release_manifest.json"
    release = json.loads(release_path.read_text()) if release_path.exists() else {}
    if release.get("transport_layout") == "range_containers_v1":
        candidates += [
            (BUNDLE / key, r["sha256"])
            for key, r in sorted(release["containers"].items())
        ]
    else:
        for p in sorted((BUNDLE / "modules").glob("*.receipt.json")):
            r = json.loads(p.read_text())
            path = BUNDLE / "modules" / r["path"]
            candidates.append((path, r["sha256"]))
            if r.get("range_path"):
                candidates.append((BUNDLE / r["range_path"], r["range_sha256"]))
    # Clinical metadata is explicitly approved by the user. R/C clinical values
    # are stripped; identities are existing public pseudonyms only.
    for p in sorted((BUNDLE / "metadata").glob("*.parquet")):
        candidates.append((p, None))
    if complete:
        release = json.loads((BUNDLE / "release_manifest.json").read_text())
        if release["status"] != "complete":
            raise ValueError("Full packing has not finished")
        for directory in ("coverage", "associations"):
            for p in sorted((BUNDLE / directory).iterdir()):
                if p.is_file():
                    candidates.append((p, None))
        for key in (
            "catalog.json",
            "masks/manifest.json",
            "release_manifest.json",
            "manifest.json",
        ):
            candidates.append((BUNDLE / key, None))
    result = []
    for path, expected in candidates:
        key = path.relative_to(BUNDLE).as_posix()
        if key.startswith(("private/", "file_indexes/")) or path.suffix in (
            ".npz",
            ".npy",
        ):
            raise ValueError("Private artifact in release plan")
        if receipt_path(key).exists():
            old = json.loads(receipt_path(key).read_text())
            if old.get("sha256") == (expected or digest(path, "sha256")) and old.get(
                "id"
            ):
                continue
            raise ValueError("Uploaded immutable artifact changed: " + key)
        actual = digest(path, "sha256")
        if expected and expected != actual:
            raise ValueError("Sealed module changed: " + key)
        result.append(
            dict(
                key=key,
                path=str(path),
                sha256=actual,
                md5=digest(path, "md5"),
                size=path.stat().st_size,
                drive_name=key.replace("/", "__"),
                receipt_path=str(receipt_path(key)),
            )
        )
        if len(result) >= limit:
            break
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--complete", action="store_true")
    p.add_argument("--compact", action="store_true")
    a = p.parse_args()
    tasks = plan(a.limit, a.complete)
    if a.compact:
        tasks = [
            dict(
                key=t["key"],
                sha256=t["sha256"],
                md5=t["md5"],
                size=t["size"],
                receipt_name=Path(t["receipt_path"]).name,
            )
            for t in tasks
        ]
    print(json.dumps(tasks, separators=(",", ":")))
