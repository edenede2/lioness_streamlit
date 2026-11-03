#!/usr/bin/env python3
"""Install a completely uploaded, independently verified release index locally.

The live app is unchanged until the user pushes Git. Old index entries, files,
sharing and predictions are preserved. No remote write or Git command runs here.
"""
from pathlib import Path
import sys, json
from datetime import datetime, timezone

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / "scripts")]
from control_calibrated_release_plan import OUT, BUNDLE, plan
from app_helpers.control_calibrated.inputs import atomic_json, sha


def main():
    approval = json.loads((OUT / "publication_approval.json").read_text())
    audit = json.loads((OUT / "release_audit.json").read_text())
    release = json.loads((BUNDLE / "release_manifest.json").read_text())
    state = json.loads((OUT / "drive_publication_state.json").read_text())
    if (
        not approval.get("approved")
        or audit.get("status") != "passed"
        or release["status"] != "complete"
    ):
        raise ValueError("Explicit approval and complete numerical audit are required")
    transport = json.loads((OUT / "transport_audit.json").read_text())
    if transport.get("status") != "passed" or transport.get(
        "release_manifest_sha256"
    ) != sha(BUNDLE / "release_manifest.json"):
        raise ValueError("The current transport must pass a complete byte audit")
    pending = plan(10000, complete=True)
    if pending:
        raise ValueError(f"{len(pending)} artifacts still need upload")
    records = [
        json.loads(p.read_text()) for p in (OUT / "upload_receipts").glob("*.json")
    ]
    if release.get("transport_layout") == "range_containers_v1":
        records = [r for r in records if not r["key"].startswith("modules/")]
    if not records or not all(r.get("reader_verified") for r in records):
        raise ValueError("Every upload must pass the app-identity readback")
    index_path = APP / "drive_file_index.json"
    previous = json.loads(index_path.read_text())
    before = sha(index_path)
    if previous["root_folder_id"] != state["root_folder_id"]:
        raise ValueError("App Drive root changed")
    prefix = "control_calibrated_v2/releases/" + release["release_id"]
    files = dict(previous["files"])
    added = {}
    for r in records:
        key = prefix + "/" + r["key"]
        if (
            not key.startswith(prefix + "/")
            or "/private/" in key
            or key.endswith((".npz", ".npy"))
        ):
            raise ValueError("Private input in upload receipts")
        entry = dict(
            id=r["id"],
            size=r["size"],
            md5=r["md5"],
            sha256=r["sha256"],
            drive_name=r["drive_name"],
            calibrated_release=release["release_id"],
        )
        if key in files and files[key] != entry:
            raise ValueError("Existing immutable release index differs: " + key)
        files[key] = entry
        added[key] = entry
    for key, entry in previous["files"].items():
        if not key.startswith(prefix + "/") and files[key] != entry:
            raise ValueError("Unrelated Drive index entry changed")
    backup = OUT / "drive_file_index.before_release.json"
    if not backup.exists():
        backup.write_bytes(index_path.read_bytes())
    result = dict(
        previous,
        files=dict(sorted(files.items())),
        file_count=len(files),
        total_bytes=sum(int(v.get("size", 0)) for v in files.values()),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        calibrated_release=release["release_id"],
    )
    temporary = index_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    if temporary.stat().st_size >= 95 * 1024**2:
        raise ValueError("Index exceeds the repository file limit")
    temporary.replace(index_path)
    atomic_json(
        APP / "control_calibrated_release.json",
        dict(
            schema_version=1,
            release_id=release["release_id"],
            data_prefix=prefix,
            backend="existing_read_only_google_drive_index",
            manifest_sha256=sha(BUNDLE / "manifest.json"),
            catalog_sha256=sha(BUNDLE / "catalog.json"),
            release_manifest_sha256=sha(BUNDLE / "release_manifest.json"),
            edge_reconstruction="local_only",
            edge_search="local_only",
            prediction=False,
        ),
    )
    atomic_json(
        OUT / "index_installation_receipt.json",
        dict(
            status="complete",
            release_id=release["release_id"],
            previous_index_sha256=before,
            index_sha256=sha(index_path),
            old_entries_preserved=len(previous["files"]),
            release_entries=len(added),
            total_entries=len(files),
            data_prefix=prefix,
            git_pushed=False,
            prediction_entries_changed=False,
            permissions_changed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="complete",
                release_entries=len(added),
                old_entries_preserved=len(previous["files"]),
                git_pushed=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
