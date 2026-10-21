#!/usr/bin/env python3
"""Publish one approved, immutable CAMEL snapshot; never commit or push Git.

The uploader lists only its own new folder. It neither inventories unrelated
Drive files nor changes sharing. All writes use the existing user OAuth token;
the app's read-only identity independently verifies representative file bytes.
The local index/availability manifest changes only after complete verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

from run_camel_validation import RUN, atomic_json, sha
from upload_prediction_data import create_file, create_folder, md5_file

PREFIX = "control_derived/camel_validation"
DRIVE = "https://www.googleapis.com/drive/v3/files"
READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_thread = threading.local()


def snapshot(root: Path) -> Path:
    source = root / "public_bundle"
    approval = json.loads((root / "publication_approval.json").read_text())
    report = json.loads((source / "release_manifest.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if not approval.get("approved") or report.get("status") != "passed":
        raise ValueError("Explicit approval and a passing release audit are required")
    if set(manifest.get("completed_rules", [])) != set(report["completed_rules"]):
        raise ValueError("Repeat the release audit after the latest statistics milestone")
    target = root / "publication_snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True)
    completed = set(manifest["completed_rules"])
    for relative, info in report["files"].items():
        path = Path(relative)
        if path.parts[0] == "concordance":
            # The app uses narrow review shards, not duplicate 59-MiB tables.
            continue
        if path.parts[0] in {"statistics", "concordance"} and path.parts[1] not in completed:
            continue
        if path.name in {"manifest.json", "release_manifest.json"}:
            continue
        source_path = source / path
        if sha(source_path) != info["sha256"]:
            raise ValueError("A release file changed; rebuild/audit the snapshot")
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, destination)  # atomic analysis replacements cannot mutate this inode
    # Copy (not hard-link) mutable registry files and install the review manifest last.
    atomic_json(manifest, target / "manifest.json")
    review = source / "review"
    if review.exists():
        for p in review.rglob("*"):
            if p.is_file():
                dst = target / p.relative_to(source)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(p, dst)
    atomic_json({"status": "passed", "completed_rules": sorted(completed), "source_audit_sha256": sha(source / "release_manifest.json"),
                 "source_manifest_sha256": sha(source / "manifest.json"), "publication_approved": True}, target / "release_manifest.json")
    return target


def session(token):
    if not hasattr(_thread, "session"):
        _thread.session = AuthorizedSession(Credentials.from_authorized_user_file(str(token)))
    return _thread.session


def list_own_folder(s, folder):
    found = {}
    page = None
    while True:
        params = {"q": f"'{folder}' in parents and trashed = false", "pageSize": 1000,
                  "fields": "nextPageToken,files(id,name,size,md5Checksum,parents)"}
        if page:
            params["pageToken"] = page
        response = s.get(DRIVE, params=params, timeout=60)
        response.raise_for_status()
        body = response.json()
        for item in body.get("files", []):
            if item["name"] in found:
                raise ValueError("Duplicate immutable upload name")
            found[item["name"]] = item
        page = body.get("nextPageToken")
        if not page:
            return found


def install(snapshot_root: Path, index, root: Path):
    stamp = snapshot_root.name
    rollback = root / "rollback" / ("publication_" + stamp)
    target = APP / "data" / PREFIX
    # Keep small review/default caches in GitHub. Other aggregate and all donor
    # files are read lazily through the verified index, not copied into Git.
    chosen = ["sample_metadata.parquet", "cohort_counts.tsv", "scales.parquet", "mdc/resolved.parquet",
              "release_manifest.json", "backend_verification.json"]
    chosen += [str(p.relative_to(snapshot_root)) for p in (snapshot_root / "review").glob("*.parquet")]
    chosen += ["review/manifest.json"]
    for norm in ("standard_pruned", "retained_edge"):
        chosen.append(f"review/associations/all/{norm}/connectivity/cogn_global.parquet")
    # The Drive index must be installed before an availability manifest can
    # expose large files that are intentionally absent from the repository.
    index_path = APP / "drive_file_index.json"
    rollback.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, rollback / "drive_file_index.json")
    # Preserve the existing compact index format; only semantic additions to
    # the approved prefix should enlarge this already substantial registry.
    temporary_index = index_path.with_suffix(".json.partial")
    temporary_index.write_text(json.dumps(index, separators=(",", ":")) + "\n")
    temporary_index.replace(index_path)
    for relative in [*chosen, "manifest.json"]:
        p = snapshot_root / relative
        if not p.exists():
            continue
        destination = target / relative
        if destination.exists():
            previous = rollback / relative
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, previous)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(p, temporary)
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--oauth-token", type=Path, default=APP / "google-drive-oauth-token.json")
    parser.add_argument("--reader-credentials", type=Path, default=APP / "lionessexplorer-05e37b3ea2a6.json")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    sroot = args.snapshot or snapshot(args.root)
    if not sroot.resolve().is_relative_to((args.root / "publication_snapshots").resolve()):
        raise ValueError("Snapshot must belong to this approved run")
    index_path = APP / "drive_file_index.json"
    old_index_hash = sha(index_path)
    index = json.loads(index_path.read_text())
    s = session(args.oauth_token)
    root_id = index["root_folder_id"]
    response = s.get(DRIVE + "/" + root_id, params={"fields": "id,mimeType,capabilities(canAddChildren)"}, timeout=30)
    response.raise_for_status()
    if not response.json().get("capabilities", {}).get("canAddChildren"):
        raise ValueError("Existing root is not writable")
    checkpoint_path = sroot.parent / (sroot.name + "_upload.json")
    state = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    folder = state.get("folder_id")
    if folder:
        response = s.get(DRIVE + "/" + folder, params={"fields": "id,parents"}, timeout=30)
        response.raise_for_status()
        if root_id not in response.json().get("parents", []):
            raise ValueError("Resume folder no longer belongs to the approved app root")
    else:
        folder = create_folder(s, "camel_validation_" + sroot.name, root_id)["id"]
        state = dict(folder_id=folder, uploaded={})
        atomic_json(state, checkpoint_path)
    inventory = list_own_folder(s, folder)
    paths = [p for p in sorted(sroot.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "backend_verification.json", "release_manifest.json"}]
    # A nested review registry is also uploaded last, after its shards.
    paths += [sroot / "review/manifest.json", sroot / "release_manifest.json"]
    def upload(path):
        relative = path.relative_to(sroot).as_posix()
        digest = md5_file(path)
        name = relative.replace("/", "__") + "__" + digest[:12]
        remote = inventory.get(name)
        if remote is None:
            remote = create_file(session(args.oauth_token), path, name, folder, digest)
        if int(remote.get("size", -1)) != path.stat().st_size or remote.get("md5Checksum") != digest:
            raise ValueError("Upload byte count / checksum mismatch")
        return relative, {"id": remote["id"], "size": path.stat().st_size, "md5": digest}
    uploaded = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, path) for path in paths]
        for future in as_completed(futures):
            relative, info = future.result()
            uploaded[relative] = info
            if len(uploaded) % 100 == 0:
                state["uploaded"] = uploaded
                atomic_json(state, checkpoint_path)
                print(json.dumps(dict(uploaded=len(uploaded), total=len(paths))), flush=True)
    # Independent reader identity verifies content, not merely OAuth write access.
    reader = AuthorizedSession(service_account.Credentials.from_service_account_file(str(args.reader_credentials), scopes=[READ_SCOPE]))
    samples = ["sample_metadata.parquet", "modules/M121.parquet", "rint/M121.parquet", "mdc/adjacency/TS_DLPFC/M121.npz"]
    for relative in samples:
        info = uploaded[relative]
        response = reader.get(DRIVE + "/" + info["id"], params={"alt": "media"}, timeout=90)
        response.raise_for_status()
        if len(response.content) != info["size"] or hashlib.md5(response.content, usedforsecurity=False).hexdigest() != info["md5"]:
            raise ValueError("Read-only app identity failed content readback")
    verification = dict(status="passed", immutable_snapshot=sroot.name, files_verified_by_upload_checksum=len(uploaded),
                        read_only_identity_content_samples=samples, sharing_changes=False,
                        generated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(verification, sroot / "backend_verification.json")
    relative, info = upload(sroot / "backend_verification.json")
    uploaded[relative] = info
    manifest = json.loads((sroot / "manifest.json").read_text())
    manifest.update(backend_verified=True, backend_verified_utc=verification["generated_utc"], publication_snapshot=sroot.name)
    atomic_json(manifest, sroot / "manifest.json")
    relative, info = upload(sroot / "manifest.json")
    uploaded[relative] = info
    if sha(index_path) != old_index_hash:
        raise ValueError("Drive index changed concurrently; merge only after reviewing the other writer")
    for relative, info in uploaded.items():
        index["files"][PREFIX + "/" + relative] = info
    index.update(file_count=len(index["files"]), total_bytes=sum(int(v.get("size") or 0) for v in index["files"].values()),
                 generated_at_utc=verification["generated_utc"], refresh_scope="Approved CAMEL incomplete-region validation snapshot; existing files unchanged")
    state.update(status="complete", uploaded=uploaded)
    atomic_json(state, checkpoint_path)
    install(sroot, index, args.root)
    print(json.dumps(dict(status="backend_verified_and_installed", files=len(uploaded), snapshot=sroot.name,
                          completed_rules=manifest["completed_rules"], git_pushed=False)), flush=True)


if __name__ == "__main__":
    main()
