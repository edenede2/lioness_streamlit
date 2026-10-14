#!/usr/bin/env python3
"""Upload only this release's sanitized eigengene shards and shared registries.

No recursive Drive inventory, deletion, permission changes or unrelated writes.
Existing target IDs are grounded in the app's verified Drive index; changed
remote files abort rather than overwrite a newer unindexed release.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "scripts"))
from upload_prediction_data import create_file, create_folder, find_child, update_file  # noqa: E402
from build_drive_index import md5_file, parquet_predicates  # noqa: E402

SOURCE = "single_motor_tissue_complete_case_l3"
BASE = "https://www.googleapis.com/drive/v3/files"


def metadata(session, identifier):
    response = session.get(f"{BASE}/{identifier}", params={"fields": "id,name,mimeType,parents,size,md5Checksum,modifiedTime", "supportsAllDrives": "true"}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    index_path = APP / "drive_file_index.json"
    previous = json.loads(index_path.read_text())
    token_path = APP / "google-drive-oauth-token.json"
    credentials = Credentials.from_authorized_user_file(str(token_path), scopes=["https://www.googleapis.com/auth/drive"])
    if credentials.expired:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Existing app-backend OAuth authorization is unavailable")
    session = AuthorizedSession(credentials)
    reader = AuthorizedSession(service_account.Credentials.from_service_account_file(
        str(APP / "lionessexplorer-05e37b3ea2a6.json"), scopes=["https://www.googleapis.com/auth/drive.readonly"]))
    shared = ["regional_modules/module_details.parquet", "regional_modules/kegg_enrichment.parquet", "regional_modules/manifest.json", "data_manifest.json"]
    # Check all mutable targets before the first write.
    grounded = {}
    for key in shared:
        indexed = previous["files"][key]
        item = metadata(reader, indexed["id"])
        if item["name"] != Path(key).name or item.get("md5Checksum") != indexed.get("md5"):
            raise RuntimeError(f"Remote target changed since the indexed release: {key}")
        grounded[key] = item
    regional_parent = grounded["regional_modules/manifest.json"]["parents"]
    if len(regional_parent) != 1:
        raise RuntimeError("Regional folder parent is ambiguous")
    parent = regional_parent[0]
    root_check = metadata(reader, parent)
    if root_check["name"] != "regional_modules" or previous["root_folder_id"] not in root_check["parents"]:
        raise RuntimeError("Regional folder is outside the configured app backend")
    # Ground/create exactly the source/cohort/score directories.
    for name in (SOURCE, "common_357", "eigengene"):
        children = find_child(session, name, parent)
        if len(children) > 1:
            raise RuntimeError("Duplicate source directory in app backend")
        folder = children[0] if children else create_folder(session, name, parent)
        if folder["mimeType"] != "application/vnd.google-apps.folder":
            raise RuntimeError("Source directory name is occupied by a file")
        parent = folder["id"]
    changed = dict(previous["files"])
    files = [f"regional_modules/{SOURCE}/common_357/eigengene/{t}.parquet" for t in ("VH", "SMA", "QUAD")]
    for key in [*files, *shared]:
        local = APP / "data" / key
        expected_md5 = md5_file(local)
        indexed = previous["files"].get(key)
        if indexed:
            old = grounded.get(key) or metadata(reader, indexed["id"])
            if old.get("md5Checksum") == expected_md5:
                uploaded = old
            else:
                if old.get("md5Checksum") != indexed.get("md5"):
                    raise RuntimeError(f"Remote file changed outside this release: {key}")
                uploaded = update_file(session, local, indexed["id"])
        else:
            existing = find_child(session, local.name, parent)
            if existing:
                if len(existing) != 1 or existing[0].get("md5Checksum") != expected_md5:
                    raise RuntimeError("Unindexed source file differs; refusing replacement")
                uploaded = existing[0]
            else:
                uploaded = create_file(session, local, local.name, parent, expected_md5)
        # Reader identity must see and verify every uploaded file before index commit.
        verified = metadata(reader, uploaded["id"])
        if int(verified["size"]) != local.stat().st_size or verified["md5Checksum"] != expected_md5:
            raise AssertionError("Reader visibility/checksum validation failed")
        entry = {"id": verified["id"], "size": int(verified["size"]), "md5": expected_md5,
                 "modified_time": verified["modifiedTime"]}
        predicates = parquet_predicates(local)
        if predicates:
            entry["parquet_predicates"] = predicates
        changed[key] = entry
        print(json.dumps({"status": "uploaded and reader-verified", "path": key, "bytes": entry["size"]}), flush=True)
    result = {**previous, "files": dict(sorted(changed.items())), "file_count": len(changed),
              "total_bytes": sum(int(row.get("size", 0)) for row in changed.values()),
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "refresh_scope": "Independent motor eigengene release: three score shards plus four shared registries; unrelated entries unchanged"}
    backup = APP.parents[1] / "out/regional_motor_eigengenes/20261014_table3_common357/rollback_regional_metadata/drive_file_index.json"
    if not backup.exists():
        backup.write_bytes(index_path.read_bytes())
    temporary = index_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(index_path)
    print(json.dumps({"status": "validated", "updated_files": 7, "new_donor_metadata_files": 0}))


if __name__ == "__main__":
    main()
