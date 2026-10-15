#!/usr/bin/env python3
"""Publish approved pathology metadata and additive statistics to the app backend.

Grounds exact existing targets/checksums; no recursive listing, permission changes,
prediction writes, or unrelated Drive/index edits. New statistics share one folder.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts"))
from upload_prediction_data import create_file, create_folder, find_child
from build_drive_index import md5_file, parquet_predicates

BASE = "https://www.googleapis.com/drive/v3/files"


def metadata(session, identifier):
    response = session.get(f"{BASE}/{identifier}", params={"fields":"id,name,mimeType,parents,size,md5Checksum,modifiedTime", "supportsAllDrives":"true"}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    path = APP / "drive_file_index.json"
    previous = json.loads(path.read_text())
    manifest = json.loads((APP / "data/pathology/manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["statistics"]) != 22:
        raise ValueError("Pathology catalog is not reconciled and complete")
    credentials = Credentials.from_authorized_user_file(str(APP / "google-drive-oauth-token.json"), scopes=["https://www.googleapis.com/auth/drive"])
    if credentials.expired:
        credentials.refresh(Request())
    session = AuthorizedSession(credentials)
    reader = AuthorizedSession(service_account.Credentials.from_service_account_file(
        str(APP / "lionessexplorer-05e37b3ea2a6.json"), scopes=["https://www.googleapis.com/auth/drive.readonly"]))
    shared = [*manifest["metadata"], "data_manifest.json", "regional_modules/manifest.json"]
    grounded = {}
    for key in shared:
        indexed = previous["files"][key]
        remote = metadata(reader, indexed["id"])
        if remote["name"] != indexed.get("drive_name", Path(key).name) or remote.get("md5Checksum") != indexed.get("md5"):
            raise ValueError(f"Remote file changed outside the indexed release: {key}")
        grounded[key] = remote
    root = previous["root_folder_id"]
    parents = grounded["data_manifest.json"]["parents"]
    if root not in parents and not any(root in metadata(reader, parent).get("parents", []) for parent in parents):
        raise ValueError("Configured Drive root does not contain the indexed app manifest")
    folders = find_child(session, "pathology", root)
    if len(folders) > 1:
        raise ValueError("Duplicate pathology backend directory")
    folder = folders[0] if folders else create_folder(session, "pathology", root)
    if folder["mimeType"] != "application/vnd.google-apps.folder":
        raise ValueError("Pathology directory name is occupied by a non-folder")
    changed = dict(previous["files"])
    # Data first; manifests last. One flat folder avoids a large directory-list walk.
    new = [*manifest["statistics"], "pathology/manifest.json"]
    for key in [*new[:-1], *shared[:-2], new[-1], *shared[-2:]]:
        local = APP / "data" / key
        expected = md5_file(local)
        indexed = previous["files"].get(key)
        if indexed:
            old = grounded.get(key) or metadata(reader, indexed["id"])
            if old.get("md5Checksum") == expected:
                result = old
            else:
                if old.get("md5Checksum") != indexed["md5"]:
                    raise ValueError(f"Remote file changed: {key}")
                result = None
        else:
            result = None
        if result is None:
            # Immutable copy-on-write: the deployed app's old index/IDs remain
            # valid until the user pushes this release's new local index.
            name = "20261015_" + expected[:12] + "__" + key.replace("/", "__")
            found = find_child(session, name, folder["id"])
            if found:
                if len(found) != 1 or found[0].get("md5Checksum") != expected:
                    raise ValueError("An unindexed release file differs; refusing replacement")
                result = found[0]
            else:
                result = create_file(session, local, name, folder["id"], expected)
        verified = metadata(reader, result["id"])
        if verified.get("md5Checksum") != expected or int(verified["size"]) != local.stat().st_size:
            raise AssertionError("Read-only backend checksum/access check failed")
        entry = dict(id=verified["id"], size=int(verified["size"]), md5=expected,
                     modified_time=verified["modifiedTime"], drive_name=verified["name"])
        predicates = parquet_predicates(local)
        if predicates:
            entry["parquet_predicates"] = predicates
        changed[key] = entry
        print(json.dumps({"status":"reader_verified", "path":key,"bytes":entry["size"]}), flush=True)
    backup = APP.parents[1] / "out/pathology_app_20261015/rollback/drive_file_index.json"
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    result = {**previous, "files":dict(sorted(changed.items())), "file_count":len(changed),
              "total_bytes":sum(int(v.get("size",0)) for v in changed.values()),
              "generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "refresh_scope":"Three approved pathology columns in six existing metadata files plus additive score statistics; predictions untouched"}
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(path)
    print(json.dumps({"status":"complete","updated_files":len(shared)+len(new)}))


if __name__ == "__main__":
    main()
