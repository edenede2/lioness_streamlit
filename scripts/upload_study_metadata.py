#!/usr/bin/env python3
"""Copy-on-write publication of the six explicitly approved ROS/MAP fields.

Old Drive files remain untouched. The live application changes only when the new
index/code commit is pushed by the user. No sharing permissions are changed.
"""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

import pandas as pd
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts"))
from build_study_metadata import METADATA_PATHS, OUTPUT, describe
from build_drive_index import md5_file, parquet_predicates
from upload_prediction_data import create_file, create_folder, find_child

BASE = "https://www.googleapis.com/drive/v3/files"


def remote_metadata(session, identifier):
    response = session.get(f"{BASE}/{identifier}", params={
        "fields":"id,name,mimeType,parents,size,md5Checksum,modifiedTime", "supportsAllDrives":"true"}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    index_path = APP / "drive_file_index.json"
    original_bytes = index_path.read_bytes()
    index = json.loads(original_bytes)
    manifest = json.loads((APP / "data/study/manifest.json").read_text())
    if manifest["status"] != "complete" or set(manifest["metadata"]) != set(METADATA_PATHS):
        raise ValueError("The approved metadata set is not complete")
    for key, record in manifest["metadata"].items():
        if describe(APP / "data" / key)["sha256"] != record["sha256"]:
            raise ValueError("A staged metadata file changed after validation")
    credentials = Credentials.from_authorized_user_file(
        str(APP / "google-drive-oauth-token.json"), scopes=["https://www.googleapis.com/auth/drive"])
    if credentials.expired:
        credentials.refresh(Request())
    writer = AuthorizedSession(credentials)
    reader = AuthorizedSession(service_account.Credentials.from_service_account_file(
        str(APP / "lionessexplorer-05e37b3ea2a6.json"), scopes=["https://www.googleapis.com/auth/drive.readonly"]))
    root = index["root_folder_id"]
    shared = (*METADATA_PATHS, "data_manifest.json", "regional_modules/manifest.json")
    for key in shared:
        entry = index["files"][key]
        actual = remote_metadata(reader, entry["id"])
        if actual["name"] != entry.get("drive_name", Path(key).name) or actual.get("md5Checksum") != entry["md5"]:
            raise ValueError(f"The indexed remote file changed: {key}")
    folders = find_child(writer, "study_metadata", root)
    if len(folders) > 1:
        raise ValueError("Ambiguous study metadata destination")
    folder = folders[0] if folders else create_folder(writer, "study_metadata", root)
    if folder["mimeType"] != "application/vnd.google-apps.folder":
        raise ValueError("The study metadata destination is not a folder")
    files = dict(index["files"])
    paths = (*METADATA_PATHS, "study/sample_study.parquet", "study/manifest.json", "data_manifest.json", "regional_modules/manifest.json")
    for key in paths:
        path = APP / "data" / key
        checksum = md5_file(path)
        name = "20261021_" + checksum[:12] + "__" + key.replace("/", "__")
        found = find_child(writer, name, folder["id"])
        if found:
            if len(found) != 1 or found[0].get("md5Checksum") != checksum:
                raise ValueError("An existing immutable release file differs")
            uploaded = found[0]
        else:
            uploaded = create_file(writer, path, name, folder["id"], checksum)
        verified = remote_metadata(reader, uploaded["id"])
        if verified.get("md5Checksum") != checksum or int(verified["size"]) != path.stat().st_size:
            raise ValueError("Backend reader/checksum verification failed")
        entry = dict(id=verified["id"], size=int(verified["size"]), md5=checksum,
                     modified_time=verified["modifiedTime"], drive_name=verified["name"])
        predicates = parquet_predicates(path)
        if predicates:
            entry["parquet_predicates"] = predicates
        files[key] = entry
        print(json.dumps({"status":"verified", "path":key, "bytes":entry["size"]}), flush=True)
    # One bounded media read additionally verifies the cloud reader's table/schema.
    response = reader.get(f"{BASE}/{files['sample_metadata.parquet']['id']}", params={"alt":"media"}, timeout=30)
    response.raise_for_status()
    table = pd.read_parquet(io.BytesIO(response.content))
    if table.study.value_counts().to_dict() != {"MAP":241,"ROS":209} or {"donor","projid"}.intersection(table):
        raise ValueError("Cloud metadata content failed study/privacy validation")
    if index_path.read_bytes() != original_bytes:
        raise ValueError("Drive index changed concurrently; no index replacement was made")
    backup = OUTPUT / "rollback/drive_file_index.json"
    if not backup.exists():
        backup.write_bytes(original_bytes)
    payload = {**index, "files":dict(sorted(files.items())), "file_count":len(files),
               "total_bytes":sum(int(v.get("size",0)) for v in files.values()),
               "generated_at_utc":datetime.now(timezone.utc).isoformat(),
               "refresh_scope":"Approved ROS/MAP labels in six existing metadata files; predictions untouched"}
    temporary = index_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(index_path)
    print(json.dumps({"status":"complete", "files":len(paths), "cloud_counts":{"ROS":209,"MAP":241}}))


if __name__ == "__main__":
    main()
