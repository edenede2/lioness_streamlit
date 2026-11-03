#!/usr/bin/env python3
"""Resumable, immutable publication to the explicitly approved app folder.

No Git push, existing file replacement, permission change, private-array upload,
prediction publication or unrelated Drive listing is performed.
"""
from pathlib import Path
import sys, json, threading, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / "scripts")]
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from control_calibrated_release_plan import OUT, BUNDLE, plan, receipt_path
from app_helpers.control_calibrated.inputs import atomic_json

BASE = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
_thread = threading.local()
_config = {}


def session():
    if not hasattr(_thread, "session"):
        _thread.session = AuthorizedSession(
            Credentials.from_authorized_user_file(str(_config["token"]))
        )
    return _thread.session


def own_folder_files(s, folder, name=None):
    files = []
    page = None
    q = f"'{folder}' in parents and trashed = false"
    if name:
        q += " and name = '" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"
    while True:
        params = dict(
            q=q,
            pageSize=1000,
            fields="nextPageToken,files(id,name,size,md5Checksum,parents)",
            supportsAllDrives="true",
            includeItemsFromAllDrives="true",
        )
        if page:
            params["pageToken"] = page
        r = s.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        files += data.get("files", [])
        page = data.get("nextPageToken")
        if not page:
            return files


def persist(task, remote):
    if (
        remote.get("name") != task["drive_name"]
        or int(remote.get("size", -1)) != task["size"]
        or remote.get("md5Checksum") != task["md5"]
        or _config["folder"] not in remote.get("parents", [])
    ):
        raise ValueError("Uploaded bytes or destination do not match the approved file")
    atomic_json(
        receipt_path(task["key"]),
        dict(
            task,
            id=remote["id"],
            uploaded_via="Existing app Drive API; immutable resumable upload",
            reader_verified=False,
        ),
    )


def upload(task):
    s = session()
    name = task["drive_name"]
    existing = _config["existing"].get(name, [])
    for attempt in range(3):
        if attempt:
            existing = own_folder_files(s, _config["folder"], name)
        matching = [
            x
            for x in existing
            if x.get("md5Checksum") == task["md5"]
            and int(x.get("size", -1)) == task["size"]
        ]
        if matching:
            persist(task, matching[0])
            return task["size"]
        if existing:
            raise ValueError(
                "An immutable release filename has different existing bytes: " + name
            )
        try:
            r = s.post(
                UPLOAD,
                params={
                    "uploadType": "resumable",
                    "supportsAllDrives": "true",
                    "fields": "id,name,size,md5Checksum,parents",
                },
                headers={
                    "X-Upload-Content-Type": "application/octet-stream",
                    "X-Upload-Content-Length": str(task["size"]),
                },
                json={"name": name, "parents": [_config["folder"]]},
                timeout=60,
            )
            r.raise_for_status()
            location = r.headers["Location"]
            if not location.startswith("https://www.googleapis.com/upload/drive/"):
                raise ValueError("Unexpected Drive resumable-upload host")
            with Path(task["path"]).open("rb") as handle:
                result = s.put(
                    location,
                    data=handle,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(task["size"]),
                    },
                    timeout=(30, 900),
                )
            result.raise_for_status()
            persist(task, result.json())
            return task["size"]
        except Exception:
            if attempt == 2:
                raise
    raise AssertionError("Unreachable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oauth-token", type=Path, default=APP / "google-drive-oauth-token.json"
    )
    p.add_argument("--workers", type=int, choices=[2, 4, 8], default=8)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    approval = json.loads((OUT / "publication_approval.json").read_text())
    state = json.loads((OUT / "drive_publication_state.json").read_text())
    release = json.loads((BUNDLE / "release_manifest.json").read_text())
    if (
        not approval.get("approved")
        or approval["root_folder_id"] != state["root_folder_id"]
        or release["status"] != "complete"
    ):
        raise ValueError(
            "Explicit publication approval and complete artifacts are required"
        )
    _config.update(token=a.oauth_token, folder=state["folder_id"])
    s = session()
    response = s.get(
        BASE + "/" + state["folder_id"],
        params={
            "fields": "id,parents,capabilities(canAddChildren)",
            "supportsAllDrives": "true",
        },
        timeout=30,
    )
    response.raise_for_status()
    folder = response.json()
    if state["root_folder_id"] not in folder.get("parents", []) or not folder.get(
        "capabilities", {}
    ).get("canAddChildren"):
        raise ValueError(
            "Existing OAuth identity cannot upload into the approved release folder"
        )
    if a.check_only:
        print(json.dumps(dict(status="passed", approved_release_folder_writable=True)))
        return
    existing = {}
    for f in own_folder_files(s, state["folder_id"]):
        existing.setdefault(f["name"], []).append(f)
    _config["existing"] = existing
    tasks = plan(limit=10000, complete=True)
    started = time.perf_counter()
    done = 0
    size = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(upload, t): t["key"] for t in tasks}
        for future in as_completed(futures):
            size += future.result()
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(
                    json.dumps(
                        dict(
                            stage="upload",
                            completed=done,
                            total=len(tasks),
                            bytes=size,
                            seconds=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
    atomic_json(
        OUT / "bulk_upload_receipt.json",
        dict(
            status="complete",
            uploaded_or_recovered=done,
            bytes=size,
            seconds=time.perf_counter() - started,
            git_pushed=False,
            permissions_changed=False,
            private_arrays_uploaded=False,
        ),
    )


if __name__ == "__main__":
    main()
