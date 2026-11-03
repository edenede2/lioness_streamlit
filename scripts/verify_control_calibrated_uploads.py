#!/usr/bin/env python3
"""Independently verify connector uploads using the app's read-only identity."""
from pathlib import Path
import sys, json, argparse, threading, hashlib
from concurrent.futures import ThreadPoolExecutor

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from app_helpers.control_calibrated.inputs import atomic_json
from app_helpers.control_calibrated.hosted import range_receipt
from control_calibrated_release_plan import plan, receipt_path

OUT = APP / "artifacts/control_calibrated_hosted_release_20261103"
_thread = threading.local()


def session(credentials):
    if not hasattr(_thread, "session"):
        _thread.session = AuthorizedSession(
            service_account.Credentials.from_service_account_file(
                str(credentials),
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
        )
    return _thread.session


def verify(task):
    path, credentials, folder = task
    record = json.loads(path.read_text())
    s = session(credentials)
    response = s.get(
        "https://www.googleapis.com/drive/v3/files/" + record["id"],
        params={
            "fields": "id,name,size,md5Checksum,parents,capabilities(canDownload)",
            "supportsAllDrives": "true",
        },
        timeout=30,
    )
    response.raise_for_status()
    meta = response.json()
    if (
        meta.get("name") != record["drive_name"]
        or int(meta.get("size", -1)) != record["size"]
        or meta.get("md5Checksum") != record["md5"]
        or folder not in meta.get("parents", [])
    ):
        raise ValueError(
            "Read-only backend metadata/content mismatch: " + record["key"]
        )
    if meta.get("capabilities", {}).get("canDownload") is False:
        raise ValueError("App read-only identity cannot download " + record["key"])
    record.update(
        reader_verified=True,
        reader_verification="App service account: exact file ID, parent, size, name and Google-computed MD5",
        remote_modified_time=meta.get("modifiedTime"),
    )
    atomic_json(path, record)
    return record["key"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--credentials", type=Path, default=APP / "lionessexplorer-05e37b3ea2a6.json"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--byte-probe", action="store_true")
    p.add_argument("--recover", action="store_true")
    a = p.parse_args()
    state = json.loads((OUT / "drive_publication_state.json").read_text())
    paths = [
        p
        for p in sorted((OUT / "upload_receipts").glob("*.json"))
        if not json.loads(p.read_text()).get("reader_verified")
    ]
    if a.recover:
        s = session(a.credentials)
        items = []
        page = None
        while True:
            params = dict(
                q=f"'{state['folder_id']}' in parents and trashed = false",
                pageSize=1000,
                fields="nextPageToken,files(id,name,size,md5Checksum,parents)",
                supportsAllDrives="true",
            )
            if page:
                params["pageToken"] = page
            response = s.get(
                "https://www.googleapis.com/drive/v3/files", params=params, timeout=60
            )
            response.raise_for_status()
            body = response.json()
            items += body.get("files", [])
            page = body.get("nextPageToken")
            if not page:
                break
        by_name = {}
        for item in items:
            by_name.setdefault(item["name"], []).append(item)
        recovered = 0
        for task in plan(10000, complete=True):
            matching = [
                x
                for x in by_name.get(task["drive_name"], [])
                if x.get("md5Checksum") == task["md5"]
                and int(x.get("size", -1)) == task["size"]
            ]
            if matching:
                atomic_json(
                    receipt_path(task["key"]),
                    dict(
                        task,
                        id=matching[0]["id"],
                        uploaded_via="Recovered approved connector upload",
                        reader_verified=True,
                    ),
                )
                recovered += 1
        print(json.dumps(dict(recovered_upload_receipts=recovered)), flush=True)
        paths = [
            p
            for p in sorted((OUT / "upload_receipts").glob("*.json"))
            if not json.loads(p.read_text()).get("reader_verified")
        ]
    if a.limit:
        paths = paths[: a.limit]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, key in enumerate(
            pool.map(verify, [(p, a.credentials, state["folder_id"]) for p in paths])
        ):
            if i % 25 == 0:
                print(
                    json.dumps(
                        dict(status="reader_verified", files=i + 1, total=len(paths))
                    ),
                    flush=True,
                )
    if a.byte_probe:
        receipts = [
            json.loads(p.read_text()) for p in (OUT / "upload_receipts").glob("*.json")
        ]
        release = json.loads((OUT / "public_bundle/release_manifest.json").read_text())
        containers = {
            r["key"]: r for r in receipts if r["key"].startswith("containers/")
        }
        if containers:
            record = next(
                r
                for r in release["modules"].values()
                if r["container_path"] in containers
            )
            module = containers[record["container_path"]]
            ranges = json.loads(
                (OUT / "public_bundle" / record["range_path"]).read_text()
            )
            offset = record["module_offset"]
        else:
            module = next(
                r
                for r in receipts
                if r["key"].endswith(".parquet") and r["key"].startswith("modules/")
            )
            ranges = range_receipt(Path(module["path"]))
            offset = 0
        block = ranges["footer"]
        s = session(a.credentials)
        response = s.get(
            "https://www.googleapis.com/drive/v3/files/" + module["id"],
            params={"alt": "media", "supportsAllDrives": "true"},
            headers={
                "Range": f"bytes={offset+block['offset']}-{offset+block['offset']+block['length']-1}"
            },
            timeout=60,
        )
        response.raise_for_status()
        if (
            response.status_code != 206
            or hashlib.sha256(response.content).hexdigest() != block["sha256"]
        ):
            raise ValueError("Real app-identity range read failed")
        atomic_json(
            OUT / "backend_range_probe.json",
            dict(
                status="passed",
                file=module["key"],
                bytes=len(response.content),
                full_module_bytes=module["size"],
                container_offset=offset,
                transport_layout=release.get("transport_layout"),
                sha256=block["sha256"],
                read_only_app_identity=True,
            ),
        )
    print(json.dumps(dict(status="passed", verified_files=len(paths))), flush=True)


if __name__ == "__main__":
    main()
