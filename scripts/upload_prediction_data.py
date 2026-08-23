#!/usr/bin/env python3
"""Upload only changed prediction bundle files to the app's Google Drive folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from build_drive_index import DRIVE_FOLDER_MIME, DRIVE_LIST_URL, list_visible_items, path_below_root


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"


def raise_drive_error(response, operation: str) -> None:
    """Expose Drive's structured error reason without leaking credentials."""

    if response.ok:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:2_000]
    raise RuntimeError(
        f"Google Drive {operation} failed with HTTP {response.status_code}: "
        f"{json.dumps(detail, sort_keys=True)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--root-folder-id", required=True)
    parser.add_argument("--local-prediction", type=Path, default=Path("data/prediction"))
    parser.add_argument(
        "--remote-directory",
        default="prediction",
        help="Directory below the configured Drive root (for example prediction_targeted).",
    )
    parser.add_argument(
        "--manifest-name",
        default="prediction_public_manifest.json",
        help="Required manifest filename in the local prediction directory.",
    )
    return parser.parse_args()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_folder(session: AuthorizedSession, name: str, parent_id: str) -> dict[str, object]:
    response = session.post(
        DRIVE_FILES_URL,
        params={"supportsAllDrives": "true", "fields": "id,name,mimeType,parents"},
        json={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
        timeout=60,
    )
    raise_drive_error(response, "folder creation")
    return response.json()


def create_file(
    session: AuthorizedSession, path: Path, name: str, parent_id: str
) -> dict[str, object]:
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        response = session.post(
            DRIVE_UPLOAD_URL,
            params={"uploadType": "multipart", "supportsAllDrives": "true", "fields": "id,name,size,md5Checksum,parents"},
            files={
                "metadata": (None, json.dumps({"name": name, "parents": [parent_id]}), "application/json; charset=UTF-8"),
                "file": (name, handle, mime),
            },
            timeout=(30, 600),
        )
    raise_drive_error(response, f"file creation for {name}")
    return response.json()


def update_file(session: AuthorizedSession, path: Path, file_id: str) -> dict[str, object]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        response = session.patch(
            f"{DRIVE_UPLOAD_URL}/{file_id}",
            params={"uploadType": "media", "supportsAllDrives": "true", "fields": "id,name,size,md5Checksum,parents"},
            headers={"Content-Type": mime},
            data=handle,
            timeout=(30, 600),
        )
    raise_drive_error(response, f"file update for {path.name}")
    return response.json()


def main() -> None:
    args = parse_args()
    local_root = args.local_prediction.resolve()
    if not (local_root / args.manifest_name).is_file():
        raise FileNotFoundError("Prediction public manifest is missing from the local bundle")
    remote_directory = str(args.remote_directory).strip("/")
    if not remote_directory or ".." in Path(remote_directory).parts:
        raise ValueError("--remote-directory must be a safe path below the Drive root")
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(args.credentials.read_text(encoding="utf-8")), scopes=[DRIVE_SCOPE]
    )
    session = AuthorizedSession(credentials)
    items, list_requests = list_visible_items(session)
    by_id = {str(item["id"]): item for item in items}
    if args.root_folder_id not in by_id:
        raise RuntimeError("The Drive root is not visible to the service account")
    by_path = {
        relative: item
        for item in items
        if (relative := path_below_root(item, args.root_folder_id, by_id)) is not None
    }
    folder_ids = {"": args.root_folder_id}
    uploaded = updated = reused = created_folders = 0
    for local in sorted(path for path in local_root.rglob("*") if path.is_file()):
        local_relative = local.relative_to(local_root).as_posix()
        remote_relative = f"{remote_directory}/{local_relative}"
        parent_relative = Path(remote_relative).parent.as_posix()
        if parent_relative == ".":
            parent_relative = ""
        current_parent = ""
        for part in Path(parent_relative).parts:
            next_parent = f"{current_parent}/{part}".strip("/")
            if next_parent not in folder_ids:
                existing = by_path.get(next_parent)
                if existing is not None and existing.get("mimeType") == DRIVE_FOLDER_MIME:
                    folder_ids[next_parent] = str(existing["id"])
                else:
                    folder = create_folder(session, part, folder_ids[current_parent])
                    folder_ids[next_parent] = str(folder["id"])
                    created_folders += 1
            current_parent = next_parent
        existing = by_path.get(remote_relative)
        local_md5 = md5_file(local)
        if (
            existing is not None
            and int(existing.get("size") or -1) == local.stat().st_size
            and str(existing.get("md5Checksum") or "").lower() == local_md5
        ):
            reused += 1
            continue
        if existing is None:
            create_file(session, local, local.name, folder_ids[current_parent])
            uploaded += 1
        else:
            update_file(session, local, str(existing["id"]))
            updated += 1
    print(
        json.dumps(
            {
                "drive_list_requests": list_requests,
                "created_folders": created_folders,
                "created_files": uploaded,
                "updated_files": updated,
                "unchanged_files": reused,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
