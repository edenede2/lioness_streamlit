#!/usr/bin/env python3
"""Upload only changed prediction bundle files to the app's Google Drive folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2 import service_account

from build_drive_index import DRIVE_FOLDER_MIME, DRIVE_LIST_URL, list_visible_items, path_below_root


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
TRANSIENT_DRIVE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_DRIVE_ATTEMPTS = 6


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


def _drive_query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_child(
    session: AuthorizedSession, name: str, parent_id: str
) -> list[dict[str, object]]:
    query = (
        f"'{_drive_query_literal(parent_id)}' in parents and "
        f"name = '{_drive_query_literal(name)}' and trashed = false"
    )
    response = session.get(
        DRIVE_FILES_URL,
        params={
            "q": query,
            "pageSize": 100,
            "fields": "files(id,name,mimeType,parents,size,md5Checksum)",
            "spaces": "drive",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        },
        timeout=60,
    )
    if not response.ok:
        return []
    return list(response.json().get("files", []))


def _wait_before_retry(response, operation: str, attempt: int) -> None:
    retry_after = str(response.headers.get("Retry-After", "")).strip()
    try:
        delay = float(retry_after) if retry_after else min(2**attempt, 60)
    except ValueError:
        delay = min(2**attempt, 60)
    print(
        json.dumps(
            {
                "status": "drive_retry",
                "operation": operation,
                "http_status": response.status_code,
                "attempt": attempt + 1,
                "max_attempts": MAX_DRIVE_ATTEMPTS,
                "delay_seconds": delay,
            }
        ),
        flush=True,
    )
    time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    authentication = parser.add_mutually_exclusive_group(required=True)
    authentication.add_argument(
        "--credentials",
        type=Path,
        help="Service-account JSON (read/update only in a user-owned My Drive folder).",
    )
    authentication.add_argument(
        "--oauth-client",
        type=Path,
        help="Installed/Desktop OAuth client JSON used to upload as a Google user.",
    )
    parser.add_argument(
        "--oauth-token",
        type=Path,
        default=Path("google-drive-oauth-token.json"),
        help="Ignored local refresh-token file for user OAuth.",
    )
    parser.add_argument(
        "--inventory-credentials",
        type=Path,
        help=(
            "Optional read-only service-account JSON used only to inventory the shared "
            "LIONESS folder. This avoids listing an OAuth user's unrelated My Drive files."
        ),
    )
    parser.add_argument(
        "--oauth-port",
        type=int,
        default=8765,
        help="Local callback port; forward this port when authorizing over SSH.",
    )
    parser.add_argument(
        "--oauth-no-browser",
        action="store_true",
        help="Print the authorization URL instead of opening a browser on this host.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Authenticate and inspect the configured root without syncing the bundle.",
    )
    parser.add_argument(
        "--probe-write",
        action="store_true",
        help="With --check-only, create and immediately delete a small test folder.",
    )
    parser.add_argument("--root-folder-id", required=True)
    parser.add_argument("--local-prediction", type=Path, default=Path("data/prediction"))
    parser.add_argument(
        "--remote-directory",
        default="prediction",
        help=(
            "Directory below the configured Drive root (for example "
            "prediction_targeted). Use an empty value to synchronize a complete public "
            "data bundle directly below the root."
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="prediction_public_manifest.json",
        help="Required manifest filename in the local prediction directory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a resumable progress record after this many local files.",
    )
    return parser.parse_args()


def _write_oauth_token(path: Path, credentials: UserCredentials) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(credentials.to_json(), encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_credentials(args: argparse.Namespace):
    """Load a service account or authorize a user-owned Drive upload."""

    if args.credentials is not None:
        return service_account.Credentials.from_service_account_info(
            json.loads(args.credentials.read_text(encoding="utf-8")),
            scopes=[DRIVE_SCOPE],
        )

    token_path = args.oauth_token.expanduser().resolve()
    credentials: UserCredentials | None = None
    if token_path.is_file():
        credentials = UserCredentials.from_authorized_user_file(
            str(token_path), scopes=[DRIVE_SCOPE]
        )
    if credentials is not None and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        _write_oauth_token(token_path, credentials)
    if credentials is None or not credentials.valid:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as error:
            raise RuntimeError(
                "User OAuth requires google-auth-oauthlib. Install requirements-dev.txt."
            ) from error
        flow = InstalledAppFlow.from_client_secrets_file(
            str(args.oauth_client.expanduser().resolve()), [DRIVE_SCOPE]
        )
        credentials = flow.run_local_server(
            host="127.0.0.1",
            port=args.oauth_port,
            open_browser=not args.oauth_no_browser,
            authorization_prompt_message=(
                "Open this URL in your browser to authorize the Drive upload:\n{url}"
            ),
            success_message=(
                "Google Drive authorization completed. You may close this browser tab."
            ),
        )
        _write_oauth_token(token_path, credentials)
    return credentials


def load_service_account(path: Path, scope: str):
    return service_account.Credentials.from_service_account_info(
        json.loads(path.expanduser().read_text(encoding="utf-8")), scopes=[scope]
    )


def check_root(
    session: AuthorizedSession, root_folder_id: str, probe_write: bool
) -> dict[str, object]:
    response = session.get(
        f"{DRIVE_FILES_URL}/{root_folder_id}",
        params={
            "supportsAllDrives": "true",
            "fields": "id,name,mimeType,capabilities(canAddChildren)",
        },
        timeout=60,
    )
    raise_drive_error(response, "root-folder inspection")
    root = response.json()
    if root.get("mimeType") != DRIVE_FOLDER_MIME:
        raise RuntimeError("--root-folder-id does not identify a Google Drive folder")
    result: dict[str, object] = {
        "root_folder_id": root.get("id"),
        "root_folder_name": root.get("name"),
        "can_add_children": bool(
            (root.get("capabilities") or {}).get("canAddChildren", False)
        ),
        "write_probe": "not_requested",
    }
    if probe_write:
        probe = create_folder(session, ".lioness_oauth_write_probe", root_folder_id)
        delete = session.delete(
            f"{DRIVE_FILES_URL}/{probe['id']}",
            params={"supportsAllDrives": "true"},
            timeout=60,
        )
        raise_drive_error(delete, "write-probe cleanup")
        result["write_probe"] = "created_and_deleted"
    return result


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_folder(session: AuthorizedSession, name: str, parent_id: str) -> dict[str, object]:
    for attempt in range(MAX_DRIVE_ATTEMPTS):
        response = session.post(
            DRIVE_FILES_URL,
            params={"supportsAllDrives": "true", "fields": "id,name,mimeType,parents"},
            json={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
            timeout=60,
        )
        if response.ok:
            return response.json()
        if response.status_code not in TRANSIENT_DRIVE_STATUS:
            raise_drive_error(response, "folder creation")
        matches = [
            item
            for item in find_child(session, name, parent_id)
            if item.get("mimeType") == DRIVE_FOLDER_MIME
        ]
        if len(matches) == 1:
            return matches[0]
        if attempt + 1 == MAX_DRIVE_ATTEMPTS:
            raise_drive_error(response, "folder creation after retries")
        _wait_before_retry(response, "folder_creation", attempt)
    raise AssertionError("unreachable")


def create_file(
    session: AuthorizedSession,
    path: Path,
    name: str,
    parent_id: str,
    local_md5: str,
) -> dict[str, object]:
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    for attempt in range(MAX_DRIVE_ATTEMPTS):
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
        if response.ok:
            return response.json()
        if response.status_code not in TRANSIENT_DRIVE_STATUS:
            raise_drive_error(response, f"file creation for {name}")
        matches = [
            item
            for item in find_child(session, name, parent_id)
            if int(item.get("size") or -1) == path.stat().st_size
            and str(item.get("md5Checksum") or "").lower() == local_md5
        ]
        if len(matches) == 1:
            return matches[0]
        if attempt + 1 == MAX_DRIVE_ATTEMPTS:
            raise_drive_error(response, f"file creation for {name} after retries")
        _wait_before_retry(response, f"file_creation:{name}", attempt)
    raise AssertionError("unreachable")


def update_file(session: AuthorizedSession, path: Path, file_id: str) -> dict[str, object]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    for attempt in range(MAX_DRIVE_ATTEMPTS):
        with path.open("rb") as handle:
            response = session.patch(
                f"{DRIVE_UPLOAD_URL}/{file_id}",
                params={"uploadType": "media", "supportsAllDrives": "true", "fields": "id,name,size,md5Checksum,parents"},
                headers={"Content-Type": mime},
                data=handle,
                timeout=(30, 600),
            )
        if response.ok:
            return response.json()
        if response.status_code not in TRANSIENT_DRIVE_STATUS:
            raise_drive_error(response, f"file update for {path.name}")
        if attempt + 1 == MAX_DRIVE_ATTEMPTS:
            raise_drive_error(response, f"file update for {path.name} after retries")
        _wait_before_retry(response, f"file_update:{path.name}", attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    if args.probe_write and not args.check_only:
        raise ValueError("--probe-write may only be used with --check-only")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    credentials = load_credentials(args)
    session = AuthorizedSession(credentials)
    if args.check_only:
        print(json.dumps(check_root(session, args.root_folder_id, args.probe_write), indent=2))
        return

    local_root = args.local_prediction.resolve()
    if not (local_root / args.manifest_name).is_file():
        raise FileNotFoundError("Prediction public manifest is missing from the local bundle")
    remote_directory = str(args.remote_directory).strip("/")
    if ".." in Path(remote_directory).parts:
        raise ValueError("--remote-directory must be a safe path below the Drive root")
    inventory_session = session
    inventory_mode = "upload_identity"
    if args.inventory_credentials is not None:
        inventory_session = AuthorizedSession(
            load_service_account(args.inventory_credentials, DRIVE_SCOPE)
        )
        inventory_mode = "read_only_service_account"
    items, list_requests = list_visible_items(inventory_session)
    by_id = {str(item["id"]): item for item in items}
    if args.root_folder_id not in by_id:
        raise RuntimeError("The Drive root is not visible to the authenticated account")
    by_path = {
        relative: item
        for item in items
        if (relative := path_below_root(item, args.root_folder_id, by_id)) is not None
    }
    folder_ids = {"": args.root_folder_id}
    uploaded = updated = reused = created_folders = 0
    local_files = sorted(path for path in local_root.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "status": "sync_started",
                "local_files": len(local_files),
                "drive_items_visible": len(items),
                "drive_list_requests": list_requests,
                "inventory_mode": inventory_mode,
            }
        ),
        flush=True,
    )
    for file_number, local in enumerate(local_files, start=1):
        local_relative = local.relative_to(local_root).as_posix()
        remote_relative = f"{remote_directory}/{local_relative}".strip("/")
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
        elif existing is None:
            create_file(
                session,
                local,
                local.name,
                folder_ids[current_parent],
                local_md5,
            )
            uploaded += 1
        else:
            update_file(session, local, str(existing["id"]))
            updated += 1
        if file_number % args.progress_every == 0 or file_number == len(local_files):
            print(
                json.dumps(
                    {
                        "status": "sync_progress",
                        "files_examined": file_number,
                        "local_files": len(local_files),
                        "created_folders": created_folders,
                        "created_files": uploaded,
                        "updated_files": updated,
                        "unchanged_files": reused,
                    }
                ),
                flush=True,
            )
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
