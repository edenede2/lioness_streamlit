#!/usr/bin/env python3
"""Build the static Google Drive file index used by the Streamlit app."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


DRIVE_LIST_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
PREDICATE_COLUMNS = ("module", "estimator", "network_method", "fdr_scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--root-folder-id", required=True)
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("drive_file_index.json"))
    parser.add_argument(
        "--skip-local-md5",
        action="store_true",
        help="Validate names and sizes but do not hash every local file.",
    )
    return parser.parse_args()


def list_visible_items(session: AuthorizedSession) -> tuple[list[dict[str, object]], int]:
    items: list[dict[str, object]] = []
    page_token: str | None = None
    requests = 0
    while True:
        parameters = {
            "q": "trashed = false",
            "pageSize": 1000,
            "fields": (
                "nextPageToken,files(id,name,mimeType,parents,size,md5Checksum,modifiedTime)"
            ),
            "spaces": "drive",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token:
            parameters["pageToken"] = page_token
        response = session.get(DRIVE_LIST_URL, params=parameters, timeout=60)
        response.raise_for_status()
        requests += 1
        payload = response.json()
        items.extend(payload.get("files", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return items, requests


def path_below_root(
    item: dict[str, object],
    root_folder_id: str,
    items_by_id: dict[str, dict[str, object]],
) -> str | None:
    names = [str(item["name"])]
    seen = {str(item["id"])}
    parents = item.get("parents") or []
    while parents:
        parent_id = str(parents[0])
        if parent_id == root_folder_id:
            return "/".join(reversed(names))
        if parent_id in seen or parent_id not in items_by_id:
            return None
        seen.add(parent_id)
        parent = items_by_id[parent_id]
        names.append(str(parent["name"]))
        parents = parent.get("parents") or []
    return None


def parquet_predicates(path: Path) -> dict[str, dict[str, object]]:
    if ".parquet/" not in path.as_posix():
        return {}
    parquet = pq.ParquetFile(path)
    columns = [name for name in PREDICATE_COLUMNS if name in parquet.schema_arrow.names]
    if not columns:
        return {}
    table = parquet.read(columns=columns)
    result: dict[str, dict[str, object]] = {}
    for column in columns:
        values = table[column].combine_chunks().drop_null()
        if not len(values):
            continue
        if column == "module":
            extrema = pc.min_max(values).as_py()
            result[column] = {
                "min": int(extrema["min"]),
                "max": int(extrema["max"]),
            }
        else:
            unique = sorted(str(value) for value in pc.unique(values).to_pylist())
            result[column] = {"values": unique}
    return result


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    credentials_info = json.loads(args.credentials.read_text(encoding="utf-8"))
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info, scopes=[DRIVE_SCOPE]
    )
    items, request_count = list_visible_items(AuthorizedSession(credentials))
    items_by_id = {str(item["id"]): item for item in items}
    if args.root_folder_id not in items_by_id:
        raise RuntimeError(
            "The configured root folder is not visible to the service account. "
            "Share the folder with the service-account client_email."
        )
    if items_by_id[args.root_folder_id].get("mimeType") != DRIVE_FOLDER_MIME:
        raise RuntimeError("--root-folder-id does not identify a Google Drive folder")

    indexed: dict[str, dict[str, object]] = {}
    for item in items:
        if item.get("mimeType") == DRIVE_FOLDER_MIME:
            continue
        relative = path_below_root(item, args.root_folder_id, items_by_id)
        if relative is None:
            continue
        if relative in indexed:
            raise RuntimeError(f"Duplicate path within Drive folder: {relative}")
        metadata: dict[str, object] = {
            "id": item["id"],
            "size": int(item["size"]) if item.get("size") is not None else None,
            "md5": item.get("md5Checksum"),
            "modified_time": item.get("modifiedTime"),
        }
        local_path = args.local_data / relative
        if local_path.is_file():
            predicates = parquet_predicates(local_path)
            if predicates:
                metadata["parquet_predicates"] = predicates
        indexed[relative] = metadata

    local_files = {
        path.relative_to(args.local_data).as_posix(): path
        for path in args.local_data.rglob("*")
        if path.is_file()
    }
    missing_remote = sorted(set(local_files).difference(indexed))
    extra_remote = sorted(set(indexed).difference(local_files))
    size_mismatches: list[str] = []
    checksum_mismatches: list[str] = []
    for relative in sorted(set(local_files).intersection(indexed)):
        local = local_files[relative]
        remote = indexed[relative]
        if remote.get("size") is not None and local.stat().st_size != int(remote["size"]):
            size_mismatches.append(relative)
            continue
        expected_md5 = str(remote.get("md5") or "").lower()
        if expected_md5 and not args.skip_local_md5 and md5_file(local) != expected_md5:
            checksum_mismatches.append(relative)

    if missing_remote or extra_remote or size_mismatches or checksum_mismatches:
        details = {
            "missing_remote": missing_remote,
            "extra_remote": extra_remote,
            "size_mismatches": size_mismatches,
            "checksum_mismatches": checksum_mismatches,
        }
        raise RuntimeError("Drive/local bundle mismatch:\n" + json.dumps(details, indent=2))

    payload = {
        "schema_version": 1,
        "root_folder_id": args.root_folder_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(indexed),
        "total_bytes": sum(int(item.get("size") or 0) for item in indexed.values()),
        "drive_list_requests": request_count,
        "files": dict(sorted(indexed.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Indexed and validated {len(indexed)} files using {request_count} Drive list "
        f"request(s): {args.output}"
    )


if __name__ == "__main__":
    main()
