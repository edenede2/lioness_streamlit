"""Lazy, integrity-checked access to the public bundle stored in Google Drive.

The repository can still use a local ``data/`` directory for development.  When
that directory is absent, files are downloaded individually into an ephemeral
cache using a committed Drive file index and read-only service-account secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable


APP_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = APP_ROOT / "data"
DRIVE_INDEX_PATH = APP_ROOT / "drive_file_index.json"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_MEDIA_URL = "https://www.googleapis.com/drive/v3/files/{file_id}"


def _selected_data_dir() -> Path:
    override = os.environ.get("LIONESS_APP_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    force_drive = os.environ.get("LIONESS_FORCE_DRIVE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not force_drive and (LOCAL_DATA_DIR / "data_manifest.json").is_file():
        return LOCAL_DATA_DIR.resolve()
    cache_override = os.environ.get("LIONESS_DRIVE_CACHE_DIR")
    cache_root = (
        Path(cache_override).expanduser()
        if cache_override
        else Path(tempfile.gettempdir()) / "lioness_streamlit_data"
    )
    return cache_root.resolve()


DATA_DIR = _selected_data_dir()


@lru_cache(maxsize=1)
def load_drive_index() -> dict[str, object]:
    if not DRIVE_INDEX_PATH.is_file():
        return {"files": {}}
    payload = json.loads(DRIVE_INDEX_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("drive_file_index.json has an unsupported schema")
    return payload


def _relative_key(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(DATA_DIR)
    except ValueError as error:
        raise ValueError(f"Data path is outside the configured bundle: {path}") from error
    key = relative.as_posix().lstrip("/")
    if not key or key.startswith("../"):
        raise ValueError(f"Invalid data path: {path}")
    return key


def _indexed_entries(path: Path) -> dict[str, dict[str, object]]:
    files = load_drive_index().get("files", {})
    if not isinstance(files, dict):
        return {}
    key = _relative_key(path)
    exact = files.get(key)
    if isinstance(exact, dict):
        return {key: exact}
    prefix = key.rstrip("/") + "/"
    return {
        str(relative): metadata
        for relative, metadata in files.items()
        if str(relative).startswith(prefix) and isinstance(metadata, dict)
    }


def data_path_available(path: Path) -> bool:
    """Return whether a local or indexed remote file/dataset is available."""
    return path.exists() or bool(_indexed_entries(path))


def _secret_mapping() -> dict[str, object]:
    credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credential_path:
        return json.loads(Path(credential_path).expanduser().read_text(encoding="utf-8"))

    try:
        import streamlit as st

        section = st.secrets["google_drive"]
    except Exception as error:
        raise RuntimeError(
            "Google Drive credentials are unavailable. Add a [google_drive] section "
            "to Streamlit Secrets or set GOOGLE_APPLICATION_CREDENTIALS locally."
        ) from error

    credentials_json = section.get("credentials_json")
    if credentials_json:
        info = json.loads(str(credentials_json))
    else:
        ignored = {"folder_id", "cache_dir"}
        info = {key: value for key, value in dict(section).items() if key not in ignored}
    private_key = info.get("private_key")
    if isinstance(private_key, str):
        info["private_key"] = private_key.replace("\\n", "\n")
    return info


def _configured_folder_id() -> str | None:
    value = os.environ.get("LIONESS_GOOGLE_DRIVE_FOLDER_ID")
    if value:
        return value.strip()
    try:
        import streamlit as st

        value = st.secrets["google_drive"].get("folder_id")
    except Exception:
        return None
    return str(value).strip() if value else None


@lru_cache(maxsize=1)
def _authorized_session():
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    index_folder = str(load_drive_index().get("root_folder_id", "")).strip()
    configured_folder = _configured_folder_id()
    if configured_folder and index_folder and configured_folder != index_folder:
        raise RuntimeError(
            "The Google Drive folder_id in Streamlit Secrets does not match "
            "drive_file_index.json."
        )
    credentials = service_account.Credentials.from_service_account_info(
        _secret_mapping(), scopes=[DRIVE_SCOPE]
    )
    return AuthorizedSession(credentials)


_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}


def _download_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _DOWNLOAD_LOCKS.setdefault(key, threading.Lock())


def _valid_cached_file(path: Path, metadata: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    expected_size = metadata.get("size")
    return expected_size is None or path.stat().st_size == int(expected_size)


def _download_file(relative: str, metadata: dict[str, object]) -> Path:
    destination = DATA_DIR / relative
    if _valid_cached_file(destination, metadata):
        return destination

    with _download_lock(relative):
        if _valid_cached_file(destination, metadata):
            return destination
        file_id = str(metadata.get("id", "")).strip()
        if not file_id:
            raise RuntimeError(f"Drive index entry lacks a file ID: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        try:
            response = _authorized_session().get(
                DRIVE_MEDIA_URL.format(file_id=file_id),
                params={"alt": "media", "supportsAllDrives": "true"},
                stream=True,
                timeout=(30, 300),
            )
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            expected_size = metadata.get("size")
            if expected_size is not None and size != int(expected_size):
                raise IOError(
                    f"Incomplete Drive download for {relative}: {size} bytes, "
                    f"expected {expected_size}"
                )
            expected_md5 = str(metadata.get("md5", "")).strip().lower()
            if expected_md5 and digest.hexdigest().lower() != expected_md5:
                raise IOError(f"Drive checksum mismatch for {relative}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _entry_may_match(
    metadata: dict[str, object], filters: Iterable[tuple[str, str, object]]
) -> bool:
    """Use optional Parquet part summaries to avoid unrelated Drive downloads."""
    predicates = metadata.get("parquet_predicates")
    if not isinstance(predicates, dict):
        return True
    for column, operator, value in filters:
        if operator != "=" or column not in predicates:
            continue
        summary = predicates[column]
        if not isinstance(summary, dict):
            continue
        values = summary.get("values")
        if isinstance(values, list) and value not in values and str(value) not in {
            str(item) for item in values
        }:
            return False
        minimum = summary.get("min")
        maximum = summary.get("max")
        if minimum is not None and maximum is not None:
            try:
                if float(value) < float(minimum) or float(value) > float(maximum):
                    return False
            except (TypeError, ValueError):
                pass
    return True


def ensure_data_path(
    path: Path,
    filters: Iterable[tuple[str, str, object]] = (),
) -> Path | list[Path]:
    """Materialize one indexed file or only matching parts of a dataset."""
    if path.is_file():
        return path
    entries = _indexed_entries(path)
    if path.is_dir() and not entries:
        return path
    if not entries:
        raise FileNotFoundError(f"Data file is absent locally and from Drive: {path}")
    exact_key = _relative_key(path)
    if exact_key in entries:
        return _download_file(exact_key, entries[exact_key])

    selected = {
        relative: metadata
        for relative, metadata in entries.items()
        if _entry_may_match(metadata, filters)
    }
    if not selected:
        raise FileNotFoundError(
            f"No indexed Parquet parts can satisfy the requested filters for {path}"
        )
    return [_download_file(relative, metadata) for relative, metadata in selected.items()]


def data_source_label() -> str:
    return "local bundle" if DATA_DIR == LOCAL_DATA_DIR.resolve() else "Google Drive (lazy cache)"
