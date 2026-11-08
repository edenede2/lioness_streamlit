"""Bounded UI caches with dataset/catalog versions, never a global cache clear."""

from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import threading

import streamlit as st

from app_helpers.drive_data import DATA_DIR, DRIVE_INDEX_PATH, ensure_data_path

_lock = threading.RLock()
_versions: dict[str, str] = {}
_manifest_stamp: tuple | None = None


def file_stamp(path: Path) -> tuple:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return (str(path), None, None)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def register_manifest(manifest: dict, stamp: tuple) -> None:
    """Only rehash manifest entries after the deployed manifest changes."""
    global _manifest_stamp
    with _lock:
        if _manifest_stamp == stamp:
            return
        _versions.clear()
        _versions["manifest"] = _digest(stamp)
        common = {
            k: v
            for k, v in manifest.items()
            if k not in {"module_sets", "files", "created_utc"}
        }
        common_hash = _digest(common)
        for key, value in manifest.get("module_sets", {}).items():
            _versions[key] = _digest((value, common_hash))
        _manifest_stamp = stamp


def version_for(scope: str, module_set: str | None = None) -> str:
    if scope in {"prediction", "prediction_targeted"}:
        name = (
            "prediction_public_manifest.json"
            if scope == "prediction"
            else "targeted_prediction_public_manifest.json"
        )
        # Only a selected prediction page asks for this catalog's version.
        path = DATA_DIR / scope / name
        return _digest(file_stamp(ensure_data_path(path)))
    with _lock:
        return _versions.get(
            module_set or "manifest",
            _digest(file_stamp(DATA_DIR / "data_manifest.json")),
        )


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_manifest(stamp: tuple, index_stamp: tuple) -> dict:
    from app_helpers.data import load_data_manifest, require_data_files

    manifest = load_data_manifest()
    require_data_files(manifest)
    return manifest


def initialize_bundle() -> dict:
    path = ensure_data_path(DATA_DIR / "data_manifest.json")
    stamp = file_stamp(path)
    manifest = _load_manifest(stamp, file_stamp(DRIVE_INDEX_PATH))
    register_manifest(manifest, stamp)
    return manifest


def cache_data(**options):
    """Drop-in UI decorator; scientific readers and their signatures stay intact."""
    options.setdefault("max_entries", 4)
    options.setdefault("ttl", 1800)

    def decorate(function):
        signature = inspect.signature(function)
        scope = (
            "prediction_targeted"
            if "targeted_prediction" in function.__name__
            else "prediction" if "prediction" in function.__name__ else "exploration"
        )

        def entry(version_token, *args, **kwargs):
            return function(*args, **kwargs)

        entry.__name__ = function.__name__
        entry.__qualname__ = function.__qualname__
        entry.__module__ = function.__module__
        cached = st.cache_data(**options)(entry)

        @wraps(function)
        def read(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            module_set = bound.arguments.get("module_set")
            return cached(version_for(scope, module_set), *args, **kwargs)

        read.clear = cached.clear
        return read

    return decorate
