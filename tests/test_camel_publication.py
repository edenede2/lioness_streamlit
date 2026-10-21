"""Publication tests use only temporary local files and mocked Drive responses."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts"))
from run_camel_validation import atomic_json, sha
import publish_camel_validation as P


def test_snapshot_is_immutable_and_excludes_duplicate_wide_comparisons(tmp_path):
    source = tmp_path / "public_bundle"
    atomic_json(dict(approved=True), tmp_path / "publication_approval.json")
    atomic_json(dict(completed_rules=["all"], backend_verified=False), source / "manifest.json")
    payload = source / "modules/M121.parquet"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"example-only-in-test")
    wide = source / "concordance/all/standard_pruned.parquet"
    wide.parent.mkdir(parents=True)
    wide.write_bytes(b"duplicate wide table")
    atomic_json(dict(status="passed", completed_rules=["all"], files={
        str(p.relative_to(source)): dict(sha256=sha(p)) for p in (payload, wide)
    }), source / "release_manifest.json")
    snapshot = P.snapshot(tmp_path)
    assert (snapshot / "modules/M121.parquet").read_bytes() == b"example-only-in-test"
    replacement = source / "modules/replacement.parquet"
    replacement.write_bytes(b"later analysis replacement")
    replacement.replace(payload)
    assert (snapshot / "modules/M121.parquet").read_bytes() == b"example-only-in-test"
    assert not (snapshot / "concordance").exists()


def test_snapshot_rejects_unapproved_or_unaudited_milestone(tmp_path):
    atomic_json(dict(approved=False), tmp_path / "publication_approval.json")
    atomic_json(dict(status="passed", completed_rules=["all"], files={}), tmp_path / "public_bundle/release_manifest.json")
    atomic_json(dict(completed_rules=["all"]), tmp_path / "public_bundle/manifest.json")
    with pytest.raises(ValueError, match="approval"):
        P.snapshot(tmp_path)
    atomic_json(dict(approved=True), tmp_path / "publication_approval.json")
    atomic_json(dict(completed_rules=["all", "new_mask"]), tmp_path / "public_bundle/manifest.json")
    with pytest.raises(ValueError, match="Repeat the release audit"):
        P.snapshot(tmp_path)


def test_inventory_is_scoped_to_exact_upload_folder_and_paginated():
    class Response:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body
    class Session:
        def __init__(self): self.queries = []
        def get(self, url, *, params, timeout):
            self.queries.append(params.copy())
            if len(self.queries) == 1:
                return Response(dict(nextPageToken="second", files=[dict(id="a", name="one")]))
            return Response(dict(files=[dict(id="b", name="two")]))
    session = Session()
    assert set(P.list_own_folder(session, "exact-approved-folder")) == {"one", "two"}
    assert all(p["q"] == "'exact-approved-folder' in parents and trashed = false" for p in session.queries)
    assert session.queries[1]["pageToken"] == "second"


def test_incremental_release_reuses_only_exact_verified_content_in_this_prefix():
    entry = dict(id="verified-file", size=100, md5="checksum")
    index = dict(files={P.PREFIX + "/modules/M121.parquet": entry,
                        "other_dataset/modules/M122.parquet": entry})
    assert P.verified_existing_entry(index, "modules/M121.parquet", 100, "checksum") == entry
    assert P.verified_existing_entry(index, "modules/M121.parquet", 101, "checksum") is None
    assert P.verified_existing_entry(index, "modules/M121.parquet", 100, "changed") is None
    assert P.verified_existing_entry(index, "modules/M122.parquet", 100, "checksum") is None
