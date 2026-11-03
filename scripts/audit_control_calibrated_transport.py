#!/usr/bin/env python3
"""Check every published container byte against the sealed numeric modules."""
from pathlib import Path
import hashlib
import json
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers.control_calibrated.inputs import atomic_json, sha

OUT = APP / "artifacts/control_calibrated_hosted_release_20261103"
ROOT = OUT / "public_bundle"


def main():
    release = json.loads((ROOT / "release_manifest.json").read_text())
    assert release["transport_layout"] == "range_containers_v1"
    spans = {key: [] for key in release["containers"]}
    for key, record in release["modules"].items():
        spans[record["container_path"]].extend(
            [
                (record["module_offset"], record["bytes"], record["sha256"], key),
                (
                    record["range_offset"],
                    record["range_length"],
                    record["range_sha256"],
                    key + ":ranges",
                ),
            ]
        )
    total = 0
    for i, (key, expected) in enumerate(release["containers"].items()):
        path = ROOT / key
        assert path.stat().st_size == expected["bytes"] < 100 * 1024**2, key
        cursor = 0
        whole = hashlib.sha256()
        with path.open("rb") as stream:
            for offset, length, digest, module in sorted(spans[key]):
                assert offset == cursor, (key, module, "gap or overlap")
                part = hashlib.sha256()
                remaining = length
                while remaining:
                    block = stream.read(min(remaining, 8 * 1024**2))
                    assert block, (key, module, "truncated")
                    whole.update(block)
                    part.update(block)
                    remaining -= len(block)
                assert part.hexdigest() == digest, (key, module, "changed bytes")
                cursor += length
            assert not stream.read(1), (key, "unexpected payload")
        assert (
            cursor == expected["bytes"] and whole.hexdigest() == expected["sha256"]
        ), key
        assert len(spans[key]) == 2 * expected["modules"], key
        total += cursor
        if i % 25 == 0:
            print(json.dumps(dict(checked=i + 1, total=len(spans))), flush=True)
    result = dict(
        status="passed",
        release_id=release["release_id"],
        release_manifest_sha256=sha(ROOT / "release_manifest.json"),
        containers=len(spans),
        modules=len(release["modules"]),
        bytes=total,
        all_subrange_checksums_verified=True,
        no_gaps_or_extra_bytes=True,
    )
    atomic_json(OUT / "transport_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
