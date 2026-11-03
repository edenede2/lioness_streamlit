"""Lossless release encoding, narrow verified reads and hosted privacy gates."""

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import pytest
from app_helpers.control_calibrated import adapter as A, hosted as H
from app_helpers.control_calibrated.inputs import atomic_json, sha
from test_control_calibrated_pipeline import catalog_root


@pytest.fixture
def release(catalog_root, tmp_path):
    source = catalog_root
    dest = tmp_path / "release"
    catalog = A.catalog(source)
    job = next(j for j in catalog["modules"] if j["family"] == "full_cohort")
    meta = pd.read_parquet(source / "metadata/brain.parquet")
    sources = []
    for beta in (3, 1):
        path = (
            source
            / "scores"
            / f"beta{beta}"
            / "full_cohort/maximum_component"
            / A.module_filename(job["module_key"])
        )
        sources.append((beta, "all", path, sha(path)))
        masked = tmp_path / f"masked_beta{beta}.parquet"
        pd.read_parquet(path).assign(
            mask="frozen", mask_provenance="Frozen synthetic discovery mask"
        ).to_parquet(masked, index=False)
        sources.append((beta, "frozen", masked, sha(masked)))
    path = H.module_path(dest, job["module_key"])
    record = H.pack_module(path, job, sources, meta)
    sidecar = path.with_suffix(".ranges.json")
    atomic_json(sidecar, H.range_receipt(path))
    record.update(range_path=str(sidecar.relative_to(dest)), range_sha256=sha(sidecar))
    atomic_json(
        dest / "manifest.json",
        dict(status="complete", storage_layout=H.LAYOUT, families=catalog["families"]),
    )
    atomic_json(
        dest / "release_manifest.json", dict(modules={job["module_key"]: record})
    )
    atomic_json(dest / "catalog.json", dict(catalog, modules=[job]))
    (dest / "metadata").mkdir()
    meta.to_parquet(dest / "metadata/brain.parquet", index=False)
    return dest, job, meta, sources


def test_exact_frames_both_betas_masks_and_scope_subsets(release, catalog_root):
    root, job, meta, sources = release
    for beta, mask, path, _ in sources:
        restored = H.read_scores(root, job, beta, mask, meta)
        expected = pd.read_parquet(path)
        order = ["component", "sample_id"]
        pd.testing.assert_frame_equal(
            restored.sort_values(order).reset_index(drop=True),
            expected.sort_values(order).reset_index(drop=True),
            check_exact=True,
        )
    for beta in (3, 1):
        for scope in (
            "complete_450",
            "incomplete_region_validation",
            "maximum_component",
        ):
            kw = dict(
                module_set="full_cohort",
                module=1,
                estimator=f"calibrated_beta{beta}",
                cohort_scope=scope,
            )
            actual = (
                A.scores(root=root, **kw)
                .sort_values(["sample_id", "component", "metric_family"])
                .reset_index(drop=True)
            )
            expected = (
                A.scores(root=catalog_root, **kw)
                .sort_values(["sample_id", "component", "metric_family"])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)


class Response:
    def __init__(self, data, status):
        self.data = data
        self.status_code = status

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data

    def close(self):
        pass


class Session:
    def __init__(self, files):
        self.files = files
        self.calls = []
        self.corrupt = False

    def get(self, url, params=None, headers=None, **kwargs):
        path = self.files[url.rsplit("/", 1)[1]]
        data = path.read_bytes()
        spec = (headers or {}).get("Range")
        if spec:
            start, end = map(int, spec.removeprefix("bytes=").split("-"))
            data = data[start : end + 1]
            if self.corrupt:
                data = data[:-1] + bytes([data[-1] ^ 1])
        self.calls.append(dict(path=path.name, range=spec, bytes=len(data)))
        return Response(data, 206 if spec else 200)


@pytest.mark.parametrize("container", [False, True])
def test_drive_ranges_exact_cached_bounded_and_checksum_protected(
    release, tmp_path, monkeypatch, container
):
    source, job, meta, _ = release
    cache = tmp_path / "cache"
    cache.mkdir()
    index = {}
    files = {}
    if container:
        release_path = source / "release_manifest.json"
        manifest = json.loads(release_path.read_text())
        r = manifest["modules"][job["module_key"]]
        blob = source / "transport.bin"
        module = H.module_path(source, job["module_key"])
        sidecar = source / r["range_path"]
        prefix = b"container-prefix"
        blob.write_bytes(prefix + module.read_bytes() + sidecar.read_bytes())
        r.update(
            container_path="transport.bin",
            module_offset=len(prefix),
            range_offset=len(prefix) + module.stat().st_size,
            range_length=sidecar.stat().st_size,
        )
        atomic_json(release_path, manifest)
    for i, p in enumerate(source.rglob("*")):
        if p.is_file():
            if container and p.is_relative_to(source / "modules"):
                continue
            identifier = f"f{i}"
            files[identifier] = p
            data = p.read_bytes()
            index["release/" + p.relative_to(source).as_posix()] = dict(
                id=identifier,
                size=len(data),
                md5=hashlib.md5(data).hexdigest(),
                sha256=sha(p),
            )
    session = Session(files)
    monkeypatch.setattr(H.D, "DATA_DIR", cache)
    monkeypatch.setattr(H.D, "load_drive_index", lambda: dict(files=index))
    monkeypatch.setattr(H.D, "_authorized_session", lambda: session)
    root = cache / "release"
    actual = H.read_scores(root, job, 3, "all", meta)
    expected = H.read_scores(source, job, 3, "all", meta)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    module_name = (
        "transport.bin" if container else H.module_path(source, job["module_key"]).name
    )
    module_calls = [c for c in session.calls if c["path"] == module_name]
    assert module_calls and all(c["range"] for c in module_calls)
    assert (
        sum(c["bytes"] for c in module_calls)
        < H.module_path(source, job["module_key"]).stat().st_size
    )
    calls = len(session.calls)
    H.read_scores(root, job, 3, "all", meta)
    assert len(session.calls) == calls
    session.corrupt = True
    with pytest.raises(IOError, match="checksum"):
        H.read_scores(root, job, 1, "all", meta)


def test_role_membership_and_local_only_edge_gate(release, monkeypatch):
    root, job, meta, _ = release
    broken = meta.copy()
    broken.loc[0, "role"] = "target"
    with pytest.raises(ValueError, match="membership"):
        H.read_scores(root, job, 3, "all", broken)
    from app_helpers.control_calibrated.inspection import iter_edges

    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **kw: pytest.fail("Hosted edge path opened a private array"),
    )
    with pytest.raises(A.ScoreUnavailable, match="remains local"):
        list(iter_edges("full_cohort", 1, ["unknown"], root=root))
