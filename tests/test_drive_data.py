from __future__ import annotations

import hashlib
from pathlib import Path

from app_helpers import drive_data


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.payload[:3]
        yield self.payload[3:]


class FakeSession:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, *args, **kwargs) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.payload)


def test_drive_download_is_integrity_checked_and_cached(tmp_path, monkeypatch) -> None:
    payload = b"sample parquet bytes"
    metadata = {
        "id": "drive-file-id",
        "size": len(payload),
        "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }
    session = FakeSession(payload)
    monkeypatch.setattr(drive_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        drive_data,
        "load_drive_index",
        lambda: {"schema_version": 1, "files": {"sample.parquet": metadata}},
    )
    monkeypatch.setattr(drive_data, "_authorized_session", lambda: session)

    destination = tmp_path / "sample.parquet"
    assert drive_data.ensure_data_path(destination) == destination
    assert destination.read_bytes() == payload
    assert session.calls == 1

    assert drive_data.ensure_data_path(destination) == destination
    assert session.calls == 1


def test_partition_predicates_skip_unrelated_drive_parts(tmp_path, monkeypatch) -> None:
    payload = b"part"
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    files = {
        "volcano.parquet/part-1.parquet": {
            "id": "one",
            "size": len(payload),
            "md5": md5,
            "parquet_predicates": {
                "module": {"min": 1, "max": 100},
                "estimator": {"values": ["lioness"]},
            },
        },
        "volcano.parquet/part-2.parquet": {
            "id": "two",
            "size": len(payload),
            "md5": md5,
            "parquet_predicates": {
                "module": {"min": 101, "max": 200},
                "estimator": {"values": ["bonobo"]},
            },
        },
    }
    session = FakeSession(payload)
    monkeypatch.setattr(drive_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        drive_data,
        "load_drive_index",
        lambda: {"schema_version": 1, "files": files},
    )
    monkeypatch.setattr(drive_data, "_authorized_session", lambda: session)

    materialized = drive_data.ensure_data_path(
        tmp_path / "volcano.parquet",
        [("module", "=", 150), ("estimator", "=", "bonobo")],
    )
    assert materialized == [tmp_path / "volcano.parquet/part-2.parquet"]
    assert session.calls == 1


def test_streamlit_secrets_example_is_valid_toml() -> None:
    import toml

    example = Path(__file__).resolve().parents[1] / ".streamlit/secrets.toml.example"
    parsed = toml.loads(example.read_text(encoding="utf-8"))
    assert parsed["google_drive"]["folder_id"]
    assert '"type": "service_account"' in parsed["google_drive"]["credentials_json"]


def test_private_key_normalization_repairs_escaping_whitespace_and_padding() -> None:
    escaped = (
        "-----BEGIN PRIVATE KEY-----\\n"
        "QU JD RA\\n"
        "-----END PRIVATE KEY-----\\n"
    )
    assert drive_data._normalize_private_key(escaped) == (
        "-----BEGIN PRIVATE KEY-----\n"
        "QUJDRA==\n"
        "-----END PRIVATE KEY-----\n"
    )


def test_service_account_normalization_reports_missing_fields() -> None:
    try:
        drive_data._normalize_service_account_info({"type": "service_account"})
    except RuntimeError as error:
        assert "credentials are incomplete" in str(error)
        assert "private_key" in str(error)
    else:
        raise AssertionError("Incomplete service-account data should fail clearly")
