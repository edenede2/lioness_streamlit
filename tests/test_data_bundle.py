from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers import data  # noqa: E402


def test_public_plot_files_have_expected_rows_and_no_identifiers() -> None:
    forbidden = {
        "donor",
        "projid",
        "msex.x",
        "age_death.x",
        "educ.x",
        "apoe_genotype",
        "cogdx.y",
        "braaksc",
        "ceradsc",
        "adnc",
        "parkinsonism_yn_lv",
    }
    expected = {
        data.AGGREGATE_DATA: 2 * 450 * 154 * 6,
        data.RESOLVED_DATA: 2 * 450 * 154 * 6 * 6,
    }
    for path, expected_rows in expected.items():
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == expected_rows
        assert forbidden.isdisjoint(parquet.schema_arrow.names)
        assert "sample_id" in parquet.schema_arrow.names

    metadata_columns = set(pq.ParquetFile(data.SAMPLE_METADATA).schema_arrow.names)
    assert {"donor", "projid"}.isdisjoint(metadata_columns)
    assert {
        "sample_id",
        "age_at_death",
        "apoe_genotype",
        "cogdx",
        "braak_stage",
        "cerad_score",
        "adnc",
        "parkinsonism",
    }.issubset(metadata_columns)
    assert pq.ParquetFile(data.SAMPLE_METADATA).metadata.num_rows == 450


def test_all_modules_and_m1918_are_packaged() -> None:
    annotations = data.load_module_annotations()
    assert annotations["module"].nunique() == 154
    assert 1918 in set(annotations["module"].astype(int))
    for method in data.METHOD_LABELS:
        selected = data.load_aggregate(method, 1918, "abs_sum")
        assert len(selected) == 450
        assert selected["sample_id"].nunique() == 450


def test_dlpfc_public_label_replaces_internal_mf_code() -> None:
    resolved = data.load_resolved("control_anchored", 935, "connectivity")
    assert resolved["component"].str.contains("DLPFC").any()
    assert not resolved.astype(str).apply(lambda col: col.str.contains("MFBA9", regex=False)).any().any()
    kegg_columns = pq.ParquetFile(data.KEGG_PARQUET).schema_arrow.names
    assert "overlap_DLPFC" in kegg_columns
    assert "overlap_MFBA9BA46" not in kegg_columns


def test_kegg_table_is_complete_and_module_filter_works() -> None:
    full = data.load_kegg()
    source = pd.read_csv(data.KEGG_TSV, sep="\t")
    assert len(full) == len(source) == 8866
    assert len(data.load_kegg(1918)) == 156
