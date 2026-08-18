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


def test_module_details_match_level4_modules_and_tissue_counts() -> None:
    details = data.load_module_details()
    annotations = data.load_module_annotations()
    assert len(details) == details["module"].nunique() == 154
    assert set(details["module"].astype(int)) == set(annotations["module"].astype(int))
    assert details[["n_genes_ac", "n_genes_dlpfc", "n_genes_pcg"]].sum(axis=1).eq(
        details["module_size"]
    ).all()
    proportions = details[["proportion_ac", "proportion_dlpfc", "proportion_pcg"]]
    assert (proportions.sum(axis=1) - 1.0).abs().max() < 1e-8
    assert not details.astype(str).apply(
        lambda column: column.str.contains("MFBA9", regex=False)
    ).any().any()

    m1012 = data.load_module_details(1012).iloc[0]
    assert int(m1012["module_size"]) == 152
    assert int(m1012["n_genes_ac"]) == 151
    assert int(m1012["n_genes_dlpfc"]) == 0
    assert int(m1012["n_genes_pcg"]) == 1
    assert m1012["tissues"] == "AC, PCG"

    manifest = data.load_data_manifest()
    assert manifest["rows"]["module_details_rows"] == 154
    assert "module_details.tsv" in manifest["files"]


def test_dlpfc_public_label_replaces_internal_mf_code() -> None:
    resolved = data.load_resolved("control_anchored", 935, "connectivity")
    resolved_stats = data.load_resolved_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    assert resolved["component"].str.contains("DLPFC").any()
    assert not resolved.astype(str).apply(lambda col: col.str.contains("MFBA9", regex=False)).any().any()
    assert not resolved_stats.astype(str).apply(
        lambda col: col.str.contains("MFBA9", regex=False)
    ).any().any()
    assert set(resolved["component"].astype(str)) == set(
        resolved_stats["component"].astype(str)
    )
    kegg_columns = pq.ParquetFile(data.KEGG_PARQUET).schema_arrow.names
    assert "overlap_DLPFC" in kegg_columns
    assert "overlap_MFBA9BA46" not in kegg_columns


def test_m263_dlpfc_statistics_match_the_scatter_component() -> None:
    plot_data = data.load_resolved("control_anchored", 263, "negative_density")
    statistics = data.load_resolved_statistics(
        "control_anchored", 263, "cogn_global", "negative_density"
    )
    component = "CT_AC__DLPFC"
    assert component in set(plot_data["component"].astype(str))
    selected = statistics.loc[
        statistics["component"].astype(str).eq(component)
        & statistics["diagnosis_group"].eq("AD")
    ]
    assert len(selected) == 1
    row = selected.iloc[0]
    assert int(row["n"]) == 124
    assert round(float(row["r_rint"]), 6) == 0.076391
    assert round(float(row["p_rint"]), 6) == 0.399079


def test_kegg_table_is_complete_and_module_filter_works() -> None:
    full = data.load_kegg()
    source = pd.read_csv(data.KEGG_TSV, sep="\t")
    assert len(full) == len(source) == 8866
    assert len(data.load_kegg(1918)) == 156


def test_mdc_table_matches_modules_and_directional_fdr() -> None:
    mdc = data.load_mdc_summary()
    annotations = data.load_module_annotations()
    assert len(mdc) == mdc["module"].nunique() == 154
    assert set(mdc["module"].astype(int)) == set(annotations["module"].astype(int))
    assert mdc[["mdc_total", "mdc_ts"]].gt(0).all().all()

    m1918 = mdc.loc[mdc["module"].eq(1918)].iloc[0]
    assert round(float(m1918["mdc_total"]), 6) == 1.246025
    assert round(float(m1918["mdc_ts"]), 6) == 1.099733
    assert round(float(m1918["mdc_ct"]), 6) == 1.285022
    assert m1918["direction_total"] == "Higher in AD"
    assert round(float(m1918["directional_fdr_total"]), 5) == 0.01666
    assert bool(m1918["significant_total_fdr05"])
    assert not bool(m1918["significant_ts_fdr05"])
    assert bool(m1918["significant_ct_fdr05"])

    missing_ct = mdc.loc[mdc["mdc_ct"].isna()]
    assert len(missing_ct) == 6
    assert missing_ct["n_ct_edges"].eq(0).all()

    manifest = data.load_data_manifest()
    assert manifest["rows"]["mdc_rows"] == 154
    assert manifest["mdc"]["sample_permutations"] == 200
    assert manifest["mdc"]["gene_permutations"] == 200
