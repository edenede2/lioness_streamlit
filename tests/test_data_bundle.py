from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers import data  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def test_all_module_kegg_filters_cover_modules_categories_fdr_and_search() -> None:
    full = data.load_kegg()
    target = full.loc[full["significant"].fillna(False).astype(bool)].iloc[0]
    filtered = data.filter_kegg_enrichments(
        full,
        modules=[int(target["cluster_id"])],
        categories=[str(target["category_level1"])],
        subcategories=[str(target["category_level2"])],
        significance="significant",
        maximum_fdr=0.05,
        search=str(target["pathway_name"]),
    )
    assert not filtered.empty
    assert set(filtered["cluster_id"].astype(int)) == {int(target["cluster_id"])}
    assert set(filtered["category_level1"]) == {str(target["category_level1"])}
    assert set(filtered["category_level2"]) == {str(target["category_level2"])}
    assert filtered["significant"].fillna(False).astype(bool).all()
    assert filtered["fdr"].le(0.05).all()
    assert filtered["pathway_name"].str.contains(
        str(target["pathway_name"]), case=False, regex=False
    ).all()
    assert filtered["fdr"].is_monotonic_increasing

    all_rows = data.filter_kegg_enrichments(full, modules=[])
    assert len(all_rows) == len(full)
    non_significant = data.filter_kegg_enrichments(
        full, significance="not_significant"
    )
    assert not non_significant["significant"].fillna(False).astype(bool).any()


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


def test_control_derived_bundle_is_isolated_complete_and_private() -> None:
    module_set = "control_derived"
    annotations = data.load_module_annotations(module_set)
    details = data.load_module_details(module_set=module_set)
    mdc = data.load_mdc_summary(module_set)
    assert len(annotations) == annotations["module"].nunique() == 186
    assert len(details) == details["module"].nunique() == 186
    assert len(mdc) == mdc["module"].nunique() == 186
    assert set(annotations["module"].astype(int)) == set(details["module"].astype(int))
    assert set(annotations["module"].astype(int)) == set(mdc["module"].astype(int))

    aggregate_path = data.module_set_path("aggregate_plot_data.parquet", module_set)
    resolved_path = data.module_set_path("resolved_plot_data.parquet", module_set)
    aggregate_file = pq.ParquetFile(aggregate_path)
    resolved_file = pq.ParquetFile(resolved_path)
    assert aggregate_file.metadata.num_rows == 450 * 186 * 6
    assert resolved_file.metadata.num_rows == 450 * 186 * 6 * 6
    for parquet in (aggregate_file, resolved_file):
        assert {"donor", "projid"}.isdisjoint(parquet.schema_arrow.names)
    assert aggregate_path.stat().st_size < 100 * 1024 * 1024
    assert resolved_path.stat().st_size < 100 * 1024 * 1024

    aggregate = data.load_aggregate(
        "control_anchored", 935, "connectivity", module_set=module_set
    )
    resolved = data.load_resolved(
        "control_anchored", 935, "connectivity", module_set=module_set
    )
    assert len(aggregate) == aggregate["sample_id"].nunique() == 450
    assert set(aggregate["lioness_method"].astype(str)) == {"control_anchored"}
    assert len(resolved) == 450 * 6
    assert resolved["component"].nunique() == 6
    assert set(aggregate["sample_id"]) == set(
        data.load_aggregate("control_anchored", 935, "connectivity")["sample_id"]
    )

    # M935 exists in both definitions but represents different gene memberships.
    full_m935 = data.load_module_details(935).iloc[0]
    control_m935 = data.load_module_details(935, module_set=module_set).iloc[0]
    assert int(full_m935["module_size"]) == 58
    assert int(control_m935["module_size"]) == 527

    unavailable_ct = mdc.loc[mdc["n_ct_edges"].eq(0), "module"].astype(int).tolist()
    assert unavailable_ct == [356]
    assert mdc.loc[mdc["module"].eq(356), "mdc_ct"].isna().all()

    manifest = data.load_data_manifest()["module_sets"][module_set]
    assert manifest["methods"] == ["control_anchored"]
    assert manifest["rows"]["aggregate_rows"] == 502_200
    assert manifest["rows"]["resolved_rows"] == 3_013_200
    assert manifest["rows"]["aggregate_stat_rows"] == 16_740
    assert manifest["rows"]["resolved_stat_rows"] == 100_440
    assert manifest["ct_unavailable_modules"] == [356]
    assert manifest["ts_labeled_modules"] == [355, 356, 867]
    assert manifest["kegg"]["input_modules"] == 186
    assert manifest["kegg"]["admitted_modules"] == 186
    assert manifest["formula_validation"]["groups"] == ["AD", "Control", "MCI"]
    assert manifest["formula_validation"]["max_relative_error"] < 1e-8


def test_control_derived_kegg_keeps_ts_modules_and_unavailable_annotations() -> None:
    module_set = "control_derived"
    annotations = data.load_module_annotations(module_set)
    reported = data.load_kegg(module_set=module_set)
    assert {355, 356, 867}.issubset(set(reported["cluster_id"].astype(int)))
    unavailable = annotations.loc[
        ~annotations["annotation_available"].fillna(False).astype(bool), "module"
    ]
    manifest = data.load_data_manifest()["module_sets"][module_set]
    assert sorted(unavailable.astype(int)) == manifest["kegg"][
        "modules_without_reported_pathways"
    ]


def test_manifest_file_hashes_and_github_size_limit() -> None:
    manifest = data.load_data_manifest()
    for relative_path, expected in manifest["files"].items():
        path = data.DATA_DIR / relative_path
        assert path.exists()
        assert path.stat().st_size == expected["bytes"]
        assert path.stat().st_size < 100 * 1024 * 1024
        assert file_sha256(path) == expected["sha256"]
