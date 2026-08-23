from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
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


def test_targeted_prediction_loader_uses_predicates_and_preserves_privacy(
    tmp_path: Path, monkeypatch
) -> None:
    directory = tmp_path / "prediction_targeted"
    directory.mkdir()
    table = pd.DataFrame(
        {
            "evidence_tier": ["primary", "secondary"],
            "module_definition": ["control_derived", "full_cohort"],
            "model_outcome": ["diagnosis_binary", "cogdx"],
            "sample_id": ["T-AAA", "T-BBB"],
            "value": [0.71, 1.25],
        }
    )
    path = directory / "targeted_oof_performance.parquet"
    table.to_parquet(path, index=False)
    manifest = directory / "targeted_prediction_public_manifest.json"
    manifest.write_text('{"complete": true}\n', encoding="utf-8")
    files = dict(data.TARGETED_PREDICTION_FILES)
    files["oof_performance"] = path
    monkeypatch.setattr(data, "TARGETED_PREDICTION_DIR", directory)
    monkeypatch.setattr(data, "TARGETED_PREDICTION_MANIFEST", manifest)
    monkeypatch.setattr(data, "TARGETED_PREDICTION_FILES", files)
    assert data.targeted_prediction_data_available()
    selected = data.load_targeted_prediction_table(
        "oof_performance", evidence_tier="primary"
    )
    assert selected["sample_id"].tolist() == ["T-AAA"]
    assert {"donor", "projid"}.isdisjoint(selected.columns)


def test_all_modules_and_m1918_are_packaged() -> None:
    annotations = data.load_module_annotations()
    assert annotations["module"].nunique() == 154
    assert 1918 in set(annotations["module"].astype(int))
    for method in data.MODULE_SET_METHODS["full_cohort"]:
        selected = data.load_aggregate(method, 1918, "abs_sum")
        assert len(selected) == 450
        assert selected["sample_id"].nunique() == 450


def test_differential_scores_keep_all_edges_and_offer_both_bh_scopes() -> None:
    assert data.differential_data_available("full_cohort")
    unfiltered = data.load_aggregate(
        "control_anchored", 935, "connectivity", module_set="full_cohort"
    )
    explicit_unfiltered = data.load_aggregate(
        "control_anchored",
        935,
        "connectivity",
        module_set="full_cohort",
        differential_edge_rule="all",
        differential_fdr_scope="per_module",
        differential_fdr_threshold=0.10,
    )
    pd.testing.assert_frame_equal(unfiltered, explicit_unfiltered)

    selections = []
    for scope in ("global", "per_module"):
        for threshold in (0.05, 0.10):
            selected = data.load_aggregate(
                "control_anchored",
                935,
                "connectivity",
                module_set="full_cohort",
                differential_edge_rule="ad_control_discovery_fdr05",
                differential_fdr_scope=scope,
                differential_fdr_threshold=threshold,
            )
            assert len(selected) == selected["sample_id"].nunique() == 450
            assert set(selected["differential_fdr_scope"].astype(str)) == {scope}
            assert set(
                pd.to_numeric(selected["differential_fdr_threshold"]).round(2)
            ) == {threshold}
            selections.append(selected)
    assert not selections[0].equals(selections[-1])

    candidates = data.load_volcano_candidates(
        "full_cohort", "lioness", "control_anchored", 935
    )
    assert {
        "discovery_fdr_global",
        "discovery_fdr_per_module",
        "validation_fdr_global",
        "validation_fdr_per_module",
    }.issubset(candidates.columns)
    for scope in ("global", "per_module"):
        bins = data.load_volcano_bins(
            "full_cohort", "lioness", "control_anchored", 935, scope
        )
        assert not bins.empty
        assert set(bins["fdr_scope"].astype(str)) == {scope}
    summaries = data.load_edge_summaries(
        "lioness",
        "control_anchored",
        935,
        module_set="full_cohort",
        differential_edge_rule="ad_control_discovery_fdr05",
        differential_fdr_scope="per_module",
        differential_fdr_threshold=0.10,
    )
    assert len(summaries) == 450 * 9
    assert set(summaries["differential_fdr_scope"].astype(str)) == {"per_module"}
    assert set(pd.to_numeric(summaries["differential_fdr_threshold"]).round(2)) == {
        0.10
    }


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
    assert details["tissue_entropy"].between(0.0, 1.5849625008).all()
    assert details["tissue_entropy_normalized"].between(0.0, 1.0).all()

    m1012 = data.load_module_details(1012).iloc[0]
    assert int(m1012["module_size"]) == 152
    assert int(m1012["n_genes_ac"]) == 151
    assert int(m1012["n_genes_dlpfc"]) == 0
    assert int(m1012["n_genes_pcg"]) == 1
    assert m1012["tissues"] == "AC, PCG"

    manifest = data.load_data_manifest()
    assert manifest["rows"]["module_details_rows"] == 154
    assert "module_details.tsv" in manifest["files"]


def test_pathway_mdc_join_uses_component_matched_kegg_fdrs() -> None:
    expected_components = {
        "total",
        "TS",
        "CT",
        "TS_AC",
        "TS_DLPFC",
        "TS_PCGBA23",
        "CT_AC__DLPFC",
        "CT_AC__PCGBA23",
        "CT_DLPFC__PCGBA23",
    }
    for module_set in data.MODULE_SET_LABELS:
        rows = data.build_pathway_mdc_rows(
            data.load_mdc_summary(module_set),
            data.load_mdc_resolved(module_set),
            data.load_kegg(module_set=module_set),
            enrichment_fdr_threshold=0.05,
        )
        assert not rows.empty
        assert set(rows["component"].astype(str)) == expected_components
        assert rows["enrichment_fdr"].le(0.05).all()

        ac = rows.loc[rows["component"].eq("TS_AC")]
        assert np.allclose(ac["enrichment_fdr"], ac["fdr_AC"])
        pair = rows.loc[rows["component"].eq("CT_AC__DLPFC")]
        expected_pair_fdr = pair[["fdr_AC", "fdr_DLPFC"]].max(axis=1)
        assert np.allclose(pair["enrichment_fdr"], expected_pair_fdr)
        assert pair["fdr_AC"].le(0.05).all()
        assert pair["fdr_DLPFC"].le(0.05).all()

        summary = data.summarize_pathway_mdc_rows(
            rows, mdc_fdr_threshold=0.05, minimum_modules=2
        )
        assert not summary.empty
        assert summary["n_modules"].ge(2).all()
        assert np.allclose(
            summary["geometric_mean_mdc"], np.exp2(summary["mean_log2_mdc"])
        )
        assert summary["proportion_mdc_significant"].between(0.0, 1.0).all()

        resolution_counts = {}
        for resolution in ("pathway", "subcategory", "category"):
            collapsed = data.collapse_pathway_mdc_rows(
                rows, resolution=resolution
            )
            assert not collapsed.duplicated(
                ["pathway_id", "module", "component"]
            ).any()
            assert collapsed["supporting_pathway_count"].ge(1).all()
            resolution_summary = data.summarize_pathway_mdc_rows(
                rows,
                mdc_fdr_threshold=0.05,
                minimum_modules=1,
                resolution=resolution,
            )
            assert not resolution_summary.empty
            assert set(resolution_summary["enrichment_resolution"]) == {
                resolution
            }
            assert resolution_summary["n_pathways"].ge(1).all()
            resolution_counts[resolution] = resolution_summary[
                "pathway_id"
            ].nunique()
        assert (
            resolution_counts["category"]
            <= resolution_counts["subcategory"]
            <= resolution_counts["pathway"]
        )


def test_legacy_module_details_reconstruct_tissue_entropy() -> None:
    legacy = pd.DataFrame(
        {
            "proportion_ac": [1.0, 1.0 / 3.0],
            "proportion_dlpfc": [0.0, 1.0 / 3.0],
            "proportion_pcg": [0.0, 1.0 / 3.0],
        }
    )
    restored = data.ensure_tissue_entropy(legacy)
    assert float(restored.loc[0, "tissue_entropy"]) == 0.0
    assert float(restored.loc[0, "tissue_entropy_normalized"]) == 0.0
    assert abs(float(restored.loc[1, "tissue_entropy"]) - np.log2(3.0)) < 1e-12
    assert abs(float(restored.loc[1, "tissue_entropy_normalized"]) - 1.0) < 1e-12


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


def test_kegg_contains_valid_whole_and_per_region_statistics() -> None:
    region_columns = {
        "AC": ("p_AC", "fdr_AC", "significant_AC"),
        "DLPFC": ("p_DLPFC", "fdr_DLPFC", "significant_DLPFC"),
        "PCG": ("p_PCGBA23", "fdr_PCGBA23", "significant_PCGBA23"),
    }
    manifest = data.load_data_manifest()
    for module_set in data.MODULE_SET_LABELS:
        frame = data.load_kegg(module_set=module_set)
        assert {"p", "fdr", "significant"}.issubset(frame.columns)
        assert not any("MFBA9BA46" in column for column in frame.columns)
        for p_column, fdr_column, significant_column in region_columns.values():
            assert {p_column, fdr_column, significant_column}.issubset(frame.columns)
            assert frame[p_column].notna().all()
            assert frame[fdr_column].notna().all()
            assert frame[p_column].between(0.0, 1.0).all()
            assert frame[fdr_column].between(0.0, 1.0).all()
            assert frame[significant_column].astype(bool).eq(
                frame[fdr_column].le(0.05)
            ).all()

        kegg_manifest = manifest["module_sets"][module_set]["kegg"]
        assert kegg_manifest["join_coverage"] == 1.0
        assert kegg_manifest["per_region_pathways_per_module_min"] == 350
        assert kegg_manifest["per_region_pathways_per_module_max"] == 350
        assert set(kegg_manifest["region_statistics"]) == {"AC", "DLPFC", "PCG"}


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

    dlpfc_target = full.loc[full["significant_DLPFC"].astype(bool)].iloc[0]
    dlpfc = data.filter_kegg_enrichments(
        full,
        modules=[int(dlpfc_target["cluster_id"])],
        significance="significant",
        maximum_fdr=0.05,
        significance_columns=["significant_DLPFC"],
        fdr_columns=["fdr_DLPFC"],
    )
    assert not dlpfc.empty
    assert dlpfc["significant_DLPFC"].astype(bool).all()
    assert dlpfc["fdr_DLPFC"].le(0.05).all()
    assert dlpfc["fdr_DLPFC"].is_monotonic_increasing

    any_region = data.filter_kegg_enrichments(
        full,
        significance="significant",
        significance_columns=[
            "significant_AC",
            "significant_DLPFC",
            "significant_PCGBA23",
        ],
        fdr_columns=["fdr_AC", "fdr_DLPFC", "fdr_PCGBA23"],
    )
    assert any_region[
        ["significant_AC", "significant_DLPFC", "significant_PCGBA23"]
    ].astype(bool).any(axis=1).all()


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


def test_resolved_mdc_has_six_components_and_conservative_fdr() -> None:
    for module_set, module_count in {"full_cohort": 154, "control_derived": 186}.items():
        resolved = data.load_mdc_resolved(module_set)
        assert len(resolved) == module_count * 6
        assert resolved["module"].nunique() == module_count
        assert resolved["component"].nunique() == 6
        assert resolved["mdc"].isna().eq(resolved["n_edges"].eq(0)).all()
        unavailable = resolved["n_edges"].eq(0)
        inferential_columns = [
            "p_loss_sample", "p_loss_gene", "q_loss_sample", "q_loss_gene",
            "p_gain_sample", "p_gain_gene", "q_gain_sample", "q_gain_gene",
            "directional_p_sample", "directional_p_gene", "directional_fdr",
        ]
        assert not resolved.loc[unavailable, inferential_columns].notna().any().any()
        assert not resolved.astype(str).apply(
            lambda column: column.str.contains("MFBA9", regex=False)
        ).any().any()
        finite = resolved["directional_fdr"].notna()
        expected = resolved.loc[finite, ["directional_p_sample", "directional_p_gene"]]
        assert expected.notna().all().all()
        ad_higher = finite & resolved["mdc"].ge(1)
        control_higher = finite & resolved["mdc"].lt(1)
        assert (
            resolved.loc[ad_higher, "directional_fdr"]
            - resolved.loc[ad_higher, ["q_loss_sample", "q_loss_gene"]].max(axis=1)
        ).abs().max() < 1e-12
        assert (
            resolved.loc[control_higher, "directional_fdr"]
            - resolved.loc[control_higher, ["q_gain_sample", "q_gain_gene"]].max(axis=1)
        ).abs().max() < 1e-12


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
    assert aggregate_path.stat().st_size < 95 * 1024 * 1024
    assert resolved_path.stat().st_size < 95 * 1024 * 1024

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
    assert manifest["rows"]["aggregate_stat_rows"] == 40_176
    assert manifest["rows"]["resolved_stat_rows"] == 241_056
    assert manifest["ct_unavailable_modules"] == [356]
    assert manifest["ts_labeled_modules"] == [355, 356, 867]
    assert manifest["kegg"]["input_modules"] == 186
    assert manifest["kegg"]["admitted_modules"] == 186
    assert manifest["formula_validation"]["groups"] == ["AD", "Control", "MCI"]
    assert manifest["formula_validation"]["max_relative_error"] < 1e-8


def test_lioness_statistics_cover_all_twelve_outcomes() -> None:
    configurations = [
        ("full_cohort", "standard", 154),
        ("full_cohort", "control_anchored", 154),
        ("control_derived", "control_anchored", 186),
    ]
    for module_set, method, module_count in configurations:
        aggregate = data.load_aggregate_statistics(method, module_set=module_set)
        resolved = data.load_resolved_statistics(method, module_set=module_set)
        assert len(aggregate) == module_count * 6 * 12 * 3
        assert len(resolved) == module_count * 6 * 6 * 12 * 3
        assert set(aggregate["phenotype"]) == set(data.NUMERIC_OUTCOMES)
        assert {
            "q_spearman_CT_all12_global",
            "q_spearman_TS_all12_global",
            "q_component_rank_all12_global",
            "q_rint_CT_all12_global",
            "q_rint_TS_all12_global",
            "q_component_rint_all12_global",
        }.issubset(aggregate.columns)
        families = data.load_data_manifest()["module_sets"][module_set][
            "fdr_test_families"
        ]
        assert families["lioness"]["all12_aggregate_global"] == (
            module_count * 6 * 12 * 3
        )
        assert families["bonobo_per_edge_rule"]["all12_aggregate_global"] == (
            module_count * 3 * 12 * 3
        )


def test_feature_definitions_cover_lioness_and_bonobo_rules() -> None:
    definitions = data.load_feature_definitions()
    assert set(definitions["estimator"]) == {"LIONESS", "BONOBO"}
    assert set(
        definitions.loc[definitions["estimator"].eq("BONOBO"), "feature_family"]
    ) == {"connectivity", "positive_density", "negative_density"}
    bonobo_rules = " ".join(
        definitions.loc[definitions["estimator"].eq("BONOBO"), "edge_rules"]
        .astype(str)
        .tolist()
    )
    assert "posterior p<0.05" in bonobo_rules
    assert "BH FDR<0.05" in bonobo_rules


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
        assert path.stat().st_size < 95 * 1024 * 1024
        assert file_sha256(path) == expected["sha256"]
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            assert {"donor", "projid"}.isdisjoint(parquet.schema_arrow.names)
            assert expected["rows"] == parquet.metadata.num_rows
            assert expected["schema"] == {
                field.name: str(field.type) for field in parquet.schema_arrow
            }


def test_bonobo_rules_are_complete_private_and_explicit_about_zero_states() -> None:
    for module_set, module_count in {"full_cohort": 154, "control_derived": 186}.items():
        for edge_rule in ("all", "native_p05", "bh_fdr05"):
            aggregate_path = data.estimator_path(
                "aggregate_plot_data.parquet", module_set, "bonobo", edge_rule
            )
            resolved_path = data.estimator_path(
                "resolved_plot_data.parquet", module_set, "bonobo", edge_rule
            )
            assert pq.ParquetFile(aggregate_path).metadata.num_rows == module_count * 450 * 3
            assert pq.ParquetFile(resolved_path).metadata.num_rows == module_count * 450 * 3 * 6
            for path in (aggregate_path, resolved_path):
                assert {"donor", "projid"}.isdisjoint(
                    pq.ParquetFile(path).schema_arrow.names
                )
                assert path.stat().st_size < 95 * 1024 * 1024
            aggregate_stats = data.load_aggregate_statistics(
                "bonobo", module_set=module_set, estimator="bonobo",
                edge_rule=edge_rule,
            )
            resolved_stats = data.load_resolved_statistics(
                "bonobo", module_set=module_set, estimator="bonobo",
                edge_rule=edge_rule,
            )
            assert len(aggregate_stats) == module_count * 3 * 12 * 3
            assert len(resolved_stats) == module_count * 3 * 6 * 12 * 3
            assert set(aggregate_stats["phenotype"]) == set(data.NUMERIC_OUTCOMES)
            assert {
                "q_spearman_CT_all12_global",
                "q_spearman_TS_all12_global",
                "q_component_rank_all12_global",
                "q_rint_CT_all12_global",
                "q_rint_TS_all12_global",
                "q_component_rint_all12_global",
            }.issubset(aggregate_stats.columns)
        sparse = data.load_aggregate(
            "bonobo", 34 if module_set == "full_cohort" else 10,
            "connectivity", module_set=module_set,
            estimator="bonobo", edge_rule="bh_fdr05",
        )
        assert len(sparse) == 450
        if module_set == "full_cohort":
            m1918 = data.load_aggregate(
                "bonobo", 1918, "connectivity", module_set=module_set,
                estimator="bonobo", edge_rule="all",
            )
            assert len(m1918) == m1918["sample_id"].nunique() == 450


def test_edge_summaries_cover_nine_scopes_and_obey_identities() -> None:
    for module_set, method, module in [
        ("full_cohort", "standard", 935),
        ("full_cohort", "control_anchored", 935),
        ("control_derived", "control_anchored", 935),
    ]:
        frame = data.load_edge_summaries(
            "lioness", method, module, module_set=module_set
        )
        assert len(frame) == 450 * 9
        assert frame["scope"].nunique() == 9
        assert (frame["n_retained_edges"] == frame["n_possible_edges"]).all()
        assert (frame["n_pruned_edges"] == 0).all()
        assert (
            frame["n_positive_edges"]
            + frame["n_negative_edges"]
            + frame["n_zero_edges"]
            == frame["n_retained_edges"]
        ).all()
        assert (
            frame["positive_weight_sum"] + frame["negative_weight_magnitude"]
            - frame["absolute_weight_sum"]
        ).abs().max() < 1e-8

    for module_set, module in [("full_cohort", 935), ("control_derived", 935)]:
        for edge_rule in ("all", "native_p05", "bh_fdr05"):
            frame = data.load_edge_summaries(
                "bonobo", "bonobo", module, module_set=module_set,
                edge_rule=edge_rule,
            )
            assert len(frame) == 450 * 9
            assert frame["scope"].nunique() == 9
            unavailable = frame["n_possible_edges"].eq(0)
            assert frame.loc[unavailable, "n_retained_edges"].isna().all()
            available = frame.loc[~unavailable]
            assert (
                available["n_retained_edges"] <= available["n_possible_edges"]
            ).all()
            assert (
                available["n_positive_edges"]
                + available["n_negative_edges"]
                + available["n_zero_edges"]
                == available["n_retained_edges"]
            ).all()
            assert (
                available["n_possible_edges"]
                - available["n_retained_edges"]
                == available["n_pruned_edges"]
            ).all()
            assert (
                available["positive_weight_sum"]
                + available["negative_weight_magnitude"]
                - available["absolute_weight_sum"]
            ).abs().max() < 1e-8


def test_shared_anonymous_samples_cover_all_estimators_and_module_sets() -> None:
    expected = set(data.load_sample_metadata()["sample_id"])
    assert len(expected) == 450
    for module_set, module in [("full_cohort", 935), ("control_derived", 935)]:
        lioness = data.load_aggregate(
            "control_anchored", module, "connectivity", module_set=module_set
        )
        bonobo = data.load_aggregate(
            "bonobo", module, "connectivity", module_set=module_set,
            estimator="bonobo", edge_rule="all",
        )
        assert set(lioness["sample_id"]) == expected
        assert set(bonobo["sample_id"]) == expected
