from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (
    kegg_region_heatmap_figure,
    mdc_resolved_heatmap_figure,
    module_region_composition_figure,
)
from app_helpers.data import (
    add_kegg_region_sharing,
    module_set_components,
    module_set_tissue_detail_columns,
    module_set_tissues,
)
from scripts.build_muscle_public_data import (
    _fast_corr,
    _fast_spearman,
    _leave_one_out,
    _williams_test,
    robust_scale,
)
from scripts.run_muscle_m5_mdc_significance import (
    bh as mdc_bh,
    empirical_lower,
    empirical_upper,
)


MUSCLE_DATA = APP_ROOT / "data/muscle_control_l4"


def test_muscle_module_definition_has_distinct_tissues_and_components() -> None:
    assert module_set_tissues("muscle_control_l4") == ("VH", "SMA", "QUAD")
    assert module_set_components("muscle_control_l4") == (
        "CT_VH__SMA",
        "CT_VH__QUAD",
        "CT_SMA__QUAD",
        "TS_VH",
        "TS_SMA",
        "TS_QUAD",
    )


def test_fast_statistics_match_scipy_with_missing_values_and_ties() -> None:
    x = np.asarray([1.0, 2.0, 2.0, 4.0, np.nan, 7.0, 8.0])
    y = np.asarray([2.0, 1.0, 3.0, 8.0, 5.0, np.nan, 9.0])
    valid = np.isfinite(x) & np.isfinite(y)
    n, coefficient, p_value = _fast_corr(x, y)
    expected = pearsonr(x[valid], y[valid])
    assert n == int(valid.sum())
    assert np.isclose(coefficient, expected.statistic, atol=1e-14)
    assert np.isclose(p_value, expected.pvalue, atol=1e-14)

    complete_x = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 8.0])
    complete_y = np.asarray([2.0, 1.0, 3.0, 8.0, 5.0, 6.0, 9.0])
    full = pearsonr(complete_x, complete_y).statistic
    deleted = np.asarray(
        [
            pearsonr(np.delete(complete_x, index), np.delete(complete_y, index)).statistic
            for index in range(len(complete_x))
        ]
    )
    loo = _leave_one_out(complete_x, complete_y)
    assert np.isclose(loo[0], np.max(np.abs(deleted - full)))
    assert np.isclose(loo[1], np.min(np.abs(deleted)))
    assert loo[2] == float(np.any(np.sign(deleted) != np.sign(full)))
    assert robust_scale(np.asarray([-10.0, -2.0, 1.0, 3.0, 7.0])) == 3.0
    statistic, p_value = _williams_test(0.3, 0.1, 0.4, 30)
    assert np.isclose(statistic, 0.9921424868308382)
    assert np.isclose(p_value, 0.3299368033663729)
    n, coefficient, p_value = _fast_spearman(x, y)
    expected = spearmanr(x[valid], y[valid])
    assert n == int(valid.sum())
    assert np.isclose(coefficient, expected.statistic, atol=1e-14)
    assert np.isclose(p_value, expected.pvalue, atol=1e-14)
    assert module_set_tissue_detail_columns("muscle_control_l4") == (
        ("VH", "n_genes_vh", "proportion_vh"),
        ("SMA", "n_genes_sma", "proportion_sma"),
        ("QUAD", "n_genes_quad", "proportion_quad"),
    )


def test_muscle_kegg_sharing_and_heatmap_use_only_muscle_regions() -> None:
    frame = pd.DataFrame(
        {
            "cluster_id": [7],
            "pathway_name": ["Example pathway"],
            "fdr_VH": [0.01],
            "fdr_SMA": [0.03],
            "fdr_QUAD": [0.4],
            "p_VH": [0.001],
            "p_SMA": [0.002],
            "p_QUAD": [0.2],
            "overlap_VH": [4],
            "overlap_SMA": [5],
            "overlap_QUAD": [1],
        }
    )
    annotated = add_kegg_region_sharing(frame, 0.05)
    assert annotated.loc[0, "significant_region_count"] == 2
    assert annotated.loc[0, "significant_regions"] == "VH + SMA"
    figure = kegg_region_heatmap_figure(annotated)
    assert list(figure.data[0].x) == ["VH", "SMA", "QUAD"]


def test_muscle_module_composition_chart_uses_vh_sma_quad() -> None:
    frame = pd.DataFrame(
        {
            "module": [1, 2],
            "module_size": [60, 45],
            "cluster_type": ["CT", "TS"],
            "n_genes_vh": [20, 45],
            "n_genes_sma": [25, 0],
            "n_genes_quad": [15, 0],
        }
    )
    figure = module_region_composition_figure(frame)
    assert [trace.name for trace in figure.data] == ["VH", "SMA", "QUAD"]
    np.testing.assert_array_equal(np.asarray(figure.data[0].x), [20, 45])


def test_muscle_resolved_mdc_heatmap_keeps_all_six_components() -> None:
    frame = pd.DataFrame(
        {
            "module": np.repeat([1, 2], 6),
            "component_label": [
                "TS: VH", "TS: SMA", "TS: QUAD", "CT: VH - SMA",
                "CT: VH - QUAD", "CT: SMA - QUAD",
            ] * 2,
            "mdc": np.linspace(0.5, 1.6, 12),
            "log2_mdc": np.log2(np.linspace(0.5, 1.6, 12)),
            "directional_fdr": [np.nan] * 12,
        }
    )
    figure = mdc_resolved_heatmap_figure(
        frame, 0.05, selected_module=1, scale="raw"
    )
    assert list(figure.data[0].x) == [
        "TS: VH", "TS: SMA", "TS: QUAD", "CT: VH - SMA",
        "CT: VH - QUAD", "CT: SMA - QUAD",
    ]


def test_muscle_mdc_empirical_p_values_and_bh() -> None:
    null = np.asarray([0.5, 1.0, 1.5, np.nan])
    assert empirical_upper(1.0, null) == 0.75
    assert empirical_lower(1.0, null) == 0.75
    adjusted = mdc_bh(pd.Series([0.01, 0.04, np.nan, 0.03]))
    np.testing.assert_allclose(adjusted.iloc[[0, 1, 3]], [0.03, 0.04, 0.04])
    assert pd.isna(adjusted.iloc[2])


def test_muscle_extreme_mdc_ratios_are_marked_descriptive_only() -> None:
    resolved = pd.read_csv(
        MUSCLE_DATA / "mdc_resolved_ad_vs_control.tsv", sep="\t"
    )
    expected = {
        (542, "TS_SMA"): "near-zero AD or Control mean absolute weight",
        (972, "TS_QUAD"): "fewer than 10 edges",
    }
    for key, reason in expected.items():
        module, component = key
        row = resolved.loc[
            resolved["module"].eq(module) & resolved["component"].eq(component)
        ].iloc[0]
        assert not bool(row["mdc_stable_for_inference"])
        assert row["mdc_stability_reason"] == reason
        assert pd.isna(row["directional_fdr"])


def test_public_muscle_bundle_dimensions_and_privacy() -> None:
    details = pd.read_csv(MUSCLE_DATA / "module_details.tsv", sep="\t")
    metadata = pd.read_parquet(MUSCLE_DATA / "sample_metadata.parquet")
    aggregate = pd.read_parquet(
        MUSCLE_DATA / "aggregate_plot_data.parquet", columns=["module", "sample_id"]
    )
    resolved = pd.read_parquet(
        MUSCLE_DATA / "resolved_plot_data.parquet",
        columns=["module", "sample_id", "component"],
    )
    kegg = pd.read_parquet(MUSCLE_DATA / "kegg_tissue_expanded_full.parquet")
    mdc = pd.read_csv(MUSCLE_DATA / "mdc_ad_vs_control_summary.tsv", sep="\t")
    mdc_resolved = pd.read_csv(
        MUSCLE_DATA / "mdc_resolved_ad_vs_control.tsv", sep="\t"
    )

    assert len(details) == details["module"].nunique() == 155
    assert details["module_size"].sum() == 33_491
    assert len(metadata) == metadata["sample_id"].nunique() == 357
    assert metadata["diagnosis_group"].value_counts().to_dict() == {
        "AD": 155,
        "Control": 120,
        "MCI": 82,
    }
    assert len(aggregate) == 332_010
    assert len(resolved) == 1_992_060
    assert set(resolved["component"].astype(str)) == set(
        module_set_components("muscle_control_l4")
    )
    assert len(kegg) > 0
    assert set(kegg["cluster_id"].astype(int)).issubset(set(details["module"].astype(int)))
    assert not kegg.duplicated(["cluster_id", "pathway_id"]).any()
    assert len(mdc) == 155
    assert len(mdc_resolved) == 155 * 6
    for frame in (metadata, aggregate, resolved, kegg, mdc, mdc_resolved):
        assert {"donor", "projid", "ProjID"}.isdisjoint(frame.columns)


def test_public_muscle_statistics_cover_all_modules_and_outcomes() -> None:
    aggregate_path = MUSCLE_DATA / "aggregate_statistics.parquet"
    resolved_path = MUSCLE_DATA / "resolved_statistics.parquet"
    if not aggregate_path.exists() or not resolved_path.exists():
        # The builder publishes data atomically; this guard permits focused UI
        # tests while a local statistics-only checkpoint is still running.
        return
    aggregate = pd.read_parquet(
        aggregate_path, columns=["module", "phenotype", "lioness_method"]
    )
    resolved = pd.read_parquet(
        resolved_path,
        columns=["module", "phenotype", "lioness_method", "component"],
    )
    assert len(aggregate) == 33_480
    assert len(resolved) == 200_880
    assert aggregate["module"].nunique() == resolved["module"].nunique() == 155
    assert aggregate["phenotype"].nunique() == resolved["phenotype"].nunique() == 12
    assert set(aggregate["lioness_method"].astype(str)) == {"control_anchored"}
    assert set(resolved["lioness_method"].astype(str)) == {"control_anchored"}


def test_public_muscle_eigengenes_are_tissue_specific_and_private() -> None:
    path = MUSCLE_DATA / "module_eigengenes.parquet"
    if not path.exists():
        return
    scores = pd.read_parquet(path)
    assert len(scores) == (124 + 146 + 138) * 357
    assert scores["sample_id"].nunique() == 357
    assert scores["module"].nunique() == 155
    assert scores["component"].value_counts().to_dict() == {
        "TS_SMA": 146 * 357,
        "TS_QUAD": 138 * 357,
        "TS_VH": 124 * 357,
    }
    assert set(scores["metric_family"]) == {"eigengene"}
    assert not scores[["metric_raw", "metric_asinh", "metric_rint"]].isna().any().any()
    assert {"donor", "projid", "ProjID"}.isdisjoint(scores.columns)
