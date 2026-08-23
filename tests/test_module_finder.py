from __future__ import annotations

import numpy as np

from app_helpers.charts import module_finder_figure
from app_helpers.data import load_aggregate_statistics, load_module_annotations
from app_helpers.module_finder import build_module_finder_table


def test_module_finder_reconciles_both_association_differences() -> None:
    statistics = load_aggregate_statistics(
        "control_anchored", None, "motor10_demog_slope", "connectivity"
    )
    result = build_module_finder_table(
        statistics,
        ct_ts_diagnosis="AD",
        correlation_method="Spearman",
        criterion="both",
    )
    assert len(result) == 154
    assert result["module"].nunique() == 154
    assert result["finder_rank"].tolist() == list(range(1, 155))
    assert result["finder_score"].dropna().between(0, 1, inclusive="both").all()
    expected_joint = result[
        ["ct_ts_percentile_score", "ad_control_percentile_score"]
    ].min(axis=1, skipna=False)
    assert np.allclose(result["finder_score"], expected_joint, equal_nan=True)

    module = int(result.iloc[0]["module"])
    source = statistics.loc[statistics["module"].eq(module)].set_index(
        "diagnosis_group"
    )
    row = result.loc[result["module"].eq(module)].iloc[0]
    assert np.isclose(
        row["ct_ts_delta_correlation"],
        source.loc["AD", "rho_CT"] - source.loc["AD", "rho_TS"],
    )
    assert np.isclose(
        row["ad_control_delta_correlation_CT"],
        source.loc["AD", "rho_CT"] - source.loc["Control", "rho_CT"],
    )
    assert np.isclose(
        row["ad_control_delta_correlation_TS"],
        source.loc["AD", "rho_TS"] - source.loc["Control", "rho_TS"],
    )
    expected_fisher_z_ct = (
        np.arctanh(source.loc["AD", "rho_CT"])
        - np.arctanh(source.loc["Control", "rho_CT"])
    ) / np.sqrt(
        1 / (source.loc["AD", "n"] - 3)
        + 1 / (source.loc["Control", "n"] - 3)
    )
    assert np.isclose(row["ad_control_fisher_z_CT"], expected_fisher_z_ct)
    assert result[["ad_control_fdr_CT", "ad_control_fdr_TS"]].min().min() >= 0
    assert result[["ad_control_fdr_CT", "ad_control_fdr_TS"]].max().max() <= 1


def test_module_finder_chart_has_two_explicit_criteria_axes() -> None:
    statistics = load_aggregate_statistics(
        "control_anchored", None, "cogn_global", "positive_density"
    )
    result = build_module_finder_table(
        statistics,
        ct_ts_diagnosis="AD",
        correlation_method="Pearson",
        criterion="both",
    )
    annotations = load_module_annotations()
    result = result.merge(
        annotations[["module", "displayed_pathway", "displayed_fdr"]],
        on="module",
        how="left",
        validate="one_to_one",
    )
    result.insert(0, "display_rank", np.arange(1, len(result) + 1))
    figure = module_finder_figure(
        result,
        phenotype_label="Global cognition",
        feature_label="Positive density",
        correlation_method="Pearson",
        ct_ts_diagnosis="AD",
        criterion_label="Both criteria",
        selected_module=935,
    )
    assert "CT−TS" in str(figure.layout.xaxis.title.text)
    assert "Control−AD" in str(figure.layout.yaxis.title.text)
    assert "Both criteria" in str(figure.layout.title.text)
    assert len(figure.data[0].x) == 154
