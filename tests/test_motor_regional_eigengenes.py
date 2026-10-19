from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, rankdata, spearmanr
from streamlit.testing.v1 import AppTest

from app_helpers import regional_modules as R
from app_helpers import score_comparison as C


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
EXPECTED = {"VH": 56, "SMA": 99, "QUAD": 126}


def _widget(elements, label):
    return next(element for element in elements if element.label == label)


def _view(app, name):
    return _widget(app.pills, "Analysis view").set_value(name).run()


def _motor_view(view):
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = _view(app, view)
    app = _widget(app.radio, "Module organization").set_value("independent_regional").run()
    app = _widget(app.selectbox, "Regional partition source").set_value(R.MOTOR_SOURCE).run()
    return app


def test_motor_manifest_grain_privacy_transforms_and_isolated_metadata():
    manifest = R.load_regional_manifest()
    completed = R.completed_combinations(manifest)
    motor = completed.loc[completed.partition_source.eq(R.MOTOR_SOURCE)]
    assert len(motor) == 3 and motor["rows"].sum() == 100317
    assert set(motor.score_variant) == {"eigengene"}
    assert set(motor.cohort_scope) == {"common_357"}
    assert manifest["source_definitions"][R.MOTOR_SOURCE]["discovery_donors"] == 364
    details = R.load_regional_details()
    details = details.loc[details.partition_source.eq(R.MOTOR_SOURCE)]
    assert details.groupby("tissue").module.nunique().to_dict() == EXPECTED
    assert len(details) == 281 and details.n_genes.sum() == 34862
    assert details.module_key.is_unique and details.n_genes.min() >= 30
    metadata = R.regional_metadata(R.MOTOR_SOURCE, "common_357")
    assert len(metadata) == 357
    for tissue, count in EXPECTED.items():
        scores = R.load_regional_scores(R.MOTOR_SOURCE, "common_357", "eigengene", "not_applicable", tissue)
        assert len(scores) == 357 * count
        assert not scores.duplicated(["sample_id", "module"]).any()
        assert set(scores.sample_id) == set(metadata.sample_id)
        assert not {"donor", "projid"}.intersection(scores.columns)
        assert np.allclose(scores.metric_asinh, np.arcsinh(scores.metric_raw))
        for _, part in scores.groupby("module"):
            expected = norm.ppf((rankdata(part.metric_raw, method="average") - .5) / len(part))
            assert np.allclose(part.metric_rint, expected)
        joined = scores[["sample_id", "cogn_global"]].merge(metadata[["sample_id", "cogn_global"]], on="sample_id")
        assert np.allclose(joined.cogn_global_x, joined.cogn_global_y, equal_nan=True)
    kegg = R.load_regional_kegg()
    kegg = kegg.loc[kegg.partition_source.eq(R.MOTOR_SOURCE)]
    assert not kegg.empty
    assert not kegg.overlap_genes.str.contains("ENSG", na=False).any()
    assert set(kegg.tissue) == set(EXPECTED)


def test_motor_grouped_fdr_and_module_comparison_are_tissue_specific():
    catalog = R.regional_correlation_catalog(
        R.MOTOR_SOURCE, "common_357", "eigengene", "not_applicable", "VH", "raw",
        ("Control", "MCI", "AD"), "diagnosis_group", ("Control", "MCI", "AD"), 10,
        ("cogn_global",), True,
    )
    assert len(catalog) == 56 * 4
    assert set(catalog.spearman_fdr_module_family_n) == {56}
    spec = C.ComparisonSeriesSpec(R.MOTOR_SOURCE, feature="eigengene", method="not_applicable",
                                 component="regional", cohort_scope="common_357", tissues=R.MOTOR_TISSUE_ORDER)
    result = C.calculate_series(spec, "cogn_global", "all_donors", 10, "motor-qa")
    assert len(result) == 281 and result.module_key.is_unique
    for tissue, group in result.groupby("tissue"):
        assert set(group.spearman_fdr_module_family_n) == {EXPECTED[tissue]}
    donor_n = int(pd.to_numeric(R.regional_metadata(R.MOTOR_SOURCE, "common_357").cogn_global).notna().sum())
    assert set(result.n) == {donor_n}
    row = result.iloc[0]
    scores = R.load_regional_scores(R.MOTOR_SOURCE, "common_357", "eigengene", "not_applicable", row.tissue, int(row.module))
    good = scores[["metric_raw", "cogn_global"]].dropna()
    assert row.spearman_rho == pytest.approx(spearmanr(good.metric_raw, good.cogn_global).statistic)
    with pytest.raises(ValueError, match="Unknown regional tissue"):
        C.ComparisonSeriesSpec(R.MOTOR_SOURCE, tissues=("AC",))
    with pytest.raises(ValueError, match="No donor metadata"):
        R.regional_metadata(R.MOTOR_SOURCE, "common_450")


@pytest.mark.parametrize("view", ["Associations", "Feature distributions", "Correlation heatmaps"])
def test_motor_regional_views_render_and_can_switch_back_to_brain(view):
    app = _motor_view(view)
    assert not app.exception
    assert _widget(app.selectbox, "Regional donor cohort").value == "common_357"
    assert _widget(app.multiselect, "Regions shown side by side").value == list(R.MOTOR_TISSUE_ORDER)
    assert {box.label for box in app.selectbox}.issuperset({"VH module", "SMA module", "QUAD module"})
    assert len(app.get("plotly_chart")) == 1
    app = _widget(app.selectbox, "Regional partition source").set_value("single_region_complete_case_l3").run()
    assert not app.exception
    assert _widget(app.selectbox, "Regional donor cohort").value == "common_450"
    assert _widget(app.multiselect, "Regions shown side by side").value == list(R.REGION_ORDER)


def test_motor_source_is_selectable_in_module_outcome_comparison():
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = _view(app, "Module–outcome comparison")
    app = _widget(app.selectbox, "Partition / score source").set_value(R.MOTOR_SOURCE).run()
    assert not app.exception
    app = _widget(app.checkbox, "All regions in one box").check().run()
    assert not app.exception
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception
    summary = app.dataframe[0].value
    motor = summary.loc[summary.source.eq(R.MOTOR_SOURCE)]
    assert len(motor) == 1 and motor.total_modules.iloc[0] == 281
    assert motor.cohort_label.iloc[0] == "Common 357 motor-tissue donors"
    assert len(app.get("plotly_chart")) == 1
    app = _widget(app.selectbox, "Score component").set_value(C.COMBINED_COMPONENT).run()
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception
    combined = app.dataframe[0].value.loc[lambda table: table.component.eq(C.COMBINED_COMPONENT)]
    assert len(combined) == 1 and combined.total_modules.iloc[0] == 186
    assert combined.total_score_observations.iloc[0] == 372
    assert combined.plotted_score_observations.iloc[0] > combined.plotted_modules.iloc[0]
