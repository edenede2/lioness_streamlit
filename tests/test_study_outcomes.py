"""Nominal study membership: privacy, source joins, statistics and app routing."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal
from streamlit.testing.v1 import AppTest

from app_helpers import data as D, regional_modules as R, score_comparison as C
from app_helpers.correlations import benjamini_hochberg
from app_helpers.study import normalize_study, with_study_metadata

APP = Path(__file__).resolve().parents[1]


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_study_is_nominal_group_color_hover_but_never_numeric_or_prediction():
    assert "study" in D.ASSOCIATION_OUTCOME_LABELS
    assert "study" in D.ASSOCIATION_GROUP_LABELS
    assert "study" in D.COLOR_LABELS and "study" in D.HOVER_LABELS
    assert "study" in D.CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
    assert D.ASSOCIATION_LEVEL_ORDERS["study"] == ["ROS", "MAP"]
    assert "study" not in D.NUMERIC_OUTCOMES and "study" not in D.PREDICTION_OUTCOME_LABELS
    assert normalize_study(pd.Series(["ROS ", " map", None, ""])).tolist() == ["ROS", "MAP", pd.NA, pd.NA]
    with pytest.raises(ValueError, match="Unexpected study"):
        normalize_study(pd.Series(["unverified code"]))
    old = pd.DataFrame({"sample_id": ["S1", "S2"]})
    assert with_study_metadata(old).study.isna().all()
    assert "study" not in old  # No inferred study or source mutation.


def test_repository_overlay_supports_old_drive_metadata_without_mixing_namespaces(tmp_path, monkeypatch):
    from app_helpers import study
    path = tmp_path / "study.parquet"
    pd.DataFrame({"metadata_namespace":["brain","motor"],"sample_id":["S1","S1"],
                  "study":["ROS","MAP"]}).to_parquet(path,index=False)
    monkeypatch.setattr(study,"STUDY_CACHE",path)
    old = pd.DataFrame({"sample_id":["S1","unknown"]})
    assert study.with_study_metadata(old,"brain").study.iloc[0] == "ROS"
    assert study.with_study_metadata(old,"motor").study.iloc[0] == "MAP"
    assert pd.isna(study.with_study_metadata(old,"brain").study.iloc[1])
    with pytest.raises(ValueError,match="disagree"):
        study.with_study_metadata(pd.DataFrame({"sample_id":["S1"],"study":["MAP"]}),"brain")


def test_approved_metadata_matches_manifest_and_preserves_every_old_value():
    path = APP / "data/study/manifest.json"
    if not path.exists():
        pytest.skip("Study bundle not installed; no Drive requests")
    manifest = json.loads(path.read_text())
    expected = {"sample_metadata.parquet": {"ROS":209,"MAP":241},
                "muscle_control_l4/sample_metadata.parquet":{"MAP":357},
                "muscle_all_donors_l4/sample_metadata.parquet":{"MAP":357}}
    for relative, record in manifest["metadata"].items():
        file = APP / "data" / relative
        frame = pd.read_parquet(file)
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"]
        assert len(frame) == record["rows"] and frame.sample_id.is_unique
        assert not {"donor", "projid"}.intersection(frame)
        assert set(frame.study.dropna()).issubset({"ROS","MAP"})
        if relative in expected:
            assert frame.study.value_counts().to_dict() == expected[relative]
        backup = APP.parents[1] / "out/study_metadata_app_20261021/rollback" / relative
        if backup.exists():
            old = pd.read_parquet(backup)
            columns = [c for c in old if c != "study"]
            pd.testing.assert_frame_equal(frame[columns], old[columns])


def test_study_module_comparison_uses_kruskal_not_numeric_study_codes():
    rng = np.random.default_rng(42)
    metadata = pd.DataFrame({"sample_id":[f"S{i}" for i in range(40)],
                             "study":["ROS"]*20+["MAP"]*20})
    frames = [(module, "", pd.DataFrame({"sample_id":metadata.sample_id,
                "metric_value":rng.normal(size=40)+np.r_[np.zeros(20),np.ones(20)*module/4]}))
              for module in (1,2,3)]
    spec = C.ComparisonSeriesSpec("control_derived")
    result = C.statistics_from_module_frames(spec, iter(frames), {"":metadata}, "study",10)
    for row, (_,_,values) in zip(result.itertuples(), frames):
        direct = kruskal(values.metric_value.iloc[:20], values.metric_value.iloc[20:])
        assert row.categorical_p == pytest.approx(direct.pvalue)
        assert row.epsilon_squared == pytest.approx(max(0,(direct.statistic-1)/38))
    assert result.pearson_r.isna().all() and result.spearman_rho.isna().all()
    assert result.categorical_fdr_module_family_n.eq(3).all()
    np.testing.assert_allclose(result.categorical_fdr_across_modules,benjamini_hochberg(result.categorical_p))
    shown, _, summary = C.display_results({spec.key:result},[("a","Study effects",spec)],top_k=2)
    assert shown.correlation_method.eq("not_applicable").all()
    assert shown.absolute_correlation.isna().all() and shown.association_effect.notna().all()
    figure = C.comparison_figure(shown,summary,"study","spearman")
    assert "epsilon-squared" in figure.layout.yaxis.title.text
    assert "correlation:" not in figure.data[1].hovertemplate
    one_study = metadata.assign(study="MAP")
    unavailable = C.statistics_from_module_frames(spec,iter(frames),{"":one_study},"study",10)
    assert not unavailable.eligible.any()
    assert unavailable.categorical_p.isna().all()
    assert unavailable.categorical_fdr_module_family_n.eq(0).all()
    assert unavailable.unavailable_reason.str.contains("two eligible groups").all()


def test_study_metadata_routes_brain_motor_and_maximum_tissue_separately():
    brain = D.load_sample_metadata()
    motor = R.regional_metadata(R.MOTOR_SOURCE,"common_357")
    assert brain.study.nunique() == 2 and set(motor.study) == {"MAP"}
    for tissue in ("AC","DLPFC","PCG"):
        frame = R.regional_metadata("single_region_complete_case_l3","maximum_tissue",tissue)
        assert frame.sample_id.is_unique and frame.study.nunique() == 2
        assert frame.study.isna().sum() == {"AC":11,"DLPFC":13,"PCG":8}[tissue]


def test_real_association_view_supports_study_outcome_and_grouping():
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.selectbox,"Association outcome").set_value("study").run()
    assert not app.exception
    assert any("ROS/MAP is nominal" in item.value for item in app.caption)
    table = next(e.value for e in app.dataframe if "category_variable" in e.value)
    assert table.category_variable.eq("study").all()
    assert not {"pearson_r","spearman_rho"}.intersection(table)
    app = widget(app.selectbox,"Association outcome").set_value("cogn_global").run()
    app = widget(app.selectbox,"Group correlations by").set_value("study").run()
    assert not app.exception
    assert widget(app.multiselect,"Group levels").value == ["ROS","MAP"]
    table = next(e.value for e in app.dataframe if "grouping_variable" in e.value)
    assert table.grouping_variable.eq("study").all()
    assert set(table.loc[~table.is_pooled,"grouping_level"]) == {"ROS","MAP"}


def test_real_study_screen_finder_statistics_and_heatmap_routes():
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    for view in ("CT–TS screen","Module finder","Statistics"):
        app = widget(app.pills,"Analysis view").set_value(view).run()
        app = widget(app.selectbox,"Association outcome").set_value("study").run()
        assert not app.exception, view
        table = next(e.value for e in app.dataframe if any("epsilon_squared" in str(c) for c in e.value))
        assert len(table) > 0
    app = widget(app.pills,"Analysis view").set_value("Correlation heatmaps").run()
    app = widget(app.radio,"Association type").set_value("Study / project (ROS / MAP)").run()
    assert not app.exception
    assert widget(app.radio,"Study heatmap scope")
    assert app.get("plotly_chart")


def test_motor_single_study_screen_is_explicitly_unavailable():
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.selectbox,"Module definition").set_value("muscle_control_l4").run()
    app = widget(app.pills,"Analysis view").set_value("CT–TS screen").run()
    app = widget(app.selectbox,"Association outcome").set_value("study").run()
    assert not app.exception
    assert any("fewer than two observed studies" in e.value for e in app.info)


def test_study_distribution_and_numeric_heatmap_group_filters():
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.pills, "Analysis view").set_value("Feature distributions").run()
    app = widget(app.selectbox, "Group distributions by").set_value("study").run()
    assert not app.exception
    assert widget(app.multiselect, "Distribution group levels").value == ["ROS", "MAP"]
    summary = next(e.value for e in app.dataframe
                   if {"grouping_variable", "distribution_group"}.issubset(e.value))
    assert summary.grouping_variable.eq("study").all()
    assert set(summary.distribution_group) == {"ROS", "MAP"}

    app = widget(app.pills, "Analysis view").set_value("Correlation heatmaps").run()
    app = widget(app.selectbox, "Group heatmap correlations by").set_value("study").run()
    app = widget(app.multiselect, "Heatmap group levels").set_value(["ROS"]).run()
    assert not app.exception
    table = next(e.value for e in app.dataframe
                 if {"grouping_variable", "grouping_label", "correlation_method"}.issubset(e.value))
    assert table.grouping_variable.eq("study").all()
    assert set(table.grouping_label) == {"ROS"}


def test_regional_study_catalog_and_grouped_correlations_use_tissue_families():
    stats = R.regional_categorical_catalog("single_region_complete_case_l3","common_450", "eigengene",
        "not_applicable","AC","raw",("Control","MCI","AD"),"study",("ROS","MAP"),10)
    assert stats.outcome.eq("study").all() and stats.module.nunique() == 142
    assert stats.categorical_fdr_module_family_n.max() <= 142
    grouped = R.regional_correlation_catalog("single_region_complete_case_l3","common_450","eigengene",
        "not_applicable","AC","raw",("Control","MCI","AD"),"study",("ROS","MAP"),10,("cogn_global",),False)
    assert set(grouped.grouping_level) == {"ROS","MAP"}
    assert grouped.spearman_fdr_module_family_n.max() <= 142
