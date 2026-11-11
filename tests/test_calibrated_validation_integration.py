"""Local-only publication boundary, immutable reads and actual app exports."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app_helpers.control_calibrated import validation_data as V, adapter as A
from app_helpers.control_calibrated.inputs import sha
from app_helpers.control_calibrated.registry import RESEARCH_ROOT


def test_reader_rejects_hosted_context_and_modified_files(monkeypatch, tmp_path):
    identity = "a" * 64
    root = tmp_path / identity
    root.mkdir()
    frame = root / "results.parquet"
    pd.DataFrame({"effect": [0.25]}).to_parquet(frame)
    (root / "complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                analysis_id=identity,
                files={"results.parquet": sha(frame)},
            )
        )
    )
    (tmp_path / "current_local.json").write_text(json.dumps(dict(analysis_id=identity)))
    monkeypatch.setattr(V, "BASE", tmp_path)
    monkeypatch.setattr(V.D, "DATA_DIR", V.D.LOCAL_DATA_DIR.resolve())
    assert V.local_root() == root and V.verified(root, "results.parquet") == frame
    pd.DataFrame({"effect": [0.5]}).to_parquet(frame)
    with pytest.raises(ValueError, match="checksum"):
        V.verified(root, "results.parquet")
    monkeypatch.setattr(V.D, "DATA_DIR", tmp_path / "hosted-cache")
    assert V.local_root() is None
    assert V.baseline_path(RESEARCH_ROOT, "brain", "complete_450") is None
    with pytest.raises(FileNotFoundError):
        V.baseline_path(RESEARCH_ROOT, "brain", "complete_450", "repaired")


@pytest.mark.skipif(
    V.local_root() is None, reason="Completed local validation bundle required"
)
def test_actual_completed_catalog_exports_and_baseline_audit():
    spec = V.read_json("analysis_spec.json")
    assert sum(v["modules"] for v in spec["families"].values()) == 1697
    for family in spec["families"]:
        f = V.results(
            family, 3, models=["B", "C"], strata=["all"], outcome="cogn_global"
        )
        assert not f.empty and f.family.eq(family).all()
        assert not {"sample_id", "donor_id", "IID", "projid"}.intersection(f.columns)
        pivot = f.pivot(
            index=["cohort", "module", "component", "metric_family"],
            columns="model",
            values="donor_set_hash",
        )
        assert pivot.B.equals(pivot.C)
        assert f.analysis_id.eq(spec["analysis_id"]).all()
        assert {
            "design_rank",
            "residual_df",
            "p_method",
            "permutation_seed",
            "q_scan",
            "q_GL_outcome",
            "correction_scope",
            "q_across_modules",
            "module_family_n",
        } <= set(f.columns)
        from scipy.stats import false_discovery_control
        from app_helpers.control_calibrated.association_fdr import FAMILY_COLUMNS
        for _, group in f.groupby([c for c in FAMILY_COLUMNS if c in f], dropna=False, sort=False):
            valid = group.p.notna()
            assert group.module_family_n.eq(valid.sum()).all()
            if valid.any():
                np.testing.assert_allclose(group.loc[valid, "q_across_modules"], false_discovery_control(group.loc[valid, "p"]))
    old = pd.read_parquet(
        RESEARCH_ROOT / "associations/brain_incomplete_region_validation.parquet"
    )
    new = V.read_table("repaired_baseline/brain_incomplete_region_validation.parquet")
    bad = old.n.eq(4) & old.p.eq(0) & old.effect.abs().gt(0.999999)
    assert bad.sum() == 98
    keys = ["module_key", "component", "beta", "metric_family", "outcome"]
    repaired = old.loc[bad, keys].merge(new, on=keys, validate="one_to_one")
    np.testing.assert_allclose(repaired.p, 1 / 12)
    filtered = A.baseline_associations(
        "control_derived",
        "calibrated_beta3",
        "incomplete_region_validation",
        inference_version="repaired",
    )
    assert filtered.family.eq("control_derived").all() and filtered.beta.eq(3).all()


@pytest.mark.skipif(
    V.local_root() is None, reason="Completed local validation bundle required"
)
def test_new_validation_app_panels_and_plot_metadata():
    from streamlit.testing.v1 import AppTest
    from app_helpers.control_calibrated.registry import APP

    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180)
    app.query_params.update(
        view="Correlation heatmaps",
        screen_partition="control_derived",
        score_source="calibrated_beta3",
    )
    app = app.run()
    app = (
        app.selectbox(key="cc_validation_mode")
        .set_value("Covariate validation / 8-outcome scan")
        .run()
    )
    assert not app.exception
    assert app.radio(key="ccv_panel").value == "Covariate triplets"
    table = app.dataframe[-1].value
    assert set(table.model) == {"A", "B", "C"}
    assert set(table.cohort) == {"original", "incomplete"}
    assert table.beta.eq(3).all() and "donor_set_hash" in table
    for panel in [
        "ROS / MAP",
        "Composition / missingness",
        "Baseline inference repair",
        "Methods / provenance",
        "Focused comparisons",
    ]:
        app = app.radio(key="ccv_panel").set_value(panel).run()
        assert not app.exception, panel
    for kind in [
        "Score-by-study interaction",
        "Shared-donor Connectivity / eigengenes",
        "Joint G/L beyond comparators",
        "Triplet / age spline",
    ]:
        app = app.selectbox(key="ccv_focused_kind").set_value(kind).run()
        assert not app.exception, kind
        frame = app.dataframe[-1].value
        assert (
            frame.analysis_id.nunique() == 1
            and "ci_low" in frame
            and "donor_set_hash" in frame
        )
        chart = app.get("plotly_chart")[0]
        figure = json.loads(chart.proto.spec)
        assert figure["layout"]["meta"]["beta"] == 3
        assert figure["layout"]["meta"]["mask"] == "all"
        assert "correction_scope" in figure["layout"]["meta"]
    app = app.radio(key="ccv_panel").set_value("Covariate triplets").run()
    app = app.multiselect(key="ccv_modules").set_value([747]).run()
    app = app.checkbox(key="ccv_residual_diagnostic").check().run()
    assert not app.exception
    meta = json.loads(app.get("plotly_chart")[-1].proto.spec)["layout"]["meta"]
    assert meta["plot_units"] == "residualized midranks; not native connectivity units"
    assert "donor_set_hash" in meta and meta["n"] > 30
