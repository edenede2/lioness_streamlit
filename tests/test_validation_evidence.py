"""Evidence gates: cohorts, physical edges, q scopes and rendered saved data."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest
from app_helpers import validation_evidence as E
from app_helpers.validation_evidence_ui import forest

APP = Path(__file__).resolve().parents[1]


def example_meta():
    return pd.DataFrame(
        dict(
            sample_id=["a", "b", "c", "d", "e"],
            in_original_analysis=[True, True, False, False, False],
            role=["reference", "target", "target", "target", "target"],
            has_AC=[True, True, True, False, False],
            has_DLPFC=[True] * 5,
            has_PCG=[True, True, True, False, True],
            ad_control_split=[
                "Validation",
                "Discovery",
                "Validation",
                "Validation",
                "Validation",
            ],
            study=["ROS", "MAP", "ROS", "MAP", "MAP"],
        )
    )


def test_missing_membership_is_not_zero_overlap():
    assert E.intersection_audit({"a", "b"}, {"b", "c"})["overlap_n"] == 1
    assert E.intersection_audit({"a"}, {"b"})["status"] == "verified disjoint"
    result = E.intersection_audit(None, {"b"})
    assert result["status"] == "independence not verified" and np.isnan(
        result["overlap_n"]
    )


def test_cohort_roles_exclude_other_complete_donors_and_absent_regions():
    m = example_meta()
    assert np.flatnonzero(
        E.cohort_selection(m, "TS_DLPFC", "original_complete")
    ).tolist() == [0, 1]
    assert np.flatnonzero(
        E.cohort_selection(m, "TS_DLPFC", "original_complete", independent=True)
    ).tolist() == [1]
    assert np.flatnonzero(E.cohort_selection(m, "TS_DLPFC", "incomplete")).tolist() == [
        3,
        4,
    ]
    assert np.flatnonzero(
        E.cohort_selection(m, "CT_DLPFC__PCGBA23", "incomplete")
    ).tolist() == [4]
    assert np.flatnonzero(
        E.cohort_selection(m, "TS_DLPFC", "legacy_heldout")
    ).tolist() == [0]
    with pytest.raises(ValueError):
        E.cohort_selection(m, "TS_unknown", "incomplete")


def test_actual_region_mapping_and_signed_powers_match_direct():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 5))
    x[:, 1] = -0.8 * x[:, 0] + rng.normal(size=30) * 0.1
    r, c = np.triu_indices(5, 1)
    for power in (2, 3):
        raw, adj = E.group_edges(x, r, c, power)
        direct = np.corrcoef(x, rowvar=False)[r, c]
        np.testing.assert_allclose(raw, direct, atol=1e-12)
        np.testing.assert_allclose(
            adj, np.sign(direct) * abs(direct) ** power, atol=1e-12
        )
        assert adj[0] < 0
    assert E.canonical_component("CT_DLPFC__PCGBA23") == "CT_DLPFC__PCG"


def test_densities_have_no_arbitrary_epsilon_or_zero_imputation():
    ad, control, ratio = E.density_ratio(np.array([2.0]), np.array([1e-12]))
    assert ratio == 2e12 and control == 1e-12
    assert np.isnan(E.density_ratio(np.array([1.0]), np.array([0.0]))[2])
    assert np.isnan(E.concordance([0, 0, 0], [1, 2, 3]))


def test_rectangular_hubs_are_separate_and_resampling_preserves_donor_strata():
    r = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    c = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
    values = np.arange(1, 10, dtype=float)
    result = E.hub_concordance(values, values * 2, r, c, ["DLPFC"] * 3 + ["PCG"] * 3)
    assert result["hub_DLPFC"] == pytest.approx(1) and result[
        "hub_PCG"
    ] == pytest.approx(1)
    meta = example_meta()
    a = E.resample_indices(meta, np.random.default_rng(42))
    b = E.resample_indices(meta, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert (
        meta.iloc[a].study.value_counts().to_dict()
        == meta.study.value_counts().to_dict()
    )
    assert E.interval([1, np.nan, np.nan], 3)[2] == 1 and np.isnan(
        E.interval([1, np.nan, np.nan], 3)[0]
    )


def test_mdc_denominators_and_noninferential_counts_are_explicit():
    f = pd.DataFrame(
        dict(
            original_mdc=[2.0, 2.0, 0.5, np.nan],
            validation_mdc=[1.5, 0.8, 0.7, 1.0],
            original_directional_fdr=[0.01, 0.01, 0.2, 0.01],
            validation_directional_fdr=[0.04, 0.03, np.nan, np.nan],
            inferential=[True, True, False, False],
            p_loss_sample=[0.01, 0.01, np.nan, np.nan],
            p_loss_gene=[0.02, 0.01, np.nan, np.nan],
            p_gain_sample=[0.01, 0.01, np.nan, np.nan],
            p_gain_gene=[0.01, 0.01, np.nan, np.nan],
        )
    )
    original = f.copy(deep=True)
    s = E.mdc_summary(f)
    assert s["planned"] == 4 and s["comparable"] == 3 and s["unavailable"] == 1
    assert s["original_significant"] == 3 and s["original_significant_evaluable"] == 2
    assert s["direction_reversed"] == 1 and s["original_signal_adjusted_support"] == 1
    pd.testing.assert_frame_equal(original, f)
    assert E.evidence_status(np.nan, 0.2) == "Unavailable / not tested"
    assert E.evidence_status(0.2, -0.1, 0.001, 0.001) == "Direction reversal"
    assert E.evidence_status(0.2, 0.1, 0.01, 0.2) == "Same direction + nominal support"
    assert E.evidence_status(0, 0.2) == "Zero direction / descriptive"


def test_artifact_version_and_hash_are_checked_without_fallback(tmp_path):
    assert E.manifest(tmp_path)["status"] == "unavailable"
    (tmp_path / "manifest.json").write_text(json.dumps(dict(version="other")))
    with pytest.raises(ValueError, match="version"):
        E.manifest(tmp_path)
    (tmp_path / "cohorts.csv").write_text("cohort,n\noriginal,10\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                version=E.VERSION,
                status="complete",
                files={"cohorts.csv": {"sha256": "wrong"}},
            )
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        E.table("cohorts", tmp_path)


def test_public_adjusted_report_q_values_survive_filtering_and_BC_match():
    full = E.report_table("focused_all_endpoints")
    selected = full.loc[
        full.family.eq("control_derived")
        & full.module.eq(747)
        & full.outcome.eq("gpath")
    ]
    source = pd.read_csv(E.REPORT / "focused_all_endpoints.csv")
    pd.testing.assert_frame_equal(selected, source.loc[selected.index])
    pairs = E.report_table("focused_selection_adjustment")
    assert pairs.B_C_same_membership.all()
    assert pairs.hash_B.eq(pairs.hash_C).all()


def test_interval_artist_does_not_clip_or_recenter_bootstrap_endpoints():
    frame = pd.DataFrame(
        dict(
            effect=[0.7],
            ci_low=[0.3],
            ci_high=[0.6],
            row_label=["fixed"],
            cohort=["Controls"],
        )
    )
    figure = forest(frame)
    assert list(figure.data[0].x) == [0.3, 0.6, None]
    assert list(figure.data[1].x) == [0.7]
    assert list(figure.layout.yaxis.tickvals) == ["fixed"]
    assert figure.layout.xaxis.range[0] < 0.3
    assert figure.layout.xaxis.range[1] > 0.7


def test_meeting_artifacts_require_verified_safe_paths(tmp_path):
    (tmp_path / "RESULTS.md").write_text("Aggregate scientific note")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                version=E.VERSION,
                artifacts={"RESULTS.md": {"sha256": E.sha(tmp_path / "RESULTS.md")}},
            )
        )
    )
    assert E.public_artifact("RESULTS.md", tmp_path) == b"Aggregate scientific note"
    with pytest.raises(ValueError):
        E.public_artifact("../private/edges.parquet", tmp_path)
    (tmp_path / "RESULTS.md").write_text("Changed")
    with pytest.raises(ValueError, match="checksum"):
        E.public_artifact("RESULTS.md", tmp_path)


def test_candidate_ranking_is_original_only_and_has_deterministic_ties(monkeypatch):
    monkeypatch.syspath_prepend(str(APP / "scripts"))
    monkeypatch.syspath_prepend(str(APP.parents[1]))
    from validation_evidence_edges import choose_candidates

    frame = pd.DataFrame(
        dict(
            edge_code=[9, 3, 4],
            original_g=[0.4, -0.4, 0.1],
            original_cogn_global_effect=[0.2, 0.2, 0.1],
            original_gpath_effect=[0.1, 0.1, 0.05],
        )
    )
    first = choose_candidates(frame, 1)
    frame["validation_effect"] = [100.0, -100.0, 10.0]
    pd.testing.assert_frame_equal(
        first, choose_candidates(frame, 1).drop(columns="validation_effect")
    )
    assert first.edge_code.tolist() == [3]


def test_discovery_header_orientation_or_absence_never_becomes_zero_overlap(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(APP / "scripts"))
    from build_validation_evidence import discovery_membership

    mapping = {"ROSMAP-00000001": "S1", "ROSMAP-00000002": "S2"}
    column_oriented = tmp_path / "controls.tsv"
    column_oriented.write_text("module,ROSMAP-00000001,ROSMAP-00000002\nM1,0.1,0.2\n")
    row_oriented = tmp_path / "full.tsv"
    row_oriented.write_text("donor\tM1\nROSMAP-00000001\t0.1\nROSMAP-00000002\t0.2\n")
    assert discovery_membership(column_oriented, mapping) == {"S1", "S2"}
    assert discovery_membership(row_oriented, mapping) == {"S1", "S2"}
    missing = tmp_path / "absent.tsv"
    assert discovery_membership(missing, mapping) is None
    missing.write_text("unknown\n")
    assert discovery_membership(missing, mapping) is None


def test_resume_rejects_changed_source_before_loading_networks(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(APP / "scripts"))
    import build_validation_evidence as builder

    monkeypatch.setattr(E, "PUBLIC", tmp_path)
    monkeypatch.setattr(builder, "APP", tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("Original source")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                version=E.VERSION,
                input_sha256={"source.csv": E.sha(source)},
            )
        )
    )
    source.write_text("Changed source metadata")
    with pytest.raises(ValueError, match="Frozen evidence source changed"):
        builder.inputs()


def test_public_artifacts_have_no_identifiers_or_private_edge_arrays():
    m = E.manifest()
    for name, item in m.get("files", {}).items():
        f = pd.read_csv(E.PUBLIC / name)
        assert len(f) == item["rows"] and E.sha(E.PUBLIC / name) == item["sha256"]
        assert not {
            "donor",
            "projid",
            "sample_id",
            "gene_a",
            "gene_b",
            "edge_code",
        }.intersection(f)
        assert (E.PUBLIC / name).stat().st_size < 95 * 1024**2
    structure = E.table("structure")
    assert (
        len(structure) == 8
        and structure[["partition", "module", "component"]].duplicated().sum() == 0
    )
    assert structure.n_edges.gt(0).all()
    assert set(structure.original_control_n) == {164} and set(
        structure.validation_control_n
    ) == {229, 65}


def test_private_selected_family_q_covers_full_frozen_union():
    frame = E.private_table("same_edges")
    if frame.empty:
        pytest.skip("Local-only detailed evidence not installed")
    from app_helpers.correlations import benjamini_hochberg

    assert (
        frame[["partition", "module", "component", "edge_code", "test", "outcome"]]
        .duplicated()
        .sum()
        == 0
    )
    for _, g in frame.groupby(["test", "outcome"], dropna=False):
        np.testing.assert_allclose(
            g.selected_family_q, benjamini_hochberg(g.p), atol=1e-12
        )
        assert g.selected_family_n.eq(g.p.notna().sum()).all()
    candidates = E.private_table("edge_candidates")
    assert len(candidates) == 461
    assert not candidates.columns.str.startswith("validation_").any()


def widget(app, kind, label):
    return next(value for value in getattr(app, kind) if value.label == label)


def app():
    instance = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180)
    instance.query_params["view"] = "Validation & replication"
    return instance.run()


def test_new_overview_is_aggregate_only_and_existing_views_are_linked(monkeypatch):
    from app_helpers import camel_validation as V

    def forbidden(*args, **kwargs):
        raise AssertionError("No donor read or reconstruction in overview")

    for name in ("metadata", "scores", "summaries"):
        monkeypatch.setattr(V, name, forbidden)
    a = app()
    assert not a.exception
    assert widget(a, "radio", "Validation section").value == "Evidence overview"
    assert len(a.get("plotly_chart")) >= 1
    assert widget(a, "button", "Open legacy MDC replication")
    a = widget(a, "button", "Open legacy MDC replication").click().run()
    assert (
        not a.exception
        and widget(a, "radio", "Validation section").value == "MDC replication"
    )


@pytest.mark.parametrize(
    "section",
    [
        "Control structure",
        "MDC decomposition",
        "Same-edge follow-up",
        "Adjusted evidence (A/B/C)",
    ],
)
def test_focused_sections_render_saved_tables_and_exporters(section):
    a = app()
    a = widget(a, "selectbox", "Evidence view").set_value(section).run()
    assert not a.exception, [(e.type, e.message) for e in a.exception]
    assert a.dataframe and a.get("plotly_chart")
    assert widget(a, "button", "Export figure")
    if section == "Control structure" and len(a.get("plotly_chart")) > 1:
        figure = json.loads(a.get("plotly_chart")[-1].proto.spec)
        heatmaps = [t for t in figure["data"] if t["type"] == "heatmap"]
        assert len(heatmaps) == 2
        assert heatmaps[0]["coloraxis"] == heatmaps[1]["coloraxis"] == "coloraxis"
        assert figure["layout"]["coloraxis"]["cmin"] == 0
        assert figure["layout"]["coloraxis"]["cmax"] > 0
        # Both matrices use the exact same implicit frozen node-axis order.
        assert heatmaps[0].get("x") == heatmaps[1].get("x")
        assert heatmaps[0].get("y") == heatmaps[1].get("y")
        assert figure["data"][-1]["mode"] == "lines"
        assert figure["data"][-1]["x"] == figure["data"][-1]["y"]


def test_pending_evidence_module_does_not_relabel_saved_figure():
    a = app()
    a = widget(a, "selectbox", "Evidence view").set_value("MDC decomposition").run()
    before = a.get("plotly_chart")[0].proto.spec
    a = widget(a, "selectbox", "Evidence module").set_value(1116).run()
    assert not a.exception and a.get("plotly_chart")[0].proto.spec == before
    a = widget(a, "button", "Apply analysis").click().run()
    assert not a.exception and a.get("plotly_chart")[0].proto.spec != before


def test_hosted_empty_private_state_still_shows_aggregate_followup(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(E, "PRIVATE", tmp_path)
    a = app()
    a = widget(a, "selectbox", "Evidence view").set_value("Same-edge follow-up").run()
    assert not a.exception and a.dataframe
    assert any("local-only" in str(v.value) for v in a.info)
    assert not a.get("plotly_chart")


def test_missing_adjusted_report_is_explicit_without_substitute(monkeypatch):
    monkeypatch.setattr(E, "report_table", lambda *args, **kwargs: pd.DataFrame())
    a = app()
    a = (
        widget(a, "selectbox", "Evidence view")
        .set_value("Adjusted evidence (A/B/C)")
        .run()
    )
    assert not a.exception
    assert any("No legacy bundle" in str(v.value) for v in a.info)
    assert not a.get("plotly_chart")
