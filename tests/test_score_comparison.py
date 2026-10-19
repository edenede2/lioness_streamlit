from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from streamlit.testing.v1 import AppTest

from app_helpers import score_comparison as C
from app_helpers import score_comparison_ui as U
from app_helpers.correlations import benjamini_hochberg


def _fixture(spec, tissues=("",), n=30):
    rng = np.random.default_rng(42)
    metadata = pd.DataFrame({"sample_id": [f"S-{i}" for i in range(n)], "cogn_global": rng.normal(size=n)})
    y = metadata["cogn_global"].to_numpy()
    frames = []
    for tissue in tissues:
        for module in (1, 2, 3):
            x = y * module / 3 + rng.normal(size=n)
            frames.append((module, tissue, pd.DataFrame({"sample_id": metadata["sample_id"], "metric_value": x})))
        frames.append((4, tissue, pd.DataFrame({"sample_id": metadata["sample_id"], "metric_value": np.ones(n)})))
        frames.append((5, tissue, pd.DataFrame()))
        frames.append((6, tissue, pd.DataFrame({"sample_id": metadata["sample_id"].head(4), "metric_value": np.arange(4)})))
    result = C.statistics_from_module_frames(spec, iter(frames), {t: metadata for t in tissues}, "cogn_global", 10)
    return result, metadata, frames


def _combined_fixture(spec):
    single = dataclasses.replace(spec, component="CT")
    _, metadata, frames = _fixture(single)
    combined_frames = []
    for module, _, frame in frames:
        combined_frames.append((module, "CT", frame))
        ts = frame.copy()
        if not ts.empty:
            ts["metric_value"] = np.roll(ts["metric_value"], module)
        combined_frames.append((module, "TS", ts))
    result = C.statistics_from_module_frames(
        spec, iter(combined_frames), {"": metadata}, "cogn_global", 10,
    )
    return result, metadata, combined_frames


def test_combined_ct_ts_retains_exact_component_correlations_and_separate_fdr():
    spec = C.ComparisonSeriesSpec("control_derived", component=C.COMBINED_COMPONENT)
    result, metadata, frames = _combined_fixture(spec)
    assert len(result) == 12 and result["module_key"].nunique() == 6
    assert result["observation_key"].is_unique
    assert set(result["tissue"]) == {""}
    for component, part in result.groupby("component", sort=False):
        single = dataclasses.replace(spec, component=component)
        expected = C.statistics_from_module_frames(
            single, ((m, "", f) for m, c, f in frames if c == component),
            {"": metadata}, "cogn_global", 10,
        )
        for method in ("pearson", "spearman"):
            for suffix in ("p", "fdr_across_modules", "fdr_module_family_n"):
                assert np.allclose(part[f"{method}_{suffix}"], expected[f"{method}_{suffix}"], equal_nan=True)
            assert set(part[f"{method}_fdr_module_family_n"]) == {3}
        assert np.allclose(part["pearson_r"], expected["pearson_r"], equal_nan=True)
        assert np.allclose(part["spearman_rho"], expected["spearman_rho"], equal_nan=True)


def test_combined_top_k_counts_unique_modules_and_fdr_filters_individual_scores():
    spec = C.ComparisonSeriesSpec("control_derived", component=C.COMBINED_COMPONENT)
    result, _, _ = _combined_fixture(spec)
    result.loc[result["eligible"], "spearman_rho"] = [.9, -.1, .8, -.2, .7, -.3]
    result.loc[result["eligible"], "spearman_fdr_across_modules"] = .01
    boxes = [("both", "CT + TS", spec)]
    shown, full, summary = C.display_results({spec.key: result}, boxes, top_k=2)
    assert set(shown["module"]) == {1, 2}
    assert len(shown) == 4 and shown["observation_key"].is_unique
    assert summary.iloc[0]["plotted_modules"] == 2
    assert summary.iloc[0]["plotted_score_observations"] == 4
    assert summary.iloc[0]["total_modules"] == 6
    assert summary.iloc[0]["total_score_observations"] == 12
    # One component may fail a display filter without hiding the module's other score.
    result.loc[result["module"].eq(1) & result["component"].eq("TS"), "spearman_fdr_across_modules"] = .3
    shown, full, summary = C.display_results({spec.key: result}, boxes, top_k=1, fdr_cutoff=.05)
    assert list(shown[["module", "component"]].itertuples(index=False, name=None)) == [(1, "CT")]
    assert full.loc[full["displayed"], "observation_key"].tolist() == shown["observation_key"].tolist()
    larger, _, summary = C.display_results({spec.key: result}, boxes, top_k=999)
    assert len(larger) == 6 and summary.iloc[0]["plotted_modules"] == 3
    figure = C.comparison_figure(larger, summary, "cogn_global", "spearman")
    assert "3 modules" in figure.layout.xaxis.ticktext[0]
    assert "6 CT/TS scores" in figure.layout.xaxis.ticktext[0]
    points = next(trace for trace in figure.data if trace.type == "scatter")
    assert len(points.y) == 6
    assert set(points.customdata[:, 2]) == {C.D.EDGE_SCOPE_LABELS["CT"], C.D.EDGE_SCOPE_LABELS["TS"]}
    assert len(set(points.x)) == 6


@pytest.mark.parametrize("source", C.R.REGIONAL_SOURCE_ORDER)
def test_all_regional_tissues_keep_distinct_modules_and_tissue_fdr(source):
    tissues = C.R.regional_tissues(source)
    spec = C.ComparisonSeriesSpec(source, feature="eigengene", method="not_applicable", component="regional", tissues=tissues)
    result, _, _ = _fixture(spec, tissues)
    shown, _, summary = C.display_results({spec.key: result}, [("regions", "All regional eigengenes", spec)])
    assert len(shown) == 9 and shown["module_key"].is_unique
    assert set(shown["tissue"]) == set(tissues)
    assert set(shown["spearman_fdr_module_family_n"]) == {3}
    assert summary.iloc[0]["plotted_modules"] == summary.iloc[0]["plotted_score_observations"] == 9


def test_direct_correlations_full_family_and_invalid_reasons():
    spec = C.ComparisonSeriesSpec("control_derived")
    result, metadata, frames = _fixture(spec)
    first = result.iloc[0]
    x = frames[0][2]["metric_value"]
    y = metadata["cogn_global"]
    assert first["pearson_r"] == pytest.approx(pearsonr(x, y).statistic)
    assert first["spearman_rho"] == pytest.approx(spearmanr(x, y).statistic)
    for method in ("pearson", "spearman"):
        assert np.allclose(result[f"{method}_fdr_across_modules"], benjamini_hochberg(result[f"{method}_p"]), equal_nan=True)
        assert set(result[f"{method}_fdr_module_family_n"]) == {3}
    assert result["eligible"].sum() == 3
    assert "constant" in result.loc[result["module"].eq(4), "unavailable_reason"].iloc[0]
    assert "structurally" in result.loc[result["module"].eq(5), "unavailable_reason"].iloc[0]
    assert result.loc[result["module"].eq(6), "n"].iloc[0] == 4
    assert not {"sample_id", "donor", "projid"}.intersection(result.columns)


def test_regional_namespaces_and_fdr_families_do_not_mix_tissues():
    spec = C.ComparisonSeriesSpec("single_region_complete_case_l3", feature="eigengene", method="not_applicable", component="regional", cohort_scope="common_450", tissues=("AC", "DLPFC"))
    result, _, _ = _fixture(spec, spec.tissues)
    assert result["module_key"].is_unique
    assert len(result) == 12
    for _, part in result.groupby("tissue"):
        assert set(part["spearman_fdr_module_family_n"]) == {3}
        assert np.allclose(part["spearman_fdr_across_modules"], benjamini_hochberg(part["spearman_p"]), equal_nan=True)
    assert "DLPFC" in result.loc[result["tissue"].eq("DLPFC"), "module_key"].iloc[0]


def test_top_k_fdr_precedes_selection_and_ties_are_deterministic():
    spec = C.ComparisonSeriesSpec("control_derived")
    result, _, _ = _fixture(spec)
    # Deliberate ties with opposite directions; namespaced identity is the tie breaker.
    result.loc[result["module"].isin([1, 2, 3]), "spearman_rho"] = [.5, -.5, .8]
    result.loc[result["module"].isin([1, 2, 3]), "spearman_fdr_across_modules"] = [.01, .03, .20]
    shown, full, summary = C.display_results({spec.key: result}, [("a", "Box A", spec)], "spearman", 1, .05)
    assert shown["module"].tolist() == [1]
    assert shown["absolute_correlation"].tolist() == [.5]
    assert shown["spearman_fdr_module_family_n"].iloc[0] == 3
    assert summary["eligible_modules"].iloc[0] == 3
    assert summary["fdr_qualified_modules"].iloc[0] == 2
    assert summary["excluded_from_plot"].iloc[0] == 5
    assert len(full) == 6
    larger, _, summary = C.display_results({spec.key: result}, [("a", "Renamed", spec)], "spearman", 2000, .05)
    assert len(larger) == 2 and larger["module_key"].is_unique
    assert summary["plotted_modules"].iloc[0] == 2
    assert larger["signed_correlation"].tolist() == [.5, -.5]
    empty, _, summary = C.display_results({spec.key: result}, [("a", "A", spec)], "spearman", None, .00001)
    assert empty.empty and summary["plotted_modules"].iloc[0] == 0


def test_duplicate_boxes_have_one_module_observation_each_and_independent_selection():
    a = C.ComparisonSeriesSpec("full_cohort", component="CT")
    b = dataclasses.replace(a, component="TS")
    first, _, _ = _fixture(a)
    second, _, _ = _fixture(b)
    shown, full, summary = C.display_results({a.key: first, b.key: second}, [("1", "CT", a), ("2", "TS", b), ("3", "CT duplicate", a)], top_k=2)
    assert len(shown) == 6
    assert not shown.duplicated(["box_id", "module_key"]).any()
    assert set(summary["plotted_modules"]) == {2}
    assert len(full) == 18


def test_plot_points_are_modules_and_exports_have_fixed_abs_axis():
    spec = C.ComparisonSeriesSpec("control_derived")
    result, _, _ = _fixture(spec)
    shown, _, summary = C.display_results({spec.key: result}, [("a", "Very long descriptive partition and component label", spec)])
    figure = C.comparison_figure(shown, summary, "cogn_global", "spearman")
    points = [trace for trace in figure.data if trace.type == "scatter"]
    assert sum(len(trace.y) for trace in points) == 3
    assert figure.layout.yaxis.range == (0, 1)
    assert "<br>" in figure.layout.xaxis.ticktext[0]
    assert all(len(trace.customdata[0]) == 7 for trace in points)
    assert len(C.comparison_figure(shown, summary, "cogn_global", "pearson", False).data) == 1


@pytest.mark.parametrize("source", ["full_cohort", "control_derived", "muscle_control_l4", "muscle_all_donors_l4"])
@pytest.mark.parametrize("scale", C.SCALE_ORDER)
def test_projected_multitissue_reader_routes_filters_and_components(monkeypatch, source, scale):
    called = []
    def loader(path, filters, value, modules):
        called.append(dict(path=str(path), filters=filters, value=value, modules=modules))
        for module in modules:
            yield module, pd.DataFrame({"sample_id": ["A", "B"], "metric_value": [1, 2]})
    monkeypatch.setattr(C.D, "load_module_details", lambda **kwargs: pd.DataFrame({"module": [10, 11]}))
    monkeypatch.setattr(C, "_scan_score_vectors", loader)
    rule = C.D.effect_rule_key("hedges_g", "fixed", .4, "ad_higher")
    spec = C.ComparisonSeriesSpec(source, scale=scale, differential_edge_rule=rule, score_normalization="retained_edge")
    rows = list(C._module_frames(spec))
    assert len(rows) == 2
    assert all(list(row[2].columns) == ["sample_id", "metric_value"] for row in rows)
    assert all(call["value"] == f"CT_{scale}" and call["modules"] == [10, 11] and rule in call["path"] for call in called)
    called.clear()
    component = C.D.module_set_components(source)[0]
    spec = dataclasses.replace(spec, component=component, differential_edge_rule="ad_control_discovery_fdr05", differential_fdr_scope="global", differential_fdr_threshold=.1)
    list(C._module_frames(spec))
    assert all(("component", "=", component) in call["filters"] and "/global/fdr_0.10/" in call["path"] and call["value"] == f"metric_{scale}" for call in called)


@pytest.mark.parametrize("source", ["full_cohort", "control_derived", *C.D.MUSCLE_MODULE_SETS])
@pytest.mark.parametrize("scale", C.SCALE_ORDER)
def test_combined_reader_scans_ct_ts_once_and_preserves_saved_values(monkeypatch, source, scale):
    calls = []
    def scanner(path, filters, value, modules):
        calls.append((path, filters, value, modules))
        yield 10, pd.DataFrame({"sample_id": ["A", "B"], f"CT_{scale}": [1., 2.], f"TS_{scale}": [7., -9.]})
        yield 11, pd.DataFrame()
    monkeypatch.setattr(C.D, "load_module_details", lambda **_: pd.DataFrame({"module": [10, 11]}))
    monkeypatch.setattr(C, "_scan_score_vectors", scanner)
    spec = C.ComparisonSeriesSpec(source, component=C.COMBINED_COMPONENT, scale=scale)
    frames = list(C._module_frames(spec))
    assert len(calls) == 1 and calls[0][2] == (f"CT_{scale}", f"TS_{scale}")
    assert C.score_path(spec) == C.score_path(dataclasses.replace(spec, component="CT"))
    assert [row[:2] for row in frames] == [(10, "CT"), (10, "TS"), (11, "CT"), (11, "TS")]
    assert frames[0][2]["metric_value"].tolist() == [1., 2.]
    assert frames[1][2]["metric_value"].tolist() == [7., -9.]
    assert frames[2][2].empty and frames[3][2].empty


def test_monotonic_raw_asinh_spearman_and_duplicate_donor_guard():
    spec = C.ComparisonSeriesSpec("control_derived")
    raw, metadata, frames = _fixture(spec)
    transformed = [(module, tissue, frame.assign(metric_value=np.arcsinh(frame["metric_value"])) if not frame.empty else frame) for module, tissue, frame in frames]
    asinh = C.statistics_from_module_frames(dataclasses.replace(spec, scale="asinh"), iter(transformed), {"": metadata}, "cogn_global", 10)
    assert np.allclose(raw["spearman_rho"], asinh["spearman_rho"], equal_nan=True)
    assert np.allclose(raw["spearman_p"], asinh["spearman_p"], equal_nan=True)
    duplicate = pd.concat([frames[0][2], frames[0][2].head(1)])
    with pytest.raises(ValueError, match="More than one score"):
        C._module_statistic(duplicate, metadata, "cogn_global", 10)


def test_nominal_outcomes_and_invalid_eigengene_masks_are_rejected():
    with pytest.raises(ValueError, match="Eigengenes cannot"):
        C.ComparisonSeriesSpec("control_derived", feature="eigengene", component="TS_AC", differential_edge_rule="ad_control_discovery_fdr05")
    with pytest.raises(ValueError, match="tissue-specific"):
        C.ComparisonSeriesSpec("control_derived", feature="eigengene", component="CT")
    with pytest.raises(ValueError, match="distinct"):
        C.ComparisonSeriesSpec("single_region_full_tissue_l3", tissues=("AC", "AC"))
    with pytest.raises(ValueError, match="Nominal"):
        C.calculate_series(C.ComparisonSeriesSpec("control_derived"), "clusters", "all_donors", 10, "fixture")


def test_batched_reader_projects_one_family_and_handles_cross_batch_modules(tmp_path, monkeypatch):
    path = tmp_path / "scores.parquet"
    frame = pd.DataFrame({
        "sample_id": [f"S-{i}" for i in range(12000)],
        "module": np.where(np.arange(12000) < 10000, 10, 11),
        "metric_family": "connectivity", "lioness_method": "control_anchored",
        "CT_raw": np.arange(12000, dtype=float), "TS_raw": -np.arange(12000, dtype=float), "CT_asinh": 99.,
    })
    frame.to_parquet(path, row_group_size=12000)
    monkeypatch.setattr(C, "ensure_data_path", lambda path, *_: path)
    rows = list(C._scan_score_vectors(path, [("metric_family", "=", "connectivity"), ("lioness_method", "=", "control_anchored")], "CT_raw", [10, 11, 12]))
    assert [len(row[1]) for row in rows] == [10000, 2000, 0]
    assert rows[0][1]["metric_value"].tolist() == list(range(10000))
    assert list(rows[0][1].columns) == ["sample_id", "metric_value"]
    combined = list(C._scan_score_vectors(path, [("metric_family", "=", "connectivity")], ("CT_raw", "TS_raw"), [10, 11, 12]))
    assert [len(row[1]) for row in combined] == [10000, 2000, 0]
    assert list(combined[0][1].columns) == ["sample_id", "CT_raw", "TS_raw"]
    assert combined[0][1]["TS_raw"].tolist() == list(-np.arange(10000, dtype=float))
    assert list(C._scan_score_vectors(path, [("metric_family", "=", "negative_density")], "CT_raw", [10]))[0][1].empty
    with pq.ParquetFile(path) as parquet:
        assert not C._row_group_matches(parquet, 0, [("module", "=", 99)])


def test_subset_metadata_pooled_without_diagnosis_group_filter(monkeypatch):
    meta = pd.DataFrame({"sample_id": list("ABCDE"), "cogn_global": [1, 2, 3, 4, 5],
                         "ad_control_split": ["Discovery", "Discovery", "Validation", "MCI_external", None],
                         "diagnosis_group": ["Control", "AD", "AD", "MCI", None]})
    monkeypatch.setattr(C.D, "load_sample_metadata", lambda *_: meta)
    spec = C.ComparisonSeriesSpec("full_cohort")
    assert len(C._metadata(spec, "cogn_global", "all_donors")) == 5
    assert C._metadata(spec, "cogn_global", "discovery_ad_control")["sample_id"].tolist() == ["A", "B"]
    assert C._metadata(spec, "cogn_global", "validation_ad_control")["sample_id"].tolist() == ["C"]
    assert C._metadata(spec, "cogn_global", "mci_external")["sample_id"].tolist() == ["D"]


@pytest.mark.parametrize("minimum_n", [5, 10, 20])
def test_minimum_n_boundary_and_nonfinite_exclusions(minimum_n):
    metadata = pd.DataFrame({"sample_id": [f"S-{i}" for i in range(minimum_n + 2)],
                             "cogn_global": np.arange(minimum_n + 2, dtype=float)})
    values = np.arange(minimum_n + 2, dtype=float)
    values[-2:] = [np.nan, np.inf]
    vector = pd.DataFrame({"sample_id": metadata["sample_id"], "metric_value": values})
    valid = C._module_statistic(vector, metadata, "cogn_global", minimum_n)
    assert valid["eligible"] and valid["n"] == minimum_n
    invalid = C._module_statistic(vector.iloc[1:], metadata, "cogn_global", minimum_n)
    assert not invalid["eligible"] and invalid["n"] == minimum_n - 1
    assert np.isnan(invalid["spearman_p"])


def _ui_app():
    from app_helpers.score_comparison_ui import render_view
    render_view({"module_sets": {}})


def _widget(elements, label):
    return next(element for element in elements if element.label == label)


def _mock_ui(monkeypatch):
    registry = {
        "control_derived": {"label": "Control-derived brain", "regional": False, "methods": ("control_anchored",), "modules": 6},
        "single_region_complete_case_l3": {
            "label": "Matched regional", "regional": True,
            "combinations": [{"cohort_scope": "common_450", "score_variant": "eigengene", "network_method": "not_applicable", "tissue": tissue, "modules": 6} for tissue in C.R.REGION_ORDER],
        },
    }
    calls = []
    monkeypatch.setattr(U, "source_registry", lambda *_: registry)
    monkeypatch.setattr(U.R, "load_regional_manifest", lambda: {})
    monkeypatch.setattr(U, "effect_settings", lambda *_: {})
    monkeypatch.setattr(U.D, "differential_data_available", lambda *_: False)
    monkeypatch.setattr(U, "availability_reason", lambda *_: "")
    def calculate(spec, *args):
        calls.append(spec)
        if spec.combines_ct_ts:
            return _combined_fixture(spec)[0]
        return _fixture(spec, spec.tissues if spec.is_regional else ("",))[0]
    monkeypatch.setattr(U, "calculate_series", calculate)
    return calls


def test_ui_requires_calculate_and_display_edits_never_read_donors(monkeypatch):
    calls = _mock_ui(monkeypatch)
    app = AppTest.from_function(_ui_app, default_timeout=30).run()
    assert not app.exception and not calls
    assert len(app.session_state[U.BOX_STATE]) == 5
    assert _widget(app.selectbox, "Correlation statistic").value == "spearman"
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception and len(calls) == 5
    assert len(app.get("plotly_chart")) == 1
    summary = app.dataframe[0].value
    assert set(summary["plotted_modules"]) == {3}
    app = _widget(app.radio, "Modules shown").set_value("Top K").run()
    app = _widget(app.number_input, "K per box").set_value(1).run()
    app = _widget(app.selectbox, "Correlation statistic").set_value("pearson").run()
    app = app.text_input[0].set_value("Renamed AC").run()
    assert not app.exception and len(calls) == 5
    assert set(app.dataframe[0].value["plotted_modules"]) == {1}
    app = next(button for button in app.button if button.label == "Duplicate").click().run()
    assert not app.exception and len(calls) == 5
    assert len(app.session_state[U.BOX_STATE]) == 6
    assert len(app.get("plotly_chart")) == 1  # Same spec; no re-read needed.
    app = next(button for button in app.button if button.label == "Move down" and not button.disabled).click().run()
    assert not app.exception and len(calls) == 5
    app = next(button for button in app.button if button.label == "Remove").click().run()
    assert not app.exception and len(calls) == 5


def test_ui_changed_outcome_is_stale_until_calculated_and_empty_region_is_safe(monkeypatch):
    calls = _mock_ui(monkeypatch)
    app = AppTest.from_function(_ui_app).run()
    app = _widget(app.button, "Calculate comparison").click().run()
    app = _widget(app.selectbox, "Comparison outcome").set_value("cogng_path_slope").run()
    assert not app.exception and len(calls) == 5
    assert len(app.get("plotly_chart")) == 0
    app = app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert _widget(app.button, "Calculate comparison").disabled


def test_ui_all_regions_and_ct_ts_combined_boxes_render_and_do_not_reread_for_top_k(monkeypatch):
    calls = _mock_ui(monkeypatch)
    app = AppTest.from_function(_ui_app, default_timeout=30).run()
    assert not _widget(app.checkbox, "All regions in one box").value
    app = _widget(app.checkbox, "All regions in one box").check().run()
    assert not app.exception and not calls
    app = _widget(app.selectbox, "Score component").set_value(C.COMBINED_COMPONENT).run()
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception and len(calls) == 5
    assert calls[0].tissues == C.R.regional_tissues(calls[0].source)
    assert sum(spec.combines_ct_ts for spec in calls) == 1
    summary = app.dataframe[0].value
    assert summary.iloc[0]["plotted_modules"] == 9  # Three regions, three eligible modules each.
    both = summary.loc[summary["component"].eq(C.COMBINED_COMPONENT)].iloc[0]
    assert both["plotted_modules"] == 3 and both["plotted_score_observations"] == 6
    app = _widget(app.radio, "Modules shown").set_value("Top K").run()
    app = _widget(app.number_input, "K per box").set_value(1).run()
    assert not app.exception and len(calls) == 5
    both = app.dataframe[0].value.loc[lambda frame: frame["component"].eq(C.COMBINED_COMPONENT)].iloc[0]
    assert both["plotted_modules"] == 1 and both["plotted_score_observations"] == 2
    app = _widget(app.checkbox, "All regions in one box").uncheck().run()
    assert not app.exception
    assert app.multiselect[0].value == ["AC"]  # Original custom choice is preserved.


def test_ui_all_regions_unavailable_for_partial_method_cohort(monkeypatch):
    _mock_ui(monkeypatch)
    registry = U.source_registry({})
    registry["single_region_complete_case_l3"]["combinations"] = [
        row for row in registry["single_region_complete_case_l3"]["combinations"] if row["tissue"] != "PCG"
    ]
    monkeypatch.setattr(U, "source_registry", lambda *_: registry)
    app = AppTest.from_function(_ui_app).run()
    assert not app.exception
    assert all(widget.disabled and not widget.value for widget in app.checkbox if widget.label == "All regions in one box")


def test_unfinished_sources_not_registered(monkeypatch):
    monkeypatch.setattr(C, "data_path_available", lambda _: True)
    registry = C.source_registry({"module_sets": {
        "control_derived": {"modules": 186, "status": "complete"},
        "full_cohort": {"modules": 154, "status": "running"},
    }}, {"completed_combinations": []})
    assert set(registry) == {"control_derived"}


def test_ui_effect_controls_percentiles_directions_and_eigengene_is_unfiltered(monkeypatch):
    calls = _mock_ui(monkeypatch)
    monkeypatch.setattr(U, "effect_settings", lambda *_: {"status": "complete", "completed_networks": ["lioness/control_anchored"]})
    app = AppTest.from_function(_ui_app).run()
    edges = next(widget for widget in app.selectbox if widget.label == "Edge subset")
    app = edges.set_value("effect").run()
    assert not app.exception
    assert _widget(app.selectbox, "Effect threshold").value == .4
    app = _widget(app.selectbox, "Effect direction").set_value("control_higher").run()
    app = _widget(app.selectbox, "Effect statistic").set_value("mean_difference").run()
    assert not app.exception
    assert _widget(app.selectbox, "Effect threshold rule").value == "module_top_fraction"
    app = _widget(app.selectbox, "Effect threshold").set_value(.1).run()
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception and len(calls) == 5
    filtered = next(spec for spec in calls if spec.differential_edge_rule != "all")
    assert filtered.differential_edge_rule == C.D.effect_rule_key("mean_difference", "module_top_fraction", .1, "control_higher")
    score = next(widget for widget in app.selectbox if widget.label == "Module score" and "Positive density" in widget.options)
    app = score.set_value("eigengene").run()
    assert not app.exception
    assert next(widget for widget in app.selectbox if widget.label == "Edge subset").value == "all"
    assert next(widget for widget in app.selectbox if widget.label == "Edge subset").disabled


def test_box_limit_and_atomic_failure_keep_old_results(monkeypatch):
    _mock_ui(monkeypatch)
    app = AppTest.from_function(_ui_app).run()
    app = _widget(app.button, "Calculate comparison").click().run()
    old_context = app.session_state[U.RESULT_STATE]["context"]
    for _ in range(7):
        app = _widget(app.button, "Add box").click().run()
    assert len(app.session_state[U.BOX_STATE]) == 12
    assert _widget(app.button, "Add box").disabled
    assert all(button.disabled for button in app.button if button.label == "Duplicate")
    def fail(*_):
        raise OSError("fixture I/O failure")
    monkeypatch.setattr(U, "calculate_series", fail)
    app = _widget(app.selectbox, "Comparison outcome").set_value("braak_stage").run()
    app = _widget(app.button, "Calculate comparison").click().run()
    assert not app.exception
    assert any("not replaced" in error.value for error in app.error)
    assert app.session_state[U.RESULT_STATE]["context"] == old_context


def test_local_saved_bundle_default_boxes_match_direct_statistics():
    """Optional source-backed QA, without initiating any Drive downloads in CI."""
    manifest_path = C.D.DATA_DIR / "data_manifest.json"
    regional_files = [C.R.regional_score_path("single_region_complete_case_l3", "common_450", "eigengene", "not_applicable", tissue) for tissue in C.R.REGION_ORDER]
    brain_file = C.D.module_set_path("aggregate_plot_data.parquet", "control_derived")
    required = [manifest_path, brain_file, *regional_files, C.R.REGIONAL_MANIFEST,
                C.R.REGIONAL_DETAILS, C.D.module_set_path("module_details.tsv", "control_derived"),
                C.D.module_set_path("sample_metadata.parquet", "full_cohort")]
    if not all(path.is_file() for path in required):
        pytest.skip("Saved local score bundle not installed; no remote download attempted")
    registry = C.source_registry(json.loads(manifest_path.read_text()))
    specs = [C.ComparisonSeriesSpec(**{k: v for k, v in row.items() if k not in {"uid", "label"}}) for row in U.default_boxes(registry)]
    counts = []
    for spec in specs:
        result = C.calculate_series(spec, "cogn_global", "all_donors", 10, "automated-local-saved-bundle")
        counts.append(len(result))
        assert result["module_key"].is_unique
        row = result.loc[result["eligible"]].iloc[0]
        if spec.is_regional:
            path = C.R.regional_score_path(spec.source, spec.cohort_scope, spec.feature, spec.method, row["tissue"])
            vector = pq.read_table(path, filters=[("module", "=", int(row["module"]))], columns=["sample_id", "metric_raw"]).to_pandas().rename(columns={"metric_raw": "metric_value"})
        else:
            frame = C.D.load_aggregate_scope(spec.method, module=int(row["module"]), metric_family=spec.feature, module_set=spec.source, metric_scale="raw", include_embedded_metadata=False)
            vector = frame[["sample_id", f"{spec.component}_raw"]].rename(columns={f"{spec.component}_raw": "metric_value"})
        metadata = C._metadata(spec, "cogn_global", "all_donors", row["tissue"])
        joined = vector.merge(metadata, on="sample_id").dropna()
        assert row["n"] == len(joined)
        assert row["pearson_r"] == pytest.approx(pearsonr(joined["metric_value"], joined["cogn_global"]).statistic)
        assert row["spearman_rho"] == pytest.approx(spearmanr(joined["metric_value"], joined["cogn_global"]).statistic)
    assert counts == [142, 122, 116, 186, 186]


@pytest.mark.parametrize("source", ["full_cohort", "control_derived", *C.D.MUSCLE_MODULE_SETS])
def test_saved_combined_catalog_matches_both_separate_component_catalogs(source):
    spec = C.ComparisonSeriesSpec(source, component=C.COMBINED_COMPONENT)
    needed = [C.score_path(spec), C.D.module_set_path("module_details.tsv", source),
              C.D.module_set_path("sample_metadata.parquet", source)]
    if not all(path.is_file() for path in needed):
        pytest.skip("Saved local scores unavailable; never download for this test")
    combined = C.calculate_series(spec, "cogn_global", "all_donors", 10, "combined-ct-ts-local-qa")
    expected_n = len(C.D.load_module_details(module_set=source))
    assert len(combined) == 2 * expected_n
    assert combined["module_key"].nunique() == expected_n
    assert combined["observation_key"].is_unique
    for component in ("CT", "TS"):
        single = C.calculate_series(dataclasses.replace(spec, component=component), "cogn_global", "all_donors", 10, "combined-ct-ts-local-qa")
        both = combined.loc[combined["component"].eq(component)].reset_index(drop=True)
        columns = ["module_key", "n", "eligible", "pearson_r", "spearman_rho", "pearson_p", "spearman_p",
                   "pearson_fdr_across_modules", "spearman_fdr_across_modules",
                   "pearson_fdr_module_family_n", "spearman_fdr_module_family_n"]
        pd.testing.assert_frame_equal(both[columns], single[columns])
