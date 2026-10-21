"""Coverage, metadata-grain, namespace, privacy and live Streamlit contracts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers import donor_explorer as D
from app_helpers import donor_explorer_ui as UI

APP = Path(__file__).resolve().parents[1]


def mini_catalog():
    base = pd.DataFrame(dict(sample_id=["one", "two", "three", "four"], diagnosis_group=["Control", "AD", "MCI", None],
                             age_at_death=[70., 80., np.nan, 90.], study=["ROS", "MAP", "ROS", None],
                             has_AC=[True, False, True, False], has_DLPFC=[True, True, False, False],
                             has_PCG=[True, False, True, True]))
    base["n_available_tissues"] = base[["has_" + t for t in D.BRAIN_TISSUES]].sum(axis=1)
    return base


def test_coverage_distinguishes_available_only_pairs_and_empty_selection():
    f = mini_catalog()
    assert f.loc[D.coverage_mask(f, D.BRAIN_TISSUES, ["DLPFC"]), "sample_id"].tolist() == ["one", "two"]
    assert f.loc[D.coverage_mask(f, D.BRAIN_TISSUES, ["DLPFC"], "exactly"), "sample_id"].tolist() == ["two"]
    assert f.loc[D.coverage_mask(f, D.BRAIN_TISSUES, ["AC", "PCG"], "exactly"), "sample_id"].tolist() == ["three"]
    assert not D.coverage_mask(f, D.BRAIN_TISSUES, []).any()
    with pytest.raises(ValueError):
        D.coverage_mask(f, D.BRAIN_TISSUES, ["VH"])


def test_coverage_table_counts_each_component_correctly_not_unique_union():
    table = D.coverage_summary(mini_catalog(), D.BRAIN_TISSUES).set_index("Component")
    assert table.at["DLPFC", "Total"] == 2
    assert table.at["AC–PCG", "Total"] == 2
    assert table.at["AC–DLPFC–PCG", "Total"] == 1
    assert table.at["PCG", D.MISSING] == 1
    assert table.Total.sum() > len(mini_catalog())


def test_filters_preserve_missing_values_unless_explicitly_excluded_and_clear_means_empty():
    f = mini_catalog()
    assert len(D.filter_donors(f, {"study": ["ROS"]})) == 2
    assert len(D.filter_donors(f, {"study": []})) == 0
    assert len(D.filter_donors(f, numeric_ranges={"age_at_death": (75, 85, True)})) == 2
    assert len(D.filter_donors(f, numeric_ranges={"age_at_death": (75, 85, False)})) == 1
    assert len(D.filter_donors(f, require_observed=["age_at_death", "study"])) == 2
    missing = D.missingness(f, ["age_at_death", "study"]).set_index("field")
    assert missing.at["age_at_death", "missing_percent"] == 25
    assert missing.at["study", "observed"] == 3


def test_reconciliation_rejects_conflicts_duplicates_and_private_fields(monkeypatch, tmp_path):
    a = pd.DataFrame(dict(sample_id=["same"], age_at_death=[70.], study=["ROS"]))
    b = a.copy(); b["age_at_death"] = 80.
    with pytest.raises(ValueError, match="Regional metadata conflict"):
        D.reconcile_regional_frames({"AC": a, "DLPFC": b})
    path = tmp_path / "metadata.parquet"
    pd.DataFrame(dict(sample_id=["public"], donor=["not-allowed"])).to_parquet(path)
    with pytest.raises(ValueError, match="public-identifier"):
        D._read(path, "brain")
    pd.DataFrame(dict(sample_id=["same", "same"])).to_parquet(path)
    with pytest.raises(ValueError, match="unique public sample"):
        D._read(path, "brain")


def test_snapshot_comparison_tolerates_float_roundoff_and_records_real_changes():
    a = pd.DataFrame(dict(sample_id=["a", "b", "c"], cogn_global=[.1, .2, np.nan]))
    b = pd.DataFrame(dict(sample_id=["a", "b", "c"], cogn_global=[.1 + 8e-17, .4, .5]))
    row = D.snapshot_differences(a, b).iloc[0]
    assert row.differing_values == 1
    assert row.missing_only_original == 1
    assert D.same_metadata_value(.1, .1 + 8e-17)
    assert not D.same_metadata_value(.1, .2)
    assert D.same_metadata_value(np.nan, np.nan)
    assert not D.same_metadata_value(np.nan, .1)
    merged = D.reconcile_regional_frames({"AC": a.iloc[:1], "DLPFC": b.iloc[:1]})
    assert len(merged) == 1 and merged.has_AC.all() and merged.has_DLPFC.all()


@pytest.fixture
def catalogs():
    if not all(p.is_file() for dataset in ("brain", "motor") for p in D.source_paths(dataset)):
        pytest.skip("Approved metadata not available locally; do not contact Drive in tests")
    return {dataset: D.load_catalog(dataset) for dataset in ("brain", "motor")}


def test_source_backed_counts_and_validation_cohort_reproduce_existing_membership(catalogs):
    catalog, original = catalogs["brain"]
    expected = {"all": 1257, "complete": 464, "original": 450, "incomplete": 793,
                "at_least:AC": 730, "at_least:DLPFC": 1216, "at_least:PCG": 659,
                "exactly:AC": 22, "exactly:DLPFC": 346, "exactly:PCG": 5,
                "at_least:AC+DLPFC": 694, "at_least:AC+PCG": 478, "at_least:DLPFC+PCG": 640,
                "exactly:AC+DLPFC": 230, "exactly:AC+PCG": 14, "exactly:DLPFC+PCG": 176}
    for preset, count in expected.items():
        result = D.select_cohort("brain", catalog, original, preset)
        assert len(result) == count and result.sample_id.is_unique
    validation = D.select_cohort("brain", catalog, original, "incomplete")
    assert validation.diagnosis_group.value_counts().to_dict() == {"AD": 350, "Control": 244, "MCI": 177, "Unclassified": 22}
    validation_file = APP / "data/control_derived/camel_validation/sample_metadata.parquet"
    if validation_file.exists():
        assert set(validation.sample_id) == set(pd.read_parquet(validation_file, columns=["sample_id"]).sample_id)
    assert catalog.memory_usage(deep=True).sum() < 2 * 1024**2


def test_motor_and_brain_namespaces_are_never_joined_and_sources_deduplicate(catalogs):
    brain, _ = catalogs["brain"]
    motor, original = catalogs["motor"]
    assert len(motor) == 357 and original.empty
    assert motor.metadata_namespace.eq("motor").all() and brain.metadata_namespace.eq("brain").all()
    assert motor[["has_" + t for t in D.MOTOR_TISSUES]].all().all()
    # The real source IDs overlap; do not interpret that as linkage.
    assert set(motor.sample_id) & set(brain.sample_id)
    assert list(D.cohort_options("motor", motor, original)) == ["motor_common"]
    assert not D.PRIVATE_FIELDS.intersection(motor.columns)
    with pytest.raises(ValueError):
        D.source_paths("brain_plus_motor")


def test_original_values_remain_intact_and_snapshots_match_to_numerical_precision(catalogs):
    regional, original = catalogs["brain"]
    selected = D.select_cohort("brain", regional, original, "original")
    saved = pd.read_parquet(APP / "data/sample_metadata.parquet").set_index("sample_id")
    selected = selected.set_index("sample_id").loc[saved.index]
    for c in saved.columns:
        if c in selected:
            pd.testing.assert_series_equal(selected[c].reset_index(drop=True), saved[c].reset_index(drop=True),
                                           check_names=False, check_dtype=False, check_exact=True)
    differences = D.snapshot_differences(original, regional)
    assert differences.differing_values.eq(0).all()
    assert differences[["missing_only_original", "missing_only_regional"]].eq(0).all().all()


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_donor_views_filters_details_and_dataset_switch_render_without_scores(catalogs, monkeypatch):
    UI.cached_catalog.clear()
    monkeypatch.setattr(D, "load_catalog", lambda dataset: tuple(x.copy() for x in catalogs[dataset]))
    # The new standalone view uses no score loaders; explicitly fail if added.
    from app_helpers import camel_validation
    monkeypatch.setattr(camel_validation, "scores", lambda *a, **k: pytest.fail("No donor score loads in metadata view"))
    script = '''
from app_helpers.donor_explorer_ui import render_view
from app_helpers.figure_export import render_exportable_plot
render_view(render_exportable_plot)
'''
    app = AppTest.from_string(script, default_timeout=120).run()
    assert not app.exception
    assert app.metric[0].value == "1257"
    app = widget(app.selectbox, "Donor cohort / region coverage").set_value("exactly:DLPFC").run()
    assert not app.exception and app.metric[0].value == "346"
    app = widget(app.selectbox, "Donor cohort / region coverage").set_value("complete").run()
    assert app.metric[0].value == "464"
    app = widget(app.selectbox, "Donor cohort / region coverage").set_value("all").run()
    for section in ("Phenotype distributions", "Missingness", "Donor table & details"):
        app = widget(app.radio, "Donor view").set_value(section).run()
        assert not app.exception, section
    app = widget(app.radio, "Donor dataset").set_value("motor").run()
    assert not app.exception and app.metric[0].value == "357"
    assert any("Single-tissue-only" in x.value for x in app.info)
    app = widget(app.radio, "Donor view").set_value("Overview").run()
    app = widget(app.radio, "Donor dataset").set_value("brain").run()
    assert not app.exception and app.metric[0].value == "1257"
    app = widget(app.multiselect, "Diagnosis group values").set_value([]).run()
    assert not app.exception and app.metric[0].value == "0"
    assert any("No donors match" in x.value for x in app.info)
    UI.cached_catalog.clear()


def test_main_navigation_enters_donor_explorer_before_module_controls(catalogs):
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.pills, "Analysis view").set_value("Donor explorer").run()
    assert not app.exception
    assert any(x.value == "Donor explorer" for x in app.header)
    assert not any(x.label == "Module definition" for x in app.selectbox)


def test_phenotype_modes_custom_coverage_and_selected_sample_provenance(catalogs):
    app = AppTest.from_string('''
from app_helpers.donor_explorer_ui import render_view
from app_helpers.figure_export import render_exportable_plot
render_view(render_exportable_plot)
''', default_timeout=120).run()
    app = widget(app.selectbox, "Donor cohort / region coverage").set_value("custom").run()
    app = widget(app.radio, "Coverage rule").set_value("exactly").run()
    assert not app.exception and app.metric[0].value == "346"
    app = widget(app.multiselect, "Required brain regions").set_value([]).run()
    assert not app.exception and app.metric[0].value == "0"
    app = widget(app.selectbox, "Donor cohort / region coverage").set_value("all").run()
    app = widget(app.multiselect, "Numeric donor filters").set_value(["age_at_death", "motor10_demog_slope"]).run()
    assert not app.exception and app.metric[0].value == "1257"
    app = widget(app.radio, "Donor view").set_value("Phenotype distributions").run()
    for mode in ("Histogram", "ECDF", "Two-phenotype scatter"):
        app = widget(app.selectbox, "Donor phenotype view").set_value(mode).run()
        assert not app.exception and app.get("plotly_chart"), mode
    app = widget(app.radio, "Donor view").set_value("Donor table & details").run()
    sample = catalogs["brain"][1].sample_id.iloc[0]
    app = widget(app.selectbox, "Inspect one public sample").set_value(sample).run()
    assert not app.exception
    comparison = next(x.value for x in app.dataframe if "different" in x.value)
    assert not comparison.different.any()
    columns = widget(app.multiselect, "Donor table columns").value
    assert "metadata_namespace" in columns and "metadata_snapshot" in columns
