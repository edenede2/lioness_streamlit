from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_helpers import protocol_v11


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _widget(elements, label: str):
    return next(element for element in elements if element.label == label)


def _analysis_selector(app: AppTest):
    return _widget(
        app.pills if hasattr(app, "pills") else app.get("button_group"),
        "Analysis view",
    )


def test_protocol_public_bundle_is_aggregate_only() -> None:
    manifest = protocol_v11.load_protocol_manifest()
    assert manifest["protocol_version"] == "1.1"
    assert manifest["complete_expression_n"] == 464
    assert manifest["privacy"]["donor_identifiers_present"] is False
    for name in (
        "outcome_dictionary.tsv",
        "outcome_exclusions.tsv",
        "source_inventory.tsv",
        "expression_inventory.tsv",
        "split_registry_summary.tsv",
    ):
        frame = protocol_v11.load_protocol_table(name)
        assert not frame.empty
        assert {"donor", "projid", "donor_id"}.isdisjoint(frame.columns)


def test_protocol_outcomes_preserve_verified_and_blocked_definitions() -> None:
    outcomes = protocol_v11.load_protocol_table("outcome_dictionary.tsv")
    status = outcomes.set_index("key")["audit_status"].to_dict()
    assert status["diagnosis_binary"] == "verified"
    assert status["cogng_path_slope"] == "blocked_definition"
    assert outcomes.set_index("key").loc["amyloid", "actual_n"] == 456


def test_protocol_results_loader_exposes_only_complete_aggregate_performance() -> None:
    performance = protocol_v11.load_protocol_table("performance")
    assert not performance.empty
    assert set(performance["outcome"]) == {
        "diagnosis_binary",
        "diagnosis_three_class",
        "gpath",
    }
    binary = performance.loc[performance["outcome"].eq("diagnosis_binary")]
    three_class = performance.loc[
        performance["outcome"].eq("diagnosis_three_class")
    ]
    gpath = performance.loc[performance["outcome"].eq("gpath")]
    assert set(binary["representation"]) == {"B", "G", "E", "C0"}
    assert set(three_class["representation"]) == {"B", "E", "C0"}
    assert set(gpath["representation"]) == {"B", "E", "C0"}
    assert set(performance["model_family"]) == {
        "logistic_l2",
        "logistic_elastic_net",
        "ridge",
        "elastic_net",
    }
    assert set(performance["completed_folds"]) == {25}
    assert set(binary["n_donors"]) == {331}
    assert set(three_class["n_donors"]) == {450}
    assert set(gpath["n_donors"]) == {457}
    assert {"donor", "projid", "donor_id"}.isdisjoint(performance.columns)
    assert protocol_v11.load_protocol_table("coefficients").empty


def test_protocol_view_renders_validated_performance() -> None:
    app = AppTest.from_file(APP, default_timeout=300).run()
    view_value = "Prediction" if hasattr(app, "pills") else ["Prediction"]
    app = _analysis_selector(app).set_value(view_value).run(timeout=300)
    if not hasattr(app, "pills"):
        _analysis_selector(app)._value = ["Prediction"]
    app = _widget(app.radio, "Prediction mode").set_value("protocol_v11").run(
        timeout=300
    )
    app.session_state["protocol_v11_tabs"] = "Results"
    if not hasattr(app, "pills"):
        _analysis_selector(app)._value = ["Prediction"]
    app = app.run(timeout=300)
    assert not app.exception
    assert len(app.get("plotly_chart")) >= 1
    assert any(
        "metric" in dataframe.value.columns
        and "representation" in dataframe.value.columns
        for dataframe in app.dataframe
    )
