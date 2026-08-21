from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def widget_with_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def assert_app_clean(app: AppTest) -> None:
    assert not app.exception
    assert not [error for error in app.error if "Traceback" in str(error.value)]


def test_streamlit_smoke_lioness_and_bonobo_module_definitions() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)

    outcome = widget_with_label(app.selectbox, "Association outcome")
    assert len(outcome.options) == 12
    outcome.set_value("adnc")
    widget_with_label(app.selectbox, "Color points by").set_value(
        "age_at_death"
    ).run()
    widget_with_label(app.selectbox, "Continuous color scale").set_value("Viridis")
    widget_with_label(app.radio, "Network estimator").set_value("bonobo").run()
    assert_app_clean(app)

    edge_subset = widget_with_label(app.radio, "BONOBO edge subset")
    edge_subset.set_value("Significant edges").run()
    widget_with_label(app.radio, "Significant-edge rule").set_value("bh_fdr05").run()
    assert_app_clean(app)

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    network_method = widget_with_label(app.radio, "Network method")
    assert network_method.value == "bonobo"
    assert network_method.options == ["All-donor empirical-Bayes BONOBO"]
