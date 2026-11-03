#!/usr/bin/env python3
"""Run the real Streamlit views against the installed, read-only Drive release."""
import json
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from streamlit.testing.v1 import AppTest
from app_helpers import drive_data as D
from app_helpers.control_calibrated import adapter as A, registry as C
from app_helpers.control_calibrated.inputs import atomic_json

OUT = APP / "artifacts/control_calibrated_hosted_release_20261103"


def check(app, label):
    errors = [str(e.message) for e in app.exception]
    if errors:
        raise AssertionError((label, errors))
    return app


def app_for(view="Associations", family="full_cohort"):
    job = min(
        (j for j in A.catalog()["modules"] if j["family"] == family),
        key=lambda j: j["physical_nodes"],
    )
    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=300)
    app.query_params.update(
        view=view,
        screen_partition=family,
        screen_module=str(job["module"]),
        score_source="calibrated_beta3",
    )
    app = check(app.run(), view + ":" + family)
    if view == "Gene & edge associations":
        app = check(
            app.radio(key="screening_calibrated_source")
            .set_value("calibrated_beta3")
            .run(),
            view,
        )
    return app


def main():
    assert os.environ.get("LIONESS_FORCE_DRIVE") == "1"
    assert D.DATA_DIR != D.LOCAL_DATA_DIR.resolve()
    checks = []
    catalog = A.catalog()
    for family, info in catalog["families"].items():
        if info["regional"]:
            app = app_for()
            organization = next(
                w for w in app.radio if w.label == "Module organization"
            )
            app = check(organization.set_value("independent_regional").run(), family)
            app = check(
                app.selectbox(key="regional_partition_source").set_value(family).run(),
                family,
            )
            app = check(
                app.radio(key="regional_calibrated_source")
                .set_value("calibrated_beta3")
                .run(),
                family,
            )
        else:
            app = app_for(family=family)
        assert app.get("plotly_chart") and app.get("download_button"), family
        assert app.multiselect(key="cc_metrics").value == ["gain", "loss"]
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        assert spec["layout"]["meta"]["family"] == family
        checks.append(
            dict(family=family, view="Associations", plotted=True, downloadable=True)
        )
        print(json.dumps(checks[-1]), flush=True)
    for view in (
        "Feature distributions",
        "Correlation heatmaps",
        "Statistics",
        "Module finder",
        "CT–TS screen",
        "Edge summaries",
        "Methods & data",
        "Module details",
        "KEGG enrichment",
        "Donor edge explorer",
        "Gene & edge associations",
        "Edge volcano",
    ):
        app = app_for(view=view)
        if view in ("Donor edge explorer", "Gene & edge associations", "Edge volcano"):
            assert any(
                "remain available in the local research app" in item.value
                for item in app.info
            ), view
            assert not any(
                b.key in ("cc_inspect_edges", "cc_run_edge_association")
                for b in app.button
            )
        checks.append(dict(family="full_cohort", view=view, no_exception=True))
        print(json.dumps(checks[-1]), flush=True)
    app = app_for()
    app = check(app.radio(key="cc_scope").set_value("maximum_component").run(), "scope")
    app = check(app.selectbox(key="cc_scale").set_value("asinh").run(), "transform")
    estimator = next(w for w in app.radio if w.label == "Network estimator")
    app = check(estimator.set_value("calibrated_beta1").run(), "beta")
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["beta"] == 1
    assert spec["layout"]["meta"]["transform"] == "asinh"
    assert spec["layout"]["meta"]["cohort_scope"] == "maximum_component"
    result = dict(
        status="passed",
        real_read_only_backend=True,
        checks=checks,
        changed_scope_transform_beta=True,
        edge_explanation_verified=True,
        deployment_pushed=False,
        prediction=False,
    )
    atomic_json(OUT / "hosted_ui_smoke.json", result)
    print(json.dumps(dict(status="passed", views=len(checks))), flush=True)


if __name__ == "__main__":
    main()
