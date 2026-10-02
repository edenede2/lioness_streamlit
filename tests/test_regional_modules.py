from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app_helpers.regional_modules import (
    completed_combinations,
    load_regional_details,
    load_regional_manifest,
    load_regional_scores,
    regional_categorical_catalog,
    regional_correlation_catalog,
)


APP_ROOT = Path(__file__).resolve().parents[1]
APP = APP_ROOT / "streamlit_app.py"
REGIONAL_ROOT = APP_ROOT / "data/regional_modules"
EXPECTED_MODULES = {
    "single_region_complete_case_l3": {"AC": 142, "DLPFC": 122, "PCG": 116},
    "single_region_full_tissue_l3": {"AC": 118, "DLPFC": 144, "PCG": 106},
}
EXPECTED_MAXIMUM_DONORS = {"AC": 730, "DLPFC": 1216, "PCG": 659}


def _widget(elements, label: str):
    return next(element for element in elements if element.label == label)


def _preserve_pills(app: AppTest) -> None:
    if not hasattr(app, "pills"):
        selector = _widget(app.get("button_group"), "Analysis view")
        if isinstance(selector.value, str):
            selector._value = [selector.value]


def _select_view(app: AppTest, label: str) -> AppTest:
    _preserve_pills(app)
    selector = _widget(
        app.pills if hasattr(app, "pills") else app.get("button_group"),
        "Analysis view",
    )
    return selector.set_value(label if hasattr(app, "pills") else [label]).run()


def _regional_view(label: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=300).run()
    app = _select_view(app, label)
    _preserve_pills(app)
    app = _widget(app.radio, "Module organization").set_value(
        "independent_regional"
    ).run(timeout=300)
    return app


def test_regional_manifest_and_score_grain() -> None:
    manifest = load_regional_manifest()
    completed = completed_combinations(manifest)
    assert len(completed) == 36
    assert set(completed["status"]) == {"complete"}
    connectivity = completed.loc[completed["score_variant"].eq("connectivity")]
    assert len(connectivity) == 24
    assert set(connectivity["network_method"]) == {"standard", "control_anchored"}
    details = load_regional_details()
    observed = (
        details.groupby(["partition_source", "tissue"], observed=True)["module"]
        .nunique().to_dict()
    )
    expected = {
        (source, tissue): count
        for source, tissues in EXPECTED_MODULES.items()
        for tissue, count in tissues.items()
    }
    assert observed == expected
    assert details["module_key"].is_unique

    for source, tissues in EXPECTED_MODULES.items():
        for tissue, count in tissues.items():
            scores = pd.read_parquet(
                REGIONAL_ROOT / source / "common_450/eigengene" / f"{tissue}.parquet"
            )
            assert len(scores) == 450 * count
            assert scores["sample_id"].nunique() == 450
            assert not scores.duplicated(["sample_id", "module"]).any()
            assert not {"donor", "projid"}.intersection(scores.columns)
            assert np.allclose(
                scores["metric_asinh"], np.arcsinh(scores["metric_raw"])
            )
            maximum = pd.read_parquet(
                REGIONAL_ROOT / source / "maximum_tissue/eigengene" / f"{tissue}.parquet"
            )
            assert len(maximum) == EXPECTED_MAXIMUM_DONORS[tissue] * count
            assert maximum["sample_id"].nunique() == EXPECTED_MAXIMUM_DONORS[tissue]
            assert not maximum.duplicated(["sample_id", "module"]).any()

    common_ids = set(pd.read_parquet(APP_ROOT / "data/sample_metadata.parquet")["sample_id"])
    for tissue, donor_count in EXPECTED_MAXIMUM_DONORS.items():
        metadata = pd.read_parquet(
            REGIONAL_ROOT / "metadata/maximum_tissue" / f"{tissue}.parquet"
        )
        assert len(metadata) == donor_count
        assert common_ids.issubset(set(metadata["sample_id"]))
        assert not {"donor", "projid"}.intersection(metadata.columns)


def test_regional_association_fdr_is_within_tissue_partition() -> None:
    result = regional_correlation_catalog(
        "single_region_complete_case_l3",
        "common_450",
        "eigengene",
        "not_applicable",
        "AC",
        "raw",
        ("Control", "MCI", "AD"),
        "diagnosis_group",
        ("Control", "MCI", "AD"),
        10,
        ("cogn_global",),
        True,
    )
    assert len(result) == 142 * 4
    assert set(result["spearman_fdr_module_family_n"]) == {142}
    assert set(result["pearson_fdr_module_family_n"]) == {142}
    assert set(result["tissue"]) == {"AC"}

    nominal = regional_categorical_catalog(
        "single_region_complete_case_l3",
        "common_450",
        "eigengene",
        "not_applicable",
        "AC",
        "raw",
        ("Control", "MCI", "AD"),
        "clusters",
        (1, 2, 3, 4),
        10,
    )
    assert len(nominal) == 142
    assert set(nominal["categorical_fdr_module_family_n"]) == {142}


def test_regional_association_view_renders_three_tissues() -> None:
    app = _regional_view("Associations")
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert {box.label for box in app.selectbox}.issuperset(
        {"AC module", "DLPFC module", "PCG module"}
    )
    assert load_regional_manifest()["status"] == "complete"


def test_regional_distribution_and_heatmap_views_render() -> None:
    distributions = _regional_view("Feature distributions")
    assert not distributions.exception
    assert len(distributions.get("plotly_chart")) == 1
    heatmap = _regional_view("Correlation heatmaps")
    assert not heatmap.exception
    assert len(heatmap.get("plotly_chart")) == 1


def test_regional_maximum_tissue_view_renders() -> None:
    app = _regional_view("Associations")
    _preserve_pills(app)
    app = _widget(app.selectbox, "Regional donor cohort").set_value(
        "maximum_tissue"
    ).run(timeout=300)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1


def test_regional_control_referenced_lioness_view_renders() -> None:
    app = _regional_view("Associations")
    _preserve_pills(app)
    score = _widget(app.selectbox, "Regional module score")
    assert len(score.options) == 3
    app = score.set_value(("connectivity", "control_anchored")).run(timeout=300)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1


def test_regional_manifest_contains_no_private_identifiers() -> None:
    manifest = json.loads((REGIONAL_ROOT / "manifest.json").read_text())
    assert manifest["privacy"]["mapping_persisted"] is False
    assert manifest["privacy"]["forbidden_fields"] == ["donor", "projid"]
