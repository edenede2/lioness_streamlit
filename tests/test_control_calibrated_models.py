"""Independent descriptive-model checks, with no predictive fitting workflow."""

import numpy as np
import pandas as pd
from scipy.stats import t
from app_helpers.control_calibrated.ui import _ols


def test_hc3_nullable_complete_cases_against_lstsq_and_outer_products():
    rng = np.random.default_rng(272)
    x = rng.normal(size=40)
    age = rng.normal(80, 4, size=40)
    sex = np.array(["F", "M"] * 20, dtype=object)
    y = 1.3 * x + 0.2 * age + (sex == "M") + rng.normal(size=40)
    age[2] = np.nan
    sex[4] = None
    y[8] = np.nan
    cov = pd.DataFrame({"age": pd.array(age, dtype="Float64"), "sex": sex})
    result = _ols(x, y, cov, ["age", "sex"])
    ok = np.isfinite(age) & pd.notna(sex) & np.isfinite(y)
    design = np.column_stack(
        [np.ones(ok.sum()), x[ok], age[ok], (sex[ok] == "M").astype(float)]
    )
    b = np.linalg.lstsq(design, y[ok], rcond=None)[0]
    pinv = np.linalg.pinv(design)
    h = np.diag(design @ pinv)
    residual = y[ok] - design @ b
    variance = sum(
        np.outer(pinv[:, i], pinv[:, i]) * (residual[i] / (1 - h[i])) ** 2
        for i in range(ok.sum())
    )
    se = np.sqrt(variance[1, 1])
    assert result["n"] == ok.sum() == 37
    np.testing.assert_allclose(result["effect"], b[1], rtol=1e-10)
    np.testing.assert_allclose(result["standard_error_HC3"], se, rtol=1e-10)
    np.testing.assert_allclose(
        result["p"], 2 * t.sf(abs(b[1] / se), ok.sum() - 4), rtol=1e-10
    )
    assert not result["unavailable_reason"]


def test_nonestimable_models_keep_na_p_and_specific_reasons():
    x = np.arange(10, dtype=float)
    frame = pd.DataFrame({"duplicate": x})
    rank_deficient = _ols(x, x + 0.1, frame, ["duplicate"])
    constant = _ols(x, np.ones(10), frame, [])
    assert (
        np.isnan(rank_deficient["p"])
        and "rank-deficient" in rank_deficient["unavailable_reason"]
    )
    assert (
        np.isnan(constant["p"]) and constant["unavailable_reason"] == "constant outcome"
    )


def test_numeric_nominal_covariate_uses_indicators_not_ordering():
    rng = np.random.default_rng(243)
    x = rng.normal(size=60)
    group = np.tile([1, 2, 4], 20)
    y = 0.8 * x + (group == 2) * 1.5 - (group == 4) * 2 + rng.normal(size=60)
    result = _ols(x, y, pd.DataFrame({"clusters": group}), ["clusters"])
    design = np.column_stack([np.ones(60), x, group == 2, group == 4])
    np.testing.assert_allclose(
        result["effect"], np.linalg.lstsq(design, y, rcond=None)[0][1], rtol=1e-12
    )
    assert result["model_rank"] == 4
    assert (
        "clusters_2" in result["expanded_model_terms"]
        and "clusters_4" in result["expanded_model_terms"]
    )


def test_nominal_comparison_display_uses_kw_for_all_category_fields():
    from app_helpers.score_comparison import (
        ComparisonSeriesSpec,
        statistics_from_module_frames,
        display_results,
    )

    spec = ComparisonSeriesSpec(
        "full_cohort", estimator="calibrated_beta3", feature="gain", component="TS"
    )
    rng = np.random.default_rng(54)
    metadata = pd.DataFrame(
        {
            "sample_id": ["synthetic" + str(i) for i in range(30)],
            "clusters": np.repeat([1, 2, 4], 10),
        }
    )
    frame = pd.DataFrame(
        {
            "sample_id": metadata.sample_id,
            "metric_value": rng.normal(size=30) + np.repeat([0, 2, -1], 10),
        }
    )
    result = statistics_from_module_frames(
        spec, iter([(1, "", frame)]), {"": metadata}, "clusters", 5
    )
    shown, full, summary = display_results(
        {spec.key: result}, [("box", "Calibrated gain", spec)]
    )
    assert (
        len(shown) == 1
        and shown.association_measure.eq("Kruskal epsilon-squared").all()
    )
    assert shown.correlation_method.eq("not_applicable").all()
    assert shown.selected_p.notna().all()
