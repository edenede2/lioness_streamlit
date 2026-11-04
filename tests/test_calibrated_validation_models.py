import json
import numpy as np
import pandas as pd
from scipy import stats, interpolate
import statsmodels.api as sm
from app_helpers.control_calibrated import (
    validation_inference as I,
    validation_focused as F,
    validation_run as R,
)


def metadata(n):
    rng = np.random.default_rng(731)
    return pd.DataFrame(
        dict(
            sample_id=[f"synthetic_{i}" for i in range(n)],
            age_at_death=rng.normal(86, 6, n),
            education_years=rng.integers(10, 22, n),
            sex_code=np.where(np.arange(n) % 2, "Code 1", "Code 0"),
            study=np.where(np.arange(n) % 3, "MAP", "ROS"),
            diagnosis_group="Control",
        )
    )


def test_HC3_coefficients_intervals_and_joint_F_against_statsmodels():
    rng = np.random.default_rng(951)
    n = 100
    x = rng.normal(size=(n, 4))
    z = np.column_stack([np.ones(n), x])
    y = 0.6 * x[:, 2] - 0.5 * x[:, 3] + rng.normal(size=n) * (1 + abs(x[:, 0]))
    actual = F.hc3_fit(y, z, [3, 4])
    reference = sm.OLS(y, z).fit().get_robustcov_results(cov_type="HC3", use_t=True)
    test = reference.f_test(np.eye(5)[[3, 4]])
    np.testing.assert_allclose(
        actual["coefficient"], reference.params[[3, 4]], rtol=1e-10
    )
    np.testing.assert_allclose(
        actual["standard_error"], reference.bse[[3, 4]], rtol=1e-10
    )
    np.testing.assert_allclose(
        actual["ci_low"], reference.conf_int()[[3, 4], 0], rtol=1e-10
    )
    np.testing.assert_allclose(
        actual["ci_high"], reference.conf_int()[[3, 4], 1], rtol=1e-10
    )
    np.testing.assert_allclose(
        [actual["F_hc3"], actual["p_hc3"]],
        [float(test.fvalue), float(test.pvalue)],
        rtol=1e-10,
    )


def test_partial_rank_HC3_matches_independent_standard_OLS():
    rng = np.random.default_rng(138)
    meta = metadata(80)
    x = rng.normal(size=80)
    y = 0.4 * x + 0.1 * meta.age_at_death + rng.normal(size=80)
    actual = I.infer_group(
        x[None, :], np.asarray(y), meta, adjusted=True, draws=39, blocks=meta.study
    )
    z = np.column_stack(
        [
            np.ones(80),
            stats.rankdata(meta.age_at_death),
            meta.sex_code.eq("Code 1"),
            stats.rankdata(meta.education_years),
            meta.study.eq("ROS"),
            stats.rankdata(x),
        ]
    )
    ref = (
        sm.OLS(stats.rankdata(y), z)
        .fit()
        .get_robustcov_results(cov_type="HC3", use_t=True)
    )
    np.testing.assert_allclose(
        [actual["rank_beta"][0], actual["hc3_se"][0], actual["hc3_p"][0]],
        [ref.params[-1], ref.bse[-1], ref.pvalues[-1]],
        rtol=1e-10,
    )


def test_natural_age_spline_span_matches_scipy_natural_cubic_basis():
    meta = metadata(90)
    age = meta.age_at_death.to_numpy()
    d = I.nuisance_design(meta, ("age_at_death",), spline_age=True)
    knots = np.array(d.age_knots)
    spline = interpolate.CubicSpline(knots, np.eye(4), bc_type="natural")
    z = spline(age)
    for bound, select in [(knots[0], age < knots[0]), (knots[-1], age > knots[-1])]:
        z[select] = spline(bound) + (age[select, None] - bound) * spline(bound, 1)
    q1 = np.linalg.qr(d.matrix)[0]
    q2 = np.linalg.qr(z)[0]
    np.testing.assert_allclose(q1 @ q1.T, q2 @ q2.T, atol=1e-10)
    assert d.rank == 4


def test_joint_permutation_matches_explicit_F_refits():
    rng = np.random.default_rng(28)
    n = 70
    z = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
    y = 0.5 * z[:, -1] + rng.normal(size=n)
    interest = [3, 4]
    answer = F.linear_permutation(
        y, z, interest, draws=59, seed=66, blocks=np.repeat(["ROS", "MAP"], 35)
    )
    reduced = z[:, :3]
    fit = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    error = y - fit

    def statistic(value):
        e0 = value - reduced @ np.linalg.lstsq(reduced, value, rcond=None)[0]
        e1 = value - z @ np.linalg.lstsq(z, value, rcond=None)[0]
        return ((e0 @ e0 - e1 @ e1) / 2) / (e1 @ e1 / (n - 5))

    observed = statistic(y)
    count = 0
    for ids in I.permutation_indices(n, 59, 66, np.repeat(["ROS", "MAP"], 35)):
        for p in ids:
            count += statistic(fit + error[p]) >= observed - 1e-10
    assert answer["p"] == (count + 1) / 60


def test_triplet_B_C_identity_is_actual_membership_with_missing_covariates(tmp_path):
    root = tmp_path
    dest = root / "prepared"
    dest.mkdir()
    n = 80
    meta = metadata(n)
    meta["role"] = "target"
    meta["in_original_analysis"] = np.arange(n) < 40
    meta["has_AC"] = True
    meta["has_DLPFC"] = True
    meta["has_PCG"] = np.arange(n) < 40
    rng = np.random.default_rng(6)
    meta["cogn_global"] = rng.normal(size=n)
    meta.loc[:4, "age_at_death"] = np.nan
    meta.to_parquet(dest / "brain_metadata.parquet", index=False)
    x = rng.normal(size=(1, n))
    np.save(dest / "brain.npy", x)
    pd.DataFrame(
        [
            dict(
                row_id=0,
                module_key="full_cohort::M1",
                family="full_cohort",
                module=1,
                regional_tissue=None,
                dataset="brain",
                input_variant="brain",
                component="TS_DLPFC",
                metric_family="gain",
                beta=3,
                allocation_id="synthetic",
                focused=False,
                is_canonical=True,
            )
        ]
    ).to_parquet(dest / "brain_rows.parquet", index=False)
    base = dict(
        root=str(root),
        analysis_id="synthetic",
        stage="scan",
        variant="brain",
        scope="original",
        outcome="cogn_global",
        stratum="all",
        spline=False,
        rows=[0],
        focus=False,
        covariates=list(I.COVARIATES),
        draws=39,
        bootstrap_draws=19,
    )
    bc = dict(base, model="BC", donors=list(range(5, 40)), task_id="bc")
    a = dict(base, model="A", donors=list(range(40)), task_id="a")
    R.execute_task(bc)
    R.execute_task(a)
    matched = pd.read_parquet(root / "scan_shards/bc.parquet")
    all_available = pd.read_parquet(root / "scan_shards/a.parquet")
    assert set(matched.model) == {"B", "C"} and matched.n.eq(35).all()
    assert matched.donor_set_hash.nunique() == 1
    assert matched.donor_set_hash.iloc[0] == I.membership_hash(
        meta.sample_id.iloc[5:40]
    )
    assert (
        all_available.n.iloc[0] == 40
        and all_available.donor_set_hash.iloc[0] != matched.donor_set_hash.iloc[0]
    )
    assert matched.missing_covariates_after_pair_n.eq(5).all()
    np.testing.assert_array_equal(np.load(dest / "brain.npy"), x)


def test_BH_families_include_secondary_beta1_but_separate_models():
    base = dict(
        dataset="brain",
        cohort="original",
        allocation_id="fixed",
        model="C",
        study_stratum="all",
        mask="all",
        transform="native",
        outcome="cogn_global",
        beta=3,
        p_method="exact_tie_aware_pairings",
        exceedances=1,
    )
    frame = pd.DataFrame(
        [
            dict(base, metric_family="gain", p=0.001),
            dict(base, metric_family="loss", p=0.01),
            dict(base, metric_family="net_change", p=0.04),
            dict(base, metric_family="gain", beta=1, p=0.2),
            dict(base, metric_family="gain", p=np.nan),
            dict(base, metric_family="gain", model="A", p=0.00001),
        ]
    )
    result, families = R.new_correction(frame)
    np.testing.assert_allclose(
        result.q_scan.iloc[:4], stats.false_discovery_control([0.001, 0.01, 0.04, 0.2])
    )
    np.testing.assert_allclose(
        result.q_GL_outcome.iloc[:2], stats.false_discovery_control([0.001, 0.01])
    )
    assert np.isnan(result.q_GL_outcome.iloc[2]) and np.isnan(result.q_scan.iloc[4])
    assert result.q_scan.iloc[5] == 0.00001
    family = families.loc[families.model.eq("C")].iloc[0]
    assert family.planned == 5 and family.estimable == 4 and family.excluded == 1


def test_membership_uses_tissue_flags_and_frozen_target_roles():
    f = pd.DataFrame(
        dict(
            role=["target", "target", "reference", "target"],
            in_original_analysis=[True, False, True, False],
            has_AC=True,
            has_DLPFC=True,
            has_PCG=[True, False, True, True],
        )
    )
    np.testing.assert_array_equal(
        R.cohort_mask(f, "original"), [True, False, False, False]
    )
    np.testing.assert_array_equal(
        R.cohort_mask(f, "incomplete"), [False, True, False, False]
    )
    f.loc[0, "has_PCG"] = False
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        R.cohort_mask(f, "original")


def test_adjusted_df_guard_does_not_drop_nonconstant_nuisance_terms():
    rng = np.random.default_rng(9)
    meta = metadata(35)
    extra = [f"extra_{i}" for i in range(15)]
    for name in extra:
        meta[name] = rng.normal(size=35)
    answer = I.infer_group(
        rng.normal(size=(1, 35)),
        rng.normal(size=35),
        meta,
        adjusted=True,
        draws=19,
        covariates=(*I.COVARIATES, *extra),
    )
    assert (
        np.isnan(answer["p"][0])
        and answer["reason"][0] == "primary_ineligible_residual_df_below_20"
    )
    assert answer["design"].rank > 15


def test_calibrated_custom_search_repairs_n4_p_in_app_adapter(monkeypatch, tmp_path):
    from app_helpers.control_calibrated import ui as U

    monkeypatch.setattr(U, "ROOT", tmp_path)
    U.descriptive_analysis.clear()
    f = pd.DataFrame(
        dict(
            sample_id=["a", "b", "c", "d"],
            module_key="full_cohort::M1",
            component="TS_DLPFC",
            metric_family="gain",
            metric_value=[1, 2, 3, 4],
            cogn_global=[1, 2, 3, 4],
        )
    )
    result = U.descriptive_analysis(
        f,
        dict(
            family="full_cohort",
            module=1,
            beta=3,
            cohort_scope="original",
            allocation_id="fixed",
            transform="native",
            mask="all",
            outcome="cogn_global",
        ),
        ["cogn_global"],
    )
    assert result.p.iloc[0] == 1 / 12 and result.q_search.iloc[0] == 1 / 12
    assert result.p_method.iloc[0] == "exact_tie_aware_pairings"
    U.descriptive_analysis.clear()


def test_interval_plot_keeps_bounds_even_if_they_exclude_point_estimate():
    from app_helpers.control_calibrated.validation_ui import _plot

    f = pd.DataFrame(
        [
            dict(
                module_key="control_derived::M747",
                component="TS_DLPFC",
                metric_family="gain",
                cohort="original",
                model="C",
                n=100,
                effect=-0.2,
                ci_low=-0.8,
                ci_high=-0.5,
                p=0.2,
                q_scan=0.4,
                beta=3,
                primary_eligible=True,
            )
        ]
    )
    captured = []
    source = dict(
        analysis_id="fixed",
        beta=3,
        outcome="cogn_global",
        correction_scope="within model/cohort",
        plot_units="rank association",
    )
    _plot(f, "test", lambda fig, **kw: captured.append((fig, kw)), source)
    fig, kw = captured[0]
    assert (
        fig.layout.meta["analysis_id"] == "fixed"
        and kw["metadata"]["correction_scope"] == "within model/cohort"
    )
    interval = next(t for t in fig.data if t.mode == "lines")
    assert list(interval.x) == [-0.8, -0.5, None]
    assert list(fig.layout.yaxis.range) == [0.5, -0.5]
    assert list(fig.layout.yaxis.categoryarray) == [
        "control_derived::M747 · TS_DLPFC · gain"
    ]


def test_triplet_plot_order_and_all_category_rows():
    from app_helpers.control_calibrated.validation_ui import _plot, MODEL_LABELS

    rows = []
    for model in ["B", "C", "A"]:
        for module in range(20):
            rows.append(
                dict(
                    family="control_derived",
                    module=module,
                    component="TS_DLPFC",
                    metric_family="gain",
                    cohort="original",
                    model=model,
                    n=100,
                    effect=0.1,
                    ci_low=-0.1,
                    ci_high=0.3,
                    p=0.3,
                    beta=3,
                )
            )
    captured = []
    _plot(
        pd.DataFrame(rows),
        "test",
        lambda f, **kw: captured.append(f),
        dict(beta=3, outcome="cogn_global"),
    )
    fig = captured[0]
    assert list(fig.layout.yaxis.range) == [19.5, -0.5]
    assert len(fig.layout.yaxis.categoryarray) == 20
    assert [a.text for a in fig.layout.annotations] == [
        MODEL_LABELS[m] for m in ["A", "B", "C"]
    ]
