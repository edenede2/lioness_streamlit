"""All-module heatmap grain, unavailable cells and complete-search correction."""

import json
import numpy as np
import pandas as pd
from scipy import stats
from app_helpers.control_calibrated import ui as U


def test_heatmap_keeps_all_module_rows_na_cells_and_pair_counts():
    rows = []
    for module in (10, 2):
        for metric in ("gain", "loss"):
            for outcome in ("cogn_global", "amyloid"):
                rows.append(
                    dict(
                        module=module,
                        module_key=f"full_cohort::M{module}",
                        metric_family=metric,
                        component="TS",
                        outcome=outcome,
                        method="Spearman",
                        effect=np.nan if module == 10 else -0.25,
                        n=0 if module == 10 else 23,
                        q_scan=0.4,
                    )
                )
    # Nominal tests do not become signed correlations.
    rows.append(
        dict(rows[0], outcome="diagnosis_group", method="Kruskal-Wallis", effect=0.8)
    )
    frame = pd.DataFrame(rows)
    fig = U.correlation_heatmap_figure(
        frame, ["gain", "loss"], ["TS"], ["cogn_global", "amyloid"], True, "All modules"
    )
    assert list(fig.data[0].y) == [
        "M2 · Gain / strengthening · TS",
        "M2 · Loss / weakening · TS",
        "M10 · Gain / strengthening · TS",
        "M10 · Loss / weakening · TS",
    ]
    np.testing.assert_allclose(np.asarray(fig.data[0].z)[:2], -0.25)
    assert np.isnan(np.asarray(fig.data[0].z)[2:]).all()
    np.testing.assert_array_equal(
        np.asarray(fig.data[0].customdata)[:, 0], [23, 23, 0, 0]
    )
    assert fig.layout.coloraxis.cmin == -1 and fig.layout.coloraxis.cmax == 1
    assert frame.q_scan.eq(0.4).all()
    complete = U.correlation_heatmap_figure(
        frame,
        ["gain", "loss"],
        ["TS"],
        ["cogn_global", "amyloid"],
        True,
        "All modules",
        modules=[2, 10, 20],
    )
    assert len(complete.data[0].y) == 6
    assert np.isnan(np.asarray(complete.data[0].z)[4:]).all()
    assert np.isnan(np.asarray(complete.data[0].customdata)[4:]).all()


def test_multioutcome_search_corrects_across_modules_and_outcomes(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(114)
    ids = [f"target_{i}" for i in range(18)]
    jobs = [
        dict(
            family="full_cohort",
            module=i,
            module_key=f"full_cohort::M{i}",
            regional_tissue=None,
        )
        for i in (2, 10, 20)
    ]
    frames = {}
    for job in jobs[:2]:
        frames[job["module"]] = pd.DataFrame(
            dict(
                sample_id=ids,
                module_key=job["module_key"],
                component="TS",
                metric_family="gain",
                metric_raw=rng.normal(size=len(ids)),
                metric_asinh=rng.normal(size=len(ids)),
                cogn_global=rng.normal(size=len(ids)),
                amyloid=rng.normal(size=len(ids)),
            )
        )
    frames[10].loc[:4, "amyloid"] = np.nan

    def scores(**kw):
        assert kw["metric_families"] == ["gain"]
        if kw["module"] == 20:
            raise U.A.ScoreUnavailable("Unavailable synthetic module")
        return frames[kw["module"]].copy()

    monkeypatch.setattr(U, "ROOT", tmp_path)
    monkeypatch.setattr(U.A, "catalog", lambda: dict(modules=jobs))
    monkeypatch.setattr(U.A, "scores", scores)
    U.descriptive_analysis.clear()
    donors = ids[:-2]
    context = dict(
        family="full_cohort",
        module=None,
        transform="asinh",
        outcome="cogn_global",
        outcome_interpretation="numeric",
        beta=3,
        cohort_scope="complete_450",
        mask="all",
        allocation_id="synthetic",
    )
    result = U.full_partition_search(
        context,
        "calibrated_beta3",
        "complete_450",
        ["gain"],
        ["TS"],
        "cogn_global",
        donors,
        dict(rule="all"),
        None,
        outcomes=["cogn_global", "amyloid"],
    )
    expected_p = []
    for module in (2, 10):
        f = frames[module].loc[frames[module].sample_id.isin(donors)]
        for outcome in ("cogn_global", "amyloid"):
            paired = f[["metric_asinh", outcome]].dropna()
            rho, p = stats.spearmanr(paired.metric_asinh, paired[outcome])
            row = result.loc[
                result.module.eq(module) & result.outcome.eq(outcome)
            ].iloc[0]
            np.testing.assert_allclose([row.effect, row.p], [rho, p], rtol=1e-12)
            assert row.n == len(paired)
            expected_p.append(p)
    np.testing.assert_allclose(
        result.loc[result.module.ne(20), "q_search"],
        stats.false_discovery_control(expected_p),
        rtol=1e-12,
    )
    unavailable = result.loc[result.module.eq(20)]
    assert len(unavailable) == 2 and unavailable.q_search.isna().all()
    assert unavailable.unavailable_reason.eq("Unavailable synthetic module").all()
    assert "q_scan" not in result
    intent = json.loads(
        (
            tmp_path / "associations/saved" / f"{result.analysis_id.iloc[0]}.json"
        ).read_text()
    )
    assert intent["outcomes"] == ["cogn_global", "amyloid"]
    assert len(intent["planned_modules"]) == 3
    U.descriptive_analysis.clear()
