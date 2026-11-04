import itertools
import numpy as np
import pandas as pd
from scipy import stats
from app_helpers.control_calibrated import validation_inference as V


def metadata(n=80):
    rng = np.random.default_rng(72)
    return pd.DataFrame(
        dict(
            age_at_death=rng.normal(87, 6, n),
            education_years=rng.integers(9, 22, n),
            sex_code=np.where(np.arange(n) % 2, "Code 1", "Code 0"),
            study=np.where(np.arange(n) % 3, "MAP", "ROS"),
        )
    )


def test_exact_n4_perfect_and_ties():
    meta = metadata(4)
    answer = V.infer_group(np.array([[1, 2, 3, 4]]), np.array([1, 2, 3, 4]), meta)
    assert answer["p"][0] == 1 / 12
    assert answer["draws"][0] == 24
    x = np.array([1, 1, 2, 3])
    y = np.array([1, 2, 2, 4])
    observed = abs(stats.spearmanr(x, y).statistic)
    expected = np.mean(
        [
            abs(stats.spearmanr(x, p).statistic) >= observed - 1e-12
            for p in itertools.permutations(y)
        ]
    )
    answer = V.infer_group(x[None, :], y, meta)
    assert answer["p"][0] == expected


def test_monte_carlo_positive_reproducible_and_chunk_invariant():
    x = np.arange(10, dtype=float)[None, :]
    a = V.infer_group(x, x[0], metadata(10), draws=199, seed=81, batch=17)
    b = V.infer_group(x, x[0], metadata(10), draws=199, seed=81, batch=31)
    assert a["p"][0] == b["p"][0] == 1 / 200
    assert a["method"][0] == "monte_carlo_pairings"
    blocks = np.repeat(["ROS", "MAP"], 5)
    a = np.concatenate(list(V.permutation_indices(10, 53, 31, blocks, batch=7)))
    b = np.concatenate(list(V.permutation_indices(10, 53, 31, blocks, batch=17)))
    np.testing.assert_array_equal(a, b)
    assert (blocks[a] == blocks[None, :]).all()


def test_confounding_and_retained_conditional_signal():
    rng = np.random.default_rng(931)
    meta = metadata(140)
    age = stats.rankdata(meta.age_at_death)
    age = (age - age.mean()) / age.std()
    x = 5 * age + rng.normal(size=len(meta))
    y = 5 * age + rng.normal(size=len(meta))
    plain = V.infer_group(x[None, :], y, meta, draws=499, seed=12)
    adjusted = V.infer_group(
        x[None, :], y, meta, adjusted=True, draws=499, seed=12, blocks=meta.study
    )
    assert plain["effect"][0] > 0.8 and abs(adjusted["effect"][0]) < 0.2
    signal = V.infer_group(
        x[None, :],
        y + 2 * x,
        meta,
        adjusted=True,
        draws=499,
        seed=12,
        blocks=meta.study,
    )
    assert signal["effect"][0] > 0.5 and signal["p"][0] < 0.02


def test_freedman_lane_matches_explicit_refit_with_fixed_initial_ranks():
    rng = np.random.default_rng(420)
    meta = metadata(65)
    x = rng.normal(size=65)
    y = 0.5 * x + 0.07 * meta.age_at_death.to_numpy() + rng.normal(size=65)
    answer = V.infer_group(
        x[None, :],
        y,
        meta,
        adjusted=True,
        draws=79,
        seed=99,
        blocks=meta.study,
        batch=13,
    )
    z = V.nuisance_design(meta).matrix
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    design = np.column_stack([z, rx])
    df = len(y) - design.shape[1]

    def t(value):
        beta = np.linalg.lstsq(design, value, rcond=None)[0]
        error = value - design @ beta
        se = np.sqrt((error @ error / df) * np.linalg.inv(design.T @ design)[-1, -1])
        return beta[-1] / se

    fitted = z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    residual = ry - fitted
    observed = abs(t(ry))
    b = 0
    for indices in V.permutation_indices(len(y), 79, 99, meta.study, batch=11):
        for ids in indices:
            b += abs(t(fitted + residual[ids])) >= observed - 1e-10
    assert answer["exceedances"][0] == b
    assert answer["p"][0] == (b + 1) / 80


def test_constant_covariates_rank_deficiency_eligibility_and_immutability():
    rng = np.random.default_rng(3)
    meta = metadata(60)
    meta["study"] = "MAP"
    design = V.nuisance_design(meta)
    assert any("study" in x for x in design.removed)
    assert not any("study" in x for x in design.names)
    x = rng.normal(size=(2, 60))
    x[1] = 1
    y = rng.normal(size=60)
    snapshot = x.copy()
    r = V.infer_group(x, y, meta, adjusted=True, draws=19)
    np.testing.assert_array_equal(x, snapshot)
    assert np.isnan(r["p"][1]) and "constant" in r["reason"][1]
    small = V.infer_group(x[:, :24], y[:24], meta.iloc[:24], adjusted=True, draws=19)
    assert np.isnan(small["p"]).all() and "below_30" in small["reason"][0]
    meta["education_years"] = meta.age_at_death
    r = V.infer_group(x, y, meta, adjusted=True, draws=19)
    assert (
        np.isnan(r["p"]).all()
        and (r["reason"] == "rank_deficient_nuisance_design").all()
    )


def test_bootstrap_reranks_refits_and_records_degeneracy():
    rng = np.random.default_rng(21)
    meta = metadata(50)
    x = rng.normal(size=(2, 50))
    y = x[0] + rng.normal(size=50)
    a = V.bootstrap_group(x, y, meta, draws=49, seed=18)
    b = V.bootstrap_group(x, y, meta, draws=49, seed=18)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], equal_nan=True)
    assert a["adjusted_low"][0] < a["adjusted_high"][0]
    assert 0 < a["adjusted_valid"][0] <= 49
