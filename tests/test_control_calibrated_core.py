from dataclasses import replace
import numpy as np
import pytest
from app_helpers.camel import frozen_edge_weights
from app_helpers.control_calibrated.core import (
    Settings,
    fit_component,
    calibrate,
    aggregate_weights,
    generate_weights,
    component_available,
)
from app_helpers.control_calibrated.store import save_fit, load_fit


def fixture(policy="R_mean", batch=2, extra=False):
    rng = np.random.default_rng(210611)
    R = rng.normal(size=(12, 3))
    C = rng.normal(size=(8, 3))
    T = rng.normal(size=(5, 3))
    nodes = [("A", str(i)) for i in range(3)]
    s = Settings(
        12, 8, sparse_cell_policy=policy, edge_batch=batch, sign_seeds=tuple(range(10))
    )
    return R, C, T, nodes, s


@pytest.mark.parametrize("policy", ["unavailable", "R_mean"])
@pytest.mark.parametrize("where", ["calibration", "target"])
def test_missing_endpoint_does_not_invalidate_batched_complete_edge(policy, where):
    R, C, T, nodes, s = fixture(policy)
    C[0, 2] = np.nan if where == "calibration" else C[0, 2]
    T[0, 2] = np.nan if where == "target" else T[0, 2]
    results = []
    for size in (1, 2):
        fit = fit_component(
            R,
            C,
            nodes,
            [0, 1],
            [1, 2],
            3,
            range(12),
            range(12, 20),
            replace(s, edge_batch=size),
            {"source": "synthetic"},
            frozen_edge_weights,
            R_available=np.ones(12, bool),
            C_available=np.ones(8, bool),
        )
        w = generate_weights(
            fit.reference,
            T,
            np.array([0, 1]),
            np.array([1, 2]),
            np.full(2, 3.0),
            s,
            frozen_edge_weights,
            np.ones(5, bool),
        )
        results.append((fit, w))
    a, b = results
    assert a[0].calibration["n"][0] == 8 and b[0].calibration["n"][0] == 8
    assert a[0].calibration["valid"][0] and b[0].calibration["valid"][0]
    np.testing.assert_allclose(a[1][:, 0], b[1][:, 0])
    assert a[0].edge_keys[0] == b[0].edge_keys[0]
    if policy == "unavailable":
        if where == "calibration":
            assert a[0].calibration["n"][1] == 7 and not a[0].calibration["valid"][1]
        else:
            assert np.isnan(a[1][0, 1]) and np.isfinite(a[1][0, 0])


@pytest.mark.parametrize("policy", ["R_mean", "unavailable"])
@pytest.mark.parametrize("present", [True, False])
def test_authoritative_presence_not_module_genes(policy, present):
    R, C, T, nodes, s = fixture(policy)
    T[0, :2] = np.nan
    outputs = []
    for n in (2, 3):
        fit = fit_component(
            R[:, :n],
            C[:, :n],
            nodes[:n],
            [0],
            [1],
            3,
            range(12),
            range(12, 20),
            s,
            {"source": "synthetic"},
            frozen_edge_weights,
        )
        score = fit.score(
            T[:, :n],
            range(20, 25),
            frozen_edge_weights,
            available={"A": np.array([present, True, True, True, True])},
        )
        outputs.append((fit, score))
    assert outputs[0][0].edge_keys == outputs[1][0].edge_keys
    for k in ("G", "L", "S", "A", "NA_reason"):
        np.testing.assert_equal(outputs[0][1][k], outputs[1][1][k])
    if present and policy == "R_mean":
        assert np.isfinite(outputs[0][1]["G"][0])
    elif not present:
        assert outputs[0][1]["NA_reason"][0] == "missing_component_tissue"
    else:
        assert outputs[0][1]["NA_reason"][0] == "incomplete_target_edge_coverage"


def test_missing_third_tissue_preserves_edge_and_full_union_na():
    R, C, T, nodes, s = fixture()
    nodes = [("A", "x"), ("A", "y"), ("B", "z")]
    fit = fit_component(
        R,
        C,
        nodes,
        [0, 1],
        [1, 2],
        3,
        range(12),
        range(12, 20),
        s,
        {"source": "synthetic"},
        frozen_edge_weights,
    )
    presence = {"A": np.ones(5, bool), "B": np.zeros(5, bool)}
    p = component_available(T, nodes, fit.rows, fit.columns, presence)
    w = generate_weights(
        fit.reference, T, fit.rows, fit.columns, fit.powers, s, frozen_edge_weights, p
    )
    assert np.isfinite(w[:, 0]).all() and np.isnan(w[:, 1]).all()
    o = fit.score(T, range(20, 25), frozen_edge_weights, available=presence)
    assert (o["NA_reason"] == "missing_component_tissue").all()
    masked = fit.score(
        T,
        range(20, 25),
        frozen_edge_weights,
        available=presence,
        edge_mask=np.array([True, False]),
    )
    assert np.isfinite(masked["G"]).all()
    assert masked["possible_edges"].tolist() == [1] * 5
    assert masked["signed_eligible_edges"].tolist() == [1] * 5


def test_identities_signs_and_own_signed_mask():
    s = Settings(12, 8, sign_seeds=tuple(range(20)))
    w = np.arange(1, 9.0)[:, None]
    c = calibrate(w, s)
    o = aggregate_weights(np.array([[-10.0], [-4.5], [-1.0], [0.0]]), c, s)
    np.testing.assert_allclose(o["G"] - o["L"], o["S"])
    np.testing.assert_allclose(o["G"] + o["L"], o["A"])
    assert o["G_reversed"][0] > 0 and o["L_reversed"][2] > 0 and o["L_zero"][3] > 0
    c = calibrate(np.array([-2, 2] * 4)[:, None], s)
    o = aggregate_weights(np.array([[3.0]]), c, s)
    assert np.isnan(o["G"][0]) and np.isfinite(o["D"][0])


def test_cache_content_roundtrip_and_corruption(tmp_path):
    R, C, T, nodes, s = fixture()
    f = fit_component(
        R,
        C,
        nodes,
        [0],
        [1],
        3,
        range(12),
        range(12, 20),
        s,
        {"source": "synthetic"},
        frozen_edge_weights,
    )
    save_fit(tmp_path / "fit", f)
    g = load_fit(tmp_path / "fit")
    np.testing.assert_allclose(
        f.score(T, range(20, 25), frozen_edge_weights)["G"],
        g.score(T, range(20, 25), frozen_edge_weights)["G"],
    )
    p = tmp_path / "fit.npz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="content hash"):
        load_fit(tmp_path / "fit")


def test_role_counts_beta_and_unit_invariance():
    R, C, T, nodes, s = fixture()
    args = (R, C, nodes, [0], [1])
    a = fit_component(
        *args,
        3,
        range(12),
        range(12, 20),
        s,
        {"source": "synthetic"},
        frozen_edge_weights,
    )
    b = fit_component(
        *args,
        1,
        range(12),
        range(12, 20),
        s,
        {"source": "synthetic"},
        frozen_edge_weights,
    )
    assert a.edge_keys != b.edge_keys
    with pytest.raises(ValueError, match="Training-role"):
        a.score(C, range(12, 20), frozen_edge_weights)
    c = calibrate(C, s)
    base = aggregate_weights(T, c, s)
    for factor in (1e-100, 1e100):
        o = aggregate_weights(T * factor, calibrate(C * factor, s), s)
        np.testing.assert_allclose(o["G"], base["G"], rtol=1e-12, atol=1e-12)
