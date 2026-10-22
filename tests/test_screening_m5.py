from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.screening_m5 import (M5Settings, PROFILES, canonical_input, full_m5, m5_donor_network, priority, subsample_plan)


def fixture():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(29, 12))
    x[:, 5] = .4*x[:, 0]+rng.normal(size=len(x))*.7
    ids = [f"private-fixture-{i:03d}" for i in range(len(x))]
    nodes = [("VH", f"G{i}") for i in range(4)]+[("SMA", f"G{i}") for i in range(4)]+[("QUAD", f"G{i}") for i in range(4)]
    return x, ids, nodes


def test_m5_complete_estimator_direct_bootstrap_and_null():
    from models.se2.methods.soft_weighted import _full_bootstrap_stability
    x, ids, nodes = fixture()
    settings = M5Settings(subsamples=7, null_permutations=4, null_edge_cap=5)
    result, diagnostics = full_m5(x, ids, nodes, "control_fitted", settings=settings)
    plan = iter(subsample_plan(ids, settings))
    class PlannedRNG:
        def choice(self, n, size, replace):
            assert not replace
            selected = next(plan)
            assert len(selected) == size
            return selected
    # Existing bootstrap kernel, coupled to precisely the same resamples.
    freq, mean = _full_bootstrap_stability(x[:, :4].T, x[:, 4:8].T, B=7,
        donor_frac=.8, corr_method="pearson", tau_r=.10, is_intra=False, rng=PlannedRNG())
    src, dst = np.where(freq>0)
    order = sorted(range(len(src)), key=lambda k: (priority(42, "null_edges:VH-SMA", f"G{src[k]}|G{dst[k]}"), k))[:5]
    src, dst = src[order], dst[order]
    null = []
    for rep in range(4):
        perm = sorted(range(len(ids)), key=lambda k: (priority(42, f"null_donors:VH-SMA:{rep}", ids[k]), ids[k]))
        for i,j in zip(src, dst, strict=True):
            null.append(abs(np.corrcoef(x[:, i], x[perm, 4+j])[0,1]))
    mu, sigma = np.mean(null), np.std(null)
    direct = np.sign(mean)*freq*np.abs(mean)**8*np.maximum((np.abs(mean)-mu)/(sigma+1e-8), 0)
    np.testing.assert_allclose(result[:4, 4:8], direct, rtol=3e-5, atol=1e-8)
    assert diagnostics["VH-SMA"]["null_mean"] == pytest.approx(mu, abs=1e-14)
    assert diagnostics["VH-SMA"]["null_sd"] == pytest.approx(sigma, abs=1e-14)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, result.T)
    np.testing.assert_array_equal(result.diagonal(), np.zeros(len(nodes)))


def test_m5_gene_and_donor_reordering_invariant():
    x, ids, nodes = fixture(); s = M5Settings(subsamples=5, null_permutations=3, null_edge_cap=4)
    a, _ = full_m5(x, ids, nodes, "all_donor_fitted", settings=s)
    di = np.arange(len(x))[::-1]; ni = np.random.default_rng(19).permutation(len(nodes))
    b, _ = full_m5(x[np.ix_(di,ni)], [ids[i] for i in di], [nodes[i] for i in ni], "all_donor_fitted", settings=s)
    np.testing.assert_array_equal(b, a[np.ix_(ni, ni)])


def test_m5_loo_addone_rebuild_complete_components_and_sign():
    x, ids, nodes = fixture(); s = M5Settings(subsamples=5, null_permutations=3, null_edge_cap=4)
    ref, ref_ids = x[:-1], ids[:-1]
    base, _ = full_m5(ref, ref_ids, nodes, "control_fitted", settings=s)
    add, diag = m5_donor_network(ref, ref_ids, x[-1], ids[-1], nodes, "control_fitted", settings=s, base=base)
    changed, _ = full_m5(x, ids, nodes, "control_fitted", settings=s)
    np.testing.assert_array_equal(add, 29*changed-28*base)
    assert diag["role"] == "add_one"
    loo, diag = m5_donor_network(ref, ref_ids, x[0], ids[0], nodes, "control_fitted", settings=s, base=base)
    changed, _ = full_m5(ref[1:], ref_ids[1:], nodes, "control_fitted", settings=s)
    np.testing.assert_array_equal(loo, 28*base-27*changed)
    assert diag["role"] == "leave_one_out"
    assert "diagnosis" not in __import__('inspect').signature(m5_donor_network).parameters


def test_m5_beta_profiles_and_no_dense_genome_launch():
    assert [PROFILES['control_fitted'][t] for t in ('VH','SMA','QUAD')] == [5,7,13]
    assert [PROFILES['all_donor_fitted'][t] for t in ('VH-SMA','VH-QUAD','SMA-QUAD')] == [6,6,7]
    with pytest.raises(ValueError, match="genome-wide"):
        full_m5(np.zeros((20,3001)), [str(i) for i in range(20)], [('VH',str(i)) for i in range(3001)], 'control_fitted')
