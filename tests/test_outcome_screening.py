from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr, false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_helpers.outcome_screening import bh, correlations, edge_batches, selected_union, signed_alt_edges, top_hits
from scripts.screening_bh import external_bh


def test_correlations_pairwise_complete_ties_missing_constants():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 6, size=(43, 7)).astype(float)
    y = rng.normal(size=43); y[[2, 7, 16]] = np.nan
    x[[1, 7, 13], 1] = np.nan; x[:35, 2] = np.nan
    x[:, 3] = 2; x[:, 4] = np.nan
    result = correlations(x, y)
    for j in (0, 1, 5, 6):
        mask = np.isfinite(x[:, j]) & np.isfinite(y)
        for name, test in (("pearson", pearsonr), ("spearman", spearmanr)):
            direct = test(x[mask, j], y[mask])
            assert result.loc[j, f"{name}_r"] == pytest.approx(direct.statistic, abs=1e-14)
            assert result.loc[j, f"{name}_p"] == pytest.approx(direct.pvalue, abs=1e-14)
        assert result.loc[j, "n"] == mask.sum()
    assert result.loc[2, "unavailable_reason"] == "too_few_complete_pairs"
    assert result.loc[3, "unavailable_reason"] == "constant_predictor"
    assert result.loc[4, "unavailable_reason"] == "too_few_complete_pairs"
    assert correlations(x, np.ones(43)).loc[0, "unavailable_reason"] == "constant_outcome"


def test_bh_missing_and_full_search_before_topk():
    p = np.array([.0001, .01, .05, .2, np.nan, 1.])
    q, n = bh(p)
    np.testing.assert_allclose(q[np.isfinite(p)], false_discovery_control(p[np.isfinite(p)]), atol=1e-16)
    assert n == 5 and np.isnan(q[4])
    assert q[0] != bh(p[:2])[0][0]
    with pytest.raises(ValueError):
        bh([1.1])


def test_external_bh_matches_scipy_across_spilled_batches(tmp_path):
    pytest.importorskip("duckdb")
    rng = np.random.default_rng(42)
    p = rng.uniform(size=1307); p[:5] = [0, 1, .05, .05, np.nan]
    paths = []
    for i, block in enumerate(np.array_split(np.arange(len(p)), 7)):
        path = tmp_path/f"{i}.parquet"
        pd.DataFrame(dict(test_id=block, p=p[block])).to_parquet(path, index=False)
        paths.append(path)
    selected = [0, 1, 2, 3, 4, 331, 1043]
    q, n = external_bh(paths, "p", selected, temporary_directory=tmp_path/"sort", memory_gib=1, batch_size=37)
    expected, total = bh(p)
    assert total == n
    for i in selected:
        if np.isfinite(expected[i]):
            assert q.loc[i] == pytest.approx(expected[i], abs=1e-15)
        else:
            assert i not in q.index
    assert not any((tmp_path/"sort").iterdir())


def test_topk_ties_union_frozen_independent_of_input_order():
    f = pd.DataFrame(dict(identity=["b", "c", "a", "d"], pearson_r=[.6, -.6, .6, np.nan], spearman_r=[.2, -.8, .7, np.nan]))
    assert top_hits(f, 2, "pearson").identity.tolist() == ["a", "b"]
    union = selected_union(f, 2)
    assert union.identity.tolist() == ["a", "b", "c"]
    pd.testing.assert_frame_equal(union, selected_union(f.sample(frac=1, random_state=1), 2))


@pytest.mark.parametrize("n,batch", [(3, 1), (17, 7), (40, 97)])
def test_bounded_edge_order(n, batch):
    values = list(edge_batches(n, batch))
    i = np.concatenate([v[0] for v in values]); j = np.concatenate([v[1] for v in values])
    expected = np.triu_indices(n, 1)
    np.testing.assert_array_equal(i, expected[0]); np.testing.assert_array_equal(j, expected[1])
    assert max(len(v[0]) for v in values) <= batch


def test_direct_signed_alt_loo_add_one_and_invariance():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(31, 9)); new = rng.normal(size=(4, 9))
    rows, cols = np.triu_indices(9, 1); powers = np.where(rows < 3, 2., 3.)
    targets = np.vstack((ref[:3], new)); positions = np.r_[np.arange(3), [-1]*4]
    result = signed_alt_edges(ref, targets, rows, cols, powers, reference_positions=positions)
    def direct(x):
        r = np.corrcoef(x, rowvar=False)[rows, cols]
        return np.sign(r)*np.abs(r)**powers
    base = direct(ref)
    for i in range(len(targets)):
        expected = (31*base-30*direct(np.delete(ref, i, axis=0))) if i < 3 else (32*direct(np.vstack((ref, targets[i])))-31*base)
        np.testing.assert_allclose(result[i], expected, rtol=0, atol=1e-13)
    isolated = signed_alt_edges(ref, new[2:3], rows, cols, powers)
    np.testing.assert_array_equal(isolated[0], result[5])
    reversed_targets = signed_alt_edges(ref, new[::-1], rows, cols, powers)
    np.testing.assert_array_equal(reversed_targets[::-1], result[3:])


def test_full_cohort_preserves_m188_after_unique_node_resolution():
    from scripts.screening_sources import assignments
    a = assignments("full_cohort")
    assert a.module.nunique() == 154
    assert len(a) == 46868
    assert a.loc[a.module.eq(188)].shape[0] == 27
    assert 1918 in set(a.module)
    assert not a.duplicated(["tissue", "gene"]).any()
