import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, rankdata
from app_helpers.camel import FrozenRINT, frozen_edge_weights, feature_values, score_framework


def test_camel_label_is_scoped_not_a_new_estimator():
    assert score_framework("control_derived") == "CAMEL"
    assert score_framework("muscle_control_l4") == "CAMEL"
    assert score_framework("full_cohort") == "LIONESS"
    assert score_framework("muscle_all_donors_l4") == "LIONESS"
    assert score_framework("control_derived", estimator="bonobo") == "BONOBO"


def test_add_one_direct_loo_and_target_invariance():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(25, 8))
    targets = np.vstack([ref[2], rng.normal(size=(3, 8))])
    rows, columns = np.tril_indices(8, -1)
    powers = np.where(rows//3 == columns//3, 3, 2)
    def adjacency(x):
        c = np.corrcoef(x, rowvar=False)[rows, columns]
        return np.sign(c)*np.abs(c)**powers
    a = adjacency(ref)
    result = frozen_edge_weights(ref, targets, rows, columns, powers, reference_positions=np.array([2,-1,-1,-1]))
    direct = np.vstack([25*a-24*adjacency(np.delete(ref,2,axis=0)),
                       *[26*adjacency(np.vstack([ref,t]))-25*a for t in targets[1:]]])
    np.testing.assert_allclose(result, direct, atol=1e-12)
    alone = frozen_edge_weights(ref, targets[2:3], rows, columns, powers)
    np.testing.assert_array_equal(alone[0], result[2])
    reordered = frozen_edge_weights(ref, targets[[3,2,1]], rows, columns, powers)
    np.testing.assert_array_equal(reordered[1], result[2])
    with pytest.raises(ValueError, match="absent regions"):
        frozen_edge_weights(ref, np.full((1,8),np.nan), rows, columns, powers)


def test_frozen_rint_ties_interpolation_missing_constants_and_outside_range():
    x=np.array([1,1,2,4,np.nan])
    fit=FrozenRINT.fit(x)
    np.testing.assert_allclose(fit.transform(x[:4]),norm.ppf((rankdata(x[:4])-.5)/4))
    mapped=fit.transform(np.array([-10,1,3,4,100,np.nan]))
    assert mapped[0] == mapped[1] and mapped[3] == mapped[4]
    assert mapped[1] < mapped[2] < mapped[3] and np.isnan(mapped[-1])
    assert np.isnan(FrozenRINT.fit(np.ones(10)).transform(np.array([1,2]))).all()


def test_feature_identities_and_empty_vs_structural_masks():
    f=pd.DataFrame(dict(n_retained_edges=[3,0,0],n_positive_edges=[2,0,0],
        n_negative_edges=[1,0,0],positive_weight_sum=[4.,0,0],negative_weight_magnitude=[1.,0,0],
        n_features=[10]*3,n_possible_edges=[45,45,0]))
    np.testing.assert_allclose(feature_values(f,"connectivity","standard_pruned"),[.6,0,np.nan],equal_nan=True)
    np.testing.assert_allclose(feature_values(f,"negative_density","standard_pruned"),[-1,0,np.nan],equal_nan=True)
    np.testing.assert_allclose(feature_values(f,"connectivity","retained_edge"),[1,np.nan,np.nan],equal_nan=True)
