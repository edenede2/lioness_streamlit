import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import build_partition_validation_statistics as B
from app_helpers.validation_datasets import dataset


def test_bh_separates_groups_and_excludes_invalid_tests():
    rows=[]
    for component in ["TS_AC","TS_DLPFC"]:
        for group,values in [("All donors",[.01,.02]),("AD",[.9,np.nan])]:
            for module,p in zip([1,2],values):
                rows.append(dict(module=module,component=component,metric_family="connectivity",phenotype="cogn_global",
                    diagnosis_group=group,network_method="standard",**{column:p for column in B.P_COLUMNS}))
    frame=B.correct_statistics(pd.DataFrame(rows),[1,2])
    pooled=frame.loc[frame.diagnosis_group.eq("All donors")]
    np.testing.assert_allclose(pooled.q_spearman_across_modules,.02)
    assert pooled.q_spearman_raw_across_modules_n.eq(2).all()
    ad=frame.loc[frame.diagnosis_group.eq("AD")]
    assert ad.q_spearman_raw_across_modules_n.eq(1).all()
    assert ad.loc[ad.module.eq(1),"q_raw_across_modules"].eq(.9).all()
    with pytest.raises(ValueError,match="complete"): B.correct_statistics(frame.loc[frame.module.eq(1)],[1,2])
    mixed=frame.copy();mixed.loc[mixed.module.eq(2),"network_method"]="control_anchored"
    with pytest.raises(ValueError,match="mixes"): B.correct_statistics(mixed,[1,2])


def test_evaluate_uses_pooled_donors_directly_and_excludes_transform_constants():
    x=np.arange(20,dtype=float); y=np.sin(x)+x/8
    scores=pd.DataFrame(dict(sample_id=[f"S{i}" for i in range(20)],module=188,metric_family="connectivity",
        component="TS_AC",metric_raw=x,metric_asinh=np.arcsinh(x),metric_rint=np.ones(20),metric_scale=1.))
    meta=pd.DataFrame(dict(sample_id=scores.sample_id,diagnosis_group=["Control"]*10+["AD"]*10,cogn_global=y))
    result=B.evaluate(scores,meta,dataset("full_cohort"),"standard","all","standard_pruned","test","unique")
    row=result.loc[result.diagnosis_group.eq("All donors")].iloc[0]
    assert row.n==20 and row.rho_raw==pytest.approx(spearmanr(x,y).statistic)
    assert row.p_spearman_raw==pytest.approx(spearmanr(x,y).pvalue)
    assert np.isnan(row.p_rint) and np.isnan(row.p_spearman_rint)
    assert row.unavailable_reason_rint=="constant transformed score"
    assert set(result.diagnosis_group)=={"All donors","Control","AD"}


def test_vectorized_robustness_matches_independent_legacy_kernels():
    from app_helpers.validation_statistics import robust_rows
    from app_helpers.camel import FrozenRINT
    from build_camel_validation_statistics import robust_row
    rng=np.random.default_rng(42);x=np.round(rng.normal(size=73),1)
    x[4]=np.nan
    frame=pd.DataFrame(dict(metric_raw=x,metric_asinh=np.arcsinh(x),metric_rint=FrozenRINT.fit(x).transform(x),metric_scale=1.))
    for index in range(6):
        values=rng.normal(size=73)+x/3
        values[rng.choice(73,10,replace=False)]=np.nan
        frame[f"target{index}"]=values
    outcomes=[f"target{j}" for j in range(6)]
    vector=robust_rows({},frame,outcomes)
    for outcome,row in zip(outcomes,vector):
        original=robust_row({},frame,outcome)
        for key,value in original.items():
            if isinstance(value,(bool,str)):
                assert row[key]==value
            elif key not in ("phenotype",):
                assert row[key]==pytest.approx(value,rel=1e-10,abs=1e-12,nan_ok=True),key


def test_statistics_row_groups_support_narrow_context_reads(tmp_path):
    import pyarrow.parquet as pq
    frame=pd.DataFrame(dict(module=[1,2,1,2],metric_family=["a","a","b","b"],phenotype="cogn_global",
        p_raw=[.01,.02,.2,np.nan],same_sign=[pd.NA,pd.NA,True,False]))
    path=tmp_path/"statistics.parquet";B.write_statistics(frame,path)
    assert pq.ParquetFile(path).metadata.num_row_groups==2
    result=pd.read_parquet(path,filters=[("metric_family","=","b")])
    assert result.metric_family.eq("b").all() and len(result)==2
