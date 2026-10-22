"""Small end-to-end fixtures test module, partition, and selected-set families."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_gene_edge_screen as runner
from scripts.screening_sources import Sources


def fixture_source():
    rng=np.random.default_rng(42)
    original=[f'd{i:03d}' for i in range(35)]
    validation=[f'v{i:03d}' for i in range(18)]
    all_ids=original+validation
    expression={t:pd.DataFrame(rng.normal(size=(len(all_ids),4)),index=all_ids,columns=['A','B','C','D']) for t in ('AC','DLPFC')}
    expression['DLPFC']=expression['DLPFC'].drop(index=validation[-4:])
    meta=pd.DataFrame({'outcome':rng.normal(size=len(all_ids)),'diagnosis_group':['Control']*16+['AD']*19+['Control']*8+['AD']*10},index=all_ids)
    nodes=pd.DataFrame({'module':[10]*4+[20]*4,'tissue':['AC','AC','DLPFC','DLPFC']*2,'gene':['A','B','A','B','C','D','C','D'],
        'source_node_count':[4]*8,'unique_node_count':[4]*8})
    return Sources('brain',expression,original,validation,original[:16],meta,{d:f'S-{i}' for i,d in enumerate(all_ids)},(),35),nodes


def test_module_partition_and_replication_families(tmp_path,monkeypatch):
    pytest.importorskip('duckdb')
    source,nodes=fixture_source()
    monkeypatch.setattr(runner,'load_sources',lambda dataset:source)
    monkeypatch.setattr(runner,'assignments',lambda p:nodes)
    monkeypatch.setattr(runner,'OUTCOMES',('outcome',))
    monkeypatch.setattr(runner,'PARTITIONS',{'fixture':(2,'unused',('control_anchored',))})
    runner.atomic_json({'status':'passed','identity':'fixture'},tmp_path/'audit.json')
    for module in (10,20):
        result=runner.module_job(tmp_path,'fixture','control_anchored',module,edge_batch=2)
        assert result['tested_edges']==6
    before=sorted((tmp_path/'private/selected_weights/fixture/control_anchored').glob('*.parquet'))
    for p in before:
        scores=pd.read_parquet(p)
        assert not scores.duplicated(['identity','cohort','sample_id']).any()
    runner.finalise_modules(tmp_path,'fixture','control_anchored')
    catalog=pd.read_parquet(tmp_path/'public_bundle/module_edges/fixture/control_anchored.parquet')
    assert len(catalog)==12
    assert catalog.identity.nunique()==12
    assert set(catalog.spearman_tested_n)=={6}
    assert set(catalog.spearman_partition_tested_n)=={12}
    # Every module hit is selected in this tiny fixture; consequently full-search
    # and selected-validation families can be reproduced directly.
    np.testing.assert_allclose(catalog.spearman_partition_fdr,false_discovery_control(catalog.spearman_p))
    for _,group in catalog.groupby('module'):
        np.testing.assert_allclose(group.validation_spearman_selected_fdr,false_discovery_control(group.validation_spearman_p))
        assert set(group.validation_spearman_tested_n)=={6}
    np.testing.assert_allclose(catalog.validation_spearman_partition_selected_fdr,false_discovery_control(catalog.validation_spearman_p))
    cached=runner.module_job(tmp_path,'fixture','control_anchored',10,edge_batch=3)
    assert cached['cached']
    # New batch boundaries must not change scores, tests, selection or p/q.
    other=tmp_path/'resized'
    runner.atomic_json({'status':'passed','identity':'fixture'},other/'audit.json')
    runner.module_job(other,'fixture','control_anchored',10,edge_batch=5)
    a=pd.read_parquet(tmp_path/'checkpoints/module_edges/fixture/control_anchored/M10.parquet').sort_values('identity').reset_index(drop=True)
    b=pd.read_parquet(other/'checkpoints/module_edges/fixture/control_anchored/M10.parquet').sort_values('identity').reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_topk_and_ranking_do_not_change_validation_outcome(tmp_path,monkeypatch):
    source,nodes=fixture_source()
    monkeypatch.setattr(runner,'load_sources',lambda dataset:source)
    monkeypatch.setattr(runner,'assignments',lambda p:nodes)
    monkeypatch.setattr(runner,'OUTCOMES',('outcome',))
    runner.atomic_json({'status':'passed','identity':'fixture'},tmp_path/'a/audit.json')
    runner.module_job(tmp_path/'a','fixture','control_anchored',10,edge_batch=3)
    a=pd.read_parquet(tmp_path/'a/checkpoints/module_edges/fixture/control_anchored/M10.parquet')
    source.metadata.loc[source.validation,'outcome']*= -5
    runner.atomic_json({'status':'passed','identity':'fixture'},tmp_path/'b/audit.json')
    runner.module_job(tmp_path/'b','fixture','control_anchored',10,edge_batch=3)
    b=pd.read_parquet(tmp_path/'b/checkpoints/module_edges/fixture/control_anchored/M10.parquet')
    pd.testing.assert_frame_equal(a[['identity','pearson_rank','spearman_rank','pearson_r','spearman_r']],b[['identity','pearson_rank','spearman_rank','pearson_r','spearman_r']])
    assert not np.array_equal(a.validation_pearson_r,b.validation_pearson_r)


def test_no_eligible_tests_is_explicit_not_an_invented_zero(tmp_path,monkeypatch):
    source,nodes=fixture_source()
    source.metadata['outcome']=np.nan
    monkeypatch.setattr(runner,'load_sources',lambda dataset:source)
    monkeypatch.setattr(runner,'assignments',lambda p:nodes)
    monkeypatch.setattr(runner,'OUTCOMES',('outcome',))
    runner.atomic_json({'status':'passed','identity':'fixture'},tmp_path/'audit.json')
    result=runner.module_job(tmp_path,'fixture','control_anchored',10,edge_batch=3)
    assert result['rows']==0
    assert result['unavailable_reason']=='no_variable_edges_or_eligible_outcomes'


def test_selected_weights_deduplicate_across_partition_catalogs(tmp_path):
    pytest.importorskip('duckdb')
    from scripts.consolidate_screening_weights import consolidate
    runner.atomic_json({'status':'passed','identity':'fixture'},tmp_path/'audit.json')
    runner.atomic_json({'brain_module_edges_status':'complete'},tmp_path/'public_bundle/manifest.json')
    for partition in ('full_cohort','control_derived'):
        path=tmp_path/f'private/selected_weights/{partition}/control_anchored/M10.parquet'
        path.parent.mkdir(parents=True)
        pd.DataFrame(dict(sample_id=['S-fixture'],identity=['AC:G1|DLPFC:G2'],cohort=['original'],component=['CT_AC__DLPFC'],weight=[.3])).to_parquet(path,index=False)
    report=consolidate(tmp_path)
    assert report['files']==1
    out=pd.read_parquet(tmp_path/'private/selected_network_weights/brain/control_anchored/CT_AC__DLPFC.parquet')
    assert len(out)==1 and out.weight.iloc[0]==.3
