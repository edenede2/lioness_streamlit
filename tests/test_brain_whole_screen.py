from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from app_helpers.outcome_screening import correlations, signed_alt_edges
from scripts.screening_brain_whole import (
    FrozenSignedAlt, FiniteOutcomeCorrelator, TopUnion, component_indices,
    edge_ids, edge_nodes, scoped_external_bh,
)


def test_random_access_triangle_rectangle_and_global_identity():
    for n in (2,3,20,17306):
        total=n*(n-1)//2
        ids=np.unique(np.r_[np.arange(min(total,100)),total-1])
        left,right=edge_nodes(ids,n)
        np.testing.assert_array_equal(edge_ids(left,right,n),ids)
        if n<100:
            a,b=component_indices(0,total,n)
            expected=np.triu_indices(n,1)
            np.testing.assert_array_equal(a,expected[0]);np.testing.assert_array_equal(b,expected[1])
    a,b=component_indices(2,18,4,7)
    np.testing.assert_array_equal(a*7+b,np.arange(2,18))
    with pytest.raises(ValueError): component_indices(0,7,3)


@pytest.mark.parametrize('power',[2,3])
def test_frozen_network_matches_original_and_direct_correlation(power):
    rng=np.random.default_rng(42);reference=rng.normal(size=(30,9));targets=np.vstack([reference[:2],rng.normal(size=(4,9))])
    roles=np.array([0,1,-1,-1,-1,-1]);left,right=np.triu_indices(9,1)
    network=FrozenSignedAlt(reference,targets,roles)
    weights=network.weights(left,right,power)
    np.testing.assert_allclose(weights,signed_alt_edges(reference,targets,left,right,power,reference_positions=roles),atol=1e-12,rtol=0)
    base=np.corrcoef(reference,rowvar=False)[left,right]
    base=np.sign(base)*np.abs(base)**power
    for i,row in enumerate(targets):
        changed=np.delete(reference,roles[i],axis=0) if roles[i]>=0 else np.vstack([reference,row])
        r=np.corrcoef(changed,rowvar=False)[left,right];adj=np.sign(r)*np.abs(r)**power
        direct=30*base-29*adj if roles[i]>=0 else 31*adj-30*base
        np.testing.assert_allclose(weights[i],direct,atol=1e-12,rtol=0)
    separate=FrozenSignedAlt(reference,targets[2:3]).weights(left,right,power)
    np.testing.assert_array_equal(separate[0],weights[2])
    reordered=FrozenSignedAlt(reference,targets[::-1],roles[::-1]).weights(left,right,power)
    np.testing.assert_array_equal(reordered[::-1],weights)


def test_shared_rank_kernel_exact_missingness_ties_constants_and_minimum_n():
    rng=np.random.default_rng(1042);x=rng.normal(size=(45,55));x[:,1]=0;x[:,2]=np.round(x[:,2])
    y=rng.normal(size=45);z=y.copy();z[::3]=np.nan
    outcomes={'full':y,'same_mask':-2*y,'missing':z,'constant':np.ones(45),'sparse':np.r_[np.ones(4),np.full(41,np.nan)]}
    result=FiniteOutcomeCorrelator(outcomes).calculate(x)
    for name,outcome in outcomes.items():
        direct=correlations(x,outcome)
        for column in result[name]:
            np.testing.assert_allclose(result[name][column],direct[column],atol=2e-13,rtol=1e-12,equal_nan=True)
    with pytest.raises(ValueError): FiniteOutcomeCorrelator(outcomes).calculate(x*np.nan)


def test_top_union_exact_boundary_ties_and_resume_batch_invariance():
    ids=np.arange(1100);rng=np.random.default_rng(42)
    r=np.round(rng.normal(size=len(ids)),1)
    stats=dict(n=np.full(len(ids),45),pearson_r=r,pearson_p=np.ones(len(ids)),spearman_r=-r,spearman_p=np.ones(len(ids)))
    expected=ids[np.lexsort((ids,-np.abs(r)))[:10]]
    first=TopUnion(['y'],10);second=TopUnion(['y'],10)
    first.update(ids,{'y':stats})
    for ix in np.array_split(np.arange(len(ids)),17):
        second.update(ids[ix],{'y':{k:v[ix] for k,v in stats.items()}})
    for selected in (first,second):
        frame=selected.frame().sort_values('pearson_rank')
        np.testing.assert_array_equal(frame.test_id,expected)
    pd.testing.assert_frame_equal(first.frame().sort_values('test_id').reset_index(drop=True),
                                  second.frame().sort_values('test_id').reset_index(drop=True))


def test_single_sort_exact_network_and_component_bh_with_ties_and_missing(tmp_path):
    pytest.importorskip('duckdb')
    rng=np.random.default_rng(42);files=[];frames=[]
    for scope,n in enumerate((151,72,39)):
        frame=pd.DataFrame(dict(test_id=np.arange(n)+1000*scope,scope_id=np.full(n,scope),p=rng.uniform(size=n)))
        frame.loc[:4,'p']=.01;frame.loc[6,'p']=np.nan
        path=tmp_path/f'{scope}.parquet';frame.to_parquet(path);files.append(path);frames.append(frame)
    all_rows=pd.concat(frames,ignore_index=True).set_index('test_id')
    ids=all_rows.loc[all_rows.p.notna()].index[::3]
    result,counts=scoped_external_bh(files,'p',ids,temporary_directory=tmp_path/'sort',batch_size=7,memory_gib=.0625)
    direct=pd.Series(np.nan,index=all_rows.index)
    direct.loc[all_rows.p.notna()]=false_discovery_control(all_rows.p.dropna())
    np.testing.assert_allclose(result.network_fdr,result.test_id.map(direct),atol=1e-15)
    assert counts['network_tested_n']==len(all_rows)-3
    for scope,frame in all_rows.groupby('scope_id'):
        local=pd.Series(false_discovery_control(frame.p.dropna()),index=frame.p.dropna().index)
        part=result.loc[result.scope_id.eq(scope)]
        np.testing.assert_allclose(part.component_fdr,part.test_id.map(local),atol=1e-15)
        assert counts['component_tested_n'][scope]==len(frame)-1


def miniature_context():
    from types import SimpleNamespace
    from scripts.screening_sources import component
    rng=np.random.default_rng(42)
    old=[f'old-{i}' for i in range(15)];new=[f'new-{i}' for i in range(12)]
    tissues=['AC','DLPFC','PCG'];genes=['APOE','MAPT','TREM2']
    expression={t:pd.DataFrame(rng.normal(size=(27,3)),index=old+new,columns=genes) for t in tissues}
    matrix=np.concatenate([expression[t].loc[old].to_numpy() for t in tissues],axis=1)
    metadata=pd.DataFrame({'diagnosis_group':['Control']*6+['MCI']*4+['AD']*5+['Control']*4+['MCI']*4+['AD']*4,
        'cogn_global':rng.normal(size=27),'gpath':rng.normal(size=27)},index=old+new)
    source=SimpleNamespace(original=old,metadata=metadata,expression=expression,
        mapping={d:f'S-{i:03d}' for i,d in enumerate(old+new)},donors=lambda required,validation=False:new if validation else old)
    nodes=[(t,g) for t in tissues for g in genes];scopes=[]
    for i,a in enumerate(tissues):
        for j in range(i,3):
            b=tissues[j]
            scopes.append(dict(scope_id=len(scopes),component=component(a,b),tissues=[a] if i==j else [a,b],
                first=a,second=b,left_offset=i*3,right_offset=j*3,left_n=3,right_n=None if i==j else 3,
                edges=3 if i==j else 9,power=3 if i==j else 2))
    outcomes={o:metadata.loc[old,o].to_numpy() for o in ('cogn_global','gpath')}
    return dict(source=source,genes={t:genes for t in tissues},nodes=nodes,original=old,
        reference_ids=old[:6],reference=matrix[:6],method='control_anchored',
        network=FrozenSignedAlt(matrix[:6],matrix,np.r_[np.arange(6),np.full(9,-1)]),
        correlator=FiniteOutcomeCorrelator(outcomes),outcomes=outcomes,scopes=scopes)


def test_miniature_complete_pipeline_checkpoints_fdr_validation_and_ui(tmp_path,monkeypatch):
    import hashlib,json
    import scripts.run_brain_whole_screen as runner
    from scripts.run_gene_edge_screen import public_catalog,atomic_json
    from streamlit.testing.v1 import AppTest
    ctx=miniature_context()
    monkeypatch.setattr(runner,'OUTCOMES',tuple(ctx['outcomes']))
    monkeypatch.setattr(runner,'TOP_K',2);monkeypatch.setattr(runner,'SHARD',4);monkeypatch.setattr(runner,'BATCH',3)
    monkeypatch.setattr(runner,'_CONTEXT',ctx)
    root=tmp_path/'analysis';root.mkdir();method='control_anchored';identity='fixture-v1'
    records=[]
    for scope in ctx['scopes']:
        for start in range(0,scope['edges'],runner.SHARD):
            stop=min(scope['edges'],start+runner.SHARD)
            record=runner.shard_job(root,method,scope,start,stop,identity)
            assert runner.verified_shard(root,method,scope,start,stop,identity)==record
            records.append(record)
    assert sum(r['edges'] for r in records)==36
    hits,files=runner.merge_hits(root,ctx,identity)
    result=runner.fdr(root,method,hits,files,identity)
    repeated=runner.fdr(root,method,hits,files,identity)
    pd.testing.assert_frame_equal(result,repeated)
    validated=runner.validate_hits(root,ctx,result)
    assert runner.reconcile_results(validated)['status']=='passed'
    assert validated.validation_n.eq(12).all()
    assert validated.pearson_network_tested_n.eq(36).all()
    assert validated.spearman_network_tested_n.eq(36).all()
    # Actual Streamlit whole-network routing and both FDR scopes/cohorts.
    public=public_catalog(validated)
    part=public.loc[public.component.eq('TS_AC')]
    deploy=tmp_path/'deploy';relative=Path('whole_network/brain/control_anchored__TS_AC.parquet')
    path=deploy/relative;path.parent.mkdir(parents=True);part.to_parquet(path,index=False)
    atomic_json(dict(schema_version=1,outcomes=list(ctx['outcomes']),files=[dict(path=relative.as_posix(),
        status='complete',rows=len(part),sha256=hashlib.sha256(path.read_bytes()).hexdigest())]),deploy/'manifest.json')
    app=AppTest.from_string(f'from app_helpers.outcome_screening_ui import render_view\nrender_view({str(deploy)!r})',default_timeout=30).run()
    next(w for w in app.radio if w.label=='Screening view').set_value('whole_network').run()
    assert not app.exception and len(app.get('plotly_chart'))==2
    for cohort in ('original','validation'):
        next(w for w in app.selectbox if w.label=='Displayed statistics').set_value(cohort).run()
        for scope in ('component','network'):
            next(w for w in app.radio if w.label=='FDR scope').set_value(scope).run()
            assert not app.exception and len(app.get('plotly_chart'))==2
    with pytest.raises(ValueError,match='identity'):
        runner.verified_shard(root,method,ctx['scopes'][0],0,3,'other-engine')
    # Cleanup is gated and removes only files created for this exact method.
    with pytest.raises(FileNotFoundError): runner.recycle_pvalues(root,method,ctx,identity)
    atomic_json(dict(status='passed',engine_identity=identity),root/'qa/whole_network_brain_control_anchored.json')
    runner.recycle_pvalues(root,method,ctx,identity)
    assert not any(p.exists() for p in files)
    assert (root/'whole_network_brain/recycled_control_anchored.json').is_file()
