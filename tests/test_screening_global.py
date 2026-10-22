import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.screening_global import WholeNetworkScreen,apply_global_scopes,require_genome_review


def test_whole_network_two_exact_fdr_scopes_and_topk(tmp_path):
    pytest.importorskip('duckdb')
    rng=np.random.default_rng(42);y=rng.normal(size=40)
    files={};hits=[];raw=[]
    for k,component in enumerate(('TS_AC','CT_AC__DLPFC')):
        x=rng.normal(size=(40,13))
        metadata=pd.DataFrame(dict(test_id=np.arange(k*100,k*100+13),identity=[f'{component}:{i:03d}' for i in range(13)],component=component))
        path=tmp_path/f'{component}.parquet';files[component]=path
        screen=WholeNetworkScreen({'target':y},path,top_k=3)
        for block in (np.arange(5),np.arange(5,9),np.arange(9,13)):
            screen.append(x[:,block],metadata.iloc[block])
        chosen=screen.finish();hits.append(chosen)
        assert len(chosen)<=6
        raw.append(pd.read_parquet(path))
    result=apply_global_scopes(files,pd.concat(hits,ignore_index=True),temporary_directory=tmp_path/'sort')
    original=pd.concat(raw,ignore_index=True).set_index('test_id')
    for r in ('pearson','spearman'):
        direct=pd.Series(false_discovery_control(original[f'target__{r}_p']),index=original.index)
        np.testing.assert_allclose(result[f'{r}_network_fdr'],result.test_id.map(direct))
        assert set(result[f'{r}_network_tested_n'])=={26}
        assert set(result[f'{r}_tested_n'])=={13}


def test_genome_gate_never_infers_approval_from_pilot(tmp_path):
    with pytest.raises(ValueError):
        require_genome_review(tmp_path)
    path=tmp_path/'pilot/m5_resource_report.json';path.parent.mkdir()
    path.write_text(json.dumps({'status':'resource_review_required'}))
    with pytest.raises(PermissionError):
        require_genome_review(tmp_path)
    with pytest.raises(PermissionError):
        require_genome_review(tmp_path,approved_report_hash='wrong')
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert require_genome_review(tmp_path,approved_report_hash=expected)==expected
