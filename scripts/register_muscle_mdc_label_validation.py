#!/usr/bin/env python3
"""Validate/register the new MDC catalog without touching existing result files."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import pyarrow.parquet as pq

APP=Path(__file__).resolve().parents[1]
PUBLIC=APP/'data/muscle_control_l4/mdc_label_validation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(data,path):
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def main():
    manifest=json.loads((PUBLIC/'manifest.json').read_text())
    required={'results.parquet','label_replicates.parquet','balanced_replicates.parquet'}
    assert set(manifest['files'])==required
    for name in required:
        path=PUBLIC/name
        assert path.stat().st_size<95*1024**2
        assert sha(path)==manifest['files'][name]['sha256']
        assert not {'donor','projid','ProjID','gene_id','Gene ID','role_hash'}.intersection(pq.read_schema(path).names)
    result=pd.read_parquet(PUBLIC/'results.parquet')
    assert len(result)==3100 and result.module.nunique()==155
    assert not result.duplicated(['normalization_variant','module','scope']).any()
    for _,family in result.groupby(['normalization_variant','scope']):
        assert len(family)==155
        assert family.tested_module_count.eq(family.label_p.notna().sum()).all()
    if manifest['completed_label_permutations']<manifest['label_permutations']:
        assert result.label_p.isna().all() and result.label_fdr.isna().all()
    for kind,expected in [('label',manifest['completed_label_permutations']),('balanced',manifest['completed_balanced_resamples'])]:
        rep=pd.read_parquet(PUBLIC/f'{kind}_replicates.parquet')
        assert len(rep)==3100*expected
        assert not rep.duplicated(['normalization_variant','module','scope','repetition']).any()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rollback=APP.parents[1]/'out/mdc_muscle_control_l4/public_rollback'/stamp/'label_validation'
    rollback.mkdir(parents=True,exist_ok=True)
    module_path=PUBLIC.parent/'module_set_manifest.json'
    data_path=APP/'data/data_manifest.json'
    shutil.copy2(module_path,rollback/module_path.name);shutil.copy2(data_path,rollback/data_path.name)
    module=json.loads(module_path.read_text());data=json.loads(data_path.read_text())
    before={k:v.copy() for k,v in module['files'].items() if k.startswith('mdc_sensitivity/') or k.startswith('mdc_') and k.endswith('.tsv')}
    module['mdc_label_validation']={**manifest,'directory':'mdc_label_validation','conditional_on_fixed_control_modules':True}
    for name in [*sorted(required),'manifest.json']:
        path=PUBLIC/name;rec={'bytes':path.stat().st_size,'sha256':sha(path)}
        module['files'][f'mdc_label_validation/{name}']=rec
        data['files'][f'muscle_control_l4/mdc_label_validation/{name}']=rec
    for k,v in before.items():
        assert module['files'][k]==v
    atomic_json(module,module_path)
    data['module_sets']['muscle_control_l4']=module
    data['files']['muscle_control_l4/module_set_manifest.json']={'bytes':module_path.stat().st_size,'sha256':sha(module_path)}
    atomic_json(data,data_path)
    print(json.dumps({'status':'validated_and_registered','analysis_status':manifest['status'],'rows':len(result),'rollback':str(rollback)},indent=2))


if __name__=='__main__':main()
