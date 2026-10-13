#!/usr/bin/env python3
"""Validate and register the compact public MDC sensitivity hot cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

APP = Path(__file__).resolve().parents[1]
ROOT = APP / 'data/muscle_control_l4/mdc_sensitivity'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(data, path):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    variants=manifest['variants']
    for name,nrows,key in [('summary.parquet',155,['normalization_variant','module']),
                           ('resolved.parquet',930,['normalization_variant','module','component'])]:
        path=ROOT/name
        assert path.stat().st_size<95*1024**2
        assert digest(path)==manifest['files'][name]['sha256']
        assert not {'donor','projid','gene_id','Gene ID'}.intersection(pq.read_schema(path).names)
        frame=pd.read_parquet(path)
        assert not frame.duplicated(key).any()
        assert set(frame.normalization_variant)==set(variants)
        assert frame.groupby('normalization_variant').size().eq(nrows).all()
        for variant,part in frame.groupby('normalization_variant'):
            if variant!='original' and not variants[variant]['permutation_significance_complete']:
                fdr_cols=[c for c in part if c.startswith('directional_fdr')]
                assert part[fdr_cols].isna().all().all()
    if manifest['status']=='complete':
        assert manifest['completed_sample_permutations']==manifest['sample_permutations']==200
        assert manifest['completed_gene_permutations']==manifest['gene_permutations']==200
    stamps=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=APP.parents[1]/'out/mdc_muscle_control_l4/public_rollback'/stamps
    backup.mkdir(parents=True,exist_ok=True)
    for name in ['module_set_manifest.json']:
        shutil.copy2(ROOT.parent/name,backup/name)
    shutil.copy2(APP/'data/data_manifest.json',backup/'data_manifest.json')
    record={**manifest,'directory':'mdc_sensitivity','default_variant':'m5_high_no_q90'}
    module_path=ROOT.parent/'module_set_manifest.json'
    module_manifest=json.loads(module_path.read_text())
    module_manifest['mdc_normalization_sensitivity']=record
    for name in ['summary.parquet','resolved.parquet','manifest.json']:
        path=ROOT/name
        module_manifest['files'][f'mdc_sensitivity/{name}']={'bytes':path.stat().st_size,'sha256':digest(path)}
    images=ROOT.parent/'mdc_split_heatmaps/m5_high_no_q90'
    if images.exists():
        assert len(list(images.glob('M*.png')))==155
        for path in images.iterdir():
            if path.is_file():
                relative=path.relative_to(ROOT.parent).as_posix()
                module_manifest['files'][relative]={'bytes':path.stat().st_size,'sha256':digest(path)}
    atomic_json(module_manifest,module_path)
    data_path=APP/'data/data_manifest.json'
    data_manifest=json.loads(data_path.read_text())
    data_manifest['module_sets']['muscle_control_l4']=module_manifest
    data_manifest['files']['muscle_control_l4/module_set_manifest.json']={'bytes':module_path.stat().st_size,'sha256':digest(module_path)}
    for name in ['summary.parquet','resolved.parquet','manifest.json']:
        path=ROOT/name
        data_manifest['files'][f'muscle_control_l4/mdc_sensitivity/{name}']={'bytes':path.stat().st_size,'sha256':digest(path)}
    atomic_json(data_manifest,data_path)
    print(json.dumps({'status':'validated_and_registered','variants':len(variants),
                      'inference_status':manifest['status'],'public_bytes':sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file()),
                      'rollback':str(backup)},indent=2))


if __name__=='__main__':main()
