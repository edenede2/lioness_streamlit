#!/usr/bin/env python3
"""Publish validated immutable brain images on Drive and compact routing metadata.

Keep legacy images, donor matrices and unrelated Drive index entries untouched.
This command uploads and prepares a public bundle; Git commits/pushes are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
from PIL import Image
from google.auth.transport.requests import AuthorizedSession

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP/'scripts'))
from build_brain_mdc_heatmaps import DEFAULT_OUTPUT, CONFIGS, sha
from update_muscle_drive_index import list_children
from upload_prediction_data import (check_root, create_file, create_folder, update_file,
                                    load_credentials, load_service_account, md5_file)

VERSION='brain_20261014'
DRIVE_ROOT='1dTj5SkLxuDIvII5LayLViqOzjKQt-dos'


def folder_at(upload, reader, parent, name):
    children,_=list_children(reader,parent)
    matches=[item for item in children if item['name']==name]
    if len(matches)>1:raise ValueError(f'Duplicate Drive folder: {name}')
    if matches:
        if matches[0]['mimeType']!='application/vnd.google-apps.folder':
            raise ValueError(f'Expected folder: {name}')
        return matches[0]['id']
    return create_folder(upload,name,parent)['id']


def sync_verified(upload,reader,parent,files):
    existing,_=list_children(reader,parent)
    by_name={item['name']:item for item in existing}
    if len(by_name)!=len(existing):raise ValueError('Duplicate Drive filenames')
    for file in files:
        item=by_name.get(file.name)
        digest=md5_file(file)
        if item and int(item.get('size') or -1)==file.stat().st_size and item.get('md5Checksum')==digest:
            continue
        if item:update_file(upload,file,item['id'])
        else:create_file(upload,file,file.name,parent,digest)
    # Verify visibility/checksums through the app's actual read-only identity.
    observed,_=list_children(reader,parent)
    by_name={item['name']:item for item in observed}
    result={}
    for file in files:
        item=by_name.get(file.name)
        if not item or int(item.get('size') or -1)!=file.stat().st_size or item.get('md5Checksum')!=md5_file(file):
            raise RuntimeError(f'App readback/checksum failed: {file.name}')
        result[file.name]=dict(id=item['id'],size=int(item['size']),md5=item['md5Checksum'],modified_time=item.get('modifiedTime'))
    return result


def verified_files(source,key):
    rows=pd.read_csv(source/key/'manifest.tsv',sep='\t')
    stats=pd.read_csv(CONFIGS[key]['data']/'mdc_ad_vs_control_summary.tsv',sep='\t')
    assert len(rows)==CONFIGS[key]['count'] and set(rows.module)==set(stats.module)
    assert rows.module.is_unique
    files=[]
    for row in rows.itertuples(index=False):
        path=source/key/row.png
        assert path.name==f'M{int(row.module)}.png' and path.parent==source/key
        assert sha(path)==row.png_sha256 and path.stat().st_size<95*1024**2
        with Image.open(path) as im:
            assert im.size==(3200,3200)
            im.verify()
        files.append(path)
    for name in ('manifest.tsv','AD_Control_MDC_split_triangles.pdf'):
        files.append(source/key/name)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--root-folder-id',default=DRIVE_ROOT)
    args=parser.parse_args();source=args.source.resolve()
    run=json.loads((source/'manifest.json').read_text())
    assert run['status']=='complete'
    catalogs={key:verified_files(source,key) for key in CONFIGS}
    for key in CONFIGS:
        assert sha(CONFIGS[key]['data']/'mdc_ad_vs_control_summary.tsv')==run['datasets'][key]['source_statistics_sha256']
        assert sha(source/key/'AD_Control_MDC_split_triangles.pdf')==run['datasets'][key]['pdf_sha256']
    oauth=SimpleNamespace(credentials=None,oauth_client=APP/'client_secret_688842367267-8o25rcsq6jvrb1go8d0gsr1o304v1knq.apps.googleusercontent.com.json',
                          oauth_token=APP/'google-drive-oauth-token.json',oauth_port=8765,oauth_no_browser=True)
    upload=AuthorizedSession(load_credentials(oauth))
    reader=AuthorizedSession(load_service_account(APP/'lionessexplorer-05e37b3ea2a6.json','https://www.googleapis.com/auth/drive.readonly'))
    check_root(upload,args.root_folder_id,False)
    previous=json.loads((APP/'drive_file_index.json').read_text())
    assert previous['root_folder_id']==args.root_folder_id
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rollback=source/'publication_rollback'/stamp
    rollback.mkdir(parents=True,exist_ok=False)
    shutil.copy2(APP/'drive_file_index.json',rollback/'drive_file_index.json')
    additions={}
    prepared=[]
    for key,files in catalogs.items():
        base=CONFIGS[key]['data']
        prefix=base.relative_to(APP/'data').as_posix()
        parts=[] if prefix=='.' else list(Path(prefix).parts)
        parent=args.root_folder_id
        for part in [*parts,'mdc_split_heatmaps']:
            parent=folder_at(upload,reader,parent,part)
        remote=folder_at(upload,reader,parent,VERSION)
        inventory=sync_verified(upload,reader,remote,files)
        metadata=dict(schema_version=1,status='complete',module_set=key,modules=CONFIGS[key]['count'],
                      image_directory=VERSION,upper_triangle='AD',lower_triangle='Control',
                      width=3200,height=3200,dpi=300,display='Absolute signedAlt adjacency',
                      color_scale='Shared within each module; pooled 99th-percentile display cap only',
                      beta_ts=3,beta_ct=2,mdc_statistics_unchanged=True,
                      source_statistics_sha256=run['datasets'][key]['source_statistics_sha256'],
                      released_utc=datetime.now(timezone.utc).isoformat())
        meta_path=rollback/key/'catalog.json'
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
        meta_inventory=sync_verified(upload,reader,parent,[meta_path])
        root_relative=Path(*parts,'mdc_split_heatmaps').as_posix()
        additions.update({f'{root_relative}/{VERSION}/{name}':item for name,item in inventory.items()})
        additions[f'{root_relative}/catalog.json']=meta_inventory['catalog.json']
        prepared.append((key,files,meta_path))
        print(f'{key}: verified {len(files)} remote images/catalog files through the app reader',flush=True)
    # Only expose the new release after both complete catalogs have read back.
    for key,files,meta_path in prepared:
        parent=CONFIGS[key]['data']/'mdc_split_heatmaps'
        old=parent/'catalog.json'
        if old.exists():shutil.copy2(old,rollback/f'{key}_previous_catalog.json')
        target=parent/VERSION
        target.mkdir(parents=True,exist_ok=True)
        for file in files:shutil.copy2(file,target/file.name)
        temporary=parent/'catalog.json.partial'
        shutil.copy2(meta_path,temporary);temporary.replace(old)
    index={**previous,'files':{**previous['files'],**additions},'generated_at_utc':datetime.now(timezone.utc).isoformat(),
           'refresh_scope':'Verified brain split-triangle heatmap release; unrelated entries preserved'}
    index['file_count']=len(index['files'])
    index['total_bytes']=sum(int(item.get('size') or 0) for item in index['files'].values())
    temporary=APP/'drive_file_index.json.partial'
    temporary.write_text(json.dumps(index,indent=2)+'\n');temporary.replace(APP/'drive_file_index.json')
    report=dict(status='verified_uploaded',modules=340,indexed_files=len(additions),
                privacy='No donor IDs, expression or private adjacency caches uploaded',
                previous_figures_preserved=True,rollback=str(rollback))
    (source/'publication_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
