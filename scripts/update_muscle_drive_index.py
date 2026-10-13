#!/usr/bin/env python3
"""Refresh only the new muscle Drive prefix, preserving unrelated index entries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from scripts.build_drive_index import DRIVE_FOLDER_MIME, DRIVE_LIST_URL, md5_file, parquet_predicates

PREFIX='muscle_all_donors_l4'


def list_children(session, parent):
    files=[];token=None;requests=0
    while True:
        params={'q':f"'{parent}' in parents and trashed=false",'pageSize':1000,
                'fields':'nextPageToken,files(id,name,mimeType,size,md5Checksum,modifiedTime)',
                'supportsAllDrives':'true','includeItemsFromAllDrives':'true'}
        if token:params['pageToken']=token
        response=session.get(DRIVE_LIST_URL,params=params,timeout=60)
        response.raise_for_status();payload=response.json();requests+=1
        files.extend(payload.get('files',[]));token=payload.get('nextPageToken')
        if not token:return files,requests


def scoped_inventory(session, root):
    response=session.get(DRIVE_LIST_URL,params={
        'q':f"'{root}' in parents and name='{PREFIX}' and trashed=false",
        'fields':'files(id,name,mimeType)','pageSize':100},timeout=60)
    response.raise_for_status();folders=response.json().get('files',[])
    if len(folders)!=1 or folders[0]['mimeType']!=DRIVE_FOLDER_MIME:
        raise RuntimeError('New muscle folder is not uniquely visible to the app service account')
    pending=[(folders[0]['id'],PREFIX)];indexed={};requests=1
    while pending:
        parent,prefix=pending.pop()
        children,n=list_children(session,parent);requests+=n
        seen=set()
        for item in children:
            name=item['name']
            if name in seen or name in {'.','..'} or '/' in name:
                raise RuntimeError('Duplicate or unsafe path in new muscle Drive folder')
            seen.add(name);relative=f'{prefix}/{name}'
            if item['mimeType']==DRIVE_FOLDER_MIME:pending.append((item['id'],relative));continue
            indexed[relative]=dict(id=item['id'],size=int(item['size']),md5=item.get('md5Checksum'),modified_time=item.get('modifiedTime'))
    return indexed,requests


def merge_index(previous, scoped, root, requests):
    if previous['root_folder_id']!=root:raise ValueError('Cannot merge indexes from different Drive roots')
    if any(not key.startswith(PREFIX+'/') for key in scoped):raise ValueError('Scoped inventory escaped the requested dataset')
    files={k:v for k,v in previous['files'].items() if not k.startswith(PREFIX+'/')}
    files.update(scoped)
    return {**previous,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'files':dict(sorted(files.items())),'file_count':len(files),
            'total_bytes':sum(int(item.get('size') or 0) for item in files.values()),
            'drive_list_requests':requests,'validated_prefixes':[PREFIX],
            'refresh_scope':'Incremental muscle prefix; unrelated entries retained from their prior verified snapshot'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials',type=Path,required=True)
    p.add_argument('--root-folder-id',required=True)
    p.add_argument('--local-data',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,default=Path('drive_file_index.json'))
    args=p.parse_args()
    previous=json.loads(args.output.read_text())
    credential=service_account.Credentials.from_service_account_info(json.loads(args.credentials.read_text()),scopes=['https://www.googleapis.com/auth/drive.readonly'])
    session=AuthorizedSession(credential)
    local={str(file.relative_to(args.local_data)):file for file in (args.local_data/PREFIX).rglob('*') if file.is_file()}
    requests=0
    for attempt in range(3):
        scoped,n=scoped_inventory(session,args.root_folder_id);requests+=n
        missing=set(local)-set(scoped);extra=set(scoped)-set(local)
        if not missing and not extra:break
        if attempt==2:raise RuntimeError(f'Drive visibility incomplete: {len(missing)} missing, {len(extra)} extra files')
        time.sleep(10)
    for key,path in local.items():
        if path.stat().st_size!=scoped[key]['size'] or md5_file(path)!=scoped[key]['md5']:
            raise RuntimeError(f'Drive checksum mismatch: {key}')
        predicates=parquet_predicates(path)
        if predicates:scoped[key]['parquet_predicates']=predicates
    result=merge_index(previous,scoped,args.root_folder_id,requests)
    temporary=args.output.with_name(args.output.name+'.partial')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps(dict(status='validated',prefix=PREFIX,files=len(scoped),drive_list_requests=requests)))


if __name__=='__main__':main()
