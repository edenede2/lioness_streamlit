#!/usr/bin/env python3
"""Deduplicate selected weights by physical network, never publish donor rows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

APP=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(APP),str(APP/'scripts')]
from scripts.run_gene_edge_screen import RUN,atomic_json,require_audit
from scripts.screening_sources import sha
from scripts.screening_bh import DUCKDB_VERSION


def consolidate(root=RUN):
    import duckdb
    if duckdb.__version__ != DUCKDB_VERSION:
        raise RuntimeError('Pinned private DuckDB is required')
    root=Path(root);manifest=json.loads((root/'public_bundle/manifest.json').read_text())
    if manifest.get('brain_module_edges_status')!='complete':
        raise ValueError('Wait for all brain catalogs before cross-partition deduplication')
    record=[]
    for method in ('standard','control_anchored'):
        paths=sorted((root/'private/selected_weights').glob(f'*/{method}/M*.parquet'))
        if not paths:
            continue
        output=root/'private/selected_network_weights/brain'/method
        output.mkdir(parents=True,exist_ok=True)
        connection=duckdb.connect()
        try:
            connection.execute("SET memory_limit='4GB'")
            connection.execute("SET threads=1")
            connection.execute('SET temp_directory=?',[str(root/'scratch/duckdb')])
            connection.read_parquet([str(p) for p in paths]).create_view('weights')
            # Tiny floating summation-order differences between two module
            # contexts are permitted, never material disagreement.
            bad=connection.execute('SELECT count(*) FROM (SELECT max(weight)-min(weight) AS gap FROM weights GROUP BY sample_id,identity,cohort,component HAVING gap>1e-10)').fetchone()[0]
            if bad:
                raise ValueError('Same physical brain edge has conflicting donor weights')
            for component, in connection.execute('SELECT DISTINCT component FROM weights ORDER BY component').fetchall():
                temporary=output/f'{component}.partial'
                # Sorted min is deterministic. No averaging across donors,
                # cohorts, methods or independent normalization profiles.
                query="SELECT sample_id,identity,cohort,component,min(weight) AS weight FROM weights WHERE component=? GROUP BY sample_id,identity,cohort,component ORDER BY cohort,sample_id,identity"
                connection.sql(query,params=[component]).write_parquet(str(temporary),compression='zstd',row_group_size=32768)
                target=temporary.with_suffix('.parquet');temporary.replace(target)
                record.append(dict(method=method,component=component,bytes=target.stat().st_size,sha256=sha(target)))
        finally:
            connection.close()
    report=dict(status='complete',audit_identity=require_audit(root),files=record,publication='private_only_not_authorized',
        network_identity='Brain signedAlt TS3 CT2; original frozen reference; same physical tissue-gene edge',
        intermediate_module_caches='retained for resumable checkpoints; consolidated files are canonical selected-weight outputs')
    atomic_json(report,root/'private/selected_network_weights/manifest.json')
    return dict(status='complete',files=len(record),bytes=sum(e['bytes'] for e in record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=RUN)
    print(json.dumps(consolidate(parser.parse_args().root)))
