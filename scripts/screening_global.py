"""Shared bounded whole-network search primitives; no automatic execution.

The full-M5 pilot/resource gate is deliberately external to these kernels.
Individual donor networks must be supplied by a validated estimator, not a
beta-only substitute. Every p-value batch is kept until both BH scopes finish.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app_helpers.outcome_screening import correlations, selected_union, top_hits
from scripts.screening_bh import external_bh


def require_genome_review(root, *, approved_report_hash=None):
    """Never infer approval from a passing pilot, file existence, or time."""
    path = Path(root)/'pilot/m5_resource_report.json'
    if not path.is_file():
        raise ValueError('Measured full-M5 resource report is required')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not approved_report_hash or approved_report_hash != digest:
        raise PermissionError('Genome-wide full M5 requires explicit review of the exact measured resource report')
    record = json.loads(path.read_text())
    if record.get('status') != 'resource_review_required':
        raise ValueError('Unresolved numerical/resource pilot state')
    return digest


class WholeNetworkScreen:
    """One component's streaming rankings and all-edge p-values.

    `test_id` must be a network-global canonical int64 edge index, independent
    of components and chunks. Callers must checkpoint the original cursor and
    top-hit state together. Repeated identities or overlapping batches fail.
    """
    def __init__(self, outcomes, pvalue_path, *, top_k=1000):
        self.outcomes = {k:np.asarray(v,float) for k,v in outcomes.items()}
        self.path = Path(pvalue_path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.partial = self.path.with_suffix('.partial')
        self.top_k = top_k; self.top = {k:pd.DataFrame() for k in outcomes}
        self.writer = None; self.count = 0; self.last_id = -1

    def append(self, weights, metadata):
        if len(metadata) != weights.shape[1] or metadata.test_id.duplicated().any():
            raise ValueError('One canonical physical identity per edge is required')
        indices = metadata.test_id.to_numpy(np.int64)
        if len(indices) and (indices[0] <= self.last_id or np.any(np.diff(indices)<=0)):
            raise ValueError('Edge cursor repeated or reordered; use a checkpointed canonical iterator')
        block = {'test_id':indices}
        for outcome,y in self.outcomes.items():
            stats = correlations(weights,y)
            for r in ('pearson','spearman'):
                block[f'{outcome}__{r}_p'] = stats[f'{r}_p'].to_numpy(float)
            candidate = pd.concat([metadata.reset_index(drop=True),stats],axis=1)
            combined = pd.concat([self.top[outcome],candidate],ignore_index=True)
            keep = set(top_hits(combined,self.top_k,'pearson').identity)|set(top_hits(combined,self.top_k,'spearman').identity)
            self.top[outcome] = combined.loc[combined.identity.isin(keep)]
        table = pa.table(block)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.partial,table.schema,compression='zstd')
        self.writer.write_table(table)
        self.count += len(indices)
        if len(indices):
            self.last_id = int(indices[-1])

    def finish(self):
        if self.writer is None:
            raise ValueError('Empty component has no search family')
        self.writer.close();self.writer=None;self.partial.replace(self.path)
        selected=[]
        for outcome,rows in self.top.items():
            hits=selected_union(rows,self.top_k);hits['outcome']=outcome;selected.append(hits)
        return pd.concat(selected,ignore_index=True)


def apply_global_scopes(component_files, selected, *, temporary_directory):
    """Both component and network-global BH BEFORE returning selected results."""
    result=selected.copy()
    all_paths=list(component_files.values())
    for outcome,ix in result.groupby('outcome').groups.items():
        for r in ('pearson','spearman'):
            p=f'{outcome}__{r}_p'
            global_q,n=external_bh(all_paths,p,result.loc[ix,'test_id'],temporary_directory=temporary_directory)
            result.loc[ix,f'{r}_network_fdr']=result.loc[ix,'test_id'].map(global_q)
            result.loc[ix,f'{r}_network_tested_n']=n
            for component,ci in result.loc[ix].groupby('component').groups.items():
                q,nc=external_bh([component_files[component]],p,result.loc[ci,'test_id'],temporary_directory=temporary_directory)
                result.loc[ci,f'{r}_fdr']=result.loc[ci,'test_id'].map(q)
                result.loc[ci,f'{r}_tested_n']=nc
    return result
