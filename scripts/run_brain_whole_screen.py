#!/usr/bin/env python3
"""Pilot-gated, resumable all-edge brain screening. Never launches motor M5.

The user's run authorization permits automatic brain expansion only after the
technical pilot passes. No Git push, Drive upload, or donor publication occurs.
"""
from __future__ import annotations
import os
for _thread_option in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP/"scripts")]
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import false_discovery_control, pearsonr, spearmanr
from app_helpers.gene_symbols import gene_symbol
from app_helpers.outcome_screening import bh
from scripts.run_gene_edge_screen import (
    RUN, atomic_json, atomic_parquet, public_catalog, require_audit,
    resource_guard, stats_for_selected,
)
from scripts.screening_sources import BRAINS, OUTCOMES, component, load_sources, sha
from scripts.screening_brain_whole import (
    FrozenSignedAlt, FiniteOutcomeCorrelator, TopUnion, component_indices,
    edge_ids, edge_nodes, scoped_external_bh,
)

BATCH = 8192
SHARD = 262144
TOP_K = 1000
METHODS = ("control_anchored", "standard")
STAT_COLUMNS = ("n","pearson_r","pearson_p","spearman_r","spearman_p")
_CONTEXT = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def workdir(root):
    return Path(root)/"whole_network_brain"


def status(root, phase, **fields):
    record = dict(phase=phase, updated_at=stamp(), **fields)
    atomic_json(record, workdir(root)/"status.json")
    print(json.dumps(record), flush=True)


def make_context(method):
    if method not in METHODS:
        raise ValueError("Unsupported brain method")
    source = load_sources("brain")
    genes = {t:sorted(source.expression[t].columns) for t in BRAINS}
    ids = source.original
    arrays = [source.expression[t].loc[ids,genes[t]].to_numpy(float) for t in BRAINS]
    matrix = np.concatenate(arrays, axis=1)
    nodes = [(t,g) for t in BRAINS for g in genes[t]]
    if len(ids) != 450 or len(source.controls) != 164 or len(nodes) != 51918:
        raise ValueError("Whole-brain source dimensions changed")
    reference_ids = ids if method == "standard" else source.controls
    lookup = {d:i for i,d in enumerate(ids)}
    reference = matrix[[lookup[d] for d in reference_ids]]
    roles = {d:i for i,d in enumerate(reference_ids)}
    positions = np.array([roles.get(d,-1) for d in ids])
    offsets = {}; cursor = 0
    for t in BRAINS:
        offsets[t] = cursor; cursor += len(genes[t])
    scopes = []
    for ti,first in enumerate(BRAINS):
        for second in BRAINS[ti:]:
            n,m = len(genes[first]),len(genes[second])
            scopes.append(dict(scope_id=len(scopes),component=component(first,second),
                tissues=[first] if first == second else [first,second],
                first=first,second=second,left_offset=offsets[first],right_offset=offsets[second],
                left_n=n,right_n=None if first == second else m,
                edges=n*(n-1)//2 if first == second else n*m,
                power=3 if first == second else 2))
    outcomes = {o:source.metadata.loc[ids,o].to_numpy(float) for o in OUTCOMES}
    return dict(source=source, genes=genes, nodes=nodes, original=ids,
        reference_ids=reference_ids, reference=reference, method=method,
        network=FrozenSignedAlt(reference,matrix,positions),
        correlator=FiniteOutcomeCorrelator(outcomes),outcomes=outcomes,scopes=scopes)


def scope_edges(scope, start, stop):
    a,b = component_indices(start,stop,scope["left_n"],scope["right_n"])
    return a+scope["left_offset"],b+scope["right_offset"]


def paths(root,method,scope,start):
    base = Path(root)/"scratch/whole_network_brain"/method/scope["component"]/f"part_{start:012d}"
    return base.with_suffix(".parquet"),base.with_suffix(".hits.parquet"),base.with_suffix(".json")


def shard_job(root,method,scope,start,stop,identity):
    ctx = _CONTEXT
    if ctx is None or ctx["method"] != method:
        raise ValueError("Numerical worker context differs from task")
    pfile,hfile,marker = paths(root,method,scope,start)
    pfile.parent.mkdir(parents=True,exist_ok=True)
    partial = pfile.with_suffix(".partial")
    started = time.monotonic(); top = TopUnion(OUTCOMES,TOP_K); writer = None; peak_combined = 0.
    try:
        for cursor in range(start,stop,BATCH):
            end = min(stop,cursor+BATCH)
            left,right = scope_edges(scope,cursor,end)
            ids = edge_ids(left,right,len(ctx["nodes"]))
            weights = ctx["network"].weights(left,right,scope["power"])
            stats = ctx["correlator"].calculate(weights)
            top.update(ids,stats)
            block = dict(test_id=ids,scope_id=np.full(len(ids),scope["scope_id"],np.uint8))
            for outcome,values in stats.items():
                for r in ("pearson","spearman"):
                    block[f"{outcome}__{r}_p"] = values[f"{r}_p"]
            table = pa.table(block)
            if writer is None:
                writer = pq.ParquetWriter(partial,table.schema,compression="zstd")
            writer.write_table(table,row_group_size=BATCH)
            if (cursor-start) % (BATCH*4) == 0:
                peak_combined = max(peak_combined,resource_guard(root)["current_rss_bytes"]/1024**3)
        if writer is None:
            raise ValueError("Empty whole-network shard")
        writer.close(); writer = None; partial.replace(pfile)
        atomic_parquet(top.frame(),hfile)
        record = dict(status="complete",engine_identity=identity,method=method,
            component=scope["component"],scope_id=scope["scope_id"],start=start,stop=stop,
            edges=stop-start,seconds=time.monotonic()-started,pvalue_bytes=pfile.stat().st_size,
            pvalue_sha256=sha(pfile),hits_sha256=sha(hfile),
            peak_combined_rss_gib=peak_combined,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        atomic_json(record,marker)
        return record
    finally:
        if writer is not None:
            writer.close()


def verified_shard(root,method,scope,start,stop,identity):
    pfile,hfile,marker = paths(root,method,scope,start)
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text())
    if (record.get("engine_identity") != identity or record.get("status") != "complete"
            or record.get("start") != start or record.get("stop") != stop):
        raise ValueError("Checkpoint identity/cursor changed")
    if sha(pfile) != record["pvalue_sha256"] or sha(hfile) != record["hits_sha256"]:
        raise ValueError("Completed checkpoint hash mismatch")
    if pq.ParquetFile(pfile).metadata.num_rows != stop-start:
        raise ValueError("Checkpoint edge count mismatch")
    return record


def run_tasks(root,method,tasks,identity,workers,phase):
    completed = []; pending = []
    for scope,start,stop in tasks:
        cached = verified_shard(root,method,scope,start,stop,identity)
        if cached:
            completed.append(cached)
        else:
            pending.append((scope,start,stop))
    done = sum(r["edges"] for r in completed)
    total = sum(stop-start for _,start,stop in tasks)
    status(root,phase,method=method,completed_edges=done,total_edges=total,pending_shards=len(pending))
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        iterator = iter(pending); active = {}
        def submit():
            task = next(iterator,None)
            if task is not None:
                scope,start,stop = task
                active[pool.submit(shard_job,str(root),method,scope,start,stop,identity)] = task
        for _ in range(workers): submit()
        while active:
            finished,_ = wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                active.pop(future)
                record = future.result();completed.append(record);done += record["edges"]
                status(root,phase,method=method,completed_edges=done,total_edges=total,
                    last_component=record["component"],last_shard_seconds=record["seconds"])
                submit()
    return completed


def numerical_pilot(ctx):
    rng = np.random.default_rng(42)
    max_edge_error = max_stat_error = 0.
    source = ctx["source"]
    original_roles = [next(i for i,d in enumerate(ctx["original"])
        if source.metadata.loc[d,"diagnosis_group"] == group) for group in ("Control","MCI","AD")]
    for scope in ctx["scopes"]:
        chosen = np.unique(rng.integers(0,scope["edges"],size=24))
        if scope["right_n"] is None:
            local_a,local_b = edge_nodes(chosen,scope["left_n"])
        else:
            local_a,local_b = chosen//scope["right_n"],chosen % scope["right_n"]
        left,right = local_a+scope["left_offset"],local_b+scope["right_offset"]
        weights = ctx["network"].weights(left,right,scope["power"])
        for i in original_roles:
            for j in range(min(8,len(left))):
                reference = ctx["reference"][:,[left[j],right[j]]]
                row = ctx["network"].delta[i,[left[j],right[j]]]+ctx["network"].mean[[left[j],right[j]]]
                position = ctx["network"].positions[i]
                changed = np.delete(reference,position,axis=0) if position >= 0 else np.vstack([reference,row])
                def adjacency(matrix):
                    correlation = np.nan_to_num(np.corrcoef(matrix,rowvar=False)[0,1])
                    return np.sign(correlation)*abs(np.clip(correlation,-1,1))**scope["power"]
                n = len(reference)
                direct = n*adjacency(reference)-(n-1)*adjacency(changed) if position >= 0 else (n+1)*adjacency(changed)-n*adjacency(reference)
                max_edge_error = max(max_edge_error,abs(direct-weights[i,j]))
        stats = ctx["correlator"].calculate(weights)
        for outcome,y in ctx["outcomes"].items():
            mask = np.isfinite(y)
            for j in range(min(4,len(left))):
                if mask.sum()<10 or np.ptp(weights[mask,j]) == 0 or np.ptp(y[mask]) == 0:
                    continue
                for r,fn in (("pearson",pearsonr),("spearman",spearmanr)):
                    direct = fn(weights[mask,j],y[mask])
                    max_stat_error = max(max_stat_error,abs(direct.statistic-stats[outcome][f"{r}_r"][j]),
                                          abs(direct.pvalue-stats[outcome][f"{r}_p"][j]))
        # Actual validation Control: add-one only, independent of other targets.
        eligible = source.donors(tuple(scope["tissues"]),True)
        donor = next(d for d in eligible if source.metadata.loc[d,"diagnosis_group"] == "Control")
        used = np.unique(np.r_[left,right]); lookup = {int(index):i for i,index in enumerate(used)}
        target = np.array([[source.expression[ctx["nodes"][i][0]].loc[donor,ctx["nodes"][i][1]] for i in used]])
        reference = ctx["reference"][:,used]
        local_left = np.array([lookup[int(i)] for i in left]);local_right = np.array([lookup[int(i)] for i in right])
        one = FrozenSignedAlt(reference,target).weights(local_left,local_right,scope["power"])
        many = FrozenSignedAlt(reference,np.vstack([target,target*1.01])).weights(local_left,local_right,scope["power"])
        if not np.array_equal(one[0],many[0]):
            raise ValueError("Validation donor independence failed")
        for j in range(min(4,len(left))):
            rr = reference[:,[local_left[j],local_right[j]]]
            xx = target[:,[local_left[j],local_right[j]]]
            r0 = np.nan_to_num(np.corrcoef(rr,rowvar=False)[0,1]);r1 = np.nan_to_num(np.corrcoef(np.vstack([rr,xx]),rowvar=False)[0,1])
            p = scope["power"];n = len(reference)
            direct = (n+1)*np.sign(r1)*abs(np.clip(r1,-1,1))**p-n*np.sign(r0)*abs(np.clip(r0,-1,1))**p
            max_edge_error = max(max_edge_error,abs(direct-one[0,j]))
    if max_edge_error > 1e-8 or max_stat_error > 1e-10:
        raise ValueError(f"Brain numerical pilot failed: {max_edge_error}, {max_stat_error}")
    return dict(max_direct_edge_error=max_edge_error,max_scipy_statistics_error=max_stat_error,
        validation_controls_add_one=True,unrelated_validation_donor_invariant=True)


def pilot(root,identity,workers):
    global _CONTEXT
    measurements = []; numerical = {}; peak = 0.; sort_seconds = []
    for method in METHODS:
        _CONTEXT = make_context(method)
        numerical[method] = numerical_pilot(_CONTEXT)
        scopes = [_CONTEXT["scopes"][0],_CONTEXT["scopes"][1]]
        tasks = [(s,0,min(SHARD,s["edges"])) for s in scopes]
        records = run_tasks(root,method,tasks,identity,workers,"pilot_discovery")
        measurements.extend(records)
        files = [paths(root,method,s,0)[0] for s in scopes]
        # Independently check all 30 BH families on actual pilot p-values.
        selected = np.concatenate([pd.read_parquet(paths(root,method,s,0)[1],columns=["test_id"]).test_id.unique() for s in scopes])
        for outcome in OUTCOMES:
            for r in ("pearson","spearman"):
                column = f"{outcome}__{r}_p";started = time.monotonic()
                adjusted,counts = scoped_external_bh(files,column,selected,
                    temporary_directory=Path(root)/"scratch/whole_network_brain/sort_pilot",memory_gib=.0625)
                sort_seconds.append(time.monotonic()-started)
                frame = pd.concat([pd.read_parquet(p,columns=["test_id","scope_id",column]) for p in files],ignore_index=True).set_index("test_id")
                valid = frame[column].notna();direct = pd.Series(false_discovery_control(frame.loc[valid,column]),index=frame.index[valid])
                np.testing.assert_allclose(adjusted.network_fdr,adjusted.test_id.map(direct),atol=1e-14,rtol=0)
                for scope,part in frame.groupby("scope_id"):
                    valid = part[column].notna();direct = pd.Series(false_discovery_control(part.loc[valid,column]),index=part.index[valid])
                    actual = adjusted.loc[adjusted.scope_id.eq(scope)]
                    np.testing.assert_allclose(actual.component_fdr,actual.test_id.map(direct),atol=1e-14,rtol=0)
                peak = max(peak,resource_guard(root)["current_rss_bytes"]/1024**3)
        status(root,"pilot_fdr_verified",method=method,independent_families=30)
    total = 51918*51917//2
    existing = resource_guard(root)["scratch_bytes"]
    pbytes_per_edge = max(r["pvalue_bytes"]/r["edges"] for r in measurements)
    # Conservative 80 extra bytes per edge for external-sort spill, plus 20 GiB
    # of candidate/checkpoint data; methods are deliberately serialized.
    projected = existing+(pbytes_per_edge+80)*total+20*1024**3
    peak = max(peak,max(r["peak_combined_rss_gib"] for r in measurements))
    report = dict(status="passed" if peak<=32 and projected<=500*1024**3 else "failed_resource_gate",
        engine_identity=identity,updated_at=stamp(),numerical=numerical,measurements=measurements,
        numerical_workers=workers,peak_combined_rss_gib=peak,projected_scratch_gib=projected/1024**3,
        measured_pvalue_bytes_per_edge=pbytes_per_edge,median_pilot_sort_seconds=float(np.median(sort_seconds)),
        fdr_families_independently_verified=60,sort_memory_gib=.0625,production_sort_memory_gib=4,
        expected_edges_per_method=total,methods_sequential=True,motor_launch_authorized=False,
        runtime_projection_note="Small-component throughput only; billion-edge external-sort runtime is not measured.",
        compute_hours_two_workers={m:sum(r["seconds"] for r in measurements if r["method"]==m)/sum(r["edges"] for r in measurements if r["method"]==m)*total/workers/3600 for m in METHODS})
    atomic_json(report,workdir(root)/"pilot_report.json")
    if report["status"] != "passed":
        raise RuntimeError("Measured brain resource pilot did not pass")
    return report


def merge_hits(root,ctx,identity):
    frames = []; pfiles = []
    for scope in ctx["scopes"]:
        top = TopUnion(OUTCOMES,TOP_K)
        for start in range(0,scope["edges"],SHARD):
            pfile,hfile,_ = paths(root,ctx["method"],scope,start)
            frame = pd.read_parquet(hfile)
            for outcome,part in frame.groupby("outcome"):
                top.update(part.test_id.to_numpy(),{outcome:{k:part[k].to_numpy() for k in STAT_COLUMNS}})
            pfiles.append(pfile)
        frame = top.frame();frame["scope_id"] = scope["scope_id"];frame["component"] = scope["component"]
        frames.append(frame)
    return pd.concat(frames,ignore_index=True),pfiles


def fdr(root,method,selected,pfiles,identity):
    result = selected.copy()
    for outcome,ix in result.groupby("outcome").groups.items():
        for r in ("pearson","spearman"):
            base = workdir(root)/"fdr"/method/f"{outcome}__{r}"
            marker = base.with_suffix(".json"); data = base.with_suffix(".parquet")
            if marker.is_file():
                record = json.loads(marker.read_text())
                if record["engine_identity"] != identity or sha(data) != record["sha256"]:
                    raise ValueError("FDR checkpoint changed")
                adjusted = pd.read_parquet(data);counts = record["counts"]
            else:
                status(root,"network_fdr",method=method,outcome=outcome,correlation=r)
                adjusted,counts = scoped_external_bh(pfiles,f"{outcome}__{r}_p",result.loc[ix,"test_id"],
                    temporary_directory=Path(root)/"scratch/whole_network_brain/sort",memory_gib=4)
                if set(adjusted.test_id) != set(result.loc[ix,"test_id"]):
                    raise ValueError("Selected hit missing from full-search BH stream")
                atomic_parquet(adjusted,data)
                atomic_json(dict(engine_identity=identity,sha256=sha(data),counts=counts),marker)
            lookup = adjusted.set_index("test_id")
            result.loc[ix,f"{r}_network_fdr"] = result.loc[ix,"test_id"].map(lookup.network_fdr)
            result.loc[ix,f"{r}_fdr"] = result.loc[ix,"test_id"].map(lookup.component_fdr)
            result.loc[ix,f"{r}_network_tested_n"] = counts["network_tested_n"]
            component_counts = {int(k):v for k,v in counts["component_tested_n"].items()}
            result.loc[ix,f"{r}_tested_n"] = result.loc[ix,"scope_id"].map(component_counts)
            resource_guard(root)
    return result


def validate_hits(root,ctx,selected):
    from app_helpers.outcome_screening import correlations
    source = ctx["source"];results = []
    for scope_id,frame in selected.groupby("scope_id"):
        scope = ctx["scopes"][int(scope_id)];selected_ids = np.sort(frame.test_id.unique())
        left,right = edge_nodes(selected_ids,len(ctx["nodes"]))
        new_ids = source.donors(tuple(scope["tissues"]),True)
        used = np.unique(np.r_[left,right]); lookup = {int(i):j for j,i in enumerate(used)}
        targets = np.column_stack([source.expression[ctx["nodes"][i][0]].loc[new_ids,ctx["nodes"][i][1]].to_numpy(float) for i in used])
        local_left = np.array([lookup[int(i)] for i in left]);local_right = np.array([lookup[int(i)] for i in right])
        validation_network = FrozenSignedAlt(ctx["reference"][:,used],targets)
        original_parts = [];validation_parts = []
        for start in range(0,len(left),BATCH):
            end = start+BATCH
            original_parts.append(ctx["network"].weights(left[start:end],right[start:end],scope["power"]))
            validation_parts.append(validation_network.weights(local_left[start:end],local_right[start:end],scope["power"]))
        old = np.concatenate(original_parts,axis=1);new = np.concatenate(validation_parts,axis=1)
        hits = frame.copy();hits["identity"] = hits.test_id.map(lambda i:f"brain-edge:{int(i):012d}")
        value_index = {int(i):j for j,i in enumerate(selected_ids)}
        hits["value_index"] = hits.test_id.map(value_index)
        for outcome,part in hits.groupby("outcome"):
            direct = correlations(old[:,part.value_index.to_numpy(int)],ctx["outcomes"][outcome])
            for column in STAT_COLUMNS:
                np.testing.assert_allclose(part[column],direct[column],atol=2e-12,rtol=1e-10,equal_nan=True)
        result = stats_for_selected(source,ctx["original"],new_ids,old,new,hits)
        result["network_method"] = ctx["method"];result["normalization_profile"] = "signedAlt_TS3_CT2"
        result["dataset"] = "brain";result["analysis"] = "whole_network"
        a,b = edge_nodes(result.test_id.to_numpy(),len(ctx["nodes"]))
        result["tissue_left"] = [ctx["nodes"][i][0] for i in a];result["tissue_right"] = [ctx["nodes"][i][0] for i in b]
        result["gene_symbol_left"] = [gene_symbol(ctx["nodes"][i][1]) for i in a]
        result["gene_symbol_right"] = [gene_symbol(ctx["nodes"][i][1]) for i in b]
        result["original_fdr_scope"] = "all valid edges within component and whole network, before selection"
        result["validation_fdr_scope"] = "frozen selected-hit union, separately within component and whole network"
        results.append(result)
        # Only the selected union is retained. No new donor-weight publication.
        for cohort,donors,values in (("original",ctx["original"],old),("validation",new_ids,new)):
            for start in range(0,len(selected_ids),2048):
                stop = min(len(selected_ids),start+2048)
                frame = pd.DataFrame(dict(sample_id=np.repeat([source.mapping[d] for d in donors],stop-start),
                    test_id=np.tile(selected_ids[start:stop],len(donors)),weight=values[:,start:stop].ravel()))
                atomic_parquet(frame,Path(root)/"private/whole_network_weights/brain"/ctx["method"]/scope["component"]/cohort/f"part_{start:06d}.parquet")
        status(root,"selected_hit_validation",method=ctx["method"],component=scope["component"],selected_edges=len(selected_ids),validation_donors=len(new_ids))
        resource_guard(root)
    result = pd.concat(results,ignore_index=True)
    for outcome,ix in result.groupby("outcome").groups.items():
        for r in ("pearson","spearman"):
            q,n = bh(result.loc[ix,f"validation_{r}_p"])
            result.loc[ix,f"validation_{r}_network_selected_fdr"] = q
            result.loc[ix,f"validation_{r}_network_tested_n"] = n
    return result


def reconcile_results(result):
    if result.duplicated(["component","outcome","test_id"]).any():
        raise ValueError("Duplicate whole-network selected identity")
    for (_, _),part in result.groupby(["component","outcome"]):
        for r in ("pearson","spearman"):
            ranks = sorted(part[f"{r}_rank"].dropna().astype(int))
            if ranks != list(range(1,TOP_K+1)):
                raise ValueError("Whole-network top-K ranks are incomplete")
            valid = part[f"validation_{r}_p"].notna()
            direct = false_discovery_control(part.loc[valid,f"validation_{r}_p"]) if valid.any() else []
            np.testing.assert_allclose(part.loc[valid,f"validation_{r}_selected_fdr"],direct,atol=1e-14,rtol=0)
            if part.loc[part.validation_n.lt(10),f"validation_{r}_p"].notna().any():
                raise ValueError("Undersized validation association received a p-value")
    for outcome,part in result.groupby("outcome"):
        for r in ("pearson","spearman"):
            counts = part.groupby("component")[f"{r}_tested_n"].first()
            if not part[f"{r}_network_tested_n"].eq(counts.sum()).all():
                raise ValueError("Component/global FDR family sizes disagree")
            valid = part[f"validation_{r}_p"].notna()
            direct = false_discovery_control(part.loc[valid,f"validation_{r}_p"]) if valid.any() else []
            np.testing.assert_allclose(part.loc[valid,f"validation_{r}_network_selected_fdr"],direct,atol=1e-14,rtol=0)
    return dict(status="passed",rows=len(result),components=result.component.nunique(),outcomes=result.outcome.nunique())


def publish_ready(root,method,result,identity):
    # Aggregate-only local staging is permitted; no remote operations occur.
    reconciliation = reconcile_results(result)
    public = public_catalog(result)
    entries = []
    for component_name,part in public.groupby("component"):
        relative = Path("whole_network/brain")/f"{method}__{component_name}.parquet"
        path = Path(root)/"public_bundle"/relative
        atomic_parquet(part.sort_values(["outcome","identity"]),path)
        if path.stat().st_size >= 95*1024**2:
            raise ValueError("Whole-network aggregate shard exceeds public file limit")
        entries.append(dict(path=relative.as_posix(),rows=len(part),bytes=path.stat().st_size,
            sha256=sha(path),status="complete"))
    receipt = dict(status="passed",engine_identity=identity,method=method,
        audit_identity=require_audit(root),files=entries,donor_values_public=False,
        full_search_fdr="network and component, all valid tests before selection",
        validation_fdr="frozen selected-hit union only",reconciliation=reconciliation,updated_at=stamp())
    atomic_json(receipt,Path(root)/"qa"/f"whole_network_brain_{method}.json")
    manifest_path = Path(root)/"public_bundle/manifest.json"
    with (manifest_path.parent/".whole_network_manifest.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        replace_paths = {e["path"] for e in entries}
        manifest["files"] = [e for e in manifest["files"] if e["path"] not in replace_paths]+entries
        done = set(manifest.get("whole_network_brain_methods_complete",[]))|{method}
        manifest.update(whole_network_status="brain_complete_motor_pending" if done==set(METHODS) else "brain_partial_motor_pending",
            whole_network_brain_methods_complete=sorted(done),updated_at=stamp())
        atomic_json(manifest,manifest_path)
    from scripts.stage_outcome_screening import stage
    stage(root)
    return receipt


def recycle_pvalues(root,method,ctx,identity):
    """Delete only this runner's own temporary p-values after validated release."""
    receipt = json.loads((Path(root)/"qa"/f"whole_network_brain_{method}.json").read_text())
    if receipt["status"] != "passed" or receipt["engine_identity"] != identity:
        raise ValueError("No validated release authorizes scratch recycling")
    parent = (Path(root)/"scratch/whole_network_brain"/method).resolve()
    count = size = 0
    for scope in ctx["scopes"]:
        for start in range(0,scope["edges"],SHARD):
            path = paths(root,method,scope,start)[0]
            if not path.resolve().is_relative_to(parent):
                raise ValueError("Temporary cleanup escaped exact method directory")
            if path.is_file():
                size += path.stat().st_size;path.unlink();count += 1
    atomic_json(dict(files=count,bytes_recycled=size,engine_identity=identity,
        reason="Temporary full-search p-values recycled after exact BH and validated selected-hit release; regenerable from frozen inputs"),
        workdir(root)/f"recycled_{method}.json")


def initialize(root):
    root = Path(root);require_audit(root)
    source = load_sources("brain")
    audit = json.loads((root/"audit.json").read_text())
    for path in source.paths:
        if audit["sources"].get(str(path)) != sha(path):
            raise ValueError("Frozen brain source changed; do not reuse checkpoints")
    sources = [Path(__file__),APP/"scripts/screening_brain_whole.py",APP/"scripts/screening_sources.py",
               APP/"scripts/screening_bh.py",APP/"app_helpers/outcome_screening.py"]
    settings = dict(schema_version=1,audit_identity=require_audit(root),batch_edges=BATCH,shard_edges=SHARD,top_k=TOP_K,
        outcomes=OUTCOMES,methods=METHODS,code_hashes={str(p.relative_to(APP)):sha(p) for p in sources},
        reference_definitions=dict(standard=450,control_anchored=164),beta_TS=3,beta_CT=2,
        ranking="absolute coefficient then canonical upper-triangle index over sorted tissue-gene nodes",
        selected_hit_weights_private=True,donor_publication_authorized=False,motor_launch_authorized=False,
        brain_execution_authorized=True,authorization_basis="User approved pilot-gated whole-network execution on 2026-10-25")
    identity = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    path = workdir(root)/"run_manifest.json"
    if path.is_file() and json.loads(path.read_text())["engine_identity"] != identity:
        raise ValueError("Runner settings/code changed; reconcile before resuming")
    atomic_json(dict(settings,engine_identity=identity),path)
    return identity


def run(root=RUN,workers=2,pilot_only=False):
    global _CONTEXT
    if workers not in (1,2):
        raise ValueError("At most two numerical workers")
    root = Path(root);workdir(root).mkdir(parents=True,exist_ok=True)
    os.environ["SCREEN_COORDINATOR_PID"] = str(os.getpid())
    os.nice(5)  # Background priority; do not modify other analysis processes.
    with (workdir(root)/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status(root,"source_audit",pid=os.getpid())
            identity = initialize(root)
            path = workdir(root)/"pilot_report.json"
            if path.is_file():
                report = json.loads(path.read_text())
                if report["engine_identity"] != identity or report["status"] != "passed":
                    raise ValueError("Pilot gate does not authorize this engine")
            else:
                report = pilot(root,identity,workers)
            status(root,"pilot_passed",report=report)
            if pilot_only:
                return
            for method in METHODS:
                completion = workdir(root)/f"complete_{method}.json"
                if completion.is_file():
                    if json.loads(completion.read_text())["engine_identity"] != identity:
                        raise ValueError("Completed method identity changed")
                    _CONTEXT = make_context(method)
                    recycle_pvalues(root,method,_CONTEXT,identity)
                    continue
                _CONTEXT = make_context(method)
                tasks = [(s,start,min(s["edges"],start+SHARD)) for s in _CONTEXT["scopes"] for start in range(0,s["edges"],SHARD)]
                run_tasks(root,method,tasks,identity,workers,"discovery")
                selected,pfiles = merge_hits(root,_CONTEXT,identity)
                selected = fdr(root,method,selected,pfiles,identity)
                result = validate_hits(root,_CONTEXT,selected)
                receipt = publish_ready(root,method,result,identity)
                atomic_json(dict(status="complete",engine_identity=identity,receipt=receipt),completion)
                recycle_pvalues(root,method,_CONTEXT,identity)
            status(root,"brain_complete_motor_gated",methods=list(METHODS),donor_values_published=False,git_pushed=False)
        except BaseException as error:
            status(root,"stopped_requires_attention",error_type=type(error).__name__,message=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--detach",action="store_true")
    args = parser.parse_args()
    if args.detach:
        directory = workdir(args.root);directory.mkdir(parents=True,exist_ok=True)
        logfile = directory/f"runner_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        command = [sys.executable,str(Path(__file__).resolve()),"--root",str(args.root),"--workers",str(args.workers)]
        if args.pilot_only: command.append("--pilot-only")
        with logfile.open("ab",buffering=0) as output:
            process = subprocess.Popen(command,cwd=APP,stdout=output,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,log=str(logfile),status="launched_pilot_gate_then_brain_search")))
    else:
        run(args.root,args.workers,args.pilot_only)
