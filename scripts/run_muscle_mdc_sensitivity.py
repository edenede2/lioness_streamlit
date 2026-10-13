#!/usr/bin/env python3
"""Resumable, paired muscle MDC sensitivity with variant-specific nulls.

One bootstrap calculation supplies every M5 power/scale/clipping variant.
The Control-derived scale is shared between diagnoses and re-estimated from
the Control null network in each replicate. Fixed modules are retained; this
analysis does not establish independence from Control-only module discovery.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
from app_helpers.mdc_sensitivity import VARIANTS, public_mdc_tables
from scripts import run_muscle_m5_mdc as M
from scripts import run_muscle_m5_mdc_significance as S
from models.se2.correlation import pairwise_corr
from models.se2.methods.soft_weighted import _full_bootstrap_stability

DEFAULT_OUTPUT = M.REPO_ROOT / "out/mdc_muscle_control_l4/20261013_no_q90_mdc"
FAMILIES = ("pearson", "signed_alt_low", "high_m5", "low_m5")
SCHEMA_VERSION = 1
_STATE = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def block_parts(rows, tissues, group_tag, nulls, seed=42):
    """Generate packed undirected absolute weights without a dense supra matrix."""
    labels = rows["Tissue"].astype(str).to_numpy()
    genes = rows["Gene ID"].astype(str).to_numpy()
    n = len(labels)
    i, j = np.triu_indices(n, k=1)
    codes = np.full(len(i), -1, dtype=np.int8)
    values = {family: np.zeros(len(i), dtype=np.float32) for family in FAMILIES}
    for code, (left, right) in enumerate(M.BLOCKS):
        li, ri = np.flatnonzero(labels == left), np.flatnonzero(labels == right)
        if not len(li) or not len(ri) or (left == right and len(li) < 2):
            continue
        X = tissues[left].loc[genes[li]].to_numpy(dtype=np.float32)
        Y = tissues[right].loc[genes[ri]].to_numpy(dtype=np.float32)
        freq, mean_r = _full_bootstrap_stability(
            X, Y, B=200, donor_frac=.8, corr_method="pearson", tau_r=.1,
            is_intra=left == right,
            rng=np.random.default_rng(M.stable_seed("muscle-m5-bootstrap", group_tag, int(rows["Cluster ID"].iloc[0]), left, right, seed)),
        )
        direct_r = pairwise_corr(X, Y, method="pearson")
        low_beta = 3 if left == right else 2
        high_beta = M._resolve_beta(left, right)
        z = np.ones_like(mean_r)
        if left != right:
            stats = nulls[f"{left}-{right}"]
            z = np.maximum((np.abs(mean_r) - stats["mu_null"]) / (stats["sigma_null"] + 1e-8), 0)
        matrices = {
            "pearson": np.abs(direct_r),
            "signed_alt_low": np.abs(direct_r) ** low_beta,
            "high_m5": freq * np.abs(mean_r) ** high_beta * z,
            "low_m5": freq * np.abs(mean_r) ** low_beta * z,
        }
        if left == right:
            u, v = np.triu_indices(len(li), k=1)
            gi, gj = li[u], li[v]
            selector = (u, v)
        else:
            gi, gj = np.repeat(li, len(ri)), np.tile(ri, len(li))
            selector = None
        lo, hi = np.minimum(gi, gj), np.maximum(gi, gj)
        packed = lo * (2 * n - lo - 1) // 2 + hi - lo - 1
        codes[packed] = code
        for family, array in matrices.items():
            values[family][packed] = array[selector] if selector is not None else array.ravel()
    if np.any(codes < 0):
        raise AssertionError("A possible undirected edge has no tissue block")
    return {"labels": labels, "genes": genes, "i": i, "j": j, "codes": codes, "values": values}


def group_parts(rows_by_module, tissues, group_tag, nulls, seed=42):
    parts = {int(module): block_parts(rows, tissues, group_tag, nulls, seed)
             for module, rows in rows_by_module.items()}
    scales = {}
    for family in ("high_m5", "low_m5"):
        for code, block in enumerate(M.BLOCKS):
            chunks = []
            for part in parts.values():
                v = part["values"][family][part["codes"] == code]
                if np.any(v):
                    chunks.append(v[v != 0])
            v = np.concatenate(chunks) if chunks else np.empty(0)
            scales[(family, code)] = float(np.quantile(v, .9)) if len(v) else 0.
    return parts, scales


def variant_weights(part, variant, group_scales, control_scales):
    definition = VARIANTS[variant]
    values = part["values"][definition["family"]].copy()
    if definition["scale"] != "none":
        scales = control_scales if definition["scale"] == "control" else group_scales
        for code in np.unique(part["codes"]):
            scale = scales[(definition["family"], int(code))]
            if scale > 0:
                selected = part["codes"] == code
                values[selected] /= scale
    if definition["clip"]:
        np.minimum(values, 10., out=values)
    return values


def summarize_parts(ad_parts, control_parts, ad_scales, control_scales):
    records = []
    for module in sorted(control_parts):
        a, c = ad_parts[module], control_parts[module]
        if not np.array_equal(a["labels"], c["labels"]) or not np.array_equal(a["genes"], c["genes"]):
            raise AssertionError("AD/Control endpoints are misaligned")
        codes = a["codes"]
        masks = {"total": np.ones(len(codes), bool), "TS": codes < 3, "CT": codes >= 3}
        masks.update({scope: codes == code for code, scope in enumerate(M.SCOPES[3:])})
        for variant in VARIANTS:
            av = variant_weights(a, variant, ad_scales, control_scales)
            cv = variant_weights(c, variant, control_scales, control_scales)
            for scope in M.SCOPES:
                mask = masks[scope]
                count = int(mask.sum())
                ma = float(av[mask].mean()) if count else np.nan
                mc = float(cv[mask].mean()) if count else np.nan
                ratio = ma / mc if mc > 1e-12 else np.nan
                records.append({"normalization_variant": variant, "module": module, "scope": scope,
                    "n_edges": count, "mean_abs_ad": ma, "mean_abs_control": mc,
                    "mdc_raw": ratio, "mdc_log2": np.log2(ratio) if ratio > 0 else np.nan})
    return pd.DataFrame(records)


def init_worker(expression_root, assignments_path, observed_root, seed, config_hash):
    assignments = M.read_filtered_assignments(assignments_path)
    control, _, _ = M.load_group(expression_root, "Control")
    ad, _, _ = M.load_group(expression_root, "AD")
    _STATE.update(assignments=assignments, control=control, ad=ad, seed=seed,
                  observed_root=observed_root, config_hash=config_hash)
    _STATE["common_genes"] = {t: np.asarray(sorted(set(control[t].index).intersection(ad[t].index)), dtype=str) for t in M.TISSUES}


def calculate(kind, permutation, checkpoint):
    started = time.monotonic()
    seed = _STATE["seed"]
    if kind == "sample":
        rows = S.original_assignments(_STATE["assignments"])
        control = S.independently_permute_genes(_STATE["control"], "control", permutation, seed)
        ad = S.independently_permute_genes(_STATE["ad"], "ad", permutation, seed)
        ct, at = f"control_sample_{permutation}_{seed}", f"ad_sample_{permutation}_{seed}"
        cn, an = M._group_null_statistics(control, ct), M._group_null_statistics(ad, at)
    elif kind == "gene":
        rows = S.sampled_assignments(_STATE["assignments"], _STATE["common_genes"], permutation, seed)
        control, ad = _STATE["control"], _STATE["ad"]
        ct, at = f"control_gene_{permutation}_{seed}", f"ad_gene_{permutation}_{seed}"
        cn = S.observed_ct_nulls(_STATE["observed_root"], "control")
        an = S.observed_ct_nulls(_STATE["observed_root"], "ad")
    else:
        raise ValueError(kind)
    cp, cq = group_parts(rows, control, ct, cn, seed)
    ap, aq = group_parts(rows, ad, at, an, seed)
    frame = summarize_parts(ap, cp, aq, cq)
    modules = np.asarray(sorted(rows), dtype=np.int64)
    lookup = pd.MultiIndex.from_product([modules, M.SCOPES], names=["module", "scope"])
    values = np.stack([frame[frame.normalization_variant.eq(v)].set_index(["module", "scope"]).reindex(lookup).mdc_raw.to_numpy().reshape(len(modules),len(M.SCOPES)) for v in VARIANTS])
    temporary = checkpoint.with_name(checkpoint.name + ".partial")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.monotonic() - started
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, modules=modules, scopes=np.asarray(M.SCOPES), variants=np.asarray(list(VARIANTS)),
                            mdc=values, config_hash=_STATE["config_hash"], kind=kind,
                            permutation=permutation, seed=seed, seconds=elapsed, peak_rss_mib=peak_mib)
    os.replace(temporary, checkpoint)
    return {"kind":kind, "permutation":permutation, "seconds":round(elapsed,3), "peak_rss_mib":round(peak_mib,1)}


def checkpoint_count(root, kind):
    return len(list((root/"checkpoints"/kind).glob("perm_*.npz")))


def summarize_inference(root, sample_n, gene_n, config_hash):
    observed = pd.read_parquet(root/"mdc_observed.parquet")
    arrays = {}
    for kind, n in [("sample",sample_n),("gene",gene_n)]:
        chunks = []
        for rep in range(1,n+1):
            with np.load(root/"checkpoints"/kind/f"perm_{rep:03d}.npz", allow_pickle=False) as item:
                if str(item['config_hash']) != config_hash or list(item['variants']) != list(VARIANTS) or list(item['scopes']) != list(M.SCOPES):
                    raise ValueError("Null checkpoint identity differs")
                if chunks and not np.array_equal(modules,item['modules']):
                    raise ValueError("Null module order differs")
                modules = item['modules']
                chunks.append(item['mdc'])
        arrays[kind] = np.stack(chunks,axis=-1)
    results=[]
    for idx,variant in enumerate(VARIANTS):
        result = S.build_significance_from_arrays(observed[observed.normalization_variant.eq(variant)],
            modules,arrays['sample'][idx],arrays['gene'][idx],sample_n,gene_n)
        results.append(result)
    data=pd.concat(results,ignore_index=True)
    atomic_parquet(data,root/"mdc_significance.parquet")
    return data


def build_public(root, public_root):
    manifest=json.loads((root/"run_manifest.json").read_text())
    complete=manifest.get('status')=='complete'
    frame=pd.read_parquet(root/('mdc_significance.parquet' if complete else 'mdc_observed.parquet'))
    details=pd.read_csv(APP_ROOT/'data/muscle_control_l4/module_details.tsv',sep='\t')
    summary,resolved=[],[]
    for variant,part in frame.groupby('normalization_variant',sort=False):
        a,b=public_mdc_tables(part,details)
        for d in (a,b):
            d['normalization_variant']=variant
            d['normalization_label']=VARIANTS[variant]['label']
        summary.append(a);resolved.append(b)
    public_root.mkdir(parents=True,exist_ok=True)
    atomic_parquet(pd.concat(summary,ignore_index=True),public_root/'summary.parquet')
    atomic_parquet(pd.concat(resolved,ignore_index=True),public_root/'resolved.parquet')
    files={name:{'bytes':(public_root/name).stat().st_size,'sha256':M.sha256_file(public_root/name)} for name in ['summary.parquet','resolved.parquet']}
    public_manifest={'schema_version':1,'status':manifest['status'],'module_count':155,
        'diagnosis_counts':{'Control':120,'AD':155},'sample_permutations':manifest['sample_permutations'],
        'gene_permutations':manifest['gene_permutations'],'completed_sample_permutations':checkpoint_count(root,'sample'),
        'completed_gene_permutations':checkpoint_count(root,'gene'),'files':files,
        'variants':{v:{**d,'observed_complete':True,'permutation_significance_complete':complete,
                       'status':'complete' if complete else 'observed_complete_inference_pending'} for v,d in VARIANTS.items()},
        'notes':manifest['notes'],'config_hash':manifest['config_hash'],'updated_utc':utc_now()}
    M.atomic_json(public_manifest,public_root/'manifest.json')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['observed','pilot','nulls','summarize','public','all'],default='all')
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--observed-root',type=Path,default=M.DEFAULT_OUTPUT)
    p.add_argument('--expression-root',type=Path,default=M.DEFAULT_EXPRESSION)
    p.add_argument('--assignments',type=Path,default=M.DEFAULT_ASSIGNMENTS)
    p.add_argument('--public-root',type=Path,default=APP_ROOT/'data/muscle_control_l4/mdc_sensitivity')
    p.add_argument('--sample-permutations',type=int,default=200)
    p.add_argument('--gene-permutations',type=int,default=200)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args()


def main():
    args=parse_args()
    if min(args.workers,args.sample_permutations,args.gene_permutations)<1 or args.seed!=42:
        raise ValueError('Positive counts and the original seed 42 are required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    sources=[M.file_identity(args.assignments)]
    for group in ['Control','AD']:
        for tissue in M.TISSUES:
            sources.append(M.file_identity(M.expression_path(args.expression_root,tissue,group)))
    config={'schema_version':SCHEMA_VERSION,'variants':VARIANTS,'seed':42,'parameters':M.M5_PARAMETERS,
            'sources':sources,'runner_sha256':M.sha256_file(Path(__file__))}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    old=json.loads((root/'run_manifest.json').read_text()) if (root/'run_manifest.json').exists() else {}
    if old and old['config_hash']!=config_hash:
        raise ValueError('Existing sensitivity checkpoints have different provenance; use a separate output root')
    manifest={**old,'schema_version':1,'config_hash':config_hash,'configuration':config,'status':old.get('status','running'),
        'sample_permutations':args.sample_permutations,'gene_permutations':args.gene_permutations,
        'workers':args.workers,'notes':[
            'Identical beta values are used in AD and Control for every corresponding variant and tissue block.',
            'Low powers are beta_TS=3 and beta_CT=2; M5 variants retain bootstrap stability and group-specific CT null-excess calibration.',
            'Shared scale means Control-derived block q90 applied identically to both groups. It is refitted from the Control network within each permutation.',
            'The sample null independently permutes donors within genes and rebuilds all affected network terms; the gene null preserves tissue counts.',
            'FDR is computed separately by variant, scope and direction across valid modules; it requires support from both nulls.',
            'Modules were discovered in Controls. These sensitivities remain conditional on the fixed module definition and do not remove discovery selection dependence.',
            'No existing MDC, LIONESS or prediction result is replaced.']}
    M.atomic_json(manifest,root/'run_manifest.json')
    try:
        if args.stage in ['observed','all','pilot'] and not (root/'mdc_observed.parquet').exists():
            init_worker(args.expression_root,args.assignments,args.observed_root,42,config_hash)
            rows=S.original_assignments(_STATE['assignments'])
            cp,cq=group_parts(rows,_STATE['control'],'Control',S.observed_ct_nulls(args.observed_root,'control'))
            ap,aq=group_parts(rows,_STATE['ad'],'AD',S.observed_ct_nulls(args.observed_root,'ad'))
            observed=summarize_parts(ap,cp,aq,cq)
            baseline=pd.read_parquet(args.observed_root/'mdc_observed.parquet')
            joined=observed[observed.normalization_variant.eq('original')].merge(baseline,on=['module','scope'],validate='one_to_one')
            finite=np.isfinite(joined.mdc_raw_x)&np.isfinite(joined.mdc_raw_y)
            np.testing.assert_allclose(joined.loc[finite,'mdc_raw_x'],joined.loc[finite,'mdc_raw_y'],rtol=1e-6,atol=1e-10)
            if not np.array_equal(joined.mdc_raw_x.isna(),joined.mdc_raw_y.isna()):
                raise AssertionError('Original missingness changed')
            manifest['original_formula_reconciled']=True
            manifest['original_max_relative_error']=float(((joined.loc[finite,'mdc_raw_x']-joined.loc[finite,'mdc_raw_y']).abs()/joined.loc[finite,'mdc_raw_y']).max())
            manifest['observed_rows']=len(observed)
            manifest['status']='observed_complete_inference_pending'
            atomic_parquet(observed,root/'mdc_observed.parquet')
            scales=[{'group':g,'family':f,'block':f'{M.BLOCKS[c][0]}-{M.BLOCKS[c][1]}','q90':q} for g,gs in [('Control',cq),('AD',aq)] for (f,c),q in gs.items()]
            pd.DataFrame(scales).to_csv(root/'block_scales.tsv',sep='\t',index=False)
            M.atomic_json(manifest,root/'run_manifest.json')
            _STATE.clear()
        if args.stage=='pilot':
            init_worker(args.expression_root,args.assignments,args.observed_root,42,config_hash)
            results=[]
            for kind in ['sample','gene']:
                checkpoint=root/'checkpoints'/kind/'perm_001.npz'
                if not checkpoint.exists():results.append(calculate(kind,1,checkpoint))
            manifest['pilot_measurements']=results
            _STATE.clear()
        if args.stage in ['nulls','all']:
            if not (root/'mdc_observed.parquet').exists():raise FileNotFoundError('Observed sensitivity must pass QA first')
            tasks=[(kind,rep,root/'checkpoints'/kind/f'perm_{rep:03d}.npz') for kind,n in [('sample',args.sample_permutations),('gene',args.gene_permutations)] for rep in range(1,n+1) if not (root/'checkpoints'/kind/f'perm_{rep:03d}.npz').exists()]
            manifest['status']='running'
            M.atomic_json(manifest,root/'run_manifest.json')
            with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                initargs=(args.expression_root,args.assignments,args.observed_root,42,config_hash)) as pool:
                futures={pool.submit(calculate,*task):task[:2] for task in tasks}
                for future in as_completed(futures):
                    result=future.result();print(json.dumps(result),flush=True)
                    manifest['last_completed']=result
                    manifest['updated_utc']=utc_now()
                    manifest['completed_sample_permutations']=checkpoint_count(root,'sample')
                    manifest['completed_gene_permutations']=checkpoint_count(root,'gene')
                    M.atomic_json(manifest,root/'run_manifest.json')
        if args.stage in ['summarize','all','nulls']:
            if checkpoint_count(root,'sample')==args.sample_permutations and checkpoint_count(root,'gene')==args.gene_permutations:
                frame=summarize_inference(root,args.sample_permutations,args.gene_permutations,config_hash)
                manifest['status']='complete';manifest['completed_utc']=utc_now()
                manifest['significance_rows']=len(frame)
                manifest['significant_fdr05_by_variant']=frame.groupby('normalization_variant').significant_fdr_05.sum().astype(int).to_dict()
        manifest['updated_utc']=utc_now()
        manifest['completed_sample_permutations']=checkpoint_count(root,'sample')
        manifest['completed_gene_permutations']=checkpoint_count(root,'gene')
        M.atomic_json(manifest,root/'run_manifest.json')
        if (root/'mdc_observed.parquet').exists():build_public(root,args.public_root)
        print(json.dumps({k:v for k,v in manifest.items() if k!='configuration'},indent=2),flush=True)
    except Exception as error:
        manifest.update(status='failed',error=str(error),updated_utc=utc_now())
        M.atomic_json(manifest,root/'run_manifest.json')
        raise


if __name__=='__main__':main()
