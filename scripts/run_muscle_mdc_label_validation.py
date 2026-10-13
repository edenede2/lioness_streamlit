#!/usr/bin/env python3
"""Resumable MDC steps 1/2/4; no module discovery or legacy-output replacement."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
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

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import run_muscle_m5_mdc as M
from scripts import run_muscle_m5_mdc_significance as S
from scripts import run_muscle_mdc_sensitivity as R
from scripts.build_muscle_public_data import MUSCLE_METADATA
from scripts.run_muscle_lioness import donor_to_projid
from app_helpers.mdc_validation import VARIANTS, SCOPES, add_contrasts, permute_within_strata, proportional_subset, summarize_validation

ROOT = M.REPO_ROOT / "out/mdc_muscle_control_l4/20261013_label_balance_ct_ts"
PUBLIC = APP / "data/muscle_control_l4/mdc_label_validation"
MODULE_SET = "muscle_control_l4"
MODULE_COUNT = 155
DISCOVERY_NOTE = "Fixed Control-derived modules; discovery-selection bias is not corrected."
STATE = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_cohort():
    control, cd, cs = M.load_group(M.DEFAULT_EXPRESSION, "Control")
    ad, ad_ids, ads = M.load_group(M.DEFAULT_EXPRESSION, "AD")
    if set(cd).intersection(ad_ids):
        raise ValueError("AD and Control donors overlap")
    pooled = {}
    for t in M.TISSUES:
        if not np.array_equal(control[t].index, ad[t].index):
            raise ValueError("Expression backgrounds differ; reconcile before validation")
        pooled[t] = pd.concat([control[t], ad[t]], axis=1)
    donors = cd + ad_ids
    source = pd.read_csv(MUSCLE_METADATA, usecols=["ProjID", "study", "msex"], dtype={"ProjID": str})
    source.ProjID = source.ProjID.str.replace(r"\.0$", "", regex=True)
    ids = [donor_to_projid(d) for d in donors]
    source = source[source.ProjID.isin(ids)]
    if source.groupby("ProjID")[["study", "msex"]].nunique(dropna=True).gt(1).any().any():
        raise ValueError("Conflicting study/sex metadata")
    meta = source.groupby("ProjID")[["study", "msex"]].first().reindex(ids)
    if meta.isna().any().any():
        raise ValueError("Missing stratification metadata; do not silently drop donors")
    strata = np.asarray([f"{study}|{sex}" for study, sex in meta.itertuples(index=False, name=None)])
    labels = np.asarray(["Control"] * len(cd) + ["AD"] * len(ad_ids))
    return pooled, labels, strata, cs + ads + [M.file_identity(MUSCLE_METADATA)]


def init_worker(config_hash, observed_path):
    pooled, labels, strata, _ = load_cohort()
    assignments = M.read_filtered_assignments(M.DEFAULT_ASSIGNMENTS)
    STATE.update(pooled=pooled, labels=labels, strata=strata,
                 rows=S.original_assignments(assignments), config_hash=config_hash,
                 observed=pd.read_parquet(observed_path) if Path(observed_path).exists() else None)


def selected_frames(indices):
    # Match the original gene×donor C layout. Pandas column selection otherwise
    # changes float32 summation order, amplified by M5's high powers.
    return {t: pd.DataFrame(np.ascontiguousarray(f.iloc[:, indices].to_numpy()),
                           index=f.index, columns=f.columns[indices], copy=False)
            for t, f in STATE["pooled"].items()}


def calculate_scores(control_idx, ad_idx, observed=False, balanced=False):
    """All label-dependent calibration is rebuilt; only fixed modules are reused."""
    c, a = selected_frames(control_idx), selected_frames(ad_idx)
    cn = S.observed_ct_nulls(M.DEFAULT_OUTPUT, "control") if observed else None
    an = S.observed_ct_nulls(M.DEFAULT_OUTPUT, "ad") if observed else M._group_null_statistics(a, "AD")
    if not balanced and cn is None:
        cn = M._group_null_statistics(c, "Control")
    cached_c = STATE["observed"].set_index(["normalization_variant", "module", "scope"]) if balanced else None
    records = []
    for module, rows in STATE["rows"].items():
        ap = R.block_parts(rows, a, "AD", an)
        cp = None if balanced else R.block_parts(rows, c, "Control", cn)
        codes = ap["codes"]
        masks = {"total": np.ones(len(codes), bool), "TS": codes < 3, "CT": codes >= 3}
        masks.update({s: codes == k for k, s in enumerate(M.SCOPES[3:])})
        for v in VARIANTS:
            family = "pearson" if v == "pearson" else "high_m5"
            for scope, mask in masks.items():
                n = int(mask.sum())
                ma = float(ap["values"][family][mask].mean()) if n else np.nan
                mc = float(cached_c.loc[(v, module, scope), "mean_abs_control"]) if balanced else (float(cp["values"][family][mask].mean()) if n else np.nan)
                ratio = ma / mc if mc > 1e-12 else np.nan
                statistic = np.log2(ratio) if ratio > 0 else np.nan
                eligible = bool(n >= 10 and ma > 1e-8 and mc > 1e-8 and np.isfinite(statistic))
                reason = "" if eligible else ("Fewer than 10 edges" if n < 10 else "Near-zero weight or unavailable ratio")
                records.append(dict(normalization_variant=v, module=module, scope=scope,
                    mean_abs_ad=ma, mean_abs_control=mc, n_edges=n, mdc_raw=ratio,
                    statistic=statistic, test_eligible=eligible, unavailable_reason=reason))
    return add_contrasts(pd.DataFrame(records))


def checkpoint_valid(path, config_hash):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        if str(z["config_hash"]) != config_hash:
            raise ValueError("Checkpoint provenance mismatch")
    return True


def run_rep(kind, rep, path):
    started = time.monotonic()
    labels, strata = STATE["labels"], STATE["strata"]
    if kind == "label":
        selected = permute_within_strata(labels, strata, M.stable_seed("mdc-label", rep, 42))
        for key in set(strata):
            if sorted(labels[strata == key]) != sorted(selected[strata == key]):
                raise AssertionError("Stratum counts changed")
        ci, ai = np.flatnonzero(selected == "Control"), np.flatnonzero(selected == "AD")
    else:
        ci = np.flatnonzero(labels == "Control")
        ai = proportional_subset(np.flatnonzero(labels == "AD"), strata, len(ci), M.stable_seed("mdc-balanced", rep, 42))
    if set(ci).intersection(ai) or len(ci) != 120 or len(ai) != (155 if kind == "label" else 120):
        raise AssertionError("Invalid group roles")
    frame = calculate_scores(ci, ai, balanced=kind == "balanced")
    modules = sorted(STATE["rows"])
    order = pd.MultiIndex.from_product([list(VARIANTS), modules, SCOPES], names=["normalization_variant", "module", "scope"])
    values = frame.set_index(["normalization_variant", "module", "scope"]).reindex(order).statistic.to_numpy().reshape(2, len(modules), len(SCOPES))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as h:
        np.savez_compressed(h, statistic=values, config_hash=STATE["config_hash"], modules=modules,
            variants=list(VARIANTS), scopes=SCOPES, kind=kind, repetition=rep,
            role_hash=hashlib.sha256(np.asarray([*ci, -1, *ai], np.int64).tobytes()).hexdigest())
    os.replace(temporary, path)
    return dict(kind=kind, repetition=rep, seconds=round(time.monotonic()-started, 3),
                peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1))


def load_reps(root, kind, expected, config_hash):
    result, rows = [], []
    for rep in range(1, expected+1):
        path = root / "checkpoints" / kind / f"rep_{rep:03d}.npz"
        if not checkpoint_valid(path, config_hash):
            continue
        with np.load(path, allow_pickle=False) as z:
            if list(z["variants"]) != list(VARIANTS) or list(z["scopes"]) != list(SCOPES):
                raise ValueError("Checkpoint axes mismatch")
            item = {}
            for vi, v in enumerate(VARIANTS):
                for mi, m in enumerate(z["modules"]):
                    for si, s in enumerate(SCOPES):
                        value = float(z["statistic"][vi,mi,si])
                        item[(v,int(m),s)] = value
                        rows.append(dict(normalization_variant=v,module=int(m),scope=s,repetition=rep,statistic=value))
            result.append(item)
    return result, pd.DataFrame(rows, columns=["normalization_variant","module","scope","repetition","statistic"])


def write_public(root, config, config_hash, status):
    if not (root/"observed.parquet").exists():
        return
    observed = pd.read_parquet(root/"observed.parquet")
    label, ln = load_reps(root,"label",config["label_permutations"],config_hash)
    balance, bn = load_reps(root,"balanced",config["balanced_resamples"],config_hash)
    result = summarize_validation(observed,label,balance,config["label_permutations"],config["balanced_resamples"])
    assert len(result)==len(VARIANTS)*MODULE_COUNT*len(SCOPES) and not result.duplicated(["normalization_variant","module","scope"]).any()
    result['validation_level']=DISCOVERY_NOTE
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for name, data in [("results.parquet",result),("label_replicates.parquet",ln),("balanced_replicates.parquet",bn)]:
        R.atomic_parquet(data,root/name)
        R.atomic_parquet(data,PUBLIC/name)
    files={p.name:{"bytes":p.stat().st_size,"sha256":M.sha256_file(p)} for p in PUBLIC.glob("*.parquet")}
    manifest=dict(schema_version=1,status=status,config_hash=config_hash,variants=VARIANTS,
        module_set=MODULE_SET,module_count=MODULE_COUNT,label_permutations=config["label_permutations"],balanced_resamples=config["balanced_resamples"],
        completed_label_permutations=len(label),completed_balanced_resamples=len(balance),
        diagnosis_counts={"Control":120,"AD":155},balanced_counts={"Control":120,"AD":120},
        strata="study × sex; exact within-stratum counts for label permutations",
        source_hashes=[s["sha256"] for s in config["sources"]],files=files,updated_utc=utc(),
        notes=[DISCOVERY_NOTE,
               "Each donor's gene and tissue expression profile is preserved; only diagnosis membership is permuted.",
               "Label-dependent M5 bootstrap and CT calibration are rebuilt; no q90 scaling or clipping.",
               "Two-sided equal-tail plus-one label-permutation p; BH across valid modules separately by weighting and scope.",
               "Primary CT–TS contrast uses unpowered correlations. M5 contrast is weighting-dependent sensitivity.",
               "Balanced results describe repeated AD subsampling to 120; 2.5/97.5 percentiles are not inferential confidence intervals.",
               "Existing disrupted-coexpression/gene-set null FDRs are retained and not reused here."])
    M.atomic_json(manifest,PUBLIC/"manifest.json")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["observed","pilot","all","public"],default="all")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--label-permutations",type=int,default=200)
    p.add_argument("--balanced-resamples",type=int,default=100)
    args=p.parse_args()
    if args.workers not in range(1,13) or min(args.label_permutations,args.balanced_resamples)<1:
        raise ValueError("Invalid worker/repetition count")
    ROOT.mkdir(parents=True,exist_ok=True)
    _, labels, strata, sources=load_cohort()
    sources.append(M.file_identity(M.DEFAULT_ASSIGNMENTS))
    config=dict(schema_version=1,seed=42,label_permutations=args.label_permutations,
        balanced_resamples=args.balanced_resamples,sources=sources,variants=VARIANTS,
        module_set=MODULE_SET,
        m5_parameters=M.M5_PARAMETERS,runner_sha256=M.sha256_file(Path(__file__)),
        helper_sha256=M.sha256_file(APP/"app_helpers/mdc_validation.py"))
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    mp=ROOT/"run_manifest.json"
    manifest=json.loads(mp.read_text()) if mp.exists() else {}
    if manifest and manifest["config_hash"]!=identity:
        raise ValueError("Run configuration changed; use a separate output directory")
    manifest.update(config_hash=identity,configuration=config,status="running",workers=args.workers)
    M.atomic_json(manifest,mp)
    try:
        if not (ROOT/"observed.parquet").exists():
            init_worker(identity,ROOT/"observed.parquet")
            frame=calculate_scores(np.flatnonzero(labels=="Control"),np.flatnonzero(labels=="AD"),observed=True)
            legacy=pd.read_parquet(R.DEFAULT_OUTPUT/"mdc_observed.parquet")
            joined=frame[frame.scope!="CT_minus_TS"].merge(legacy,on=["normalization_variant","module","scope"],validate="one_to_one")
            for col in ["mdc_raw","mean_abs_ad","mean_abs_control"]:
                np.testing.assert_allclose(joined[col+"_x"],joined[col+"_y"],rtol=1e-7,atol=1e-12,equal_nan=True)
            R.atomic_parquet(frame,ROOT/"observed.parquet")
            manifest["observed_reproduces_legacy"]=True
            STATE.clear()
        if args.stage in ["pilot","all"]:
            counts={"label":(2 if args.stage=="pilot" else args.label_permutations),
                    "balanced":(1 if args.stage=="pilot" else args.balanced_resamples)}
            tasks=[(k,rep,ROOT/"checkpoints"/k/f"rep_{rep:03d}.npz") for k,n in counts.items()
                   for rep in range(1,n+1) if not checkpoint_valid(ROOT/"checkpoints"/k/f"rep_{rep:03d}.npz",identity)]
            with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(identity,ROOT/"observed.parquet")) as pool:
                futures=[pool.submit(run_rep,*task) for task in tasks]
                for future in as_completed(futures):
                    rec=future.result();print(json.dumps(rec),flush=True)
                    manifest.update(last_completed=rec,updated_utc=utc())
                    for kind in ["label","balanced"]:
                        manifest["completed_"+kind]=len(list((ROOT/"checkpoints"/kind).glob("rep_*.npz")))
                    M.atomic_json(manifest,mp)
        complete=all(len(list((ROOT/"checkpoints"/kind).glob("rep_*.npz")))==n
                     for kind,n in [("label",args.label_permutations),("balanced",args.balanced_resamples)])
        manifest.update(status="complete" if complete else "partial",updated_utc=utc())
        M.atomic_json(manifest,mp)
        pub=write_public(ROOT,config,identity,manifest["status"])
        print(json.dumps({k:v for k,v in pub.items() if k not in ["files","notes","source_hashes"]},indent=2),flush=True)
    except Exception as e:
        manifest.update(status="failed",error=str(e),updated_utc=utc());M.atomic_json(manifest,mp)
        raise


if __name__=="__main__":main()
