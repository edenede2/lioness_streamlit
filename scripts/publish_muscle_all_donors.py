#!/usr/bin/env python3
"""Publish validated all-donor muscle milestones; never expose private outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import run_muscle_all_donors_analysis as A

DESTINATION = APP / "data" / A.KEY
DRIVE_ROOT = "1dTj5SkLxuDIvII5LayLViqOzjKQt-dos"
OAUTH_CLIENT = "client_secret_688842367267-8o25rcsq6jvrb1go8d0gsr1o304v1knq.apps.googleusercontent.com.json"
SERVICE_ACCOUNT = "lionessexplorer-05e37b3ea2a6.json"


def command(args):
    subprocess.run([str(a) for a in args], cwd=APP, check=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT":"0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1",
                        "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})


def confirm_push():
    """A local commit is not deployment; verify the remote even after push errors."""
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=APP,text=True).strip()
    try:
        command(['git','push','origin','master'])
    except subprocess.CalledProcessError:
        # Some workspaces synchronize commits through an authenticated external
        # connection while the shell lacks HTTPS credentials. Accept that only
        # after observing the exact commit on the actual remote master branch.
        pass
    for attempt in range(6):
        output=subprocess.check_output(['git','ls-remote','origin','refs/heads/master'],cwd=APP,text=True,
                                       env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
        if output.split() and output.split()[0]==commit:return
        if attempt<5:time.sleep(10)
    raise RuntimeError('Validated commit is push-ready but remote publication is not confirmed')


def validate_snapshot(source, files, final):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    details = pd.read_csv(source / "module_details.tsv", sep="\t")
    assignment = pd.read_csv(A.ASSIGNMENTS)
    expected = set(assignment["Cluster ID"].astype(int))
    assert set(details.module) == expected and len(details) == 173
    assert details.module_size.sum() == 31948
    metadata = pd.read_parquet(source / "sample_metadata.parquet")
    assert len(metadata) == 357 and metadata.sample_id.nunique() == 357
    assert metadata.diagnosis_group.value_counts().to_dict() == {"AD":155,"Control":120,"MCI":82}
    existing = pd.read_parquet(APP / "data/muscle_control_l4/sample_metadata.parquet")
    assert set(metadata.sample_id) == set(existing.sample_id)
    for name, record in files.items():
        path = source / name
        assert path.is_relative_to(source) and path.is_file(), path
        assert path.stat().st_size == record["bytes"] < 95 * 1024**2
        assert A.sha(path) == record["sha256"], path
        if path.suffix == ".parquet":
            names = pq.read_schema(path).names
            assert not {"donor","projid","ProjID","role_hash","Gene ID","gene_id"}.intersection(names), path
            if "module" in names:
                assert set(pd.read_parquet(path, columns=["module"]).module.astype(int)).issubset(expected)
            for gene in [c for c in ["gene_a","gene_b"] if c in names]:
                assert not pd.read_parquet(path, columns=[gene])[gene].astype(str).str.contains(r"ENSG\d", regex=True).any(), path
    for name, rows in [("aggregate_plot_data.parquet",173*357*6), ("resolved_plot_data.parquet",173*357*36)]:
        assert pq.ParquetFile(source / name).metadata.num_rows == rows
    annotations = pd.read_csv(source / "module_kegg_annotations.tsv", sep="\t")
    assert set(annotations.module) == expected
    kegg = pd.read_parquet(source / "kegg_tissue_expanded_full.parquet")
    assert set(kegg.cluster_id) == expected and kegg.groupby("cluster_id").size().eq(328).all()
    if final:
        status = json.loads((A.ROOT / "final_qa.json").read_text())
        assert status["status"] == "passed"
        label = json.loads((source / "mdc_label_validation/manifest.json").read_text())
        assert label["status"] == "complete" and label["module_count"] == 173
        assert label["completed_label_permutations"] == 200 and label["completed_balanced_resamples"] == 100
        assert len(pd.read_parquet(source / "mdc_label_validation/results.parquet")) == 3460
    return dict(status="passed", modules=173, donors=357, files=len(files), final=final)


def snapshot(final):
    manifest = json.loads((A.PUBLIC / "module_set_manifest.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = A.ROOT / "release_snapshots" / (stamp + ("_final" if final else "_core"))
    path.mkdir(parents=True, exist_ok=False)
    for name, record in manifest["files"].items():
        source = A.PUBLIC / name
        if not source.is_relative_to(A.PUBLIC) or ".." in Path(name).parts:
            raise ValueError("Unsafe public manifest path")
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert A.sha(target) == record["sha256"]
    manifest["release_status"] = "complete" if final else "core_complete_inference_pending"
    manifest["overall_catalog_interim"] = not final
    # These fixed modules were discovered in all donors, not in Controls alone.
    for catalog in ["mdc_sensitivity", "mdc_label_validation"]:
        if catalog in manifest:
            notes = manifest[catalog].get("notes", [])
            manifest[catalog]["notes"] = [n.replace("Modules were discovered in Controls.", "Modules were discovered in the all-donor cohort.") for n in notes]
    A.atomic_json(manifest, path / "module_set_manifest.json")
    report = validate_snapshot(path, manifest["files"], final)
    A.atomic_json(report, A.ROOT / "publication_qa" / (stamp + ".json"))
    return path


def publish(source, final, push):
    import pyarrow.parquet as pq
    label = "complete" if final else "core"
    staged = subprocess.check_output(["git","diff","--cached","--name-only"], cwd=APP, text=True).splitlines()
    if staged: raise RuntimeError("Unrelated staged changes require review")
    if subprocess.check_output(["git","branch","--show-current"],cwd=APP,text=True).strip() != "master":
        raise RuntimeError("Unexpected app branch")
    allowed = {"data/data_manifest.json", "drive_file_index.json"}
    dirty = subprocess.check_output(["git","diff","--name-only"],cwd=APP,text=True).splitlines()
    if any(p not in allowed and not p.startswith(f"data/{A.KEY}/") for p in dirty):
        raise RuntimeError("Unrelated tracked changes require review")
    rollback = A.ROOT / "public_rollback" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback.mkdir(parents=True,exist_ok=False)
    for name in allowed:
        file = APP / name
        if file.exists(): shutil.copy2(file, rollback / file.name)
    incoming = APP / "data" / (f"._{A.KEY}_{os.getpid()}")
    if incoming.exists(): raise RuntimeError("Pending publication directory already exists")
    shutil.copytree(source, incoming)
    if DESTINATION.exists(): DESTINATION.rename(rollback / A.KEY)
    incoming.rename(DESTINATION)
    committed=False
    try:
        # Upload first. The public selector is not registered until all indexed
        # files are remotely available and render tests pass.
        command([sys.executable,"scripts/upload_prediction_data.py","--oauth-client",OAUTH_CLIENT,
                 "--oauth-token","google-drive-oauth-token.json","--inventory-credentials",SERVICE_ACCOUNT,
                 "--root-folder-id",DRIVE_ROOT,"--local-prediction",DESTINATION,
                 "--remote-directory",A.KEY,"--manifest-name","module_set_manifest.json"])
        command([sys.executable,"scripts/update_muscle_drive_index.py","--credentials",SERVICE_ACCOUNT,
                 "--root-folder-id",DRIVE_ROOT,"--local-data","data","--output","drive_file_index.json",
                 ])
        config = A.configuration()
        B = A.engines(config)[2]
        manifest = json.loads((DESTINATION / "module_set_manifest.json").read_text())
        B.update_app_manifest(manifest, DESTINATION)
        command([sys.executable,"-m","pytest","-q","tests/test_streamlit_app.py::test_streamlit_all_donor_muscle_core_views_render",
                 "tests/test_streamlit_app.py::test_streamlit_muscle_mdc_label_validation_keeps_existing_catalog"])
        # Keep only compact tables and metadata in Git. All donor-score shards,
        # edge summaries, drivers and adjacency PNGs remain lazy on Drive.
        keep = []
        for path in DESTINATION.rglob("*"):
            if not path.is_file(): continue
            rel = path.relative_to(DESTINATION).as_posix()
            hot = path.suffix in {".json", ".tsv"} or "statistics" in path.name or rel.endswith("results.parquet") or rel.endswith("mask_retention_summary.parquet") or rel == "sample_metadata.parquet" or rel == "kegg_tissue_expanded_full.parquet" or rel in {"mdc_sensitivity/summary.parquet","mdc_sensitivity/resolved.parquet"}
            if "mdc_split_heatmaps" in rel and path.suffix == ".png": hot = False
            if hot:
                if path.stat().st_size >= 95*1024**2: raise ValueError("Oversized Git file")
                keep.append(str(path.relative_to(APP)))
        command(["git","add","-f","--",*keep])
        command(["git","add","-u","--","data/data_manifest.json","drive_file_index.json"])
        staged = subprocess.check_output(["git","diff","--cached","--name-only"],cwd=APP,text=True).splitlines()
        assert all(p in allowed or p.startswith(f"data/{A.KEY}/") for p in staged)
        if staged:
            command(["git","commit","-m",f"Publish {label} all-donor muscle L4 non-prediction results"])
        committed=True
        if push: confirm_push()
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=APP,text=True).strip()
    except Exception:
        if committed:
            # A failed push must leave the validated, push-ready commit intact.
            raise
        # Preserve failed candidate for inspection; restore the previous registry
        # and bundle instead of leaving a selectable, incomplete deployment.
        if DESTINATION.exists(): DESTINATION.rename(rollback / "failed_candidate")
        if (rollback / A.KEY).exists(): (rollback / A.KEY).rename(DESTINATION)
        for name in allowed:
            saved = rollback / Path(name).name
            if saved.exists(): shutil.copy2(saved, APP / name)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--detach",action="store_true")
    parser.add_argument("--push",action="store_true")
    args=parser.parse_args()
    if args.detach:
        with (A.ROOT/"publication.log").open("ab",buffering=0) as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--detach']],
                                  cwd=APP,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(publisher_pid=proc.pid)));return
    path=A.ROOT/"publication_status.json"
    record=json.loads(path.read_text()) if path.exists() else dict(status="waiting",milestones={})
    # Resume a verified commit if the previous process stopped only during its
    # push step. Never repeat data uploads merely because shell auth was absent.
    if args.push and not record['milestones'] and DESTINATION.exists():
        local=json.loads((DESTINATION/'module_set_manifest.json').read_text())
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=APP,text=True).strip()
        changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=APP,text=True).splitlines()
        if local.get('release_status')=='core_complete_inference_pending' and not changed:
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/master'],cwd=APP,text=True,
                                           env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}).split()
            if remote and remote[0]==head:
                record['milestones']['core']=dict(commit=head,pushed=True)
                record['status']='published_core'
                A.atomic_json(record,path)
    while True:
        run=json.loads((A.ROOT/"pipeline_manifest.json").read_text())
        done=run.get("stages",{})
        core=done.get("eigengenes",{}).get("status")=="complete"
        final=done.get("validate",{}).get("status")=="complete" and run['status']=='complete'
        milestone='final' if final else 'core' if core else None
        if milestone and milestone not in record['milestones']:
            try:
                record.update(status="validating_"+milestone,updated_utc=datetime.now(timezone.utc).isoformat())
                A.atomic_json(record,path)
                source=snapshot(final)
                commit=publish(source,final,args.push)
                record['milestones'][milestone]=dict(commit=commit,pushed=args.push)
                record.update(status="published_"+milestone if args.push else "committed_"+milestone,
                              updated_utc=datetime.now(timezone.utc).isoformat())
                A.atomic_json(record,path)
            except Exception as e:
                record.update(status="needs_review",error=str(e),updated_utc=datetime.now(timezone.utc).isoformat())
                A.atomic_json(record,path);raise
        if final or not args.wait:return
        if run['status']=='needs_review':
            record.update(status="analysis_needs_review",error=run.get('error',''))
            A.atomic_json(record,path);return
        time.sleep(30)


if __name__=='__main__':main()
