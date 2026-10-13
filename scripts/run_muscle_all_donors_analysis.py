#!/usr/bin/env python3
"""Isolated, resumable non-prediction analysis of the all-donor muscle table_3.

Reuse the validated muscle engines with an explicit dataset configuration.
No Control-derived result, prediction run, or source partition is overwritten.
All stages run in separate processes; caches are guarded by source/code hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(REPO))
KEY = "muscle_all_donors_l4"
LABEL = "All-donor VH/SMA/QUAD M5 L4 modules (173)"
PARTITION = REPO / "results/se2_m5_muscle_all_genes/all_donors/20261012_all_donors_fitted_betas"
ASSIGNMENTS = PARTITION / "modules_filtered_min30/M5_all_genes_table_3.csv"
DETAILS = PARTITION / "modules_filtered_min30/M5_all_genes_details_3.csv"
ROOT = REPO / "out/muscle_all_donors_l4/20261013_nonprediction"
PUBLIC = ROOT / "public_staging" / KEY
LIONESS = ROOT / "lioness"
MDC = ROOT / "mdc_m5"
SENSITIVITY = ROOT / "mdc_sensitivity"
VALIDATION = ROOT / "mdc_label_validation"
EFFECT = ROOT / "effect_edges"
KEGG = ROOT / "kegg_full"
METHODS = ("control_anchored",)
STAGES = ("qc", "kegg", "lioness", "mdc_observed", "public_core", "eigengenes",
          "mdc_sensitivity", "mdc_validation", "heatmaps", "effect_edges",
          "effect_public", "public_final", "validate")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def configuration():
    import pandas as pd
    frame = pd.read_csv(ASSIGNMENTS)
    beta = json.loads((PARTITION / "beta_selection.json").read_text())
    status = json.loads((PARTITION / "runner_status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Source partition is unfinished")
    if len(frame) != 31948 or frame["Cluster ID"].nunique() != 173:
        raise ValueError("Unexpected biological L4 partition; audit before accepting a changed source")
    if frame.duplicated(["Tissue", "Gene ID"]).any() or frame.duplicated(["Tissue", "Gene Symbol"]).any():
        raise ValueError("Duplicate tissue-gene assignments")
    if frame.groupby("Cluster ID").size().min() < 30:
        raise ValueError("Unfiltered small modules found")
    if set(frame.Tissue) != {"VH", "SMA", "QUAD"}:
        raise ValueError("Unexpected tissues")
    return dict(module_set=KEY, label=LABEL, modules=173, assignments=31948,
                methods=list(METHODS), biological_level=4, zero_indexed_table=3,
                beta_ts=beta["beta_ts"], beta_ct=beta["beta_ct"],
                sources={p.name: sha(p) for p in [ASSIGNMENTS, DETAILS, PARTITION / "beta_selection.json"]},
                partition_cohort="All donors", inference_caveat="Fixed all-donor modules; discovery-selection bias is not corrected.",
                prediction=False, bonobo=False)


def engines(config):
    """Bind each legacy engine to one explicit namespace before starting work."""
    L = importlib.import_module("scripts.run_muscle_lioness")
    M = importlib.import_module("scripts.run_muscle_m5_mdc")
    B = importlib.import_module("scripts.build_muscle_public_data")
    # The older edge scripts use bare imports; make their dependencies the same
    # configured module instances instead of accidentally loading another copy.
    sys.modules["run_muscle_lioness"] = L
    sys.modules["run_muscle_m5_mdc"] = M
    sys.modules["build_muscle_public_data"] = B
    for mod in (L, M):
        mod.DEFAULT_ASSIGNMENTS = ASSIGNMENTS
        mod.BETAS_TS = dict(config["beta_ts"])
        mod.BETAS_CT = dict(config["beta_ct"])
    L.DEFAULT_OUTPUT = LIONESS
    M.DEFAULT_OUTPUT = MDC
    M.MODULE_COUNT = config["modules"]
    M.M5_PARAMETERS = {**M.M5_PARAMETERS, "beta_ts": M.BETAS_TS, "beta_ct": M.BETAS_CT}
    B.RUN_ROOT, B.ASSIGNMENTS, B.OUTPUT = LIONESS, ASSIGNMENTS, PUBLIC
    B.KEGG_ROOT, B.MDC_ROOT = KEGG, MDC
    B.MDC_SIGNIFICANCE_ROOT = MDC / "significance_200perms"
    B.MODULE_SET, B.MODULE_COUNT, B.ASSIGNMENT_COUNT = KEY, config["modules"], config["assignments"]
    B.MODULE_LABEL, B.PARTITION_COHORT = LABEL, "All donors"
    B.BETAS_TS, B.BETAS_CT = M.BETAS_TS, M.BETAS_CT
    B.METHODS = METHODS
    # Same 357 donors: retain the existing private map, and never publish it.
    S = importlib.import_module("scripts.run_muscle_m5_mdc_significance")
    R = importlib.import_module("scripts.run_muscle_mdc_sensitivity")
    R.DEFAULT_OUTPUT, R.MODULE_SET, R.PUBLIC_ROOT = SENSITIVITY, KEY, PUBLIC
    V = importlib.import_module("scripts.run_muscle_mdc_label_validation")
    V.ROOT, V.PUBLIC = VALIDATION, PUBLIC / "mdc_label_validation"
    V.MODULE_SET, V.MODULE_COUNT, V.DISCOVERY_NOTE = KEY, config["modules"], config["inference_caveat"]
    E = importlib.import_module("scripts.run_muscle_effect_size_edges")
    E.DEFAULT_ASSIGNMENTS, E.DEFAULT_OUTPUT, E.MODULE_DEFINITION = ASSIGNMENTS, EFFECT, KEY
    P = importlib.import_module("scripts.build_muscle_effect_size_public_data")
    P.ASSIGNMENTS, P.RUN_ROOT, P.MODULE_SET = ASSIGNMENTS, LIONESS, KEY
    P.MODULE_COUNT, P.DEFAULT_ANALYSIS, P.DEFAULT_OUTPUT = config["modules"], EFFECT, PUBLIC
    P.RUN_NAME = "20261013_muscle_all_donors_hedges_g_040"
    P.METHODS = METHODS
    return L, M, B, S, R, V, E, P


def call_main(module, argv):
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *[str(a) for a in argv]]
        module.main()
    finally:
        sys.argv = previous


def qc(config, L, B):
    import numpy as np
    import pandas as pd
    assignments = L.read_assignments(ASSIGNMENTS, 30)
    cohort = L.load_cohort(L.DEFAULT_EXPRESSION, assignments)
    detail = B.module_details(assignments)
    assert detail.module_size.sum() == config["assignments"]
    assert detail.tissue_entropy_normalized.between(0, 1).all()
    source = pd.read_csv(DETAILS).set_index("Cluster ID")
    for tissue, column in [("VH", "n_genes_vh"), ("SMA", "n_genes_sma"), ("QUAD", "n_genes_quad")]:
        np.testing.assert_array_equal(detail.set_index("module")[column].sort_index(),
                                      source[tissue].fillna(0).sort_index())
    sizes = assignments.groupby("Cluster ID").size().sort_values()
    checked = []
    for module in [int(sizes.index[0]), int(sizes.index[-1])]:
        matrix, tissues, _ = L.module_matrix(cohort, assignments, module)
        i, j, powers, _ = L.edge_metadata(tissues)
        control = np.asarray(cohort.diagnoses) == "Control"
        reference = matrix[control]
        def adjacency(x):
            corr = np.nan_to_num(np.corrcoef(x, rowvar=False)[i, j], nan=0.)
            return L.signed_power(corr, powers)
        baseline = adjacency(reference)
        requested = {g: next(i for i, d in enumerate(cohort.diagnoses) if d == g)
                     for g in ["Control", "MCI", "AD"]}
        for ids, weights in L.method_weights(matrix, np.asarray(cohort.diagnoses), "control_anchored", i, j, powers, 4):
            for pos, donor in enumerate(ids):
                if int(donor) not in requested.values():
                    continue
                if control[donor]:
                    ri = list(np.flatnonzero(control)).index(donor)
                    direct = 120 * baseline - 119 * adjacency(np.delete(reference, ri, axis=0))
                else:
                    direct = 121 * adjacency(np.vstack([reference, matrix[donor]])) - 120 * baseline
                np.testing.assert_allclose(weights[pos], direct, rtol=1e-8, atol=1e-8)
                checked.append(dict(module=module, diagnosis=cohort.diagnoses[donor], max_edge_error=float(np.max(np.abs(weights[pos]-direct)))))
                requested.pop(cohort.diagnoses[donor])
            if not requested:
                break
    PUBLIC.mkdir(parents=True, exist_ok=True)
    B.atomic_tsv(detail, PUBLIC / "module_details.tsv")
    atomic_json(dict(status="passed", **config, donors=len(cohort.donors),
                     diagnosis_counts=pd.Series(cohort.diagnoses).value_counts().to_dict(), direct_formula_checks=checked), ROOT / "input_qa.json")


def run_kegg():
    import pandas as pd
    from models.kegg_enrichment.gene_sets import load_kegg_pathway_names, parse_kegg_brite_json, build_pathway_annotation_table, save_pathway_annotation_table
    resources = REPO / "resources/kegg"
    KEGG.mkdir(parents=True, exist_ok=True)
    annotation = KEGG / "pathway_annotations.tsv"
    names = load_kegg_pathway_names(str(resources / "kegg_list_pathway_hsa.tsv"))
    lookup = parse_kegg_brite_json(str(resources / "br08901.json"))
    save_pathway_annotation_table(build_pathway_annotation_table(names, lookup), str(annotation))
    command = [sys.executable, str(REPO / "scripts/python/kegg_multi_enrichment.py"),
               "--gene2ensembl", str(resources / "gene2ensembl.gz"),
               "--kegg-link", str(resources / "kegg_link_hsa_pathway.tsv"),
               "--kegg-list", str(resources / "kegg_list_pathway_hsa.tsv"),
               "--cluster-table", str(ASSIGNMENTS), "--cluster-details", str(DETAILS),
               "--gene-col", "Gene ID", "--keep-types", "ALL", "--outdir", str(KEGG),
               "--min-cluster-size", "30", "--min-term-size", "10", "--min-overlap", "3",
               "--kegg-annotations", str(annotation), "--expr-map"]
    command.extend(f"{t}={REPO}/data/raw/ROSMAP_muscle_split/ROSMAP_tissue_{t.lower()}_Control.csv" for t in ["VH", "SMA", "QUAD"])
    subprocess.run(command, check=True, cwd=REPO)
    regional = pd.read_csv(KEGG / "method2_meta_kegg_annotated.tsv", sep="\t")
    assert regional.cluster_id.nunique() == 173


def register_local(B):
    """Refresh staged metadata only; deployment is a separate validated action."""
    manifest = B.refresh_existing_manifest(PUBLIC)
    manifest["module_set"] = KEY
    manifest["partition_network"]["beta_ts"] = B.BETAS_TS
    manifest["partition_network"]["beta_ct"] = B.BETAS_CT
    manifest["capabilities"]["prediction"] = False
    for name in ["mdc_sensitivity", "mdc_label_validation"]:
        path = PUBLIC / name / "manifest.json"
        if path.exists():
            manifest[name] = json.loads(path.read_text())
    atomic_json(manifest, PUBLIC / "module_set_manifest.json")


def original_inference_export(B):
    """Reuse the original variant's completed nulls, without a duplicate null run."""
    import pandas as pd
    source = SENSITIVITY / "mdc_significance.parquet"
    frame = pd.read_parquet(source, filters=[("normalization_variant", "=", "original")])
    if len(frame) != 173 * 9 or frame.module.nunique() != 173:
        raise ValueError("Incomplete original-functional null results")
    status = json.loads((SENSITIVITY / "run_manifest.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Original null inference is pending")
    B.atomic_parquet(frame, B.MDC_SIGNIFICANCE_ROOT / "mdc_significance.parquet")
    atomic_json(dict(status="complete", sample_permutations=200, gene_permutations=200,
                     completed_sample_checkpoints=200, completed_gene_checkpoints=200,
                     derived_from="original variant of the ten-functional sensitivity catalog",
                     source_sha256=sha(source)), B.MDC_SIGNIFICANCE_ROOT / "run_manifest.json")
    path = PUBLIC / "mdc_sensitivity/manifest.json"
    public = json.loads(path.read_text())
    public["notes"] = [note.replace("Modules were discovered in Controls.", "Modules were discovered in the all-donor cohort.") for note in public["notes"]]
    atomic_json(public, path)


def validate(config, L):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    for method in METHODS:
        frame = pd.read_parquet(LIONESS / method / "donor_module_metrics.parquet")
        assert len(frame) == 173 * 357 and frame.module.nunique() == 173
        assert frame.groupby("module").size().eq(357).all()
        assert not frame.duplicated(["module", "donor"]).any()
        L.validate_feature_identities(frame)
    modules = set(pd.read_csv(ASSIGNMENTS)["Cluster ID"])
    for path in PUBLIC.rglob("*.parquet"):
        schema = pq.read_schema(path)
        assert not {"donor", "projid", "ProjID", "Gene ID", "gene_id", "role_hash"}.intersection(schema.names), path
        assert path.stat().st_size < 95 * 1024**2, path
        if "module" in schema.names:
            values = pq.read_table(path, columns=["module"]).column(0).to_numpy()
            assert set(values).issubset(modules), path
    for name, rows in [("aggregate_plot_data.parquet", 173*357*6), ("resolved_plot_data.parquet", 173*357*36)]:
        assert pq.ParquetFile(PUBLIC / name).metadata.num_rows == rows
    from scripts import validate_muscle_mdc_label_results as check
    check.PUBLIC, check.PRIVATE = PUBLIC / "mdc_label_validation", VALIDATION
    check.main()
    atomic_json(dict(status="passed", module_set=KEY, modules=173, donors=357, source_hashes=config["sources"]), ROOT / "final_qa.json")


def stage(name, workers):
    config = configuration()
    L, M, B, S, R, V, E, P = engines(config)
    if name == "qc": qc(config, L, B)
    elif name == "kegg": run_kegg()
    elif name == "lioness": call_main(L, ["--methods", *METHODS, "--workers", min(workers, 4)])
    elif name == "mdc_observed": call_main(M, [])
    elif name == "public_core": call_main(B, [])
    elif name == "eigengenes": call_main(B, ["--eigengenes-only"])
    elif name == "mdc_sensitivity": call_main(R, ["--workers", workers, "--public-root", PUBLIC / "mdc_sensitivity"])
    elif name == "mdc_validation": call_main(V, ["--workers", workers])
    elif name == "heatmaps":
        H = importlib.import_module("scripts.build_muscle_mdc_heatmaps")
        for variant in ["original", "m5_high_no_q90"]:
            call_main(H, ["--assignments", ASSIGNMENTS, "--mdc-root", MDC, "--sensitivity-root", SENSITIVITY,
                          "--normalization-variant", variant, "--output", PUBLIC / "mdc_split_heatmaps" / ("all_edges" if variant == "original" else variant)])
    elif name == "effect_edges": call_main(E, ["--methods", *METHODS, "--workers", min(workers, 4)])
    elif name == "effect_public":
        # The general packager refreshes the global app manifest; suppress that
        # one callback while building the private staging directory.
        P.update_app_manifest = lambda manifest, output: None
        call_main(P, ["--statistics-workers", min(workers, 4)])
    elif name == "public_final":
        original_inference_export(B)
        call_main(B, ["--refresh-mdc-only"])
        register_local(B)
    elif name == "validate": validate(config, L)
    else: raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--accept-pre-mdc-fix", action="store_true",
                        help="Revalidate already completed QC/KEGG/LIONESS when adapting the legacy MDC module-count guard.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError("Use 1–8 workers to preserve prediction resources")
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (ROOT / "pipeline.log").open("ab", buffering=0) as log:
            command=[sys.executable, str(Path(__file__).resolve()), "--stage", args.stage, "--workers", str(args.workers)]
            if args.accept_pre_mdc_fix: command.append("--accept-pre-mdc-fix")
            proc = subprocess.Popen(command,
                                    cwd=APP, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})
        print(json.dumps(dict(pid=proc.pid, status="started", root=str(ROOT))))
        return
    if args.stage != "all": stage(args.stage, args.workers); return
    with (ROOT / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = configuration()
        code_paths = [Path(__file__), *[APP / "scripts" / f"{name}.py" for name in ["build_muscle_public_data", "build_muscle_effect_size_public_data", "run_muscle_lioness", "run_muscle_m5_mdc", "run_muscle_mdc_sensitivity", "run_muscle_mdc_label_validation", "run_muscle_effect_size_edges"]]]
        config["code_hashes"] = {p.name: sha(p) for p in code_paths}
        identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        path = ROOT / "pipeline_manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else dict(stages={})
        if manifest.get("config_hash", identity) != identity:
            old=manifest["configuration"].copy();old.pop("code_hashes",None)
            new=config.copy();new.pop("code_hashes",None)
            completed={k for k,v in manifest['stages'].items() if v.get('status')=='complete'}
            if not args.accept_pre_mdc_fix or old!=new or not completed.issubset({'qc','kegg','lioness'}) or manifest.get('active_stage')!='mdc_observed':
                raise ValueError("Pipeline source/code changed; review provenance before resuming")
            if list((MDC/'recovered_modules').rglob('M*.npz')):
                raise ValueError("Cannot adopt new MDC code over already computed networks")
            atomic_json(manifest,ROOT/'provenance_revisions/pre_mdc_count_guard.json')
            # Revalidate completed numerical features under the unchanged input
            # and beta configuration. Completed source artifacts are not deleted.
            import pandas as pd
            L=engines(config)[0]
            frame=pd.read_parquet(LIONESS/'control_anchored/donor_module_metrics.parquet')
            assert len(frame)==173*357 and frame.module.nunique()==173
            L.validate_feature_identities(frame)
            stage('qc',args.workers)
        manifest.update(configuration=config, config_hash=identity, status="running", pid=os.getpid())
        try:
            for name in STAGES:
                if manifest["stages"].get(name, {}).get("status") == "complete": continue
                manifest.update(active_stage=name, updated_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(manifest, path)
                start = time.monotonic()
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", name, "--workers", str(args.workers)], check=True, cwd=APP)
                manifest["stages"][name] = dict(status="complete", seconds=round(time.monotonic()-start, 3))
                atomic_json(manifest, path)
            manifest.update(status="complete", updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(manifest, path)
        except Exception as error:
            manifest.update(status="needs_review", error=str(error), updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(manifest, path)
            raise


if __name__ == "__main__": main()
