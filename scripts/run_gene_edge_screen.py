#!/usr/bin/env python3
"""Resumable offline screening. Never uploads, pushes or stores full networks.

Use --stage first-release for genes and brain module edges. Full M5 and whole
networks are deliberately gated on a measured pilot and explicit review.
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
import multiprocessing
from pathlib import Path
import resource
import shutil
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1]), str(APP/"scripts")]
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from app_helpers.gene_symbols import gene_symbol, ENSEMBL_PATTERN
from app_helpers.outcome_screening import (bh, correlations, edge_batches, selected_union, signed_alt_edges, top_hits)
from scripts.screening_sources import (OUTCOMES, PARTITIONS, REPO, assignments, component, load_sources, sha)
from scripts.screening_bh import external_bh

RUN = REPO/"out/gene_edge_outcome_screen/20261022_original_and_validation"
VERSION = 1


def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n")
    temporary.replace(path)


def atomic_parquet(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    frame.to_parquet(temporary, compression="zstd", index=False, row_group_size=32768)
    temporary.replace(path)


def resource_guard(root, budget_gib=32):
    import psutil
    process = psutil.Process(int(os.environ.get("SCREEN_COORDINATOR_PID", os.getpid())))
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if rss > budget_gib*1024**3:
        raise MemoryError("Combined screening RSS exceeds the 32-GiB acceptance budget")
    scratch = Path(root)/"scratch"
    used = 0
    if scratch.exists():
        for p in scratch.rglob("*"):
            try:
                if p.is_file():
                    used += p.stat().st_size
            except FileNotFoundError:  # another checkpoint atomically renamed
                pass
    if used > 500*1024**3:
        raise OSError("Screening scratch budget exceeds 500 GiB")
    if shutil.disk_usage(root).free < 25*1024**3:
        raise OSError("Insufficient reserve storage")
    return dict(current_rss_bytes=rss, scratch_bytes=used)


def audit(root):
    rows = []; hashes = {}; component_rows = []
    for dataset in ("brain", "motor"):
        source = load_sources(dataset)
        for tissue, expression in source.expression.items():
            for role in ("original", "validation"):
                ids = source.donors((tissue,), validation=role == "validation")
                meta = source.metadata.loc[ids]
                rows.append(dict(dataset=dataset, tissue=tissue, cohort=role, donors=len(ids), genes=len(expression.columns),
                    **{f"n_{g.lower()}":int(meta.diagnosis_group.eq(g).sum()) for g in ("Control", "MCI", "AD", "Unclassified")},
                    **{f"n_{outcome}":int(meta[outcome].notna().sum()) for outcome in OUTCOMES}))
        for p in source.paths:
            hashes[str(p)] = sha(p)
        tissues = list(source.expression)
        expected = [266,752,195,230,14,176] if dataset == "brain" else [110,88,67,58,40,25]
        components = [(t,) for t in tissues]+[(tissues[i], tissues[j]) for i in range(len(tissues)) for j in range(i+1,len(tissues))]
        for required, expected_n in zip(components, expected, strict=True):
            ids = source.donors(required, True)
            if len(ids) != expected_n:
                raise ValueError("Validation tissue/pair source counts changed")
            meta = source.metadata.loc[ids]
            component_rows.append(dict(dataset=dataset, component="–".join(required), donors=len(ids),
                **{f"n_{g.lower()}":int(meta.diagnosis_group.eq(g).sum()) for g in ("Control", "MCI", "AD", "Unclassified")}))
    module_rows = []
    for partition, (expected, relative, methods) in PARTITIONS.items():
        frame = assignments(partition)
        source = load_sources("motor" if partition.startswith("muscle_") else "brain")
        for tissue, group in frame.groupby("tissue"):
            if not set(group.gene) <= set(source.expression[tissue].columns):
                raise ValueError("Assignment expression mapping is incomplete")
        for module, group in frame.groupby("module"):
            module_rows.append(dict(partition=partition, module=int(module), source_node_count=int(group.source_node_count.iloc[0]), unique_node_count=len(group),
                n_edges=len(group)*(len(group)-1)//2))
        hashes[str(REPO/relative)] = sha(REPO/relative)
    identity = hashlib.sha256(json.dumps(dict(version=VERSION, sources=hashes, outcomes=OUTCOMES), sort_keys=True).encode()).hexdigest()
    target = Path(root)/"audit.json"
    if target.exists() and json.loads(target.read_text()).get("identity") != identity:
        raise ValueError("Source audit changed: use a new output root, not old checkpoints")
    record = dict(version=VERSION, status="passed", identity=identity, sources=hashes, outcomes=OUTCOMES,
        cohorts=rows, nodes="Unique physical tissue-gene nodes; preserve all input module IDs, including M188",
        validation="Brain excludes all 464 complete-expression donors; motor excludes 357 complete donors",
        ram_budget_gib=32, scratch_budget_gib=500, workers_max=2,
        original_fdr="Full valid search, before top-K", validation_fdr="Prespecified Pearson/Spearman selected-hit union",
        public_donor_publication="not_authorized_for_this_experiment", full_m5_genome_gate="pilot_and_user_review_required")
    atomic_json(record, target)
    pd.DataFrame(rows).to_csv(Path(root)/"cohort_audit.tsv", sep="\t", index=False)
    pd.DataFrame(component_rows).to_csv(Path(root)/"validation_component_audit.tsv", sep="\t", index=False)
    pd.DataFrame(module_rows).to_csv(Path(root)/"module_audit.tsv", sep="\t", index=False)
    return record


def require_audit(root):
    record = json.loads((Path(root)/"audit.json").read_text())
    if record["status"] != "passed":
        raise ValueError("Source audit is required")
    return record["identity"]


def stats_for_selected(source, original_ids, validation_ids, original_values, validation_values, selected):
    """Selected values have one column per physical hit, shared across outcomes."""
    if selected.empty:
        return selected.copy()
    rows = []
    for outcome, hits in selected.groupby("outcome", sort=False):
        indices = hits.value_index.to_numpy(int)
        valid = correlations(validation_values[:, indices], source.metadata.loc[validation_ids, outcome].to_numpy(float))
        valid.index = hits.index
        result = hits.copy()
        for col in valid:
            result[f"validation_{col}"] = valid[col]
        for method in ("pearson", "spearman"):
            result[f"validation_{method}_selected_fdr"], n = bh(valid[f"{method}_p"])
            result[f"validation_{method}_tested_n"] = n
            a, b = result[f"{method}_r"], result[f"validation_{method}_r"]
            result[f"{method}_direction_agreement"] = np.where(a.notna() & b.notna(), np.sign(a) == np.sign(b), None)
        # Diagnosis checks are descriptive, only on frozen pooled hits.
        for role, ids, values in (("original", original_ids, original_values), ("validation", validation_ids, validation_values)):
            labels = source.metadata.loc[ids, "diagnosis_group"].to_numpy()
            y = source.metadata.loc[ids, outcome].to_numpy(float)
            for group in ("Control", "MCI", "AD"):
                mask = labels == group
                stats = correlations(values[mask][:, indices], y[mask])
                for col in stats:
                    result[f"{role}_{group.lower()}_{col}"] = stats[col].to_numpy()
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def public_catalog(frame):
    """Aggregate statistics only, no donor rows or internal Ensembl identifiers."""
    result = frame.drop(columns=["value_index", "gene", "gene_left", "gene_right", "edge_index"], errors="ignore").copy()
    result["identity"] = result.identity.map(lambda v: hashlib.sha256(str(v).encode()).hexdigest()[:24])
    for col in result.select_dtypes(include=["object", "string"]):
        if result[col].dropna().astype(str).str.contains(ENSEMBL_PATTERN).any():
            raise ValueError(f"Internal gene IDs remain in public {col}")
    if {"sample_id", "donor", "projid"} & set(result):
        raise ValueError("This release is aggregate only")
    return result


def genes(root):
    audit_id = require_audit(root)
    for dataset in ("brain", "motor"):
        source = load_sources(dataset)
        for tissue, expression in source.expression.items():
            checkpoint = Path(root)/"checkpoints/genes"/dataset/f"{tissue}.json"
            if checkpoint.exists() and json.loads(checkpoint.read_text()).get("audit_identity") == audit_id:
                continue
            started = time.monotonic()
            ids = source.donors((tissue,)); new_ids = source.donors((tissue,), True)
            order = sorted(expression.columns)
            original = expression.loc[ids, order].to_numpy(float)
            validation = expression.loc[new_ids, order].to_numpy(float)
            raw, chosen = [], []
            for outcome in OUTCOMES:
                stats = correlations(original, source.metadata.loc[ids, outcome].to_numpy(float))
                stats["identity"] = [f"{tissue}:{g}" for g in order]
                stats["gene"] = order; stats["gene_symbol"] = stats.gene.map(gene_symbol)
                stats["outcome"] = outcome; stats["value_index"] = np.arange(len(order))
                for method in ("pearson", "spearman"):
                    stats[f"{method}_fdr"], n = bh(stats[f"{method}_p"])
                    stats[f"{method}_tested_n"] = n
                chosen.append(selected_union(stats, 50))
                raw.append(stats)
                resource_guard(root)
            selected = stats_for_selected(source, ids, new_ids, original, validation, pd.concat(chosen, ignore_index=True))
            selected["dataset"] = dataset; selected["tissue"] = tissue
            selected["analysis"] = "genes"; selected["network_method"] = "not_applicable"
            selected["normalization_profile"] = "processed_expression"
            selected["original_fdr_scope"] = "all valid genes within this tissue"
            selected["validation_fdr_scope"] = "frozen selected-hit union within tissue and outcome"
            atomic_parquet(pd.concat(raw, ignore_index=True), Path(root)/"private/gene_tests"/dataset/f"{tissue}.parquet")
            atomic_parquet(selected, Path(root)/"private/gene_hits"/dataset/f"{tissue}.parquet")
            public = public_catalog(selected)
            destination = Path(root)/"public_bundle/genes"/dataset/f"{tissue}.parquet"
            atomic_parquet(public, destination)
            public.to_csv(destination.with_suffix(".tsv"), sep="\t", index=False)
            atomic_json(dict(status="complete", audit_identity=audit_id, rows=len(public), sha256=sha(destination), seconds=time.monotonic()-started), checkpoint)
            print(json.dumps(dict(stage="genes", dataset=dataset, tissue=tissue, rows=len(public), seconds=round(time.monotonic()-started, 1))), flush=True)
            catalog_manifest(root)


def module_job(root, partition, method, module, edge_batch=2048):
    """One bounded original module pass, then selected-hit-only rescoring."""
    root = Path(root); audit_id = require_audit(root)
    base = root/"checkpoints/module_edges"/partition/method/f"M{module}"
    marker = base.with_suffix(".json")
    if marker.exists() and json.loads(marker.read_text()).get("audit_identity") == audit_id:
        return dict(module=module, cached=True)
    started = time.monotonic()
    source = load_sources("brain")
    table = assignments(partition)
    all_modules = sorted(table.module.unique())
    nodes = table.loc[table.module.eq(module)].copy()
    names = list(zip(nodes.tissue, nodes.gene, strict=True))
    ids = source.original
    expression = np.column_stack([source.expression[t].loc[ids, g].to_numpy(float) for t, g in names])
    reference_ids = ids if method == "standard" else source.controls
    reference_positions = {d:i for i, d in enumerate(reference_ids)}
    reference = expression[[ids.index(d) for d in reference_ids]]
    positions = np.asarray([reference_positions.get(d, -1) for d in ids])
    centered = reference-reference.mean(axis=0)
    variance = np.sum(centered*centered, axis=0); mean = reference.mean(axis=0)
    offset = sum(int(n*(n-1)//2) for m in all_modules if m < module for n in [int(table.module.eq(m).sum())])
    p_path = root/"scratch/module_pvalues"/partition/method/f"M{module}.parquet"
    p_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = p_path.with_suffix(".partial")
    top = {o:pd.DataFrame() for o in OUTCOMES}
    outcomes = {o:source.metadata.loc[ids, o].to_numpy(float) for o in OUTCOMES}
    writer = None; edge_index = 0
    try:
        for left, right in edge_batches(len(nodes), edge_batch):
            cross = np.sum(centered[:, left]*centered[:, right], axis=0)
            powers = np.asarray([3. if names[i][0] == names[j][0] else 2. for i, j in zip(left, right, strict=True)])
            weights = signed_alt_edges(reference, expression, left, right, powers, reference_positions=positions,
                batch_statistics=(mean, variance, cross))
            edge_indices = np.arange(edge_index, edge_index+len(left), dtype=np.int64)
            block = dict(test_id=offset+edge_indices)
            identity = np.asarray([f"{names[i][0]}:{names[i][1]}|{names[j][0]}:{names[j][1]}" for i, j in zip(left, right, strict=True)])
            for outcome in OUTCOMES:
                stats = correlations(weights, outcomes[outcome])
                block[f"{outcome}__pearson_p"] = stats.pearson_p.to_numpy()
                block[f"{outcome}__spearman_p"] = stats.spearman_p.to_numpy()
                stats["identity"] = identity; stats["test_id"] = offset+edge_indices
                stats["edge_index"] = edge_indices; stats["left_index"] = left; stats["right_index"] = right
                combined = pd.concat([top[outcome], stats], ignore_index=True)
                keep = set(top_hits(combined, 10, "pearson").identity) | set(top_hits(combined, 10, "spearman").identity)
                top[outcome] = combined.loc[combined.identity.isin(keep)]
            arrow = pa.table(block)
            if writer is None:
                writer = pq.ParquetWriter(temporary, arrow.schema, compression="zstd")
            writer.write_table(arrow)
            edge_index += len(left)
            if edge_index % (edge_batch*32) == 0:
                resource_guard(root)
        if writer:
            writer.close(); writer = None
        temporary.replace(p_path)
    finally:
        if writer:
            writer.close()
    all_hits = []
    for outcome in OUTCOMES:
        selected = selected_union(top[outcome], 10)
        selected["outcome"] = outcome
        for method_r in ("pearson", "spearman"):
            column = f"{outcome}__{method_r}_p"
            pvalues = pq.read_table(p_path, columns=[column])[column].to_numpy()
            q, n = bh(pvalues)
            selected[f"{method_r}_fdr"] = q[selected.edge_index.to_numpy(int)]
            selected[f"{method_r}_tested_n"] = n
        all_hits.append(selected)
    selected = pd.concat(all_hits, ignore_index=True)
    unique = selected[["identity", "left_index", "right_index"]].drop_duplicates().sort_values("identity")
    unique["component"] = [component(names[int(i)][0], names[int(j)][0]) for i, j in zip(unique.left_index, unique.right_index, strict=True)]
    all_original = []; all_validation = []; selected_parts = []; weight_parts = []
    for c, group in unique.groupby("component"):
        tissues = tuple(dict.fromkeys([names[int(i)][0] for i in [group.left_index.iloc[0], group.right_index.iloc[0]]]))
        new_ids = source.donors(tissues, True)
        new_expression = np.zeros((len(new_ids), len(nodes)), dtype=float)
        # Fill only required node columns. Absent tissues do not enter scoring.
        used = np.unique(np.r_[group.left_index, group.right_index]).astype(int)
        for index in used:
            tissue, gene = names[index]
            new_expression[:, index] = source.expression[tissue].loc[new_ids, gene].to_numpy(float)
        left, right = group.left_index.to_numpy(int), group.right_index.to_numpy(int)
        powers = np.where(np.asarray([names[i][0] for i in left]) == np.asarray([names[i][0] for i in right]), 3., 2.)
        cross = np.sum(centered[:, left]*centered[:, right], axis=0)
        old_values = signed_alt_edges(reference, expression, left, right, powers, reference_positions=positions, batch_statistics=(mean, variance, cross))
        new_values = signed_alt_edges(reference, new_expression, left, right, powers, batch_statistics=(mean, variance, cross))
        lookup = {identity:i for i, identity in enumerate(group.identity)}
        hits = selected.loc[selected.identity.isin(lookup)].copy()
        hits["value_index"] = hits.identity.map(lookup)
        result = stats_for_selected(source, ids, new_ids, old_values, new_values, hits)
        result["component"] = c
        selected_parts.append(result)
        for role, donor_ids, values in (("original", ids, old_values), ("validation", new_ids, new_values)):
            weight_parts.append(pd.DataFrame(dict(sample_id=np.repeat([source.mapping[d] for d in donor_ids], len(group)),
                identity=np.tile(group.identity.to_numpy(), len(donor_ids)), weight=values.ravel(), cohort=role, component=c)))
    # Validation BH is over the full selected module set, not separately split
    # into components. Unobserved components contribute missing, not zero.
    result = pd.concat(selected_parts, ignore_index=True) if selected_parts else selected.copy()
    for outcome, ix in result.groupby("outcome").groups.items():
        for r in ("pearson", "spearman"):
            result.loc[ix, f"validation_{r}_selected_fdr"], n = bh(result.loc[ix, f"validation_{r}_p"])
            result.loc[ix, f"validation_{r}_tested_n"] = n
    result["partition"] = partition; result["module"] = int(module)
    result["network_method"] = method; result["normalization_profile"] = "signedAlt_TS3_CT2"
    result["dataset"] = "brain"; result["analysis"] = "module_edges"
    result["source_node_count"] = int(nodes.source_node_count.iloc[0]); result["unique_node_count"] = len(nodes)
    result["gene_symbol_left"] = result.left_index.map(lambda i: gene_symbol(names[int(i)][1]))
    result["gene_symbol_right"] = result.right_index.map(lambda i: gene_symbol(names[int(i)][1]))
    result["tissue_left"] = result.left_index.map(lambda i: names[int(i)][0])
    result["tissue_right"] = result.right_index.map(lambda i: names[int(i)][0])
    result["original_fdr_scope"] = "all valid physical edges within module; partition-wide also retained"
    result["validation_fdr_scope"] = "frozen selected-hit union within module and outcome"
    atomic_parquet(result, base.with_suffix(".parquet"))
    saved_weights = pd.concat(weight_parts, ignore_index=True) if weight_parts else pd.DataFrame(columns=["sample_id", "identity", "weight", "cohort", "component"])
    atomic_parquet(saved_weights, root/"private/selected_weights"/partition/method/f"M{module}.parquet")
    record = dict(status="complete", audit_identity=audit_id, module=int(module), rows=len(result),
        tested_edges=edge_index, pvalue_sha256=sha(p_path), seconds=time.monotonic()-started,
        unavailable_reason="no_variable_edges_or_eligible_outcomes" if result.empty else "",
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(record, marker)
    return record


def brain_modules(root, partition, method, workers=2, limit_modules=None):
    if partition not in {"full_cohort", "control_derived"} or method not in PARTITIONS[partition][2]:
        raise ValueError("Unsupported brain module/method combination")
    table = assignments(partition); modules = sorted(map(int, table.module.unique()))
    if limit_modules:
        modules = modules[:limit_modules]
    load_sources("brain")  # fork read-only source frames, no repeated CSV I/O.
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        tasks = [executor.submit(module_job, str(root), partition, method, module) for module in modules]
        for task in as_completed(tasks):
            print(json.dumps(dict(stage="module_edges", partition=partition, network_method=method, **task.result())), flush=True)
    if limit_modules:
        return
    finalise_modules(root, partition, method)


def finalise_modules(root, partition, method):
    root = Path(root); count = PARTITIONS[partition][0]
    base = root/"checkpoints/module_edges"/partition/method
    markers = list(base.glob("M*.json"))
    if len(markers) != count or any(json.loads(p.read_text())["audit_identity"] != require_audit(root) for p in markers):
        raise ValueError("Partition correction requires every module checkpoint")
    selected = pd.concat([pd.read_parquet(p.with_suffix(".parquet")) for p in sorted(markers)], ignore_index=True)
    paths = sorted((root/"scratch/module_pvalues"/partition/method).glob("M*.parquet"))
    for outcome, ix in selected.groupby("outcome").groups.items():
        for r in ("pearson", "spearman"):
            adjusted, n = external_bh(paths, f"{outcome}__{r}_p", selected.loc[ix, "test_id"], temporary_directory=root/"scratch/duckdb", memory_gib=4)
            selected.loc[ix, f"{r}_partition_fdr"] = selected.loc[ix, "test_id"].map(adjusted)
            selected.loc[ix, f"{r}_partition_tested_n"] = n
            selected.loc[ix, f"validation_{r}_partition_selected_fdr"], nv = bh(selected.loc[ix, f"validation_{r}_p"])
            selected.loc[ix, f"validation_{r}_partition_tested_n"] = nv
        print(json.dumps(dict(stage="partition_BH", partition=partition, method=method, outcome=outcome)), flush=True)
    public = public_catalog(selected)
    destination = root/"public_bundle/module_edges"/partition/f"{method}.parquet"
    atomic_parquet(public, destination)
    public.to_csv(destination.with_suffix(".tsv"), sep="\t", index=False)
    atomic_json(dict(status="complete", modules=count, rows=len(public), sha256=sha(destination), audit_identity=require_audit(root)), base/"completion.json")
    catalog_manifest(root)


def catalog_manifest(root):
    root = Path(root); bundle = root/"public_bundle"
    entries = []
    for p in sorted(bundle.rglob("*.parquet")):
        file = pq.ParquetFile(p)
        if p.stat().st_size >= 95*1024**2:
            raise ValueError("Public aggregate shard exceeds 95 MiB; split before release")
        entries.append(dict(path=p.relative_to(bundle).as_posix(), rows=file.metadata.num_rows, bytes=p.stat().st_size,
            sha256=sha(p), status="complete"))
    record = dict(schema_version=1, status="interim", updated_at=datetime.now(timezone.utc).isoformat(),
        audit_identity=require_audit(root), outcomes=OUTCOMES, files=entries,
        genes_status="complete" if len(list((bundle/"genes").rglob("*.parquet"))) == 6 else "running",
        brain_module_edges_status="complete" if len(list((bundle/"module_edges").rglob("*.parquet"))) == 3 else "running",
        motor_module_edges_status="unavailable_pending_full_M5_pilot",
        whole_network_status="unavailable_pending_pilot_resource_review",
        donor_values_status="private_only_requires_separate_publication_approval")
    atomic_json(record, bundle/"manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--stage", choices=["audit", "genes", "brain-modules", "finalise", "first-release"], required=True)
    parser.add_argument("--partition", choices=["full_cohort", "control_derived"], default="control_derived")
    parser.add_argument("--method", choices=["standard", "control_anchored"], default="control_anchored")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--limit-modules", type=int)
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    os.environ["SCREEN_COORDINATOR_PID"] = str(os.getpid())
    record = audit(args.root)
    print(json.dumps(dict(stage="audit", status=record["status"], identity=record["identity"])), flush=True)
    if args.stage in {"genes", "first-release"}:
        genes(args.root)
    if args.stage == "brain-modules":
        brain_modules(args.root, args.partition, args.method, args.workers, args.limit_modules)
    if args.stage == "finalise":
        finalise_modules(args.root, args.partition, args.method)
    if args.stage == "first-release":
        for partition, method in (("control_derived", "control_anchored"), ("full_cohort", "control_anchored"), ("full_cohort", "standard")):
            brain_modules(args.root, partition, method, args.workers)


if __name__ == "__main__":
    main()
