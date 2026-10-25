#!/usr/bin/env python3
"""Independent sampled SciPy/direct-estimator QA for completed screening hits."""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr, false_discovery_control

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1]), str(APP/"scripts")]
from scripts.screening_sources import load_sources, OUTCOMES, assignments, sha
from scripts.run_gene_edge_screen import RUN, atomic_json, require_audit


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def validate_genes(root):
    checked = []
    for dataset in ("brain", "motor"):
        source = load_sources(dataset)
        for tissue, expression in source.expression.items():
            hits = pd.read_parquet(root/"private/gene_hits"/dataset/f"{tissue}.parquet")
            full = pd.read_parquet(root/"private/gene_tests"/dataset/f"{tissue}.parquet")
            old_ids = source.donors((tissue,)); new_ids = source.donors((tissue,), True)
            for outcome, rows in hits.groupby("outcome"):
                all_tests = full.loc[full.outcome.eq(outcome)].copy()
                for method in ("pearson", "spearman"):
                    expected_top = all_tests.loc[all_tests[f"{method}_r"].notna()].assign(absolute=lambda f:f[f"{method}_r"].abs()).sort_values(["absolute", "identity"], ascending=[False, True]).head(50)
                    actual_top = rows.loc[rows[f"{method}_rank"].notna()].sort_values(f"{method}_rank")
                    assert actual_top.identity.tolist() == expected_top.identity.tolist()
                    eligible = all_tests[f"{method}_p"].notna()
                    adjusted = pd.Series(false_discovery_control(all_tests.loc[eligible, f"{method}_p"].to_numpy()), index=all_tests.loc[eligible, "identity"])
                    close(rows[f"{method}_fdr"], rows.identity.map(adjusted))
                    assert set(rows[f"{method}_tested_n"]) == {int(eligible.sum())}
                    v = rows[f"validation_{method}_p"].notna()
                    q = false_discovery_control(rows.loc[v, f"validation_{method}_p"].to_numpy())
                    close(rows.loc[v, f"validation_{method}_selected_fdr"], q)
                for row in rows.sort_values("identity").head(3).itertuples():
                    for role, ids in (("original",old_ids),("validation",new_ids)):
                        x = expression.loc[ids,row.gene].to_numpy(float)
                        y = source.metadata.loc[ids,outcome].to_numpy(float)
                        mask = np.isfinite(x)&np.isfinite(y)
                        prefix = "" if role == "original" else "validation_"
                        for method, test in (("pearson",pearsonr),("spearman",spearmanr)):
                            answer = test(x[mask],y[mask])
                            close(getattr(row,f"{prefix}{method}_r"),answer.statistic)
                            close(getattr(row,f"{prefix}{method}_p"),answer.pvalue)
                checked.append(dict(dataset=dataset,tissue=tissue,outcome=outcome,selected_hits=len(rows),status="passed"))
    return checked


def validate_module_samples(root, partition, method):
    source = load_sources("brain"); table = assignments(partition)
    paths = sorted((root/"checkpoints/module_edges"/partition/method).glob("M*.parquet"))
    if not paths:
        return []
    checked = []
    for path in paths[:2]:
        hits = pd.read_parquet(path)
        weights = pd.read_parquet(root/"private/selected_weights"/partition/method/path.name)
        nodes = table.loc[table.module.eq(int(hits.module.iloc[0]))]
        identities = list(zip(nodes.tissue,nodes.gene,strict=True))
        ref_ids = source.original if method == "standard" else source.controls
        for row in hits.drop_duplicates("identity").head(3).itertuples():
            endpoints = [identities[int(row.left_index)],identities[int(row.right_index)]]
            ref = np.column_stack([source.expression[t].loc[ref_ids,g].to_numpy(float) for t,g in endpoints])
            power = 3. if endpoints[0][0] == endpoints[1][0] else 2.
            def adjacency(x):
                r = np.corrcoef(x,rowvar=False)[0,1]
                return np.sign(r)*abs(r)**power
            base = adjacency(ref); n = len(ref_ids)
            expected = []
            for d in source.original:
                values = np.asarray([source.expression[t].loc[d,g] for t,g in endpoints])
                if d in ref_ids:
                    score = n*base-(n-1)*adjacency(np.delete(ref,ref_ids.index(d),axis=0))
                else:
                    score = (n+1)*adjacency(np.vstack((ref,values)))-n*base
                expected.append(score)
            selected_weights = weights.loc[weights.identity.eq(row.identity)&weights.cohort.eq("original")].set_index("sample_id")
            close(selected_weights.loc[[source.mapping[d] for d in source.original],"weight"], expected)
            y = source.metadata.loc[source.original,row.outcome].to_numpy(float)
            mask = np.isfinite(y)
            for name,test in (("pearson",pearsonr),("spearman",spearmanr)):
                direct = test(np.asarray(expected)[mask],y[mask])
                close(getattr(row,f"{name}_r"),direct.statistic)
            checked.append(dict(partition=partition,network_method=method,module=int(row.module),status="passed"))
    return checked


def validate_completed_catalogs(root, search_outcomes=("cogn_global", "gpath", "sqrt_parksc_demog_slope")):
    """Reconcile every completed hit family and sampled full-search BH families.

    Independent SciPy correction materializes one p-value column (not weights)
    at a time: <=13.2 million float64 p-values for current brain partitions.
    The remaining full-search outcomes are covered by the same tested sorter;
    this is explicitly a sampled, not exhaustive independent recomputation.
    """
    root = Path(root)
    manifest = json.loads((root/"public_bundle/manifest.json").read_text())
    checked = []
    for entry in manifest["files"]:
        if not entry["path"].startswith("module_edges/"):
            continue
        path = root/"public_bundle"/entry["path"]
        frame = pd.read_parquet(path)
        _, partition, filename = entry["path"].split("/")
        method = Path(filename).stem
        table = assignments(partition)
        assert len(frame) == entry["rows"] and sha(path) == entry["sha256"]
        assert set(frame.module) == set(table.module)
        assert not frame.duplicated(["module", "outcome", "identity"]).any()
        assert set(frame.outcome) == set(OUTCOMES)
        assert {"donor", "projid", "sample_id"}.isdisjoint(frame.columns)
        sparse = frame.validation_n.lt(10)
        for correlation in ("pearson", "spearman"):
            for (_, _), group in frame.groupby(["module", "outcome"]):
                ranks = sorted(group[f"{correlation}_rank"].dropna().astype(int))
                assert ranks == list(range(1, 11))
                valid = group[f"validation_{correlation}_p"].notna()
                p = group.loc[valid, f"validation_{correlation}_p"].to_numpy(float)
                expected = false_discovery_control(p) if len(p) else np.array([])
                close(group.loc[valid, f"validation_{correlation}_selected_fdr"], expected)
                assert set(group[f"validation_{correlation}_tested_n"]) == {int(valid.sum())}
            assert frame.loc[sparse, f"validation_{correlation}_p"].isna().all()
            for _, group in frame.groupby("outcome"):
                valid = group[f"validation_{correlation}_p"].notna()
                p = group.loc[valid, f"validation_{correlation}_p"].to_numpy(float)
                expected = false_discovery_control(p) if len(p) else np.array([])
                close(group.loc[valid, f"validation_{correlation}_partition_selected_fdr"], expected)
                assert set(group[f"validation_{correlation}_partition_tested_n"]) == {int(valid.sum())}
        independent = []
        ppaths = sorted((root/"scratch/module_pvalues"/partition/method).glob("M*.parquet"))
        assert len(ppaths) == table.module.nunique()
        for outcome in search_outcomes:
            for correlation in ("pearson", "spearman"):
                column = f"{outcome}__{correlation}_p"
                pieces = [pq.read_table(p, columns=["test_id", column], use_threads=False).to_pandas() for p in ppaths]
                all_tests = pd.concat(pieces, ignore_index=True)
                assert not all_tests.test_id.duplicated().any()
                valid = all_tests[column].notna()
                expected = pd.Series(false_discovery_control(all_tests.loc[valid, column].to_numpy(float)),
                    index=all_tests.loc[valid, "test_id"])
                hits = frame.loc[frame.outcome.eq(outcome)]
                close(hits[f"{correlation}_partition_fdr"], hits.test_id.map(expected))
                assert set(hits[f"{correlation}_partition_tested_n"]) == {int(valid.sum())}
                # Independent within-module correction from the full tests,
                # covering all modules for these sampled outcome families.
                for p, tests in zip(ppaths, pieces, strict=True):
                    module = int(p.stem.removeprefix("M"))
                    module_hits = hits.loc[hits.module.eq(module)]
                    valid_m = tests[column].notna()
                    adjusted = pd.Series(false_discovery_control(tests.loc[valid_m, column].to_numpy(float)),
                        index=tests.loc[valid_m, "test_id"])
                    close(module_hits[f"{correlation}_fdr"], module_hits.test_id.map(adjusted))
                independent.append(dict(outcome=outcome, correlation=correlation, tested_edges=int(valid.sum())))
                print(json.dumps(dict(stage="independent_catalog_QA", partition=partition, method=method,
                    outcome=outcome, correlation=correlation, status="passed")), flush=True)
        checked.append(dict(path=entry["path"], sha256=entry["sha256"], rows=len(frame), modules=int(frame.module.nunique()),
            selected_validation_families="all outcomes/modules and partition families",
            independent_full_search_families=independent, status="passed"))
    record = dict(status="passed", audit_identity=require_audit(root), files=checked,
        independent_scope="All selected-set families; six full-search corrections per catalog, sampled outcomes, all modules")
    atomic_json(record, root/"qa/catalog_reconciliation.json")
    return record


def validate(root=RUN):
    root = Path(root)
    record = dict(status="passed",audit_identity=require_audit(root),gene_families=validate_genes(root),module_samples=[])
    for partition,method in (("control_derived","control_anchored"),("full_cohort","control_anchored"),("full_cohort","standard")):
        record["module_samples"].extend(validate_module_samples(root,partition,method))
    atomic_json(record,root/"qa/independent_direct_checks.json")
    return dict(status=record["status"],gene_families=len(record["gene_families"]),module_edges_checked=len(record["module_samples"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--completed-catalogs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.root)))
    if args.completed_catalogs:
        result = validate_completed_catalogs(args.root)
        print(json.dumps(dict(status=result["status"], completed_catalogs=len(result["files"]))))
