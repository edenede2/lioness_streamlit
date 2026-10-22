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
from scipy.stats import pearsonr, spearmanr, false_discovery_control

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1]), str(APP/"scripts")]
from scripts.screening_sources import load_sources, OUTCOMES, assignments
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


def validate(root=RUN):
    root = Path(root)
    record = dict(status="passed",audit_identity=require_audit(root),gene_families=validate_genes(root),module_samples=[])
    for partition,method in (("control_derived","control_anchored"),("full_cohort","control_anchored"),("full_cohort","standard")):
        record["module_samples"].extend(validate_module_samples(root,partition,method))
    atomic_json(record,root/"qa/independent_direct_checks.json")
    return dict(status=record["status"],gene_families=len(record["gene_families"]),module_edges_checked=len(record["module_samples"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=RUN)
    print(json.dumps(validate(parser.parse_args().root)))
