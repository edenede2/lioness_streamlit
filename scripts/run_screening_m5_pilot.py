#!/usr/bin/env python3
"""Measured full no-q90 M5 pilot only. Never starts a genome-wide search."""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1]), str(APP/"scripts")]
import numpy as np
import pandas as pd
from scripts.screening_m5 import M5Settings, full_m5, m5_donor_network
from scripts.screening_sources import load_sources, assignments
from scripts.run_gene_edge_screen import RUN, atomic_json, atomic_parquet, require_audit, resource_guard


def measured_pilot(root=RUN):
    root = Path(root); audit_id = require_audit(root)
    source = load_sources("motor"); settings = M5Settings()
    measured = []
    # Small and large CT-containing modules in each beta profile, fixed by size
    # and tissue membership, never by association or prediction performance.
    for partition, profile in (("muscle_control_l4", "control_fitted"), ("muscle_all_donors_l4", "all_donor_fitted")):
        a = assignments(partition)
        counts = a.groupby("module").agg(size=("gene", "size"), tissues=("tissue", "nunique"))
        candidates = counts.loc[counts.tissues.eq(3)].sort_values(["size"], kind="stable")
        modules = [int(candidates.index[0]), int(candidates.index[-1])]
        for module in modules:
            nodes = list(a.loc[a.module.eq(module), ["tissue", "gene"]].itertuples(index=False, name=None))
            ids = source.original
            matrix = np.column_stack([source.expression[t].loc[ids, g].to_numpy(float) for t,g in nodes])
            control_ids = source.controls
            cases = [control_ids[0], next(d for d in ids if source.metadata.loc[d,"diagnosis_group"] == "AD")]
            for method in ("control_anchored", "standard"):
                marker = root/f"pilot/m5/{partition}/{method}/M{module}.json"
                if marker.is_file():
                    existing = json.loads(marker.read_text())
                    if existing.get("audit_identity") != audit_id:
                        raise ValueError("Pilot source identity changed")
                    measured.extend(existing["measurements"])
                    continue
                reference_ids = control_ids if method == "control_anchored" else ids
                reference = matrix[[ids.index(d) for d in reference_ids]]
                started = time.monotonic()
                base, diagnostics = full_m5(reference, reference_ids, nodes, profile, settings=settings)
                base_seconds = time.monotonic()-started
                summary = []; upper = np.triu_indices(len(nodes), 1)
                seed_weights = {}
                for seed in (42, 1042, 2042):
                    config = replace(settings, seed=seed)
                    seed_base = base if seed == 42 else full_m5(reference, reference_ids, nodes, profile, settings=config)[0]
                    for donor in cases:
                        pos = ids.index(donor); started = time.monotonic()
                        network, diag = m5_donor_network(reference, reference_ids, matrix[pos], donor, nodes, profile, settings=config, base=seed_base)
                        donor_seconds = time.monotonic()-started
                        # Independent direct estimator recomputation for seed42.
                        direct_error = 0.
                        if seed == 42:
                            if donor in reference_ids:
                                i = reference_ids.index(donor)
                                changed = full_m5(np.delete(reference, i, axis=0), reference_ids[:i]+reference_ids[i+1:], nodes, profile, settings=config)[0]
                                direct = len(reference_ids)*seed_base-(len(reference_ids)-1)*changed
                            else:
                                changed = full_m5(np.vstack((reference,matrix[pos])), reference_ids+[donor], nodes, profile, settings=config)[0]
                                direct = (len(reference_ids)+1)*changed-len(reference_ids)*seed_base
                            direct_error = float(np.max(np.abs(direct-network)))
                            if direct_error > 1e-10:
                                raise ValueError("M5 direct-formula gate failed")
                        weights = network[upper]
                        seed_weights[(donor,seed)] = weights
                        summary.append(dict(partition=partition, profile=profile, method=method, module=module,
                            role=diag["role"], seed=seed, nodes=len(nodes), edges=len(weights), reference_n=len(reference_ids),
                            base_seconds=base_seconds, donor_seconds=donor_seconds, direct_qa_seconds=time.monotonic()-started-donor_seconds,
                            direct_error=direct_error, signed_mean=float(weights.mean()), absolute_mean=float(np.abs(weights).mean()),
                            retained_fraction=float(np.count_nonzero(weights)/len(weights)),
                            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
                        resource_guard(root)
                # Actual validation donor, observed single tissue only; every
                # validation Control is add-one. This is not missing-tissue fill.
                validation_ids = source.donors(("VH",), True)
                validation_control = next(d for d in validation_ids if source.metadata.loc[d,"diagnosis_group"] == "Control")
                selected_nodes = [(t,g) for t,g in nodes if t == "VH"]
                index = [i for i,(t,g) in enumerate(nodes) if t == "VH"]
                small_reference = reference[:, index]
                target = np.asarray([source.expression[t].loc[validation_control,g] for t,g in selected_nodes])
                b, _ = full_m5(small_reference, reference_ids, selected_nodes, profile, settings=settings)
                first, role = m5_donor_network(small_reference, reference_ids, target, validation_control, selected_nodes, profile, settings=settings, base=b)
                perm = np.arange(len(reference_ids))[::-1]
                again, _ = m5_donor_network(small_reference[perm], [reference_ids[i] for i in perm], target, validation_control, selected_nodes, profile, settings=settings)
                if role["role"] != "add_one" or not np.array_equal(first,again):
                    raise ValueError("Validation Control/reference-order invariance gate failed")
                seed_stability = []
                for donor in cases:
                    base_w = seed_weights[(donor,42)]
                    for seed in (1042,2042):
                        other = seed_weights[(donor,seed)]
                        seed_stability.append(dict(role="loo" if donor in reference_ids else "add_one", seed=seed,
                            pearson=float(np.corrcoef(base_w,other)[0,1]) if np.std(base_w)>0 and np.std(other)>0 else None,
                            signed_mean_difference=float(other.mean()-base_w.mean()), rms_difference=float(np.sqrt(np.mean((other-base_w)**2)))))
                atomic_json(dict(status="measured", audit_identity=audit_id, settings=asdict(settings), q90=False,
                    measurements=summary, seed_stability=seed_stability, block_diagnostics=diagnostics,
                    validation_control_add_one=True, donor_order_invariant=True, genome_launch_authorized=False), marker)
                measured.extend(summary)
                print(json.dumps(dict(stage="full_M5_pilot", partition=partition, method=method, module=module, status="measured")), flush=True)
    table = pd.DataFrame(measured)
    atomic_parquet(table, root/"pilot/m5_measurements.parquet")
    table.to_csv(root/"pilot/m5_measurements.tsv", sep="\t", index=False)
    report = dict(status="resource_review_required", audit_identity=audit_id, measurements=len(table),
        pilot_modules_per_partition=2, profiles=2, methods=2, seeds=[42,1042,2042],
        max_peak_rss_gib=float(table.peak_rss_bytes.max()/2**30),
        complete_reference_recalculated=True, masks_or_outcomes_used=False,
        genome_estimate="Not authorized from module timings alone: whole-block eligible CT null and all-edge Spearman/FDR costs require an explicit resource review",
        genome_launch_authorized=False, no_q90=True, data_dependent_components_frozen=False)
    atomic_json(report, root/"pilot/m5_resource_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=RUN)
    print(json.dumps(measured_pilot(parser.parse_args().root)), flush=True)
