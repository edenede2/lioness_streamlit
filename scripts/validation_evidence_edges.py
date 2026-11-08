"""Local-only frozen candidate follow-up; imported only by the offline builder."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from app_helpers import validation_evidence as E
from app_helpers.control_calibrated import validation_inference as I
from app_helpers.control_calibrated.analysis import bh
from app_helpers.control_calibrated.inputs import atomic_json, atomic_parquet
from app_helpers.control_calibrated.generate import pearson_block, weights, unpack
from app_helpers.control_calibrated.core import edge_attributes
from app_helpers.control_calibrated.inspection import calibration_cache
from app_helpers.control_calibrated.registry import RESEARCH_ROOT
from models.differential_edges import StreamingMoments, welch_hedges_from_moments
from build_validation_evidence import context, public_csv, refresh_manifest, SEED

OUTCOMES = ("cogn_global", "gpath")
DRAWS = 9999


def difference(w, meta):
    groups = []
    for diagnosis in ("AD", "Control"):
        v = np.asarray(w[meta.diagnosis_group.eq(diagnosis)])
        groups.append(
            StreamingMoments(count=len(v), total=v.sum(0), total_squares=(v * v).sum(0))
        )
    return welch_hedges_from_moments(*groups)


def partial_ranks(w, meta, outcome, *, inference=True):
    valid = I.complete_covariates(meta) & meta[outcome].notna().to_numpy()
    sub = meta.loc[valid].reset_index(drop=True)
    matrix = np.asarray(w[valid]).T
    seed = I.stable_seed(SEED, "same_edge", outcome, I.membership_hash(sub.sample_id))
    result = I.infer_group(
        matrix,
        sub[outcome].to_numpy(float),
        sub,
        adjusted=True,
        draws=DRAWS if inference else 0,
        seed=seed,
        blocks=sub.study.to_numpy(),
        minimum_n=30,
    )
    return result, valid, sub, seed


def subset_reference(ref, nodes):
    return {
        k: (
            v[:, nodes]
            if k == "values"
            else v[nodes] if k in {"mean", "ss", "endpoint_valid", "impute_mean"} else v
        )
        for k, v in ref.items()
    }


def donor_weights(values, reference, presence, node_tissues, r, c):
    r0, r1, valid, _ = pearson_block(
        values, reference, presence, node_tissues, r, c, np.array([], int)
    )
    w = weights(r0, r1, 3, np.array([], int), len(reference["values"]))
    w[~valid] = np.nan
    return w


def choose_candidates(frame, k=20):
    """Selection has no validation input. Stable physical edge-code tie break."""
    picks = {}
    for name, column in (
        ("AD_Control_raw_w", "original_g"),
        ("cognition_raw_w", "original_cogn_global_effect"),
        ("pathology_raw_w", "original_gpath_effect"),
    ):
        selected = frame.loc[np.isfinite(frame[column])].assign(
            _absolute=lambda v: abs(v[column])
        )
        selected = selected.sort_values(
            ["_absolute", "edge_code"], ascending=[False, True], kind="stable"
        ).head(k)
        for rank, index in enumerate(selected.index, 1):
            picks.setdefault(index, []).append((name, rank))
    result = frame.loc[sorted(picks)].copy()
    result["selected_by"] = [
        "; ".join(f"{name}:rank{rank}" for name, rank in picks[ix])
        for ix in result.index
    ]
    return result


def freeze_candidates(job, component, info, meta, x, presence, ref, params):
    prefix = f"{job['family']}__M{job['module']}__{component}"
    path = E.PRIVATE / (prefix + "__selection.parquet")
    receipt = path.with_suffix(".json")
    nodes, r, c, tissues = context(job, info, component)
    signature = dict(
        version=E.VERSION,
        score_beta=3,
        mask="all",
        top_k=20,
        module=job["module"],
        family=job["family"],
        component=component,
        expression_sha256=info["expression_sha256"],
        reference_sha256=E.sha(RESEARCH_ROOT / "private/brain_reference.npz"),
    )
    if path.is_file() and receipt.is_file():
        check = json.loads(receipt.read_text())
        if check["signature"] != signature or E.sha(path) != check["sha256"]:
            raise ValueError(
                "Frozen candidate input changed; use a new artifact version"
            )
        return pd.read_parquet(path), nodes, r, c
    codes = np.minimum(nodes[r], nodes[c]) * info["n_nodes"] + np.maximum(
        nodes[r], nodes[c]
    )
    all_codes = np.load(RESEARCH_ROOT / "private/brain/edges.npy", mmap_mode="r")
    positions = np.searchsorted(all_codes, codes)
    if not np.array_equal(all_codes[positions], codes):
        raise ValueError("Physical-edge alignment failed")
    eligible = np.asarray(params[positions, 10], bool)
    keep = E.cohort_selection(meta, component, "original_complete", independent=True)
    original = meta.loc[keep].reset_index(drop=True)
    values = np.asarray(x[np.ix_(np.flatnonzero(keep), nodes)])
    localref = subset_reference(ref, nodes)
    nt = np.array([info["tissues"].index(t) for t in tissues])
    pieces = []
    for start in range(0, len(r), 1024):
        end = min(start + 1024, len(r))
        w = donor_weights(
            values, localref, presence[keep], nt, r[start:end], c[start:end]
        )
        d = difference(w, original)
        part = pd.DataFrame(
            dict(
                edge_code=codes[start:end],
                row=r[start:end],
                column=c[start:end],
                original_g=d["hedges_g"],
                original_difference=d["mean_difference"],
                original_p_difference=d["p_value"],
                eligible=eligible[start:end],
            )
        )
        for outcome in OUTCOMES:
            results, valid, _, _ = partial_ranks(w, original, outcome)
            part["original_" + outcome + "_effect"] = results["effect"]
            part["original_" + outcome + "_p"] = results["p"]
            part["original_" + outcome + "_n"] = int(valid.sum())
        part.loc[~part.eligible, [c for c in part if c.startswith("original_")]] = (
            np.nan
        )
        pieces.append(part)
    full = pd.concat(pieces, ignore_index=True)
    for p, q in [
        ("original_p_difference", "original_q_difference"),
        *[(f"original_{o}_p", f"original_{o}_q") for o in OUTCOMES],
    ]:
        full[q] = bh(full[p])
        full[q + "_family_n"] = int(full[p].notna().sum())
    chosen = choose_candidates(full)
    chosen["partition"] = job["family"]
    chosen["module"] = job["module"]
    chosen["component"] = component
    chosen["score_beta"] = 3
    chosen["edge_readout"] = (
        "raw signed frozen-R82 add-one LIONESS weight (signed cube adjacency)"
    )
    chosen["source_ranks"] = (
        "New original-target-only ranks: historical R164 module-edge ranks use a different readout and are not substituted"
    )
    for endpoint, index in (
        ("a", chosen.row.to_numpy(int)),
        ("b", chosen.column.to_numpy(int)),
    ):
        chosen["tissue_" + endpoint] = [info["nodes"][nodes[i]][0] for i in index]
        chosen["gene_" + endpoint] = [info["nodes"][nodes[i]][1] for i in index]
    # The entire candidate list and full-search q-values are frozen on disk
    # before the caller reads ANY validation expression/outcome values.
    atomic_parquet(path, chosen)
    atomic_parquet(E.PRIVATE / (prefix + "__original_search.parquet"), full)
    atomic_json(
        receipt,
        dict(
            signature=signature,
            sha256=E.sha(path),
            selection="union of top20 absolute Hedges g, top20 absolute adjusted cognition partial rho, top20 absolute adjusted gpath partial rho; original targets only; physical code ties",
            original_correction="BH all valid physical edges within this module/component/readout/endpoint BEFORE selecting candidates",
            original_role="target only; calibration/reference Controls excluded",
            original_n=len(original),
            original_member_sha256=I.membership_hash(original.sample_id),
            frozen_before_validation=True,
        ),
    )
    return chosen, nodes, r, c


def follow(job, component, info, meta, x, presence, ref, params, bootstrap):
    chosen, nodes, r_all, c_all = freeze_candidates(
        job, component, info, meta, x, presence, ref, params
    )
    prefix = f"{job['family']}__M{job['module']}__{component}"
    output = E.PRIVATE / (prefix + "__followup.parquet")
    if output.is_file():
        return pd.read_parquet(output), pd.read_parquet(
            E.PRIVATE / (prefix + "__contributions.parquet")
        )
    r, c = chosen.row.to_numpy(int), chosen.column.to_numpy(int)
    codes = chosen.edge_code.to_numpy(np.int64)
    edge_codes = np.load(RESEARCH_ROOT / "private/brain/edges.npy", mmap_mode="r")
    positions = np.searchsorted(edge_codes, codes)
    component_codes = np.minimum(nodes[r_all], nodes[c_all]) * info[
        "n_nodes"
    ] + np.maximum(nodes[r_all], nodes[c_all])
    denominator = int(
        np.asarray(params[np.searchsorted(edge_codes, component_codes), 10], bool).sum()
    )
    cal = unpack(np.asarray(params[positions]))
    localref = subset_reference(ref, nodes)
    nt = np.array([info["tissues"].index(info["nodes"][i][0]) for i in nodes])
    rows, contributions = [], []
    arrays = {}
    for cohort in ("original_complete", "incomplete"):
        keep = E.cohort_selection(meta, component, cohort, independent=True)
        sub = meta.loc[keep].reset_index(drop=True)
        w = donor_weights(
            np.asarray(x[np.ix_(np.flatnonzero(keep), nodes)]),
            localref,
            presence[keep],
            nt,
            r,
            c,
        )
        departure, _ = edge_attributes(w, cal)
        arrays[cohort + "_weights"] = w
        arrays[cohort + "_departure"] = departure
        d = difference(w, sub)
        rng = np.random.default_rng(
            I.stable_seed(SEED, prefix, cohort, "Hedges_bootstrap")
        )
        g_boot = []
        for _ in range(bootstrap):
            indices = np.concatenate(
                [
                    ix[E.resample_indices(sub.iloc[ix].reset_index(drop=True), rng)]
                    for diagnosis in ("AD", "Control")
                    for ix in [np.flatnonzero(sub.diagnosis_group.eq(diagnosis))]
                ]
            )
            g_boot.append(
                difference(w[indices], sub.iloc[indices].reset_index(drop=True))[
                    "hedges_g"
                ]
            )
        for k, (_, candidate) in enumerate(chosen.iterrows()):
            lo, hi, valid = E.interval(np.array(g_boot)[:, k], bootstrap)
            rows.append(
                dict(
                    **candidate.to_dict(),
                    cohort=cohort,
                    test="AD_Control",
                    outcome="diagnosis",
                    effect=d["hedges_g"][k],
                    mean_difference=d["mean_difference"][k],
                    mean_ad=d["mean_ad"][k],
                    mean_control=d["mean_control"][k],
                    n_ad=d["n_ad"],
                    n_control=d["n_control"],
                    n=d["n_ad"] + d["n_control"],
                    p=d["p_value"][k],
                    ci_low=lo,
                    ci_high=hi,
                    bootstrap_valid=valid,
                    bootstrap_budget=bootstrap,
                    adjustment="none; unadjusted AD–Control raw-edge Hedges g",
                    inference="Welch two-sided p; donor-bootstrap Hedges CI",
                    original_search_q=candidate.original_q_difference,
                    original_search_family_n=candidate.original_q_difference_family_n,
                )
            )
            for diagnosis in ("All target donors", "AD", "Control"):
                dk = (
                    np.ones(len(sub), bool)
                    if diagnosis.startswith("All")
                    else sub.diagnosis_group.eq(diagnosis).to_numpy()
                )
                contributions.append(
                    dict(
                        partition=job["family"],
                        module=job["module"],
                        component=component,
                        edge_code=int(candidate.edge_code),
                        cohort=cohort,
                        diagnosis=diagnosis,
                        n=int(dk.sum()),
                        raw_mean=float(np.mean(w[dk, k])),
                        calibrated_departure_mean=float(np.nanmean(departure[dk, k])),
                        gain_contribution=float(
                            np.nanmean(np.maximum(departure[dk, k], 0)) / denominator
                        ),
                        loss_contribution=float(
                            np.nanmean(np.maximum(-departure[dk, k], 0)) / denominator
                        ),
                        calibration_nu=cal["nu"][k],
                        calibration_tau=cal["tau"][k],
                        full_module_component_denominator=denominator,
                        meaning="mean calibrated magnitude departure / full frozen eligible-edge denominator; not clinical effect size",
                    )
                )
        for outcome in OUTCOMES:
            result, valid, tested, seed = partial_ranks(w, sub, outcome)
            boot = I.bootstrap_group(
                w[valid].T,
                tested[outcome].to_numpy(float),
                tested,
                draws=bootstrap,
                seed=I.stable_seed(SEED, prefix, cohort, outcome, "boot"),
            )
            for k, (_, candidate) in enumerate(chosen.iterrows()):
                rows.append(
                    dict(
                        **candidate.to_dict(),
                        cohort=cohort,
                        test="edge_outcome",
                        outcome=outcome,
                        effect=result["effect"][k],
                        p=result["p"][k],
                        n=int(valid.sum()),
                        ci_low=boot["adjusted_low"][k],
                        ci_high=boot["adjusted_high"][k],
                        bootstrap_valid=boot["adjusted_valid"][k],
                        bootstrap_budget=bootstrap,
                        adjustment="partial midranks: age_at_death + sex_code + education_years + study; exact complete cases",
                        inference="study-restricted Freedman–Lane on fixed initial ranks; 9999 draws",
                        permutation_seed=seed,
                        permutation_draws=int(result["draws"][k]),
                        p_resolution=1 / (DRAWS + 1),
                        unavailable_reason=result["reason"][k],
                        original_search_q=candidate["original_" + outcome + "_q"],
                        original_search_family_n=candidate[
                            "original_" + outcome + "_q_family_n"
                        ],
                    )
                )
    result = pd.DataFrame(rows)
    atomic_parquet(output, result)
    contribution = pd.DataFrame(contributions)
    atomic_parquet(E.PRIVATE / (prefix + "__contributions.parquet"), contribution)
    np.savez_compressed(
        E.PRIVATE / (prefix + "__selected_donor_edges.npz"), edge_code=codes, **arrays
    )
    return result, contribution


def run_edges(info, meta, x, presence, ref, jobs, bootstrap):
    params_path = RESEARCH_ROOT / "private/brain/calibration_beta3.npy"
    params = calibration_cache(
        str(RESEARCH_ROOT), "brain", 3, params_path.stat().st_mtime_ns
    )
    followed, contributions, candidates = [], [], []
    for job in jobs:
        for component in E.COMPONENTS:
            result, contribution = follow(
                job, component, info, meta, x, presence, ref, params, bootstrap
            )
            followed.append(result)
            contributions.append(contribution)
            selection = (
                E.PRIVATE
                / f"{job['family']}__M{job['module']}__{component}__selection.parquet"
            )
            candidates.append(pd.read_parquet(selection))
            print(
                json.dumps(
                    dict(
                        stage="same_edges",
                        partition=job["family"],
                        module=job["module"],
                        component=component,
                        status="complete",
                        candidates=result.edge_code.nunique(),
                    )
                ),
                flush=True,
            )
    result = pd.concat(followed, ignore_index=True)
    # The entire frozen tested union is declared before any display filtering.
    # Families are separate by cohort and biological meaning/endpoint, not Top K.
    for _, ix in result.groupby(
        ["cohort", "test", "outcome"], dropna=False, sort=False
    ).groups.items():
        result.loc[ix, "selected_family_q"] = bh(result.loc[ix, "p"])
        result.loc[ix, "selected_family_n"] = int(result.loc[ix, "p"].notna().sum())
    result["selected_family_scope"] = (
        "All frozen selected candidate module-edge hypotheses across the 4 named namespaces × 2 components, separately per cohort/test/endpoint; not all module edges"
    )
    keys = ["partition", "module", "component", "edge_code", "test", "outcome"]
    orig = result.loc[result.cohort.eq("original_complete")].set_index(keys)
    val = result.loc[result.cohort.eq("incomplete")].set_index(keys)
    match = val.join(
        orig[["effect", "p", "selected_family_q", "n", "ci_low", "ci_high"]].add_prefix(
            "original_"
        ),
        how="left",
        validate="one_to_one",
    ).reset_index()
    match["evidence"] = [
        E.evidence_status(a, b, p, q)
        for a, b, p, q in zip(
            match.original_effect, match.effect, match.p, match.selected_family_q
        )
    ]
    summaries = []
    for key, group in match.groupby(
        ["partition", "module", "component", "test", "outcome"], sort=False
    ):
        comparable = np.isfinite(group.original_effect) & np.isfinite(group.effect)
        same = (
            comparable
            & (np.sign(group.original_effect) == np.sign(group.effect))
            & group.effect.ne(0)
            & group.original_effect.ne(0)
        )
        reversed_ = comparable & (group.original_effect * group.effect < 0)
        original_signal = group.original_search_q.lt(0.05)
        summary = dict(
            zip(["partition", "module", "component", "test", "outcome"], key)
        )
        summary.update(
            planned_candidates=len(group),
            tested=int(group.p.notna().sum()),
            unavailable=int(group.p.isna().sum()),
            comparable=int(comparable.sum()),
            same_direction=int(same.sum()),
            direction_reversed=int(reversed_.sum()),
            zero_direction=int((comparable & ~same & ~reversed_).sum()),
            nominal_support=int((group.p.lt(0.05) & same).sum()),
            adjusted_support=int((group.selected_family_q.lt(0.05) & same).sum()),
            original_full_search_significant=int(original_signal.sum()),
            original_signal_evaluable=int((original_signal & comparable).sum()),
            original_signal_nominal_support=int(
                (original_signal & same & group.p.lt(0.05)).sum()
            ),
            original_signal_adjusted_support=int(
                (original_signal & same & group.selected_family_q.lt(0.05)).sum()
            ),
            median_original_effect=float(group.original_effect.median()),
            median_validation_effect=float(group.effect.median()),
            min_n=int(group.n.min()),
            max_n=int(group.n.max()),
            score_beta=3,
            mask="all",
            correction="BH full frozen selected union by test/endpoint; original full-search q retained in local table",
            local_detail="Candidate identities, donor edge arrays and full original search remain local-only",
        )
        summaries.append(summary)
    atomic_parquet(E.PRIVATE / "same_edges.parquet", match)
    atomic_parquet(
        E.PRIVATE / "edge_candidates.parquet", pd.concat(candidates, ignore_index=True)
    )
    atomic_parquet(
        E.PRIVATE / "edge_contributions.parquet",
        pd.concat(contributions, ignore_index=True),
    )
    public_csv("edge_summary", pd.DataFrame(summaries))
    atomic_json(
        E.PRIVATE / "manifest.json",
        dict(
            version=E.VERSION,
            local_only=True,
            files={
                name: E.sha(E.PRIVATE / name)
                for name in (
                    "same_edges.parquet",
                    "edge_candidates.parquet",
                    "edge_contributions.parquet",
                )
            },
            local_artifacts={
                p.name: E.sha(p)
                for p in E.PRIVATE.iterdir()
                if p.suffix in (".png", ".pdf", ".npz")
                and "__selected_donor_edges" not in p.name
            },
            selection="Frozen original-only top20 by absolute raw-edge Hedges g and separate cogn_global/gpath adjusted partial-rank associations; union, no validation reranking",
            score_beta=3,
            reference="frozen R=82; C=41; original and incomplete target donors only",
            correction="Original BH all module-component eligible physical edges per endpoint; validation BH frozen selected union per meaning/endpoint",
            inference_draws=DRAWS,
            bootstrap_budget=bootstrap,
        ),
    )
