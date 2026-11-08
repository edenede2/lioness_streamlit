#!/usr/bin/env python3
"""Bounded, offline evidence for four frozen brain modules; never uploads.

Public output is aggregate-only. Reconstructed donor edges, candidate IDs and
group matrices/figures stay in artifacts/.../private and are never committed.
"""

from __future__ import annotations
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import resource
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP.parents[1])]
import numpy as np
import pandas as pd
from scipy import stats
from app_helpers import validation_evidence as E
from app_helpers import validation_review as VR
from app_helpers.control_calibrated.registry import RESEARCH_ROOT
from app_helpers.control_calibrated.inputs import atomic_json, atomic_parquet, canon
from app_helpers.control_calibrated import validation_inference as I
from app_helpers.control_calibrated.analysis import bh
from app_helpers.control_calibrated.generate import pearson_block, weights, unpack
from app_helpers.control_calibrated.core import edge_attributes

SEED = 611108


@lru_cache(maxsize=32)
def _input_sha(path, size, mtime_ns):
    return E.sha(path)


def input_sha(path):
    path = Path(path)
    stat = path.stat()
    return _input_sha(str(path), stat.st_size, stat.st_mtime_ns)


def public_csv(name, frame):
    if {"donor", "projid", "sample_id", "edge_code", "gene_a", "gene_b"}.intersection(
        frame
    ):
        raise ValueError("Private field in aggregate output")
    E.PUBLIC.mkdir(parents=True, exist_ok=True)
    frame.to_csv(E.PUBLIC / (name + ".csv"), index=False)


def inputs():
    # A resume must not relabel stale checkpoints with changed metadata or
    # calibration inputs. Producer-code changes are tracked separately.
    previous = E.PUBLIC / "manifest.json"
    if previous.is_file():
        receipt = json.loads(previous.read_text())
        if receipt.get("version") != E.VERSION:
            raise ValueError("Focused evidence version mismatch")
        for name, expected in receipt.get("input_sha256", {}).items():
            if input_sha(APP / name) != expected:
                raise ValueError(
                    "Frozen evidence source changed; use a new version: " + name
                )
    catalog = json.loads((RESEARCH_ROOT / "catalog.json").read_text())
    info = catalog["datasets"]["brain"]
    meta = pd.read_parquet(RESEARCH_ROOT / info["metadata_file"])
    if not meta.sample_id.is_unique:
        raise ValueError("Nonunique calibrated donor metadata")
    for name, expected in (
        ("expression", info["expression_sha256"]),
        ("presence", info["presence_sha256"]),
    ):
        if E.sha(RESEARCH_ROOT / f"private/brain_{name}.npy") != expected:
            raise ValueError("Frozen input changed: " + name)
    x = np.load(RESEARCH_ROOT / "private/brain_expression.npy", mmap_mode="r")
    presence = np.load(RESEARCH_ROOT / "private/brain_presence.npy")
    receipt = json.loads((RESEARCH_ROOT / "private/brain_reference.json").read_text())
    path = RESEARCH_ROOT / "private/brain_reference.npz"
    if E.sha(path) != receipt["array_sha256"]:
        raise ValueError("Frozen reference mismatch")
    ref = dict(np.load(path))
    jobs = [
        j for j in catalog["modules"] if j["module"] in E.FOCUS.get(j["family"], ())
    ]
    if len(jobs) != 4:
        raise ValueError("Focused module identity mismatch")
    return catalog, info, meta, x, presence, ref, jobs


def context(job, info, component):
    wanted = component[3:].split("__")
    nodes = np.array(
        [n for n in job["node_indices"] if info["nodes"][n][0] in wanted], int
    )
    tissues = np.array([info["nodes"][n][0] for n in nodes])
    r, c = np.triu_indices(len(nodes), 1)
    keep = (
        tissues[r] == tissues[c]
        if component.startswith("TS_")
        else tissues[r] != tissues[c]
    )
    return nodes, r[keep], c[keep], tissues


def discovery_membership(path, private_to_public):
    """Associated eigengene files have two real orientations/delimiters.

    An unreadable/empty header is NOT an empty discovery cohort. Never emit
    zero-overlap evidence unless an actual nonempty membership was resolved.
    """
    if not path.is_file():
        return None
    with path.open() as stream:
        header = stream.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    columns = pd.read_csv(path, sep=sep, nrows=0).columns[1:]
    candidates = [columns, pd.read_csv(path, sep=sep, usecols=[0]).iloc[:, 0]]
    for values in candidates:
        if len(values) == 0:
            continue
        try:
            donors = [canon(d) for d in values]
            if all(d in private_to_public for d in donors):
                return {private_to_public[d] for d in donors}
        except (ValueError, KeyError):
            continue
    return None


def audit(info, meta, jobs):
    rows, intersections = [], []
    all_complete = meta[["has_AC", "has_DLPFC", "has_PCG"]].all(axis=1)
    for name, keep in [
        ("original_complete", meta.in_original_analysis),
        ("incomplete", ~all_complete),
        (
            "legacy_heldout",
            meta.in_original_analysis & meta.ad_control_split.eq("Validation"),
        ),
        ("other_complete_targets", all_complete & ~meta.in_original_analysis),
    ]:
        for (diagnosis, role), group in meta.loc[keep].groupby(
            ["diagnosis_group", "role"], observed=True, dropna=False
        ):
            rows.append(
                dict(
                    dataset="brain",
                    component="Any observed brain region",
                    cohort=name,
                    population="all applicable roles",
                    diagnosis=diagnosis,
                    role=role,
                    unique_donors=group.sample_id.nunique(),
                    cogn_global_n=int(group.cogn_global.notna().sum()),
                    gpath_n=int(group.gpath.notna().sum()),
                    mask="all",
                    artifact_version=E.VERSION,
                )
            )
    matched_path = APP / "data/matched_brain_donors/sample_metadata.parquet"
    if matched_path.is_file():
        matched = pd.read_parquet(matched_path)
        for (role, diagnosis), group in matched.groupby(
            ["analysis_role", "diagnosis_group"], observed=True, dropna=False
        ):
            rows.append(
                dict(
                    dataset="matched_control_brain_l4 (separate namespace)",
                    component="Registered matched cohort; region inclusion is context-specific",
                    cohort=role,
                    population="all applicable roles",
                    diagnosis=diagnosis,
                    role="matched analysis role",
                    unique_donors=group.sample_id.nunique(),
                    cogn_global_n=int(group.cogn_global.notna().sum()),
                    gpath_n=int(group.gpath.notna().sum()),
                    mask="see matched-control catalog",
                    artifact_version="existing approved matched_brain_donors metadata",
                )
            )
    for component in E.COMPONENTS:
        for cohort in ("original_complete", "incomplete", "legacy_heldout"):
            for independent in (
                (False, True) if cohort != "legacy_heldout" else (False,)
            ):
                selected = meta.loc[
                    E.cohort_selection(meta, component, cohort, independent=independent)
                ]
                for (diagnosis, role), group in selected.groupby(
                    ["diagnosis_group", "role"], dropna=False, observed=True
                ):
                    rows.append(
                        dict(
                            dataset="brain",
                            component=component,
                            cohort=cohort,
                            population=(
                                "independent targets"
                                if independent
                                else "all applicable roles"
                            ),
                            diagnosis=diagnosis,
                            role=role,
                            unique_donors=group.sample_id.nunique(),
                            cogn_global_n=int(group.cogn_global.notna().sum()),
                            gpath_n=int(group.gpath.notna().sum()),
                            mask="all",
                            artifact_version=E.VERSION,
                        )
                    )
        original = set(
            meta.loc[
                E.cohort_selection(meta, component, "original_complete"), "sample_id"
            ]
        )
        validation = set(
            meta.loc[E.cohort_selection(meta, component, "incomplete"), "sample_id"]
        )
        intersections.append(
            dict(
                dataset="brain",
                component=component,
                comparison="complete-expression original versus incomplete validation",
                **E.intersection_audit(original, validation),
                audit_source="frozen calibrated membership and dataset-wide tissue-presence records",
            )
        )
    public_csv("cohorts", pd.DataFrame(rows))
    ids = json.loads((RESEARCH_ROOT / "private/brain_identities.json").read_text())
    private_to_public = dict(zip(ids["donors"], ids["public_ids"], strict=True))
    for family, path in [
        (
            "control_derived",
            APP.parents[1]
            / "data/proccessed/se2_signed_alt_control_filtered/se2_rosmap_control_signed-alt/se2_4_rosmap_full_Control_Brain_ME_commonDonors_cross_tissue.tsv",
        ),
        (
            "full_cohort",
            APP.parents[1]
            / "data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_4_rosmap_full_signed_alt_global2_global2_ME_commonDonors_cross_tissue.tsv",
        ),
    ]:
        discovery = discovery_membership(path, private_to_public)
        validation = set(meta.loc[~all_complete, "sample_id"])
        intersections.append(
            dict(
                dataset=family,
                component="All fixed module tissues",
                comparison="Associated historical discovery eigengene donor header versus incomplete",
                **E.intersection_audit(discovery, validation),
                audit_source=str(path.relative_to(APP.parents[1]))
                + "; associated artifact donor axis, not a re-executed discovery log",
            )
        )
    historic = {}
    for diagnosis in ("AD", "Control"):
        for tissue, old in (("AC", "AC"), ("DLPFC", "MFBA9BA46"), ("PCG", "PCGBA23")):
            path = (
                APP.parents[1]
                / "data/external/ROSMAP_partitioned"
                / f"ROSMAP_fixed_{diagnosis}{old}.csv"
            )
            if path.is_file():
                raw = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str)
                donor = raw.str.extract(r"^([^-]+-[^-]+)", expand=False).map(canon)
                historic[diagnosis, tissue] = {
                    private_to_public[d] for d in donor if d in private_to_public
                }
    for component in E.COMPONENTS:
        ts = component[3:].split("__")
        for diagnosis in ("AD", "Control"):
            sets = [historic.get((diagnosis, t)) for t in ts]
            old = set.intersection(*sets) if all(s is not None for s in sets) else None
            validation = set(
                meta.loc[
                    E.cohort_selection(meta, component, "incomplete")
                    & meta.diagnosis_group.eq(diagnosis),
                    "sample_id",
                ]
            )
            intersections.append(
                dict(
                    dataset="brain",
                    component=component,
                    comparison=f"Historical Control-derived MDC reconstructed pairwise inputs ({diagnosis}) versus incomplete",
                    **E.intersection_audit(old, validation),
                    audit_source="Current historical input CSVs; pairwise-complete region membership. Retrospective reconstruction, not an archived donor manifest.",
                )
            )
        intersections.append(
            dict(
                dataset="brain",
                component=component,
                comparison="Historical full-cohort MDC saved RDS versus incomplete",
                **E.intersection_audit(None, validation),
                audit_source="Saved adjacency cache has no donor membership record; independence not verified",
            )
        )
    public_csv("intersections", pd.DataFrame(intersections))
    existing_mdc = []
    overview = []
    for family, method in [
        ("control_derived", "control_anchored"),
        ("full_cohort", "standard"),
        ("full_cohort", "control_anchored"),
        ("muscle_control_l4", "control_anchored"),
        ("muscle_all_donors_l4", "control_anchored"),
    ]:
        reader = VR.for_dataset(family, method)
        f = reader.table("mdc_comparison")
        if f.empty:
            continue
        f["partition"] = family
        f["network_method"] = method
        existing_mdc.append(f)
        for (component, rule), group in f.groupby(
            ["component", "differential_edge_rule"], observed=True
        ):
            overview.append(
                dict(
                    partition=family,
                    network_method=method,
                    component=component,
                    mask=rule,
                    **E.mdc_summary(group),
                    artifact_version=reader.manifest().get("run_id", "unavailable"),
                    inference="existing sample-shuffle + tissue-preserving gene null; conservative max BH",
                    correction="across valid modules within component/mask/direction/null; no display-time correction",
                )
            )
    public_csv("mdc_overview", pd.DataFrame(overview))
    # Overview uses every mask/test above. Keep the downloadable detail compact:
    # all modules for all-edges, plus every mask for the fixed focused modules.
    detail = pd.concat(existing_mdc, ignore_index=True)
    focused = np.array(
        [m in E.FOCUS.get(f, ()) for f, m in zip(detail.partition, detail.module)]
    )
    public_csv(
        "historical_mdc", detail.loc[detail.differential_edge_rule.eq("all") | focused]
    )
    return rows, intersections


def historical_densities(job, component):
    root = (
        APP.parents[1]
        / "out/brain_mdc_split_triangle_heatmaps/20261014/_private_adjacencies"
        / job["family"]
    )
    path = root / "manifest.tsv"
    if not path.is_file():
        return dict(
            historical_reason="Exact historical numeric group adjacency cache absent"
        )
    frame = pd.read_csv(path, sep="\t")
    row = frame.loc[frame.module.eq(job["module"])].iloc[0]
    n = int(row.n_nodes)
    codes = np.fromfile(root / f"M{job['module']}__tissues.i32", dtype="<i4")
    r, c = np.triu_indices(n, 1)
    keep = (
        (codes[r] == 1) & (codes[c] == 1)
        if component == "TS_DLPFC"
        else ((codes[r] == 1) & (codes[c] == 2)) | ((codes[r] == 2) & (codes[c] == 1))
    )
    means = {}
    for label in ("AD", "Control"):
        values = np.memmap(
            root / f"M{job['module']}__{label}.f64",
            mode="r",
            dtype="<f8",
            shape=(n, n),
            order="F",
        )
        means[label] = (
            float(np.mean(abs(values[r[keep], c[keep]]))) if keep.any() else np.nan
        )
    return dict(
        historical_mean_abs_ad=means["AD"],
        historical_mean_abs_control=means["Control"],
        historical_mdc=(
            means["AD"] / means["Control"] if means["Control"] > 0 else np.nan
        ),
        historical_reason="Historical tissue-union/pairwise-complete networks; not the disjoint complete-expression comparison",
    )


def structure_job(job, component, info, meta, x, ref, budget):
    nodes, r, c, tissues = context(job, info, component)
    prefix = f"{job['family']}__M{job['module']}__{component}"
    checkpoint = E.PRIVATE / (prefix + ".json")
    signature = dict(
        version=E.VERSION,
        catalog=E.sha(RESEARCH_ROOT / "catalog.json"),
        budget=budget,
        component=component,
        family=job["family"],
        module=job["module"],
        diagnostic_revision=2,
    )
    if checkpoint.is_file():
        value = json.loads(checkpoint.read_text())
        if value.get("signature") == signature:
            return value["structure"], value["mdc"]
    power = 3 if component.startswith("TS_") else 2
    base = dict(
        partition=job["family"],
        module=job["module"],
        component=component,
        mask="all",
        adjacency_power=power,
        score_beta="not applicable: group network",
        n_nodes=len(nodes),
        n_edges=len(r),
        node_convention="fixed unique tissue-gene nodes; not positional duplicate edges",
        seed=SEED,
        bootstrap_budget=budget,
        uncertainty="study-stratified donor percentile bootstrap; rebuild all group correlations/powers",
        independence="verified disjoint complete-expression versus incomplete donors",
        retrospective="User-selected modules; original Controls participated in historical discovery/calibration",
        artifact_version=E.VERSION,
    )
    if not len(r):
        return dict(**base, unavailable_reason="structurally absent component"), dict(
            **base, unavailable_reason="structurally absent component"
        )
    values, metadata = {}, {}
    for cohort in ("original_complete", "incomplete"):
        for diagnosis in ("AD", "Control"):
            key = cohort + "_" + diagnosis
            keep = (
                E.cohort_selection(meta, component, cohort)
                & meta.diagnosis_group.eq(diagnosis).to_numpy()
            )
            metadata[key] = meta.loc[keep].reset_index(drop=True)
            value = np.asarray(x[np.ix_(np.flatnonzero(keep), nodes)], float)
            value = np.where(np.isfinite(value), value, ref["impute_mean"][nodes])
            values[key] = value
    oc, vc = "original_complete_Control", "incomplete_Control"
    structure = {
        **base,
        **E.structure_metrics(values[oc], values[vc], r, c, tissues, power),
        "original_control_n": len(values[oc]),
        "validation_control_n": len(values[vc]),
        "original_reference_controls": int(metadata[oc].role.eq("reference").sum()),
        "original_calibration_controls": int(metadata[oc].role.eq("calibration").sum()),
        "original_target_controls": int(metadata[oc].role.eq("target").sum()),
        "unavailable_reason": "",
    }
    edges = {k: E.group_edges(v, r, c, power)[1] for k, v in values.items()}
    mdc = {**base, "unavailable_reason": "", **historical_densities(job, component)}
    for cohort in ("original_complete", "incomplete"):
        ad, control, ratio = E.density_ratio(
            edges[cohort + "_AD"], edges[cohort + "_Control"]
        )
        mdc.update(
            {
                cohort + "_mean_abs_ad": ad,
                cohort + "_mean_abs_control": control,
                cohort + "_mdc": ratio,
                cohort + "_log2_mdc": np.log2(ratio) if ratio > 0 else np.nan,
                cohort + "_ad_n": len(values[cohort + "_AD"]),
                cohort + "_control_n": len(values[cohort + "_Control"]),
            }
        )
    mdc["log2_ad_density_change"] = np.log2(
        mdc["incomplete_mean_abs_ad"] / mdc["original_complete_mean_abs_ad"]
    )
    mdc["log2_control_density_change"] = np.log2(
        mdc["incomplete_mean_abs_control"] / mdc["original_complete_mean_abs_control"]
    )
    mdc["delta_log2_mdc"] = (
        mdc["log2_ad_density_change"] - mdc["log2_control_density_change"]
    )
    draws_s, draws_m, matched = [], [], []
    rng = np.random.default_rng(
        I.stable_seed(SEED, job["family"], job["module"], component)
    )
    for _ in range(budget):
        sampled = {
            k: v[E.resample_indices(metadata[k], rng)] for k, v in values.items()
        }
        draws_s.append(
            E.structure_metrics(sampled[oc], sampled[vc], r, c, tissues, power)
        )
        match_n = min(len(values[oc]), len(values[vc]))
        om = values[oc][rng.choice(len(values[oc]), match_n, replace=False)]
        vm = values[vc][rng.choice(len(values[vc]), match_n, replace=False)]
        matched.append(E.structure_metrics(om, vm, r, c, tissues, power))
        e = {k: E.group_edges(v, r, c, power)[1] for k, v in sampled.items()}
        cell = {}
        for cohort in ("original_complete", "incomplete"):
            ad, control, ratio = E.density_ratio(
                e[cohort + "_AD"], e[cohort + "_Control"]
            )
            cell.update(
                {
                    cohort + "_mean_abs_ad": ad,
                    cohort + "_mean_abs_control": control,
                    cohort + "_mdc": ratio,
                    cohort + "_log2_mdc": np.log2(ratio) if ratio > 0 else np.nan,
                }
            )
        cell["log2_ad_density_change"] = np.log2(
            cell["incomplete_mean_abs_ad"] / cell["original_complete_mean_abs_ad"]
        )
        cell["log2_control_density_change"] = np.log2(
            cell["incomplete_mean_abs_control"]
            / cell["original_complete_mean_abs_control"]
        )
        cell["delta_log2_mdc"] = (
            cell["log2_ad_density_change"] - cell["log2_control_density_change"]
        )
        draws_m.append(cell)
    for target, draws in ((structure, draws_s), (mdc, draws_m)):
        for key in draws[0] if draws else ():
            low, high, valid = E.interval([row[key] for row in draws], budget)
            target.update(
                {
                    key + "_ci_low": low,
                    key + "_ci_high": high,
                    key + "_bootstrap_valid": valid,
                }
            )
    for cohort in ("original_complete", "incomplete"):
        dens = np.array([row[cohort + "_mean_abs_control"] for row in draws_m])
        mdc[cohort + "_zero_control_density_draws"] = int((dens == 0).sum())
        mdc[cohort + "_undefined_ratio_draws"] = int(
            sum(not np.isfinite(row[cohort + "_mdc"]) for row in draws_m)
        )
        mdc[cohort + "_control_density_bootstrap_cv"] = (
            float(np.std(dens, ddof=1) / np.mean(dens)) if np.mean(dens) > 0 else np.nan
        )
    # Re-estimating thousands of noisy edge weights can attenuate the resampled
    # concordance distribution. Do not present that distribution as a calibrated
    # population-correlation CI or hide the fact it may exclude the point.
    for key in draws_s[0]:
        if "concordance" in key or key.startswith("hub_"):
            structure[key + "_resampling_low"] = structure.pop(key + "_ci_low")
            structure[key + "_resampling_high"] = structure.pop(key + "_ci_high")
    structure["concordance_uncertainty"] = (
        "95% donor-resampling stability range, not a calibrated population-concordance CI; rebuilding noisy networks can attenuate it below the point estimate"
    )
    structure["sample_matched_control_n"] = match_n
    for key in matched[0]:
        low, high, valid = E.interval([row[key] for row in matched], budget)
        structure["matched_" + key] = float(np.nanmedian([row[key] for row in matched]))
        structure["matched_" + key + "_resampling_low"] = low
        structure["matched_" + key + "_resampling_high"] = high
    structure["sample_matching"] = (
        "Equal Control N; uniform without-replacement donor subsamples, no outcome or validation-guided selection; descriptive sensitivity"
    )
    matrices = {
        k: np.asarray(E.group_edges(v, r, c, power)[1]) for k, v in values.items()
    }
    np.savez_compressed(
        E.PRIVATE / (prefix + ".npz"), rows=r, columns=c, tissues=tissues, **matrices
    )
    group_figures(prefix, matrices, r, c, tissues, structure)
    atomic_json(checkpoint, dict(signature=signature, structure=structure, mdc=mdc))
    return structure, mdc


def group_figures(prefix, matrices, r, c, tissues, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    maximum = max(
        float(np.max(abs(matrices[k])))
        for k in ("original_complete_Control", "incomplete_Control")
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, key, title in zip(
        axes[:2],
        ("original_complete_Control", "incomplete_Control"),
        ("Original Controls", "Incomplete-region Controls"),
    ):
        matrix = np.full((len(tissues), len(tissues)), np.nan)
        matrix[r, c] = abs(matrices[key])
        matrix[c, r] = abs(matrices[key])
        if len(set(tissues)) == 2:
            matrix = matrix[np.ix_(tissues == "DLPFC", tissues == "PCG")]
            ax.set_xlabel("PCG nodes")
            ax.set_ylabel("DLPFC nodes")
        else:
            ax.set_xlabel("DLPFC nodes")
            ax.set_ylabel("DLPFC nodes")
        image = ax.imshow(
            matrix,
            cmap="YlOrRd",
            norm=Normalize(0, maximum),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(
            title
            + f" (N={summary['original_control_n'] if key.startswith('original') else summary['validation_control_n']})"
        )
    axes[2].scatter(
        abs(matrices["original_complete_Control"]),
        abs(matrices["incomplete_Control"]),
        s=1,
        alpha=0.15,
    )
    axes[2].plot([0, maximum], [0, maximum], color="black", ls="--", lw=1)
    axes[2].set(
        xlabel="Original |adjacency|",
        ylabel="Validation |adjacency|",
        xlim=(0, maximum),
        ylim=(0, maximum),
        title=f"Edge magnitude concordance r={summary['magnitude_concordance']:.3f}",
    )
    fig.colorbar(
        image, ax=list(axes[:2]), label="Absolute signedAlt adjacency", shrink=0.7
    )
    fig.suptitle(
        prefix.replace("__", " · ")
        + " · fixed nodes/order; descriptive edges, donor-bootstrap uncertainty"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(E.PRIVATE / (prefix + "." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def refresh_manifest(stage):
    files = {}
    for path in E.PUBLIC.glob("*.csv"):
        files[path.name] = dict(sha256=E.sha(path), rows=len(pd.read_csv(path)))
    source_spec = json.loads((E.REPORT / "analysis_spec.json").read_text())
    atomic_json(
        E.PUBLIC / "manifest.json",
        dict(
            version=E.VERSION,
            status=stage,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            focus=E.FOCUS,
            components=E.COMPONENTS,
            source_score_release=source_spec["source_release"],
            source_adjusted_analysis_id=source_spec["analysis_id"],
            input_sha256={
                str(p.relative_to(APP)): input_sha(p)
                for p in [
                    RESEARCH_ROOT / "catalog.json",
                    RESEARCH_ROOT / "manifest.json",
                    RESEARCH_ROOT / "private/brain_reference.json",
                    RESEARCH_ROOT / "private/brain_reference.npz",
                    RESEARCH_ROOT / "private/brain/edges.npy",
                    RESEARCH_ROOT / "private/brain/calibration_beta3.npy",
                    RESEARCH_ROOT / "metadata/brain.parquet",
                    E.REPORT / "report_manifest.json",
                ]
            },
            producer_sha256={
                str(p.relative_to(APP)): E.sha(p)
                for p in [
                    Path(__file__),
                    APP / "scripts/validation_evidence_edges.py",
                    APP / "app_helpers/validation_evidence.py",
                    APP / "app_helpers/validation_evidence_ui.py",
                    APP / "scripts/export_validation_evidence.py",
                ]
            },
            files=files,
            artifacts={
                str(p.relative_to(E.PUBLIC)): dict(
                    sha256=E.sha(p), bytes=p.stat().st_size
                )
                for p in sorted(
                    [
                        *E.PUBLIC.glob("figures/*.png"),
                        *E.PUBLIC.glob("figures/*.pdf"),
                        *E.PUBLIC.glob("RESULTS.md"),
                        *E.PUBLIC.glob("QA.json"),
                    ]
                )
            },
            local_only="Donor-edge arrays, exact candidates, paired network heatmaps and edge scatter figures are not published",
            definitions=dict(
                score_beta="power within frozen calibrated donor LIONESS estimator; beta3 primary, beta1 in existing report",
                adjacency_power="group signedAlt adjacency: TS=3, CT=2; distinct from donor score beta",
                uncertainty="199 study-stratified donor bootstrap draws; rebuild group networks, not independent edge resampling",
                new_diagnostic_inference="bootstrap intervals only; no new structure/MDC permutation p or FDR",
                legacy_mdc="existing dual-null permutation p and full-family BH retained, historical donor overlap shown separately",
            ),
        ),
    )


def finalize(info, meta, x, ref, jobs):
    """Small direct checks and exact availability, not another scientific scan."""
    rows = []
    for job in jobs:
        for component in E.COMPONENTS:
            prefix = f"{job['family']}__M{job['module']}__{component}"
            selection = E.PRIVATE / (prefix + "__selection.parquet")
            if not selection.is_file():
                rows.append(
                    dict(
                        partition=job["family"],
                        module=job["module"],
                        component=component,
                        analysis="same edges",
                        status="unavailable",
                        reason="Frozen original-only selection not completed",
                    )
                )
                continue
            candidates = pd.read_parquet(selection)
            edge = int(candidates.edge_code.iloc[0])
            a, b = divmod(edge, info["n_nodes"])
            frozen = ref["values"][:, [a, b]]
            r0 = np.corrcoef(frozen, rowvar=False)[0, 1]
            with np.load(E.PRIVATE / (prefix + "__selected_donor_edges.npz")) as stored:
                for cohort in ("original_complete", "incomplete"):
                    keep = E.cohort_selection(meta, component, cohort, independent=True)
                    sub = meta.loc[keep].reset_index(drop=True)
                    values = np.asarray(x[np.ix_(np.flatnonzero(keep), [a, b])])
                    values = np.where(np.isfinite(values), values, ref["mean"][[a, b]])
                    for diagnosis in ("AD", "Control"):
                        index = int(
                            np.flatnonzero(sub.diagnosis_group.eq(diagnosis))[0]
                        )
                        r1 = np.corrcoef(
                            np.vstack([frozen, values[index]]), rowvar=False
                        )[0, 1]
                        direct = (len(frozen) + 1) * r1**3 - len(frozen) * r0**3
                        error = abs(
                            direct - float(stored[cohort + "_weights"][index, 0])
                        )
                        if error > 1e-10:
                            raise ValueError("Direct frozen-R add-one check failed")
                        rows.append(
                            dict(
                                partition=job["family"],
                                module=job["module"],
                                component=component,
                                analysis="sampled direct add-one "
                                + cohort
                                + " "
                                + diagnosis,
                                status="passed",
                                max_abs_error=error,
                                tolerance=1e-10,
                            )
                        )
            rows.append(
                dict(
                    partition=job["family"],
                    module=job["module"],
                    component=component,
                    analysis="structure + disjoint MDC",
                    status="complete",
                    reason="199 donor resamples; structure concordance uses stability ranges; MDC has bootstrap CIs, no new permutation q",
                )
            )
    for name in E.REPORT_TABLES:
        frame = E.report_table(name)
        rows.append(
            dict(
                partition="four fixed brain namespaces",
                module=np.nan,
                component="DLPFC and DLPFC–PCG",
                analysis="read-only adjusted report " + name,
                status="complete",
                rows=len(frame),
                reason="Exact existing public source/hash; original q scopes retained, eight endpoints and beta1/beta3",
            )
        )
    rows.append(
        dict(
            partition="four fixed brain namespaces",
            module=np.nan,
            component="DLPFC and DLPFC–PCG",
            analysis="incomplete incremental Connectivity + PC1",
            status="unavailable",
            reason="Saved incremental.csv: no_shared_comparator_donors. Existing module_eigengenes.parquet provides original450 scores only; no compatible frozen incomplete-cohort PC1 projection is registered. No validation PCA refit or substitute.",
        )
    )
    public_csv("availability", pd.DataFrame(rows))
    # Local figures/arrays were regenerated by the structure stage; refresh only
    # their receipts, not their contents, after the expensive work has completed.
    receipt = E.PRIVATE / "manifest.json"
    value = json.loads(receipt.read_text())
    value["local_artifacts"] = {
        p.name: E.sha(p)
        for p in E.PRIVATE.iterdir()
        if p.suffix in (".png", ".pdf", ".npz")
        and "__selected_donor_edges" not in p.name
    }
    atomic_json(receipt, value)
    refresh_manifest("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "structure", "edges", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--bootstrap", type=int, default=199)
    args = parser.parse_args()
    E.PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    catalog, info, meta, x, presence, ref, jobs = inputs()
    if args.stage in ("audit", "all"):
        audit(info, meta, jobs)
        refresh_manifest("cohort audit complete; focused diagnostics running")
        print(json.dumps(dict(stage="audit", status="complete")), flush=True)
    if args.stage in ("structure", "all"):
        structures, ratios = [], []
        for job in jobs:
            for component in E.COMPONENTS:
                s, m = structure_job(job, component, info, meta, x, ref, args.bootstrap)
                structures.append(s)
                ratios.append(m)
                public_csv("structure", pd.DataFrame(structures))
                public_csv("mdc_decomposition", pd.DataFrame(ratios))
                refresh_manifest("structure and MDC running")
                print(
                    json.dumps(
                        dict(
                            stage="structure",
                            family=job["family"],
                            module=job["module"],
                            component=component,
                            status="complete",
                            elapsed_seconds=round(time.monotonic() - started, 1),
                        )
                    ),
                    flush=True,
                )
    if args.stage in ("edges", "all"):
        from validation_evidence_edges import run_edges

        run_edges(info, meta, x, presence, ref, jobs, args.bootstrap)
    if args.stage in ("finalize", "all"):
        finalize(info, meta, x, ref, jobs)
    else:
        refresh_manifest(args.stage + " complete; see table coverage")
    print(
        json.dumps(
            dict(
                status="complete",
                seconds=time.monotonic() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
