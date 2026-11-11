"""Shared public-summary adapters for every descriptive score consumer.

These readers never fit references, train models, or open private expression.
"""

from pathlib import Path
from functools import lru_cache
import json, hashlib
import numpy as np
import pandas as pd
from .registry import ROOT, METRICS, ESTIMATORS, is_calibrated, require_capability
from .inputs import sha
from . import hosted as H


class ScoreUnavailable(ValueError):
    pass


def catalog(root=ROOT):
    path = H.ensure(Path(root).resolve() / "catalog.json")
    stat = path.stat()
    return _catalog_content(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _catalog_content(path, revision, size):
    """Reuse the immutable catalog across bounded per-module readers.

    Catalogs include physical node identities and can be tens of megabytes.
    Re-parsing them for every module would turn one catalog scan into hundreds
    of redundant JSON decodes. Atomic catalog replacement changes this key.
    """
    return json.loads(Path(path).read_text())


def component_public(value):
    return str(value).replace("PCGBA23", "PCG")


def component_legacy(value):
    return str(value).replace("PCG", "PCGBA23")


def module_filename(key):
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".parquet"


@lru_cache(maxsize=256)
def verify_score(path, mtime, size, expected):
    if sha(path) != expected:
        raise ValueError("Cached score content hash mismatch")
    return True


@lru_cache(maxsize=4)
def file_index(root, variant, revision):
    rows = json.loads((Path(root) / "file_indexes" / f"{variant}.json").read_text())
    return {r["path"]: r["sha256"] for r in rows}


def variant_for(family, root=ROOT):
    c = catalog(root)
    matches = [j["input_variant"] for j in c["modules"] if j["family"] == family]
    if not matches:
        raise ScoreUnavailable("Partition has no verified calibrated input alignment")
    return matches[0]


def metadata(module_set="full_cohort", cohort_scope="complete_450", root=ROOT, **_):
    c = catalog(root)
    variant = variant_for(module_set, root)
    f = pd.read_parquet(H.ensure(Path(root) / c["datasets"][variant]["metadata_file"]))
    keep = f.role.eq("target")
    if cohort_scope in ("complete_450", "common_450", "common_357"):
        keep &= f.in_original_analysis
    elif cohort_scope == "incomplete_region_validation":
        keep &= ~f[[col for col in f if col.startswith("has_")]].all(axis=1)
    elif cohort_scope not in ("maximum_component", "maximum_tissue"):
        raise ScoreUnavailable("Unknown calibrated cohort scope")
    return f.loc[keep].copy()


def scores(
    method="control_anchored",
    module=None,
    metric_family=None,
    component=None,
    module_set="full_cohort",
    estimator="calibrated_beta3",
    cohort_scope="complete_450",
    metric_scale=None,
    include_embedded_metadata=True,
    edge_rule="all",
    differential_edge_rule="all",
    differential_fdr_scope="global",
    differential_fdr_threshold=0.05,
    score_normalization="standard_pruned",
    regional_tissue=None,
    root=ROOT,
    metric_families=None,
    **_,
):
    require_capability(estimator, "donor_scores")
    if method != "control_anchored":
        raise ScoreUnavailable(
            "Calibrated scores require the frozen B control R/C protocol"
        )
    if edge_rule != "all":
        raise ScoreUnavailable(
            "This edge rule has no verified frozen physical-edge mapping"
        )
    if metric_family is not None and metric_family not in METRICS:
        raise ScoreUnavailable("The selected metric is not a calibrated donor score")
    selected_metrics = (
        [metric_family]
        if metric_family
        else list(METRICS if metric_families is None else metric_families)
    )
    if not set(selected_metrics) <= set(METRICS):
        raise ScoreUnavailable("Unknown calibrated donor measure")
    root = Path(root)
    c = catalog(root)
    variant = variant_for(module_set, root)
    beta = ESTIMATORS[estimator]
    scope = {
        "common_450": "complete_450",
        "common_357": "complete_450",
        "maximum_tissue": "maximum_component",
    }.get(cohort_scope, cohort_scope)
    jobs = [
        j
        for j in c["modules"]
        if j["family"] == module_set
        and (module is None or j["module"] == int(module))
        and (
            regional_tissue is None
            or j["regional_tissue"] == component_public(regional_tissue)
        )
    ]
    if (
        c["families"][module_set]["regional"]
        and regional_tissue is None
        and module is not None
    ):
        raise ScoreUnavailable("Regional module IDs require their tissue namespace")
    parts = []
    for job in jobs:
        if H.is_hosted(root):
            from .masks import key as mask_key

            metadata_path = H.ensure(root / c["datasets"][variant]["metadata_file"])
            full_metadata = pd.read_parquet(metadata_path)
            selected_mask = (
                "all"
                if differential_edge_rule == "all"
                else mask_key(
                    differential_edge_rule,
                    differential_fdr_scope,
                    differential_fdr_threshold,
                )
            )
            frame = H.read_scores(
                root,
                job,
                beta,
                selected_mask,
                full_metadata,
                component_public(component) if component else None,
            )
            selected_ids = metadata(module_set, scope, root).sample_id
            frame = frame.loc[frame.sample_id.isin(selected_ids)].copy()
            frame["cohort_scope"] = scope
        else:
            frame = _local_score_frame(
                root,
                c,
                job,
                variant,
                beta,
                scope,
                module_set,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
            )
        if len(frame) and (
            not frame.role.eq("target").all()
            or not frame.beta.eq(beta).all()
            or not frame.module_key.eq(job["module_key"]).all()
        ):
            raise ValueError("Cached score role/beta/module identity mismatch")
        if component is not None:
            frame = frame.loc[frame.component.eq(component_public(component))]
        parts.extend(
            _metric_frames(
                frame,
                selected_metrics,
                metric_scale,
                estimator,
                beta,
                edge_rule,
                differential_edge_rule,
            )
        )
    result = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=[
                "sample_id",
                "module",
                "metric_family",
                "component",
                "component_class",
                "component_label",
                "metric_raw",
                "metric_asinh",
                "metric_rint",
                "lioness_method",
            ]
        )
    )
    if include_embedded_metadata and not result.empty:
        meta = metadata(module_set, scope, root)
        extra = [col for col in meta if col not in result or col == "sample_id"]
        result = result.merge(
            meta[extra], on="sample_id", how="left", validate="many_to_one"
        )
    if not include_embedded_metadata:
        result = result.drop(columns=["diagnosis_group"], errors="ignore")
    return result


def _local_score_frame(
    root,
    c,
    job,
    variant,
    beta,
    scope,
    module_set,
    differential_edge_rule,
    differential_fdr_scope,
    differential_fdr_threshold,
):
    path = (
        root
        / "scores"
        / f"beta{beta}"
        / module_set
        / scope
        / module_filename(job["module_key"])
    )
    if differential_edge_rule != "all":
        from .masks import mask_artifact_path

        path = mask_artifact_path(
            root,
            beta,
            module_set,
            scope,
            job["module_key"],
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        )
    if not path.exists():
        raise ScoreUnavailable(
            f"Cached calibrated score is unavailable: {module_set}, beta{beta}, {scope}, {differential_edge_rule}"
        )
    if differential_edge_rule == "all":
        index_path = root / "file_indexes" / f"{variant}.json"
        expected = file_index(str(root), variant, index_path.stat().st_mtime_ns).get(
            str(path.relative_to(root))
        )
    else:
        from .masks import key

        mm = json.loads((root / "masks/manifest.json").read_text())
        mask_key = key(
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        )
        record = mm["modules"][job["module_key"]][mask_key]
        if "files" not in record:
            record = json.loads(
                (
                    root
                    / "masks/module_completion"
                    / module_filename(job["module_key"]).replace(".parquet", ".json")
                ).read_text()
            )["masks"][mask_key]
        expected = next(
            x["sha256"]
            for x in record["files"]
            if x["path"] == str(path.relative_to(root))
        )
    if expected is None:
        raise ScoreUnavailable("Score artifact is absent from its completed file index")
    stat = path.stat()
    verify_score(str(path), stat.st_mtime_ns, stat.st_size, expected)
    frame = pd.read_parquet(path)
    return frame


def _metric_frames(
    frame,
    selected_metrics,
    metric_scale,
    estimator,
    beta,
    edge_rule,
    differential_edge_rule,
):
    parts = []
    identity = [
        col
        for col in frame
        if col not in METRICS
        and not any(col == m + "_" + s for m in METRICS for s in ("asinh", "rint"))
    ]
    for metric in selected_metrics:
        out = frame[identity].copy()
        out["metric_family"] = metric
        out["magnitude_unavailable_reason"] = frame.unavailable_reason.to_numpy()
        out["magnitude_eligible_edges"] = frame.eligible_edges.to_numpy()
        if metric in ("signed_distance", "signed_abs_z"):
            finite = np.isfinite(frame[metric])
            out["eligible_edges"] = frame.signed_eligible_edges.to_numpy()
            out["scored_edges"] = np.where(finite, frame.signed_eligible_edges, np.nan)
            out["unavailable_reason"] = np.where(
                finite,
                "ok",
                np.where(
                    frame.unavailable_reason.eq("missing_required_tissue"),
                    "missing_required_tissue",
                    np.where(
                        frame.signed_eligible_edges.eq(0),
                        "no_eligible_signed_edges",
                        "incomplete_signed_target_edge_coverage",
                    ),
                ),
            )
        out["metric_eligible_edges"] = out.eligible_edges
        for scale in [metric_scale] if metric_scale else ("raw", "asinh", "rint"):
            out["metric_" + scale] = frame[
                metric if scale == "raw" else metric + "_" + scale
            ].to_numpy()
        out["component"] = out.component.map(component_legacy)
        out["component_label"] = out.component.map(
            lambda v: (
                v.replace("TS_", "")
                .replace("CT_", "")
                .replace("__", "–")
                .replace("PCGBA23", "PCG")
                if v not in ("TS", "CT", "all")
                else {
                    "TS": "All within-tissue edges",
                    "CT": "All cross-tissue edges",
                    "all": "Whole module (fixed physical union)",
                }[v]
            )
        )
        out["lioness_method"] = "control_anchored"
        out["network_method"] = "control_anchored"
        out["estimator"] = estimator
        out["score_version"] = f"control_calibrated_connectivity.v2.beta{beta}"
        out["edge_rule"] = edge_rule
        out["differential_edge_rule"] = differential_edge_rule
        out["score_normalization"] = "common_eligible_physical_edges"
        parts.append(out)
    return parts


def aggregate(**kwargs):
    frame = scores(**kwargs)
    parts = []
    contextual = [
        "eligible_edges",
        "possible_edges",
        "scored_edges",
        "signed_eligible_edges",
        "loss_bound",
        "unavailable_reason",
        "artifact_hash",
        "edge_universe_sha256",
        "sparse_endpoint_cells_imputed",
        "absent_tissue_endpoint_cells",
        "metric_eligible_edges",
        "magnitude_eligible_edges",
        "magnitude_unavailable_reason",
    ] + [
        side + "_" + c
        for side in ("gain", "loss")
        for c in ("zero", "near_zero", "uncertain", "preserved", "reversed")
    ]
    for (module_key, module, metric), group in frame.groupby(
        ["module_key", "module", "metric_family"], sort=False
    ):
        pooled = group[group.component.isin(["TS", "CT"])]
        base = pooled.drop_duplicates("sample_id").drop(
            columns=[
                c
                for c in pooled
                if c.startswith("metric_")
                or c in ("component", "component_class", "component_label")
            ]
        )
        base["metric_family"] = metric
        base = base.drop(columns=contextual, errors="ignore")
        for component in ("CT", "TS"):
            component_rows = pooled[pooled.component.eq(component)].set_index(
                "sample_id"
            )
            for col in contextual:
                if col in component_rows:
                    base[component + "_" + col] = base.sample_id.map(
                        component_rows[col]
                    )
            for scale in ("raw", "asinh", "rint"):
                col = "metric_" + scale
                if col in pooled:
                    base[component + "_" + scale] = base.sample_id.map(
                        pooled[pooled.component.eq(component)].set_index("sample_id")[
                            col
                        ]
                    )
        parts.append(base)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def baseline_associations(
    module_set="full_cohort",
    estimator="calibrated_beta3",
    cohort_scope="complete_450",
    module=None,
    metric_family=None,
    phenotype=None,
    component=None,
    regional_tissue=None,
    root=ROOT,
    inference_version="auto",
    minimum_n=3,
    **_,
):
    import pyarrow.parquet as pq

    c = catalog(root)
    dataset = c["families"][module_set]["dataset"]
    from .validation_data import baseline_path

    path = baseline_path(root, dataset, cohort_scope, inference_version)
    if path is None:
        path = Path(root) / "associations" / f"{dataset}_{cohort_scope}.parquet"
        if not H.available(path):
            raise ScoreUnavailable(
                "The full baseline association catalog has not finished"
            )
        path = H.ensure(path)
        completion = Path(root) / "associations/completion.json"
        if not H.available(completion):
            raise ScoreUnavailable(
                "The complete baseline scan is still being finalized"
            )
        report = H.read_json(completion)
        expected = next(
            item["sha256"]
            for item in report["files"]
            if item["path"] == str(path.relative_to(root))
        )
        stat = path.stat()
        verify_score(str(path), stat.st_mtime_ns, stat.st_size, expected)
    filters = [("family", "=", module_set), ("beta", "=", ESTIMATORS[estimator])]
    if c["families"][module_set]["regional"] and module is not None:
        if regional_tissue is None:
            raise ScoreUnavailable(
                "Regional association module IDs require their tissue namespace"
            )
    for col, value in [
        ("metric_family", metric_family),
        ("outcome", phenotype),
        ("component", component_public(component) if component else None),
    ]:
        if value is not None:
            filters.append((col, "=", value))
    out = pq.read_table(path, filters=filters).to_pandas()
    out["component"] = out.component.map(component_legacy)
    if regional_tissue is not None:
        out = out[
            out.module_key.str.contains(
                "::" + component_public(regional_tissue) + "::", regex=False
            )
        ]
    # Correct the complete fixed partition family BEFORE selecting a module.
    # Preserve the immutable scan-wide q columns as historical alternatives.
    from .association_fdr import from_baseline
    saved_p = out.p.copy()
    out.loc[out.n.lt(int(minimum_n)), "p"] = np.nan
    out = from_baseline(out)
    out["p"] = saved_p
    out["module_fdr_minimum_n"] = int(minimum_n)
    out["module_fdr_exclusion_reason"] = np.where(out.n.lt(int(minimum_n)),
        f"n < {int(minimum_n)}; saved descriptive test retained, excluded from module BH", "")
    if module is not None:
        out = out.loc[out.module.eq(int(module))].copy()
    return out


def statistics(**kwargs):
    """Native baseline rows and aliases; additional searches are explicit."""
    if kwargs.get("method", "control_anchored") != "control_anchored":
        raise ScoreUnavailable(
            "Calibrated baseline statistics require the frozen control R/C protocol"
        )
    if (
        kwargs.get("differential_edge_rule", "all") != "all"
        or kwargs.get("analysis_subset", "all_donors") != "all_donors"
        or kwargs.get("diagnosis_group") not in (None, "All donors")
    ):
        raise ScoreUnavailable(
            "This mask or donor subgroup is an additional descriptive search; baseline statistics will not be substituted. Use its saved analysis context."
        )
    if kwargs.get("metric_family") not in (None, *METRICS.keys()):
        raise ScoreUnavailable("Unknown calibrated metric")
    frame = baseline_associations(**kwargs)
    frame["phenotype"] = frame.outcome
    frame["diagnosis_group"] = "All donors"
    frame["scale"] = "raw"
    frame["lioness_method"] = "control_anchored"
    frame["component_label"] = frame.component
    frame["spearman_rho"] = np.where(frame.method.eq("Spearman"), frame.effect, np.nan)
    frame["spearman_p"] = np.where(frame.method.eq("Spearman"), frame.p, np.nan)
    frame["spearman_fdr"] = np.where(frame.method.eq("Spearman"), frame.q_across_modules, np.nan)
    frame["spearman_fdr_across_modules"] = frame.spearman_fdr
    frame["spearman_fdr_module_family_n"] = np.where(frame.method.eq("Spearman"), frame.module_family_n, 0)
    frame["pearson_r"] = np.nan
    frame["pearson_p"] = np.nan
    frame["pearson_fdr"] = np.nan
    frame["categorical_p"] = np.where(
        frame.method.eq("Kruskal-Wallis"), frame.p, np.nan
    )
    frame["categorical_fdr"] = np.where(
        frame.method.eq("Kruskal-Wallis"), frame.q_across_modules, np.nan
    )
    frame["categorical_fdr_across_modules"] = frame.categorical_fdr
    frame["epsilon_squared"] = np.where(
        frame.method.eq("Kruskal-Wallis"), frame.effect, np.nan
    )
    frame["rho"] = frame.spearman_rho
    frame["p_spearman"] = frame.spearman_p
    for scale in ("raw", "asinh", "rint"):
        frame["r_" + scale] = np.nan
        frame["p_" + scale] = np.nan
        frame["rho_" + scale] = frame.spearman_rho if scale == "raw" else np.nan
        frame["p_spearman_" + scale] = frame.spearman_p if scale == "raw" else np.nan
    return frame


def coverage(module_set=None, root=ROOT):
    paths = [
        H.ensure(Path(root) / "coverage" / (v + ".parquet"))
        for v in catalog(root)["datasets"]
    ]
    f = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return f if module_set is None else f[f.family.eq(module_set)].copy()


def role_status(dataset, sample_ids=None, root=ROOT):
    c = catalog(root)
    info = c["datasets"][dataset]
    frame = pd.read_parquet(H.ensure(Path(root) / info["metadata_file"]))
    frame = frame[["sample_id", "role", "in_original_analysis"]].copy()
    frame["independent_B_score_available"] = frame.role.eq("target")
    frame["independent_B_unavailable_reason"] = frame.role.map(
        {
            "reference": "Reference role under frozen B; no independent target score",
            "calibration": "Calibration role under frozen B; no independent target score",
            "target": "",
        }
    )
    return (
        frame if sample_ids is None else frame[frame.sample_id.isin(sample_ids)].copy()
    )


def edge_summaries(**kwargs):
    frame = scores(**kwargs)
    base = frame.drop_duplicates(["sample_id", "component"]).copy()
    values = frame.pivot(
        index=["sample_id", "component"], columns="metric_family", values="metric_raw"
    ).reset_index()
    base = base.drop(
        columns=["metric_family", "metric_raw", "metric_asinh", "metric_rint"],
        errors="ignore",
    ).merge(values, on=["sample_id", "component"], validate="one_to_one")
    base["scope"] = base.component
    base["scope_label"] = base.component_label
    return base
