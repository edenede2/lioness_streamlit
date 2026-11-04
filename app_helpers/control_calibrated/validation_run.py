"""Checkpointed catalog execution on immutable native donor-score summaries."""

from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, time, math, os, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import validation_inference as I
from .validation_spec import (
    WORK,
    BASE,
    OUTCOMES,
    METRICS,
    COHORTS,
    FOCUS,
    FOCUS_COMPONENTS,
)
from .registry import RESEARCH_ROOT
from .inputs import atomic_json, atomic_parquet, sha
from .core import canonical_hash
from .analysis import correct, bh


def cohort_mask(meta, scope):
    original = meta.role.eq("target") & meta.in_original_analysis
    incomplete = meta.role.eq("target") & ~meta[
        [c for c in meta if c.startswith("has_")]
    ].all(axis=1)
    if (original & incomplete).any():
        raise ValueError("Original and incomplete target donors overlap")
    if scope in ("original", "complete_450"):
        return original.to_numpy()
    if scope in ("incomplete", "incomplete_region_validation"):
        return incomplete.to_numpy()
    if scope == "maximum_component":
        return meta.role.eq("target").to_numpy()
    raise ValueError("Unknown donor scope")


def _component_order(component):
    return (0 if component.startswith(("TS_", "CT_")) else 1, component)


def prepare(root, spec):
    root = Path(root)
    dest = root / "prepared"
    dest.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((RESEARCH_ROOT / "catalog.json").read_text())
    totals = []
    receipts = []
    for variant, info in catalog["datasets"].items():
        completed = dest / (variant + ".json")
        if completed.exists():
            old = json.loads(completed.read_text())
            if old["analysis_id"] != spec["analysis_id"]:
                raise ValueError("Prepared input identity differs")
            totals.append(old)
            continue
        meta = pd.read_parquet(RESEARCH_ROOT / info["metadata_file"])
        meta = (
            meta.loc[meta.role.eq("target")]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        assert meta.sample_id.is_unique
        original = cohort_mask(meta, "original")
        incomplete = cohort_mask(meta, "incomplete")
        atomic_parquet(dest / (variant + "_metadata.parquet"), meta)
        index = json.loads(
            (RESEARCH_ROOT / "file_indexes" / f"{variant}.json").read_text()
        )
        source = [r for r in index if r["scope"] == "maximum_component"]
        arrays = []
        rows = []
        jobs = {j["module_key"]: j for j in catalog["modules"]}
        for i, item in enumerate(source):
            path = RESEARCH_ROOT / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError("Native score source changed: " + item["path"])
            frame = pd.read_parquet(
                path,
                columns=[
                    "sample_id",
                    "component",
                    *METRICS,
                    "possible_edges",
                    "eligible_edges",
                    "edge_universe_sha256",
                    "role",
                    "beta",
                    "module_key",
                    "allocation_id",
                ],
            )
            if (
                not frame.role.eq("target").all()
                or not frame.beta.eq(item["beta"]).all()
                or not frame.module_key.eq(item["module_key"]).all()
            ):
                raise ValueError("Native score identity/role mismatch")
            job = jobs[item["module_key"]]
            physical = {}
            for component, g in frame.groupby("component", sort=False):
                if g.possible_edges.iloc[0] <= 0:
                    continue
                if (
                    g.sample_id.duplicated().any()
                    or not g.sample_id.isin(meta.sample_id).all()
                ):
                    raise ValueError("Invalid score donor membership")
                g = g.set_index("sample_id").reindex(meta.sample_id)
                physical[component] = g
            aliases = {}
            for component, g in physical.items():
                aliases.setdefault(g.edge_universe_sha256.dropna().iloc[0], []).append(
                    component
                )
            canonical = {}
            for universe, members in aliases.items():
                groups = []
                for alias in sorted(members, key=_component_order):
                    group = next(
                        (
                            g
                            for g in groups
                            if np.allclose(
                                physical[alias][list(METRICS)],
                                physical[g[0]][list(METRICS)],
                                rtol=1e-12,
                                atol=1e-12,
                                equal_nan=True,
                            )
                            and np.allclose(
                                physical[alias].eligible_edges,
                                physical[g[0]].eligible_edges,
                                equal_nan=True,
                            )
                        ),
                        None,
                    )
                    if group is None:
                        groups.append([alias])
                    else:
                        group.append(alias)
                if (
                    job["regional"]
                    if "regional" in job
                    else bool(job["regional_tissue"])
                ):
                    if len(groups) != 1:
                        raise ValueError("Regional physical aliases differ")
                for group in groups:
                    for alias in group:
                        canonical[alias] = (group[0], group)
            for component, g in physical.items():
                chosen, members = canonical[component]
                for metric in METRICS:
                    values = g[metric].to_numpy(float)
                    arrays.append(values)
                    rows.append(
                        dict(
                            row_id=len(rows),
                            module_key=job["module_key"],
                            family=job["family"],
                            module=job["module"],
                            regional_tissue=job["regional_tissue"],
                            dataset=info["dataset"],
                            input_variant=variant,
                            component=component,
                            metric_family=metric,
                            beta=item["beta"],
                            allocation_id=info["allocation_id"],
                            possible_edges=int(g.possible_edges.dropna().iloc[0]),
                            edge_universe_sha256=g.edge_universe_sha256.dropna().iloc[
                                0
                            ],
                            canonical_component=chosen,
                            is_canonical=component == chosen,
                            aliases=json.dumps(members),
                            focused=job["module"] in FOCUS.get(job["family"], ())
                            and chosen in FOCUS_COMPONENTS,
                            native_values_sha256=hashlib.sha256(
                                values.tobytes()
                            ).hexdigest(),
                        )
                    )
            receipts.append(dict(path=item["path"], sha256=item["sha256"]))
            if i % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="prepare",
                            variant=variant,
                            modules=i + 1,
                            total=len(source),
                        )
                    ),
                    flush=True,
                )
        values = np.stack(arrays)
        temporary = dest / (variant + ".partial.npy")
        with temporary.open("wb") as f:
            np.save(f, values)
        temporary.replace(dest / (variant + ".npy"))
        specs = pd.DataFrame(rows)
        atomic_parquet(dest / (variant + "_rows.parquet"), specs)
        record = dict(
            analysis_id=spec["analysis_id"],
            variant=variant,
            dataset=info["dataset"],
            modules=len(source) // 2,
            vectors=len(rows),
            canonical_vectors=int(specs.is_canonical.sum()),
            donors=len(meta),
            original=int(original.sum()),
            incomplete=int(incomplete.sum()),
            other_complete_targets=int((~original & ~incomplete).sum()),
            overlap=int((original & incomplete).sum()),
            allocation_id=info["allocation_id"],
            matrix_sha256=sha(dest / (variant + ".npy")),
            metadata_sha256=sha(dest / (variant + "_metadata.parquet")),
            matrix_bytes=values.nbytes,
        )
        atomic_json(completed, record)
        totals.append(record)
        del arrays, values
    atomic_json(
        root / "native_input_receipts.json",
        dict(analysis_id=spec["analysis_id"], files=receipts, variants=totals),
    )
    atomic_json(
        root / "preparation_complete.json", dict(status="complete", variants=totals)
    )
    return totals


@lru_cache(maxsize=3)
def prepared(root, variant):
    dest = Path(root) / "prepared"
    return (
        np.load(dest / (variant + ".npy"), mmap_mode="r"),
        pd.read_parquet(dest / (variant + "_metadata.parquet")),
        pd.read_parquet(dest / (variant + "_rows.parquet")),
    )


def _patterns(mask, row_ids, chunk=2048):
    patterns, inverse = np.unique(
        np.packbits(mask, axis=1), axis=0, return_inverse=True
    )
    for k in range(len(patterns)):
        ix = np.flatnonzero(inverse == k)
        donors = np.flatnonzero(mask[ix[0]])
        for start in range(0, len(ix), chunk):
            yield row_ids[ix[start : start + chunk]].tolist(), donors.tolist()


def plan_tasks(root, spec, stage="scan"):
    tasks = []
    provenance_path = Path(root) / "provenance_inventory.json"
    pathology_verified = provenance_path.exists() and json.loads(
        provenance_path.read_text()
    ).get("pathology_sensitivity_verified", False)
    catalog = json.loads((RESEARCH_ROOT / "catalog.json").read_text())
    for variant, info in catalog["datasets"].items():
        matrix, meta, rows = prepared(str(root), variant)
        keep = (
            np.ones(len(rows), bool)
            if stage == "repair"
            else rows.is_canonical.to_numpy()
        )
        scopes = (
            ("complete_450", "incomplete_region_validation", "maximum_component")
            if stage == "repair"
            else COHORTS
        )
        outcomes = (
            tuple(
                json.loads(
                    (RESEARCH_ROOT / "associations/testing_manifest.json").read_text()
                )["numeric_outcomes"]
            )
            if stage == "repair"
            else OUTCOMES
        )
        finite = np.isfinite(matrix)
        for scope in scopes:
            cohort = cohort_mask(meta, scope)
            for outcome in outcomes:
                y = (
                    I.finite_numeric(meta[outcome])
                    if outcome in meta
                    else np.full(len(meta), np.nan)
                )
                a = finite & (cohort & np.isfinite(y))[None, :]
                cases = []
                if stage == "repair":
                    cases = [("repair", a, np.flatnonzero(keep), "all", False)]
                else:
                    cc = a & I.complete_covariates(meta)[None, :]
                    same = np.all(a == cc, axis=1)
                    cases = [
                        ("A", a, np.flatnonzero(keep & ~same), "all", False),
                        ("BC", cc, np.flatnonzero(keep), "all", False),
                    ]
                    if info["dataset"] == "brain":
                        for stratum in ("ROS", "MAP"):
                            mask = (
                                cc
                                & meta.study.eq(stratum)
                                .fillna(False)
                                .to_numpy()[None, :]
                            )
                            cases.append(
                                ("C", mask, np.flatnonzero(keep), stratum, False)
                            )
                    cases.append(
                        (
                            "C_age_spline",
                            cc,
                            np.flatnonzero(keep & rows.focused.to_numpy()),
                            "all",
                            True,
                        )
                    )
                    if outcome == "cogng_demog_slope" and pathology_verified:
                        pc = (
                            a
                            & I.complete_covariates(
                                meta, (*I.COVARIATES, "gpath", "amyloid", "tangles")
                            )[None, :]
                        )
                        cases.append(
                            (
                                "C_AD_pathology",
                                pc,
                                np.flatnonzero(keep & rows.focused.to_numpy()),
                                "all",
                                False,
                            )
                        )
                for model, mask, row_ids, stratum, spline in cases:
                    for focus in [False] if stage == "repair" else [False, True]:
                        selected = (
                            row_ids
                            if stage == "repair"
                            else row_ids[rows.focused.to_numpy()[row_ids] == focus]
                        )
                        if not len(selected):
                            continue
                        for vectors, donors in _patterns(mask[selected], selected):
                            covariates = (
                                (*I.COVARIATES, "gpath", "amyloid", "tangles")
                                if model == "C_AD_pathology"
                                else I.COVARIATES
                            )
                            task = dict(
                                root=str(root),
                                analysis_id=spec["analysis_id"],
                                stage=stage,
                                variant=variant,
                                scope=scope,
                                outcome=outcome,
                                model=model,
                                stratum=stratum,
                                spline=spline,
                                rows=vectors,
                                donors=donors,
                                focus=focus,
                                covariates=list(covariates),
                                draws=spec["permutations"]["draws"],
                                bootstrap_draws=(
                                    0
                                    if stage == "repair"
                                    else spec["bootstrap"][
                                        "focused_draws" if focus else "catalog_draws"
                                    ]
                                ),
                            )
                            task["task_id"] = canonical_hash(task)
                            tasks.append(task)
    return tasks


def execute_task(task):
    start = time.perf_counter()
    root = Path(task["root"])
    folder = root / ("repair_shards" if task["stage"] == "repair" else "scan_shards")
    folder.mkdir(exist_ok=True)
    path = folder / (task["task_id"] + ".parquet")
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if old["analysis_id"] != task["analysis_id"] or old["sha256"] != sha(path):
            raise ValueError("Immutable result shard changed")
        return dict(status="cached", rows=old["rows"], seconds=0)
    threadpool_limits(
        limits=int(os.environ.get("CALIBRATED_VALIDATION_BLAS_THREADS", "2"))
    )
    matrix, metadata, specs = prepared(task["root"], task["variant"])
    donors = task["donors"]
    row_ids = task["rows"]
    meta = metadata.iloc[donors].reset_index(drop=True)
    x = np.array(matrix[np.ix_(row_ids, donors)], copy=True)
    y = I.finite_numeric(meta[task["outcome"]])
    donor_hash = I.membership_hash(meta.sample_id)
    seed = I.stable_seed(
        611104,
        task["variant"],
        task["scope"],
        task["outcome"],
        donor_hash,
        "adjusted" if task["model"] not in ("repair", "A") else "unadjusted",
        task["stratum"],
        (
            task["model"]
            if task["spline"] or task["model"] == "C_AD_pathology"
            else "base"
        ),
    )
    bootseed = I.stable_seed(
        611105,
        task["variant"],
        task["scope"],
        task["outcome"],
        donor_hash,
        (
            task["model"]
            if task["spline"] or task["model"] == "C_AD_pathology"
            else "base"
        ),
    )
    models = (
        [("B", False), ("C", True)]
        if task["model"] == "BC"
        else [(task["model"], task["model"] not in ("A", "repair"))]
    )
    bootstrap = (
        I.bootstrap_group(
            x,
            y,
            meta,
            draws=task["bootstrap_draws"],
            seed=bootseed,
            covariates=task["covariates"],
            adjusted=task["model"] != "A",
            spline_age=task["spline"],
        )
        if task["bootstrap_draws"]
        else None
    )
    records = []
    composition = {
        k: meta[k].fillna("Missing").astype(str).value_counts().to_dict()
        for k in ("study", "sex_code", "diagnosis_group")
        if k in meta
    }
    base_rows = specs.iloc[row_ids].to_dict("records")
    scope_mask = cohort_mask(metadata, task["scope"])
    if task["stratum"] != "all":
        scope_mask &= metadata.study.eq(task["stratum"]).fillna(False).to_numpy()
    all_finite = np.isfinite(matrix[row_ids])
    full_y = I.finite_numeric(metadata[task["outcome"]])
    score_masks = all_finite & scope_mask[None, :]
    pairs = score_masks & np.isfinite(full_y)[None, :]
    cov = I.complete_covariates(metadata, task["covariates"])
    score_counts = score_masks.sum(axis=1)
    pair_counts = pairs.sum(axis=1)
    cc_counts = (pairs & cov[None, :]).sum(axis=1)
    tested_mask = np.zeros(len(metadata), bool)
    tested_mask[donors] = True
    identical_A = np.all(pairs == tested_mask[None, :], axis=1)
    for model, adjusted in models:
        actual_seed = (
            seed
            if adjusted
            else I.stable_seed(
                611104,
                task["variant"],
                task["scope"],
                task["outcome"],
                donor_hash,
                "unadjusted",
                "all",
                "base",
            )
        )
        answer = I.infer_group(
            x,
            y,
            meta,
            adjusted=adjusted,
            draws=task["draws"],
            seed=actual_seed,
            blocks=(
                meta.study.to_numpy()
                if adjusted and task["variant"] == "brain"
                else None
            ),
            covariates=task["covariates"],
            spline_age=task["spline"],
        )
        design = answer["design"]
        for i, row_id in enumerate(row_ids):
            row = base_rows[i]
            label = "adjusted" if adjusted else "unadjusted"
            result = dict(
                row,
                analysis_id=task["analysis_id"],
                cohort=task["scope"],
                outcome=task["outcome"],
                model=model,
                study_stratum=task["stratum"],
                mask="all",
                transform="native",
                n=len(donors),
                effect=answer["effect"][i],
                partial_rank_rho=answer["effect"][i] if adjusted else np.nan,
                p=answer["p"][i],
                p_method=answer["method"][i],
                permutation_draws=int(answer["draws"][i]),
                exceedances=int(answer["exceedances"][i]),
                permutation_seed=actual_seed,
                permutation_degenerate=answer["degenerate_permutations"],
                donor_set_hash=donor_hash,
                candidate_n=int(scope_mask.sum()),
                score_available_n=int(score_counts[i]),
                score_outcome_n=int(pair_counts[i]),
                complete_case_n=int(cc_counts[i]),
                missing_score_n=int(scope_mask.sum() - score_counts[i]),
                missing_outcome_after_score_n=int(score_counts[i] - pair_counts[i]),
                missing_covariates_after_pair_n=int(pair_counts[i] - cc_counts[i]),
                composition=json.dumps(composition, sort_keys=True),
                design_rank=design.rank,
                design_columns=json.dumps(design.names),
                removed_constant_terms=json.dumps(design.removed),
                age_spline_knots=json.dumps(design.age_knots),
                residual_df=answer["residual_df"],
                unavailable_reason=answer["reason"][i],
                primary_eligible=len(donors) >= 30 and not bool(answer["reason"][i]),
                rank_coefficient=answer["rank_beta"][i],
                hc3_standard_error=answer["hc3_se"][i],
                hc3_p_sensitivity=answer["hc3_p"][i],
                ROS_MAP_residual_variance_ratio=answer["residual_variance_ratio"][i],
                interval_method="stratified_donor_percentile",
                bootstrap_draws=task["bootstrap_draws"],
                bootstrap_seed=bootseed,
                ci_low=bootstrap[label + "_low"][i] if bootstrap else np.nan,
                ci_high=bootstrap[label + "_high"][i] if bootstrap else np.nan,
                bootstrap_valid=int(bootstrap[label + "_valid"][i]) if bootstrap else 0,
                supplied_adjusted_slope=task["outcome"]
                in (
                    "cogng_demog_slope",
                    "cogng_path_slope",
                    "motor10_demog_slope",
                    "sqrt_parksc_demog_slope",
                ),
            )
            records.append(result)
            if model == "B" and identical_A[i]:
                records.append(dict(result, model="A"))
    frame = pd.DataFrame(records)
    atomic_parquet(path, frame)
    import resource

    record = dict(
        status="complete",
        analysis_id=task["analysis_id"],
        task_id=task["task_id"],
        sha256=sha(path),
        rows=len(frame),
        seconds=time.perf_counter() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    atomic_json(receipt, record)
    return record


def run_tasks(root, spec, stage="scan", workers=4):
    start = time.perf_counter()
    tasks = plan_tasks(root, spec, stage)
    counts = []
    public_plan = [
        {k: v for k, v in t.items() if k not in ("donors", "rows", "root")}
        | dict(score_vectors=len(t["rows"]), donors_n=len(t["donors"]))
        for t in tasks
    ]
    atomic_json(
        Path(root) / (stage + "_plan.json"),
        dict(analysis_id=spec["analysis_id"], tasks=public_plan, task_count=len(tasks)),
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_task, t): t["task_id"] for t in tasks}
        for future in as_completed(futures):
            try:
                counts.append(future.result())
            except Exception as error:
                atomic_json(
                    Path(root) / (stage + "_failed.json"),
                    dict(task_id=futures[future], error=repr(error)),
                )
                raise
            if len(counts) % 25 == 0 or len(counts) == len(tasks):
                print(
                    json.dumps(
                        dict(
                            stage=stage,
                            completed=len(counts),
                            planned=len(tasks),
                            rows=sum(c["rows"] for c in counts),
                            seconds=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
    atomic_json(
        Path(root) / (stage + "_execution.json"),
        dict(
            status="complete",
            analysis_id=spec["analysis_id"],
            planned_tasks=len(tasks),
            completed_tasks=len(counts),
            failed_tasks=0,
            rows=sum(c["rows"] for c in counts),
            seconds=time.perf_counter() - start,
            workers=workers,
            code_sha256={
                str(p.name): sha(p) for p in [Path(__file__), Path(I.__file__)]
            },
        ),
    )


def new_correction(frame):
    out = frame.copy()
    out["q_scan"] = np.nan
    out["q_GL_outcome"] = np.nan
    context = [
        "dataset",
        "cohort",
        "allocation_id",
        "model",
        "study_stratum",
        "mask",
        "transform",
    ]
    families = []
    for key, ids in out.groupby(context, dropna=False, sort=False).groups.items():
        out.loc[ids, "q_scan"] = bh(out.loc[ids, "p"])
        n = int(out.loc[ids, "p"].notna().sum())
        minimum = 1 / 10000
        families.append(
            dict(zip(context, key))
            | dict(
                planned=len(ids),
                estimable=n,
                excluded=len(ids) - n,
                minimum_MC_p=minimum,
                isolated_BH_threshold=0.05 / n if n else np.nan,
                resolution_limited=n > 500,
            )
        )
    gl = out.loc[out.metric_family.isin(("gain", "loss"))]
    for _, ids in gl.groupby(
        context + ["beta", "outcome"], dropna=False, sort=False
    ).groups.items():
        out.loc[ids, "q_GL_outcome"] = bh(out.loc[ids, "p"])
    out["monte_carlo_floor"] = (
        out.p_method.str.contains("monte_carlo|freedman_lane")
        & out.exceedances.eq(0)
        & out.p.notna()
    )
    return out, pd.DataFrame(families)


def assemble_scan(root, spec):
    root = Path(root)
    files = sorted((root / "scan_shards").glob("*.parquet"))
    dest = root / "results"
    dest.mkdir(exist_ok=True)
    frames = [pd.read_parquet(p) for p in files]
    totals = []
    families = []
    for dataset in ("brain", "motor"):
        for cohort in COHORTS:
            pieces = [
                f.loc[f.dataset.eq(dataset) & f.cohort.eq(cohort)]
                for f in frames
                if not f.empty
            ]
            frame = pd.concat(pieces, ignore_index=True)
            frame, fdr = new_correction(frame)
            key = ["input_variant", "row_id", "outcome", "study_stratum"]
            b = frame.loc[frame.model.eq("B"), key + ["donor_set_hash"]]
            c = frame.loc[
                frame.model.eq("C") & frame.study_stratum.eq("all"),
                key + ["donor_set_hash"],
            ]
            matched = b.merge(c, on=key, validate="one_to_one", suffixes=("_B", "_C"))
            assert len(matched) == len(b) == len(c) and matched.donor_set_hash_B.equals(
                matched.donor_set_hash_C
            )
            path = dest / f"{dataset}_{cohort}.parquet"
            atomic_parquet(path, frame)
            totals.append(
                dict(
                    dataset=dataset,
                    cohort=cohort,
                    planned=len(frame),
                    completed=len(frame),
                    estimable=int(frame.p.notna().sum()),
                    excluded=int(frame.p.isna().sum()),
                    matched_B_C_pairs=len(matched),
                    families=sorted(frame.family.unique()),
                    sha256=sha(path),
                    file=path.name,
                )
            )
            families.append(fdr)
    atomic_parquet(
        dest / "fdr_families.parquet", pd.concat(families, ignore_index=True)
    )
    atomic_json(
        root / "scan_complete.json",
        dict(
            status="complete",
            analysis_id=spec["analysis_id"],
            contexts=totals,
            B_C_identical_membership=True,
        ),
    )
    return totals


def assemble_repair(root, spec):
    root = Path(root)
    dest = root / "repaired_baseline"
    dest.mkdir(exist_ok=True)
    numeric = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((root / "repair_shards").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    keys = ["module_key", "component", "beta", "metric_family", "outcome"]
    reports = []
    changes = []
    for path in sorted((RESEARCH_ROOT / "associations").glob("*_*.parquet")):
        if path.name == "planned_cells.parquet":
            continue
        original = pd.read_parquet(path)
        scope = original.cohort_scope.iloc[0]
        dataset = original.dataset.iloc[0]
        new = numeric.loc[
            numeric.cohort.eq(scope) & numeric.dataset.eq(dataset),
            keys
            + [
                "p",
                "p_method",
                "permutation_draws",
                "exceedances",
                "permutation_seed",
                "donor_set_hash",
                "n",
            ],
        ]
        merged = original.merge(
            new,
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("_old", "_repair"),
        )
        use = merged.method.eq("Spearman")
        assert merged.loc[use, "p_method"].notna().all()
        assert np.array_equal(merged.loc[use, "n_old"], merged.loc[use, "n_repair"])
        merged["p"] = merged.p_old.where(~use, merged.p_repair)
        merged["n"] = merged.n_old
        merged = merged.drop(columns=["p_repair", "n_repair", "n_old"])
        merged["old_q_scan"] = merged.q_scan
        merged["old_q_GL_outcome"] = merged.q_GL_outcome
        merged = correct(merged)
        merged["inference_version"] = "tie_aware_permutation_v1"
        merged["validation_analysis_id"] = spec["analysis_id"]
        atomic_parquet(dest / path.name, merged)
        change = merged[
            keys
            + [
                "cohort_scope",
                "dataset",
                "n",
                "effect",
                "p_old",
                "p",
                "old_q_scan",
                "q_scan",
                "old_q_GL_outcome",
                "q_GL_outcome",
                "p_method",
            ]
        ].copy()
        change["p_changed"] = ~np.isclose(
            change.p_old, change.p, rtol=1e-12, atol=0, equal_nan=True
        )
        changes.append(change)
        bad = merged.n.eq(4) & merged.effect.abs().ge(1 - 1e-12) & merged.p_old.eq(0)
        assert np.allclose(merged.loc[bad, "p"], 1 / 12)
        reports.append(
            dict(
                file=path.name,
                rows=len(merged),
                numeric_repaired=int(use.sum()),
                old_zero_p=int(merged.p_old.eq(0).sum()),
                new_zero_p=int(merged.p.eq(0).sum()),
                n4_perfect_repaired=int(bad.sum()),
                old_q_scan_lt05=int(merged.old_q_scan.lt(0.05).sum()),
                new_q_scan_lt05=int(merged.q_scan.lt(0.05).sum()),
                sha256=sha(dest / path.name),
            )
        )
    atomic_parquet(dest / "before_after.parquet", pd.concat(changes, ignore_index=True))
    atomic_json(
        root / "repair_complete.json",
        dict(
            status="complete",
            analysis_id=spec["analysis_id"],
            contexts=reports,
            old_release_untouched=True,
            aliases_preserved_in_legacy_FDR=True,
        ),
    )
    return reports
