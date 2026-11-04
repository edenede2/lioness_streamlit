"""Prespecified explanatory comparisons for the four user-named candidates."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json, time, math
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from . import validation_inference as I
from .validation_spec import FOCUS, FOCUS_COMPONENTS, OUTCOMES, METRICS, COHORTS
from .validation_run import prepared, cohort_mask
from .analysis import bh
from .inputs import atomic_json, atomic_parquet, sha
from .core import canonical_hash


def hc3_fit(y, design, interest):
    y, z = np.asarray(y, float), np.asarray(design, float)
    n, k = z.shape
    rank = np.linalg.matrix_rank(z) if n else 0
    result = dict(
        n=n,
        design_rank=int(rank),
        residual_df=n - rank,
        condition_number=float(np.linalg.cond(z)) if n else np.nan,
        reason="",
        coefficient=np.full(len(interest), np.nan),
        standard_error=np.full(len(interest), np.nan),
        ci_low=np.full(len(interest), np.nan),
        ci_high=np.full(len(interest), np.nan),
        p_hc3=np.nan,
        F_hc3=np.nan,
        error=np.full(n, np.nan),
        r_squared=np.nan,
    )
    if rank < k:
        result["reason"] = "rank_deficient_full_model"
        return result
    if n < 30 or n - rank < 20:
        result["reason"] = "insufficient_n_or_residual_df"
        return result
    if result["condition_number"] > 1e8:
        result["reason"] = "unstable_design_condition_number"
        return result
    if np.ptp(y) == 0:
        result["reason"] = "constant_outcome"
        return result
    inverse = np.linalg.inv(z.T @ z)
    beta = inverse @ z.T @ y
    error = y - z @ beta
    h = np.sum((z @ inverse) * z, axis=1)
    if np.any(h >= 1 - 1e-12):
        result["reason"] = "unit_leverage_HC3_unavailable"
        return result
    covariance = inverse @ (z.T @ ((error / (1 - h))[:, None] ** 2 * z)) @ inverse
    interest = np.asarray(interest)
    coef = beta[interest]
    cov = covariance[np.ix_(interest, interest)]
    se = np.sqrt(np.maximum(0, np.diag(cov)))
    if np.linalg.matrix_rank(cov) < len(interest):
        result["reason"] = "singular_HC3_contrast_covariance"
        return result
    statistic = float(coef @ np.linalg.solve(cov, coef) / len(interest))
    p = stats.f.sf(statistic, len(interest), n - k)
    t = stats.t.ppf(0.975, n - k)
    result.update(
        coefficient=coef,
        standard_error=se,
        ci_low=coef - t * se,
        ci_high=coef + t * se,
        p_hc3=float(p),
        F_hc3=statistic,
        error=error,
        r_squared=float(1 - (error @ error) / np.sum((y - y.mean()) ** 2)),
    )
    return result


def linear_permutation(y, design, interest, *, draws=9999, seed=611104, blocks=None):
    """Freedman--Lane partial F with fixed supplied design and outcome ranks."""
    y, z = np.asarray(y, float), np.asarray(design, float)
    n, k = z.shape
    interest = np.asarray(interest)
    reduced = np.delete(z, interest, axis=1)
    q = np.linalg.qr(reduced, mode="reduced")[0]
    if np.linalg.matrix_rank(z) < k or n <= k:
        return dict(p=np.nan, exceedances=0, draws=0, F=np.nan)
    target = z[:, interest] - q @ (q.T @ z[:, interest])
    qi = np.linalg.qr(target, mode="reduced")[0]
    residual = y - q @ (q.T @ y)
    norm = residual @ residual
    d = len(interest)
    df = n - k

    def statistic(values):
        ss = np.sum((values @ qi) ** 2, axis=1)
        reduced_rss = norm - np.sum((values @ q) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (ss / d) / ((reduced_rss - ss) / df)

    observed = statistic(residual[None, :])[0]
    count = 0
    for ids in I.permutation_indices(n, draws, seed, blocks):
        value = statistic(residual[ids])
        count += int(np.sum((value >= observed - 1e-12) | ~np.isfinite(value)))
    return dict(
        p=(count + 1) / (draws + 1), exceedances=count, draws=draws, F=float(observed)
    )


def standardized_ranks(value):
    a = stats.rankdata(np.asarray(value, float), method="average", axis=0)
    sd = np.std(a, axis=0)
    if np.any(sd <= 1e-12):
        raise ValueError("constant_score_or_comparator")
    return (a - a.mean(axis=0)) / sd


def interaction(x, y, meta, seed, draws):
    counts = meta.study.value_counts().to_dict()
    base = I.nuisance_design(meta)
    result = dict(
        n=len(y),
        ROS_n=int(counts.get("ROS", 0)),
        MAP_n=int(counts.get("MAP", 0)),
        effect=np.nan,
        p=np.nan,
        p_hc3=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        unavailable_reason="",
        design_rank=base.rank,
        residual_df=0,
        heteroscedasticity_levene_p=np.nan,
        condition_number=np.nan,
    )
    if any(counts.get(s, 0) < 30 for s in ("ROS", "MAP")):
        result["unavailable_reason"] = "each_study_requires_at_least_30_complete_cases"
        return result
    if base.reason:
        result["unavailable_reason"] = base.reason
        return result
    try:
        rx = standardized_ranks(x)
        ry = standardized_ranks(y)
    except ValueError as e:
        result["unavailable_reason"] = str(e)
        return result
    study = meta.study.eq("ROS").to_numpy(float)
    z = np.column_stack([base.matrix, rx, rx * study])
    fit = hc3_fit(ry, z, [z.shape[1] - 1])
    result.update(
        design_rank=fit["design_rank"],
        residual_df=fit["residual_df"],
        condition_number=fit["condition_number"],
        unavailable_reason=fit["reason"],
    )
    if fit["reason"]:
        return result
    perm = linear_permutation(
        ry, z, [z.shape[1] - 1], draws=draws, seed=seed, blocks=meta.study.to_numpy()
    )
    result.update(
        effect=float(fit["coefficient"][0]),
        p=perm["p"],
        p_hc3=fit["p_hc3"],
        ci_low=float(fit["ci_low"][0]),
        ci_high=float(fit["ci_high"][0]),
        permutation_draws=perm["draws"],
        exceedances=perm["exceedances"],
        standard_error=float(fit["standard_error"][0]),
        heteroscedasticity_levene_p=float(
            stats.levene(
                fit["error"][study == 0], fit["error"][study == 1], center="median"
            ).pvalue
        ),
    )
    return result


@lru_cache(maxsize=16)
def comparators(family, module, component):
    from app_helpers import data as D
    from .adapter import component_legacy

    legacy_component = component_legacy(component)
    columns = {}
    provenance = {}
    unavailable = {}
    try:
        frame = D.load_resolved_scope(
            "control_anchored",
            module=module,
            metric_family="connectivity",
            component=legacy_component,
            module_set=family,
            estimator="lioness",
            edge_rule="all",
            differential_edge_rule="all",
            score_normalization="standard_pruned",
            metric_scale="raw",
            include_embedded_metadata=False,
        )
        if frame.sample_id.duplicated().any():
            raise ValueError("Nonunique comparator donor")
        columns["legacy_connectivity"] = frame.set_index("sample_id").metric_raw
        provenance["legacy_connectivity"] = (
            "Existing same-family/module native unmasked LIONESS control-anchored Connectivity; historical TS/CT powers and normalization, original Control reference exposure; not disjoint B R/C"
        )
    except (OSError, ValueError, KeyError) as e:
        unavailable["legacy_connectivity"] = str(e)
    tissues = (
        component.removeprefix("TS_").split("__")
        if component.startswith("TS_")
        else component.removeprefix("CT_").split("__")
    )
    for tissue in tissues:
        name = "eigengene_" + tissue
        try:
            frame = D.load_descriptive_eigengene_scope(
                method="control_anchored",
                module=module,
                component=component_legacy("TS_" + tissue),
                module_set=family,
                metric_scale="raw",
                include_embedded_metadata=False,
            )
            if frame.empty:
                raise ValueError("Same-module regional eigengene unavailable")
            if frame.sample_id.duplicated().any():
                raise ValueError("Nonunique eigengene donor")
            columns[name] = frame.set_index("sample_id").metric_raw
            provenance[name] = (
                "Existing tissue-specific PC1 of this same multi-tissue module; cohort-median imputation/gene standardization; fit on all450 original donors; sign aligned to mean member expression; historical target/discovery overlap"
            )
        except (OSError, ValueError, KeyError) as e:
            unavailable[name] = str(e)
    required = ["legacy_connectivity", *["eigengene_" + t for t in tissues]]
    return columns, provenance, unavailable, required


def incremental(x, y, meta, comparator_count, draws, seed, bootstrap_draws, bootseed):
    base = I.nuisance_design(meta)
    n = len(y)
    result = dict(
        n=n,
        effect=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        p=np.nan,
        p_hc3=np.nan,
        unavailable_reason="",
        design_rank=base.rank,
        residual_df=0,
        bootstrap_draws=bootstrap_draws,
        bootstrap_valid=0,
        condition_number=np.nan,
        G_L_rank_correlation=np.nan,
        VIF_G=np.nan,
        VIF_L=np.nan,
    )
    if base.reason:
        result["unavailable_reason"] = base.reason
        return result
    try:
        ranks = standardized_ranks(x.T)
        ry = standardized_ranks(y)
    except ValueError as e:
        result["unavailable_reason"] = str(e)
        return result
    # x is G,L,then each required comparator; S/A never enter this model.
    reduced = np.column_stack([base.matrix, ranks[:, 2:]])
    z = np.column_stack([reduced, ranks[:, :2]])
    interest = [z.shape[1] - 2, z.shape[1] - 1]
    fit = hc3_fit(ry, z, interest)
    result.update(
        design_rank=fit["design_rank"],
        residual_df=fit["residual_df"],
        condition_number=fit["condition_number"],
        unavailable_reason=fit["reason"],
        G_L_rank_correlation=float(np.corrcoef(ranks[:, :2].T)[0, 1]),
    )
    if fit["reason"]:
        return result
    br = np.linalg.lstsq(reduced, ry, rcond=None)[0]
    er = ry - reduced @ br
    total = np.sum((ry - ry.mean()) ** 2)
    delta = float((er @ er - fit["error"] @ fit["error"]) / total)
    inverse = np.linalg.inv(z.T @ z)
    result.update(
        effect=delta,
        reduced_R_squared=float(1 - er @ er / total),
        full_R_squared=fit["r_squared"],
        p_hc3=fit["p_hc3"],
        VIF_G=float(inverse[-2, -2] * n),
        VIF_L=float(inverse[-1, -1] * n),
        collinearity_flag=bool(
            max(inverse[-2, -2] * n, inverse[-1, -1] * n) > 10
            or abs(result["G_L_rank_correlation"]) > 0.95
        ),
    )
    perm = linear_permutation(
        ry, z, interest, draws=draws, seed=seed, blocks=meta.study.to_numpy()
    )
    result.update(
        p=perm["p"], permutation_draws=perm["draws"], exceedances=perm["exceedances"]
    )
    strata = [
        np.flatnonzero(meta.study.to_numpy() == v) for v in sorted(meta.study.unique())
    ]
    rng = np.random.default_rng(bootseed)
    values = []
    for b in range(bootstrap_draws):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
        d = I.nuisance_design(meta.iloc[ids])
        if d.reason or d.names != base.names:
            continue
        try:
            r = standardized_ranks(x[:, ids].T)
            yy = standardized_ranks(y[ids])
        except ValueError:
            continue
        zr = np.column_stack([d.matrix, r[:, 2:]])
        zf = np.column_stack([zr, r[:, :2]])
        if np.linalg.matrix_rank(zf) != zf.shape[1] or np.linalg.cond(zf) > 1e8:
            continue
        e1 = yy - zr @ np.linalg.lstsq(zr, yy, rcond=None)[0]
        e2 = yy - zf @ np.linalg.lstsq(zf, yy, rcond=None)[0]
        values.append(float((e1 @ e1 - e2 @ e2) / np.sum((yy - yy.mean()) ** 2)))
    result["bootstrap_valid"] = len(values)
    if len(values) >= math.ceil(0.8 * bootstrap_draws):
        result["ci_low"], result["ci_high"] = map(
            float, np.quantile(values, [0.025, 0.975])
        )
    return result


def focus_task(task):
    threadpool_limits(limits=1)
    root = Path(task["root"])
    folder = root / "focused_shards"
    folder.mkdir(exist_ok=True)
    task_id = canonical_hash(task)
    receipt = folder / (task_id + ".json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if all(sha(folder / r["file"]) == r["sha256"] for r in old["files"]):
            return old
        raise ValueError("Focused immutable checkpoint changed")
    matrix, meta, rows = prepared(task["root"], "brain")
    chosen = rows.loc[
        rows.family.eq(task["family"])
        & rows.module.eq(task["module"])
        & rows.component.eq(task["component"])
        & rows.beta.eq(task["beta"])
    ]
    if set(chosen.metric_family) != set(METRICS):
        raise ValueError("Focused physical component missing from verified catalog")
    vectors = np.array(
        matrix[chosen.set_index("metric_family").loc[list(METRICS)].row_id], copy=True
    )
    y = I.finite_numeric(meta[task["outcome"]])
    scope = cohort_mask(meta, task["cohort"])
    cov = I.complete_covariates(meta)
    interactions = []
    base = {k: v for k, v in task.items() if k != "root"}
    base.update(
        analysis_id=task["analysis_id"],
        mask="all",
        transform="native",
        model="pooled_rank_score_by_study",
    )
    for i, metric in enumerate(METRICS):
        use = scope & cov & np.isfinite(y) & np.isfinite(vectors[i])
        f = meta.loc[use].reset_index(drop=True)
        donor_hash = I.membership_hash(f.sample_id)
        seed = I.stable_seed(
            611104,
            "interaction",
            task["family"],
            task["module"],
            task["component"],
            task["outcome"],
            task["cohort"],
            metric,
            task["beta"],
            donor_hash,
        )
        result = interaction(vectors[i, use], y[use], f, seed, task["draws"])
        interactions.append(
            dict(
                base,
                **result,
                metric_family=metric,
                donor_set_hash=donor_hash,
                permutation_seed=seed,
                interval_method="HC3_t_interval_on_common_pooled_standardized_rank_scale",
                effect_name="ROS-minus-MAP rank-score slope difference"
            )
        )
    columns, provenance, unavailable, required = comparators(
        task["family"], task["module"], task["component"]
    )
    cframe = pd.DataFrame(
        {
            name: value.reindex(meta.sample_id).to_numpy(float)
            for name, value in columns.items()
        }
    )
    use = scope & cov & np.isfinite(y) & np.isfinite(vectors[:2]).all(axis=0)
    missing = [name for name in required if name not in columns]
    for name in required:
        if name in cframe:
            use &= np.isfinite(cframe[name].to_numpy())
    if missing:
        use[:] = False
    f = meta.loc[use].reset_index(drop=True)
    donor_hash = I.membership_hash(f.sample_id)
    seed = I.stable_seed(
        611104,
        "comparison",
        task["family"],
        task["module"],
        task["outcome"],
        task["cohort"],
        donor_hash,
    )
    bootseed = I.stable_seed(
        611105,
        "comparison",
        task["family"],
        task["module"],
        task["outcome"],
        task["cohort"],
        donor_hash,
    )
    names = ["gain", "loss", *required]
    xx = np.stack(
        [
            vectors[0],
            vectors[1],
            *[
                (
                    cframe[name].to_numpy()
                    if name in cframe
                    else np.full(len(meta), np.nan)
                )
                for name in required
            ],
        ]
    )[:, use]
    partial = I.infer_group(
        xx,
        y[use],
        f,
        adjusted=True,
        draws=task["draws"],
        seed=seed,
        blocks=f.study.to_numpy(),
    )
    boot = I.bootstrap_group(
        xx, y[use], f, draws=task["bootstrap_draws"], seed=bootseed
    )
    comparisons = []
    for i, name in enumerate(names):
        # Same eigengene on the same donors is one hypothesis even if shown
        # in more than one pair-component/beta comparison panel.
        hypothesis = dict(
            family=task["family"],
            module=task["module"],
            cohort=task["cohort"],
            outcome=task["outcome"],
            series=name,
            component=task["component"] if not name.startswith("eigengene_") else name,
            beta=task["beta"] if name in ("gain", "loss") else None,
            donors=donor_hash,
        )
        comparisons.append(
            dict(
                base,
                model="shared_donor_partial_comparison",
                series=name,
                metric_family=name,
                n=len(f),
                effect=partial["effect"][i],
                ci_low=boot["adjusted_low"][i],
                ci_high=boot["adjusted_high"][i],
                p=partial["p"][i],
                p_method=partial["method"][i],
                bootstrap_draws=task["bootstrap_draws"],
                bootstrap_valid=int(boot["adjusted_valid"][i]),
                donor_set_hash=donor_hash,
                unavailable_reason=(
                    "required_comparator_missing:" + ",".join(missing)
                    if missing
                    else (
                        "no_shared_comparator_donors"
                        if not len(f)
                        else partial["reason"][i]
                    )
                ),
                comparator_provenance=provenance.get(
                    name, "Frozen B calibrated " + name
                ),
                hypothesis_id=canonical_hash(hypothesis),
                permutation_seed=seed,
                interval_method="stratified_donor_percentile",
                design_rank=partial["design"].rank,
                residual_df=partial["residual_df"],
            )
        )
    jointbase = dict(
        base,
        model="joint_G_L_beyond_legacy_and_constituent_eigengenes",
        metric_family="gain_and_loss_joint",
        donor_set_hash=donor_hash,
        comparator_columns=json.dumps(required),
        permutation_seed=seed,
        interval_method="stratified_donor_percentile_delta_R2",
        effect_name="incremental in-sample rank-outcome R_squared",
    )
    joint = incremental(
        xx,
        y[use],
        f,
        len(required),
        task["draws"],
        seed,
        task["bootstrap_draws"],
        bootseed,
    )
    if missing:
        joint["unavailable_reason"] = "required_comparator_missing:" + ",".join(missing)
    elif not len(f):
        joint["unavailable_reason"] = "no_shared_comparator_donors"
    files = []
    for name, frame in [
        ("interaction", pd.DataFrame(interactions)),
        ("comparators", pd.DataFrame(comparisons)),
        ("incremental", pd.DataFrame([dict(jointbase, **joint)])),
    ]:
        path = folder / (task_id + "_" + name + ".parquet")
        atomic_parquet(path, frame)
        files.append(dict(file=path.name, sha256=sha(path), rows=len(frame)))
    record = dict(
        status="complete", analysis_id=task["analysis_id"], task_id=task_id, files=files
    )
    atomic_json(receipt, record)
    return record


def run_focused(root, spec, workers=4):
    root = Path(root)
    tasks = []
    start = time.perf_counter()
    for family, modules in FOCUS.items():
        for module in modules:
            for component in FOCUS_COMPONENTS:
                for beta in (3, 1):
                    for cohort in COHORTS:
                        for outcome in OUTCOMES:
                            tasks.append(
                                dict(
                                    root=str(root),
                                    analysis_id=spec["analysis_id"],
                                    family=family,
                                    module=module,
                                    component=component,
                                    beta=beta,
                                    cohort=cohort,
                                    outcome=outcome,
                                    draws=9999,
                                    bootstrap_draws=1999,
                                )
                            )
    atomic_json(
        root / "focused_plan.json",
        dict(
            analysis_id=spec["analysis_id"],
            planned_contexts=len(tasks),
            components=FOCUS_COMPONENTS,
            comparator_set="Same-module legacy Connectivity AND every constituent same-module tissue eigengene; no optional substitution",
            interaction_primary="Study-restricted Freedman-Lane partial F; HC3 t/F sensitivity and intervals separately labeled",
        ),
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, f in enumerate(
            as_completed([pool.submit(focus_task, t) for t in tasks])
        ):
            f.result()
            if (i + 1) % 16 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="focused",
                            completed=i + 1,
                            planned=len(tasks),
                            seconds=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
    dest = root / "results"
    dest.mkdir(exist_ok=True)
    reports = []
    for name in ("interaction", "comparators", "incremental"):
        frame = pd.concat(
            [
                pd.read_parquet(p)
                for p in sorted(
                    (root / "focused_shards").glob("*_" + name + ".parquet")
                )
            ],
            ignore_index=True,
        )
        frame["q_focused"] = np.nan
        frame["q_HC3_focused"] = np.nan
        for cohort, ids in frame.groupby("cohort", sort=False).groups.items():
            if name == "comparators":
                unique = frame.loc[ids].drop_duplicates("hypothesis_id")
                unique = unique.assign(_q=bh(unique.p))
                frame.loc[ids, "q_focused"] = frame.loc[ids, "hypothesis_id"].map(
                    unique.set_index("hypothesis_id")._q
                )
            else:
                frame.loc[ids, "q_focused"] = bh(frame.loc[ids, "p"])
            if "p_hc3" in frame:
                frame.loc[ids, "q_HC3_focused"] = bh(frame.loc[ids, "p_hc3"])
        atomic_parquet(dest / (name + ".parquet"), frame)
        reports.append(
            dict(
                analysis=name,
                planned=len(frame),
                completed=len(frame),
                estimable=int(frame.p.notna().sum()),
                excluded=int(frame.p.isna().sum()),
                reasons=frame.loc[frame.p.isna(), "unavailable_reason"]
                .value_counts()
                .to_dict(),
                file=name + ".parquet",
                sha256=sha(dest / (name + ".parquet")),
            )
        )
    atomic_json(
        root / "focused_complete.json",
        dict(
            status="complete",
            analysis_id=spec["analysis_id"],
            planned_contexts=len(tasks),
            completed_contexts=len(tasks),
            seconds=time.perf_counter() - start,
            analyses=reports,
        ),
    )
    return reports
