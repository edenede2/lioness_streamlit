"""Frozen, full-catalog descriptive testing and explicit correction families.

No outcome participates in calibration, masks, score defaults or transforms.
The testing manifest is persisted before loading any score values.
"""

from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd
from scipy import stats
from .inputs import atomic_json, atomic_parquet, sha
from .core import canonical_hash
from .registry import ROOT, PRIMARY

COHORTS = ("complete_450", "incomplete_region_validation", "maximum_component")
BETAS = (3, 1)


def bh(values):
    p = np.asarray(values, float)
    q = np.full(p.shape, np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    if not len(ix):
        return q
    order = ix[np.argsort(p[ix], kind="stable")]
    rank = np.arange(1, len(order) + 1)
    q[order] = np.minimum(
        1, np.minimum.accumulate((p[order] * len(order) / rank)[::-1])[::-1]
    )
    return q


def correct(frame):
    """Correct the complete executed family; call before any display filtering."""
    out = frame.copy()
    out["q_scan"] = np.nan
    out["q_GL_outcome"] = np.nan
    context = ["dataset", "cohort_scope", "allocation_id"]
    for _, ix in out.groupby(context, dropna=False, sort=False).groups.items():
        out.loc[ix, "q_scan"] = bh(out.loc[ix, "p"])
    gl = out[out.metric_family.isin(("gain", "loss"))]
    for _, ix in gl.groupby(
        context
        + ["beta", "outcome", "method", "mask", "subgroup", "model", "transform"],
        dropna=False,
        sort=False,
    ).groups.items():
        out.loc[ix, "q_GL_outcome"] = bh(out.loc[ix, "p"])
    return out


def planned_cells(catalog):
    rows = []
    for j in catalog["modules"]:
        tissues = catalog["datasets"][j["input_variant"]]["tissues"]
        counts = {t: sum(n[0] == t for n in j["nodes"]) for t in tissues}
        parts = [
            ("TS_" + t, counts[t] * (counts[t] - 1) // 2) for t in tissues if counts[t]
        ] + [
            ("CT_" + a + "__" + b, counts[a] * counts[b])
            for ai, a in enumerate(tissues)
            for b in tissues[ai + 1 :]
            if counts[a] and counts[b]
        ]
        ts = sum(n for c, n in parts if c.startswith("TS"))
        ct = sum(n for c, n in parts if c.startswith("CT"))
        for c, n in parts + [("all", ts + ct), ("TS", ts), ("CT", ct)]:
            rows.append(
                dict(
                    module_key=j["module_key"],
                    family=j["family"],
                    module=j["module"],
                    dataset=j["dataset"],
                    input_variant=j["input_variant"],
                    component=c,
                    possible_edges=n,
                    structural=n > 0,
                )
            )
    return pd.DataFrame(rows)


def freeze_manifest(root, catalog):
    from app_helpers.data import OUTCOME_LABELS, CATEGORICAL_OUTCOME_LABELS

    root = Path(root)
    dest = root / "associations"
    dest.mkdir(parents=True, exist_ok=True)
    cells = planned_cells(catalog)
    atomic_parquet(dest / "planned_cells.parquet", cells)
    contract = dict(
        version=2,
        allocation_seed=611010,
        betas=list(BETAS),
        metrics=list(PRIMARY),
        cohorts=list(COHORTS),
        numeric_outcomes=dict(OUTCOME_LABELS),
        nominal_outcomes=dict(CATEGORICAL_OUTCOME_LABELS),
        numeric_method="Spearman",
        nominal_method="Kruskal-Wallis",
        numeric_min_n=3,
        nominal_min_group_n=5,
        transform="native",
        mask="all",
        subgroup="all_independent_targets",
        model="unadjusted_descriptive",
        role="target",
        catalog_sha256=sha(root / "catalog.json"),
        planned_cells_sha256=sha(dest / "planned_cells.parquet"),
        unestimable_rule="Keep NA p and exclusion reason; BH uses only finite estimable p values; structural and valid counts reported separately",
        q_GL_outcome="BH jointly G/L, all registered partitions/components within dataset/cohort/allocation/beta/outcome/method/mask/subgroup/model/transform",
        q_scan="BH all planned structural components, all 20 outcomes and type-specific methods, G/L/S/A, both betas, all registered partitions within dataset/cohort/allocation; calculated before display filtering",
        additional_searches="Not covered by baseline q_scan; saved with their own exact donor/model/mask/transform context and correction membership",
        discovery_caveat="Exploratory fixed historical partitions, including previously examined validation cohorts; not untouched confirmatory inference",
        planned_structural_cells=int(cells.structural.sum()),
        structural_exclusions=int((~cells.structural).sum()),
        planned_structural_tests=int(
            cells.structural.sum()
            * len(BETAS)
            * len(COHORTS)
            * len(PRIMARY)
            * (len(OUTCOME_LABELS) + len(CATEGORICAL_OUTCOME_LABELS))
        ),
    )
    contract["metadata_sha256"] = {
        variant: sha(root / info["metadata_file"])
        for variant, info in catalog["datasets"].items()
    }
    contract["analysis_id"] = canonical_hash(contract)
    path = dest / "testing_manifest.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError(
            "Executed testing manifest changed; use a new analysis version"
        )
    if not path.exists():
        atomic_json(path, contract)
    return contract


def matrix_test(values, outcome, nominal=False, min_n=3, min_group_n=5):
    """Vectorize donor-level tests over score rows with exact tie handling."""
    x = np.asarray(values, float)
    M = len(x)
    p = np.full(M, np.nan)
    effect = p.copy()
    H = p.copy()
    reasons = np.full(M, "", dtype="<U80")
    groups = np.full(M, "", dtype=object)
    tested = np.zeros(M, int)
    if not nominal:
        y = pd.to_numeric(pd.Series(outcome), errors="coerce").to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        n = keep.sum(1)
        xx = np.where(keep, x, np.nan)
        yy = np.where(keep, y, np.nan)
        rx = stats.rankdata(xx, axis=1, nan_policy="omit")
        ry = stats.rankdata(yy, axis=1, nan_policy="omit")
        cx = rx - (n[:, None] + 1) / 2
        cy = ry - (n[:, None] + 1) / 2
        num = np.nansum(cx * cy, axis=1)
        den = np.sqrt(np.nansum(cx * cx, axis=1) * np.nansum(cy * cy, axis=1))
        ok = (n >= min_n) & (den > 0)
        np.divide(num, den, out=effect, where=ok)
        effect[ok] = np.clip(effect[ok], -1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = effect * np.sqrt((n - 2) / ((1 + effect) * (1 - effect)))
        p[ok] = stats.t.sf(abs(t[ok]), n[ok] - 2) * 2
        reasons[n < min_n] = "insufficient paired donors"
        reasons[(n >= min_n) & ~ok] = "constant score or outcome"
        tested = n.copy()
        return dict(
            n=n,
            n_tested=tested,
            effect=effect,
            p=p,
            kruskal_h=H,
            group_counts=groups,
            unavailable_reason=reasons,
        )
    y = pd.Series(outcome)
    valid_y = y.notna().to_numpy()
    ys = y.astype(str).to_numpy()
    keep = np.isfinite(x) & valid_y
    n = keep.sum(1)
    # Identical missingness patterns are common across modules. Grouping them
    # ranks an entire block once, while preserving each test's actual donors.
    patterns, inv = np.unique(np.packbits(keep, axis=1), axis=0, return_inverse=True)
    for pattern in range(len(patterns)):
        rows = np.flatnonzero(inv == pattern)
        donors = keep[rows[0]]
        levels, counts = np.unique(ys[donors], return_counts=True)
        eligible = levels[counts >= min_group_n]
        groups[rows] = json.dumps(
            dict(zip(levels, counts.astype(int).tolist())), sort_keys=True
        )
        use = donors & np.isin(ys, eligible)
        k = len(eligible)
        nn = int(use.sum())
        tested[rows] = nn
        if k < 2:
            reasons[rows] = "fewer than two eligible groups"
            continue
        if nn <= k:
            reasons[rows] = "insufficient non-missing values"
            continue
        z = x[rows][:, use]
        r = stats.rankdata(z, axis=1)
        numer = np.zeros(len(rows))
        for level in eligible:
            choose = ys[use] == level
            numer += r[:, choose].sum(1) ** 2 / choose.sum()
        raw = 12 * numer / (nn * (nn + 1)) - 3 * (nn + 1)
        ordered = np.sort(z, axis=1)
        tie_sum = np.zeros(len(rows))
        for rr in range(len(rows)):
            _, cc = np.unique(ordered[rr], return_counts=True)
            tie_sum[rr] = np.sum(cc.astype(float) ** 3 - cc)
        correction = 1 - tie_sum / (nn**3 - nn)
        ok = correction > 0
        hv = np.divide(raw, correction, out=np.full(len(rows), np.nan), where=ok)
        hv = np.maximum(hv, 0)
        H[rows] = hv
        effect[rows] = np.maximum(0, (hv - k + 1) / (nn - k))
        p[rows[ok]] = stats.chi2.sf(hv[ok], k - 1)
        reasons[rows[~ok]] = "constant scores"
    return dict(
        n=n,
        n_tested=tested,
        effect=effect,
        p=p,
        kruskal_h=H,
        group_counts=groups,
        unavailable_reason=reasons,
    )


def analyze_catalog(root=ROOT):
    root = Path(root)
    catalog = json.loads((root / "catalog.json").read_text())
    contract = freeze_manifest(root, catalog)
    if json.loads((root / "manifest.json").read_text()).get("status") != "complete":
        raise ValueError("Full score catalog must finish before baseline analysis")
    started = time.perf_counter()
    dest = root / "associations"
    files = []
    planned = planned_cells(catalog)
    structural = set(
        zip(
            planned.loc[planned.structural, "module_key"],
            planned.loc[planned.structural, "component"],
        )
    )
    for variant, info in catalog["datasets"].items():
        metadata = pd.read_parquet(root / info["metadata_file"]).set_index("sample_id")
        index = json.loads((root / "file_indexes" / f"{variant}.json").read_text())
        for beta in BETAS:
            for scope in COHORTS:
                path = dest / "uncorrected" / f"{variant}_beta{beta}_{scope}.parquet"
                files.append(path)
                receipt_path = path.with_suffix(".json")
                if path.exists() and receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt["analysis_id"] != contract["analysis_id"]:
                        raise ValueError(
                            "Saved association shard uses a different executed testing context"
                        )
                    if receipt["sha256"] != sha(path):
                        raise ValueError("Association shard content hash mismatch")
                    continue
                records = []
                for item in index:
                    if item["beta"] != beta or item["scope"] != scope:
                        continue
                    frame = pd.read_parquet(root / item["path"])
                    components = [
                        c
                        for c in frame.component.unique()
                        if (item["module_key"], c) in structural
                    ]
                    if not components:
                        continue
                    ids = frame.sample_id.drop_duplicates().tolist()
                    meta = metadata.reindex(ids)
                    row_specs = [(c, m) for c in components for m in PRIMARY]
                    matrix = np.stack(
                        [
                            frame[frame.component.eq(c)]
                            .set_index("sample_id")[m]
                            .reindex(ids)
                            .to_numpy(float)
                            for c, m in row_specs
                        ]
                    )
                    base = frame.iloc[0] if len(frame) else None
                    if base is None:
                        continue
                    for outcome in list(contract["numeric_outcomes"]) + list(
                        contract["nominal_outcomes"]
                    ):
                        nominal = outcome in contract["nominal_outcomes"]
                        answer = matrix_test(
                            matrix,
                            (
                                meta[outcome]
                                if outcome in meta
                                else np.full(len(ids), np.nan)
                            ),
                            nominal,
                        )
                        for ri, (component, metric) in enumerate(row_specs):
                            records.append(
                                dict(
                                    dataset=info["dataset"],
                                    cohort_scope=scope,
                                    allocation_id=info["allocation_id"],
                                    family=base.family,
                                    module=int(base.module),
                                    module_key=item["module_key"],
                                    component=component,
                                    beta=beta,
                                    metric_family=metric,
                                    outcome=outcome,
                                    method="Kruskal-Wallis" if nominal else "Spearman",
                                    effect_name=(
                                        "epsilon_squared" if nominal else "spearman_rho"
                                    ),
                                    mask="all",
                                    subgroup="all_independent_targets",
                                    model="unadjusted_descriptive",
                                    transform="native",
                                    role="target",
                                    analysis_id=contract["analysis_id"],
                                    score_version=f"control_calibrated_connectivity.v2.beta{beta}",
                                    **{k: v[ri] for k, v in answer.items()},
                                )
                            )
                atomic_parquet(path, pd.DataFrame(records))
                atomic_json(
                    receipt_path,
                    dict(
                        analysis_id=contract["analysis_id"],
                        sha256=sha(path),
                        planned_structural_tests=len(records),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            stage="associations",
                            variant=variant,
                            beta=beta,
                            cohort=scope,
                            tests=len(records),
                            seconds=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
    frames = [pd.read_parquet(p) for p in files]
    totals = []
    for dataset in sorted({i["dataset"] for i in catalog["datasets"].values()}):
        for scope in COHORTS:
            pieces = [
                f.loc[f.dataset.eq(dataset) & f.cohort_scope.eq(scope)]
                for f in frames
                if not f.empty
            ]
            frame = pd.concat(pieces, ignore_index=True)
            frame = correct(frame)
            atomic_parquet(dest / f"{dataset}_{scope}.parquet", frame)
            counts = dict(
                dataset=dataset,
                cohort_scope=scope,
                planned_structural_tests=len(frame),
                estimable_tests=int(frame.p.notna().sum()),
                excluded_tests=int(frame.p.isna().sum()),
                q_scan_finite=int(frame.q_scan.notna().sum()),
                q_GL_outcome_finite=int(frame.q_GL_outcome.notna().sum()),
                families=sorted(frame.family.unique()),
                betas=sorted(frame.beta.unique().astype(int).tolist()),
                exclusion_reasons=frame.loc[frame.p.isna(), "unavailable_reason"]
                .value_counts()
                .to_dict(),
            )
            totals.append(counts)
    atomic_json(
        dest / "completion.json",
        dict(
            status="complete",
            analysis_id=contract["analysis_id"],
            seconds=time.perf_counter() - started,
            contexts=totals,
            files=[
                dict(path=str(p.relative_to(root)), sha256=sha(p))
                for p in dest.glob("*_*.parquet")
            ],
        ),
    )
    return totals
