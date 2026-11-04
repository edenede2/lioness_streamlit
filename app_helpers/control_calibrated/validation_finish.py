"""Independent family audits, aggregate summaries and local-only activation."""

from pathlib import Path
import json, hashlib, resource, time
import numpy as np
import pandas as pd
from scipy import stats
from .validation_spec import WORK, BASE, COHORTS, METRICS
from .validation_run import prepared, cohort_mask
from . import validation_inference as I
from .registry import RESEARCH_ROOT
from .inputs import sha, atomic_json, atomic_parquet


def summarize(root, spec):
    root = Path(root)
    dest = root / "results"
    summaries = []
    missing = []
    focused = []
    deltas = []
    composition = []
    study = []
    for variant in ("brain", "motor", "motor_regional"):
        _, meta, rows = prepared(str(root), variant)
        for cohort in COHORTS:
            f = meta.loc[cohort_mask(meta, cohort)]
            for field in ("study", "sex_code", "diagnosis_group"):
                for level, n in (
                    f[field].fillna("Missing").astype(str).value_counts().items()
                ):
                    composition.append(
                        dict(
                            input_variant=variant,
                            cohort=cohort,
                            field=field,
                            level=level,
                            count=int(n),
                            cohort_n=len(f),
                            proportion=n / len(f),
                        )
                    )
            for field in ("age_at_death", "education_years"):
                v = I.finite_numeric(f[field])
                finite = v[np.isfinite(v)]
                composition.append(
                    dict(
                        input_variant=variant,
                        cohort=cohort,
                        field=field,
                        level="Numeric summary",
                        count=len(finite),
                        cohort_n=len(f),
                        missing=int((~np.isfinite(v)).sum()),
                        mean=float(np.mean(finite)),
                        median=float(np.median(finite)),
                    )
                )
            study.append(
                dict(
                    input_variant=variant,
                    cohort=cohort,
                    n=len(f),
                    ROS=int(f.study.eq("ROS").sum()),
                    MAP=int(f.study.eq("MAP").sum()),
                    missing_study=int(f.study.isna().sum()),
                    applicable=variant == "brain",
                    reason="" if variant == "brain" else "single_study_MAP_motor",
                )
            )
    for dataset in ("brain", "motor"):
        for cohort in COHORTS:
            f = pd.read_parquet(dest / f"{dataset}_{cohort}.parquet")
            for key, g in f.groupby(
                [
                    "dataset",
                    "cohort",
                    "model",
                    "study_stratum",
                    "beta",
                    "metric_family",
                    "outcome",
                ],
                sort=False,
            ):
                summaries.append(
                    dict(
                        zip(
                            [
                                "dataset",
                                "cohort",
                                "model",
                                "study_stratum",
                                "beta",
                                "metric_family",
                                "outcome",
                            ],
                            key,
                        )
                    )
                    | dict(
                        planned=len(g),
                        estimable=int(g.p.notna().sum()),
                        excluded=int(g.p.isna().sum()),
                        q_scan_lt05=int(g.q_scan.lt(0.05).sum()),
                        q_GL_outcome_lt05=int(g.q_GL_outcome.lt(0.05).sum()),
                        median_effect=float(g.effect.median()),
                        median_absolute_effect=float(g.effect.abs().median()),
                        MC_floor_tests=int(g.monte_carlo_floor.sum()),
                        intervals_available=int(
                            (g.ci_low.notna() & g.ci_high.notna()).sum()
                        ),
                    )
                )
            focused.append(f.loc[f.focused])
            a = f.loc[f.model.eq("A") & f.study_stratum.eq("all")].copy()
            missing.append(
                a[
                    [
                        c
                        for c in [
                            "analysis_id",
                            "dataset",
                            "input_variant",
                            "cohort",
                            "family",
                            "module",
                            "module_key",
                            "component",
                            "beta",
                            "metric_family",
                            "outcome",
                            "candidate_n",
                            "score_available_n",
                            "score_outcome_n",
                            "complete_case_n",
                            "missing_score_n",
                            "missing_outcome_after_score_n",
                            "missing_covariates_after_pair_n",
                        ]
                        if c in a
                    ]
                ]
            )
            keys = ["input_variant", "row_id", "outcome"]
            cols = keys + ["n", "effect", "donor_set_hash"]
            abc = a[
                [
                    c
                    for c in [
                        "dataset",
                        "cohort",
                        "family",
                        "module",
                        "module_key",
                        "component",
                        "beta",
                        "metric_family",
                        *cols,
                    ]
                    if c in a
                ]
            ].rename(
                columns={"n": "N_A", "effect": "rho_A", "donor_set_hash": "hash_A"}
            )
            for model in ["B", "C"]:
                part = f.loc[
                    f.model.eq(model) & f.study_stratum.eq("all"), cols
                ].rename(
                    columns={
                        "n": "N_" + model,
                        "effect": "rho_" + model,
                        "donor_set_hash": "hash_" + model,
                    }
                )
                abc = abc.merge(part, on=keys, validate="one_to_one")
            assert abc.hash_B.equals(abc.hash_C)
            abc["A_B_same_membership"] = abc.hash_A.eq(abc.hash_B)
            abc["B_C_same_membership"] = abc.hash_B.eq(abc.hash_C)
            abc["selection_change_B_minus_A"] = abc.rho_B - abc.rho_A
            abc["adjustment_change_C_minus_B"] = abc.rho_C - abc.rho_B
            deltas.append(abc)
    for name, frame in [
        ("composition", pd.DataFrame(composition)),
        ("study_applicability", pd.DataFrame(study)),
        ("missingness", pd.concat(missing, ignore_index=True)),
        ("effect_summary", pd.DataFrame(summaries)),
        ("focused_catalog", pd.concat(focused, ignore_index=True)),
        ("selection_adjustment_deltas", pd.concat(deltas, ignore_index=True)),
    ]:
        atomic_parquet(dest / (name + ".parquet"), frame)
    atomic_json(
        root / "summary_complete.json",
        dict(
            status="complete", analysis_id=spec["analysis_id"], summaries=len(summaries)
        ),
    )


def _check_bh(frame, context, q="q_scan"):
    checked = 0
    for _, ids in frame.groupby(context, dropna=False, sort=False).groups.items():
        p = frame.loc[ids, "p"].to_numpy(float)
        actual = frame.loc[ids, q].to_numpy(float)
        finite = np.isfinite(p)
        assert np.isnan(actual[~finite]).all()
        if finite.any():
            np.testing.assert_allclose(
                actual[finite],
                stats.false_discovery_control(p[finite]),
                rtol=1e-12,
                atol=1e-14,
            )
        checked += 1
    return checked


def verify(root, spec):
    root = Path(root)
    start = time.perf_counter()
    bh_groups = 0
    tested_rows = 0
    membership_rows = 0
    exact_repaired = 0
    for dataset in ("brain", "motor"):
        for cohort in COHORTS:
            f = pd.read_parquet(root / "results" / f"{dataset}_{cohort}.parquet")
            tested_rows += len(f)
            assert f["mask"].eq("all").all() and f["transform"].eq("native").all()
            assert (
                f.is_canonical.all()
                and f.p.dropna().between(0, 1, inclusive="right").all()
            )
            adjusted = ~f.model.isin(["A", "B"])
            assert (
                f.loc[adjusted & (f.n.lt(30) | f.residual_df.lt(20)), "p"].isna().all()
            )
            assert (
                not f.loc[~f.metric_family.isin(["gain", "loss"]), "q_GL_outcome"]
                .notna()
                .any()
            )
            context = [
                "dataset",
                "cohort",
                "allocation_id",
                "model",
                "study_stratum",
                "mask",
                "transform",
            ]
            bh_groups += _check_bh(f, context)
            bh_groups += _check_bh(
                f.loc[f.metric_family.isin(["gain", "loss"])],
                context + ["beta", "outcome"],
                "q_GL_outcome",
            )
            # Independently reconstruct every tested donor set from immutable
            # native scores, tissue availability and declared covariates.
            for (variant, outcome, model, stratum), g in f.groupby(
                ["input_variant", "outcome", "model", "study_stratum"], sort=False
            ):
                matrix, meta, _ = prepared(str(root), variant)
                ids = g.row_id.to_numpy(int)
                valid = (
                    np.isfinite(matrix[ids])
                    & (
                        cohort_mask(meta, cohort)
                        & np.isfinite(I.finite_numeric(meta[outcome]))
                    )[None, :]
                )
                if model != "A":
                    valid &= I.complete_covariates(meta)[None, :]
                if stratum != "all":
                    valid &= meta.study.eq(stratum).fillna(False).to_numpy()[None, :]
                patterns, inverse = np.unique(
                    np.packbits(valid, axis=1), axis=0, return_inverse=True
                )
                hashes = []
                for k in range(len(patterns)):
                    mask = valid[np.flatnonzero(inverse == k)[0]]
                    hashes.append(I.membership_hash(meta.loc[mask, "sample_id"]))
                assert np.array_equal(
                    np.asarray(hashes, dtype=object)[inverse],
                    g.donor_set_hash.to_numpy(),
                )
                assert np.array_equal(valid.sum(axis=1), g.n.to_numpy())
                membership_rows += len(g)
    for path in sorted((root / "repaired_baseline").glob("*_*.parquet")):
        if path.name == "before_after.parquet":
            continue
        f = pd.read_parquet(path)
        context = ["dataset", "cohort_scope", "allocation_id"]
        bh_groups += _check_bh(f, context)
        bh_groups += _check_bh(
            f.loc[f.metric_family.isin(["gain", "loss"])],
            context
            + ["beta", "outcome", "method", "mask", "subgroup", "model", "transform"],
            "q_GL_outcome",
        )
        original = pd.read_parquet(RESEARCH_ROOT / "associations" / path.name)
        assert len(original) == len(f)
        keys = ["module_key", "component", "beta", "metric_family", "outcome"]
        a = original.set_index(keys).sort_index()
        b = f.set_index(keys).sort_index()
        pd.testing.assert_series_equal(a.effect, b.effect, check_exact=True)
        nominal = a.method.eq("Kruskal-Wallis")
        np.testing.assert_array_equal(a.loc[nominal, "p"], b.loc[nominal, "p"])
        bad = b.n.eq(4) & b.effect.abs().ge(1 - 1e-12) & b.p_old.eq(0)
        assert b.loc[bad, "old_q_scan"].eq(0).all()
        np.testing.assert_allclose(b.loc[bad, "p"], 1 / 12)
        exact_repaired += int(bad.sum())
    alias_rows = 0
    for variant in ("brain", "motor", "motor_regional"):
        matrix, meta, rows = prepared(str(root), variant)
        prep = json.loads((root / "prepared" / f"{variant}.json").read_text())
        assert sha(root / "prepared" / f"{variant}.npy") == prep["matrix_sha256"]
        assert (
            sha(root / "prepared" / f"{variant}_metadata.parquet")
            == prep["metadata_sha256"]
        )
        lookup = rows.set_index(
            ["module_key", "beta", "component", "metric_family"]
        ).row_id
        for r in rows.loc[~rows.is_canonical].itertuples():
            canonical = lookup.loc[
                (r.module_key, r.beta, r.canonical_component, r.metric_family)
            ]
            # Exact numerical aliases, not independent evidence.
            np.testing.assert_array_equal(matrix[r.row_id], matrix[canonical])
            alias_rows += 1
    for relative, digest in spec["input_sha256"].items():
        if sha(RESEARCH_ROOT / relative) != digest:
            raise ValueError(
                "Frozen source manifest/metadata/baseline changed: " + relative
            )
    receipts = json.loads((root / "native_input_receipts.json").read_text())["files"]
    for r in receipts:
        if sha(RESEARCH_ROOT / r["path"]) != r["sha256"]:
            raise ValueError("Original native score file changed")
    for name in ("interaction", "comparators", "incremental"):
        f = pd.read_parquet(root / "results" / f"{name}.parquet")
        if name == "comparators":
            for cohort, g in f.groupby("cohort"):
                h = g.drop_duplicates("hypothesis_id").copy()
                h["q_scan"] = h.q_focused
                bh_groups += _check_bh(h, ["cohort"])
                expected = h.set_index("hypothesis_id").q_focused
                np.testing.assert_allclose(
                    g.q_focused, g.hypothesis_id.map(expected), equal_nan=True
                )
            assert (
                f.groupby(
                    ["family", "module", "component", "cohort", "outcome", "beta"]
                )
                .donor_set_hash.nunique()
                .eq(1)
                .all()
            )
        else:
            h = f.assign(q_scan=f.q_focused)
            bh_groups += _check_bh(h, ["cohort"])
        assert (f.p.dropna() > 0).all()
    result = dict(
        status="passed",
        analysis_id=spec["analysis_id"],
        catalog_rows_verified=tested_rows,
        actual_donor_memberships_verified=membership_rows,
        independent_SciPy_BH_families=bh_groups,
        N4_zero_p_repaired=exact_repaired,
        exact_alias_vectors_verified=alias_rows,
        original_native_files_hash_verified=len(receipts),
        score_allocation_and_transforms_unchanged=True,
        old_baseline_unchanged=True,
        projection_FL_and_HC3_independent_unit_tests="See recorded pytest and statsmodels0.14.5 tests",
        seconds=time.perf_counter() - start,
    )
    atomic_json(root / "verification.json", result)
    atomic_json(WORK / "scientific_verification.json", result)
    return result


def activate(root, spec):
    root = Path(root)
    if not (root / "report_complete.json").exists():
        raise ValueError(
            "The scientific report must be generated before local activation"
        )
    if json.loads((root / "verification.json").read_text())["status"] != "passed":
        raise ValueError("Scientific verification must pass first")
    files = {}
    for folder in ("results", "repaired_baseline", "prepared"):
        for p in (root / folder).iterdir():
            if p.is_file() and p.suffix in (".json", ".parquet", ".npy"):
                files[str(p.relative_to(root))] = sha(p)
    for name in (
        "analysis_spec.json",
        "provenance_inventory.json",
        "preparation_complete.json",
        "repair_complete.json",
        "scan_complete.json",
        "focused_complete.json",
        "verification.json",
        "repair_execution.json",
        "scan_execution.json",
        "focused_plan.json",
        "summary_complete.json",
        "report_complete.json",
        "native_input_receipts.json",
    ):
        files[name] = sha(root / name)
    scan = json.loads((root / "scan_complete.json").read_text())
    repair = json.loads((root / "repair_complete.json").read_text())
    record = dict(
        schema_version=1,
        status="complete",
        analysis_id=spec["analysis_id"],
        local_only=True,
        source_release=spec["source_release"],
        catalog_modules=sum(v["modules"] for v in spec["families"].values()),
        families=list(spec["families"]),
        catalog_rows=sum(c["planned"] for c in scan["contexts"]),
        baseline_rows=sum(c["rows"] for c in repair["contexts"]),
        repaired_numeric_tests=sum(c["numeric_repaired"] for c in repair["contexts"]),
        files=files,
        publication_authorized=False,
        production_release_pointer_changed=False,
        prediction_changed=False,
    )
    atomic_json(root / "complete.json", record)
    atomic_json(
        BASE / "current_local.json",
        dict(schema_version=1, analysis_id=spec["analysis_id"], local_only=True),
    )
    return record
