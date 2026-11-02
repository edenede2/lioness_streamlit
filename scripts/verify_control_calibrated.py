#!/usr/bin/env python3
"""Read-only full-catalog audit, including independent source calculations.

No prediction experiment is invoked. Review evidence contains aggregate counts
and numerical errors, never source expression or donor identity crosswalks.
"""
from pathlib import Path
import os, sys, json, argparse, time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from app_helpers.control_calibrated import adapter as A
from app_helpers.control_calibrated.registry import ROOT, METRICS, PRIMARY
from app_helpers.control_calibrated.inputs import sha, atomic_json, atomic_parquet
from app_helpers.control_calibrated.analysis import planned_cells


def independent_module_check(root, job, info, beta):
    """Use direct correlation matrices, not the optimized sufficient statistics."""
    X = np.load(
        root / "private" / f"{job['input_variant']}_expression.npy", mmap_mode="r"
    )
    ix = np.array(job["node_indices"])
    R = np.array(info["role_indices"]["R"])
    C = np.array(info["role_indices"]["C"])
    meta = pd.read_parquet(root / info["metadata_file"])
    targets = meta.index[meta.role.eq("target") & meta.in_original_analysis].to_numpy()[
        :5
    ]
    ref = np.asarray(X[R][:, ix])
    mean = np.nanmean(ref, axis=0)
    ref = np.where(np.isfinite(ref), ref, mean)
    reference = np.corrcoef(ref, rowvar=False)
    r, c = np.triu_indices(len(ix), 1)
    r0 = reference[r, c]
    f = lambda x: np.sign(x) * abs(x) ** beta
    w = []
    for row in np.r_[C, targets]:
        value = np.asarray(X[row, ix])
        value = np.where(np.isfinite(value), value, mean)
        corr = np.corrcoef(np.vstack((ref, value)), rowvar=False)[r, c]
        w.append((len(R) + 1) * f(corr) - len(R) * f(r0))
    w = np.stack(w)
    wc = w[: len(C)]
    wt = w[len(C) :]
    nu = abs(wc).mean(0)
    tau = abs(wc).std(0, ddof=1)
    scale = abs(wc).max(0)
    valid = (
        np.isfinite(wc).all(0)
        & (scale > 0)
        & (
            tau / np.where(scale > 0, scale, 1)
            > 64 * np.finfo(float).eps * np.sqrt(len(C))
        )
    )
    aa = (abs(wt[:, valid]) - nu[valid]) / tau[valid]
    ts = np.array([n[0] for n in job["nodes"]])
    parts = {"all": np.ones(len(r), bool), "TS": ts[r] == ts[c], "CT": ts[r] != ts[c]}
    errors = []
    for component, select in parts.items():
        good = valid & select
        if not good.any():
            continue
        a = (abs(wt[:, good]) - nu[good]) / tau[good]
        expected = {"gain": np.maximum(a, 0).mean(1), "loss": np.maximum(-a, 0).mean(1)}
        expected["net_change"] = expected["gain"] - expected["loss"]
        expected["magnitude_atypicality"] = expected["gain"] + expected["loss"]
        stored = A.scores(
            module_set=job["family"],
            module=job["module"],
            component=component,
            estimator=f"calibrated_beta{beta}",
            regional_tissue=job.get("regional_tissue"),
            root=root,
        )
        for metric, value in expected.items():
            actual = (
                stored[stored.metric_family.eq(metric)]
                .set_index("sample_id")
                .reindex(meta.iloc[targets].sample_id)
                .metric_raw.to_numpy(float)
            )
            np.testing.assert_allclose(actual, value, rtol=2e-10, atol=2e-11)
            errors.append(float(np.max(abs(actual - value))))
    return dict(
        family=job["family"],
        module=job["module"],
        beta=beta,
        target_examples=len(targets),
        physical_edges=len(r),
        max_absolute_score_error=max(errors, default=0),
        method="Independent direct np.corrcoef on R and each R+donor, explicit ddof1 calibration and fixed physical unions",
    )


def verify(root=ROOT, output=None):
    root = Path(root)
    output = Path(
        output or APP / "artifacts/control_calibrated_implementation_20261102"
    )
    started = time.perf_counter()
    catalog = A.catalog(root)
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "associations/completion.json").read_text())
    assert manifest["status"] == "complete" and completion["status"] == "complete"
    contract = json.loads((root / "associations/testing_manifest.json").read_text())
    assert contract["catalog_sha256"] == sha(root / "catalog.json")
    role_counts = []
    checked_files = 0
    checked_rows = 0
    for variant, info in catalog["datasets"].items():
        meta = pd.read_parquet(root / info["metadata_file"])
        assert sha(root / info["metadata_file"]) == contract["metadata_sha256"][variant]
        assert meta.sample_id.is_unique
        training = meta[meta.role.isin(["reference", "calibration"])]
        assert training.diagnosis_group.eq("Control").all()
        count = (
            meta.groupby(["role", "diagnosis_group"], dropna=False)
            .size()
            .rename("donors")
            .reset_index()
        )
        count["variant"] = variant
        role_counts.append(count)
        roles = meta.set_index("sample_id").role
        report = next(r for r in manifest["generation"] if r["variant"] == variant)
        for b in (3, 1):
            path = root / report["parameter_files"][str(b)]["path"]
            assert sha(path) == report["parameter_files"][str(b)]["sha256"]
            arr = np.load(path, mmap_mode="r")
            valid = arr[:, 10].astype(bool)
            signed = arr[:, 11].astype(bool)
            assert np.all(arr[valid | signed, 9] == len(info["role_indices"]["C"]))
        for item in json.loads((root / "file_indexes" / f"{variant}.json").read_text()):
            path = root / item["path"]
            assert sha(path) == item["sha256"]
            f = pd.read_parquet(path)
            assert f.role.eq("target").all()
            assert f.beta.eq(item["beta"]).all()
            assert f.module_key.eq(item["module_key"]).all()
            assert roles.reindex(f.sample_id).eq("target").all()
            np.testing.assert_allclose(
                f.gain - f.loss, f.net_change, equal_nan=True, atol=1e-13
            )
            np.testing.assert_allclose(
                f.gain + f.loss, f.magnitude_atypicality, equal_nan=True, atol=1e-13
            )
            for side in ("gain", "loss"):
                np.testing.assert_allclose(
                    f[side],
                    sum(
                        f[side + "_" + c]
                        for c in (
                            "zero",
                            "near_zero",
                            "uncertain",
                            "preserved",
                            "reversed",
                        )
                    ),
                    equal_nan=True,
                    rtol=2e-12,
                    atol=2e-12,
                )
            bad = f.unavailable_reason.ne("ok")
            assert f.loc[bad, "gain"].isna().all()
            assert f.loc[~bad, "gain"].notna().all()
            assert (f.gain.dropna() >= 0).all() and (f.loss.dropna() >= 0).all()
            checked_files += 1
            checked_rows += len(f)
    atomic_parquet(
        output / "verified_role_counts.parquet",
        pd.concat(role_counts, ignore_index=True),
    )
    independent = []
    for family, entry in catalog["families"].items():
        jobs = [
            j
            for j in catalog["modules"]
            if j["family"] == family
            and (entry["regional"] or len({n[0] for n in j["nodes"]}) == 3)
        ]
        job = min(jobs, key=lambda j: j["physical_nodes"])
        for b in (3, 1):
            independent.append(
                independent_module_check(
                    root, job, catalog["datasets"][job["input_variant"]], b
                )
            )
    atomic_parquet(
        output / "independent_source_checks.parquet", pd.DataFrame(independent)
    )
    fdr = []
    for dataset in ("brain", "motor"):
        for cohort in (
            "complete_450",
            "incomplete_region_validation",
            "maximum_component",
        ):
            f = pd.read_parquet(root / "associations" / f"{dataset}_{cohort}.parquet")
            assert set(f.beta) == {1, 3}
            assert f.p.isna().equals(f.q_scan.isna())
            valid = f.p.notna()
            np.testing.assert_allclose(
                f.loc[valid, "q_scan"],
                false_discovery_control(f.loc[valid, "p"].to_numpy(), method="bh"),
                rtol=1e-12,
                atol=1e-14,
            )
            for _, g in f[f.metric_family.isin(["gain", "loss"])].groupby(
                [
                    "allocation_id",
                    "beta",
                    "outcome",
                    "method",
                    "mask",
                    "subgroup",
                    "model",
                    "transform",
                ],
                dropna=False,
            ):
                ok = g.p.notna()
                if ok.any():
                    np.testing.assert_allclose(
                        g.loc[ok, "q_GL_outcome"],
                        false_discovery_control(g.loc[ok, "p"].to_numpy(), method="bh"),
                        rtol=1e-12,
                        atol=1e-14,
                    )
            assert f.loc[f.p.isna(), "unavailable_reason"].ne("").all()
            fdr.append(
                dict(
                    dataset=dataset,
                    cohort_scope=cohort,
                    tests=len(f),
                    estimable=int(valid.sum()),
                    excluded=int((~valid).sum()),
                    families=f.family.nunique(),
                    betas=f.beta.nunique(),
                )
            )
    assert sum(c["tests"] for c in fdr) == contract["planned_structural_tests"]
    atomic_parquet(output / "verified_fdr_contexts.parquet", pd.DataFrame(fdr))
    preservation = {}
    for name in ("preexisting_changes.json", "prediction_snapshot.json"):
        original = json.loads((output / name).read_text())
        changed = [
            path
            for path, digest in original.items()
            if not (APP / path).exists() or sha(APP / path) != digest
        ]
        assert not changed, changed
        preservation[name] = dict(
            checked=len(original), unchanged=len(original), changed=0
        )
    from app_helpers.data import (
        FEATURE_LABELS,
        PREDICTION_MODULE_SETS,
        load_feature_definitions,
    )

    assert not set(METRICS) & set(FEATURE_LABELS)
    features = load_feature_definitions()
    assert (
        not features.astype(str)
        .apply(
            lambda c: c.str.contains("calibrated_beta|control_calibrated", regex=True)
        )
        .any()
        .any()
    )
    assert PREDICTION_MODULE_SETS == ("full_cohort", "control_derived")
    result = dict(
        status="passed",
        checked_score_files=checked_files,
        checked_score_rows=checked_rows,
        independent_source_calculations=independent,
        testing_manifest_id=contract["analysis_id"],
        planned_tests=contract["planned_structural_tests"],
        fdr_independent_implementation="scipy.stats.false_discovery_control BH",
        preservation=preservation,
        prediction_training_invoked=False,
        seconds=time.perf_counter() - started,
    )
    atomic_json(output / "verification.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("independent_source_calculations",)
            }
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.root, args.output)
