"""Small source-to-artifact tests, independent calculations and family contracts."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, kruskal
from app_helpers.control_calibrated import adapter as A, registry as CC
from app_helpers.control_calibrated.inputs import atomic_json, atomic_parquet, sha
from app_helpers.control_calibrated.generate import (
    generate,
    reduce_edges,
    fast_calibration,
    pearson_block,
    weights,
)
from app_helpers.control_calibrated.generate_masks import channels
from app_helpers.control_calibrated.core import (
    Settings,
    fit_reference,
    calibrate,
    aggregate_weights,
)
from app_helpers.control_calibrated.analysis import (
    bh,
    correct,
    matrix_test,
    freeze_manifest,
    analyze_catalog,
)


@pytest.fixture(scope="module")
def catalog_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("control_calibrated")
    (root / "private").mkdir()
    rng = np.random.default_rng(4541)
    datasets = {}
    modules = []
    families = {}
    for dataset, tissues, fams in [
        ("brain", ["AC", "DLPFC", "PCG"], ["control_derived", "full_cohort"]),
        ("motor", ["VH", "SMA", "QUAD"], ["muscle_control_l4", "muscle_all_donors_l4"]),
    ]:
        nodes = [[t, g] for t in tissues for g in ("x", "y")]
        X = rng.normal(size=(34, 6))
        X[14, 3] = np.nan
        X[-1, 1] = np.nan
        P = np.ones((34, 3), bool)
        P[-1, 2] = False
        np.save(root / "private" / f"{dataset}_expression.npy", X)
        np.save(root / "private" / f"{dataset}_presence.npy", P)
        ids = [dataset + str(i) for i in range(34)]
        meta = pd.DataFrame(
            dict(
                sample_id=ids,
                role=["reference"] * 12 + ["calibration"] * 8 + ["target"] * 14,
                in_original_analysis=[True] * 30 + [False] * 4,
                diagnosis_group=["Control"] * 20 + ["AD"] * 7 + ["MCI"] * 7,
                cogn_global=rng.normal(size=34),
                clusters=np.tile(["a", "b"], 17),
            )
        )
        for i, t in enumerate(tissues):
            meta["has_" + t] = P[:, i]
        atomic_parquet(root / "metadata" / f"{dataset}.parquet", meta)
        datasets[dataset] = dict(
            dataset=dataset,
            tissues=tissues,
            nodes=nodes,
            n_donors=34,
            n_nodes=6,
            shape=[34, 6],
            role_indices={
                "R": list(range(12)),
                "C": list(range(12, 20)),
                "T": list(range(20, 30)),
            },
            source_sha256=["synthetic"],
            presence_sha256=sha(root / "private" / f"{dataset}_presence.npy"),
            expression_sha256=sha(root / "private" / f"{dataset}_expression.npy"),
            metadata_file=f"metadata/{dataset}.parquet",
            allocation_id="synthetic_" + dataset,
        )
        for family in fams:
            families[family] = dict(
                modules=1, dataset=dataset, regional=False, label=family
            )
            modules.append(
                dict(
                    family=family,
                    module=1,
                    module_key=family + "::M1",
                    input_variant=dataset,
                    dataset=dataset,
                    nodes=nodes,
                    node_indices=list(range(6)),
                    regional_tissue=None,
                )
            )
    atomic_json(
        root / "catalog.json",
        dict(families=families, datasets=datasets, modules=modules),
    )
    generate(root, edge_batch=4, workers=1)
    return root


def test_each_family_scope_adapter_and_incomplete_union(catalog_root):
    for family in A.catalog(catalog_root)["families"]:
        a = A.scores(
            module_set=family,
            module=1,
            metric_family="gain",
            estimator="calibrated_beta3",
            root=catalog_root,
        )
        assert set(a.role) == {"target"}
        assert a.sample_id.nunique() == 10
        assert a.beta.eq(3).all()
        assert a.module_key.eq(family + "::M1").all()
        b = A.scores(
            module_set=family,
            module=1,
            metric_family="gain",
            estimator="calibrated_beta1",
            root=catalog_root,
        )
        assert not np.allclose(a.metric_raw, b.metric_raw, equal_nan=True)
        assert not a.artifact_hash.equals(b.artifact_hash)
        f = A.scores(
            module_set=family,
            module=1,
            metric_family="gain",
            estimator="calibrated_beta3",
            cohort_scope="incomplete_region_validation",
            root=catalog_root,
        )
        assert f.sample_id.nunique() == 1
        assert f.loc[f.component.eq("all"), "metric_raw"].isna().all()
        ts = f[f.component.eq("TS_AC" if "muscle" not in family else "TS_VH")]
        assert ts.metric_raw.notna().all()
        assert (
            f.loc[f.component.eq("all"), "unavailable_reason"]
            .eq("missing_required_tissue")
            .all()
        )


def test_physical_union_denominators_and_sign_decomposition(catalog_root):
    f = A.scores(module_set="full_cohort", module=1, root=catalog_root)
    p = f.pivot(
        index=["sample_id", "component"], columns="metric_family", values="metric_raw"
    )
    np.testing.assert_allclose(p.gain - p.loss, p.net_change)
    np.testing.assert_allclose(p.gain + p.loss, p.magnitude_atypicality)
    one = f[f.metric_family.eq("gain")]
    assert one[one.component.eq("all")].possible_edges.eq(15).all()
    assert one[one.component.eq("TS")].possible_edges.eq(3).all()
    assert one[one.component.eq("CT")].possible_edges.eq(12).all()
    for _, g in one.groupby("sample_id"):
        g = g.set_index("component")
        assert g.loc["all", "metric_raw"] == pytest.approx(
            (
                g.loc["TS", "metric_raw"] * g.loc["TS", "eligible_edges"]
                + g.loc["CT", "metric_raw"] * g.loc["CT", "eligible_edges"]
            )
            / g.loc["all", "eligible_edges"]
        )
        assert g.loc["all", "metric_raw"] == pytest.approx(
            sum(
                g.loc["all", "gain_" + c]
                for c in ("zero", "near_zero", "uncertain", "preserved", "reversed")
            )
        )
    arrays = np.load(catalog_root / "private/brain/edges.npy")
    assert len(arrays) == 15  # shared across the two module definitions


def test_selected_source_calculation_independent_of_optimized_helper(catalog_root):
    x = np.load(catalog_root / "private/brain_expression.npy")
    R = x[:12]
    C = x[12:20]
    rr = np.corrcoef(R, rowvar=False)[0, 1]
    for beta in (1, 3):
        w = []
        for donor in np.vstack((C, x[20:30])):
            r = np.corrcoef(np.vstack((R, donor)), rowvar=False)[0, 1]
            w.append(
                13 * np.sign(r) * abs(r) ** beta - 12 * np.sign(rr) * abs(rr) ** beta
            )
        nu = np.mean(abs(np.array(w[:8])))
        tau = np.std(abs(np.array(w[:8])), ddof=1)
        a = (abs(np.array(w[8:])) - nu) / tau
        f = A.scores(
            module_set="full_cohort",
            module=1,
            metric_family="gain",
            component="TS_AC",
            estimator="calibrated_beta" + str(beta),
            root=catalog_root,
        )
        np.testing.assert_allclose(f.metric_raw, np.maximum(a, 0), atol=2e-13)


def test_sparse_channel_reduction_matches_direct_accumulator():
    rng = np.random.default_rng(11)
    s = Settings(12, 8)
    cal = calibrate(rng.normal(size=(8, 13)), s)
    w = rng.normal(size=(9, 13))
    w[0, 3] = np.nan
    ix = np.array([0, 3, 7, 8])
    a = reduce_edges(w, cal, ix)
    b = channels(w, cal)[ix].sum(0)
    np.testing.assert_allclose(a, b, atol=1e-13)


def test_donor_batched_reconstruction_and_masked_artifact_parity(catalog_root):
    from app_helpers.control_calibrated.inspection import (
        prepare_reference_caches,
        reference_cache,
    )
    from app_helpers.control_calibrated import generate_masks as GM
    from app_helpers.control_calibrated.masks import mask_artifact_path

    catalog = A.catalog(catalog_root)
    info = catalog["datasets"]["brain"]
    job = next(j for j in catalog["modules"] if j["family"] == "full_cohort")
    prepare_reference_caches(catalog_root)
    refpath = catalog_root / "private/brain_reference.npz"
    ref = reference_cache(str(catalog_root), "brain", refpath.stat().st_mtime_ns)
    X = np.load(catalog_root / "private/brain_expression.npy")
    P = np.load(catalog_root / "private/brain_presence.npy")
    R = np.arange(12)
    donors = np.arange(32)
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    full = pearson_block(X, ref, P, ref["node_tissues"], rows, cols, R)
    part = pearson_block(
        X, ref, P[donors], ref["node_tissues"], rows, cols, R, donor_indices=donors
    )
    for beta in (1, 3):
        np.testing.assert_allclose(
            weights(full[0], full[1], beta, R, 12)[donors],
            weights(part[0], part[1], beta, R, 12),
        )
    edges = np.load(catalog_root / "private/brain/edges.npy")
    rule = "ad_control_discovery_effect__hedges_g__fixed_0.40__either"
    mapping = (
        catalog_root
        / "private/mask_mappings"
        / A.module_filename(job["module_key"]).replace(".parquet", ".npz")
    )
    mapping.parent.mkdir(exist_ok=True)
    np.savez(mapping, codes=edges, **{rule: np.ones(len(edges), bool)})
    atomic_json(
        mapping.with_suffix(".json"),
        dict(
            status="mapped",
            array_sha256=sha(mapping),
            mapped_keys=[rule],
            unavailable={},
        ),
    )
    cal = {
        b: np.load(catalog_root / f"private/brain/calibration_beta{b}.npy")
        for b in (1, 3)
    }
    GM._STATE.clear()
    GM._STATE.update(
        root=catalog_root,
        catalog=catalog,
        specs={
            "full_cohort": {
                rule: dict(
                    rule=rule,
                    label="Synthetic all-edge frozen mask",
                    provenance="Synthetic parity test; no target learning",
                )
            }
        },
        variants={
            "brain": dict(
                X=X,
                presence=P,
                reference=ref,
                R=R,
                meta=pd.read_parquet(catalog_root / "metadata/brain.parquet"),
                edges=edges,
                cal=cal,
                cal_hash={
                    b: sha(catalog_root / f"private/brain/calibration_beta{b}.npy")
                    for b in (1, 3)
                },
            )
        },
    )
    report = GM.module_task(job)
    assert report["status"] == "complete"
    for beta in (1, 3):
        actual = pd.read_parquet(
            mask_artifact_path(
                catalog_root,
                beta,
                "full_cohort",
                "maximum_component",
                job["module_key"],
                rule,
            )
        ).sort_values(["component", "sample_id"])
        expected = pd.read_parquet(
            catalog_root
            / "scores"
            / f"beta{beta}"
            / "full_cohort/maximum_component"
            / A.module_filename(job["module_key"])
        ).sort_values(["component", "sample_id"])
        for metric in CC.METRICS:
            np.testing.assert_allclose(
                actual[metric], expected[metric], equal_nan=True, atol=2e-12
            )


def test_fdr_independent_case_and_display_invariance():
    np.testing.assert_allclose(
        bh([0.01, 0.04, 0.03, 0.2, np.nan]),
        [0.04, 0.0533333333333333, 0.0533333333333333, 0.2, np.nan],
        equal_nan=True,
    )
    base = dict(
        dataset="brain",
        cohort_scope="complete_450",
        allocation_id="B",
        outcome="x",
        method="Spearman",
        mask="all",
        subgroup="all_independent_targets",
        model="unadjusted_descriptive",
        transform="native",
    )
    f = pd.DataFrame(
        [
            dict(base, beta=b, metric_family=m, module_key=k, p=p)
            for b, m, k, p in [
                (3, "gain", "a", 0.01),
                (3, "loss", "b", 0.04),
                (1, "gain", "c", 0.03),
                (1, "net_change", "d", 0.2),
            ]
        ]
    )
    result = correct(f)
    np.testing.assert_allclose(
        result.q_scan, [0.04, 0.0533333333333, 0.0533333333333, 0.2]
    )
    np.testing.assert_allclose(result.q_GL_outcome[:2], [0.02, 0.04])
    assert result.loc[result.module_key.eq("a"), "q_scan"].iloc[0] == 0.04


def test_full_baseline_manifest_and_saved_corrections(catalog_root):
    analyze_catalog(catalog_root)
    done = json.load(open(catalog_root / "associations/completion.json"))
    assert done["status"] == "complete"
    manifest = json.load(open(catalog_root / "associations/testing_manifest.json"))
    assert (
        len(manifest["numeric_outcomes"]) == 15
        and len(manifest["nominal_outcomes"]) == 5
    )
    f = pd.read_parquet(catalog_root / "associations/brain_complete_450.parquet")
    assert set(f.beta) == {1, 3}
    assert set(f.family) == {"control_derived", "full_cohort"}
    assert len(f) == manifest["planned_structural_tests"] // 6
    assert f.loc[f.outcome.eq("clusters"), "method"].eq("Kruskal-Wallis").all()
    assert f.loc[f.outcome.eq("age_at_death"), "p"].isna().all()
    assert f.p.isna().equals(f.q_scan.isna())


def test_prediction_capabilities_and_registry_are_separate():
    from app_helpers.data import FEATURE_LABELS, PREDICTION_MODULE_SETS

    assert not set(CC.METRICS) & set(FEATURE_LABELS)
    assert PREDICTION_MODULE_SETS == ("full_cohort", "control_derived")
    for estimator in CC.ESTIMATORS:
        for capability in ("prediction", "predictive_regression", "feature_selection"):
            with pytest.raises(ValueError, match="excluded"):
                CC.require_capability(estimator, capability)
        CC.require_capability(estimator, "regression_associations")
