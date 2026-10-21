"""Pooled replication must use exactly the original across-module BH families."""
import importlib.util
from pathlib import Path
import sys
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, pearsonr, spearmanr

from app_helpers import validation_review as R

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts"))
spec = importlib.util.spec_from_file_location("build_camel_original_pooled", APP / "scripts/build_camel_original_pooled.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture_scores():
    rng = np.random.default_rng(42)
    metadata = pd.DataFrame(dict(sample_id=[f"X{i}" for i in range(24)],
        diagnosis_group=["Control"] * 8 + ["MCI"] * 8 + ["AD"] * 8,
        y=rng.normal(size=24), z=rng.normal(size=24)))
    metadata.loc[2, "y"] = np.nan
    pieces = []
    for module in (1, 2, 3, 4):
        for component in ("TS_AC", "CT_AC__DLPFC"):
            x = rng.normal(size=24) + .5 * metadata.y.fillna(0).to_numpy()
            if module == 4:
                x[:] = 1
            pieces.append(pd.DataFrame(dict(sample_id=metadata.sample_id, module=module,
                metric_family="connectivity", component=component, component_label=component,
                metric_raw=x, metric_asinh=np.arcsinh(x), metric_rint=np.arctan(x))))
    return pd.concat(pieces, ignore_index=True), metadata


def test_pooled_statistics_and_full_families_match_direct_original_calculations():
    scores, metadata = fixture_scores()
    raw = pd.concat([builder.module_statistics(f, metadata, ("y", "z"))
                     for _, f in scores.groupby("module")], ignore_index=True)
    result = builder.correct_families(raw)
    assert len(result) == 4 * 2 * 2
    assert result.diagnosis_group.eq("All donors").all()
    assert not {"sample_id", "donor", "projid"}.intersection(result)
    for row in result.itertuples():
        f = scores.loc[scores.module.eq(row.module) & scores.component.eq(row.component)]
        f = f.merge(metadata[["sample_id", row.phenotype]], on="sample_id").dropna()
        assert row.original_n == len(f)
        if row.module == 4:
            assert pd.isna(row.original_q_spearman)
            continue
        for scale in ("raw", "asinh", "rint"):
            actual = pearsonr(f[f"metric_{scale}"], f[row.phenotype])
            np.testing.assert_allclose([getattr(row, f"original_r_{scale}"), getattr(row, f"original_p_{scale}")], actual, atol=1e-12)
        np.testing.assert_allclose([row.original_rho, row.original_p_spearman],
                                  spearmanr(f.metric_raw, f[row.phenotype]), atol=1e-12)
    for _, family in result.groupby(["component", "phenotype"]):
        valid = family.original_p_spearman.notna()
        np.testing.assert_allclose(family.loc[valid, "original_q_spearman"],
            false_discovery_control(family.loc[valid, "original_p_spearman"]), atol=1e-12)
        assert family.original_family_n_spearman.eq(3).all()
        for scale in ("raw", "asinh", "rint"):
            np.testing.assert_allclose(family.loc[valid, f"original_q_{scale}"],
                false_discovery_control(family.loc[valid, f"original_p_{scale}"]), atol=1e-12)
            assert family[f"original_family_n_{scale}"].eq(3).all()


def test_overlay_only_fills_exact_pooled_rows_without_recorrecting_or_changing_groups():
    frame = pd.DataFrame(dict(module=[1, 2, 1, 2], metric_family="connectivity", component="TS_AC",
        phenotype="y", diagnosis_group=["AD", "AD", "All donors", "All donors"],
        original_n=[10., 10., np.nan, np.nan], original_rho=[.1, .2, np.nan, np.nan],
        original_q_spearman=[.8, .7, np.nan, np.nan], original_family_n_spearman=[186,186,0,0],
        original_comparison_available=[True, True, False, False], unavailable_reason="old reason",
        same_sign_spearman=[True,True,pd.NA,pd.NA], validation_rho_raw=[.2,.3,.4,.5]))
    baseline = frame.loc[frame.diagnosis_group.eq("All donors"), R.MATCH_KEYS].copy()
    baseline["original_n"] = 450
    baseline["original_rho"] = [.5, -.5]
    baseline["original_q_spearman"] = [.04, .09]
    baseline["original_family_n_spearman"] = 180
    result = R.merge_original_pooled_statistics(frame, baseline)
    pd.testing.assert_frame_equal(result.loc[result.diagnosis_group.eq("AD"), frame.columns],
                                  frame.loc[frame.diagnosis_group.eq("AD")], check_dtype=False)
    pooled = result.loc[result.diagnosis_group.eq("All donors")]
    assert pooled.original_n.eq(450).all()
    assert pooled.original_q_spearman.tolist() == [.04, .09]
    assert pooled.original_family_n_spearman.eq(180).all()
    assert pooled.original_comparison_available.all()
    assert pooled.unavailable_reason.eq("").all()
    # A selected module retains its correction from the full 180-module family.
    assert result.loc[result.module.eq(1) & result.diagnosis_group.eq("All donors"), "original_q_spearman"].item() == .04
    with pytest.raises(ValueError, match="unique"):
        R.merge_original_pooled_statistics(frame, pd.concat([baseline, baseline]))
    wrong = baseline.copy()
    wrong["diagnosis_group"] = "AD"
    with pytest.raises(ValueError, match="diagnosis-specific"):
        R.merge_original_pooled_statistics(frame, wrong)


def test_pooled_checkpoint_rejects_duplicate_donors_and_incomplete_joins():
    scores, metadata = fixture_scores()
    one = scores.loc[scores.module.eq(1)].copy()
    with pytest.raises(ValueError, match="Duplicate original donor-score"):
        builder.module_statistics(pd.concat([one, one.iloc[:1]]), metadata, ("y",))
    with pytest.raises(ValueError, match="incomplete"):
        builder.module_statistics(one, metadata.iloc[1:], ("y",))


STAGE = APP.parents[1] / "out/camel_validation/20261021_frozen164_incomplete_regions/public_bundle"


@pytest.fixture
def restored_stage(monkeypatch):
    path = STAGE / "review/manifest.json"
    if not path.is_file() or not json.loads(path.read_text()).get("original_pooled_shards"):
        pytest.skip("Original pooled summaries have not been built; never download in tests")
    monkeypatch.setenv("CAMEL_VALIDATION_DATA_ROOT", str(STAGE))
    return STAGE


def test_restored_families_hashes_privacy_and_m1116(restored_stage):
    registry = R.manifest()
    assert len(registry["original_pooled_shards"]) == 8
    for entry in registry["original_pooled_shards"]:
        path = restored_stage / "review" / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size < 95 * 1024**2
        frame = pd.read_parquet(path)
        assert len(frame) == entry["rows"] == 186 * 6 * 6 * 15
        assert not frame.duplicated(R.MATCH_KEYS).any()
        assert not {"sample_id", "donor", "projid"}.intersection(frame)
        assert frame.diagnosis_group.eq("All donors").all()
        for _, family in frame.groupby(["metric_family", "component", "phenotype"]):
            for pcol, qcol, ncol in [
                ("original_p_spearman", "original_q_spearman", "original_family_n_spearman"),
                *[(f"original_p_{s}", f"original_q_{s}", f"original_family_n_{s}") for s in builder.SCALES],
            ]:
                valid = family[pcol].notna()
                assert family[ncol].eq(int(valid.sum())).all()
                if valid.any():
                    np.testing.assert_allclose(family.loc[valid, qcol],
                        false_discovery_control(family.loc[valid, pcol]), atol=1e-12)
    rule = "ad_control_discovery_effect__hedges_g__fixed_0.40__either"
    f = R.association_table(rule, "standard_pruned", "connectivity", "cogn_global")
    hit = f.loc[f.module.eq(1116) & f.diagnosis_group.eq("All donors")]
    assert len(hit) == 6 and hit.original_n.gt(0).all()
    assert hit.original_q_asinh.notna().all() and hit.original_q_spearman.notna().all()
    before = pd.read_parquet(restored_stage / f"review/associations/{rule}/standard_pruned/connectivity/cogn_global.parquet")
    columns = list(before.columns)
    selected = before.diagnosis_group.ne("All donors")
    pd.testing.assert_frame_equal(f.loc[selected, columns], before.loc[selected, columns], check_dtype=False)


def test_m1116_matches_direct_pooled_original_donor_scores(restored_stage):
    metadata = pd.read_parquet(APP / "data/sample_metadata.parquet")
    for direction in ("ad_higher", "control_higher", "either"):
        rule = "ad_control_discovery_effect__hedges_g__fixed_0.40__" + direction
        path = APP / f"data/control_derived/effect_size/lioness/control_anchored/{rule}/all/standard_pruned/resolved_plot_data.parquet"
        scores = pd.read_parquet(path, filters=[("module", "=", 1116), ("metric_family", "=", "connectivity")],
            columns=["sample_id", "module", "component", "metric_family", "metric_raw", "metric_asinh", "metric_rint"])
        scores = scores.merge(metadata[["sample_id", "cogn_global"]], on="sample_id", validate="many_to_one")
        baseline = R.association_table(rule, "standard_pruned", "connectivity", "cogn_global")
        baseline = baseline.loc[baseline.module.eq(1116) & baseline.diagnosis_group.eq("All donors")].set_index("component")
        for component, donors in scores.groupby("component"):
            for scale in builder.SCALES:
                valid = donors[[f"metric_{scale}", "cogn_global"]].dropna()
                row = baseline.loc[component]
                assert row.original_n == len(valid)
                np.testing.assert_allclose([row[f"original_r_{scale}"], row[f"original_p_{scale}"]],
                    pearsonr(valid[f"metric_{scale}"], valid.cogn_global), atol=1e-12)
                np.testing.assert_allclose([row.original_rho, row.original_p_spearman],
                    spearmanr(valid[f"metric_{scale}"], valid.cogn_global), atol=1e-12)
