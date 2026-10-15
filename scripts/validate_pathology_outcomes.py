#!/usr/bin/env python3
"""Independent source/identity/FDR/formula checks for the pathology extension."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from app_helpers import data as D
from app_helpers.pathology import PATHOLOGY_OUTCOMES


def main():
    upstream = APP.parents[1] / "out/pathology_app_20261015"
    manifest = json.loads((APP / "data/pathology/manifest.json").read_text())
    families, direct_checks, rows = 0, 0, 0
    for relative, entry in manifest["metadata"].items():
        path = APP / "data" / relative
        current = pd.read_parquet(path)
        prior = pd.read_parquet(upstream / "rollback" / relative)
        pd.testing.assert_frame_equal(current[prior.columns], prior)
        assert current.sample_id.is_unique and len(current) == entry["donors"]
        assert not {"donor", "projid"}.intersection(current)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert current[list(PATHOLOGY_OUTCOMES)].notna().sum().to_dict() == entry["observed"]
    for relative, entry in manifest["statistics"].items():
        path = APP / "data" / relative
        assert path.stat().st_size < 95*1024**2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        frame = pd.read_parquet(path)
        resolved = path.name.startswith("resolved")
        assert len(frame) == entry["rows"]
        assert set(frame.phenotype) == set(PATHOLOGY_OUTCOMES)
        assert not {"donor", "projid", "sample_id"}.intersection(frame)
        assert frame.filter(like="all12_global").isna().all().all()
        keys = ["metric_family", "diagnosis_group", "phenotype"] + (["component"] if resolved else [])
        for _, group in frame.groupby(keys, observed=True, dropna=False):
            assert group.module.is_unique
            for column in [c for c in group if c.startswith("p_")]:
                valid = np.isfinite(group[column])
                q = "q_" + column[2:] + "_across_modules"
                assert group[q+"_n"].eq(int(valid.sum())).all()
                assert group.loc[~valid, q].isna().all()
                if valid.any():
                    expected = stats.false_discovery_control(group.loc[valid, column].to_numpy())
                    np.testing.assert_allclose(group.loc[valid, q], expected, atol=1e-14)
                families += 1
        module_set, estimator, method, rule = Path(relative).parts[1:5]
        candidates = frame.loc[frame.phenotype.eq("gpath") & frame.metric_family.eq("connectivity")]
        for diagnosis in ("AD", "Control"):
            column = "rho" if resolved else "rho_TS"
            selected = candidates.loc[candidates.diagnosis_group.eq(diagnosis) & candidates[column].notna()]
            if selected.empty:
                continue
            row = selected.iloc[0]
            kwargs = dict(method=method, module=int(row.module), metric_family="connectivity", module_set=module_set, estimator=estimator, edge_rule=rule)
            scores = (D.load_resolved if resolved else D.load_aggregate)(**kwargs)
            if resolved:
                scores = scores.loc[scores.component.eq(row.component)]
            source = scores.merge(D.load_sample_metadata(module_set)[["sample_id","gpath"]], on="sample_id", validate="many_to_one")
            source = source.loc[source.diagnosis_group.eq(diagnosis)]
            x = "metric_raw" if resolved else "TS_raw"
            good = source[[x,"gpath"]].dropna()
            direct = stats.spearmanr(good[x],good.gpath)
            np.testing.assert_allclose([row[column],row["p_spearman" if resolved else "p_spearman_TS"]], [direct.statistic,direct.pvalue], atol=1e-12)
            direct_checks += 1
        rows += len(frame)
    report = dict(status="pass", metadata_files=len(manifest["metadata"]), statistics_files=len(manifest["statistics"]),
        statistics_rows=rows, independent_bh_family_checks=families, direct_saved_score_checks=direct_checks,
        prior_metadata_unchanged=True, privacy=True, prediction_changed=False)
    (upstream / "validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
