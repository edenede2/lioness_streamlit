from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

from build_public_data import prepare_public_kegg  # noqa: E402


def test_public_kegg_recomputes_region_bh_over_complete_pathway_family(
    tmp_path: Path,
) -> None:
    terms = ["KEGG_a", "KEGG_b", "KEGG_c", "KEGG_d"]
    p_values = [0.001, 0.02, 0.5, 1.0]
    per_region = pd.DataFrame({"cluster_id": 1, "term": terms})
    for tissue in ("AC", "MFBA9BA46", "PCGBA23"):
        per_region[f"p_{tissue}"] = p_values
        per_region[f"fdr_{tissue}"] = [1.0, 1.0, 1.0, 1.0]
        per_region[f"significant_{tissue}"] = False
        per_region[f"overlap_{tissue}"] = [3, 3, 0, 0]

    expanded = pd.DataFrame(
        {
            "cluster_id": [1, 1],
            "term": ["KEGG_a", "KEGG_b"],
            "p": [0.001, 0.02],
            "fdr": [0.002, 0.02],
            "significant": [True, True],
            "overlap_AC": [3, 3],
            "overlap_MFBA9BA46": [3, 3],
            "overlap_PCGBA23": [3, 3],
        }
    )
    expanded_path = tmp_path / "expanded.tsv"
    regional_path = tmp_path / "regional.tsv"
    expanded.to_csv(expanded_path, sep="\t", index=False)
    per_region.to_csv(regional_path, sep="\t", index=False)

    public, audit = prepare_public_kegg(expanded_path, regional_path)
    expected = multipletests(p_values, method="fdr_bh")[1][:2]
    for tissue in ("AC", "DLPFC", "PCGBA23"):
        observed = public[f"fdr_{tissue}"].to_numpy()
        assert observed.tolist() == expected.tolist()
        region_audit = audit["per_region_fdr_recalculation"]["regions"][tissue]
        assert region_audit["source_values_changed"] == 3
        assert region_audit["family_size_min"] == 4
        assert region_audit["family_size_max"] == 4
