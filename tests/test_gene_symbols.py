from __future__ import annotations

import re

from app_helpers import data
from app_helpers.gene_symbols import gene_symbol, replace_ensembl_in_text


ENSEMBL = re.compile(r"ENSG\d+")


def test_known_versioned_and_versionless_ids_use_official_symbols() -> None:
    assert gene_symbol("ENSG00000130203.9") == "APOE"
    assert replace_ensembl_in_text("ENSG00000095970(DLPFC)") == "TREM2(DLPFC)"


def test_volcano_endpoints_never_expose_ensembl_ids() -> None:
    for module_set, method, module in (
        ("full_cohort", "control_anchored", 935),
        ("control_derived", "control_anchored", 935),
    ):
        frame = data.load_volcano_candidates(
            module_set, "lioness", method, module
        )
        assert not frame.empty
        for column in ("gene_a", "gene_b"):
            assert not frame[column].astype(str).str.contains(ENSEMBL).any()


def test_kegg_tables_and_complete_download_use_symbols_only() -> None:
    for module_set in data.MODULE_SET_LABELS:
        frame = data.load_kegg(module_set=module_set)
        assert not frame["overlap_genes"].dropna().astype(str).str.contains(ENSEMBL).any()
        assert b"ENSG" not in data.load_kegg_tsv_bytes(module_set)
