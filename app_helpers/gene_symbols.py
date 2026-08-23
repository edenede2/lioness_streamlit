"""Convert internal Ensembl identifiers to public-facing official gene symbols."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
GENE_SYMBOL_ASSET = APP_ROOT / "assets" / "gene_symbols.tsv"
ENSEMBL_PATTERN = re.compile(r"ENSG\d+(?:\.\d+)?(?:_PAR_Y)?")
UNMAPPED_GENE_LABEL = "Unmapped gene"


@lru_cache(maxsize=1)
def _symbol_maps() -> tuple[dict[str, str], dict[str, str]]:
    mapping = pd.read_csv(GENE_SYMBOL_ASSET, sep="\t", dtype="string")
    required = {"ensembl_id", "ensembl_base", "gene_symbol"}
    missing = required.difference(mapping.columns)
    if missing:
        raise RuntimeError(f"Gene-symbol asset is missing columns: {sorted(missing)}")
    if mapping[list(required)].isna().any().any():
        raise RuntimeError("Gene-symbol asset contains missing identifiers or symbols")
    exact = dict(zip(mapping["ensembl_id"], mapping["gene_symbol"], strict=True))
    base_counts = mapping.groupby("ensembl_base", observed=True)["gene_symbol"].nunique()
    ambiguous = set(base_counts.loc[base_counts.gt(1)].index.astype(str))
    if ambiguous:
        raise RuntimeError(
            "Gene-symbol asset maps a versionless Ensembl ID to multiple symbols"
        )
    base = (
        mapping.drop_duplicates("ensembl_base")
        .set_index("ensembl_base")["gene_symbol"]
        .astype(str)
        .to_dict()
    )
    return exact, base


def gene_symbol(value: object) -> object:
    """Return an official symbol for an Ensembl value without leaking unmapped IDs."""
    if pd.isna(value):
        return value
    text = str(value)
    if not ENSEMBL_PATTERN.fullmatch(text):
        return text
    exact, base = _symbol_maps()
    return exact.get(text, base.get(text.split(".", 1)[0], UNMAPPED_GENE_LABEL))


def replace_ensembl_in_text(value: object) -> object:
    """Replace every Ensembl token embedded in a display string with its symbol."""
    if pd.isna(value):
        return value
    exact, base = _symbol_maps()

    def replacement(match: re.Match[str]) -> str:
        identifier = match.group(0)
        return exact.get(
            identifier,
            base.get(identifier.split(".", 1)[0], UNMAPPED_GENE_LABEL),
        )

    return ENSEMBL_PATTERN.sub(replacement, str(value))


def public_gene_labels(
    frame: pd.DataFrame,
    *,
    gene_columns: tuple[str, ...] = (),
    text_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a copy whose selected columns cannot expose Ensembl identifiers."""
    result = frame.copy()
    for column in gene_columns:
        if column in result:
            result[column] = result[column].map(gene_symbol)
    for column in text_columns:
        if column in result:
            result[column] = result[column].map(replace_ensembl_in_text)
    checked = [column for column in (*gene_columns, *text_columns) if column in result]
    for column in checked:
        contains_ensembl = (
            result[column]
            .dropna()
            .astype(str)
            .str.contains(ENSEMBL_PATTERN, regex=True)
            .any()
        )
        if contains_ensembl:
            raise RuntimeError(f"Internal gene identifiers remain in public column {column}")
    return result
