#!/usr/bin/env python3
"""Build the compact Ensembl-to-symbol asset used by the public app."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    app_root = Path(__file__).resolve().parents[1]
    repository_root = app_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gene-info",
        type=Path,
        default=repository_root / "data" / "external" / "gene_info.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=app_root / "assets" / "gene_symbols.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.gene_info, sep="\t", dtype="string")
    required = {"ensgene", "symbol"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Gene information is missing columns: {sorted(missing)}")
    mapping = source[["ensgene", "symbol"]].rename(
        columns={"ensgene": "ensembl_id", "symbol": "gene_symbol"}
    )
    if mapping.isna().any().any() or mapping.duplicated("ensembl_id").any():
        raise ValueError("Gene information contains missing or duplicate exact mappings")
    mapping["ensembl_base"] = mapping["ensembl_id"].str.split(".", n=1).str[0]
    conflicts = mapping.groupby("ensembl_base", observed=True)["gene_symbol"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("A versionless Ensembl identifier maps to multiple symbols")
    mapping = mapping[["ensembl_id", "ensembl_base", "gene_symbol"]].sort_values(
        ["ensembl_base", "ensembl_id"], kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(mapping):,} validated gene-symbol mappings to {args.output}")


if __name__ == "__main__":
    main()
