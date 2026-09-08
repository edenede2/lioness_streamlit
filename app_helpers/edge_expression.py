"""Small, testable helpers for endpoint-expression edge exploration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


EDGE_RANK_COLUMNS = {
    "absolute_hedges_g": ("discovery_hedges_g", False),
    "absolute_mean_difference": ("discovery_mean_difference", False),
    "global_fdr": ("discovery_fdr_global", True),
    "per_module_fdr": ("discovery_fdr_per_module", True),
}


def scope_edges(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    """Restrict exact edge rows to a pooled or resolved component."""

    if component in {"CT", "TS"}:
        return frame.loc[frame["component_class"].eq(component)].copy()
    return frame.loc[frame["component"].eq(component)].copy()


def rank_edges(
    frame: pd.DataFrame,
    rank_by: str,
    top_k: int,
    *,
    direction: str = "either",
) -> pd.DataFrame:
    """Rank a bounded exact-edge catalog with deterministic tie breaking."""

    if rank_by not in EDGE_RANK_COLUMNS:
        raise ValueError(f"Unknown edge ranking: {rank_by}")
    if direction not in {"either", "ad_higher", "control_higher"}:
        raise ValueError(f"Unknown effect direction: {direction}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = frame.drop_duplicates("edge_index", keep="first").copy()
    difference = pd.to_numeric(
        selected["discovery_mean_difference"], errors="coerce"
    )
    if direction == "ad_higher":
        selected = selected.loc[difference.gt(0)].copy()
    elif direction == "control_higher":
        selected = selected.loc[difference.lt(0)].copy()
    column, ascending = EDGE_RANK_COLUMNS[rank_by]
    values = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[values.notna()].copy()
    selected["rank_value"] = pd.to_numeric(selected[column], errors="coerce")
    if not ascending:
        selected["rank_value"] = selected["rank_value"].abs()
    selected = selected.sort_values(
        ["rank_value", "edge_index"],
        ascending=[ascending, True],
        kind="stable",
    ).head(int(top_k))
    selected.insert(0, "edge_rank", np.arange(1, len(selected) + 1, dtype=int))
    selected["edge_label"] = (
        selected["tissue_a"].astype(str)
        + ":"
        + selected["gene_a"].astype(str)
        + " ↔ "
        + selected["tissue_b"].astype(str)
        + ":"
        + selected["gene_b"].astype(str)
    )
    return selected.reset_index(drop=True)


def find_edge_triangles(edges: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    """Find genuine three-edge triangles in a ranked undirected edge set."""

    if edges.empty:
        return pd.DataFrame()
    edge_lookup: dict[tuple[int, int], pd.Series] = {}
    adjacency: dict[int, set[int]] = {}
    for row in edges.itertuples(index=False):
        left, right = sorted((int(row.row_index), int(row.column_index)))
        edge_lookup[(left, right)] = pd.Series(row._asdict())
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    rows: list[dict[str, object]] = []
    for left in sorted(adjacency):
        for middle, right in combinations(sorted(node for node in adjacency[left] if node > left), 2):
            key = tuple(sorted((middle, right)))
            if key not in edge_lookup:
                continue
            three_edges = [
                edge_lookup[tuple(sorted((left, middle)))],
                edge_lookup[tuple(sorted((left, right)))],
                edge_lookup[key],
            ]
            edge_ranks = sorted(int(edge["edge_rank"]) for edge in three_edges)
            node_labels: dict[int, str] = {}
            for edge in three_edges:
                node_labels[int(edge["row_index"])] = (
                    f"{edge['tissue_a']}:{edge['gene_a']}"
                )
                node_labels[int(edge["column_index"])] = (
                    f"{edge['tissue_b']}:{edge['gene_b']}"
                )
            rows.append(
                {
                    "node_x": left,
                    "node_y": middle,
                    "node_z": right,
                    "gene_x": node_labels[left],
                    "gene_y": node_labels[middle],
                    "gene_z": node_labels[right],
                    "best_edge_rank": min(edge_ranks),
                    "worst_edge_rank": max(edge_ranks),
                    "edge_rank_sum": sum(edge_ranks),
                    "triangle_label": (
                        f"{node_labels[left]} · {node_labels[middle]} · "
                        f"{node_labels[right]}"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["worst_edge_rank", "edge_rank_sum", "node_x", "node_y", "node_z"],
            kind="stable",
        )
        .head(int(limit))
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class OLSFit:
    """OLS estimates and donor-level fitted/residual values."""

    diagnostics: pd.DataFrame
    coefficients: pd.DataFrame
    n: int
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    residual_sd: float


def fit_ols(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    *,
    minimum_n: int | None = None,
) -> OLSFit | None:
    """Fit an intercept OLS model without adding a statsmodels dependency."""

    columns = [response, *predictors]
    clean = frame.copy()
    for column in columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=columns).copy()
    parameter_count = len(predictors) + 1
    required_n = minimum_n or parameter_count + 2
    if len(clean) < max(required_n, parameter_count + 1):
        return None
    design = np.column_stack(
        [np.ones(len(clean)), *[clean[column].to_numpy(float) for column in predictors]]
    )
    if np.linalg.matrix_rank(design) < parameter_count:
        return None
    response_values = clean[response].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(design, response_values, rcond=None)
    fitted = design @ beta
    residual = response_values - fitted
    rss = float(residual @ residual)
    centered = response_values - response_values.mean()
    tss = float(centered @ centered)
    r_squared = 1.0 - rss / tss if tss > 0 else np.nan
    degrees_freedom = len(clean) - parameter_count
    adjusted = (
        1.0 - (1.0 - r_squared) * (len(clean) - 1) / degrees_freedom
        if degrees_freedom > 0 and np.isfinite(r_squared)
        else np.nan
    )
    mse = rss / degrees_freedom if degrees_freedom > 0 else np.nan
    covariance = np.linalg.pinv(design.T @ design) * mse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    t_values = np.divide(
        beta,
        standard_errors,
        out=np.full(beta.shape, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees_freedom)
    terms = ["Intercept", *predictors]
    coefficients = pd.DataFrame(
        {
            "term": terms,
            "estimate": beta,
            "standard_error": standard_errors,
            "t_value": t_values,
            "p_value": p_values,
        }
    )
    diagnostics = clean.copy()
    diagnostics["ols_fitted"] = fitted
    diagnostics["ols_residual"] = residual
    residual_sd = float(np.std(residual, ddof=parameter_count))
    diagnostics["standardized_residual"] = (
        residual / residual_sd if residual_sd > 0 else np.nan
    )
    return OLSFit(
        diagnostics=diagnostics,
        coefficients=coefficients,
        n=len(clean),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        residual_sd=residual_sd,
    )


def gene_pathway_membership(
    genes: pd.DataFrame,
    enrichments: pd.DataFrame,
    *,
    significant_only: bool = True,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Expand selected tissue-gene endpoints into their module KEGG memberships."""

    required = {"role", "gene_symbol", "tissue"}
    missing = required.difference(genes.columns)
    if missing:
        raise ValueError(f"Selected genes lack columns: {sorted(missing)}")
    kegg = enrichments.copy()
    if significant_only and "fdr" in kegg:
        kegg = kegg.loc[
            pd.to_numeric(kegg["fdr"], errors="coerce").le(float(fdr_threshold))
        ].copy()
    rows: list[dict[str, object]] = []
    tissue_suffix = {"AC": "AC", "DLPFC": "DLPFC", "PCG": "PCGBA23"}
    for gene in genes.drop_duplicates(["role", "gene_symbol", "tissue"]).itertuples(index=False):
        matches: list[pd.Series] = []
        for _, pathway in kegg.iterrows():
            memberships = {
                token.strip().replace("(PCGBA23)", "(PCG)")
                for token in str(pathway.get("overlap_genes", "")).split(",")
                if token.strip()
            }
            if f"{gene.gene_symbol}({gene.tissue})" in memberships:
                matches.append(pathway)
        if not matches:
            rows.append(
                {
                    "role": gene.role,
                    "gene_symbol": gene.gene_symbol,
                    "tissue": gene.tissue,
                    "module_enriched": False,
                    "pathway_id": pd.NA,
                    "category": "Not represented in selected enrichment rows",
                    "sub_category": pd.NA,
                    "pathway": pd.NA,
                    "module_p_value": np.nan,
                    "module_fdr": np.nan,
                    "tissue_p_value": np.nan,
                    "tissue_fdr": np.nan,
                }
            )
            continue
        suffix = tissue_suffix.get(str(gene.tissue))
        for pathway in matches:
            rows.append(
                {
                    "role": gene.role,
                    "gene_symbol": gene.gene_symbol,
                    "tissue": gene.tissue,
                    "module_enriched": bool(
                        pd.to_numeric(pd.Series([pathway.get("fdr")]), errors="coerce").iloc[0]
                        <= float(fdr_threshold)
                    ),
                    "pathway_id": pathway.get("pathway_id"),
                    "category": pathway.get("category_level1"),
                    "sub_category": pathway.get("category_level2"),
                    "pathway": pathway.get("pathway_name"),
                    "module_p_value": pathway.get("p"),
                    "module_fdr": pathway.get("fdr"),
                    "tissue_p_value": pathway.get(f"p_{suffix}") if suffix else np.nan,
                    "tissue_fdr": pathway.get(f"fdr_{suffix}") if suffix else np.nan,
                }
            )
    return pd.DataFrame(rows)
