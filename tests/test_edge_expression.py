from __future__ import annotations

import numpy as np
import pandas as pd

from app_helpers.edge_expression import (
    find_edge_triangles,
    fit_ols,
    gene_pathway_membership,
    rank_edges,
    scope_edges,
)


def example_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_index": [8, 2, 5, 9],
            "row_index": [0, 0, 1, 3],
            "column_index": [1, 2, 2, 4],
            "tissue_a": ["AC", "AC", "DLPFC", "AC"],
            "gene_a": ["A", "A", "B", "D"],
            "tissue_b": ["DLPFC", "PCG", "PCG", "DLPFC"],
            "gene_b": ["B", "C", "C", "E"],
            "component": [
                "CT_AC__DLPFC", "CT_AC__PCGBA23",
                "CT_DLPFC__PCGBA23", "CT_AC__DLPFC",
            ],
            "component_class": ["CT"] * 4,
            "discovery_hedges_g": [0.5, -0.8, 0.7, 0.1],
            "discovery_mean_difference": [0.4, -0.6, 0.3, 0.05],
            "discovery_fdr_global": [0.04, 0.01, 0.02, 0.8],
            "discovery_fdr_per_module": [0.03, 0.02, 0.01, 0.7],
        }
    )


def test_edge_ranking_scope_and_triangle_are_deterministic() -> None:
    edges = scope_edges(example_edges(), "CT")
    ranked = rank_edges(edges, "absolute_hedges_g", 3)
    assert ranked["edge_index"].tolist() == [2, 5, 8]
    triangles = find_edge_triangles(ranked)
    assert len(triangles) == 1
    assert {
        int(triangles.iloc[0]["node_x"]),
        int(triangles.iloc[0]["node_y"]),
        int(triangles.iloc[0]["node_z"]),
    } == {0, 1, 2}
    assert rank_edges(edges, "absolute_hedges_g", 3, direction="ad_higher")[
        "edge_index"
    ].tolist() == [5, 8, 9]


def test_ols_recovers_line_and_plane_and_returns_residuals() -> None:
    x = np.linspace(-2, 2, 30)
    line = pd.DataFrame({"x": x, "y": 1.5 + 2.0 * x, "sample_id": range(30)})
    fitted_line = fit_ols(line, "y", ["x"])
    assert fitted_line is not None
    assert np.isclose(fitted_line.r_squared, 1.0)
    estimates = fitted_line.coefficients.set_index("term")["estimate"]
    assert np.isclose(estimates["Intercept"], 1.5)
    assert np.isclose(estimates["x"], 2.0)
    assert np.allclose(fitted_line.diagnostics["ols_residual"], 0.0, atol=1e-12)

    z = np.tile(np.linspace(-1, 1, 6), 5)
    plane = pd.DataFrame({"x": x, "z": z})
    plane["y"] = 0.25 + 0.5 * plane["x"] - 1.25 * plane["z"]
    fitted_plane = fit_ols(plane, "y", ["x", "z"])
    assert fitted_plane is not None
    assert np.isclose(fitted_plane.r_squared, 1.0)


def test_gene_pathway_membership_keeps_unrepresented_genes() -> None:
    genes = pd.DataFrame(
        [
            {"role": "X", "gene_symbol": "APOE", "tissue": "DLPFC"},
            {"role": "Y", "gene_symbol": "MAPT", "tissue": "PCG"},
        ]
    )
    pathways = pd.DataFrame(
        [
            {
                "pathway_id": "hsa05010", "pathway_name": "Alzheimer disease",
                "category_level1": "Human Diseases",
                "category_level2": "Neurodegenerative disease",
                "p": 0.001, "fdr": 0.01, "p_DLPFC": 0.02,
                "fdr_DLPFC": 0.03, "p_PCGBA23": 0.4, "fdr_PCGBA23": 0.8,
                "overlap_genes": "APOE(DLPFC),PSEN1(AC)",
            }
        ]
    )
    observed = gene_pathway_membership(genes, pathways)
    apoe = observed.loc[observed["gene_symbol"].eq("APOE")].iloc[0]
    assert apoe["pathway_id"] == "hsa05010"
    assert np.isclose(apoe["tissue_fdr"], 0.03)
    mapt = observed.loc[observed["gene_symbol"].eq("MAPT")].iloc[0]
    assert not bool(mapt["module_enriched"])
    assert pd.isna(mapt["pathway_id"])
