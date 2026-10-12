#!/usr/bin/env python3
"""Generate CCDF plots for a dual-beta brain-region adjacency sweep.

The source adjacency is the all-gene ROSMAP brain network generated with the
signed-alt transform. Absolute raw Pearson correlations are recovered with
``abs(adjacency) ** (1 / source_beta)`` separately for TS and CT blocks, then
re-transformed for each requested (beta_ts, beta_ct) combination.

To keep the run bounded, the script selects the same reproducible random set
of genes within each region and uses every edge in the resulting submatrices.
Every edge in the full source block has the same probability of inclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_RUN_DIR = Path(
    "/media/psylab-6028/DATA/Eden/inter-tissue-CoExpression/results/"
    "se2_raw_m1_all_donors_brain/20260901_all_donors_beta3_beta2"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/ccdf_beta_sweep")

REGION_SOURCE_NAMES = {
    "AC": "AC",
    "PCG": "PCGBA23",
    "DLPFC": "DLPFC",
}

TS_PAIRS = (
    ("AC", "AC"),
    ("PCG", "PCG"),
    ("DLPFC", "DLPFC"),
)
CT_PAIRS = (
    ("AC", "PCG"),
    ("PCG", "DLPFC"),
    ("AC", "DLPFC"),
)
ALL_PAIRS = TS_PAIRS + CT_PAIRS

PAIR_COLORS = {
    ("AC", "AC"): "#1565C0",
    ("PCG", "PCG"): "#C62828",
    ("DLPFC", "DLPFC"): "#F9A825",
    ("AC", "PCG"): "#7B1FA2",
    ("PCG", "DLPFC"): "#E65100",
    ("AC", "DLPFC"): "#2E7D32",
}

BETA_COMBINATIONS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (3, 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing the source adjacency and diagnostics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNGs, cached samples, and metadata.",
    )
    parser.add_argument(
        "--sample-genes",
        type=int,
        default=2_000,
        help="Random genes sampled independently per region (default: 2000).",
    )
    parser.add_argument(
        "--max-ct-edges",
        type=int,
        default=2_000_000,
        help="Maximum sampled edges retained for each CT block.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore an existing compatible sample cache.",
    )
    return parser.parse_args()


def pair_key(pair: tuple[str, str]) -> str:
    return "__".join(pair)


def pair_label(pair: tuple[str, str]) -> str:
    return " || ".join(pair)


def load_source_metadata(run_dir: Path) -> tuple[Path, dict, dict]:
    manifest_path = run_dir / "manifest.json"
    diagnostics_path = run_dir / "raw_m1_diagnostics.json"
    if not manifest_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError(
            f"Expected manifest.json and raw_m1_diagnostics.json in {run_dir}"
        )

    manifest = json.loads(manifest_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    adjacency_path = Path(manifest["paths"]["adjacency"])
    if not adjacency_path.exists():
        raise FileNotFoundError(adjacency_path)
    return adjacency_path, manifest, diagnostics


def read_sampled_submatrix(
    adjacency: np.ndarray,
    row_offset: tuple[int, int],
    col_offset: tuple[int, int],
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    """Read a sampled submatrix while keeping I/O sequential within rows."""
    result = np.empty((len(row_indices), len(col_indices)), dtype=np.float64)
    col_start, col_stop = col_offset

    for start in range(0, len(row_indices), chunk_rows):
        stop = min(start + chunk_rows, len(row_indices))
        rows = row_offset[0] + row_indices[start:stop]
        full_block_rows = np.asarray(adjacency[rows, col_start:col_stop])
        result[start:stop] = full_block_rows[:, col_indices]
    return result


def make_raw_abs_samples(
    adjacency_path: Path,
    diagnostics: dict,
    sample_genes: int,
    max_ct_edges: int,
    seed: int,
    chunk_rows: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, list[int]]]:
    offsets = diagnostics["offsets"]
    source_beta_ts = float(diagnostics["beta_ts"])
    source_beta_ct = float(diagnostics["beta_ct"])
    adjacency = np.load(adjacency_path, mmap_mode="r")

    expected_nodes = max(stop for _, stop in offsets.values())
    if adjacency.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f"Unexpected adjacency shape {adjacency.shape}; expected "
            f"{(expected_nodes, expected_nodes)}"
        )

    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray] = {}
    for display_name, source_name in REGION_SOURCE_NAMES.items():
        start, stop = offsets[source_name]
        available = stop - start
        if sample_genes > available:
            raise ValueError(
                f"sample_genes={sample_genes} exceeds {display_name} genes={available}"
            )
        selected[display_name] = np.sort(
            rng.choice(available, size=sample_genes, replace=False)
        )

    samples: dict[tuple[str, str], np.ndarray] = {}
    for pair in ALL_PAIRS:
        left, right = pair
        source_left = REGION_SOURCE_NAMES[left]
        source_right = REGION_SOURCE_NAMES[right]
        is_ts = left == right
        source_beta = source_beta_ts if is_ts else source_beta_ct

        transformed = read_sampled_submatrix(
            adjacency=adjacency,
            row_offset=tuple(offsets[source_left]),
            col_offset=tuple(offsets[source_right]),
            row_indices=selected[left],
            col_indices=selected[right],
            chunk_rows=chunk_rows,
        )

        if is_ts:
            transformed = transformed[np.triu_indices(sample_genes, k=1)]
        else:
            transformed = transformed.ravel()
            if len(transformed) > max_ct_edges:
                keep = rng.choice(len(transformed), size=max_ct_edges, replace=False)
                transformed = transformed[keep]

        raw_abs = np.abs(transformed) ** (1.0 / source_beta)
        if not np.isfinite(raw_abs).all():
            raise ValueError(f"Non-finite recovered correlations in {pair_label(pair)}")
        if raw_abs.min() < 0 or raw_abs.max() > 1.0 + 1e-8:
            raise ValueError(
                f"Recovered |r| outside [0, 1] in {pair_label(pair)}: "
                f"[{raw_abs.min()}, {raw_abs.max()}]"
            )

        samples[pair] = np.sort(raw_abs.astype(np.float32, copy=False))
        print(
            f"Sampled {pair_label(pair):<18} "
            f"n={len(raw_abs):,}, |r|=[{raw_abs.min():.4g}, {raw_abs.max():.4g}]",
            flush=True,
        )

    selected_json = {name: values.tolist() for name, values in selected.items()}
    return samples, selected_json


def cache_is_compatible(
    cache_meta: dict,
    adjacency_path: Path,
    sample_genes: int,
    max_ct_edges: int,
    seed: int,
) -> bool:
    expected = {
        "adjacency_path": str(adjacency_path.resolve()),
        "adjacency_size": adjacency_path.stat().st_size,
        "adjacency_mtime_ns": adjacency_path.stat().st_mtime_ns,
        "sample_genes": sample_genes,
        "max_ct_edges": max_ct_edges,
        "seed": seed,
    }
    return all(cache_meta.get(key) == value for key, value in expected.items())


def save_sample_cache(
    cache_path: Path,
    cache_meta_path: Path,
    samples: dict[tuple[str, str], np.ndarray],
    selected: dict[str, list[int]],
    adjacency_path: Path,
    sample_genes: int,
    max_ct_edges: int,
    seed: int,
) -> None:
    np.savez(cache_path, **{pair_key(pair): values for pair, values in samples.items()})
    cache_meta = {
        "adjacency_path": str(adjacency_path.resolve()),
        "adjacency_size": adjacency_path.stat().st_size,
        "adjacency_mtime_ns": adjacency_path.stat().st_mtime_ns,
        "sample_genes": sample_genes,
        "max_ct_edges": max_ct_edges,
        "seed": seed,
        "selected_local_gene_indices": selected,
    }
    cache_meta_path.write_text(json.dumps(cache_meta, indent=2) + "\n")


def load_or_create_samples(
    output_dir: Path,
    adjacency_path: Path,
    diagnostics: dict,
    sample_genes: int,
    max_ct_edges: int,
    seed: int,
    chunk_rows: int,
    force_resample: bool,
) -> tuple[dict[tuple[str, str], np.ndarray], dict]:
    cache_path = output_dir / "raw_abs_correlation_samples.npz"
    cache_meta_path = output_dir / "raw_abs_correlation_samples.json"

    if not force_resample and cache_path.exists() and cache_meta_path.exists():
        cache_meta = json.loads(cache_meta_path.read_text())
        if cache_is_compatible(
            cache_meta, adjacency_path, sample_genes, max_ct_edges, seed
        ):
            print(f"Reusing sample cache: {cache_path}", flush=True)
            with np.load(cache_path) as cached:
                samples = {
                    pair: cached[pair_key(pair)].copy() for pair in ALL_PAIRS
                }
            return samples, cache_meta

    samples, selected = make_raw_abs_samples(
        adjacency_path=adjacency_path,
        diagnostics=diagnostics,
        sample_genes=sample_genes,
        max_ct_edges=max_ct_edges,
        seed=seed,
        chunk_rows=chunk_rows,
    )
    save_sample_cache(
        cache_path=cache_path,
        cache_meta_path=cache_meta_path,
        samples=samples,
        selected=selected,
        adjacency_path=adjacency_path,
        sample_genes=sample_genes,
        max_ct_edges=max_ct_edges,
        seed=seed,
    )
    return samples, json.loads(cache_meta_path.read_text())


def ccdf_display_points(
    sorted_raw_abs: np.ndarray,
    beta: float,
    max_points: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(sorted_raw_abs)
    indices = np.unique(np.linspace(0, n - 1, min(n, max_points), dtype=np.int64))
    x = sorted_raw_abs[indices].astype(np.float64) ** beta
    y = 1.0 - (indices + 1) / n
    return x, y


def add_curves(
    ax: plt.Axes,
    samples: dict[tuple[str, str], np.ndarray],
    beta_ts: float,
    beta_ct: float,
    compact: bool = False,
) -> None:
    for pair in ALL_PAIRS:
        is_ts = pair[0] == pair[1]
        beta = beta_ts if is_ts else beta_ct
        x, y = ccdf_display_points(samples[pair], beta)
        ax.plot(
            x,
            y,
            color=PAIR_COLORS[pair],
            linewidth=2.2 if is_ts else 1.8,
            linestyle="-" if is_ts else "--",
            label=pair_label(pair),
            alpha=0.92,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#B8BEC6", alpha=0.30, linewidth=0.8)
    ax.set_xlabel(r"$|w| = |r|^{\beta}$", fontsize=12 if compact else 16)
    ax.set_ylabel(r"$P(|W| > x)$  —  CCDF", fontsize=12 if compact else 16)
    ax.tick_params(labelsize=10 if compact else 14)


def legend_entries() -> tuple[list[Line2D], list[str]]:
    ts_header = Line2D([0], [0], color="none")
    ct_header = Line2D([0], [0], color="none")
    handles: list[Line2D] = [ts_header]
    labels = ["— Tissue-specific (TS):"]

    for pair in TS_PAIRS:
        handles.append(Line2D([0], [0], color=PAIR_COLORS[pair], linewidth=2.2))
        labels.append(pair_label(pair))

    handles.append(ct_header)
    labels.append("– – Cross-tissue (CT):")
    for pair in CT_PAIRS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PAIR_COLORS[pair],
                linewidth=1.8,
                linestyle="--",
            )
        )
        labels.append(pair_label(pair))
    return handles, labels


def save_individual_plot(
    output_path: Path,
    samples: dict[tuple[str, str], np.ndarray],
    beta_ts: int,
    beta_ct: int,
    sample_genes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    add_curves(ax, samples, beta_ts, beta_ct)
    ax.set_title(
        "CCDF of Absolute Brain-Region Edge Weights\n"
        rf"$\beta_{{TS}}={beta_ts}$, $\beta_{{CT}}={beta_ct}$",
        fontsize=16,
        pad=12,
    )
    handles, labels = legend_entries()
    ax.legend(
        handles,
        labels,
        fontsize=12,
        loc="upper right",
        framealpha=0.94,
        handlelength=2.0,
        borderpad=0.7,
    )
    fig.text(
        0.99,
        0.012,
        f"Empirical all-gene network; reproducible {sample_genes:,}-gene sample per region",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#59636E",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_combined_plot(
    output_path: Path,
    samples: dict[tuple[str, str], np.ndarray],
    sample_genes: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    for ax, (beta_ts, beta_ct) in zip(axes.flat, BETA_COMBINATIONS):
        add_curves(ax, samples, beta_ts, beta_ct, compact=True)
        ax.set_title(
            rf"$\beta_{{TS}}={beta_ts}$, $\beta_{{CT}}={beta_ct}$",
            fontsize=14,
            pad=8,
        )

    fig.suptitle("CCDF Sensitivity to TS and CT Soft-Threshold Powers", fontsize=19)
    handles, labels = legend_entries()
    fig.legend(
        handles,
        labels,
        fontsize=11.5,
        loc="center right",
        bbox_to_anchor=(0.995, 0.52),
        framealpha=0.95,
        handlelength=2.0,
        borderpad=0.7,
    )
    fig.text(
        0.995,
        0.014,
        f"Empirical all-gene network; reproducible {sample_genes:,}-gene sample per region",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#59636E",
    )
    fig.tight_layout(rect=(0, 0.035, 0.82, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adjacency_path, manifest, diagnostics = load_source_metadata(args.run_dir)
    samples, cache_meta = load_or_create_samples(
        output_dir=args.output_dir,
        adjacency_path=adjacency_path,
        diagnostics=diagnostics,
        sample_genes=args.sample_genes,
        max_ct_edges=args.max_ct_edges,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
        force_resample=args.force_resample,
    )

    output_files: list[str] = []
    for beta_ts, beta_ct in BETA_COMBINATIONS:
        output_path = args.output_dir / (
            f"Picture1_CCDF_betaTS{beta_ts}_betaCT{beta_ct}.png"
        )
        save_individual_plot(
            output_path=output_path,
            samples=samples,
            beta_ts=beta_ts,
            beta_ct=beta_ct,
            sample_genes=args.sample_genes,
        )
        output_files.append(str(output_path.resolve()))
        print(f"Saved {output_path}", flush=True)

    combined_path = args.output_dir / "Picture1_CCDF_beta_sweep_combined.png"
    save_combined_plot(combined_path, samples, args.sample_genes)
    output_files.append(str(combined_path.resolve()))
    print(f"Saved {combined_path}", flush=True)

    artifact_metadata = {
        "description": "CCDF beta sweep for AC, PCG, and DLPFC brain regions",
        "beta_combinations": [
            {"beta_ts": beta_ts, "beta_ct": beta_ct}
            for beta_ts, beta_ct in BETA_COMBINATIONS
        ],
        "source": {
            "run_dir": str(args.run_dir.resolve()),
            "adjacency_path": str(adjacency_path.resolve()),
            "source_beta_ts": diagnostics["beta_ts"],
            "source_beta_ct": diagnostics["beta_ct"],
            "source_shape": manifest["adjacency"]["shape"],
            "genes_per_region": manifest["counts"]["genes_per_region"],
            "cohort_rule": diagnostics["cohort_rule"],
        },
        "sampling": {
            "method": (
                "Independent simple random sample of genes per region; all TS "
                "upper-triangle edges and up to max_ct_edges CT edges retained"
            ),
            "sample_genes_per_region": args.sample_genes,
            "max_ct_edges": args.max_ct_edges,
            "seed": args.seed,
            "selected_local_gene_indices": cache_meta[
                "selected_local_gene_indices"
            ],
            "edge_counts": {
                pair_label(pair): int(len(values)) for pair, values in samples.items()
            },
        },
        "region_labels": {
            "AC": "AC",
            "PCG": "PCGBA23",
            "DLPFC": "DLPFC (corrected from the legacy MF label)",
        },
        "outputs": output_files,
    }
    metadata_path = args.output_dir / "artifact_metadata.json"
    metadata_path.write_text(json.dumps(artifact_metadata, indent=2) + "\n")
    print(f"Saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
