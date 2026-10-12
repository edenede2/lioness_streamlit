#!/usr/bin/env python3
"""Render lazy split-triangle AD/Control full-M5 heatmaps for muscle modules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
DEFAULT_ASSIGNMENTS = (
    REPO_ROOT
    / "results/se2_all_genes/20260319_090116/M5_all_genes/"
    "M5_all_genes_table_3.csv"
)
DEFAULT_MDC_ROOT = REPO_ROOT / "out/mdc_muscle_control_l4/20261012_m5_ad_vs_control"


def render(
    module: int,
    ad: np.ndarray,
    control: np.ndarray,
    output: Path,
    mdc_stats: dict[str, float],
) -> dict[str, object]:
    ordering = np.argsort(-np.abs(ad).sum(axis=1), kind="stable")
    ad = ad[np.ix_(ordering, ordering)]
    control = control[np.ix_(ordering, ordering)]
    count = len(ordering)
    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    lower = np.tril(np.ones((count, count), dtype=bool), k=-1)
    combined = np.full((count, count), np.nan, dtype=np.float64)
    combined[upper] = np.abs(ad[upper])
    combined[lower] = np.abs(control[lower])
    finite = combined[np.isfinite(combined)]
    display_max = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    display_max = max(display_max, np.finfo(float).eps)

    figure, axis = plt.subplots(figsize=(8, 8), dpi=200)
    image = axis.imshow(
        combined,
        cmap="hot",
        vmin=0.0,
        vmax=display_max,
        interpolation="nearest",
        rasterized=True,
    )
    axis.plot([-0.5, count - 0.5], [-0.5, count - 0.5], color="white", lw=0.6)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Genes")
    axis.set_ylabel("Genes")
    axis.set_title(
        f"Module {module} | MDC={mdc_stats['total']:.2f} "
        f"(TS={mdc_stats['TS']:.2f}, CT={mdc_stats['CT']:.2f})\n"
        f"Upper=AD, Lower=Control | full M5 | n={count} nodes",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.035)
    colorbar.set_label("Absolute Method-5 edge weight", fontsize=10)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "module": module,
        "filename": output.name,
        "n_nodes": count,
        "display_q99": display_max,
        "upper_triangle": "AD",
        "lower_triangle": "Control",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--mdc-root", type=Path, default=DEFAULT_MDC_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=APP_ROOT / "data/muscle_control_l4/mdc_split_heatmaps/all_edges",
    )
    parser.add_argument("--module", type=int, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments = pd.read_csv(
        args.assignments,
        dtype={"Tissue": str, "Gene ID": str, "Gene Symbol": str},
    )
    sizes = assignments.groupby("Cluster ID", observed=True).size()
    assignments = assignments.loc[
        assignments["Cluster ID"].isin(sizes.loc[sizes.ge(30)].index)
    ].copy()
    modules = args.module or sorted(assignments["Cluster ID"].unique().tolist())
    module_roots = {
        group: args.mdc_root / "recovered_modules" / group / "modules"
        for group in ("ad", "control")
    }
    manifests = {
        group: json.loads(
            (args.mdc_root / "recovered_modules" / group / "manifest.json").read_text()
        )
        for group in module_roots
    }
    if any(manifest.get("status") != "complete" for manifest in manifests.values()):
        raise RuntimeError("Both AD and Control recovered-module M5 networks must be complete")
    observed = pd.read_parquet(args.mdc_root / "mdc_observed.parquet")
    mdc_lookup = {
        int(module): {
            str(row.scope): float(row.mdc_raw)
            for row in group.itertuples(index=False)
            if str(row.scope) in {"total", "TS", "CT"}
        }
        for module, group in observed.groupby("module", observed=True)
    }
    rows: list[dict[str, object]] = []
    for index, module in enumerate(modules, start=1):
        destination = args.output / f"M{module}.png"
        if destination.exists() and not args.force:
            rows.append({
                "module": module,
                "filename": destination.name,
                "n_nodes": int(assignments["Cluster ID"].eq(module).sum()),
                "display_q99": "",
                "upper_triangle": "AD",
                "lower_triangle": "Control",
            })
            continue
        ad_data = np.load(module_roots["ad"] / f"M{module}.npz", allow_pickle=False)
        control_data = np.load(
            module_roots["control"] / f"M{module}.npz", allow_pickle=False
        )
        if not np.array_equal(ad_data["tissues"], control_data["tissues"]):
            raise ValueError(f"M{module}: AD/Control node order differs")
        if not np.array_equal(ad_data["gene_ids"], control_data["gene_ids"]):
            raise ValueError(f"M{module}: AD/Control gene order differs")
        ad = ad_data["weights"]
        control = control_data["weights"]
        rows.append(render(module, ad, control, destination, mdc_lookup[int(module)]))
        if index % 10 == 0 or index == len(modules):
            print(f"Rendered {index}/{len(modules)} modules", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} heatmaps to {args.output}")


if __name__ == "__main__":
    main()
