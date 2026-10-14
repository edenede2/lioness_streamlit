#!/usr/bin/env python3
"""Restyle saved muscle adjacency images; never rerun networks or MDC inference.

Render into a new private output first. Publication uploads only validated module
images/manifests, preserving old Drive assets and the scientific result tables.
Git commits and pushes are separate actions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path.insert(0, str(APP / "scripts"))
from build_muscle_mdc_heatmaps import DEFAULT_ASSIGNMENTS, DEFAULT_MDC_ROOT
from mdc_heatmap_palette import PALETTE
from build_brain_mdc_heatmaps import sha

DEFAULT_OUTPUT = REPO / "out/mdc_split_heatmap_palette/20261014_yellow_red"
DRIVE_ROOT = "1dTj5SkLxuDIvII5LayLViqOzjKQt-dos"
VARIANTS = ("original", "m5_high_no_q90")
CONFIGS = {
    "muscle_control_l4": {
        "count": 155, "assignments": DEFAULT_ASSIGNMENTS,
        "mdc": DEFAULT_MDC_ROOT,
        "sensitivity": REPO / "out/mdc_muscle_control_l4/20261013_no_q90_mdc",
    },
    "muscle_all_donors_l4": {
        "count": 173,
        "assignments": REPO / "results/se2_m5_muscle_all_genes/all_donors/20261012_all_donors_fitted_betas/modules_filtered_min30/M5_all_genes_table_3.csv",
        "mdc": REPO / "out/muscle_all_donors_l4/20261013_nonprediction/mdc_m5",
        "sensitivity": REPO / "out/muscle_all_donors_l4/20261013_nonprediction/mdc_sensitivity",
    },
}


def directory(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown MDC normalization variant")
    return f"yellow_red_20261014_{variant}"


def verify(source, key, variant):
    config = CONFIGS[key]
    folder = source / key / directory(variant)
    rows = pd.read_csv(folder / "manifest.tsv")
    old_folder = "all_edges" if variant == "original" else variant
    old = pd.read_csv(APP / "data" / key / "mdc_split_heatmaps" / old_folder / "manifest.tsv")
    assert len(rows) == config["count"] and rows.module.is_unique
    assert set(rows.module) == set(old.module)
    assert rows.color_palette.eq(PALETTE["name"]).all()
    matched = rows.merge(old, on="module", suffixes=("_new", "_old"), validate="one_to_one")
    np.testing.assert_array_equal(matched.n_nodes_new, matched.n_nodes_old)
    existing_caps = matched.display_q99_old.notna()
    np.testing.assert_allclose(
        matched.loc[existing_caps, "display_q99_new"],
        matched.loc[existing_caps, "display_q99_old"], rtol=1e-12, atol=0,
    )
    files = []
    for row in rows.itertuples(index=False):
        path = folder / f"M{int(row.module)}.png"
        assert path.name == row.filename and path.stat().st_size < 95 * 1024**2
        with Image.open(path) as image:
            assert image.format == "PNG" and min(image.size) >= 1200
            assert abs(image.info["dpi"][0] - 200) < 1
            image.verify()
        files.append(path)
    return files + [folder / "manifest.tsv"]


def render(source):
    for key, config in CONFIGS.items():
        for variant in VARIANTS:
            folder = source / key / directory(variant)
            if (folder / "manifest.tsv").exists():
                verify(source, key, variant)
                print(f"{key}/{variant}: validated completed palette checkpoint", flush=True)
                continue
            subprocess.run([
                sys.executable, str(APP / "scripts/build_muscle_mdc_heatmaps.py"),
                "--assignments", str(config["assignments"]),
                "--mdc-root", str(config["mdc"]),
                "--sensitivity-root", str(config["sensitivity"]),
                "--normalization-variant", variant, "--output", str(folder), "--force",
            ], cwd=APP, check=True)
            verify(source, key, variant)
    report = {
        "status": "validated", "modules": {k: c["count"] for k, c in CONFIGS.items()},
        "images": 656, "color_palette": PALETTE, "mdc_statistics_unchanged": True,
        "previous_images_preserved": True,
        "statistic_sources": {
            k: {"original": sha(c["mdc"] / "mdc_observed.parquet"),
                "sensitivity": sha(c["sensitivity"] / "mdc_observed.parquet")}
            for k, c in CONFIGS.items()
        },
    }
    (source / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def publish(source):
    from google.auth.transport.requests import AuthorizedSession
    from publish_brain_mdc_heatmaps import folder_at, sync_verified
    from upload_prediction_data import load_credentials, load_service_account

    files = {(key, variant): verify(source, key, variant)
             for key in CONFIGS for variant in VARIANTS}
    validation = json.loads((source / "validation.json").read_text())
    assert validation["status"] == "validated"
    for key, config in CONFIGS.items():
        assert sha(config["mdc"] / "mdc_observed.parquet") == validation["statistic_sources"][key]["original"]
        assert sha(config["sensitivity"] / "mdc_observed.parquet") == validation["statistic_sources"][key]["sensitivity"]
    previous = json.loads((APP / "drive_file_index.json").read_text())
    assert previous["root_folder_id"] == DRIVE_ROOT
    oauth = SimpleNamespace(
        credentials=None,
        oauth_client=APP / "client_secret_688842367267-8o25rcsq6jvrb1go8d0gsr1o304v1knq.apps.googleusercontent.com.json",
        oauth_token=APP / "google-drive-oauth-token.json", oauth_port=8765, oauth_no_browser=True,
    )
    upload = AuthorizedSession(load_credentials(oauth))
    reader = AuthorizedSession(load_service_account(
        APP / "lionessexplorer-05e37b3ea2a6.json", "https://www.googleapis.com/auth/drive.readonly"
    ))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback = source / "publication_rollback" / stamp
    rollback.mkdir(parents=True, exist_ok=False)
    shutil.copy2(APP / "drive_file_index.json", rollback / "drive_file_index.json")
    additions, prepared = {}, []
    for key in CONFIGS:
        parent = folder_at(upload, reader, DRIVE_ROOT, key)
        parent = folder_at(upload, reader, parent, "mdc_split_heatmaps")
        for variant in VARIANTS:
            name = directory(variant)
            remote = folder_at(upload, reader, parent, name)
            inventory = sync_verified(upload, reader, remote, files[key, variant], workers=4)
            additions.update({f"{key}/mdc_split_heatmaps/{name}/{filename}": item
                              for filename, item in inventory.items()})
            print(f"{key}/{variant}: {len(inventory)} images/manifest files verified by the app reader", flush=True)
        metadata = {
            "schema_version": 2, "status": "complete", "module_set": key,
            "modules": CONFIGS[key]["count"], "color_palette": PALETTE,
            "image_directories": {v: directory(v) for v in VARIANTS},
            "upper_triangle": "AD", "lower_triangle": "Control",
            "display": "Absolute Method-5 edge weight", "mdc_statistics_unchanged": True,
            "color_scale": "Shared within each module; pooled 99th-percentile display cap only",
            "released_utc": datetime.now(timezone.utc).isoformat(),
        }
        staged = rollback / key / "catalog.json"
        staged.parent.mkdir(parents=True)
        staged.write_text(json.dumps(metadata, indent=2) + "\n")
        catalog_inventory = sync_verified(upload, reader, parent, [staged])
        additions[f"{key}/mdc_split_heatmaps/catalog.json"] = catalog_inventory["catalog.json"]
        prepared.append((key, staged))
    for key, staged in prepared:
        target = APP / "data" / key / "mdc_split_heatmaps/catalog.json"
        if target.exists():
            shutil.copy2(target, rollback / f"{key}_previous_catalog.json")
        temporary = target.with_suffix(".partial")
        shutil.copy2(staged, temporary)
        temporary.replace(target)
    index = {**previous, "files": {**previous["files"], **additions},
             "generated_at_utc": datetime.now(timezone.utc).isoformat(),
             "refresh_scope": "Verified yellow-to-red muscle MDC images; all previous entries preserved"}
    index["file_count"] = len(index["files"])
    index["total_bytes"] = sum(int(item.get("size") or 0) for item in index["files"].values())
    target = APP / "drive_file_index.json"
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(index, separators=(",", ":")) + "\n")
    temporary.replace(target)
    (source / "publication_qa.json").write_text(json.dumps({
        "status": "verified_uploaded", "indexed_files": len(additions), "images": 656,
        "previous_images_preserved": True, "rollback": str(rollback),
        "privacy": "Module images only; no donor rows or private adjacency arrays uploaded",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("render", "publish"), required=True)
    args = parser.parse_args()
    source = args.output.resolve()
    source.mkdir(parents=True, exist_ok=True)
    (render if args.stage == "render" else publish)(source)


if __name__ == "__main__":
    main()
