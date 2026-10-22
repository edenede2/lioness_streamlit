"""Strict read-only source adapters for the offline screening experiment."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
import secrets
from pathlib import Path
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
from app_helpers.data import NUMERIC_OUTCOMES

OUTCOMES = tuple(NUMERIC_OUTCOMES)
BRAINS = ("AC", "DLPFC", "PCG")
MOTORS = ("VH", "SMA", "QUAD")
PARTITIONS = {
    "full_cohort": (154, "data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_table_filtered_4.csv", ("standard", "control_anchored")),
    "control_derived": (186, "data/proccessed/se2_signed_alt_control_filtered/se2_rosmap_control_signed-alt/speakeasy_clusters_table_level_4_filtered.csv", ("control_anchored",)),
    "muscle_control_l4": (155, "results/se2_all_genes/20260319_090116/M5_all_genes/M5_all_genes_table_3.csv", ("control_anchored",)),
    "muscle_all_donors_l4": (173, "results/se2_m5_muscle_all_genes/all_donors/20261012_all_donors_fitted_betas/modules_filtered_min30/M5_all_genes_table_3.csv", ("control_anchored",)),
}


def canon(value):
    text = str(value).strip().replace("ROSMAP-", "")
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError("Invalid private expression identifier")
    return "ROSMAP-"+text.zfill(8)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Sources:
    dataset: str
    expression: dict[str, pd.DataFrame]
    original: list[str]
    validation: list[str]
    controls: list[str]
    metadata: pd.DataFrame  # Private donor index; never exported to public.
    mapping: dict[str, str]
    paths: tuple[Path, ...]
    excluded_complete_n: int

    def donors(self, tissues, validation=False):
        allowed = set(self.validation if validation else self.original)
        return sorted(allowed.intersection(*(set(self.expression[t].index) for t in tissues)))


def checked_expression(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(canon)
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError("Duplicate raw expression keys must be reconciled")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Processed expression must be finite; no imputation allowed")
    return frame


def collapse_metadata(frame, key, columns):
    if frame.groupby(key)[columns].nunique(dropna=True).gt(1).any().any():
        raise ValueError("Conflicting source metadata joins")
    return frame.groupby(key)[columns].first()


@lru_cache(maxsize=2)
def load_sources(dataset):
    if len(OUTCOMES) != 15:
        raise ValueError("Numeric outcome registry changed; audit the experiment specification")
    if dataset == "brain":
        paths = tuple(REPO/f"data/raw/ROSMAP_brain/ROSMAP_fixed_{t}.csv" for t in ("AC", "DLPFC", "PCGBA23"))
        expression = {t: checked_expression(p) for t, p in zip(BRAINS, paths, strict=True)}
        private = REPO/"out/lioness_reference_comparison_rosmap/20260817_standard_control_anchored_allmodules_5phenotypes_6features/standard/data/phenotypes.parquet"
        old = pd.read_parquet(private, columns=["donor", "diagnosis_group"])
        original = sorted(old.donor.map(canon))
        complete = set.intersection(*(set(x.index) for x in expression.values()))
        validation = sorted(set.union(*(set(x.index) for x in expression.values()))-complete)
        map_path = REPO/"out/lioness_regional_modules_rosmap/20261002_independent_region_app/private_regional_public_id_mapping.json"
        mapping = {canon(k): v for k, v in json.loads(map_path.read_text()).items()}
        meta_paths = tuple(APP/f"data/regional_modules/metadata/maximum_tissue/{t}.parquet" for t in BRAINS)
        meta = pd.concat([pd.read_parquet(p) for p in meta_paths], ignore_index=True)
        meta = collapse_metadata(meta, "sample_id", [*OUTCOMES, "diagnosis_group"])
        inverse = {v: k for k, v in mapping.items()}
        if not set(meta.index) <= inverse.keys():
            raise ValueError("Use the existing private mapping, never phenotype fingerprint recovery")
        meta.index = meta.index.map(inverse)
        public_original = pd.read_parquet(APP/"data/sample_metadata.parquet").set_index("sample_id")
        a = meta.loc[original, list(OUTCOMES)].to_numpy(float)
        b = public_original.loc[[mapping[d] for d in original], list(OUTCOMES)].to_numpy(float)
        # The old and regional builders differ by <1e-16 for a handful of CSV
        # round trips. This is not a change in phenotype definition or scale.
        if not np.allclose(a, b, rtol=0, atol=1e-14, equal_nan=True):
            raise ValueError("Brain outcomes disagree with existing app values")
        expected = (450, 464, 793, 164)
        paths += (private, map_path, *meta_paths)
    elif dataset == "motor":
        paths = tuple(REPO/f"data/raw/ROSMAP_muscle_split/ROSMAP_tissue_{t.lower()}_{g}.csv" for t in MOTORS for g in ("Control", "MCI", "AD"))
        expression = {}
        diagnoses = {}
        for ti, tissue in enumerate(MOTORS):
            frames = []
            for gi, group in enumerate(("Control", "MCI", "AD")):
                frame = checked_expression(paths[ti*3+gi])
                if frames and not frame.columns.equals(frames[0].columns):
                    raise ValueError("Diagnosis-specific expression headers differ")
                for donor in frame.index:
                    if diagnoses.setdefault(donor, group) != group:
                        raise ValueError("Conflicting expression diagnosis membership")
                frames.append(frame)
            expression[tissue] = pd.concat(frames, verify_integrity=True)
        complete = set.intersection(*(set(x.index) for x in expression.values()))
        original = sorted(complete)
        validation = sorted(set.union(*(set(x.index) for x in expression.values()))-complete)
        map_path = REPO/"out/muscle_control_l4/private_public_sample_mapping.tsv"
        saved_map = pd.read_csv(map_path, sep="\t", dtype=str)
        mapping = dict(zip(saved_map.donor.map(canon), saved_map.sample_id, strict=True))
        # Extension is run-private. Never alter or infer the deployed mapping.
        extra_path = REPO/"out/gene_edge_outcome_screen/private_motor_mapping_extension.json"
        extras = json.loads(extra_path.read_text()) if extra_path.exists() else {}
        used = set(mapping.values()) | set(extras.values())
        changed = False
        for donor in sorted(set(original+validation)-mapping.keys()-extras.keys()):
            candidate = "E-"+secrets.token_hex(8)
            while candidate in used:
                candidate = "E-"+secrets.token_hex(8)
            extras[donor] = candidate; used.add(candidate); changed = True
        if changed:
            extra_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = extra_path.with_suffix(".partial")
            fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(extras, handle, sort_keys=True)
            temporary.replace(extra_path)
        if mapping.keys() & extras.keys():
            raise ValueError("Private extension must not override existing sample IDs")
        mapping.update(extras)
        public_path = APP/"data/muscle_control_l4/sample_metadata.parquet"
        public = pd.read_parquet(public_path).set_index("sample_id")
        inverse = {v: k for k, v in mapping.items()}
        public.index = public.index.map(inverse)
        clinical_path = REPO/"data/external/muscle_samples_with_clinical_groups_by_IID_cogdx_RADC.csv"
        clinical = pd.read_csv(clinical_path, low_memory=False)
        clinical["donor"] = clinical.ProjID.map(canon)
        fields = {"age_death": "age_at_death", "educ": "education_years", "braaksc": "braak_stage", "ceradsc": "cerad_score", "parkinsonism_yn_lv": "parkinsonism"}
        clinical = clinical.rename(columns=fields)
        available = [o for o in OUTCOMES if o not in {"gpath", "amyloid", "tangles"}]
        meta = collapse_metadata(clinical, "donor", available)
        from scripts.build_pathology_outcomes import load_source
        pathology = load_source()
        meta = meta.join(pathology, how="outer")
        meta["diagnosis_group"] = pd.Series(diagnoses)
        # Reuse precisely the already published 357-donor phenotype values.
        for outcome in OUTCOMES:
            left = meta.reindex(original)[outcome].to_numpy(float)
            right = public.reindex(original)[outcome].to_numpy(float)
            if not np.allclose(left, right, equal_nan=True, rtol=0, atol=0):
                raise ValueError(f"Motor source disagrees with existing {outcome}; do not overwrite")
        meta.loc[original, list(OUTCOMES)] = public.loc[original, list(OUTCOMES)].to_numpy()
        expected = (357, 357, 142, 120)
        from scripts.build_pathology_outcomes import SOURCE
        paths += (map_path, extra_path, public_path, clinical_path, SOURCE)
    else:
        raise ValueError("Unknown screening dataset")
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("Pseudonym collision")
    if not set(original+validation) <= mapping.keys() or not set(original+validation) <= set(meta.index):
        raise ValueError("Source metadata/mapping coverage incomplete")
    controls = [d for d in original if meta.loc[d, "diagnosis_group"] == "Control"]
    if (len(original), len(complete), len(validation), len(controls)) != expected:
        raise ValueError("Source cohort counts differ from frozen specification")
    if set(original) & set(validation):
        raise ValueError("Discovery and validation overlap")
    if dataset == "brain":
        for row in old.itertuples():
            if meta.loc[canon(row.donor), "diagnosis_group"] != row.diagnosis_group:
                raise ValueError("Original brain diagnosis differs across sources")
    return Sources(dataset, expression, original, validation, controls, meta, mapping, paths, len(complete))


def assignments(partition):
    count, relative, methods = PARTITIONS[partition]
    frame = pd.read_csv(REPO/relative)
    frame["module"] = pd.to_numeric(frame["Cluster ID"], errors="raise").astype(int)
    frame["tissue"] = frame.Tissue.replace({"MFBA9BA46": "DLPFC", "PCGBA23": "PCG"})
    frame["gene"] = frame["Gene Symbol"].astype(str)
    if partition == "muscle_control_l4":
        # The source is the unfiltered SE2 table; reproduce the app's existing
        # minimum-30 source-row selection BEFORE unique-node resolution.
        sizes = frame.groupby("module").size()
        frame = frame.loc[frame.module.isin(sizes[sizes >= 30].index)].copy()
    if frame.module.nunique() != count:
        raise ValueError("Module count changed")
    conflicts = frame.groupby(["tissue", "gene"]).module.nunique()
    if conflicts.gt(1).any():
        raise ValueError("Physical node assigned to multiple modules")
    sizes = frame.groupby("module").size()
    frame = frame.drop_duplicates(["module", "tissue", "gene"]).copy()
    frame["source_node_count"] = frame.module.map(sizes)
    frame["unique_node_count"] = frame.module.map(frame.groupby("module").size())
    if frame.module.nunique() != count:
        raise ValueError("Do not drop modules after duplicate resolution")
    return frame.sort_values(["module", "tissue", "gene"], kind="stable").reset_index(drop=True)


def component(t1, t2):
    return f"TS_{t1}" if t1 == t2 else f"CT_{t1}__{t2}"
