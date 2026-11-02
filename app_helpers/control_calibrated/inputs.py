"""Authoritative partition/input/identity resolution for the frozen B protocol.

Only this offline layer reads expression and private identities. App loaders
read public pseudonymous summaries and never import private audit code.
"""

from pathlib import Path
import hashlib, json, uuid, re
import numpy as np
import pandas as pd
from .registry import APP, ROOT

REPO = APP.parents[1]
TISSUES = {"brain": ("AC", "DLPFC", "PCG"), "motor": ("VH", "SMA", "QUAD")}
ASSIGNMENTS = {
    "full_cohort": "data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_table_filtered_4.csv",
    "control_derived": "data/proccessed/se2_signed_alt_control_filtered/se2_rosmap_control_signed-alt/speakeasy_clusters_table_level_4_filtered.csv",
    "muscle_control_l4": "results/se2_all_genes/20260319_090116/M5_all_genes/M5_all_genes_table_3.csv",
    "muscle_all_donors_l4": "results/se2_m5_muscle_all_genes/all_donors/20261012_all_donors_fitted_betas/modules_filtered_min30/M5_all_genes_table_3.csv",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def canon(value):
    value = str(value).replace("ROSMAP-", "").removesuffix(".0").strip()
    if not value.isdigit():
        raise ValueError("Private donor identifier does not match source namespace")
    return "ROSMAP-" + value.zfill(8)


def tissue(value):
    return {"MFBA9BA46": "DLPFC", "PCGBA23": "PCG"}.get(str(value), str(value))


def gene(value):
    value = str(value)
    return value.split(".")[0] if re.fullmatch(r"ENSG[0-9]+\.[0-9]+", value) else value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
        + "\n"
    )
    tmp.replace(path)


def atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def resolve_catalog():
    main = json.loads((APP / "data/data_manifest.json").read_text())
    entries = main.get("module_sets", {})
    catalog = []
    families = {}
    sources = []
    for family, info in entries.items():
        if info.get("status", "complete") != "complete":
            continue
        if family not in ASSIGNMENTS:
            raise ValueError(
                f"Registered partition lacks explicit input resolver: {family}"
            )
        path = REPO / ASSIGNMENTS[family]
        a = pd.read_csv(path)
        a["module"] = a["Cluster ID"].astype(int)
        if family.startswith("muscle"):
            a = a[a.module.map(a.groupby("module").size()).ge(30)].copy()
        a["tissue"] = a.Tissue.map(tissue)
        a["gene"] = a["Gene Symbol"].map(gene)
        detail = (
            APP
            / "data"
            / ("" if family == "full_cohort" else family)
            / "module_details.tsv"
        )
        details = pd.read_csv(detail, sep="\t")
        expected = set(details["module"].astype(int))
        if set(a.module) != expected or len(expected) != int(info["modules"]):
            raise ValueError(f"Current registry/assignment mismatch for {family}")
        dataset = "motor" if family.startswith("muscle") else "brain"
        families[family] = dict(
            modules=len(expected),
            dataset=dataset,
            regional=False,
            label=family,
            assignment_sha256=sha(path),
        )
        sources.append(path)
        for module, b in a.groupby("module", sort=True):
            unique = b.drop_duplicates(["tissue", "gene"])
            catalog.append(
                dict(
                    family=family,
                    module=int(module),
                    module_key=f"{family}::M{module}",
                    dataset=dataset,
                    input_variant=dataset,
                    regional_tissue=None,
                    nodes=list(map(list, zip(unique.tissue, unique.gene))),
                    source_rows=len(b),
                    physical_nodes=len(unique),
                    assignment_sha256=sha(path),
                    discovery="fixed legacy partition; scoring beta is separate; target/discovery overlap remains",
                )
            )
    regional = json.loads((APP / "data/regional_modules/manifest.json").read_text())
    details = pd.read_parquet(APP / "data/regional_modules/module_details.parquet")
    for family in regional["sources"]:
        dataset = "motor" if "motor" in family else "brain"
        rows = details[details.partition_source.eq(family)]
        families[family] = dict(
            modules=len(rows),
            dataset=dataset,
            regional=True,
            label=regional["source_labels"][family],
        )
        for t in sorted(rows.tissue.unique()):
            if dataset == "motor":
                path = (
                    REPO
                    / f"results/se2_rosmap_single_motor_tissues_signed_alt_complete_case/20261014_161640/{t}/se2_{t.lower()}_signed_alt_table_3.csv"
                )
            else:
                candidates = [
                    REPO / p
                    for p in regional["input_sha256"]
                    if p.endswith("_table_3.csv")
                    and f'/{"PCGBA23" if t=="PCG" else t}/' in p
                    and ("complete_case" in p) == ("complete_case" in family)
                ]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous regional assignment source")
                path = candidates[0]
            a = pd.read_csv(path)
            a["module"] = a["Cluster ID"].astype(int)
            a["gene"] = a["Gene ID"].map(gene)
            expected = set(rows.loc[rows.tissue.eq(t), "module"].astype(int))
            if set(a.module) != expected:
                raise ValueError(f"Regional assignment/detail mismatch {family}/{t}")
            sources.append(path)
            for module, b in a.groupby("module", sort=True):
                nodes = [[t, g] for g in b.gene.drop_duplicates()]
                catalog.append(
                    dict(
                        family=family,
                        module=int(module),
                        module_key=f"{family}::{t}::M{module}",
                        dataset=dataset,
                        input_variant=(
                            "motor_regional" if dataset == "motor" else dataset
                        ),
                        regional_tissue=t,
                        nodes=nodes,
                        source_rows=len(b),
                        physical_nodes=len(nodes),
                        assignment_sha256=sha(path),
                        discovery="fixed independent regional legacy partition; new scoring functional; historical discovery selection remains",
                    )
                )
    return catalog, families, sources


def frozen_roles(baseline):
    baseline = Path(baseline)
    config = json.loads((baseline / "frozen_configuration.json").read_text())
    assert config["allocation_seeds"][0] == 611010
    a = np.load(baseline / "private_prepared_inputs.npz")
    receipt = {
        x["file"]: x
        for x in json.loads((baseline / "file_receipts.json").read_text())["files"]
    }
    if (
        sha(baseline / "private_prepared_inputs.npz")
        != receipt["private_prepared_inputs.npz"]["sha256"]
    ):
        raise ValueError("Frozen B role input hash changed")
    original_brain = (
        pd.read_parquet(
            REPO
            / "out/lioness_reference_comparison_rosmap/20260817_standard_control_anchored_allmodules_5phenotypes_6features/standard/data/phenotypes.parquet"
        )
        .donor.map(canon)
        .tolist()
    )
    mm = pd.read_csv(
        REPO / "out/muscle_control_l4/private_public_sample_mapping.tsv",
        sep="\t",
        dtype=str,
    )
    pub = pd.read_parquet(APP / "data/muscle_control_l4/sample_metadata.parquet")
    mm = mm.merge(pub[["sample_id"]], on="sample_id", validate="one_to_one")
    original_motor = sorted(mm.donor.map(canon))
    counts = {"brain": (82, 41, 327), "motor": (60, 30, 267)}
    result = {}
    for ds, ids in [("brain", original_brain), ("motor", original_motor)]:
        result[ds] = {"original": ids}
        for role, indices in [
            ("R", a[ds + "_R_alloc"][0]),
            ("C", a[ds + "_C_alloc"][0]),
            ("T", a[ds + "_T"]),
        ]:
            result[ds][role] = [ids[int(i)] for i in indices]
            result[ds][role + "_ordinal_sha256"] = hashlib.sha256(
                indices.tobytes()
            ).hexdigest()
        assert tuple(len(result[ds][k]) for k in ("R", "C", "T")) == counts[ds]
        assert not set(result[ds]["R"]) & set(result[ds]["C"]) and not set(
            result[ds]["T"]
        ) & (set(result[ds]["R"]) | set(result[ds]["C"]))
    witness = json.loads(
        (APP / "artifacts/private_tail_core_20261102/frozen_config.json").read_text()
    )["roles"]
    for ds in result:
        for role in ("R", "C", "T"):
            if result[ds][role + "_ordinal_sha256"] != witness[ds][role]["sha256"]:
                raise ValueError("Frozen exact membership hash mismatch")
    return result


def public_mapping():
    brain = {
        canon(k): v
        for k, v in json.loads(
            (
                REPO
                / "out/lioness_regional_modules_rosmap/20261002_independent_region_app/private_regional_public_id_mapping.json"
            ).read_text()
        ).items()
    }
    m = pd.read_csv(
        REPO / "out/muscle_control_l4/private_public_sample_mapping.tsv",
        sep="\t",
        dtype=str,
    )
    motor = dict(zip(m.donor.map(canon), m.sample_id))
    for values in (brain, motor):
        if len(values) != len(set(values.values())):
            raise ValueError("Public identity mapping is not one-to-one")
    return {"brain": brain, "motor": motor}


def motor_metadata(ids, mapping, diagnoses):
    from scripts.build_pathology_outcomes import augment_metadata
    from scripts.build_study_metadata import augment_metadata as study_augment

    source = pd.read_csv(
        REPO
        / "data/external/muscle_samples_with_clinical_groups_by_IID_cogdx_RADC.csv",
        low_memory=False,
    )
    source["id"] = source.ProjID.map(canon)
    fields = {
        "cogn_global": "cogn_global",
        "cogng_demog_slope": "cogng_demog_slope",
        "cogng_path_slope": "cogng_path_slope",
        "motor10_demog_slope": "motor10_demog_slope",
        "sqrt_parksc_demog_slope": "sqrt_parksc_demog_slope",
        "age_death": "age_at_death",
        "educ": "education_years",
        "cogdx": "cogdx",
        "braaksc": "braak_stage",
        "ceradsc": "cerad_score",
        "adnc": "adnc",
        "parkinsonism_yn_lv": "parkinsonism",
        "msex": "sex_code",
        "apoe_genotype": "apoe_genotype",
        "clusters": "clusters",
    }
    source = source[source.id.isin(ids)]
    for field in fields:
        if source.groupby("id")[field].nunique(dropna=True).gt(1).any():
            raise ValueError("Conflicting motor clinical values: " + field)
    clinical = source.groupby("id").clinical_group.first().to_dict()
    f = (
        source.drop_duplicates("id")
        .set_index("id")[list(fields)]
        .rename(columns=fields)
        .reindex(ids)
        .reset_index(drop=True)
    )
    f.insert(0, "sample_id", [mapping[i] for i in ids])
    f["diagnosis_group"] = [
        diagnoses.get(i, clinical.get(i, "Unclassified")) for i in ids
    ]
    f = augment_metadata(f, ids)
    f = study_augment(f, ids)
    f["sex_code"] = f.sex_code.map({0: "Code 0", 1: "Code 1"}).astype("string")
    f["apoe_genotype"] = f.apoe_genotype.map(
        lambda v: pd.NA if pd.isna(v) else "ε" + "/ε".join(str(int(float(v))))
    ).astype("string")
    return f


def prepare(root=ROOT, baseline=None):
    root = Path(root)
    private = root / "private"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    if baseline is None:
        baseline = APP / "artifacts/private_calibration_followup_20261101"
    catalog, families, sources = resolve_catalog()
    roles = frozen_roles(baseline)
    mapping = public_mapping()
    datasets = {}
    diagnoses = {}
    receipts = []
    for variant in sorted({j["input_variant"] for j in catalog}):
        ds = "motor" if variant.startswith("motor") else "brain"
        ts = TISSUES[ds]
        members = [j for j in catalog if j["input_variant"] == variant]
        nodes = sorted({tuple(n) for j in members for n in j["nodes"]})
        node_index = {n: i for i, n in enumerate(nodes)}
        frames = {}
        all_presence = {}
        if variant == "motor_regional":
            manifest_path = (
                REPO
                / "results/se2_rosmap_single_motor_tissues_signed_alt_complete_case/20261014_161640/manifest.json"
            )
            regional = json.loads(manifest_path.read_text())
            paths = {t: [Path(regional["inputs"][t]["path"])] for t in ts}
        elif ds == "brain":
            paths = {
                t: [
                    REPO
                    / f'data/raw/ROSMAP_brain/ROSMAP_fixed_{"PCGBA23" if t=="PCG" else t}.csv'
                ]
                for t in ts
            }
        else:
            paths = {
                t: [
                    REPO
                    / f"data/raw/ROSMAP_muscle_split/ROSMAP_tissue_{t.lower()}_{g}.csv"
                    for g in ("Control", "MCI", "AD")
                ]
                for t in ts
            }
        for t in ts:
            parts = []
            for path in paths[t]:
                frame = pd.read_csv(path, index_col=0)
                frame.index = frame.index.map(canon)
                frame.columns = frame.columns.map(gene)
                if frame.index.duplicated().any() or frame.columns.duplicated().any():
                    raise ValueError("Source donor/gene mapping is not unique")
                receipts.append(
                    dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                )
                parts.append(frame)
                if ds == "motor" and variant == "motor":
                    group = path.stem.rsplit("_", 1)[-1]
                    for i in frame.index:
                        if i in diagnoses and diagnoses[i] != group:
                            raise ValueError("Motor diagnosis conflict across tissues")
                        diagnoses[i] = group
            f = pd.concat(parts)
            all_presence[t] = set(f.index[np.isfinite(f.to_numpy(float)).any(1)])
            needed = [g for tt, g in nodes if tt == t]
            if not set(needed) <= set(f.columns):
                raise ValueError(f"{variant}/{t}: exact gene inputs missing")
            frames[t] = f[needed]
        ids = sorted(set.union(*(set(f.index) for f in frames.values())))
        if not set(roles[ds]["R"] + roles[ds]["C"] + roles[ds]["T"]) <= set(ids):
            raise ValueError("Frozen protocol donor absent from source variant")
        for i in ids:
            if i not in mapping[ds]:
                mapping[ds][i] = ("R-" if ds == "brain" else "M-") + uuid.uuid4().hex[
                    :16
                ]
        presence = np.column_stack([[i in all_presence[t] for i in ids] for t in ts])
        lookup = {i: q for q, i in enumerate(ids)}
        shape = (len(ids), len(nodes))
        x = np.lib.format.open_memmap(
            private / f"{variant}_expression.npy",
            mode="w+",
            dtype=np.float64,
            shape=shape,
        )
        x[:] = np.nan
        for t, f in frames.items():
            cols = [node_index[(t, g)] for g in f.columns]
            rows = [lookup[i] for i in f.index]
            x[np.ix_(rows, cols)] = f.to_numpy(float)
        x.flush()
        del x, frames
        np.save(private / f"{variant}_presence.npy", presence)
        if ds == "brain":
            pieces = [
                pd.read_parquet(
                    APP / f"data/regional_modules/metadata/maximum_tissue/{t}.parquet"
                )
                for t in ts
            ]
            meta = (
                pd.concat(pieces, ignore_index=True)
                .groupby("sample_id", as_index=False)
                .first()
            )
            meta = (
                meta.set_index("sample_id")
                .reindex([mapping[ds][i] for i in ids])
                .reset_index()
            )
        else:
            meta = motor_metadata(ids, mapping[ds], diagnoses)
        original = pd.read_parquet(
            APP
            / (
                "data/sample_metadata.parquet"
                if ds == "brain"
                else "data/muscle_control_l4/sample_metadata.parquet"
            )
        ).set_index("sample_id")
        meta = meta.set_index("sample_id")
        for col in original.columns:
            meta.loc[meta.index.intersection(original.index), col] = original.loc[
                meta.index.intersection(original.index), col
            ]
        meta = meta.reset_index()
        original_set = set(roles[ds]["original"])
        R = set(roles[ds]["R"])
        C = set(roles[ds]["C"])
        meta["role"] = [
            "reference" if i in R else "calibration" if i in C else "target"
            for i in ids
        ]
        meta["in_original_analysis"] = [i in original_set for i in ids]
        for q, t in enumerate(ts):
            meta["has_" + t] = presence[:, q]
        meta["dataset_namespace"] = ds
        meta["discovery_caveat"] = (
            "Fixed historically selected modules; previously examined validation data"
        )
        atomic_parquet(root / "metadata" / f"{variant}.parquet", meta)
        role_indices = {
            r: [lookup[i] for i in sorted(roles[ds][r])] for r in ("R", "C", "T")
        }
        datasets[variant] = dict(
            dataset=ds,
            tissues=ts,
            n_donors=len(ids),
            n_nodes=len(nodes),
            shape=shape,
            nodes=nodes,
            role_indices=role_indices,
            source_sha256=[
                r["sha256"]
                for r in receipts
                if r["path"] in {str(p) for v in paths.values() for p in v}
            ],
            expression_sha256=sha(private / f"{variant}_expression.npy"),
            presence_sha256=sha(private / f"{variant}_presence.npy"),
            availability_policy="Fixed dataset-wide source-row presence AND any finite value across the entire supplied tissue assay; never inferred from selected module genes",
            metadata_file=f"metadata/{variant}.parquet",
            allocation_id=f"ROSMAP_B_seed611010_{ds}_v2",
        )
        for j in members:
            j["node_indices"] = [node_index[tuple(n)] for n in j["nodes"]]
        atomic_json(
            private / f"{variant}_identities.json",
            dict(donors=ids, public_ids=[mapping[ds][i] for i in ids]),
        )
        print(
            json.dumps(
                dict(
                    stage="prepared_input",
                    variant=variant,
                    donors=len(ids),
                    nodes=len(nodes),
                    original_targets=len(roles[ds]["T"]),
                    external_donors=len(set(ids) - original_set),
                )
            ),
            flush=True,
        )
    atomic_json(private / "roles.json", roles)
    atomic_json(private / "mapping.json", mapping)
    atomic_json(
        root / "catalog.json",
        dict(families=families, modules=catalog, datasets=datasets),
    )
    atomic_json(
        root / "input_receipts.json",
        receipts
        + [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sources],
    )
    return catalog, families, datasets
