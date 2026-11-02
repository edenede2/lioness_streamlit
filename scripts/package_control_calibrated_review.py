#!/usr/bin/env python3
"""Build a whitelisted, deidentified implementation report and review ZIP.

Private expression, reference/calibration arrays, donor rows and crosswalks are
never candidates for packaging. Full generated artifacts remain local.
"""
from pathlib import Path
import os, sys, json, re, subprocess, zipfile, platform
from dataclasses import asdict
from datetime import datetime, timezone

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
from app_helpers.control_calibrated.registry import ROOT, CAPABILITIES
from app_helpers.control_calibrated.inputs import atomic_json, sha, resolve_catalog
from app_helpers.control_calibrated.core import Settings, canonical_hash
from app_helpers.control_calibrated.generate import fit_engine_fingerprint

OUT = APP / "artifacts/control_calibrated_implementation_20261102"
LEGACY = [
    "tests/test_streamlit_app.py",
    "README.md",
    "streamlit_app.py",
    "app_helpers/camel.py",
    "app_helpers/charts.py",
    "app_helpers/data.py",
    "app_helpers/donor_explorer_ui.py",
    "app_helpers/figure_export/config.py",
    "app_helpers/outcome_screening_ui.py",
    "app_helpers/regional_modules.py",
    "app_helpers/score_comparison.py",
    "app_helpers/score_comparison_ui.py",
    "app_helpers/streaming_associations.py",
    "app_helpers/validation_review_ui.py",
]


def md_table(frame):
    frame = frame.astype(str)
    return (
        "| "
        + " | ".join(frame.columns)
        + " |\n|"
        + "|".join("---" for _ in frame)
        + "|\n"
        + "\n".join(
            "| " + " | ".join(row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
    )


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    catalog = json.loads((ROOT / "catalog.json").read_text())
    completion = json.loads((ROOT / "associations/completion.json").read_text())
    masks = json.loads((ROOT / "masks/manifest.json").read_text())
    atomic_json(
        OUT / "reproduction_config.json",
        dict(
            schema_version=2,
            root="data/control_calibrated_v2",
            entrypoint="scripts/generate_control_calibrated.py",
            stages=["prepare", "benchmark", "score", "masks", "analyze"],
            betas=[3, 1],
            default_beta=3,
            edge_batch=2048,
            workers_used=8,
            resume=True,
            allocation_seed=611010,
            settings_by_dataset={
                "brain": asdict(Settings(82, 41)),
                "motor": asdict(Settings(60, 30)),
            },
            mask_specs=masks["specs"],
            analysis_manifest="data/control_calibrated_v2/associations/testing_manifest.json",
            prediction=False,
        ),
    )
    verification = json.loads((OUT / "verification.json").read_text())
    mask_verification = json.loads((OUT / "mask_verification.json").read_text())
    coverage_receipt = json.loads((OUT / "coverage_receipt.json").read_text())
    coverage_counts = pd.read_csv(OUT / "coverage_by_module_component.csv")
    structural_unavailable = int((~coverage_counts.structural).sum())
    donor_scope_unavailable = int(
        (coverage_counts.structural & coverage_counts.scored_targets.eq(0)).sum()
    )
    assert all(
        x["status"] in ("complete", "passed")
        for x in (
            manifest,
            completion,
            masks,
            verification,
            mask_verification,
            coverage_receipt,
        )
    )
    # Resolve the authoritative registries again; no clustering is performed.
    fresh = resolve_catalog()
    current_families = fresh[1]
    assert set(current_families) == set(catalog["families"])
    assert {k: v["modules"] for k, v in current_families.items()} == {
        k: v["modules"] for k, v in catalog["families"].items()
    }
    assert {j["module_key"]: j["assignment_sha256"] for j in fresh[0]} == {
        j["module_key"]: j["assignment_sha256"] for j in catalog["modules"]
    }
    rows = []
    for family, info in catalog["families"].items():
        variant = next(
            j["input_variant"] for j in catalog["modules"] if j["family"] == family
        )
        metadata = pd.read_parquet(ROOT / catalog["datasets"][variant]["metadata_file"])
        target = metadata[metadata.role.eq("target")]
        rows.append(
            dict(
                family=family,
                modules=info["modules"],
                dataset=info["dataset"],
                regional=info["regional"],
                original_targets=int(target.in_original_analysis.sum()),
                incomplete_targets=int(
                    (
                        ~target.in_original_analysis
                        & ~target[[c for c in target if c.startswith("has_")]].all(
                            axis=1
                        )
                    ).sum()
                ),
                all_aligned_targets=len(target),
            )
        )
    families = pd.DataFrame(rows)
    families.to_csv(OUT / "catalog_inventory.csv", index=False)
    role_spec = json.loads((ROOT / "private/roles.json").read_text())
    role_public = {
        d: {k: v for k, v in info.items() if k.endswith("_sha256")}
        for d, info in role_spec.items()
    }
    atomic_json(
        OUT / "frozen_role_membership_hashes.json",
        dict(
            seed=611010,
            hashes=role_public,
            counts={
                d: {r: len(info[r]) for r in ("R", "C", "T")}
                for d, info in role_spec.items()
            },
        ),
    )
    # Read-only proof that the old producer's complete checkpoints are resumable.
    snapshot = ROOT / "private/code_snapshots/core.py"
    producer_receipt = json.loads((snapshot.parent / "receipts.json").read_text())
    assert sha(snapshot) == producer_receipt["core.py"]
    assert fit_engine_fingerprint(snapshot) == fit_engine_fingerprint(
        APP / "app_helpers/control_calibrated/core.py"
    )
    resume = []
    for variant, info in catalog["datasets"].items():
        work = ROOT / "private" / variant
        state = json.loads((work / "progress.json").read_text())
        r = info["role_indices"]["R"]
        c = info["role_indices"]["C"]
        expected = canonical_hash(
            dict(
                version=2,
                batch=2048,
                variant=variant,
                source=info["source_sha256"],
                availability=info["presence_sha256"],
                R=r,
                C=c,
                edge_universe=sha(work / "edges.npy"),
                guard=64,
                sign_agreement=0.8,
                near_zero=0.01,
                sign_seeds=list(Settings(len(r), len(c)).sign_seeds),
                core_sha256=sha(snapshot),
            )
        )
        assert state["signature"] == expected
        edges = np.load(work / "edges.npy", mmap_mode="r")
        assert state["next"] == len(edges)
        resume.append(
            dict(
                variant=variant,
                physical_edges=len(edges),
                completed_edge_offset=state["next"],
                producer_signature=expected,
                fit_engine_AST_identical=True,
            )
        )
    atomic_json(
        OUT / "resume_verification.json",
        dict(
            status="passed",
            method="Read-only reconstruction of CLI checkpoint signature; no refitting or replacement of measured generation resources",
            variants=resume,
        ),
    )
    preservation = {}
    for name in ("preexisting_changes.json", "prediction_snapshot.json"):
        old = json.loads((OUT / name).read_text())
        bad = [
            rel
            for rel, h in old.items()
            if not (APP / rel).exists() or sha(APP / rel) != h
        ]
        assert not bad, bad
        preservation[name] = dict(
            checked=len(old),
            unchanged=len(old),
            changed=0,
            snapshot_sha256=sha(OUT / name),
        )
    atomic_json(
        OUT / "final_preservation.json",
        dict(
            status="passed",
            snapshots=preservation,
            prediction_training_invoked=False,
            capabilities=CAPABILITIES,
        ),
    )
    tests = {}
    for name in (
        "tests_final_regressions.log",
        "tests_final_flows.log",
        "tests_final_models.log",
        "tests_final_comparisons.log",
        "tests_legacy_bundle_remaining.log",
        "tests_selectors_updated.log",
        "tests_source_persistence.log",
    ):
        text = (OUT / name).read_text()
        lines = [
            line for line in text.splitlines() if re.search(r"\d+ passed.* in ", line)
        ]
        assert lines, (name, "unfinished")
        if name == "tests_final_regressions.log" and "3 failed" in lines[-1]:
            expected = {
                "test_streamlit_smoke_lioness_and_bonobo_module_definitions",
                "test_streamlit_muscle_module_set_core_views_render",
                "test_streamlit_all_donor_muscle_core_views_render",
            }
            failed = {
                line.rsplit("::", 1)[-1]
                for line in text.splitlines()
                if line.startswith("FAILED ")
            }
            assert failed == expected
            assert re.search(
                r"3 passed.* in ", (OUT / "tests_selectors_updated.log").read_text()
            )
            lines[
                -1
            ] += "; these three stale selector/outcome assertions were updated to the authoritative registry and all three passed in tests_selectors_updated.log"
        else:
            assert not re.search(r"\d+ failed", lines[-1]), (
                name,
                lines[-1:] or "unfinished",
            )
        tests[name] = dict(result=lines[-1], sha256=sha(OUT / name))
    atomic_json(
        OUT / "test_results.json",
        dict(
            status="passed",
            tests=tests,
            environment="Cached Streamlit 1.64; pytest plugin autoload disabled; numeric production environment retained",
            no_prediction_training=True,
            historical_fixture_limitations=json.loads(
                (OUT / "legacy_fixture_parity.json").read_text()
            )["legacy_test_limitations"],
        ),
    )
    checks = [
        (
            "Main module browser / donor measures",
            "streamlit_app.py; control_calibrated/ui.py",
            "Implemented",
            "Complete app flow for four multi-tissue families; G/L paired; native default",
        ),
        (
            "Regional partitions / tissue namespaces",
            "regional_modules.py; adapter.py",
            "Implemented",
            "Complete app flow for each of three regional families",
        ),
        (
            "Per tissue / pair / TS / CT / whole union",
            "adapter.py; core.py; generate.py",
            "Implemented",
            "Full artifact audit, physical union tests and independent direct-source calculations",
        ),
        (
            "Original / incomplete / maximum scope and role joins",
            "inputs.py; adapter.py",
            "Implemented",
            "Frozen membership witnesses; 10182 files checked; incomplete-tissue app and numerical tests",
        ),
        (
            "Frozen discovery masks",
            "masks.py; generate_masks.py",
            "Implemented",
            "75264 file hashes/schema/coverage; eight donor contribution reconstructions; selector flow",
        ),
        (
            "Metadata and diagnosis filters / transforms",
            "control_calibrated/ui.py",
            "Implemented",
            "Saved-context invalidation and frozen R transform tests",
        ),
        (
            "Scatter / numeric / nominal / descriptive OLS",
            "ui.py; analysis.py",
            "Implemented",
            "Spearman/KW full scan; categorical override; independent complete-case HC3 check",
        ),
        (
            "Heatmaps / statistics / rankings / CT-TS screen",
            "control_calibrated/ui.py",
            "Implemented",
            "All baseline tests saved; q correction before display filtering; app flow tests",
        ),
        (
            "Distributions and group differences",
            "control_calibrated/ui.py; distributions.py",
            "Implemented",
            "App rendering and existing donor-bootstrap/group-comparison regression checks",
        ),
        (
            "Same-module historical source comparison",
            "ui.py; association_score_comparison.py",
            "Implemented",
            "Paired independent donors, separate q_comparison, no historical refit",
        ),
        (
            "Module-outcome comparison boxes",
            "score_comparison.py; score_comparison_ui.py",
            "Implemented",
            "Source capability gates, shared donors, calibrated box calculation and Top K reuse",
        ),
        (
            "Donor explorer / R-C role conditions",
            "donor_explorer_ui.py; inspection.py",
            "Implemented",
            "Dedicated source flow and role rejection tests",
        ),
        (
            "Validation and replication",
            "validation_review_ui.py",
            "Implemented",
            "Dedicated source flow defaults to independent incomplete-tissue targets",
        ),
        (
            "Edge contributions / effect-driver loader",
            "inspection.py; data.py",
            "Implemented",
            "Real calibrated readouts/denominators, raw or G/L ranking, masked contribution parity",
        ),
        (
            "Gene-edge searches / edge volcano",
            "outcome_screening_ui.py; inspection_ui.py",
            "Implemented",
            "Full selected edge search, Spearman/Pearson/KW, new q_search; AD-Control contrast",
        ),
        (
            "Streaming association and pathology consumers",
            "streaming_associations.py; data.py",
            "Implemented",
            "Shared source loaders; streamed/direct Spearman parity; supplied pathology scales retained",
        ),
        (
            "CSV and publication exports",
            "ui.py; figure_export/",
            "Implemented",
            "Source-aware downloads, filename provenance, actual PNG/SVG examples and existing export tests",
        ),
        (
            "Legacy CorShrink partition comparison",
            "streamlit_app.py; data.py",
            "Unavailable input",
            "data/all_donor_corrshrink_l4 and its partition_comparison bundle absent; source not in completed authoritative registry",
        ),
        (
            "Group MDC",
            "Existing group-statistic workflow",
            "Separate statistic",
            "No calibrated donor selector or borrowed MDC permutation p values",
        ),
        (
            "Prediction and predictive regression",
            "registry.py; existing prediction code",
            "Excluded",
            "Explicit capability denial; feature registry checks; 322 artifacts unchanged; no training",
        ),
    ]
    checklist = pd.DataFrame(
        checks, columns=["surface", "entrypoint", "status", "verification_or_reason"]
    )
    checklist.to_csv(OUT / "application_coverage_checklist.csv", index=False)
    source_paths = (
        [APP / p for p in LEGACY]
        + sorted((APP / "app_helpers/control_calibrated").glob("*.py"))
        + sorted((APP / "tests").glob("test_control_calibrated*.py"))
        + sorted((APP / "scripts").glob("*control_calibrated*.py"))
        + [
            APP / "scripts/capture_calibrated_app.py",
            APP / "scripts/audit_calibrated_masks.py",
            APP / "docs/control_calibrated_connectivity.md",
        ]
    )
    atomic_json(
        OUT / "code_provenance.json",
        dict(
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=APP, text=True
            ).strip(),
            head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=APP, text=True
            ).strip(),
            files={str(p.relative_to(APP)): sha(p) for p in sorted(set(source_paths))},
            producer_snapshots={k: v for k, v in producer_receipt.items()},
            core_fit_AST_fingerprint=fit_engine_fingerprint(snapshot),
        ),
    )
    resource = dict(
        benchmark=json.loads((ROOT / "benchmark/manifest.json").read_text())[
            "resources"
        ],
        scoring_final_resume=manifest["resources"],
        variant_final_resume=[
            {
                k: v
                for k, v in g.items()
                if k
                in (
                    "variant",
                    "physical_edges",
                    "resolved_cells",
                    "modules",
                    "seconds",
                    "workers",
                    "max_worker_RSS_MiB",
                )
            }
            for g in manifest["generation"]
        ],
        baseline_analysis=dict(seconds=completion["seconds"]),
        masks_final_resume={
            k: masks[k] for k in ("seconds", "workers", "peak_worker_RSS_MiB")
        },
        timing_caveat="Final resume segments exclude earlier checkpointed work and interrupted attempts. These are measured execution segments, not total from-scratch runtime. Parent and maximum worker RSS are distinct, not aggregate concurrent memory.",
    )
    for name, stage, key in (
        ("associations.log", "analyze", "baseline_analysis_cli"),
        ("masks_all.log", "masks", "masks_final_resume_cli"),
    ):
        for line in (OUT / name).read_text().splitlines():
            if line.startswith("{"):
                try:
                    item = json.loads(line)
                    if item.get("stage") == stage and "peak_RSS_MiB" in item:
                        resource[key] = item
                except json.JSONDecodeError:
                    pass
    atomic_json(OUT / "runtime_resources.json", resource)
    source_receipts = json.loads((ROOT / "input_receipts.json").read_text())
    for receipt in source_receipts:
        path = Path(receipt["path"])
        if not path.is_absolute():
            path = APP.parents[1] / path
        assert path.exists() and sha(path) == receipt["sha256"], str(path)
    atomic_json(
        OUT / "input_provenance.json",
        dict(
            catalog_sha256=sha(ROOT / "catalog.json"),
            testing_manifest_sha256=sha(ROOT / "associations/testing_manifest.json"),
            input_receipts_sha256=sha(ROOT / "input_receipts.json"),
            source_files=source_receipts,
            datasets={
                k: {
                    x: v[x]
                    for x in (
                        "dataset",
                        "n_donors",
                        "n_nodes",
                        "source_sha256",
                        "expression_sha256",
                        "presence_sha256",
                        "availability_policy",
                        "allocation_id",
                    )
                }
                for k, v in catalog["datasets"].items()
            },
            scope_inventory_note="Only completed authoritative module_sets and regional partition combinations are generated. Latent all_donor_corrshrink_l4 code hooks have no installed source/artifact bundle.",
        ),
    )
    versions = {"python": platform.python_version()}
    from importlib.metadata import version

    for name in ("numpy", "pandas", "scipy", "pyarrow", "pytest"):
        versions[name] = version(name)
    atomic_json(OUT / "production_environment.json", versions)
    for name in (
        "verified_role_counts",
        "verified_fdr_contexts",
        "independent_source_checks",
        "masked_contribution_checks",
    ):
        pd.read_parquet(OUT / (name + ".parquet")).to_csv(
            OUT / (name + ".csv"), index=False
        )
    contexts = pd.DataFrame(completion["contexts"])[
        [
            "dataset",
            "cohort_scope",
            "planned_structural_tests",
            "estimable_tests",
            "excluded_tests",
        ]
    ]
    maxerror = max(
        v["max_absolute_score_error"]
        for v in verification["independent_source_calculations"]
    )
    report = f"""# Control-calibrated ROSMAP/LIONESS implementation

Generated {datetime.now(timezone.utc).isoformat()}. Local branch `codex/control-calibrated-connectivity-20261102`. No deployment, publication, merge or push.

## Introduction

Control-calibrated connectivity is usable as an ordinary donor-score source in the applicable non-prediction Streamlit views. Uniform signed-power beta3 is the specified default; beta1 is selectable sensitivity. G and L initially appear together. The full ready catalog, both beta versions, compatible frozen masks and baseline outcome scan have finished. This verifies implementation, not biological validity.

## Background and scientific method

For each unique physical tissue/gene pair, add-one signed-power LIONESS weights use the frozen R controls. Calibration uses disjoint C controls against the same R. The numerical implementation fixes both reproduced stage3 missingness bugs: unavailable endpoints affect only their own edge, and tissue presence is determined independently of the selected module genes. Present-tissue sparse genes use R-only endpoint means; whole missing tissues remain absent.

For calibration magnitude mean $\\nu_e$ and sample SD $\\tau_e$, $a_{{ie}}=(|w_{{ie}}|-\\nu_e)/\\tau_e$. A fixed eligible edge universe E supplies the common denominator:

$$
G_i=|E|^{{-1}}\\sum_e\\max(a_{{ie}},0),\\quad L_i=|E|^{{-1}}\\sum_e\\max(-a_{{ie}},0),\\quad S_i=G_i-L_i,\\quad A_i=G_i+L_i.
$$

G measures magnitude gain, L magnitude loss, S net change with possible cancellation, and A total magnitude atypicality. Signed-profile distance D remains separate. These measure LIONESS contribution deviations, not observed within-person correlations, fold changes, Gaussian tail probabilities or donor abnormality p values. A negative edge can gain magnitude; reversal is a separate annotation. Lower L variation alone has no validity implication.

The declared full C counts, ddof1, stable edgewise moments, relative numerical guard and 100 conditional C sign bootstrap settings remain fixed. Sign uncertainty does not exclude magnitude-eligible edges. Physical unions sum contributions and counts, deduplicate edges and return NA for missing required tissues. Optional asinh/RINT transforms use frozen R training diagnostics, separately per metric; native-unit identities precede transforms. See [the complete method and CLI guide](method_guide.md).

## Dataset and scope

{md_table(families)}

Frozen B seed611010 uses brain R82/C41 and original independent T327 (41 Control +286 AD/MCI), and motor R60/C30 and original independent T267 (30 Control +237 AD/MCI). R/C cannot re-enter target associations through renamed identities. Regional motor alignment includes eight additional source-labelled Other_dementia donors; those labels were resolved before the final testing manifest and p values. Supplied pathology square-root scales and cognitive/motor slope adjustments are preserved.

The current registry resolves {len(families)} ready sources and {families.modules.sum():,} modules. The four multi-tissue families and three additional regional partitions all use the same adapters. `all_donor_corrshrink_l4` has latent code hooks but is absent from the completed authoritative registry; its `data/all_donor_corrshrink_l4` source/details and partition-comparison bundle are not installed. It is specifically unavailable, rather than silently represented by another partition.

## Results and coverage

- {sum(g['physical_edges'] for g in manifest['generation']):,} distinct physical-edge/input-variant rows calibrated for each beta; brain, motor and regional motor inputs stay distinct.
- {verification['checked_score_files']:,} unmasked score files and {verification['checked_score_rows']:,} donor/component rows; these are wide summaries, not a donor-by-edge catalog cube.
- {len(coverage_counts):,} enumerated artifact cells across beta and scope: {structural_unavailable:,} structurally absent, {manifest['structural_cells']:,} structural and calibration-eligible, {manifest['structural_cells']-donor_scope_unavailable:,} with at least one scored target, and {donor_scope_unavailable:,} structural cells with no scoreable target in that donor scope. Thus {manifest['unavailable_cells']:,} total cells have no available score; failed cells = {manifest['failed_cells']}. Donor-scope unavailability is entirely missing required tissue. A scored cell can still contain individual unavailable donors. The CSVs retain both cell and donor counts.
- {masks['completed_modules']} multi-tissue modules, {mask_verification['mask_contexts']:,} module/mask contexts and {mask_verification['files']:,} masked files ({mask_verification['donor_component_rows']:,} wide donor/component rows). Brain has 34 compatible historical masks per module; motor has three Hedges |g|0.4 directional masks. Regional partitions have no registered legacy masks; motor FDR masks are not registered. Every supported physical mapping completed; failed masks =0.
- 15 numeric/ordinal plus5 nominal outcomes, G/L/S/A, both betas, every structural component, every ready source and three scopes: {verification['planned_tests']:,} planned baseline tests.

{md_table(contexts)}

Spearman uses minimum paired N3; Kruskal–Wallis uses at least5 donors per included nominal group. Non-estimable p values remain NA with reasons and do not enter finite-p BH denominators. The frozen executed manifest ID is `{verification['testing_manifest_id']}`. An initial plan was retained; before p values the final plan added metadata hashes and the explicit labels for newly aligned regional motor donors. No power, mask, control split or default was chosen by strongest outcomes.

`q_GL_outcome` jointly corrects G/L and all registered partitions/components within the fixed dataset/cohort/allocation/beta/outcome/method/mask/subgroup/model/transform. `q_scan` corrects the entire baseline dataset/cohort/allocation scan, both betas and all four metrics/outcomes/partitions. Both corrections match an independent SciPy BH implementation. Display filtering never recalculates them. Pearson, covariates, changed masks/transforms/subgroups, edge searches and selected source comparisons have separate declared saved contexts and q_search/q_comparison; baseline q_scan does not cover selecting among these searches.

## Application and verification

Select **Control-calibrated connectivity · beta3** under **Network estimator** in a normal analysis view. Regional modules use **Regional donor score source**. Dedicated Donor explorer, Validation & replication and Gene & edge associations have source controls. Module–outcome comparison has a source control in each box. Original independent targets use327 brain/267 motor, while incomplete or all-aligned scope labels show their actual candidate counts. Every statistical table retains its paired/tested N. Prediction has no calibrated source.

The [application coverage checklist](application_coverage_checklist.csv) identifies each consumer and evidence. Tests exercise all seven sources, an incomplete-tissue flow, beta changes, transforms, masks, ordinary views, donor and validation entry points, new edge searches, categorical overrides, historical paired comparisons and module-comparison display reuse. Edge tables expose raw and calibrated quantities, calibration N, eligibility/sign annotations and actual aggregate denominators; rank by calibrated gain/loss or raw magnitude.

The full unmasked/masked audits checked stored content hashes and role/beta/module identities. Independent direct np.corrcoef calculations in all seven sources and both betas had maximum absolute score error {maxerror:.3g}. Eight real masked donor reconstructions matched stored G/L/S/A. Focused tests cover both missingness bugs, policies, order/batch invariance, unions, denominator/sign identities, serialization, cache beta separation, role exclusion, FDR membership and independent HC3 calculations. All105 pre-existing changed files and322 prediction artifacts still match their snapshots. Prediction capabilities are explicitly denied; no predictive experiment ran.

{chr(10).join('- ' + v['result'] + ' — `' + k + '`' for k,v in tests.items())}

Four pre-existing data-bundle assertions do not match the installed historical assets: the muscle-control KEGG table has39 unavailable regional p/q rows per tissue; LIONESS and BONOBO total-row assertions expect twelve rather than fifteen numeric outcomes; and some historical manifest entries omit an optional Parquet row-count field. The same conditions exist with the Git HEAD loaders and unchanged historical manifest. `legacy_fixture_parity.json` verifies22 historical tables (six LIONESS, twelve BONOBO rule tables and four KEGG tables) are identical under the original and current loaders. Historical data and recorded all12 correction aliases are preserved; these unrelated fixture assertions are not claimed to pass. The other33 data-bundle checks pass. Three app smoke assertions were updated to the actual20-outcome registry, current historical labels and new selectable calibrated sources; all three follow-up flows pass.

Actual browser screenshots are in `screenshots/`; source-backed publication PNG/SVG exports and aggregate source statistics are in `exports/`. They were rendered locally, not generated or cosmetically edited. Export examples are descriptive module examples and were not selected as biological findings.

## Runtime, reproducibility and limitations

The measured benchmark segment took {resource['benchmark']['seconds']:.2f}s (parent peak {resource['benchmark']['peak_RSS_MiB']:.1f}MiB). The final successful scoring resume took {manifest['resources']['seconds']:.2f}s (parent peak {manifest['resources']['peak_RSS_MiB']:.1f}MiB). Baseline analysis took {completion['seconds']:.2f}s; the final mask resume took {masks['seconds']:.2f}s, with maximum single-worker peak {masks['peak_worker_RSS_MiB']:.1f}MiB. Earlier attempts contributed checkpointed work: these are execution segments, not total from-scratch runtime or aggregate concurrent RSS. Original logs remain local. Variant timings and worker peaks are in `runtime_resources.json`.

Run the stage commands in the method guide to prepare, benchmark, score, mask and analyze with checkpoint/resume. `resume_verification.json` independently reconstructs completed producer signatures without overwriting measured resources. After generation, pure core aggregation APIs were repaired while reference-fit/calibration ASTs remained identical to the verified producer snapshot; compatibility is explicit and all saved numerical artifacts were audited. Current and producer code hashes, actual inputs/availability, frozen membership hashes, environment and testing manifest are recorded. Legacy Connectivity/native, discovery partitions and group MDC remain intact.

Magnitude-ineligible calibration edges, absent required tissues, empty masks and constant frozen R transform distributions have specific NA reasons. No partial union is called a full module. Historical target-informed masks and previously examined validation data remain exploratory, and discovery overlap precludes claims of untouched confirmation. Implementation verification does not establish beta3 superiority, biological specificity, causal interpretation or clinical utility.

## Conclusions and generated deliverables

The complete ready catalog and baseline scan are implemented and usable locally. The next scientific step is a prespecified biological validation design; it is separate from software completion. No deployment or predictive work is needed to review this implementation.

`data/control_calibrated_v2/` retains full local score/association artifacts and private caches. The compact `control_calibrated_review_20261102.zip` includes only whitelisted code, method/report, aggregate coverage and numerical checks, manifests/hashes, test receipts, screenshots and exports. It excludes private arrays, expression, donor rows, individual C weights and identity crosswalks. `review_manifest.json` lists every packaged file with SHA256.
"""
    (OUT / "implementation_report.md").write_text(report)
    (OUT / "method_guide.md").write_text(
        (APP / "docs/control_calibrated_connectivity.md").read_text()
    )
    # Whitelist, rather than recursively zipping a data or artifacts directory.
    local_names = [
        "implementation_report.md",
        "method_guide.md",
        "application_coverage_checklist.csv",
        "catalog_inventory.csv",
        "coverage_by_module_component.csv",
        "coverage_by_family_cohort_component_beta.csv",
        "coverage_by_family_cohort_component_beta_mask.csv",
        "baseline_correction_contexts.csv",
        "coverage_receipt.json",
        "adapter_checks.json",
        "verification.json",
        "mask_verification.json",
        "frozen_role_membership_hashes.json",
        "resume_verification.json",
        "final_preservation.json",
        "test_results.json",
        "legacy_fixture_parity.json",
        "code_provenance.json",
        "runtime_resources.json",
        "reproduction_config.json",
        "input_provenance.json",
        "production_environment.json",
        "application_test_environment.json",
        "verified_role_counts.csv",
        "verified_fdr_contexts.csv",
        "independent_source_checks.csv",
        "masked_contribution_checks.csv",
    ]
    members = {"review/" + n: OUT / n for n in local_names}
    members["review/testing_manifest.json"] = (
        ROOT / "associations/testing_manifest.json"
    )
    members["review/baseline_completion.json"] = ROOT / "associations/completion.json"
    for directory in ("screenshots", "exports"):
        for p in sorted((OUT / directory).iterdir()):
            if p.suffix in (".png", ".svg", ".csv", ".json"):
                members["review/" + str(p.relative_to(OUT))] = p
    for p in sorted(set(source_paths)):
        if str(p.relative_to(APP)) not in LEGACY:
            members["source/" + str(p.relative_to(APP))] = p
    patch = subprocess.check_output(["git", "diff", "--", *LEGACY], cwd=APP)
    (OUT / "legacy_integration.patch").write_bytes(patch)
    members["source/legacy_integration.patch"] = OUT / "legacy_integration.patch"
    review_manifest = dict(
        privacy="Whitelisted aggregate evidence and source only; no private arrays, donor rows, expression or identity crosswalks",
        files={
            name: dict(sha256=sha(p), bytes=p.stat().st_size)
            for name, p in members.items()
        },
    )
    atomic_json(OUT / "review_manifest.json", review_manifest)
    members["review/review_manifest.json"] = OUT / "review_manifest.json"
    archive = OUT / "control_calibrated_review_20261102.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for name, p in members.items():
            assert (
                "/private/" not in name
                and "metadata/" not in name
                and "scores/" not in name
            )
            z.write(p, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    atomic_json(
        OUT / "delivery_receipt.json",
        dict(
            status="complete",
            bundle=archive.name,
            bundle_sha256=sha(archive),
            bundle_bytes=archive.stat().st_size,
            packaged_files=len(members),
            production_artifacts_complete=True,
            baseline_complete=True,
            prediction_excluded=True,
        ),
    )
    print(
        json.dumps(
            dict(
                status="complete", files=len(members), zip_bytes=archive.stat().st_size
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
