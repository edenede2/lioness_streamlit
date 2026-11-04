"""Prespecified, local-only validation contract and immutable result paths."""

from pathlib import Path
import json
import subprocess
import ast
import hashlib
from .registry import APP, RESEARCH_ROOT
from .inputs import sha, atomic_json
from .core import canonical_hash

WORK = APP / "artifacts/calibrated_validation_20261104"
BASE = APP / "data/control_calibrated_validation_v1"
OUTCOMES = (
    "cogn_global",
    "cogng_demog_slope",
    "cogng_path_slope",
    "motor10_demog_slope",
    "sqrt_parksc_demog_slope",
    "gpath",
    "amyloid",
    "tangles",
)
METRICS = ("gain", "loss", "net_change", "magnitude_atypicality")
COHORTS = ("original", "incomplete")
FOCUS = {"control_derived": (747, 1116, 1064), "full_cohort": (1643,)}
FOCUS_COMPONENTS = ("TS_DLPFC", "CT_DLPFC__PCG")


def freeze():
    from app_helpers.data import ASSOCIATION_OUTCOME_LABELS

    c = json.loads((RESEARCH_ROOT / "catalog.json").read_text())
    manifest = json.loads((RESEARCH_ROOT / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The actual frozen score catalog is not ready")
    inventory = (
        APP
        / "docs/reports/control_calibrated_validation_20261104/provenance_inventory.json"
    )
    if not inventory.exists():
        raise FileNotFoundError(
            "An audited slope/technical provenance inventory is required before freezing"
        )
    implementation = {
        p.name: hashlib.sha256(
            ast.dump(ast.parse(p.read_text()), include_attributes=False).encode()
        ).hexdigest()
        for p in (
            Path(__file__).parent / name
            for name in (
                "validation_inference.py",
                "validation_run.py",
                "validation_focused.py",
                "validation_spec.py",
            )
        )
    }
    contract = dict(
        schema_version=1,
        version="20261104_rank_validation_v1",
        code_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=APP, text=True
        ).strip(),
        source_release="20261103_bf01d5b35d51",
        score_definition="Frozen native G/L/S/A; no recalibration or new score transforms",
        implementation_semantic_sha256=implementation,
        provenance_inventory_sha256=sha(inventory),
        families={
            f: dict(modules=v["modules"], dataset=v["dataset"], regional=v["regional"])
            for f, v in c["families"].items()
        },
        unavailable_families={
            "matched_Control_247": "Separate namespace; no ready compatible calibrated score family in the authoritative catalog"
        },
        cohorts=dict(
            original="role=target AND frozen in_original_analysis",
            incomplete="role=target AND NOT all dataset-wide has_tissue fields; asserted disjoint from original",
            other_complete_targets="Reported in inventory; excluded from the two independent main cohorts",
            maximum_component="Repaired legacy baseline only; descriptive overlapping subset, never a third replication",
        ),
        betas=[3, 1],
        primary_beta=3,
        metrics=list(METRICS),
        primary_metrics=["gain", "loss"],
        secondary_metrics=["net_change", "magnitude_atypicality"],
        outcomes={o: ASSOCIATION_OUTCOME_LABELS[o] for o in OUTCOMES},
        old_baseline_outcomes=20,
        nuisance=dict(
            covariates=["age_at_death", "sex_code", "education_years", "study"],
            intercept=True,
            numeric="Midranks within the exact tested donor subset; centered/scaled nuisance columns for numerical conditioning",
            categorical={
                "sex_code": {"reference": "Code 0", "indicator": "Code 1"},
                "study": {"reference": "MAP", "indicator": "ROS"},
            },
            constants="Remove and record constant nuisance terms; never outcome-guided covariate deletion",
            deficiency="NA inference for rank deficiency, missing covariates, constant score/outcome or inadequate residual df",
        ),
        models=dict(
            A="Unadjusted all score/outcome-available independent targets",
            B="Unadjusted on exactly C complete-case donor IDs",
            C="Partial correlation of residualized score/outcome midranks on base nuisance design",
            study="C separately within ROS and MAP in brain; same base covariates with constant study term removed; motor not applicable",
        ),
        eligibility=dict(
            adjusted_minimum_n=30,
            minimum_residual_df=20,
            full_rank=True,
            unadjusted_minimum_n=3,
            smaller_adjusted="Descriptive effect only where estimable; no adjusted primary p/q",
        ),
        permutations=dict(
            draws=9999,
            seed=611104,
            statistic="two-sided absolute Spearman rho; adjusted absolute partial rho (monotone equivalent of full-model t on the fixed design)",
            exact="Enumerate one-vector unique multiset pairings if min permutation space <= min(100000, floor(8000000/N)); identical tie configurations share their full null distribution",
            unadjusted="Unrestricted one-vector pairings; exact where tractable, otherwise fixed-budget Monte Carlo",
            adjusted="Freedman-Lane: reduced outcome-on-nuisance residual permutation, fitted values added back, full-model refit via equivalent orthogonal projections; ranks/design fixed",
            blocks="Pooled brain adjusted residual permutations restricted within study; within-study and single-MAP motor unrestricted within their study",
            empirical_p="(exceedances+1)/(draws+1); exact tail proportion includes all pairings; no empirical zero p",
            assumptions="Residual exchangeability within study; Freedman-Lane is not universally exact or heteroscedasticity robust",
            degenerate_draws="Conservatively count undefined permuted residual norm as extreme; record count",
            resolution="Family-wide 9999 draws frozen before scan; floor 0.0001. Flag every correction family with alpha/m < floor and every floor-limited test; no outcome-driven refinement",
        ),
        bootstrap=dict(
            catalog_draws=199,
            focused_draws=1999,
            seed=611105,
            interval="95% percentile donor bootstrap; study-stratified with fixed stratum sample sizes; B/C share each resample",
            refit="Rerank score, outcome and numeric nuisance terms and refit each replicate",
            degeneracy="Record valid/degenerate replicates; CI unavailable if fewer than 80% are valid; changing nuisance design columns is degenerate",
            limitation="199-replicate catalog intervals are coarse; 1999-replicate focused intervals are the detailed readout",
        ),
        canonical_hypothesis="Within each module namespace only: identical physical edge-universe hashes AND equivalent native values/denominators share one hypothesis. Prefer resolved tissue/pair; list all aliases. Never merge module families or call aliases independent replication.",
        correction=dict(
            new_q_scan="BH within dataset/cohort/allocation/model/study-stratum/native/unmasked; all ready registered canonical physical hypotheses, eight outcomes, G/L/S/A and both betas",
            new_q_GL_outcome="BH jointly G/L within the same context and fixed beta/outcome, across all registered ready module families/components",
            exclusion="Retain NA rows with reasons; only finite p enter BH. Correction before display filters, Top K or focused selection. q is within-model, not permission to choose the best model.",
            repaired_baseline="All original 20-outcome numeric Spearman tests use declared tie-aware permutation inference; nominal KW p unchanged. Recompute both complete original legacy correction families, preserving aliases and all unchanged tests; old release stays immutable.",
        ),
        focused=dict(
            modules=FOCUS,
            components=list(FOCUS_COMPONENTS),
            outcomes=list(OUTCOMES),
            betas=[3, 1],
            cohorts=list(COHORTS),
            selection="User-named candidates and DLPFC / DLPFC-PCG components declared before running the scan; all outcomes shown, including attenuated/unavailable effects",
            spline="Natural restricted cubic age spline: four knots at 5/35/65/95% of native age in the exact complete-case subset, three age df; knots fixed for permutation and bootstrap; recorded per test",
            pathology_sensitivity="Focused cogng_demog_slope only: base covariates plus ranked gpath/amyloid/tangles, contingent on verified source provenance. Never add pathology again automatically to cogng_path_slope; do not substitute global cognition for an unavailable slope.",
            interaction="Pooled standardized midranks on the exact pooled donor subset; nuisance main effects + score + study + score-by-study. HC3 covariance with t/F reference and residual df; require each study N>=30 and full-model df>=20.",
            comparators="Same family/module unmasked historical LIONESS control-anchored native Connectivity and same-module constituent tissue PC1 eigengenes, reusing existing fits. Shared score/outcome/covariate/comparator donors; unavailable inputs reported, never substituted.",
            joint="Reduced base+available declared comparators versus same model+G+L jointly on identical donors; standardized pooled midranks, HC3 two-df Wald F, delta in-sample R2 with stratified donor bootstrap; rank/condition number/VIF diagnostics; no predictive interpretation",
            focused_fdr="Each focused model/cohort/stratum tested as a declared within-focused family across all named modules/components, endpoints and both betas; retain broad catalog q for primary A/B/C",
        ),
        technical_covariates="Inventory RIN/PMI/batch/cell composition and preprocessing provenance only; none added automatically",
        exclusions="No prediction, reclustering, beta/mask optimization, matching, weighting, new calibration, donor IDs in public aggregate tables, publication, production pointer change, push or merge",
        input_sha256={
            str(p.relative_to(RESEARCH_ROOT)): sha(p)
            for p in [
                RESEARCH_ROOT / "catalog.json",
                RESEARCH_ROOT / "manifest.json",
                *sorted((RESEARCH_ROOT / "metadata").glob("*.parquet")),
                *sorted((RESEARCH_ROOT / "file_indexes").glob("*.json")),
                *sorted((RESEARCH_ROOT / "associations").glob("*.parquet")),
                RESEARCH_ROOT / "associations/testing_manifest.json",
            ]
        },
        method_references=[
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/",
            "https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/exchangeability_blocks.html",
            "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html",
        ],
    )
    contract["analysis_id"] = canonical_hash(contract)
    root = BASE / contract["analysis_id"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_spec.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Immutable analysis specification differs")
    if not path.exists():
        atomic_json(path, contract)
    if not (root / "provenance_inventory.json").exists():
        atomic_json(
            root / "provenance_inventory.json", json.loads(inventory.read_text())
        )
    WORK.mkdir(parents=True, exist_ok=True)
    atomic_json(
        WORK / "workspace.json",
        dict(analysis_id=contract["analysis_id"], root=str(root)),
    )
    atomic_json(WORK / "analysis_spec.json", contract)
    return root, contract


def workspace():
    pointer = json.loads((WORK / "workspace.json").read_text())
    root = Path(pointer["root"])
    contract = json.loads((root / "analysis_spec.json").read_text())
    if contract["analysis_id"] != pointer["analysis_id"]:
        raise ValueError("Workspace identity changed")
    return root, contract
