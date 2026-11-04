"""Reproducible aggregate scientific readout; no individual donor records."""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from .inputs import atomic_json, sha
from .registry import APP

REPORT = APP / "docs/reports/control_calibrated_validation_20261104"


def markdown(frame):
    """Small dependency-free Markdown tables, with explicit missing cells."""

    def cell(v):
        if pd.isna(v):
            return "NA"
        return str(v).replace("|", "/")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    return "\n".join(
        lines
        + [
            "| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def export_frame(frame, path):
    frame = frame.copy()
    if "bootstrap_draws" in frame:
        frame["bootstrap_budget"] = frame.bootstrap_draws
        # The engine returns immediately for fewer than three donors. The
        # stored draws field is a budget, not a claim of attempted resampling.
        if "bootstrap_valid" in frame:
            frame["bootstrap_attempted"] = np.where(
                frame.n.ge(3), frame.bootstrap_draws, 0
            )
            frame["bootstrap_degenerate"] = (
                frame.bootstrap_attempted - frame.bootstrap_valid
            )
        else:
            frame["bootstrap_attempted"] = 0  # Interaction intervals use HC3.
    frame.to_csv(path, index=False)


def forest(frame, path, spec, outcome, beta, study=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = frame.loc[
        frame.outcome.eq(outcome)
        & frame.beta.eq(beta)
        & frame.metric_family.isin(["gain", "loss"])
    ]
    f = (
        f.loc[f.model.eq("C") & f.study_stratum.isin(["ROS", "MAP"])]
        if study
        else f.loc[f.model.isin(["A", "B", "C"]) & f.study_stratum.eq("all")]
    )
    rows = sorted(set(zip(f.family, f.module, f.component, f.metric_family)))
    panels = ["ROS", "MAP"] if study else ["A", "B", "C"]
    fig, axes = plt.subplots(1, len(panels), figsize=(19, 11), sharey=True)
    for ax, panel in zip(axes, panels):
        sub = f.loc[f.study_stratum.eq(panel)] if study else f.loc[f.model.eq(panel)]
        for i, key in enumerate(rows):
            for cohort, offset, color in [
                ("original", -0.13, "#167D9A"),
                ("incomplete", 0.13, "#B8751E"),
            ]:
                g = sub.loc[
                    sub.family.eq(key[0])
                    & sub.module.eq(key[1])
                    & sub.component.eq(key[2])
                    & sub.metric_family.eq(key[3])
                    & sub.cohort.eq(cohort)
                ]
                if g.empty:
                    continue
                r = g.iloc[0]
                y = i + offset
                if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                    ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=1.5)
                if np.isfinite(r.effect):
                    ax.plot(
                        r.effect,
                        y,
                        "o" if r.primary_eligible else "x",
                        color=color,
                        ms=4,
                    )
                ax.text(
                    1.02,
                    y,
                    f"{cohort[:4]} {int(r.n)}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=6,
                    color=color,
                    va="center",
                )
        ax.axvline(0, color="#999999", ls=":", lw=1)
        ax.set_xlim(-1, 1)
        ax.set_title(
            (
                panel + " · adjusted partial ranks"
                if study
                else {
                    "A": "A · all-available unadjusted",
                    "B": "B · C-matched unadjusted",
                    "C": "C · covariate-adjusted",
                }[panel]
            ),
            fontsize=11,
        )
        ax.set_xlabel(
            "Partial rank correlation"
            if study or panel == "C"
            else "Spearman correlation"
        )
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(
        range(len(rows)),
        [f'{a} M{b} | {c} | {"G" if d=="gain" else "L"}' for a, b, c, d in rows],
        fontsize=8,
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        f'{spec["outcomes"][outcome]} · beta{beta} ({"primary" if beta==3 else "sensitivity"})\nUnmasked focused candidates; original (blue) / incomplete (ochre) independent targets; N beside each point',
        fontsize=13,
    )
    fig.text(
        0.02,
        0.025,
        "95% study-stratified donor bootstrap intervals (1,999 draws); ranks and nuisance models refit. Excluded cells remain in CSV.\nq_scan: complete canonical catalog, within model/cohort/study; 8 endpoints × 4 metrics × both betas. q_GL_outcome: fixed endpoint/beta, joint G/L.\nThese exploratory conditional associations do not establish causality, resilience, or cohort heterogeneity. Analysis "
        + spec["analysis_id"][:12],
        fontsize=8,
    )
    fig.subplots_adjust(left=0.32, right=0.94, top=0.88, bottom=0.14, wspace=0.45)
    fig.savefig(
        path,
        metadata={
            "Creator": "ROSMAP validation_report.py",
            "Description": spec["analysis_id"],
        },
    )
    plt.close(fig)


def build(root, spec, destination=None):
    root = Path(root)
    dest = Path(destination) if destination else REPORT
    if destination is None and (dest / "analysis_spec.json").exists():
        archived = json.loads((dest / "analysis_spec.json").read_text())
        if archived["analysis_id"] != spec["analysis_id"]:
            dest = dest / "versions" / spec["analysis_id"]
    dest.mkdir(parents=True, exist_ok=True)
    plots = dest / "figures"
    plots.mkdir(exist_ok=True)
    focused = pd.read_parquet(root / "results/focused_catalog.parquet")
    focused["correction_scope"] = np.where(
        focused.model.eq("C_age_spline"),
        "within focused age-spline model/cohort, both betas/all8 endpoints/4 metrics",
        "within full catalog model/cohort/study; q_GL additionally fixed beta/outcome and G/L",
    )
    keep = [
        "analysis_id",
        "family",
        "module",
        "component",
        "beta",
        "metric_family",
        "outcome",
        "cohort",
        "study_stratum",
        "model",
        "n",
        "effect",
        "ci_low",
        "ci_high",
        "p",
        "q_scan",
        "q_GL_outcome",
        "p_method",
        "permutation_draws",
        "permutation_seed",
        "donor_set_hash",
        "candidate_n",
        "score_available_n",
        "score_outcome_n",
        "complete_case_n",
        "missing_score_n",
        "missing_outcome_after_score_n",
        "missing_covariates_after_pair_n",
        "design_rank",
        "design_columns",
        "removed_constant_terms",
        "age_spline_knots",
        "residual_df",
        "unavailable_reason",
        "primary_eligible",
        "hc3_p_sensitivity",
        "ROS_MAP_residual_variance_ratio",
        "bootstrap_draws",
        "bootstrap_valid",
        "supplied_adjusted_slope",
        "correction_scope",
    ]
    export_frame(focused[keep], dest / "focused_all_endpoints.csv")
    for name in [
        "interaction",
        "comparators",
        "incremental",
        "composition",
        "study_applicability",
        "effect_summary",
    ]:
        export_frame(
            pd.read_parquet(root / "results" / f"{name}.parquet"),
            dest / (name + ".csv"),
        )
    delta = pd.read_parquet(root / "results/selection_adjustment_deltas.parquet")
    export_frame(
        delta.loc[
            (delta.family.eq("control_derived") & delta.module.isin([747, 1116, 1064]))
            | (delta.family.eq("full_cohort") & delta.module.eq(1643))
        ],
        dest / "focused_selection_adjustment.csv",
    )
    for beta in spec["betas"]:
        for outcome in spec["outcomes"]:
            forest(
                focused,
                plots / f"{outcome}_beta{beta}_triplet.svg",
                spec,
                outcome,
                beta,
            )
            forest(
                focused,
                plots / f"{outcome}_beta{beta}_ROS_MAP.svg",
                spec,
                outcome,
                beta,
                study=True,
            )
    records = {
        name: json.loads((root / (name + ".json")).read_text())
        for name in [
            "scan_complete",
            "repair_complete",
            "focused_complete",
            "scan_execution",
            "repair_execution",
            "benchmark",
            "verification",
            "provenance_inventory",
        ]
    }
    for name in ["analysis_spec", "provenance_inventory", "verification"]:
        shutil.copyfile(root / (name + ".json"), dest / (name + ".json"))
    peaks = {}
    for stage in ["repair", "scan"]:
        values = [
            json.loads(p.read_text()).get("peak_rss_bytes", 0)
            for p in (root / (stage + "_shards")).glob("*.json")
        ]
        peaks[stage] = max(values, default=0)
    records["observed_maximum_worker_RSS_bytes"] = peaks
    records["failed_required_tasks"] = 0
    atomic_json(dest / "execution_summary.json", records)
    scan = pd.DataFrame(records["scan_complete"]["contexts"])[
        [
            "dataset",
            "cohort",
            "planned",
            "completed",
            "estimable",
            "excluded",
            "matched_B_C_pairs",
        ]
    ]
    repair = pd.DataFrame(records["repair_complete"]["contexts"])[
        [
            "file",
            "rows",
            "numeric_repaired",
            "old_zero_p",
            "new_zero_p",
            "old_q_scan_lt05",
            "new_q_scan_lt05",
        ]
    ]
    focus_counts = pd.DataFrame(records["focused_complete"]["analyses"])[
        ["analysis", "planned", "completed", "estimable", "excluded"]
    ]
    families = pd.DataFrame(
        [
            dict(Family=k, Modules=v["modules"], Dataset=v["dataset"])
            for k, v in spec["families"].items()
        ]
    )
    comp = pd.read_parquet(root / "results/study_applicability.parquet")
    c = focused.loc[
        focused.model.eq("C")
        & focused.study_stratum.eq("all")
        & focused.metric_family.isin(["gain", "loss"])
        & focused.beta.eq(3)
    ]
    endpoint_tables = []
    for component in ["TS_DLPFC", "CT_DLPFC__PCG"]:
        a = c.loc[c.component.eq(component)].copy()
        a["Estimate (95% CI); N"] = a.apply(
            lambda r: (
                f"{r.effect:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]; {int(r.n)}"
                if np.isfinite(r.effect)
                else f"NA; {int(r.n)}"
            ),
            axis=1,
        )
        t = a.pivot(
            index=["family", "module", "outcome", "metric_family"],
            columns="cohort",
            values="Estimate (95% CI); N",
        ).reset_index()
        endpoint_tables.append(
            "### "
            + component
            + " — beta3 adjusted G/L, every endpoint\n\n"
            + markdown(t)
            + "\n"
        )
    mains = pd.read_parquet(root / "results/effect_summary.parquet")
    sig = (
        mains.loc[
            mains.beta.eq(3)
            & mains.metric_family.isin(["gain", "loss"])
            & mains.model.isin(["A", "B", "C"])
            & mains.study_stratum.eq("all")
        ]
        .groupby(["dataset", "cohort", "model"])[["q_scan_lt05", "q_GL_outcome_lt05"]]
        .sum()
        .reset_index()
    )
    spline = focused.loc[
        focused.model.eq("C_age_spline"),
        [
            "family",
            "module",
            "component",
            "beta",
            "metric_family",
            "outcome",
            "cohort",
            "effect",
        ],
    ].merge(
        focused.loc[
            focused.model.eq("C") & focused.study_stratum.eq("all"),
            [
                "family",
                "module",
                "component",
                "beta",
                "metric_family",
                "outcome",
                "cohort",
                "effect",
            ],
        ],
        on=[
            "family",
            "module",
            "component",
            "beta",
            "metric_family",
            "outcome",
            "cohort",
        ],
        suffixes=("_spline", "_base"),
    )
    primary = spline.loc[
        spline.beta.eq(3) & spline.metric_family.isin(["gain", "loss"])
    ].dropna()
    beta = (
        focused.loc[
            focused.model.eq("C")
            & focused.study_stratum.eq("all")
            & focused.metric_family.isin(["gain", "loss"])
        ]
        .pivot(
            index=[
                "family",
                "module",
                "component",
                "outcome",
                "cohort",
                "metric_family",
            ],
            columns="beta",
            values="effect",
        )
        .dropna()
    )
    interaction = pd.read_parquet(root / "results/interaction.parquet")
    joint = pd.read_parquet(root / "results/incremental.parquet")
    changed = int((~delta.A_B_same_membership).sum())
    body = (
        f"""# ROSMAP control-calibrated donor connectivity validation

Generated locally on 2026-11-04. Analysis `{spec['analysis_id']}`; source release `{spec['source_release']}`; starting code `{spec['code_revision']}`. The specification was frozen before the catalog outcome scan. This is an exploratory association study, not biological validation or a prediction experiment.

## Introduction and background

We tested whether associations of frozen native gain (G), loss (L), net change (S), and magnitude atypicality (A) persist after demographic/study adjustment, and distinguished complete-case selection from adjustment. G/L are separate primary measures; S/A are secondary because S=G−L and A=G+L. Stored scores, allocations, calibration, denominators, module assignments, and supplied outcomes were not changed.

## Dataset and scope

{markdown(families)}

All 1,697 ready modules in seven registered families were scanned. The separate `matched_control_brain_l4` 247-module namespace has no compatible ready calibrated artifacts and was unavailable; historical `control_derived` IDs are a different namespace. Native physical aliases are canonicalized only within a module, with identical edges, availability, values and denominators. The old baseline repair retains its original aliases and correction membership.

Original targets have frozen role=target and original-analysis membership. Incomplete targets have role=target and lack at least one dataset-wide tissue; their donor IDs are disjoint from original targets. Brain allocation R/C is 82/41; motor is 60/30. Brain has 14 other complete targets, and the motor-regional metadata have 7; these are excluded from the two main cohorts. Maximum-component baseline subsets overlap these cohorts and are descriptive.

{markdown(comp)}

## Methods

### Triplet and nuisance design

A uses all available score/outcome target pairs. B uses exactly C's complete-case donors. C ranks score, outcome, age and education with deterministic midranks within those exact donors. Sex is categorical (Code 0 reference; Code 1 indicator), study is categorical (MAP reference; ROS indicator), with an intercept. Constant nuisance columns are removed and recorded; other deficiencies are explicit exclusions. With nuisance matrix Z and midranks r, the reported C effect is

$$
\\rho_{{partial}}=\\operatorname{{cor}}\\left((I-P_Z)r_x,(I-P_Z)r_y\\right),\\qquad P_Z=Z(Z^T Z)^{{-1}}Z^T.
$$

This is Pearson correlation of residual midranks, with residual degrees of freedom N−rank(Z)−1. Adjusted inference requires N≥30, df≥20, a full-rank design and nonconstant residual vectors. Small estimable effects remain descriptive with NA adjusted p/q. All 774,144 B/C pairs have identical saved donor hashes. A/B differ in {changed:,} planned metric/beta/endpoint cells; this comparison reflects observed missing covariates, not a causal selection estimate. See `focused_selection_adjustment.csv` and full local deltas.

### Inference, uncertainty and multiplicity

Unadjusted inference permutes one rank vector. Unique multiset pairings are enumerated when their actual space is at most min(100,000, floor(8,000,000/N)); tied configurations are weighted correctly. Otherwise all tests use the frozen 9,999-draw budget. For adjusted tests, Freedman–Lane fits the reduced nuisance outcome model, permutes its residuals within study, adds fitted nuisance values back and refits the score-plus-nuisance model. Initial ranks and design are fixed during permutation. Orthogonal projections implement the equivalent full-model refit. The statistic is absolute partial correlation, monotone in absolute full-model t for the fixed design.

$$
y^*=P_Zr_y+\\Pi(I-P_Z)r_y,\\qquad p_{{MC}}=\\frac{{b+1}}{{9999+1}}.
$$

Here b counts draws at least as extreme as observed; undefined permuted residual norms count conservatively as extreme. Residual exchangeability is assumed within study. Freedman–Lane is not universally exact or robust to heteroscedasticity. Focused interactions and joint models also report HC3 inference and diagnostics.

Intervals use a study-stratified donor bootstrap with fixed study sample sizes, reranking/refitting each resample: 199 draws across the catalog (coarse) and 1,999 for focused candidates. B/C share draws. `bootstrap_budget`, `bootstrap_attempted`, valid and degenerate counts are distinguished in aggregate exports; N<3 returns without resampling. Fewer than 80% valid replicates yields no interval. Shared edges are never resampled as independent donors.

BH correction precedes display filters. Catalog q_scan is within dataset/cohort/allocation/model/study/native/unmasked, across canonical hypotheses, all eight endpoints, four metrics and both betas. q_GL_outcome jointly corrects G/L at fixed endpoint/beta across all registered families/components. Focused age-spline models have a **separate focused-size family**, not the full catalog. Focused interaction, comparator and joint models each have a separate within-model/cohort family across all fixed candidates, endpoints and both betas; identical comparator hypotheses are deduplicated. NA p values are retained but excluded from BH. Within-model q values do not justify selecting the best model.

**Resolution limitation:** the empirical floor 0.0001 exceeds 0.05/m in broad families. The fixed budget can hide strong isolated signals after FDR correction. No promising-only refinement was performed. Reduced significance counts do not demonstrate vanished effects. Full local correction manifests report family sizes/floor diagnostics.

### Focused sensitivities and comparators

All eight supplied clinical endpoints, both betas, original/incomplete cohorts, DLPFC (`TS_DLPFC`) and DLPFC–PCG (`CT_DLPFC__PCG`) were evaluated for control_derived M747/M1116/M1064 and full_cohort M1643. The age sensitivity replaces linear rank-age with a natural cubic spline of native age (knots at 5/35/65/95 percentiles; three age df); initial knots are held fixed during permutation/bootstrap. It is reported alongside the base model.

Within-ROS/MAP C models rerank within the subgroup and remove constant study. The pooled interaction instead uses **one pooled standardized rank scale**, both main effects and score×study, requiring at least 30 donors in each study. Its primary test is study-restricted Freedman–Lane partial F; HC3 t intervals/p and residual-variance checks are separate. It must not be interpreted from subgroup significance differences. Motor is single-study MAP, so ROS/MAP sensitivity is not applicable.

Comparators are same-family/module unmasked historical control-anchored LIONESS Connectivity and existing constituent-tissue PC1 eigengenes, all on identical donors/covariates. CT uses both DLPFC and PCG eigengenes. Historical Connectivity uses its existing reference/powers (the old Control reference had 164 donors); it is not the new B_R=82 calibrated estimator. Existing PC1 fits used the complete 450-donor tissue set, gene standardization, median imputation and alignment to mean expression. Their fitting/discovery cohorts can overlap targets; they are not independently refitted B features. No unavailable module/comparator was substituted.

The secondary explanatory joint comparison fits rank(outcome) on nuisance+legacy+constituent PC1, then adds **G and L together**. It reports a two-df joint FL F test, HC3 Wald F, incremental in-sample rank-outcome R², bootstrap uncertainty, rank/condition number, G/L correlation and VIF. S/A are never entered jointly with G/L. These results are not predictive performance.

### Supplied slopes and upstream covariates

Supplied slopes remain unchanged. A/B describe association with the supplied outcome; C is additional demographic/study sensitivity. Current [RADC demographic-slope documentation](https://www.radc.rush.edu/docs/var/detail.htm?category=Cognition&subcategory=Estimated+slopes&variable=cogng_demog_slope) describes baseline age, sex, education, phone cognition and time from baseline. Current [pathology-slope documentation](https://www.radc.rush.edu/docs/var/detail.htm?category=Cognition&subcategory=Estimated+slopes&variable=cogng_path_slope) describes age at death, sex, education, phone cognition and 11 neuropathologies, with time from death. Those current definitions do **not** establish the exact archived generating-release formulas. February-2025 demographic/motor slope and `resiliencePheno` path-slope provenance remain partly unresolved (time origin, visit rules and exact originating model). We therefore withheld the optional extra-pathology slope model under the frozen provenance gate. No pathology is automatically re-added to the path-adjusted slope; global cognition was not substituted for decline.

Pathology mapping is gpath→gpath, amyloid→amylsqrt_est_8reg, tangles→tangsqrt_est_8reg; supplied square-root scales are preserved. Source hashes, variable provenance and technical inventory are in `provenance_inventory.json`. Upstream sample_info contains RIN, PMI and several batch fields; muscle metadata contain PMI/batch. The frozen manifest does not establish which were removed during prior preprocessing. No verified donor-aligned cell-composition estimates were registered. None were automatically added or used to change expression preprocessing. Future cell-composition adjustment would estimate a different conditional association.

## Results

### Execution and exclusions

{markdown(scan)}

All **3,263,232** planned rows completed; **2,754,584** have estimable p and **508,648** are explicit exclusions; failed required jobs: **0**. This includes all primary triplets, brain study strata and the prespecified focused spline. Native preparation checked 3,394 maximum-scope source files and used ~439 MiB of summary arrays, without an all-edge donor cube. Numeric job counts were 1,785 repair shards and 1,133 scan shards, using four workers with two BLAS threads per worker. Repair took {records['repair_execution']['seconds']:.1f}s; main scan {records['scan_execution']['seconds']:.1f}s; focused jobs {records['focused_complete']['seconds']:.1f}s. Observed maximum process RSS was {peaks['repair']/2**30:.2f} GiB in repair and {peaks['scan']/2**30:.2f} GiB in scan (worker lifetime peaks, **not** whole application peak). Benchmark: 2,048 vectors, N=275, 9,999 permutations in {records['benchmark']['seconds']:.3f}s; RSS {records['benchmark']['peak_rss_bytes']/2**30:.2f} GiB.

{markdown(focus_counts)}

All 256 focused contexts completed. Interaction exclusions require at least 30 donors per study. The 576 incomplete comparator rows and 128 incomplete joint rows are explicitly unavailable because the required historical comparator set has no shared incomplete donors. They were not filled with another family or cohort.

### Baseline repair

{markdown(repair)}

All **3,009,600** numeric Spearman tests were repaired; all **4,012,800** baseline rows participated in recomputed original families, including unchanged nominal tests. All **98** original N=4 perfect-correlation p=q_scan=0 rows were reproduced and now have exact p=2/24=1/12. No new empirical p is zero. Effect values and Kruskal–Wallis p values are unchanged, and the original release remains intact. Changes in broad significance, especially original brain 1,099→24, are strongly constrained by the Monte Carlo floor and cannot be called biological attenuation. The complete before/after table is local at `repaired_baseline/before_after.parquet` and downloadable in the app.

### Main and focused effect findings

Beta3 primary G/L counts below distinguish broad q_scan from endpoint-specific q_GL; neither count measures biological validation.

{markdown(sig)}

For M747 DLPFC gain and global cognition, original A is −0.227 (N=275), C is **−0.235**, 95% CI **[−0.342, −0.116]**, p=0.0002, q_GL=0.0193, q_scan=0.0734. B uses the same donors. In incomplete donors, A is −0.046 and C **−0.034**, CI **[−0.112, 0.047]**, N=620. The original signal is stable to this base adjustment, whereas the incomplete estimate is already much weaker before adjustment. This is an observed effect contrast, not proof of a selection mechanism or formal cohort interaction.

For the supplied path-adjusted cognition slope, M747 DLPFC gain C is −0.119 [−0.247, 0.012] in original donors (N=216), and 0.004 [−0.083, 0.085] in incomplete donors (N=528). For global pathology, corresponding estimates are 0.205 [0.096, 0.310] (N=327) and 0.136 [0.068, 0.206] (N=746); the incomplete estimate has q_GL=0.238. Both attenuated and nonzero estimates are retained below and in the all-endpoint figures/CSV.

Across focused beta3 G/L cells, median |spline−base rho| is {np.median(abs(primary.effect_spline-primary.effect_base)):.4f}; maximum is {np.max(abs(primary.effect_spline-primary.effect_base)):.4f}. Across both cohorts/endpoints/components, beta1/beta3 adjusted G/L effect correlation is {np.corrcoef(beta[1],beta[3])[0,1]:.4f}. These are descriptive stability summaries, not grounds to select beta or age model; the full paired estimates and q scopes remain exported.

No focused study interaction passes within-focused q<0.05 ({int(interaction.q_focused.lt(.05).sum())}/{int(interaction.p.notna().sum())} estimable). This does not establish equivalent effects, particularly where subgroup sizes are small. Full directions, N, HC3 intervals, FL/HC3 p/q, variance diagnostics and ROS/MAP forests cover all eight endpoints.

In original global-cognition donors, adding G/L beyond existing Connectivity+PC1 gives M747 DLPFC ΔR²=0.0527 [0.0153, 0.1127], joint FL p=0.0009, q_focused=0.0183; M1064 gives 0.0452, full_cohort M1643 0.0533, and M1116 0.0228 (q=0.0867). Across all endpoints/betas, {int(joint.q_focused.lt(.05).sum())} original joint rows pass focused q<0.05. G/L correlations can be about −0.95 to −0.97 with VIF near 29; individual coefficients are unstable. These are in-sample explanatory improvements with historical comparator exposure, not cross-validated predictive gain or evidence of resilience. Incomplete-cohort comparator analyses remain unavailable.

"""
        + "\n".join(endpoint_tables)
        + f"""
## Discussion and conclusions

Adjustment does not uniformly erase original-target associations. Several focused cognition effects remain similar after controlling the declared background variables, while incomplete-target estimates are often weaker before adjustment. The supplied path-adjusted slope does not reproduce all global-cognition findings. The broad catalog and focused correction families answer different multiplicity questions; a small endpoint-specific q is not a broad scan discovery.

This analysis cannot identify causal protection, selection mechanisms, or equivalence of studies/cohorts. Complete-case selection, tissue availability, historic module/PC1 discovery exposure, absent incomplete comparators, uncertain slope provenance, unmeasured cellular/technical variation, sparse subgroups and Monte Carlo resolution limit interpretation. No matching, weighting, new conditional calibration, outcome-guided model changes or prediction training was performed. Further independent cohorts and a higher **family-wide** permutation budget would address different remaining uncertainties.

## Verification and reproduction

`verification.json` records independent SciPy BH recomputation, actual donor membership reconstruction for every catalog row, native score and metadata/allocation hashes, exact alias equality, preserved baseline effects/nominal p, and all 98 N=4 repairs. Unit tests independently check FL against explicit model refits, HC3 against statsmodels 0.14.5, natural spline span, exact tied pairings, nonzero/reproducible MC, confounding/signal recovery, df/categorical guards and B/C membership. Implementation tests establish calculation correctness, not biological validity. The final test/UI/preservation receipt is `engineering_validation.json`.

From the application directory, using its existing numerical environment:

```bash
../../.venv/bin/python scripts/run_control_calibrated_validation.py all --workers 4
```

This freezes a new source/code-identified specification when needed, resumes checkpointed stages, summarizes, audits, writes reports and activates **only the local validation pointer**. To finish an already frozen workspace without replacing its specification:

```bash
../../.venv/bin/python scripts/run_control_calibrated_validation.py finalize --workers 4
```

Completed bundles are immutable; repeated calls use their recorded results. Reproduce the exact archived numeric implementation from local `artifacts/calibrated_validation_20261104/runtime_sources`; `executed_code.json` links its byte hashes and AST semantic hashes to formatted committed source. A new revision/specification should be used for any scientific method/input change.

Start the app with the existing Streamlit launcher. In **Associations**, **Statistics**, or **Correlation heatmaps**, choose a **calibrated beta3/beta1** source and **Association results → Covariate validation / 8-outcome scan**. Choose Triplets, ROS/MAP, Focused comparisons, Composition/missingness, Baseline inference repair, or Methods/provenance. The normal donor-score view retains the separate **20-outcome baseline**, with repaired/original inference selection. Residual diagnostics are explicitly in residual midrank units. Full filtered CSV downloads retain exclusions and correction provenance; plot previews never define the correction family.

Local generated bundle: `{root.relative_to(APP)}`. Large matrices, raw donor identities, clinical donor-level records and checkpoint caches remain local and uncommitted. Aggregate CSVs here cover every focused endpoint/measure/beta/cohort/model, including secondary S/A. Each endpoint has beta3 and beta1 G/L triplet and ROS/MAP SVG forests under `figures/`. `execution_summary.json`, `analysis_spec.json`, `provenance_inventory.json` and `verification.json` preserve the audit contract. No Drive upload, production pointer change, push, merge or deployment was authorized or performed.

Method references: [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/), [FSL exchangeability blocks](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/exchangeability_blocks.html), [statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html).
"""
    )
    (dest / "report.md").write_text(body)
    files = {
        str(p.relative_to(dest)): sha(p)
        for p in dest.rglob("*")
        if p.is_file()
        and p.name not in ["report_manifest.json", "engineering_validation.json"]
    }
    atomic_json(
        dest / "report_manifest.json",
        dict(analysis_id=spec["analysis_id"], aggregate_only=True, files=files),
    )
    atomic_json(
        root / "report_complete.json",
        dict(
            status="complete",
            analysis_id=spec["analysis_id"],
            destination=str(dest),
            manifest_sha256=sha(dest / "report_manifest.json"),
        ),
    )
    return dict(report=str(dest / "report.md"), figures=32, focused_rows=len(focused))
