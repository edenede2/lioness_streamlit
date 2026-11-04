# ROSMAP control-calibrated donor connectivity validation

Generated locally on 2026-11-04. Analysis `8f0ba6f5eeacf089364056c444a290c1a71056306ff8edf08c6062dd8d1f2ea4`; source release `20261103_bf01d5b35d51`; starting code `35db0003fd94408403ba3f8d8a6f61e347b6f43d`. The specification was frozen before the catalog outcome scan. This is an exploratory association study, not biological validation or a prediction experiment.

## Introduction and background

We tested whether associations of frozen native gain (G), loss (L), net change (S), and magnitude atypicality (A) persist after demographic/study adjustment, and distinguished complete-case selection from adjustment. G/L are separate primary measures; S/A are secondary because S=G−L and A=G+L. Stored scores, allocations, calibration, denominators, module assignments, and supplied outcomes were not changed.

## Dataset and scope

| Family | Modules | Dataset |
| --- | --- | --- |
| control_derived | 186 | brain |
| full_cohort | 154 | brain |
| muscle_all_donors_l4 | 173 | motor |
| muscle_control_l4 | 155 | motor |
| single_region_complete_case_l3 | 380 | brain |
| single_region_full_tissue_l3 | 368 | brain |
| single_motor_tissue_complete_case_l3 | 281 | motor |

All 1,697 ready modules in seven registered families were scanned. The separate `matched_control_brain_l4` 247-module namespace has no compatible ready calibrated artifacts and was unavailable; historical `control_derived` IDs are a different namespace. Native physical aliases are canonicalized only within a module, with identical edges, availability, values and denominators. The old baseline repair retains its original aliases and correction membership.

Original targets have frozen role=target and original-analysis membership. Incomplete targets have role=target and lack at least one dataset-wide tissue; their donor IDs are disjoint from original targets. Brain allocation R/C is 82/41; motor is 60/30. Brain has 14 other complete targets, and the motor-regional metadata have 7; these are excluded from the two main cohorts. Maximum-component baseline subsets overlap these cohorts and are descriptive.

| input_variant | cohort | n | ROS | MAP | missing_study | applicable | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brain | original | 327 | 155 | 172 | 0 | True |  |
| brain | incomplete | 793 | 317 | 467 | 9 | True |  |
| motor | original | 267 | 0 | 267 | 0 | False | single_study_MAP_motor |
| motor | incomplete | 142 | 0 | 142 | 0 | False | single_study_MAP_motor |
| motor_regional | original | 267 | 0 | 267 | 0 | False | single_study_MAP_motor |
| motor_regional | incomplete | 143 | 0 | 143 | 0 | False | single_study_MAP_motor |

## Methods

### Triplet and nuisance design

A uses all available score/outcome target pairs. B uses exactly C's complete-case donors. C ranks score, outcome, age and education with deterministic midranks within those exact donors. Sex is categorical (Code 0 reference; Code 1 indicator), study is categorical (MAP reference; ROS indicator), with an intercept. Constant nuisance columns are removed and recorded; other deficiencies are explicit exclusions. With nuisance matrix Z and midranks r, the reported C effect is

$$
\rho_{partial}=\operatorname{cor}\left((I-P_Z)r_x,(I-P_Z)r_y\right),\qquad P_Z=Z(Z^T Z)^{-1}Z^T.
$$

This is Pearson correlation of residual midranks, with residual degrees of freedom N−rank(Z)−1. Adjusted inference requires N≥30, df≥20, a full-rank design and nonconstant residual vectors. Small estimable effects remain descriptive with NA adjusted p/q. All 774,144 B/C pairs have identical saved donor hashes. A/B differ in 18,736 planned metric/beta/endpoint cells; this comparison reflects observed missing covariates, not a causal selection estimate. See `focused_selection_adjustment.csv` and full local deltas.

### Inference, uncertainty and multiplicity

Unadjusted inference permutes one rank vector. Unique multiset pairings are enumerated when their actual space is at most min(100,000, floor(8,000,000/N)); tied configurations are weighted correctly. Otherwise all tests use the frozen 9,999-draw budget. For adjusted tests, Freedman–Lane fits the reduced nuisance outcome model, permutes its residuals within study, adds fitted nuisance values back and refits the score-plus-nuisance model. Initial ranks and design are fixed during permutation. Orthogonal projections implement the equivalent full-model refit. The statistic is absolute partial correlation, monotone in absolute full-model t for the fixed design.

$$
y^*=P_Zr_y+\Pi(I-P_Z)r_y,\qquad p_{MC}=\frac{b+1}{9999+1}.
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

| dataset | cohort | planned | completed | estimable | excluded | matched_B_C_pairs |
| --- | --- | --- | --- | --- | --- | --- |
| brain | original | 1175232 | 1175232 | 1175192 | 40 | 234944 |
| brain | incomplete | 1175232 | 1175232 | 800256 | 374976 | 234944 |
| motor | original | 456384 | 456384 | 456384 | 0 | 152128 |
| motor | incomplete | 456384 | 456384 | 322752 | 133632 | 152128 |

All **3,263,232** planned rows completed; **2,754,584** have estimable p and **508,648** are explicit exclusions; failed required jobs: **0**. This includes all primary triplets, brain study strata and the prespecified focused spline. Native preparation checked 3,394 maximum-scope source files and used ~439 MiB of summary arrays, without an all-edge donor cube. Numeric job counts were 1,785 repair shards and 1,133 scan shards, using four workers with two BLAS threads per worker. Repair took 174.4s; main scan 1041.3s; focused jobs 233.4s. Observed maximum process RSS was 0.92 GiB in repair and 0.80 GiB in scan (worker lifetime peaks, **not** whole application peak). Benchmark: 2,048 vectors, N=275, 9,999 permutations in 0.311s; RSS 0.92 GiB.

| analysis | planned | completed | estimable | excluded |
| --- | --- | --- | --- | --- |
| interaction | 1024 | 1024 | 992 | 32 |
| comparators | 1152 | 1152 | 576 | 576 |
| incremental | 256 | 256 | 128 | 128 |

All 256 focused contexts completed. Interaction exclusions require at least 30 donors per study. The 576 incomplete comparator rows and 128 incomplete joint rows are explicitly unavailable because the required historical comparator set has no shared incomplete donors. They were not filled with another family or cohort.

### Baseline repair

| file | rows | numeric_repaired | old_zero_p | new_zero_p | old_q_scan_lt05 | new_q_scan_lt05 |
| --- | --- | --- | --- | --- | --- | --- |
| brain_complete_450.parquet | 832160 | 624120 | 0 | 0 | 1099 | 24 |
| brain_incomplete_region_validation.parquet | 832160 | 624120 | 98 | 0 | 30591 | 30197 |
| brain_maximum_component.parquet | 832160 | 624120 | 0 | 0 | 69565 | 69339 |
| motor_complete_450.parquet | 505440 | 379080 | 0 | 0 | 0 | 0 |
| motor_incomplete_region_validation.parquet | 505440 | 379080 | 0 | 0 | 0 | 0 |
| motor_maximum_component.parquet | 505440 | 379080 | 0 | 0 | 4 | 4 |

All **3,009,600** numeric Spearman tests were repaired; all **4,012,800** baseline rows participated in recomputed original families, including unchanged nominal tests. All **98** original N=4 perfect-correlation p=q_scan=0 rows were reproduced and now have exact p=2/24=1/12. No new empirical p is zero. Effect values and Kruskal–Wallis p values are unchanged, and the original release remains intact. Changes in broad significance, especially original brain 1,099→24, are strongly constrained by the Monte Carlo floor and cannot be called biological attenuation. The complete before/after table is local at `repaired_baseline/before_after.parquet` and downloadable in the app.

### Main and focused effect findings

Beta3 primary G/L counts below distinguish broad q_scan from endpoint-specific q_GL; neither count measures biological validation.

| dataset | cohort | model | q_scan_lt05 | q_GL_outcome_lt05 |
| --- | --- | --- | --- | --- |
| brain | incomplete | A | 0 | 122 |
| brain | incomplete | B | 0 | 122 |
| brain | incomplete | C | 0 | 0 |
| brain | original | A | 0 | 861 |
| brain | original | B | 0 | 861 |
| brain | original | C | 0 | 890 |
| motor | incomplete | A | 0 | 0 |
| motor | incomplete | B | 0 | 0 |
| motor | incomplete | C | 0 | 0 |
| motor | original | A | 0 | 13 |
| motor | original | B | 0 | 13 |
| motor | original | C | 0 | 0 |

For M747 DLPFC gain and global cognition, original A is −0.227 (N=275), C is **−0.235**, 95% CI **[−0.342, −0.116]**, p=0.0002, q_GL=0.0193, q_scan=0.0734. B uses the same donors. In incomplete donors, A is −0.046 and C **−0.034**, CI **[−0.112, 0.047]**, N=620. The original signal is stable to this base adjustment, whereas the incomplete estimate is already much weaker before adjustment. This is an observed effect contrast, not proof of a selection mechanism or formal cohort interaction.

For the supplied path-adjusted cognition slope, M747 DLPFC gain C is −0.119 [−0.247, 0.012] in original donors (N=216), and 0.004 [−0.083, 0.085] in incomplete donors (N=528). For global pathology, corresponding estimates are 0.205 [0.096, 0.310] (N=327) and 0.136 [0.068, 0.206] (N=746); the incomplete estimate has q_GL=0.238. Both attenuated and nonzero estimates are retained below and in the all-endpoint figures/CSV.

Across focused beta3 G/L cells, median |spline−base rho| is 0.0044; maximum is 0.0214. Across both cohorts/endpoints/components, beta1/beta3 adjusted G/L effect correlation is 0.9948. These are descriptive stability summaries, not grounds to select beta or age model; the full paired estimates and q scopes remain exported.

No focused study interaction passes within-focused q<0.05 (0/992 estimable). This does not establish equivalent effects, particularly where subgroup sizes are small. Full directions, N, HC3 intervals, FL/HC3 p/q, variance diagnostics and ROS/MAP forests cover all eight endpoints.

In original global-cognition donors, adding G/L beyond existing Connectivity+PC1 gives M747 DLPFC ΔR²=0.0527 [0.0153, 0.1127], joint FL p=0.0009, q_focused=0.0183; M1064 gives 0.0452, full_cohort M1643 0.0533, and M1116 0.0228 (q=0.0867). Across all endpoints/betas, 34 original joint rows pass focused q<0.05. G/L correlations can be about −0.95 to −0.97 with VIF near 29; individual coefficients are unstable. These are in-sample explanatory improvements with historical comparator exposure, not cross-validated predictive gain or evidence of resilience. Incomplete-cohort comparator analyses remain unavailable.

### TS_DLPFC — beta3 adjusted G/L, every endpoint

| family | module | outcome | metric_family | incomplete | original |
| --- | --- | --- | --- | --- | --- |
| control_derived | 747 | amyloid | gain | 0.093 [0.024, 0.164]; 740 | 0.182 [0.072, 0.294]; 327 |
| control_derived | 747 | amyloid | loss | -0.105 [-0.175, -0.037]; 740 | -0.178 [-0.286, -0.072]; 327 |
| control_derived | 747 | cogn_global | gain | -0.034 [-0.112, 0.047]; 620 | -0.235 [-0.342, -0.116]; 275 |
| control_derived | 747 | cogn_global | loss | 0.032 [-0.052, 0.109]; 620 | 0.205 [0.083, 0.315]; 275 |
| control_derived | 747 | cogng_demog_slope | gain | -0.073 [-0.148, 0.008]; 676 | -0.231 [-0.336, -0.120]; 297 |
| control_derived | 747 | cogng_demog_slope | loss | 0.084 [0.005, 0.160]; 676 | 0.220 [0.112, 0.326]; 297 |
| control_derived | 747 | cogng_path_slope | gain | 0.004 [-0.083, 0.085]; 528 | -0.119 [-0.247, 0.012]; 216 |
| control_derived | 747 | cogng_path_slope | loss | -0.010 [-0.093, 0.074]; 528 | 0.141 [0.010, 0.270]; 216 |
| control_derived | 747 | gpath | gain | 0.136 [0.068, 0.206]; 746 | 0.205 [0.096, 0.310]; 327 |
| control_derived | 747 | gpath | loss | -0.147 [-0.218, -0.075]; 746 | -0.206 [-0.311, -0.098]; 327 |
| control_derived | 747 | motor10_demog_slope | gain | -0.020 [-0.095, 0.054]; 641 | -0.102 [-0.215, 0.020]; 277 |
| control_derived | 747 | motor10_demog_slope | loss | 0.000 [-0.070, 0.073]; 641 | 0.086 [-0.036, 0.199]; 277 |
| control_derived | 747 | sqrt_parksc_demog_slope | gain | 0.052 [-0.022, 0.129]; 665 | 0.012 [-0.104, 0.130]; 293 |
| control_derived | 747 | sqrt_parksc_demog_slope | loss | -0.048 [-0.123, 0.027]; 665 | 0.010 [-0.110, 0.127]; 293 |
| control_derived | 747 | tangles | gain | 0.109 [0.039, 0.177]; 740 | 0.190 [0.083, 0.296]; 326 |
| control_derived | 747 | tangles | loss | -0.142 [-0.211, -0.071]; 740 | -0.178 [-0.283, -0.071]; 326 |
| control_derived | 1064 | amyloid | gain | 0.070 [-0.002, 0.143]; 740 | 0.148 [0.035, 0.255]; 327 |
| control_derived | 1064 | amyloid | loss | -0.049 [-0.123, 0.023]; 740 | -0.154 [-0.259, -0.046]; 327 |
| control_derived | 1064 | cogn_global | gain | -0.004 [-0.083, 0.077]; 620 | -0.230 [-0.342, -0.110]; 275 |
| control_derived | 1064 | cogn_global | loss | 0.000 [-0.078, 0.079]; 620 | 0.171 [0.049, 0.287]; 275 |
| control_derived | 1064 | cogng_demog_slope | gain | 0.000 [-0.081, 0.080]; 676 | -0.200 [-0.313, -0.085]; 297 |
| control_derived | 1064 | cogng_demog_slope | loss | 0.009 [-0.071, 0.089]; 676 | 0.170 [0.053, 0.287]; 297 |
| control_derived | 1064 | cogng_path_slope | gain | 0.035 [-0.051, 0.115]; 528 | -0.088 [-0.217, 0.049]; 216 |
| control_derived | 1064 | cogng_path_slope | loss | -0.032 [-0.112, 0.052]; 528 | 0.088 [-0.050, 0.220]; 216 |
| control_derived | 1064 | gpath | gain | 0.071 [-0.000, 0.143]; 746 | 0.183 [0.086, 0.282]; 327 |
| control_derived | 1064 | gpath | loss | -0.052 [-0.123, 0.019]; 746 | -0.216 [-0.314, -0.119]; 327 |
| control_derived | 1064 | motor10_demog_slope | gain | -0.008 [-0.083, 0.067]; 641 | -0.109 [-0.224, 0.006]; 277 |
| control_derived | 1064 | motor10_demog_slope | loss | -0.001 [-0.077, 0.077]; 641 | 0.074 [-0.036, 0.191]; 277 |
| control_derived | 1064 | sqrt_parksc_demog_slope | gain | 0.008 [-0.070, 0.085]; 665 | -0.004 [-0.121, 0.110]; 293 |
| control_derived | 1064 | sqrt_parksc_demog_slope | loss | -0.014 [-0.088, 0.058]; 665 | 0.045 [-0.079, 0.159]; 293 |
| control_derived | 1064 | tangles | gain | 0.052 [-0.018, 0.119]; 740 | 0.159 [0.055, 0.268]; 326 |
| control_derived | 1064 | tangles | loss | -0.061 [-0.130, 0.011]; 740 | -0.177 [-0.283, -0.072]; 326 |
| control_derived | 1116 | amyloid | gain | 0.017 [-0.055, 0.093]; 740 | 0.072 [-0.039, 0.183]; 327 |
| control_derived | 1116 | amyloid | loss | -0.036 [-0.109, 0.034]; 740 | -0.118 [-0.224, -0.009]; 327 |
| control_derived | 1116 | cogn_global | gain | -0.006 [-0.087, 0.075]; 620 | -0.163 [-0.279, -0.039]; 275 |
| control_derived | 1116 | cogn_global | loss | 0.005 [-0.073, 0.086]; 620 | 0.179 [0.063, 0.293]; 275 |
| control_derived | 1116 | cogng_demog_slope | gain | 0.000 [-0.079, 0.079]; 676 | -0.065 [-0.171, 0.053]; 297 |
| control_derived | 1116 | cogng_demog_slope | loss | 0.025 [-0.055, 0.103]; 676 | 0.138 [0.026, 0.244]; 297 |
| control_derived | 1116 | cogng_path_slope | gain | 0.004 [-0.086, 0.087]; 528 | 0.039 [-0.099, 0.175]; 216 |
| control_derived | 1116 | cogng_path_slope | loss | 0.029 [-0.060, 0.120]; 528 | -0.019 [-0.150, 0.112]; 216 |
| control_derived | 1116 | gpath | gain | 0.029 [-0.041, 0.099]; 746 | 0.079 [-0.024, 0.194]; 327 |
| control_derived | 1116 | gpath | loss | -0.041 [-0.108, 0.029]; 746 | -0.123 [-0.227, -0.015]; 327 |
| control_derived | 1116 | motor10_demog_slope | gain | 0.033 [-0.039, 0.106]; 641 | -0.082 [-0.197, 0.033]; 277 |
| control_derived | 1116 | motor10_demog_slope | loss | -0.022 [-0.098, 0.053]; 641 | 0.071 [-0.039, 0.186]; 277 |
| control_derived | 1116 | sqrt_parksc_demog_slope | gain | 0.050 [-0.023, 0.129]; 665 | 0.022 [-0.091, 0.142]; 293 |
| control_derived | 1116 | sqrt_parksc_demog_slope | loss | -0.059 [-0.138, 0.016]; 665 | 0.040 [-0.078, 0.157]; 293 |
| control_derived | 1116 | tangles | gain | 0.013 [-0.060, 0.087]; 740 | 0.121 [0.009, 0.231]; 326 |
| control_derived | 1116 | tangles | loss | -0.024 [-0.097, 0.047]; 740 | -0.178 [-0.286, -0.072]; 326 |
| full_cohort | 1643 | amyloid | gain | 0.091 [0.019, 0.169]; 740 | 0.182 [0.072, 0.288]; 327 |
| full_cohort | 1643 | amyloid | loss | -0.080 [-0.160, -0.007]; 740 | -0.188 [-0.292, -0.076]; 327 |
| full_cohort | 1643 | cogn_global | gain | -0.025 [-0.105, 0.057]; 620 | -0.285 [-0.394, -0.169]; 275 |
| full_cohort | 1643 | cogn_global | loss | 0.021 [-0.060, 0.103]; 620 | 0.260 [0.136, 0.373]; 275 |
| full_cohort | 1643 | cogng_demog_slope | gain | -0.052 [-0.133, 0.029]; 676 | -0.271 [-0.372, -0.156]; 297 |
| full_cohort | 1643 | cogng_demog_slope | loss | 0.051 [-0.029, 0.127]; 676 | 0.275 [0.159, 0.380]; 297 |
| full_cohort | 1643 | cogng_path_slope | gain | 0.002 [-0.084, 0.085]; 528 | -0.167 [-0.287, -0.041]; 216 |
| full_cohort | 1643 | cogng_path_slope | loss | 0.004 [-0.078, 0.093]; 528 | 0.185 [0.051, 0.309]; 216 |
| full_cohort | 1643 | gpath | gain | 0.111 [0.042, 0.182]; 746 | 0.218 [0.116, 0.319]; 327 |
| full_cohort | 1643 | gpath | loss | -0.097 [-0.169, -0.026]; 746 | -0.246 [-0.350, -0.143]; 327 |
| full_cohort | 1643 | motor10_demog_slope | gain | -0.025 [-0.101, 0.047]; 641 | -0.101 [-0.211, 0.016]; 277 |
| full_cohort | 1643 | motor10_demog_slope | loss | 0.017 [-0.058, 0.094]; 641 | 0.097 [-0.023, 0.209]; 277 |
| full_cohort | 1643 | sqrt_parksc_demog_slope | gain | 0.047 [-0.029, 0.124]; 665 | 0.027 [-0.090, 0.148]; 293 |
| full_cohort | 1643 | sqrt_parksc_demog_slope | loss | -0.063 [-0.139, 0.012]; 665 | -0.010 [-0.134, 0.106]; 293 |
| full_cohort | 1643 | tangles | gain | 0.090 [0.021, 0.158]; 740 | 0.175 [0.070, 0.282]; 326 |
| full_cohort | 1643 | tangles | loss | -0.090 [-0.158, -0.018]; 740 | -0.190 [-0.295, -0.084]; 326 |

### CT_DLPFC__PCG — beta3 adjusted G/L, every endpoint

| family | module | outcome | metric_family | incomplete | original |
| --- | --- | --- | --- | --- | --- |
| control_derived | 747 | amyloid | gain | 0.108 [-0.050, 0.256]; 176 | 0.158 [0.047, 0.264]; 327 |
| control_derived | 747 | amyloid | loss | -0.102 [-0.248, 0.053]; 176 | -0.148 [-0.253, -0.037]; 327 |
| control_derived | 747 | cogn_global | gain | -0.040 [-0.208, 0.125]; 162 | -0.214 [-0.322, -0.102]; 275 |
| control_derived | 747 | cogn_global | loss | 0.068 [-0.097, 0.229]; 162 | 0.189 [0.077, 0.303]; 275 |
| control_derived | 747 | cogng_demog_slope | gain | -0.049 [-0.212, 0.101]; 157 | -0.220 [-0.326, -0.110]; 297 |
| control_derived | 747 | cogng_demog_slope | loss | 0.087 [-0.072, 0.256]; 157 | 0.210 [0.100, 0.316]; 297 |
| control_derived | 747 | cogng_path_slope | gain | 0.033 [-0.157, 0.206]; 116 | -0.070 [-0.204, 0.059]; 216 |
| control_derived | 747 | cogng_path_slope | loss | -0.037 [-0.211, 0.153]; 116 | 0.096 [-0.036, 0.223]; 216 |
| control_derived | 747 | gpath | gain | 0.209 [0.070, 0.348]; 176 | 0.207 [0.100, 0.307]; 327 |
| control_derived | 747 | gpath | loss | -0.230 [-0.371, -0.087]; 176 | -0.192 [-0.296, -0.086]; 327 |
| control_derived | 747 | motor10_demog_slope | gain | 0.024 [-0.148, 0.187]; 152 | -0.116 [-0.228, 0.002]; 277 |
| control_derived | 747 | motor10_demog_slope | loss | -0.028 [-0.187, 0.140]; 152 | 0.099 [-0.016, 0.215]; 277 |
| control_derived | 747 | sqrt_parksc_demog_slope | gain | 0.027 [-0.134, 0.193]; 153 | -0.023 [-0.141, 0.096]; 293 |
| control_derived | 747 | sqrt_parksc_demog_slope | loss | -0.047 [-0.211, 0.115]; 153 | 0.032 [-0.094, 0.151]; 293 |
| control_derived | 747 | tangles | gain | 0.172 [0.022, 0.311]; 176 | 0.196 [0.093, 0.294]; 326 |
| control_derived | 747 | tangles | loss | -0.199 [-0.337, -0.050]; 176 | -0.174 [-0.272, -0.071]; 326 |
| control_derived | 1064 | amyloid | gain | 0.110 [-0.036, 0.256]; 176 | 0.178 [0.069, 0.285]; 327 |
| control_derived | 1064 | amyloid | loss | -0.098 [-0.242, 0.049]; 176 | -0.178 [-0.283, -0.067]; 327 |
| control_derived | 1064 | cogn_global | gain | -0.020 [-0.186, 0.142]; 162 | -0.214 [-0.328, -0.096]; 275 |
| control_derived | 1064 | cogn_global | loss | 0.028 [-0.121, 0.183]; 162 | 0.168 [0.048, 0.282]; 275 |
| control_derived | 1064 | cogng_demog_slope | gain | 0.028 [-0.140, 0.183]; 157 | -0.208 [-0.321, -0.092]; 297 |
| control_derived | 1064 | cogng_demog_slope | loss | -0.012 [-0.162, 0.141]; 157 | 0.178 [0.060, 0.295]; 297 |
| control_derived | 1064 | cogng_path_slope | gain | 0.085 [-0.093, 0.246]; 116 | -0.081 [-0.211, 0.057]; 216 |
| control_derived | 1064 | cogng_path_slope | loss | -0.071 [-0.239, 0.106]; 116 | 0.056 [-0.087, 0.189]; 216 |
| control_derived | 1064 | gpath | gain | 0.202 [0.058, 0.328]; 176 | 0.228 [0.128, 0.326]; 327 |
| control_derived | 1064 | gpath | loss | -0.197 [-0.325, -0.058]; 176 | -0.245 [-0.343, -0.150]; 327 |
| control_derived | 1064 | motor10_demog_slope | gain | 0.030 [-0.131, 0.189]; 152 | -0.149 [-0.260, -0.038]; 277 |
| control_derived | 1064 | motor10_demog_slope | loss | -0.040 [-0.200, 0.110]; 152 | 0.118 [0.006, 0.234]; 277 |
| control_derived | 1064 | sqrt_parksc_demog_slope | gain | -0.055 [-0.213, 0.105]; 153 | -0.048 [-0.170, 0.081]; 293 |
| control_derived | 1064 | sqrt_parksc_demog_slope | loss | 0.055 [-0.112, 0.216]; 153 | 0.070 [-0.053, 0.196]; 293 |
| control_derived | 1064 | tangles | gain | 0.137 [-0.011, 0.280]; 176 | 0.197 [0.100, 0.296]; 326 |
| control_derived | 1064 | tangles | loss | -0.121 [-0.258, 0.025]; 176 | -0.209 [-0.307, -0.108]; 326 |
| control_derived | 1116 | amyloid | gain | 0.088 [-0.064, 0.230]; 176 | 0.106 [-0.002, 0.214]; 327 |
| control_derived | 1116 | amyloid | loss | -0.089 [-0.240, 0.077]; 176 | -0.115 [-0.219, -0.002]; 327 |
| control_derived | 1116 | cogn_global | gain | -0.038 [-0.206, 0.130]; 162 | -0.134 [-0.242, -0.020]; 275 |
| control_derived | 1116 | cogn_global | loss | 0.041 [-0.128, 0.212]; 162 | 0.146 [0.035, 0.258]; 275 |
| control_derived | 1116 | cogng_demog_slope | gain | 0.006 [-0.170, 0.165]; 157 | -0.070 [-0.183, 0.041]; 297 |
| control_derived | 1116 | cogng_demog_slope | loss | 0.031 [-0.130, 0.195]; 157 | 0.120 [0.008, 0.231]; 297 |
| control_derived | 1116 | cogng_path_slope | gain | 0.059 [-0.139, 0.242]; 116 | 0.067 [-0.062, 0.204]; 216 |
| control_derived | 1116 | cogng_path_slope | loss | -0.041 [-0.226, 0.160]; 116 | -0.028 [-0.158, 0.104]; 216 |
| control_derived | 1116 | gpath | gain | 0.200 [0.051, 0.332]; 176 | 0.085 [-0.021, 0.192]; 327 |
| control_derived | 1116 | gpath | loss | -0.203 [-0.338, -0.059]; 176 | -0.084 [-0.189, 0.023]; 327 |
| control_derived | 1116 | motor10_demog_slope | gain | 0.004 [-0.171, 0.162]; 152 | -0.102 [-0.216, 0.012]; 277 |
| control_derived | 1116 | motor10_demog_slope | loss | 0.032 [-0.123, 0.196]; 152 | 0.107 [-0.008, 0.220]; 277 |
| control_derived | 1116 | sqrt_parksc_demog_slope | gain | -0.066 [-0.230, 0.106]; 153 | -0.042 [-0.156, 0.077]; 293 |
| control_derived | 1116 | sqrt_parksc_demog_slope | loss | 0.047 [-0.126, 0.213]; 153 | 0.092 [-0.031, 0.211]; 293 |
| control_derived | 1116 | tangles | gain | 0.158 [0.009, 0.295]; 176 | 0.170 [0.058, 0.280]; 326 |
| control_derived | 1116 | tangles | loss | -0.178 [-0.318, -0.027]; 176 | -0.171 [-0.285, -0.058]; 326 |
| full_cohort | 1643 | amyloid | gain | 0.096 [-0.053, 0.237]; 176 | 0.139 [0.027, 0.243]; 327 |
| full_cohort | 1643 | amyloid | loss | -0.104 [-0.243, 0.041]; 176 | -0.154 [-0.256, -0.044]; 327 |
| full_cohort | 1643 | cogn_global | gain | -0.007 [-0.167, 0.157]; 162 | -0.233 [-0.347, -0.112]; 275 |
| full_cohort | 1643 | cogn_global | loss | -0.012 [-0.168, 0.143]; 162 | 0.220 [0.099, 0.333]; 275 |
| full_cohort | 1643 | cogng_demog_slope | gain | 0.032 [-0.138, 0.181]; 157 | -0.230 [-0.337, -0.114]; 297 |
| full_cohort | 1643 | cogng_demog_slope | loss | -0.032 [-0.185, 0.127]; 157 | 0.245 [0.132, 0.353]; 297 |
| full_cohort | 1643 | cogng_path_slope | gain | 0.138 [-0.051, 0.304]; 116 | -0.097 [-0.224, 0.038]; 216 |
| full_cohort | 1643 | cogng_path_slope | loss | -0.143 [-0.309, 0.042]; 116 | 0.127 [-0.011, 0.259]; 216 |
| full_cohort | 1643 | gpath | gain | 0.164 [0.019, 0.298]; 176 | 0.195 [0.088, 0.299]; 327 |
| full_cohort | 1643 | gpath | loss | -0.176 [-0.304, -0.034]; 176 | -0.215 [-0.315, -0.111]; 327 |
| full_cohort | 1643 | motor10_demog_slope | gain | 0.028 [-0.140, 0.193]; 152 | -0.101 [-0.215, 0.012]; 277 |
| full_cohort | 1643 | motor10_demog_slope | loss | -0.048 [-0.207, 0.119]; 152 | 0.117 [0.001, 0.227]; 277 |
| full_cohort | 1643 | sqrt_parksc_demog_slope | gain | 0.011 [-0.147, 0.173]; 153 | -0.009 [-0.135, 0.113]; 293 |
| full_cohort | 1643 | sqrt_parksc_demog_slope | loss | -0.039 [-0.198, 0.127]; 153 | 0.012 [-0.109, 0.137]; 293 |
| full_cohort | 1643 | tangles | gain | 0.131 [-0.020, 0.274]; 176 | 0.168 [0.064, 0.268]; 326 |
| full_cohort | 1643 | tangles | loss | -0.123 [-0.261, 0.022]; 176 | -0.171 [-0.270, -0.067]; 326 |

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

Local generated bundle: `data/control_calibrated_validation_v1/8f0ba6f5eeacf089364056c444a290c1a71056306ff8edf08c6062dd8d1f2ea4`. Large matrices, raw donor identities, clinical donor-level records and checkpoint caches remain local and uncommitted. Aggregate CSVs here cover every focused endpoint/measure/beta/cohort/model, including secondary S/A. Each endpoint has beta3 and beta1 G/L triplet and ROS/MAP SVG forests under `figures/`. `execution_summary.json`, `analysis_spec.json`, `provenance_inventory.json` and `verification.json` preserve the audit contract. No Drive upload, production pointer change, push, merge or deployment was authorized or performed.

Method references: [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/), [FSL exchangeability blocks](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/exchangeability_blocks.html), [statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html).
