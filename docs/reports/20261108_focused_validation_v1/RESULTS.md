# Focused validation evidence — 8 November 2026

## Scope and main finding

Artifact: `20261108_focused_validation_v1`. This is a retrospective, conditional assessment of four preselected module namespaces, not external-cohort validation or a formal WGCNA preservation test. No module memberships, donor-score definitions, prediction models, or historical q-values were changed.

**Empirical summary:** Control network structure is more concordant within DLPFC than between DLPFC and PCG. M1116's between-region structure is particularly weak. New disjoint-cohort MDC comparisons are heterogeneous; ratio changes can reflect the Control denominator as much as AD. Some original-selected edges retain AD–Control or pathology support; none retain corrected global-cognition support under the specified edge-level test. Failure to reach FDR is not evidence of no effect.

## 1. Cohorts and independence

| Input | Inclusion and actual counts |
|---|---|
| Historical original score cohort | 450 complete-expression donors: 164 Control, 119 MCI, 167 AD |
| Associated discovery inputs | Control-derived artifact: 164 Controls; full-cohort discovery eigengene donor axis: **464**, not the 450 score subset |
| Incomplete-region population | 793 donors, excluding **all 464** complete-expression donors; 244 Control, 177 MCI, 350 AD, 22 unclassified |
| DLPFC incomplete | 752: 229 Control, 165 MCI, 339 AD, 19 unclassified |
| DLPFC–PCG incomplete | 176: 65 Control, 29 MCI, 77 AD, 5 unclassified |
| Legacy complete-donor held-out split | 100: 50 Control, 50 AD; among these Controls, 29 are R, 13 C, and only 8 T in the calibrated framework |
| Frozen calibrated roles | R=82 reference Controls; C=41 magnitude-calibration Controls; original independent T=327 (41 Control, 119 MCI, 167 AD) |

The 14 other complete-expression targets are outside both the original450 and incomplete793 comparisons. Diagnosis is not required for clinical-outcome tests. After the existing complete-covariate contract, clinical edge tests have original N=275 for cognition and 327 for gpath; incomplete N=620/746 in DLPFC and 162/176 in DLPFC–PCG. Component cohorts overlap and must not be summed. Exact diagnosis, role and observed-outcome counts, including separate matched-control namespace counts, are in [cohorts.csv](cohorts.csv).

**Historical MDC overlap:** current Control-derived historical input records include all 339 AD/229 Controls from incomplete DLPFC and all 77 AD/65 Controls from incomplete DLPFC–PCG. These historical networks are not independent validation baselines. This membership reconstruction uses the actual current input files, not an archived historical receipt. Full-cohort saved adjacency caches lack membership records: **independence not verified**, not zero overlap. Historical values and inference are retained separately. New complete450-versus-incomplete comparisons have verified zero donor overlap, also against both associated discovery donor axes. See [intersections.csv](intersections.csv).

## 2. Controls-only structure

Frozen, unique tissue–gene nodes and eligible unmasked edges were used. Group networks are Pearson signedAlt: TS power=3, CT power=2. These are group adjacency powers, **not** the calibrated donor-score beta. Missing expression cells use the frozen original R imputation means; absent tissues are never filled in. These are group networks from expression, not averages of donor LIONESS scores. Original Controls participated in module discovery/reference/calibration, so this is conditional preservation.

| Namespace/module | DLPFC magnitude concordance | DLPFC–PCG magnitude concordance | Pair hub concordance, DLPFC / PCG |
|---|---:|---:|---:|
| Control-derived M747 | 0.742 | 0.506 | 0.685 / 0.740 |
| Control-derived M1064 | 0.685 | 0.453 | 0.601 / 0.619 |
| Control-derived M1116 | 0.629 | **0.154** | 0.371 / 0.505 |
| Full-cohort M1643 | 0.657 | 0.386 | 0.643 / 0.541 |

Original Control N=164; incomplete Control N=229 for DLPFC and 65 for the pair. Eligible edge counts range from 2,850 to 139,435. Signed raw-correlation, signed-adjacency and magnitude concordances, densities, separate tissue hubs and resampling-valid counts are all saved in [structure.csv](structure.csv). Equal-N without-replacement sensitivity gives magnitude-concordance medians 0.722/0.408 (M747), 0.669/0.342 (M1064), 0.599/0.101 (M1116), and 0.632/0.301 (M1643), DLPFC/pair respectively.

Uncertainty uses 199 deterministic, study-stratified donor draws (base seed 611108 and context-specific derived seeds), rebuilding all network correlations/powers; a donor's tissues remain together. Re-estimating noisy networks attenuates the edge-concordance resampling distribution, sometimes below the observed point. **Its percentile endpoints are donor-resampling stability ranges, not calibrated population-concordance CIs.** No edge-as-independent-replicate p-value is calculated. Paired local heatmaps share node order and color limits; identity scatters use corresponding edges.

## 3. MDC decomposition

New **disjoint complete450 versus incomplete** group MDC is mean absolute AD adjacency divided by mean absolute Control adjacency. Original group N=167 AD/164 Controls; validation groups are as above. Zero/undefined denominators remain unavailable; no epsilon is inserted.

| Namespace/module | Component | Original MDC → incomplete MDC | Incomplete 95% donor-bootstrap interval |
|---|---|---:|---:|
| Control-derived M747 | DLPFC | 1.236 → 1.200 | 0.850–1.580 |
| Control-derived M747 | DLPFC–PCG | 1.008 → 1.918 | 1.101–2.325 |
| Control-derived M1064 | DLPFC | 0.982 → 0.810 | 0.577–1.101 |
| Control-derived M1064 | DLPFC–PCG | 0.902 → 1.117 | 0.746–1.571 |
| Control-derived M1116 | DLPFC | 0.660 → 0.894 | 0.560–1.179 |
| Control-derived M1116 | DLPFC–PCG | 0.429 → 1.095 | 0.766–1.417 |
| Full-cohort M1643 | DLPFC | 1.349 → 0.800 | 0.570–1.095 |
| Full-cohort M1643 | DLPFC–PCG | 1.247 → 1.131 | 0.796–1.456 |

For M747's pair, AD density is 0.08879→0.09346 while Control density is 0.08811→0.04873: the higher ratio is mainly a lower Control denominator. M1116's pair changes from AD/Control 0.01105/0.02577 to 0.02650/0.02421; AD density rises and the old loss direction does not reproduce. M1643 within-DLPFC changes from 0.03720/0.02758 to 0.01508/0.01885; AD density falls more than Control and the direction reverses. These are arithmetic decompositions, not causal explanations.

Numerator, denominator and ratio are rebuilt together in each donor draw. These new 199-draw percentile intervals are **not** permutation p-values or FDR claims. Control-density CV, zero/undefined draw counts and log2 changes are in [mdc_decomposition.csv](mdc_decomposition.csv).

The full saved MDC overview spans five partition/method catalogs and every available mask. [mdc_overview.csv](mdc_overview.csv) preserves planned, evaluable, original-significant and tested denominators, same direction, nominal/adjusted support, reversal, zero direction and unavailable/descriptive-only cases. “Support” requires same log2 direction plus the saved validation dual-null p/q<0.05. Original and validation correction families are not recalculated after filtering. Historical brain full-cohort method rows share group-MDC evidence; they are not independent replications. Sparse AC–PCG remains non-inferential. Historical sample/gene-null support, and any separately saved label-swap tests, are not interchanged with these bootstrap intervals.

Examples from that **historical, not independently verified** all-edge catalog: among original-significant evaluable modules, Control-derived DLPFC–PCG has 80/121 same-direction, 19/121 nominal and 0/121 adjusted support; full-cohort DLPFC–PCG has 43/48, 27/48 and 15/48, and full-cohort DLPFC has 15/15, 14/15 and 14/15. These denominators answer ratio-replication questions conditional on the saved families, not structural preservation or independent clinical validation. Control-derived DLPFC has no original-significant denominator under that saved correction.

## 4. Same original-ranked edges

The old R164 rankings concern a different readout. Therefore new lists were frozen from **original independent T donors only**, using the existing frozen R82 add-one signed-cube edge weight (score beta3) and C41 calibration: top20 |AD–Control Hedges g| plus separate top20 |adjusted partial-rank association| for cogn_global and gpath, with deterministic physical edge-code ties. The complete lists and original full-component BH values were saved **before validation values were read** for that context. The union has 461 module–edge hypotheses; overlapping edges across modules are not independent replications.

| Test | Frozen hypotheses | Same direction | Reversed | Same + p<0.05 | Same + selected-family FDR<0.05 |
|---|---:|---:|---:|---:|---:|
| AD–Control raw-edge g | 461 | 306 | 155 | 56 | 6 |
| Global cognition association | 461 | 294 | 167 | 24 | **0** |
| Global pathology association | 461 | 319 | 142 | 72 | 10 |

Of the corrected same-direction hits, all6 diagnosis hits and 9/10 pathology hits were also original full-search q<0.05. Diagnosis hits are M747 DLPFC(5) and M1643 pair(1); pathology hits are M747 DLPFC(8) and pair(2). No original cognition edge passes this particular full-edge-search correction; this is not the same family as a module-score association.

AD–Control g uses unadjusted raw signed weights, Welch p and 199 donor-bootstrap g intervals. Clinical associations use the existing age/sex/education/study partial-midrank contract, 9,999 study-restricted Freedman–Lane draws (minimum p=0.0001), and donor-bootstrap intervals. Validation BH covers **the whole frozen selected union**, separately by test/endpoint, not a displayed top20. Original full-search q and selected-family q retain distinct names. See public [edge_summary.csv](edge_summary.csv); exact edges, clinical effects, candidate receipts, original full searches, donor arrays and contribution tables remain local-only.

These edge intervals condition on the frozen R/C reference and calibration; they do not propagate reference/calibration estimation uncertainty.

Calibrated magnitude departure `a=(|w|−ν)/τ` and gain/loss contributions retain their implemented definition. Contributions divide by the **full frozen eligible component-edge count**, not the displayed selected count. Increasing absolute connectivity is different from increasing signed correlation or having a positive clinical association.

## 5. Existing adjusted evidence and remaining gaps

The app reads the exact, hash-verified [4 November public report](../control_calibrated_validation_20261104/), analysis ID `8f0ba6f5eeacf089364056c444a290c1a71056306ff8edf08c6062dd8d1f2ea4`, source `20261103_bf01d5b35d51`. Its A/B/C, age-spline, ROS/MAP/interaction, beta1/beta3 and incremental tables are not recomputed or substituted. It covers eight endpoints, native unmasked calibrated features and its registered subset of modules/components. B and C use identical donor hashes; A→B is selection, B→C adjustment.

Example: beta3 M747 DLPFC gain–cognition changes from adjusted original ρ=−0.235 (N275, p=0.0002, q_scan=0.0734, q_GL_outcome=0.0193) to incomplete ρ=−0.0337 (N620, 95% interval −0.112–0.047, p=0.4147, q_GL_outcome=0.9995). Its pathology direction is retained but attenuated: ρ=0.205→0.136 (N327→746; validation p=0.0002, q_GL_outcome=0.238). Thus nominal pathology support is not full-family adjusted support. Different subgroup significance is not an interaction test; the actual saved interaction table is exposed separately.

All128 incomplete connectivity/PC1 incremental rows remain unavailable (`no_shared_comparator_donors`): the registered same-module eigengene artifact covers original450 only, with no compatible frozen incomplete-cohort projection. No validation PCA is fitted and unrelated regional eigengenes are not substituted. The128 original incremental rows are available, but measure **in-sample explanatory** ΔR², not held-out predictive improvement; 34 have their original focused q<0.05. Optional top-edge-removal sensitivity was not run.

## 6. App, exports, QA and limitations

Open **Compare → Validation & replication → Evidence overview**. “Evidence view” opens **Overview & cohorts**, **Control structure**, **MDC decomposition**, **Same-edge follow-up**, or **Adjusted evidence (A/B/C)**. Use **Apply analysis** after changing a scientific selection. Legacy association/MDC details and matched-control results are linked, not removed. Edge volcano labels explicitly identify its legacy complete-donor held-out split.

Every new chart uses the existing preview/publication exporter with source/version/applied-context metadata. The overview's **Meeting figures and results note** downloads the aggregate [PNG/PDF figures](figures/) and this note. Exact paired network/edge plots are available only in the local app with verified private receipts. The hosted app shows aggregates and exact read-only public report tables, with explicit unavailability for local details; no private edge or donor arrays are committed.

Build: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/lioness-evidence-matplotlib python3 scripts/build_validation_evidence.py --stage all --bootstrap 199`. Export via `scripts/export_validation_evidence.py` with the existing pinned export environment/Chrome. Stages resume from versioned private checkpoints.

See [QA.json](QA.json) for the final tests/browser/export checks and [manifest.json](manifest.json) for source, producer and output hashes. The32 sampled direct add-one checks have maximum error 5.684×10⁻¹⁴ (tolerance10⁻¹⁰). The bounded new structure stage took about28 seconds and selected-edge stage about90 seconds, each under1GiB peak RSS with two BLAS threads; these are local measured stage costs, not cloud benchmarks.

Limitations: fixed retrospectively selected modules/lists; within-ROSMAP preprocessing and cohort differences; original discovery/reference/calibration participation; historical overlap or missing membership receipts; unadjusted diagnosis effects; correlated edges; heterogeneous component Ns; finite Monte Carlo resolution and only199 new bootstrap draws. The concordance resampling distribution is not a calibrated preservation CI. Existing module-score and edge-level q families answer different questions. No prediction/reclustering/full-catalog expensive reconstruction, new reference calibration, formal preservation test, or causal analysis was performed. Code/tests/public aggregates are committed only; no push or deployment is authorized.

### Changed-file inventory

- Readers and UI: `app_helpers/validation_evidence.py`, `app_helpers/validation_evidence_ui.py`, `app_helpers/validation_review_ui.py`, `app_helpers/workspaces/donors.py`.
- Offline calculation/export: `scripts/build_validation_evidence.py`, `scripts/validation_evidence_edges.py`, `scripts/export_validation_evidence.py`.
- Tests: `tests/test_validation_evidence.py`, `tests/test_validation_review.py`, `tests/test_control_calibrated_app.py`, `tests/test_mdc_adjacency_controls.py`. Existing scientific assertions are retained; legacy navigation tests now explicitly select their detail view and use Apply-aware routing.
- Public outputs: this versioned report directory's eight CSV tables, `manifest.json`, `QA.json`, this note and six PNG/PDF files under `figures/`. No original report tables, core score implementations or Prediction files changed. Unrelated pre-existing validation-data/M5 edits are not included in the commit.
