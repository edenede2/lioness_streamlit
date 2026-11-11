# Control-calibrated donor connectivity, version 2

This source reuses the registered ROSMAP partitions. It does not recluster genes
or change discovery normalization. Its default scoring functional is uniform
signed power beta3 on every TS and CT edge. Beta1 is a sensitivity version over
the identical physical catalog and donor allocation. Historical Connectivity,
CAMEL/native-power comparators, eigengenes and group MDC remain separate.

## Navigation

In an ordinary multi-tissue analysis view, select a module definition and choose
**Control-calibrated connectivity · beta3** under **Network estimator**. Gain
and loss are selected together initially. Select a tissue, tissue pair, pooled
TS/CT, or the whole-module fixed union. Native score units are the default.

Independent regional modules expose the same source under **Regional donor
score source**. The module namespace includes its partition and tissue.
**Module–outcome comparison** exposes a donor score source in each box;
**Donor explorer**, **Gene & edge associations**, and **Validation & replication**
have corresponding source controls. Prediction has no calibrated source.

Deep links accept `score_source=calibrated_beta3` or `calibrated_beta1`, together
with the existing `view`, `screen_partition`, `screen_module` parameters and the
optional `cc_scope`. These are local application links, not published sites.

## Scientific definition

For disjoint reference controls R, calibration controls C and independent target
i, use the same R to generate both calibration and target weights:

$$
f_\beta(r)=\operatorname{sign}(r)|r|^\beta,\qquad
w_{ie}=(n_R+1)f_\beta(r_{R+i,e})-n_Rf_\beta(r_{R,e}).
$$

Each physical tissue/gene edge has its own calibration:

$$
\nu_e=\operatorname{mean}_{c\in C}|w_{ce}|,\qquad
\tau_e=\operatorname{SD}_{c\in C}(|w_{ce}|;\,ddof=1),\qquad
a_{ie}=\frac{|w_{ie}|-\nu_e}{\tau_e}.
$$

For a fixed eligible physical-edge union E, all four magnitude summaries share
one denominator:

$$
G_i=\frac{\sum_{e\in E}\max(a_{ie},0)}{|E|},\quad
L_i=\frac{\sum_{e\in E}\max(-a_{ie},0)}{|E|},\quad
S_i=G_i-L_i,\quad A_i=G_i+L_i.
$$

G measures magnitude gain and L measures magnitude loss. S can cancel opposing
changes; A preserves their total size. These are deviations in LIONESS
contribution magnitude, rather than within-person correlations, Gaussian tail
probabilities or fold changes. A more negative edge can have magnitude gain.
Per-edge loss is bounded by nu/tau; lower L variation is not evidence of better
biological validity.

Raw companions retain mean absolute w and mean magnitude change. Signed z uses
its own mu/sigma and eligibility. **Signed-profile squared distance D=mean(z²)
is not A.** The calibration numerical guard is 64 epsilon sqrt(n_C) relative to
the maximum absolute calibration weight; it detects numerical degeneracy, not
statistical reliability. Every eligible edge must have the complete declared C
count. There is no cross-edge variance pooling, strength weighting or clipping.

Sign annotations use 100 conditional C bootstrap resamples, seeds 711010–711109,
2.5/97.5 percentiles, agreement at least 0.8 and near-zero threshold 0.01nu.
Canonical donor ordering makes the finite-resample annotation invariant to
caller ordering. Zero, near-zero, uncertain, preserved and reversed contributions
partition G and L. Sign uncertainty never removes a magnitude-eligible edge.

## Roles, coverage and transformations

The frozen B allocation, seed 611010, is shared across discovery families and
betas. Exact historical membership hashes are verified during preparation.

| Dataset | R | C | Original independent targets | Target Controls | Target AD/MCI |
|---|---:|---:|---:|---:|---:|
| Brain |82|41|327|41|286|
| Motor |60|30|267|30|237|

R/C donors have no independent B score. This is a role condition, not zero
connectivity. Compatible incomplete and additional complete-tissue targets use
the frozen fit. Private canonical identities prevent a renamed R/C donor from
re-entering as a target. Brain and motor public namespaces remain separate.

Tissue presence comes from a fixed dataset-wide source-row/assay manifest,
independent of module-selected genes. Missing whole tissues are not imputed.
Sparse genes in a present tissue use R-only endpoint means. The optional
`unavailable` numerical policy preserves edge-local eligibility across batches.
Endpoint-imputation and absent-tissue counts are included in score downloads.
A missing third tissue cannot suppress another tissue's underlying edge weights.
Full unions require every tissue represented in their exact selected edge set;
missing coverage produces NA with a reason. Physical edges are deduplicated;
union means sum contributions and counts instead of averaging component means.

Asinh scales and interpolated RINT knots are fitted only to explicitly labeled
R LOO training diagnostics. Neither transform is fitted to targets. Selecting a
transformed scale applies it once to plots and additional descriptive analyses.
The saved baseline scan always uses native units. The diagnostic transforms do
not create an all-donor independent or OOF protocol.

The partitions retain historical discovery overlap. Incomplete/validation data
have been examined previously. Implementation checks are not biological or
external-cohort validation. New regional-motor donors with an explicit
`Other_dementia` source label retain that label; no diagnosis is reconstructed
from arbitrary codes.

## Frozen masks and edge inspection

Existing compatible discovery masks are mapped by verified physical endpoint
identity. Conflicting duplicate positional memberships are unavailable with an
exact reason. Self edges are excluded and physical pairs deduplicated. A mask
changes the aggregation set and denominator, while retaining the same edge
calibration. Historical target-informed masks remain exploratory; masks are
never relearned from the displayed target groups. Motor historical “global”
aliases do not acquire a new across-partition correction interpretation.

Selected-donor edge inspection reconstructs bounded blocks from frozen caches.
It reports raw w, absolute w, magnitude change, a, nu/tau, calibration N,
eligibility reason, sign class and G/L contribution with the actual denominator.
Ranking supports calibrated gain/loss contributions and raw magnitude. General
downloads do not contain reference expression, individual C weights or identity
crosswalks.

## Association families

### Historical-visualization compatibility update (2026-11-11)

Associations now reuse the historical grouped-scatter and categorical chart
helpers. Each measure (Gain, Loss, etc.) is a separate figure, with components
in separate panels, diagnosis colors/shapes, component-specific statistics and
OLS guides. Group-line visibility and the independent dashed/solid black pooled
line follow the historical controls. Feature distributions reuse the same
Histogram, Violin, Raincloud and ECDF helpers. The publication exporter is
unchanged. MDC, module composition and KEGG remain their shared historical
views; changing a donor-score beta does not create a new group-level MDC.

The default association correction is **module-only BH**, holding partition
(and tissue for independent regional partitions), beta/method, cohort, mask,
normalization, transform, component, measure, outcome, donor filter, grouping
variable/level and correlation method fixed. Pearson and Spearman are separate
families. Missing, constant and undersized tests do not enter the denominator.
FDR is calculated before module, Top-K or significance display filters. Tables
report the actual tested-module count, not a hardcoded 154/186.
The minimum paired/category N is configurable (5/10/20, default 10) in the
ordinary association/heatmap/ranking views. Saved baseline p-values and effects
remain inspectable below this threshold but are excluded from module-only BH,
with an explicit exclusion reason. Adjusted validation retains its original
N/residual-degree-of-freedom eligibility gate.

The archived `q_scan`, `q_GL_outcome`, `q_search` and `q_comparison` columns are
not overwritten or relabeled. Native pooled baseline rows retain their saved
p-value/inference method and gain `q_across_modules`. Their broad scan q values
remain inspectable in the explicitly historical baseline browser.
Grouped/custom plots use the historical two-sided SciPy Pearson/Spearman tests;
categorical outcomes use Kruskal–Wallis, never coded correlations. Use
**Calculate module-set FDR for all displayed groups** for the complete matching
grouped/custom family. Reads are serialized and only compact statistics are
cached. Until that calculation finishes, those FDRs are unavailable rather than
incorrectly corrected over the displayed module. A change of donors, mask,
scale or grouping invalidates the family. No adjusted/partial validation test
is substituted for an unadjusted plot correlation.

This is a correction-scope choice, not a guarantee of significance. Historical
scan-wide and focused validation inference retain their original definitions.
Complete-catalog adjusted validation associations also gain module-only BH,
separately by model and study stratum, before selecting named modules. Focused
age-spline panels are not complete partitions: their module-only FDR stays
unavailable and the original focused correction remains labeled as such.

The baseline tests every structurally meaningful registered component and union,
all independent targets, both betas and G/L/S/A, unmasked, in native units. It
includes all 15 numeric/ordinal and 5 nominal outcomes from the authoritative app
registry. Spearman uses minimum N 3; nominal outcomes use Kruskal–Wallis with at
least 5 donors per included group. Small groups remain visible but do not enter
the omnibus test. Results retain actual paired/tested N, group counts, effect
names, exclusion reasons and NA p values for non-estimable tests. Supplied
amyloid/tangle square-root scales and adjusted slope provenance are preserved.

* **q_GL_outcome:** BH jointly over G/L, every planned module/component and
  registered partition in one dataset/cohort/allocation/beta/outcome/method/
  mask/subgroup/model/transform context. This is outcome-specific.
* **q_scan:** BH over the entire baseline dataset/cohort/allocation scan,
  including both betas, G/L/S/A, all 20 outcomes and their type-specific methods,
  and all registered partitions. It remains an archived broad-scan alternative,
  not the default module-set association FDR.

The shared B allocation is separate from beta/component artifact hashes. Planned
structural tests, valid tests and exclusions are stored. Non-estimable p values
remain NA and are excluded from the finite-p BH denominator. Correction precedes
module/outcome/component filters and Top K.

Changed donors, metrics, masks, transforms, methods or covariate models create
explicit saved searches. Their `q_search` or `q_comparison` covers only the
declared executed family; it does not inherit baseline q_scan. Choosing among
separate searches requires an additional declared family. Descriptive OLS uses
complete cases and HC3 standard errors; extra covariates are explicit. There is
no automatic additional adjustment of a supplied adjusted slope. Pairwise group
intervals use 1,000 donor resamples within groups, seed42, not edges as replicates.

The Correlation heatmaps page offers **Selected module** and **All modules**.
All modules covers the selected partition and, for regional partitions, the
selected tissue. Each module/measure/component has its own row; undefined
components and unavailable correlations remain blank. Tall heatmaps scroll,
and the statistics table/download follows the selected numeric outcome columns.
Display filtering preserves saved baseline corrections. Changed donor groups,
masks, measures or transforms require the **Save and calculate all-module
heatmap** action; its q_search spans the complete selected module/measure/
component search and all registered outcomes before display filtering.

## Reproduction and storage

From the application directory, using the existing research environment:

```bash
../../.venv/bin/python scripts/generate_control_calibrated.py --stage prepare
../../.venv/bin/python scripts/generate_control_calibrated.py --stage benchmark --resume
../../.venv/bin/python scripts/generate_control_calibrated.py --stage score --workers 8 --resume
../../.venv/bin/python scripts/generate_control_calibrated.py --stage masks --workers 8 --resume
../../.venv/bin/python scripts/generate_control_calibrated.py --stage analyze --resume
```

`--stage all --resume` runs preparation, scoring, masks and baseline analysis.
An immutable alternative `--root` supports a fresh versioned run. A mask job can
use repeated `--family` arguments after those variants' calibrations finish.
The full catalog remains incomplete until all planned partitions finish.

`data/control_calibrated_v2` contains the catalog, input receipts, target-only
score Parquets, coverage, transformations, association testing manifest and
association tables. Calibration matrices and donor/component sufficient sums
are private. Edge work is batched and physical calibration is shared across
modules; a dense catalog-wide donor×edge cube is never persisted. Offline jobs
checkpoint/resume and write artifacts atomically. App rerenders read caches.
The loaders check content hashes as well as protocol, role, beta and module
identity. Artifacts from the old missingness-bug version are rejected.

The generated implementation report and review bundle live under
`artifacts/control_calibrated_implementation_20261102`. Its coverage/test/resource
receipts, rather than historical module counts, determine completion.

## Prediction exclusion

The central calibrated registry explicitly denies prediction, predictive
regression and feature selection. Its metrics are not added to the legacy
prediction feature catalog. No feature matrix, training/CV/tuning procedure or
predictive comparison is run. Existing prediction inputs/results and unrelated
historical artifacts are verified against their pre-task snapshots.

S and A are formed before optional transformations. Their defining identities
refer to native units. Each metric then uses its own frozen R diagnostic
transform; transformed A is not a sum of transformed G/L. A constant R diagnostic
distribution makes RINT unavailable with an explicit reason; targets never supply
replacement transform knots.

## Hosted summary release

The hosted app uses an immutable release named by
`control_calibrated_release.json`. Its files are indexed under the existing
read-only Google Drive backend. The local research bundle remains at
`data/control_calibrated_v2`; an installed local bundle takes precedence in
ordinary local runs. `LIONESS_FORCE_DRIVE=1` exercises the hosted data path.

A hosted module bundles both betas and every registered frozen mask without
recomputing scores. Identical physical unions share exact numeric rows; their
individual provenance is retained. Original and incomplete cohorts are exact
metadata subsets of maximum-component targets. Numeric values, native/frozen
transforms, common denominators, NA reasons and baseline p/q values are retained.
The release has 272 range-addressable immutable containers, each below the
connector's 100 MiB file limit. The reader downloads only the verified compressed row groups needed for the
chosen beta/mask/component. Per-range SHA256, a sealed footer and the existing
Drive size/MD5 checks protect content. The module disk cache is bounded to
512 MiB by default (`LIONESS_CALIBRATED_CACHE_MIB` can change this).

The user explicitly approved publishing deidentified donor scores and linked
registered clinical outcomes to the existing app data folder. Public pseudonyms
remain aligned with the existing app; individual R/C clinical values are removed.
Raw expression, reference/calibration arrays, preprocessing fits, private identity
crosswalks and physical source-node lists are excluded. **Edge reconstruction and
edge-level searches remain local by the user's instruction.** Hosted edge views
explain this limitation; donor scores, masks, donor-score associations, comparisons,
rankings, distributions and downloads remain available. No prediction feature or
result is enrolled or changed.

Reproduction helpers are `scripts/build_control_calibrated_release.py`,
`scripts/audit_control_calibrated_release.py`,
`scripts/pack_control_calibrated_transport.py`,
`scripts/audit_control_calibrated_transport.py`,
`scripts/control_calibrated_release_plan.py`, and
`scripts/verify_control_calibrated_uploads.py`. After complete readback,
`scripts/install_control_calibrated_release.py` adds the sealed index locally.
`scripts/smoke_control_calibrated_hosted.py` compares real Drive reads with the
frozen research sources for all families, betas and scopes. The immutable publisher accepts a
valid existing user OAuth token; connector upload receipts can also be resumed
and verified. Uploads add only new release files and do not alter existing Drive
objects or sharing. Data upload/index preparation never pushes Git. The user
pushes the reviewed release commit separately to the branch configured for the
hosted app.

## Local covariate validation (2026-11-04)

The five ROSMAP validation analyses are documented in the
[scientific report](reports/control_calibrated_validation_20261104/report.md).
They cover all seven ready families (1,697 modules), an eight-outcome adjusted
scan, linked all-pair/matched-pair/adjusted estimates, ROS/MAP sensitivities,
focused legacy/PC1 comparisons, and a versioned repair of the full 20-outcome
baseline. The local bundle is distinct from the hosted release. No production
release pointer or prediction artifact changes.

Use a calibrated beta3/beta1 source in Associations, Statistics or Correlation
heatmaps, then **Association results → Covariate validation / 8-outcome scan**.
The ordinary score view retains the 20-outcome baseline and an explicit
repaired/original inference selector. Downloads carry the exact model, donor-set
hash, N, uncertainty, inference method and correction context. Rankings and
residual plots are association models; stored native G/L/S/A units are preserved.

For a fresh local run, install the existing application dependencies and
`requirements-validation.txt` into the numerical environment, then run:

```bash
../../.venv/bin/python scripts/run_control_calibrated_validation.py all --workers 4
```

To finish the current frozen workspace without refreezing it, use `finalize`
instead of `all`. Completed bundles are immutable and reused. The default is
four workers and two BLAS threads each; checkpoint shards are bounded score
summaries, never a dense donor-by-all-edge matrix. The specification records
9,999 Monte Carlo draws; the p=0.0001 floor limits broad FDR resolution and is
flagged in the app/report. Both original and repaired baseline versions remain
available for audit. All results from this task are local and unpublished.
