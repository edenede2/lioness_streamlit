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
  and all registered partitions. This is the broad browser's default correction.

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
