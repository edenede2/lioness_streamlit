# Offline gene–outcome and edge–outcome screening

This is separate from existing predictions, validation scores, differential-edge
screens, MDC and KEGG. The application never calculates genome-wide associations.

## Scope and identities

- Six tissues, all 15 numeric/ordinal app outcomes, original brain 450/motor 357.
- Validation excludes all 464 complete-expression brain donors (793 remaining)
  and all 357 complete motor donors (142 remaining). Use observed components only.
- Top 50 genes/tissue/outcome, top 10 physical edges/module/outcome, separately
  by absolute Pearson and Spearman. Retain sign and canonical-identity tie breaks.
- The frozen Pearson/Spearman union is evaluated in validation and annotated
  within Control/MCI/AD. Diagnosis checks are descriptive post-selection checks.
- Nominal clusters, diagnosis, study, sex and APOE genotype are never numeric
  correlation targets. Supplied pathology/slope transformations are not repeated.
- Source metadata are checked against existing app values. A few historical CSV
  round trips differ below 1e-16; reconciliation tolerance is 1e-14, with identical
  missingness. This is not a new phenotype release or transformation.
- Modules are unique physical tissue–gene sets. All 154 full-cohort brain IDs
  remain present after resolving 30 repeated rows; M188 has 27 unique nodes.
  Existing legacy scores/duplicate conventions are not modified by this screen.

## Statistical families

Every test requires 10 complete pairs and nonconstant predictor/outcome. Spearman
uses average ranks recomputed on that test's observed pairs. P-values follow
SciPy Pearson and asymptotic Spearman conventions (not a permutation p-value).

Original BH precedes top-K selection:

| Analysis | Primary family | Additional family |
| --- | --- | --- |
| Genes | All valid genes in a tissue | None |
| Module edges | All valid edges in a module | All valid edges in the partition |
| Whole network (later) | All valid edges in the network | Each tissue/pair |

Outcomes, correlation methods, network methods and normalization profiles never
share a correction family. Missing/constant/undersized tests are excluded from
the denominator. Exact float64 BH uses the private pinned DuckDB external sort
and a streamed reverse cumulative minimum; no top-hit-only original correction.

Validation BH is over the prespecified original selected-hit union, separately
for each outcome/correlation method. Module-level and partition-level selected
families are both saved. It is explicitly **validation selected-hit FDR**, not
original full-search FDR. No subgroup search FDR is invented for diagnosis checks.

BH is numerically exact; this does not establish its theoretical control under
arbitrary edge dependence. These remain observational association screens.

## Networks and motor gate

Brain uses signedAlt TS3/CT2, Standard plus Control-referenced for full-cohort
modules and Control-referenced for Control-derived modules. Every validation
donor, including Controls, is add-one scored against the original reference.

Motor is **full no-q90 M5**, not existing beta-only donor CAMEL/LIONESS scores:
200 80%-donor subsamples, `abs(r) > .10`, conditional mean correlation, bootstrap
frequency, signed powers, and CT null excess. The null uses 50 permutations and
at most 100,000 **eligible** edges per block. No q90 or clipping. Both fitted beta
profiles are defined in `scripts/screening_m5.py` and independently tested.

Every LOO/add-one adjacency refits the full estimator, including the CT eligible
pool/null. Resampling priorities are deterministic donor/physical-edge hashes,
independent of diagnosis, module number, donor order and batching. This changes
the stochastic coupling convention relative to old traversal-dependent seeds;
it does not claim bitwise equality to a legacy seeded M5 run. Numerical direct
checks use identical coupled plans and the existing bootstrap formula.

The dense module/pilot M5 helper rejects more than 3,000 nodes. It is not a
whole-network production implementation. Module-free search primitives and
two-scope exact BH are tested, but **genome-wide full M5 remains unlaunched and
unavailable until the measured pilot/resource review**. A passing pilot does not
automatically authorize expansion. The pilot covers both profiles, both methods,
small/large three-tissue modules, validation Control add-one, and seeds
42/1042/2042. No hypothesis/outcome selects pilot modules. A whole-network runtime
estimate must include full eligible-null scans, all donor references and ranking.

## Commands and output

Private requirements are in `scripts/requirements-screening.txt`; do not include
them from Streamlit requirements. The existing analysis environment is:

```bash
/tmp/lioness-figure-export-312/bin/python scripts/run_gene_edge_screen.py --stage first-release --workers 2
/tmp/lioness-figure-export-312/bin/python scripts/watch_screening_release.py --pilot-after-brain
```

Run these in separate persistent terminals. The watcher stages complete aggregate
catalogs only, validates direct calculations, deduplicates selected brain weights,
then runs the **single-worker pilot after brain numerical workers finish**. It
never pushes, uploads donor rows, or launches whole-genome M5. The default root:

`out/gene_edge_outcome_screen/20261022_original_and_validation`

The audit, complete gene tests, module p-value batches, selected-hit donor weights,
private mapping extension, checkpoints, QA and pilot reports stay upstream.
Original app sample IDs are preserved; missing motor pseudonyms get a run-private
extension. No clinical values or donor linkage are reconstructed from fingerprints.

The public bundle contains aggregate selected-hit statistics only. Staging uses
immutable hashed files and an atomic manifest replacement, with rollback outside
the app directory. Only complete catalog entries are selectable. No network,
expression, MDC, eigengene, mask or prediction output is overwritten.

Combined screening RSS is guarded at 32 GiB, initial numerical workers at two,
and scratch at 500 GiB. Complete donor networks are not retained. Selected-weight
module caches are resumable intermediates; canonical selected weights are
deduplicated by underlying network configuration across partitions. The private
consolidated outputs must be sharded/sanitized and separately approved before
any donor-level release. None have been uploaded by this implementation.

## Tests and release state

```bash
/tmp/lioness-figure-export-312/bin/python -m pytest -q tests/test_outcome_screening.py tests/test_screening_pipeline.py tests/test_screening_global.py tests/test_screening_m5.py tests/test_outcome_screening_ui.py
/tmp/lioness-figure-export-312/bin/python scripts/validate_outcome_screening.py
/tmp/lioness-figure-export-312/bin/python scripts/stage_outcome_screening.py
```

Tests cover direct SciPy correlations/BH, sparse/constant/missing states,
deterministic rankings, exact component/global BH, native M5 bootstrap agreement,
full-estimator LOO/add-one, donor/node order invariance, batching/checkpoint
equivalence, frozen validation selection, privacy rejection, and Streamlit charts.

Real app startup and the new page were exercised with Streamlit AppTest. Browser
pixel/narrow-layout/export-image inspection has not been performed. One unrelated
existing validation test (`test_validation_rejects_other_definitions_fdr_and_eigengenes`)
still expects the now-supported full-cohort validation set to be rejected and
uses an outdated one-argument availability mock; this screen does not change it.

Initial milestone: all six gene catalogs (6,241 frozen outcome–gene hits) are
locally available. Brain module-edge stages run and release independently; motor
and whole-network statuses remain explicitly unfinished. No automatic Git push.
