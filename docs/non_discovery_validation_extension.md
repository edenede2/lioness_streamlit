# Non-discovery-donor validation extension

The existing 186-module Control-derived brain validation remains unchanged.
New results use separate partition/method directories and require their own
source audit, scoring pilot, association reconciliation, MDC and publication
gates. No prediction, module discovery or validation-dependent mask learning
is included. A completed scoring checkpoint is not a published result.

## Registered identities

- Full-cohort brain: 154 modules, Standard and Control-referenced LIONESS;
  450- and 164-donor frozen references respectively; 793 validation donors;
  exclude all 464 complete-expression donors. All edges and 30 frozen masks.
- Control-derived motor: 155 modules, 120-Control-referenced CAMEL.
- All-donor motor: 173 modules, 120-Control-referenced LIONESS.
- Motor validation is provisionally 142 donors, excluding the 357 complete
  motor donors; component counts 110/88/67/58/40/25 for VH, SMA, QUAD,
  VH–SMA, VH–QUAD, SMA–QUAD. Audit actual discovery membership before scoring.
  Only the existing three Hedges' g=0.40 directions are in scope.

All validation donors, including Controls, use the add-one formula. All maps
and masks remain fitted to original donors. No missing region is imputed.
The public score-reader identity includes module definition and network method;
the legacy Control-derived API and data directory remain supported.

## Source-convention discrepancy and approved resolution

The source assignment table contains 46,898 rows, including 30 repeated
tissue–gene rows in 16 modules. The historical R **unfiltered** runner maps
unique expression columns through a named gene-to-module vector, effectively
collapsing those repetitions. The later Python **effect-filtered** runner
selects the assignment rows positionally and retains their multiplicity.

For M188 the source contains 30 rows but the historical unfiltered network has
27 nodes. With 27 nodes, an independent recomputation reproduced 324 saved
values with maximum relative error 6.90e-14. With 30 nodes the historical
reproduction gate failed. Unaffected pilot modules M589 and M1918 reproduced
their saved values. These are sampled checks, not a whole-catalog audit.

On 2026-10-21 the user explicitly selected reproduction of each legacy
analysis. The source table and frozen effect masks retain all rows. The
unfiltered analysis uses one first-occurrence positional node per unique
tissue–gene identity, while effect-filtered analyses retain assignment-row
multiplicity. Connectivity denominators and possible-edge counts follow the
selected analysis. Outputs record `node_convention`, `n_features`,
`n_source_assignment_rows`, and `cross_filter_node_convention_warning`.
The 16 modules are flagged in manifests and cross-filter comparisons. Neither
historical results nor discovery masks are overwritten. A new run directory
keeps this approved policy separate from the initial diagnostic pilot.

## Release constraints

- BH is across valid modules within each fixed context, never groups × modules,
  and is calculated before any display/top-K filter. Original and validation
  corrections remain separate; pooled originals must be present from release.
- Brain MDC uses the existing signedAlt powers. Motor MDC must use the
  separately validated no-q90 M5 and unpowered comparison, not brain powers.
  Both nulls retain their actual definitions; gene-wise donor shuffles are not
  diagnosis-label permutation tests. Sparse or unstable MDC remains unavailable.
- New donor-level bundles require explicit publication approval. No uploader
  or automatic GitHub push is invoked by this framework or scoring pilot.
- Existing donor-score data remain lazy, one module/context per read. Incomplete
  identities remain hidden; repository presence alone does not prove a verified
  Drive backend. Preserve pseudonyms, private mappings and existing results.
- Pilot workers are limited to two; the measured combined worker-memory gate
  is 32 GiB. Keep prediction jobs untouched. Brain release precedes motor runs.

## Brain milestone, 2026-10-21

Both scoring methods have finished 154 modules and all 31 edge contexts:
7,795,942 summary rows and 343,728 transformation rows per method. Sampled
historical reproduction covers 46,116 values per method; the maximum relative
error is below 7e-13. This establishes technical agreement, not biological
replication or absence of leave-one-out/add-one reference-role artifacts.

The first association milestone includes all edges and the three g=0.40
directions, with both score normalizations. Pooled original statistics use
the exact saved original Raw/asinh/RINT columns, and original and validation
BH corrections use separate complete module families. Review tables may be
released before donor scores: the review page does not read donor data.

The user explicitly approved both brain donor-score/transform bundles on
2026-10-21. The private approval records constrain method, row counts, file
counts and existing sample population. Approval does not bypass the MDC,
independent release-QA or backend content-readback gates. New motor publication
is not covered by that approval.

The serialized completion runner is:

```bash
python scripts/run_partition_validation_pipeline.py
```

It completes associations, the 200+200-permutation MDC pilot/full runs,
independent release QA, and compact aggregate review staging. Its private
`pipeline_status.json` records the current method/stage. Checkpoints permit
restart. It does not upload, commit or push. After a passing release audit,
`scripts/publish_partition_validation.py` is a separate, explicit approved
publication step. It verifies upload checksums and representative reads through
the app's read-only identity before exposing the cohort.

With the separately recorded approval, `publish_ready_brain_validation.py --watch`
may wait for passing first-release and final QA manifests, then invoke that
scoped uploader. It never publishes intermediate individual masks, never
starts motor uploads, and never commits or pushes. A stale audit, analysis
failure, changed scope, upload/readback failure or concurrent index change
prevents exposure. Its private `brain_publication_status.json` is separate
from analysis status. The wait is bounded to 48 hours by default.

Repository review caches are bounded to connectivity and six cognitive/pathology
outcomes for the four first-release edge contexts. Other features, outcomes and
edge masks remain lazy backend reads after their registry is published. No
donor scores or full frozen rank curves are copied to Git.

The extension's targeted regression run passed 118 tests (one skipped).
A broader run excluding two private Matplotlib-dependent tests passed 480,
skipped 21 and reported six pre-existing catalog/test-expectation issues:
motor regional KEGG missing p-values, old fixed twelve-outcome row assertions,
a legacy manifest entry without a row count, and old UI assertions for sixteen
outcomes and the pre-CAMEL label. These are not evidence that the entire app
test suite is green; they must not be hidden or described as new-result QA.
Existing KEGG, outcome catalogs and their manifests are not modified by this
validation extension.
