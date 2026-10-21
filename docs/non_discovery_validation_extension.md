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

## Blocking source-convention discrepancy found by the brain pilot

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

The plan's requirements to retain every repeated node **and** reproduce every
historical unfiltered score cannot both hold under one node convention. Do not
silently deduplicate, replace published baselines, or weaken the reproduction
gate. Obtain a user decision between source-specific legacy conventions, a
new explicitly labeled validation-compatible baseline, or withholding the
affected modules. The scoring runner currently fails closed on this mismatch.

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
