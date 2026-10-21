# CAMEL donor validation and replication review

The **Validation & replication** page reviews the Control-derived brain-module
analysis in donors excluded from the complete-expression cohort. It is separate
from Prediction and does not launch analyses when opened.

## What the page establishes

- **Technical checks:** fixed membership/cohort audit; sampled direct formula
  checks; sampled reproduction of the original R scores; edge/count identities,
  source hashes and module-only BH reconciliation from the release audit.
- **Association replication:** original and validation Pearson/Spearman effects,
  sample sizes, p-values, module-only FDRs and tested-family sizes. Direction-only,
  nominal and FDR replication criteria are explicitly distinguished. Missing
  tests are not treated as failed replication. Original pooled/unclassified
  comparisons are unavailable in the saved stratified catalog.
- **MDC replication:** separate original/validation group-network ratios and
  FDRs, on Raw or log2 scales. AC–PCG (3 AD, 6 Controls) remains descriptive.
- **Coverage & completion:** overlapping region/pair donor counts and separate
  score/association/MDC status for each frozen mask.

This is non-discovery-donor validation within ROSMAP, conditional on prior
expression preprocessing and fixed modules—not independent external validation.
Numerical reproduction of an algorithm is not evidence that a biological
association replicated. Correlation changes are descriptive; no unregistered
between-cohort correlation test or prediction significance claim is added.

## Scoring boundaries

All 464 complete-expression donors are excluded, leaving 793 unique donors.
Component cohorts contain 266/752/195/230/14/176 donors for AC, DLPFC, PCG,
AC–DLPFC, AC–PCG and DLPFC–PCG. Never sum these overlapping counts.

All new donors, including Controls, are add-one scored against the original 164
Controls. Modules and the discovery masks remain fixed. Original-cohort fitted
robust-asinh scaling and RINT interpolation/clipping are transferred unchanged.
Clipped RINT can introduce ties and therefore change Spearman statistics.
No missing region or pooled three-region CT/TS score is fabricated.

The term CAMEL applies to Control-referenced scores on Control-derived modules;
internal `lioness` / `control_anchored` identifiers remain backward compatible.
MDC, eigengenes, BONOBO and full-cohort-module LIONESS retain their names.

## Incremental build and publication

Private outputs/checkpoints remain outside the app under
`out/camel_validation/20261021_frozen164_incomplete_regions`.

1. `scripts/run_camel_validation.py`: audit, pilot and all-mask scoring.
2. `scripts/validate_camel_existing_scores.py`: independent sampled original-R
   score reproduction.
3. `scripts/run_camel_validation_mdc.py`: frozen-mask resolved MDC and both nulls.
4. `scripts/build_camel_validation_statistics.py`: selected completed mask(s),
   both score interpretations, full-module association corrections.
5. `scripts/prepare_camel_validation_release.py --concordance`: original versus
   validation statistics, frozen scalar maps and release QA.
6. `scripts/build_camel_validation_review.py`: narrow aggregate review shards.
7. Repeat the release audit after the final review snapshot, then explicitly run
   `scripts/publish_camel_validation.py` using the ignored authorized uploader
   token. Publication requires the recorded scoped user approval.

The publisher creates an immutable snapshot under the existing app Drive root,
checks every upload's size/MD5 and verifies representative bytes using the app's
read-only service account. It changes neither sharing nor unrelated files. The
Drive index is installed before the backend-verified availability manifest.
Interrupted uploads resume with `--snapshot` pointing to that private snapshot.
Rollback copies remain outside the public app directory. No script commits or
pushes Git automatically.

Later releases reuse unchanged, checksum-verified Drive files rather than
uploading the large score/curve bundle again. Keep older immutable backend
folders: the current index may still reference their unchanged files.

Only compact metadata, coverage/QA/default review tables and manifests should be
force-added from the ignored `data/` tree. All donor scores, frozen curves and
non-default review shards are lazy-loaded from Drive; do not add the whole data
directory to Git. Clinical values and score publication require the recorded
user approvals. Credentials and private mappings must never be staged.

## Tests

`tests/test_camel_*.py` and `tests/test_validation_review.py` cover arithmetic,
frozen maps, direct statistics/module-only BH, privacy/grain, sparse MDC,
replication criteria, publication snapshots, and Streamlit interaction paths.
Source-backed cases skip explicitly when the private run is absent; they never
silently download private expression or claim a completed check. The review
page's tests prohibit donor-score and metadata reads.
