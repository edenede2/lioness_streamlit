# LIONESS ROSMAP workspace refactor

Implemented locally on 2026-11-08 against app commit `b1b71b4`, following the locked visual design. No networks, eigengenes, masks, MDC, KEGG, predictions, clinical data, or publication permissions were changed. No GitHub push or cloud deployment was performed.

## Where to find the existing tools

| Workspace | Tools |
|---|---|
| Discover — new landing | Module finder, Correlation heatmaps, CT–TS screen, Distribution differentiation |
| Module | Associations including score-comparison bars, Feature distributions, MDC, KEGG enrichment, Module details, Statistics |
| Compare | Module–outcome comparison, Partition comparison, Validation & replication, Matched-control results |
| Prediction | Existing all-module, targeted, sensitivity, transcriptomic and Protocol v1.1 modes |
| Donors & edges | Donor explorer, Gene & edge associations, Edge summaries, Edge volcano, Donor edge explorer including expression/residual inspection |
| Data & methods | Existing methods, provenance, availability and catalog downloads |

All twenty legacy `?view=...` routes remain supported. Regional modules retain independent tissue-specific selectors. The matched-control 247-module dataset is accessible in Module through its existing source-specific adapter; its specialized comparison and technical sections remain in Compare.

## Interaction changes

- Dataset and capability controls stay in the sidebar. Module, outcome, feature and scale selections appear in a compact analysis header where relevant. Advanced controls are disclosed rather than repeated above every result.
- **Apply analysis** separates pending statistical settings from applied results. Same-source pending changes retain the applied result. A pending dataset/method/cohort change stops before score reads, with an explicit notice, rather than showing a result under the wrong source.
- Colors, trend visibility, labels and export appearance do not change the applied statistical context. Relevant cached results are reused.
- Discover can open a module with its applied analysis context and return to the original discovery view. Single-row selection in the ranked table supports the same transition.
- Compare can append an applied score configuration without overwriting existing comparison boxes or pooled-donor filters. Unsupported source/estimator combinations are not substituted.
- Prediction owns its saved-catalog context; exploration donor filters never imply model refitting.
- Existing publication settings and `render_exportable_plot` remain in use. KEGG captions wrap on narrow screens; the publication legend retains its existing outside-title contract.

## Code organization

`streamlit_app.py` is now a small bootstrap/router. `app_helpers/app_shell.py` owns native navigation; `analysis_context.py` owns draft/applied state; `workspaces/` contains page renderers and scoped controls/read preparation. `ui_runtime.py` retains the existing shared UI readers/statistical helpers and compatibility checks.

The scientific readers, chart builders and statistics remain the source of truth. Optional manifest arguments in `data.py` preserve the original signatures' default behavior while avoiding repeated parsing of the large root manifest.

Only the selected workspace executes. Metadata-only views no longer load module donor-score tables. Prediction-manifest changes no longer clear exploration caches globally. UI caches are bounded and versioned by the relevant dataset or catalog. Regional statistical caches hold the three visible tissue catalogs without evicting the first tissue on every redraw. Expensive calibrated/history and deployment-manifest disclosures execute only when opened.

## Verified behavior

- All twenty legacy routes, the Discover default, draft/apply behavior, cross-source isolation, Discover-to-Module return, and bounded cache/version behavior are covered by tests.
- Existing regression checks cover multi-tissue and independent brain/motor sources, matched-control and calibrated adapters, correlations, distribution tests, score-comparison bars, comparison calculations, Drive integrity helpers and export controls.
- Two old-versus-new UI spot checks reproduced every numeric point/trend trace and statistical annotation, ignoring only newly inserted caption line breaks: full-cohort M935/global cognition/connectivity, and Control-derived M121/AC–DLPFC/negative density/pathological slope/asinh/ROSMAP Cluster 4.
- Real Kaleido scatter/enrichment exports passed PNG, TIFF, SVG and PDF checks, including raster dimensions and vector PDF page/font checks. A separate regression checks the refactored association header's publication legend placement.
- Headless Chrome rendered Discover, Associations, Module details, MDC and Prediction, the publication export preview, a 700-pixel-wide layout, and a second independent browser session without app exceptions or page-width overflow. Screenshots were inspected locally; no screenshot or donor payload is added to the repository.

Final UI/source regression run: **183 passed, 1 skipped**. A separate smoke/comparison run from the normal app checkout: **127 passed, 2 skipped**; these suites overlap and must not be summed as unique test coverage. Real scatter/enrichment export integration: **2 passed**. Test skips requiring optional integration flags or private fixtures remain explicit.

These checks are representative preservation tests, not a claim to have exhaustively revalidated every scientific dataset or every custom export combination.

## Measured local performance

Same five views, data, pinned Python environment, two runs per view, fresh benchmark processes. Times are seconds; warm runs reuse cached results.

| View | Before cold | After cold | Before warm | After warm |
|---|---:|---:|---:|---:|
| Module details | 1.495 | 0.786 | 0.705 | 0.085 |
| KEGG | 0.980 | 0.381 | 0.669 | 0.038 |
| MDC | 0.916 | 0.251 | 0.783 | 0.166 |
| Module finder | 1.234 | 0.572 | 0.639 | 0.054 |
| Associations | 3.400 | 2.681 | 0.804 | 0.132 |

Cumulative process peak RSS for this sequence decreased from **926.71 MiB to 517.47 MiB**, about **44%**. Warm runs performed no Pandas/Arrow data reads. Metadata-only views avoided unrelated donor-score reads.

This is a local AppTest high-water measurement, not per-view memory attribution, process-tree/export-child memory, a Community Cloud measurement, a concurrency capacity claim, or a guarantee against cloud OOM. The browser check exercises two sessions and an export preview but is not a sustained load test. Cloud resource monitoring remains a deployment acceptance step.

Reproduce with the app's pinned environment:

```bash
python3 scripts/benchmark_app.py --output /tmp/lioness-ui-benchmark.json
```

## Pre-existing data and test-contract failures

Four current data-bundle assertions failed identically against both the untouched entrypoint and the refactor:

1. Some motor KEGG regional p-values are missing although an older test expects all present.
2. LIONESS statistics contain more than the legacy twelve-outcome row-count expectation.
3. A manifest entry lacks the `rows` field expected by an older all-files assertion.
4. BONOBO statistics likewise exceed the legacy twelve-outcome expectation.

These failures were reproduced against the original checkout and are not silently marked passing or repaired by changing data. They need a separate data-contract review before a strict all-bundle release gate can be claimed.

One additional legacy private-fixture validation test fails in the normal checkout: it assumes full-cohort validation is unsupported and mocks `available` with a single-argument signature, although the existing full-cohort catalog uses the multi-argument API. The failing test, `data.load_resolved`, and `camel_validation.validate_identity` were verified AST-identical to the pre-refactor commit. They were not modified here. The normal-checkout scientific/export run therefore reports **121 passed, 10 skipped, 1 pre-existing failure**, not an entirely green full suite.

## Release and rollback boundaries

Only application code, tests, this design record and payload-free benchmark tooling belong to this change. Preserve pre-existing validation Parquets, M5 scripts/tests, credentials, ignored manuscripts, private mappings and running jobs.

The active analysis branch and deployment `master` have different histories and capabilities. Do not blindly push the analysis branch as a replacement for `master`; review the source baseline when preparing the cloud release. This local implementation does not merge or deploy either branch.

The tested source files are also present in the normal app checkout as uncommitted changes. All 102 pre-existing tracked dirty files were hash-checked during that copy and preserved. At the user's explicit request, the refactor is recorded separately on `codex/app-refactor-20261108`; its commit excludes validation data, M5 changes and the temporary data symlink. No `git add .` or GitHub push was performed. Push that named branch for review; merging into deployment `master` is a separate step.

Rollback is a code-only return to the recorded pre-refactor app revision, with donor data unchanged. No cache purge, data rebuild, Drive upload or scientific rerun is required. Do not use a destructive working-tree reset when unrelated changes are present.
