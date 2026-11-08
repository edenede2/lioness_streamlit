# LIONESS ROSMAP app refactor: implementation plan

Design locked by the user on 2026-11-08, Asia/Jerusalem.

Design record: [interactive visual plan](visual-plan.html). Editable design source: [visual-plan.yaml](visual-plan.yaml).

Status: implementation completed locally on 2026-11-08; release verification is recorded in [implementation and QA](implementation-and-qa.md). No scientific recalculation, data publication, GitHub push, or cloud deployment was performed.

## 1. Approved decisions and boundaries

- Use six consolidated task workspaces, not a renamed list of twenty pages.
- Make **Discover** the landing workspace, starting with finding interesting modules.
- Keep every existing tool, dataset, analysis definition, table filter, and publication export capability.
- Use **Apply analysis** for changes that require data reads or statistical work. Appearance changes update immediately from the applied result.
- Stay with Streamlit and the current hosting/backend. Do not introduce a frontend rewrite, new service, custom task queue, or another caching framework.
- Preserve the current dataset defaults and scientific settings unless the user explicitly selects alternatives. Changing the landing page must not silently change the analysis population or statistical meaning.
- Do not recompute networks, eigengenes, masks, predictions, MDC, or KEGG. Do not modify public donor permissions, credentials, private mappings, or existing data bundles.
- Preserve running analysis jobs and the import interfaces they use.
- Prepare separately testable code changes. No automatic push or deployment.

## 2. Grounded starting point

The reviewed `streamlit_app.py` contains approximately 11,000 lines and a flat twenty-item analysis selector. Several specialized pages already stop before the generic module path and use lazy imports. Preserve that behavior rather than replacing functioning lazy paths.

The generic module path currently reads donor scores and attaches metadata before reaching some MDC, KEGG, and composition branches. It also displays common KEGG/composition content before many view-specific branches. These are concrete candidates for scoped loading; they are not a measured attribution of all lag.

Startup also incorporates prediction manifests into a shared cache digest and clears all `st.cache_data` entries when that digest changes. Replace this broad coupling with relevant-version cache keys without weakening data freshness or integrity checks.

Reuse these existing components:

- `app_helpers/data.py`, `drive_data.py`, and source-specific readers for predicate reads, local hot files, checksums, and download locks.
- `streaming_associations.py`, `correlations.py`, and the existing source-specific statistical helpers.
- `streamlit_compat.py` for stateful, active-only tab execution.
- `charts.py` and `figure_export/` for figures, publication controls, rendering limits, and downloads.
- Existing regional, validation, matched-control, comparison, and control-calibrated interfaces.

The working checkout has unrelated validation-data and M5 changes. Record the starting commit and dirty paths; do not stage, revert, or move those changes. Use a clean refactor branch/worktree, and reconcile its source baseline with the intended release branch before merging.

## 3. Workspace destinations

| Workspace | Existing tools and content |
|---|---|
| **Discover** | Module finder, correlation heatmaps, CT–TS screen, distribution-differentiation rankings |
| **Module** | Overview/composition/entropy, Associations and score-comparison bars, Feature distributions, MDC/network change, KEGG/biology, contextual Statistics |
| **Compare** | Module–outcome comparison, Partition comparison, Validation & replication, relevant matched-control comparison/adjustment results |
| **Prediction** | All existing prediction modes, Protocol v1.1, catalogs, diagnostics, explanations, coefficients, and exports |
| **Donors & edges** | Donor explorer, Gene & edge associations, Edge summaries, Edge volcano, Donor edge explorer, expression/residual exploration |
| **Data & methods** | Methods, dataset inventory, provenance, availability/completion, and existing catalog downloads |

The current **Matched-control results** page is not dropped: its dataset becomes selectable through a thin adapter; donor and module inspection belong in Module, comparisons in Compare, and cohort/provenance information in Data & methods. Retain its specialized result sections until each has an explicit destination and passing tests.

MDC and KEGG retain their existing all-module/pathway/category scopes inside their workspace destinations. Consolidation must not restrict these tools to the selected module.

Independent regional sources keep separate module selectors for AC/DLPFC/PCG or VH/SMA/QUAD and their side-by-side plots. Never equate regional modules by numeric ID.

## 4. Minimal application interfaces

Keep `streamlit_app.py` as the entrypoint and reduce it to initialization, compact common context, native navigation, and execution of the selected workspace.

Proposed UI-only organization:

- `app_helpers/app_shell.py`: six-route table, supported legacy-link translation, and navigation setup.
- `app_helpers/analysis_context.py`: validated analysis selections, pending/applied state, and compact context labels.
- `app_helpers/workspaces/`: one renderer for each of the six workspaces. Extract existing renderers incrementally; do not put the old monolith behind a new wrapper.
- Existing source-specific UI/data modules: small adapters where their contracts differ. Reuse their registries rather than introduce a second data-capability registry.
- Existing chart, statistics, export, and loader modules: remain the functional source of truth.

Use native [Streamlit navigation](https://docs.streamlit.io/1.64.0/develop/api-reference/navigation/st.navigation) with callable pages. Only the selected workspace executes. Do not rely on the absence of visible output to imply the absence of computation.

### Analysis context and state

The applied context records the complete statistical identity: dataset/module definition, source, cohort, method/estimator, namespaced module selections, feature/component, transformation, edge rule and direction/threshold, score normalization, outcome, donor/group filters, correlation method, minimum group size, and relevant data versions.

- Dataset/source/cohort selectors may load small manifests to resolve dependent options. They must not query donor scores while a new analysis is being composed.
- Batch expensive settings with a native [form](https://docs.streamlit.io/develop/concepts/architecture/forms). Apply validates the whole selection before replacing the applied context.
- Display a clear pending-change notice. Existing figures, statistics, and exports retain their applied context until a valid Apply action finishes.
- Navigation uses the last valid applied context; it must not silently apply unfinished changes.
- Appearance state is separate: colors, fonts, labels, legend position, visible trend lines, and export dimensions do not invalidate statistical results.
- Reuse [fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments) where an independent appearance/export panel can rerun without reloading its analysis. Preserve the current exporter’s resource limits.
- Use active-only tabs through `streamlit_compat.py`. A collapsed panel is not inherently lazy: guard expensive content with supported open state or an explicit action. Native [tabs](https://docs.streamlit.io/1.64.0/develop/api-reference/layout/st.tabs) and [expanders](https://docs.streamlit.io/1.64.0/develop/api-reference/layout/st.expander) compute hidden content by default unless conditional execution is enabled.

### Context isolation

- A Discover result opens Module with exactly its source, cohort, mask, outcome, feature, component, and statistical context. Back restores the ranked search.
- Compare seeds a newly added series from the current context. It does not overwrite other series, which retain their own source/cohort identities.
- Prediction owns its saved-model context. Exploration donor filters, masks, and cohorts must never be applied to a saved prediction catalog as if the model had been refitted.
- Preserve CAMEL/LIONESS naming, DLPFC labeling, nominal-outcome tests, unavailable combinations, and exact statistical family labels.
- Control-calibrated quantities and their `q_search`/other correction meanings remain distinct. Do not normalize different statistical semantics into a generic FDR label.

## 5. Implementation sequence and gates

### Milestone 1 — Baseline and preservation tests

1. Inventory every current route, subview, default, deep link, source capability, and exporter entry point.
2. Record representative numeric outputs before extraction: sample counts, correlations/p-values/FDR family sizes, categorical effects, MDC values, KEGG rows, and saved prediction metrics.
3. Add test-only loader spies and a reproducible local benchmark script. Capture cold and warm startup, navigation, data reads/downloads, analysis latency, appearance-change latency, peak process-tree memory, and export-child memory.
4. Exercise one and two simultaneous sessions. Keep profiling data private and exclude clinical payloads from logs.

**Gate:** all twenty old routes have mapped destinations; existing failures are identified; baseline artifacts distinguish local observations from unknown Community Cloud limits. Freeze measured release budgets before optimizing, not after seeing favorable results.

### Milestone 2 — Page-scoped loading without a navigation redesign

1. Extract renderers and their data requirements incrementally while keeping the familiar entry routes working.
2. Move donor-score reads and metadata joins inside views that actually need them.
3. Load MDC, KEGG, details, and statistics only for their active section/scope.
4. Separate prediction-manifest initialization from general exploration startup.
5. Keep bounded data caches and relevant dataset-version keys; remove global invalidation only after freshness regression tests pass. Preserve file integrity and download locking.

**Gate:** fixed-context outputs match the baseline; detail/KEGG/MDC routes perform no unrelated donor-score reads; inactive views do not fetch their data. This milestone can be reviewed independently of the new navigation.

### Milestone 3 — Native shell and applied context

1. Introduce the six workspaces, Discover landing, and compact dataset/cohort context.
2. Add draft/applied state, Apply analysis, and context-sensitive advanced controls.
3. Translate existing `view`, `screen_*`, and supported source-selection links into the equivalent route and context. Explicit URL selections take precedence over stale session state after validation.
4. Preserve meaningful selections across navigation while resetting incompatible dataset-specific widgets safely.
5. Add scoped availability notices; never substitute a different source or original cohort silently.

**Gate:** navigation/deep-link/session-isolation tests pass; unapplied changes cannot relabel old figures or exports; switching appearance performs no donor/statistical query.

### Milestone 4 — Consolidated workflows

1. Finish Discover → Module → Back, preserving the exact ranked-analysis context.
2. Consolidate Module overview, associations, distributions, network change, and biology with one active subview at a time.
3. Route brain/motor, validation, regional, matched-control, and calibrated datasets through their existing capability-aware adapters.
4. Connect Compare, Donors & edges, and Data & methods to the shared shell.
5. Move prediction UI renderers without changing catalogs, fitted model identity, or prediction-only controls.
6. Replace repeated metric-card/composition blocks with a concise applied-context summary and explicit details action.

**Gate:** every original tool remains reachable; regional tissue selectors stay independent; comparison-series context and prediction catalog context remain isolated; adjusted and unadjusted results remain identifiable.

### Milestone 5 — Regression, resource validation, and release preparation

1. Update navigation-dependent `AppTest` helpers without deleting the scientific assertions behind the old page tests.
2. Run the relevant existing tests for correlations, distributions, regional sources, validation, matched-control, calibrated scores, comparison, MDC, KEGG, prediction exports, Drive, and publication rendering.
3. Browser-test desktop and narrow layouts, keyboard navigation, back/deep links, pending settings, long labels, table filters, downloads, and representative PNG/PDF exports.
4. Repeat the baseline benchmark with identical scenarios, cache conditions, data, and concurrent sessions.
5. Verify data/manifest hashes and donor-publication boundaries are unchanged. Review the diff for credentials, manuscripts, private outputs, unrelated changes, and accidental analysis imports with side effects.
6. Prepare a code-only release with a rollback tag/commit and a short old-page → new-location guide. Deployment and push remain separate authorized actions.

**Gate:** no scientific/result drift; no missing route; no stale context/export mismatch; no out-of-memory failure in the measured session/export workload. Report latency, memory, and I/O changes rather than claim improvement from code size alone.

## 6. Required tests and performance assertions

- **Numerical preservation:** identical donor sets, family membership, signs, p/FDR values, counts, and missingness for fixed selections; exact saved-table equality where possible, existing numerical tolerances otherwise.
- **Capabilities:** cover completed brain/motor multi-tissue, independent regional, original/validation, matched-control, LIONESS/CAMEL/BONOBO/calibrated combinations where supported. Test explicit unavailable states too.
- **State:** dataset changes cannot reuse another dataset’s module ID; grouping changes cannot reuse a wrong FDR family; draft settings cannot change applied labels or downloaded results.
- **Execution:** loader spies verify no inactive workspace/subview loads; cosmetic changes cause zero analysis reads or statistical recalculations; changing only a prediction catalog version does not invalidate unrelated exploration data.
- **Bounded memory:** preserve streaming for all-module work; do not retain whole-dataset donor frames in session state; cap result caches. Repeated navigation and two-session tests must fit the resource budget frozen in Milestone 1 and show no continuing retained-memory growth after warm-up.
- **Export:** preserve `render_exportable_plot`, presets, combined/separate panels, selected-module annotations, color/label controls, PNG/PDF, timeout cleanup, and bounded rendered-byte cache.
- **Compatibility:** test old query URLs and hot-reload public helper imports as well as new callable-page paths. Retain compatibility shims needed by analysis scripts and tests.
- **Privacy:** no new donor-level data, raw expression, permissions, or identifiers are introduced; profiles/logs cannot serialize sensitive data or credentials.

The sections above retain the approved implementation contract. See the implementation/QA record for measured local performance, known pre-existing test failures, and deployment limits. The locked visual design does not authorize scientific reruns, public data expansion, or an automatic GitHub push.
