# ROSMAP Single-Sample Network Explorer

A GitHub-ready Streamlit app for exploring the completed ROSMAP LIONESS and
BONOBO analyses.

The app includes two explicitly separated module definitions:

- **Full-cohort L4 modules (154)**, including M1918, with Standard and
  Control-referenced LIONESS results.
- **Control-derived L4 modules (186)** with Control-referenced LIONESS results.
  Identically numbered modules in the two sets are not interchangeable.
- Standard and Control-referenced LIONESS, plus all-donor empirical-Bayes
  BONOBO for both module definitions.
- Twelve numeric outcomes: five cognitive/motor phenotypes, age, education,
  CogDx, Braak, CERAD, ADNC, and Parkinsonism.
- Six LIONESS feature families and three parallel BONOBO feature families.
- BONOBO all-edge scores and significant-edge scores based on either native
  posterior `p < 0.05` or within-donor/module BH `FDR < 0.05`.
- Aggregate cross-tissue (CT) and tissue-specific (TS) views.
- Tissue-resolved CT pair and TS tissue views using the label **DLPFC**.
- Diagnosis-specific scatter plots and OLS trends for Control, MCI, and AD.
- A global association-correlation selector, defaulting to Spearman with Pearson
  available from the sidebar; it controls scatter annotations, heatmaps, the
  CT-vs-TS screen, and the primary statistics table.
- Point coloring by diagnosis or any numeric phenotype/outcome, with selectable
  continuous color scales and palette reversal.
- Expanded hover details for cognition, motor function, age, education, APOE,
  CogDx, Braak, CERAD, ADNC, and Parkinsonism.
- Feature-only histograms and violin distributions.
- Selected-module and all-module correlation heatmaps, with downloadable
  Pearson, Spearman, p-value, and displayed-family FDR tables.
- Optional average-linkage clustering of heatmap rows, columns, or both.
- Module differential connectivity (MDC) for total, pooled TS, pooled CT, three
  tissue-specific blocks, and three cross-tissue pairs, including directional
  FDR significance and an all-module resolved heatmap.
- Donor-level edge summaries for nine edge scopes, with diagnosis-level summary
  statistics calculated interactively.
- Full raw, robust, RINT, leave-one-out, CT-vs-TS, and FDR statistics.
- A descriptive CT-vs-TS screen across all modules.
- Every row and column of the tissue-expanded KEGG enrichment table, filterable
  and downloadable for the selected module or all modules, with module, FDR,
  significance, statistical scope, category, sub-category, pathway, and gene-search
  filters. Each row includes the whole expanded-tissue p/FDR and separate AC,
  DLPFC, and PCG p/FDR values.
- Level-4 module composition from the SE2 details file: total module size,
  represented tissues, dominant tissue, per-tissue gene counts, and proportions.
- Raw and normalized Shannon tissue-mixing entropy as continuous complements to
  the discrete CT/TS module class.
- Interactive module-size distributions and size-ranked AC/DLPFC/PCG composition
  bars, filterable to CT modules, TS modules, or both.

## Important public-release check

The deploy bundle removes `projid` and the source donor identifier. Plot points use
random-salted pseudonymous labels whose source mapping and salt are discarded.
Diagnosis, the five primary phenotypes, and selected clinical/neuropathology fields
remain for color, hover, and correlation views.

Pseudonymization is not the same as permission to redistribute donor-level derived
data. Before making the GitHub repository public, confirm that the ROSMAP data-use
agreement and your institutional approval allow public sharing of these derived
individual-level values. If they do not, deploy from a private repository or replace
the donor-level Parquet files with aggregate-only data.

## Run locally

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

On PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

## Publish with Streamlit Community Cloud

1. Create a new GitHub repository and copy the contents of this directory to the
   repository root. Keep the `data/` directory; the app does not use the original
   analysis filesystem.
2. Commit and push the repository. Every packaged file is below 95 MiB (and thus
   below GitHub's 100 MB per-file limit), so Git LFS is not required for this build.
3. In Streamlit Community Cloud, choose **Create app**, select the repository and
   branch, set the entrypoint to `streamlit_app.py`, and select **Python 3.12** in
   Advanced settings.
4. Deploy. No secrets are required.

`requirements.txt` pins the exact direct dependency versions validated with a clean
resolver install, `pip check`, the complete test suite, and a Streamlit app execution.

### Required Python version

Select **Python 3.12** in Streamlit Community Cloud's Advanced settings. Do not use
Python 3.14 with this dependency set: PyArrow 21 does not publish a CPython 3.14
wheel, so the platform attempts an unsupported source build and reports
`Failed to build pyarrow`.

If the app was already created with Python 3.14, changing `requirements.txt` is not
enough. Delete the Community Cloud app and deploy it again from the same repository,
selecting Python 3.12 in Advanced settings. Deleting the cloud app does not delete
the GitHub repository.

If this directory remains inside the larger analysis repository instead, use
`apps/lioness_streamlit/streamlit_app.py` as the entrypoint.

## Rebuild the public data bundle

The checked-in deploy data were generated from both completed analysis outputs.
The default rebuild creates both bundles together and gives each donor one new
pseudonymous label shared consistently between module definitions:

```bash
python scripts/build_public_data.py
```

Optional source paths can be supplied explicitly:

```bash
python scripts/build_public_data.py \
  --analysis-root /path/to/20260817_standard_control_anchored_allmodules_5phenotypes_6features \
  --control-derived-analysis-root /path/to/20260818_control_anchored_control_derived_l4_5phenotypes_6features \
  --kegg /path/to/method4_tissue_expanded_kegg_annotated.tsv \
  --kegg-per-tissue /path/to/method2_meta_kegg_annotated.tsv \
  --control-derived-kegg /path/to/control_derived_method4_tissue_expanded_kegg_annotated.tsv \
  --control-derived-kegg-per-tissue /path/to/control_derived_method2_meta_kegg_annotated.tsv \
  --mdc /path/to/AD_vs_Control_MDC_Preservation.tsv \
  --control-derived-mdc /path/to/AD_vs_Control_control_derived_l4_MDC_only.tsv \
  --module-details /path/to/se2_details_filtered_4.csv \
  --control-derived-module-details /path/to/speakeasy_clusters_details_level_4_filtered.csv \
  --full-assignments /path/to/se2_table_filtered_4.csv \
  --control-derived-assignments /path/to/speakeasy_clusters_table_level_4_filtered.csv \
  --bonobo-app-data-root /path/to/bonobo/app_data \
  --bonobo-network-root /path/to/bonobo/network_outputs \
  --lioness-expansion-root /path/to/lioness/edge_summaries_entropy \
  --lioness-all12-stats-root /path/to/lioness/all12_statistics \
  --resolved-mdc-root /path/to/resolved_mdc
```

To refresh only the compact MDC table without rebuilding the donor-level files or
changing pseudonymous sample labels:

```bash
python scripts/build_public_data.py \
  --mdc-only \
  --mdc /path/to/AD_vs_Control_MDC_Preservation.tsv
```

To refresh only the module composition table without rebuilding donor-level files:

```bash
python scripts/build_public_data.py \
  --module-details-only \
  --module-details /path/to/se2_details_filtered_4.csv
```

To refresh the all-12-outcome aggregate and tissue-resolved LIONESS statistics for
both module definitions without changing donor pseudonyms or rebuilding plot data:

```bash
python scripts/build_public_data.py --statistics-only
```

To refresh only the whole and per-region KEGG tables without changing donor
pseudonyms or rebuilding plot data:

```bash
python scripts/build_public_data.py --kegg-only
```

The build is intentionally one-way: donor/projid columns and the pseudonym salt are
not written. `data/data_manifest.json` records separate module-set row counts,
source hashes, correction-family sizes, unavailable CT modules, and deploy-file hashes.

Validate a staged or deployed bundle, including privacy, hashes, schemas, row counts,
sample consistency, edge identities, entropy ranges, resolved MDC, and the 95-MiB cap:

```bash
python scripts/validate_public_bundle.py /path/to/data
```

## Data organization

- `data/`: the existing 154-module full-cohort bundle.
- `data/control_derived/`: the isolated 186-module Control-derived bundle. It has
  its own plot data, statistics, KEGG, MDC, annotations, and module-details files.
- `aggregate_plot_data.parquet`: LIONESS CT/TS raw, asinh, and Z-score features.
- `resolved_plot_data.parquet`: LIONESS DLPFC/AC/PCG tissue and tissue-pair features.
- `bonobo/{all,native_p05,bh_fdr05}/`: separately queryable BONOBO plot and
  statistics Parquets for each edge rule.
- `edge_summaries/`: donor-level edge counts and signed/positive/negative/absolute
  weight summaries, sharded by estimator, method, and edge rule.
- `data/aggregate_statistics.parquet`: complete aggregate robust statistics.
- `data/resolved_statistics.parquet`: complete tissue-resolved robust statistics.
- `data/sample_metadata.parquet`: 450 pseudonymous rows with the hover/color outcomes.
- `data/kegg_tissue_expanded_full.tsv`: full readable KEGG source table.
- `data/kegg_tissue_expanded_full.parquet`: query-efficient KEGG copy.
- `data/module_kegg_annotations.tsv`: one displayed KEGG annotation per module.
- `data/module_details.tsv`: validated level-4 module sizes, represented tissues,
  per-tissue gene counts, tissue proportions, and tissue-mixing entropy, using
  the DLPFC label.
- `mdc_ad_vs_control_summary.tsv`: total, TS, and CT MDC ratios with directional
  FDR values for the selected module definition.
- `mdc_resolved_ad_vs_control.tsv`: AC, DLPFC, PCG, AC-DLPFC, AC-PCG, and
  DLPFC-PCG MDC ratios and directional permutation FDRs.
- `data/feature_definitions.tsv`: feature calculation definitions.
- `data/tissue_mapping.tsv`: public tissue display names.

## FDR scope

Association FDR correction is performed separately by module definition,
estimator/network method, component family, correlation method, and BONOBO edge
rule. The expanded global columns apply Benjamini-Hochberg correction over all 12
numeric outcomes; primary-five global columns are retained for backward comparison,
and within-outcome columns correct each outcome family. The
whole-regions KEGG FDR is corrected within each module across pathways reported by
the tissue-expanded test. Each AC, DLPFC, and PCG FDR is corrected separately within
its module and region across the stable set of 350 KEGG pathways; pathways below the
minimum overlap receive p=1. These values come from the supplied enrichment analyses
and are not recalculated by the app.

The additional heatmap tables calculate Pearson and Spearman correlations on demand.
Their `*_fdr_displayed_family` columns use Benjamini-Hochberg correction over the
complete correlation family used by that displayed module or all-module heatmap; the
scope is stated directly below each heatmap.

## MDC scope

The packaged MDC analysis uses AD as the reference and Control as the target, so
`MDC = mean(abs(adjacency_AD)) / mean(abs(adjacency_Control))`. Values above 1
therefore indicate higher connectivity in AD, and values below 1 indicate higher
connectivity in Control. The same calculation is supplied for all edges, TS edges,
and CT edges. Adjacencies use signedAlt with beta 3 for TS and beta 2 for CT.

Each module definition has its matching MDC source and uses 200 sample permutations
plus 200 gene permutations. For each edge scope and direction, permutation p-values
were Benjamini-Hochberg adjusted across the modules in that definition (154 or 186).
The displayed directional FDR is the maximum of the sample-permutation and
gene-permutation q-values for the observed direction.

The MDC and LIONESS cohorts are related but not identical. MDC assembled every donor
available in at least one tissue: 517 AD and 408 Control donors. This includes all 167
AD and 164 Control complete-three-tissue donors in LIONESS plus donors with partial
tissue availability. MCI was not included in MDC. The app labels MDC as contextual
module-level evidence and does not imply that it changes with the selected LIONESS
method.

The six resolved MDC components use the same AD-reference/Control-target design.
Directional FDR is calculated separately for every component and direction and is
the larger of the sample-permutation and tissue-count-preserving gene-permutation BH
values. Structurally absent components remain unavailable rather than being encoded
as zero.

## BONOBO significance and sparse features

BONOBO edge significance tests whether an individual donor-specific covariance edge
differs from zero. It does not test whether AD, MCI, and Control differ. The native
rule retains two-sided posterior `p < 0.05`; the stricter rule applies BH correction
within each donor-module over its undirected edges and retains `FDR < 0.05`.
Structurally valid scopes with no retained edge receive zero-valued sparse-network
features, and the app explicitly warns when a selected score is constant. A
structurally nonexistent tissue block remains missing.
