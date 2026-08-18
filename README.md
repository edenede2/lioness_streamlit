# ROSMAP LIONESS Network Explorer

A GitHub-ready Streamlit app for exploring the completed standard and
Control-referenced ROSMAP LIONESS analyses.

The app includes:

- All 154 level-4 modules, including M1918.
- Standard and Control-referenced LIONESS results.
- Five cognitive and motor phenotypes.
- Six module feature families.
- Aggregate cross-tissue (CT) and tissue-specific (TS) views.
- Tissue-resolved CT pair and TS tissue views using the label **DLPFC**.
- Diagnosis-specific scatter plots and OLS trends for Control, MCI, and AD.
- Feature-only histograms and violin distributions.
- Full raw, robust, RINT, leave-one-out, CT-vs-TS, and FDR statistics.
- A descriptive CT-vs-TS screen across all modules.
- Every row and column of the tissue-expanded KEGG enrichment table, filterable
  and downloadable per module or for all modules.

## Important public-release check

The deploy bundle removes `projid`, the source donor identifier, and detailed
clinical metadata. Plot points use random-salted pseudonymous labels whose source
mapping and salt are discarded. Diagnosis and the five phenotype values remain
because they are necessary for the figures.

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
2. Commit and push the repository. Every packaged file is below GitHub's 100 MB
   per-file limit, so Git LFS is not required for this build.
3. In Streamlit Community Cloud, choose **Create app**, select the repository and
   branch, and set the entrypoint to `streamlit_app.py`.
4. Deploy. No secrets are required.

If this directory remains inside the larger analysis repository instead, use
`apps/lioness_streamlit/streamlit_app.py` as the entrypoint.

## Rebuild the public data bundle

The checked-in deploy data were generated from the completed analysis outputs. To
rebuild after the analysis changes, run this from the app directory:

```bash
python scripts/build_public_data.py
```

Optional source paths can be supplied explicitly:

```bash
python scripts/build_public_data.py \
  --analysis-root /path/to/20260817_standard_control_anchored_allmodules_5phenotypes_6features \
  --kegg /path/to/method4_tissue_expanded_kegg_annotated.tsv
```

The build is intentionally one-way: donor/projid columns and the pseudonym salt are
not written. `data/data_manifest.json` records row counts, hashes, and package size.

## Data organization

- `data/aggregate_plot_data.parquet`: CT/TS raw, asinh, and Z-score features.
- `data/resolved_plot_data.parquet`: DLPFC/AC/PCG tissue and tissue-pair features.
- `data/aggregate_statistics.parquet`: complete aggregate robust statistics.
- `data/resolved_statistics.parquet`: complete tissue-resolved robust statistics.
- `data/kegg_tissue_expanded_full.tsv`: full readable KEGG source table.
- `data/kegg_tissue_expanded_full.parquet`: query-efficient KEGG copy.
- `data/module_kegg_annotations.tsv`: one displayed KEGG annotation per module.
- `data/feature_definitions.tsv`: feature calculation definitions.
- `data/tissue_mapping.tsv`: public tissue display names.

## FDR scope

Benjamini-Hochberg FDR correction was performed separately within each LIONESS
method. Aggregate global CT and TS correlation FDRs each cover 13,860 tests;
aggregate global CT-vs-TS FDR covers 13,860 dependent-correlation tests, while its
within-phenotype version covers 2,772. Tissue-resolved global correction covers
83,160 component tests and within-phenotype correction covers 16,632. KEGG FDRs
come directly from the supplied tissue-expanded enrichment table and are not
recalculated by the app.

