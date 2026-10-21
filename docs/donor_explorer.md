# Donor explorer

Open **Analysis view → Donor explorer**. This view reads only the existing public
metadata: no expression, LIONESS/CAMEL scores, edges, predictions or new analyses.

Choose **Brain** or **Motor** first. Their pseudonym namespaces are separate;
identically spelled sample IDs are not evidence that they identify the same
person. No cross-dataset donor matching or combined unique-person count is made.

Brain coverage is reconstructed from the three approved maximum-tissue metadata
tables. Presets distinguish “DLPFC available” (other regions allowed) from
“DLPFC only,” and similarly distinguish region pairs from exact pairs. Custom
coverage supports at-least, exact and any-selected-region rules. All-three
coverage includes 464 donors; the original analysis cohort includes 450, and
incomplete-region validation excludes all 464, leaving 793. The motor public
bundle contains the shared 357 VH/SMA/QUAD donors, not every raw single-tissue
motor donor. Selecting Motor does not assert that these donors lack brain tissue.

The page provides category counts, overlapping component counts and mutually
exclusive coverage patterns; phenotype box/point, histogram, ECDF and scatter
views; missingness and observed-outcome overlap; and donor tables/details. Shared
categorical, numeric-range and observed-outcome filters apply to the selected
population. No outcome or absent region is imputed, and no new association/FDR
or group-difference tests are fitted on this page.

Diagnosis groups are the saved app labels, not recomputed from CogDx. The
original-450 preset preserves its original metadata. Other brain coverage
cohorts use the regional snapshot; alternative snapshots are not silently
coalesced. A tolerance-aware snapshot audit and selected-sample comparison
distinguish actual differences from floating-point serialization roundoff.
The current snapshots agree to numerical precision (`rtol=1e-10`, `atol=1e-12`).

Downloads retain the dataset namespace and metadata snapshot. Only existing
approved clinical fields and public pseudonyms are exposed. The page adds no
donor names, projid, new donors, raw expression, private mapping, dependencies
or public data files. Its two-entry metadata cache holds only a few MiB; only
the selected analysis section renders. Tests cover counts, missingness, filters,
namespace isolation, provenance, privacy, memory and real Streamlit interactions.
