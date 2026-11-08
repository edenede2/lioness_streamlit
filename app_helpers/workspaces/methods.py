from __future__ import annotations

from app_helpers.ui_runtime import (
    DATA_DIR,
    MUSCLE_MODULE_SETS,
    cached_effect_size_manifest,
    cached_feature_definitions,
    cached_tissue_mapping,
    ensure_data_path,
    filterable_dataframe,
    st,
)


def render_methods(context):
    allowed_methods = context["allowed_methods"]
    method = context["method"]
    module_count = context["module_count"]
    module_manifest = context["module_manifest"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    st.subheader("Analysis methods and deploy data")
    donor_counts = module_manifest.get(
        "donors_per_method", module_manifest.get("donors", 450)
    )
    selected_donor_count = int(
        donor_counts.get(method, 450)
        if isinstance(donor_counts, dict)
        else donor_counts
    )
    group_counts = module_manifest.get(
        "diagnosis_counts_per_method",
        module_manifest.get(
            "diagnosis_counts", {"Control": 164, "MCI": 119, "AD": 167}
        ),
    )
    selected_group_counts = group_counts.get(method, group_counts)
    selected_control_count = int(selected_group_counts.get("Control", 164))
    st.markdown(
        f"**Selected module definition:** {module_set_label}. Module IDs are scoped to "
        "this definition; the same number in the other definition can contain a different "
        "set of tissue-gene members."
    )
    if "standard" in allowed_methods:
        st.markdown(
            f"**Standard LIONESS** scores each donor relative to the complete "
            f"{selected_donor_count}-donor reference network. "
            f"**Control-referenced LIONESS** uses the {selected_control_count} Controls "
            "as the reference and scores Controls by leave-one-out and MCI/AD by donor "
            "addition."
        )
    else:
        st.markdown(
            f"**Control-referenced LIONESS** is the only supported network method for "
            f"this Control-derived module definition. It uses the {selected_control_count} "
            "Controls as the reference and scores Controls by leave-one-out and MCI/AD "
            "by donor addition."
        )
    if "bonobo" in module_manifest.get("estimators", []):
        st.markdown(
            "**BONOBO** estimates one empirical-Bayes covariance network for each donor. "
            "The donor/module delta is estimated automatically and the app can summarize "
            "all edges or edges passing the native posterior p<0.05 or within-donor-module "
            "BH FDR<0.05 rule. BONOBO edge significance tests nonzero covariance, not "
            "AD/MCI/Control differences."
        )
    formula_col1, formula_col2 = st.columns(2)
    with formula_col1:
        st.markdown("Standard donor network")
        st.latex(
            rf"A_i = N A_{{all}} - (N-1)A_{{all\\setminus i}},\\quad N={selected_donor_count}"
        )
        st.markdown("Control donor")
        st.latex(
            rf"A_i = n_C A_C - (n_C-1)A_{{C\\setminus i}},\\quad n_C={selected_control_count}"
        )
    with formula_col2:
        st.markdown("MCI or AD donor added to the Control reference")
        st.latex(r"A_i = (n_C+1)A_{C+i} - n_C A_C")
        if module_set in MUSCLE_MODULE_SETS:
            st.markdown(
                "Donor networks use the frozen signedAlt block powers from the muscle "
                "partition: VH=5, SMA=7, QUAD=13; VH–SMA=8, VH–QUAD=10, and "
                "SMA–QUAD=10. Each undirected edge is counted once and the diagonal is excluded."
            )
        else:
            st.markdown(
                "Networks use signedAlt adjacency with β=3 for tissue-specific edges and "
                "β=2 for cross-tissue edges. Each undirected edge is counted once and the "
                "diagonal is excluded."
            )
    st.markdown(
        "For each estimator/method, module, feature, and CT/TS or resolved component, "
        "the app stores raw values, a robust-scale asinh transformation, and a rank "
        f"inverse-normal Z-score calculated across all {selected_donor_count} donors."
    )
    st.markdown(
        "For the all-donor CorShrink maximum-component sensitivity, transformations are "
        "instead fitted independently within each method, module, feature, and resolved "
        "component across that component's available donors. Pooled CT/TS remains limited "
        "to the comparable 450-donor cohort."
    )
    st.markdown(
        "**Grouped associations** use diagnosis by default and may instead stratify "
        "correlations by ROS/MAP study, clusters, CogDx, Braak, CERAD, ADNC, Parkinsonism, sex code, "
        "or APOE genotype. Pearson and Spearman are calculated separately within each "
        "eligible displayed level. Their BH FDR families contain only the "
        f"{module_count} modules in the selected definition, while holding every other analysis and "
        "grouping field fixed; missing and constant tests are excluded from the denominator. "
        "OLS trend lines are descriptive guides. Categorical comparisons use Kruskal–Wallis "
        "and epsilon-squared and never correlate nominal codes. Feature-distribution "
        "differentiation adds two-sided Mann–Whitney contrasts, Cliff’s delta, probability "
        "of superiority, deterministic 1,000-resample bootstrap intervals, and an optional "
        "two-sided KS shape diagnostic. Pairwise BH is calculated across valid modules "
        "separately for every fixed group contrast and component; category contrasts are "
        "never pooled into one correction."
    )
    st.markdown(
        "**ROSMAP clusters** are an unordered four-class donor partition available for "
        "313 of the 450 donors (Cluster 1/2/3/4: 168/71/44/30; 137 unavailable). "
        "Cluster association views use Kruskal–Wallis tests and epsilon-squared, require "
        "at least five donors per tested category, and apply BH across modules only within "
        "fixed analysis strata. The numeric class codes are never used for Pearson or "
        "Spearman correlation. Cluster prediction is exploratory because it reuses the "
        "diagnosis-derived tissue-neutral AD panel and cluster membership is strongly "
        "associated with diagnosis in this cohort."
    )
    st.markdown(
        "**ROS/MAP study membership** comes from the August-2025 source `study` field. "
        "The common brain cohort contains 209 ROS and 241 MAP donors. Both motor partitions "
        "are MAP-only (357 donors), so between-study effects and p-values are unavailable there. "
        "Expanded cohorts retain missing study labels without inference. Study is available for "
        "categorical outcomes, distribution grouping, correlation strata, coloring and hovers; "
        "study codes never enter numeric correlations or new prediction models. "
        "An approved, namespace-isolated repository overlay supports older Drive metadata."
    )
    st.markdown("#### AD–Control differential-edge filtering")
    current_effect_manifest = cached_effect_size_manifest(module_set)
    current_split_counts = {
        (str(row.get("ad_control_split")), str(row.get("diagnosis_group"))): int(
            row.get("n", 0)
        )
        for row in current_effect_manifest.get("split_counts", [])
    }
    discovery_ad = current_split_counts.get(("Discovery", "AD"), 117)
    discovery_control = current_split_counts.get(("Discovery", "Control"), 114)
    validation_ad = current_split_counts.get(("Validation", "AD"), 50)
    validation_control = current_split_counts.get(("Validation", "Control"), 50)
    external_mci = current_split_counts.get(("MCI_external", "MCI"), 119)
    st.markdown(
        f"A diagnosis-and-sex-stratified discovery cohort ({discovery_ad} AD, "
        f"{discovery_control} Control) defines "
        "the optional edge mask with two-sided Welch tests on raw signedAlt donor-edge "
        "weights. Every edge stores AD and Control means, AD-minus-Control difference, "
        "Hedges’ g, p-value, **global BH FDR**, and **per-module BH FDR**. Global BH is "
        "calculated across all tested edges within one module-definition, estimator, "
        "network-method, and discovery/validation family; per-module BH is calculated "
        "across all edges of one module. The sidebar FDR-scope and 0.05/0.10 cutoff "
        "choices control filtered features, associations, distributions, heatmaps, and "
        "edge summaries. The same scope and cutoff initialize the volcano view, where "
        "the display-only threshold can be changed without recalculating module scores. "
        "Selecting All edges continues to use the original unfiltered data."
    )
    st.markdown(
        f"The default filtered evaluation uses {validation_ad} held-out AD and "
        f"{validation_control} held-out Control donors. MCI (n={external_mci}) is "
        "external to edge selection. Validation labels do not "
        "alter the discovery mask, although the underlying unsupervised donor-network "
        "estimation is not a completely external network validation."
    )
    effect_threshold_text = (
        "the fixed |g| ≥ 0.40 mask"
        if current_effect_manifest.get("available_fixed_hedges_g") == [0.4]
        else "fixed absolute thresholds 0.20–0.50 or exact within-module top 1%, 5%, and 10% masks"
    )
    st.markdown(
        "**Effect-size filtering** reuses the same discovery edge table without rerunning "
        f"Welch tests. This selected module definition exposes {effect_threshold_text}; "
        "other completed catalogs may additionally expose percentile masks. Raw "
        "AD-minus-Control mean differences "
        "use the within-module percentage masks. Percentage selection uses absolute effect "
        "and deterministic edge-index tie breaking before applying the AD-higher or "
        "Control-higher direction. BONOBO additionally intersects this mask with the "
        "selected donor-specific native significance rule. Discovery-donor separation is "
        "selection-biased, so this option is exploratory and defaults to held-out donors."
    )
    st.markdown(
        "The **Donor edge explorer** compares pseudonymous donors with similar standardized "
        "module scores but different outcomes. It reports resolved component weights, edge "
        "counts, quantiles, retained proportions, and cancellation. Effect-filtered views "
        "load only the selected module’s bounded top-20 gene-symbol driver shards. These "
        "edge patterns are descriptive rather than causal explanations of phenotype."
    )
    st.markdown(
        "Its **endpoint-expression** section ranks the selected module’s bounded exact "
        "edge catalog and lazy-loads only that module’s pseudonymous within-gene expression "
        "Z-scores. A 2D edge fits endpoint B on endpoint A; a 3D view is available only "
        "when the top-K graph contains all three edges of a triangle and fits the chosen "
        "Z-axis gene on the other two genes. OLS lines/planes and residuals describe "
        "co-expression geometry, not phenotype prediction. Gene tables join each selected "
        "endpoint to significant or all reported module KEGG overlap memberships. Source "
        "expression values, donor IDs, projid, and Ensembl identifiers are not deployed."
    )

    st.markdown("#### Module differential connectivity")
    st.markdown(
        "All-edge MDC is an independent group-network comparison supplied as module-level context: "
        "mean absolute AD adjacency divided by mean absolute Control adjacency. The app "
        "shows total, pooled TS/CT, and six tissue-resolved ratios and their conservative "
        "directional permutation FDR on raw-ratio or log2 scales. Separate TS and CT "
        "scatter plots relate MDC to normalized Shannon tissue-mixing entropy. The "
        "pathway-resolved tab annotates module MDC with component-matched KEGG enrichment; "
        "it is not a newly recomputed edge-level pathway MDC. Its broader tissue-union "
        "AD/Control cohort and absence of MCI are stated in the MDC tab. When the sidebar "
        "selects AD–Control differential edges, MDC is recomputed on the matching estimator/"
        "method discovery mask for Global or Per-module BH at 0.05 or 0.10. That fixed-mask "
        "permutation analysis is post-selection and exploratory, not independent validation."
    )

    st.markdown("#### Module feature definitions")
    filterable_dataframe(
        cached_feature_definitions(),
        table_key="feature_definitions",
        table_name="Module feature definitions",
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Tissue labels")
    filterable_dataframe(
        cached_tissue_mapping(),
        table_key="tissue_labels",
        table_name="Tissue labels",
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Public deploy safeguards")
    st.markdown(
        "The deploy Parquet files do not contain `projid`, the source donor identifier, "
        "or a donor-to-pseudonym key. Each point has a random-salted pseudonymous sample "
        "label that is stable within this bundle only. Diagnosis, the five primary phenotypes, "
        "and selected age, education, APOE, cognitive diagnosis, neuropathology, Parkinsonism, "
        "and nominal cluster membership "
        "fields are included for color, hover, and correlation views. Public release still "
        "requires confirmation that the governing ROSMAP data-use agreement permits sharing "
        "these donor-level derived and metadata values."
    )
    st.caption(
        "Sex is displayed as source Code 0/1 because this deploy bundle does not infer a label "
        "without an accompanying reviewed data dictionary."
    )
    manifest_path = ensure_data_path(DATA_DIR / "data_manifest.json")
    if manifest_path.is_file():
        panel = st.expander(
            "Deploy data manifest", key="methods_manifest_panel", on_change="rerun"
        )
        if panel.open:
            with panel:
                st.json(manifest_path.read_text(encoding="utf-8"))
