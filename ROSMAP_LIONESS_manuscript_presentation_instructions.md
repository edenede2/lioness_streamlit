# ROSMAP LIONESS Manuscript-Oriented Presentation Package
## Instructions for Creating the Jupyter Notebook, HTML Report, and PowerPoint Presentation

## 1. Purpose

Create a **manuscript-oriented, story-driven analysis package** based on the **existing ROSMAP LIONESS/BONOBO application and its already-computed result files**.

The task is **not** to produce a catalog of everything available in the application. The task is to determine the strongest scientific story currently supported by the existing results and organize all relevant analyses around that story.

The central working hypothesis is:

> **Alzheimer’s disease reorganizes molecular coordination across brain regions in a pathway- and region-pair-specific manner, and these cross-regional alterations carry clinically relevant information at the individual-donor level that is not fully captured by within-region network organization.**

The final scientific narrative should branch into three phenotype-focused stories:

1. **AD disease state / AD-vs-Control prediction**
2. **Cognitive trajectories and cognitive phenotypes**
3. **Motor and Parkinsonian phenotypes**

All three branches should emerge from a common network-level foundation and reconnect in an integrated biological model at the end.

---

# 2. Required Deliverables

Create all of the following.

## 2.1 Reproducible Jupyter Notebook

The notebook should be the **source of truth** for the entire project.

It should contain:

- loading of existing analysis outputs;
- clear provenance for each dataset/result;
- descriptive summaries;
- integration of MDC, LIONESS, BONOBO, prediction, KEGG, module-selection, coefficient, and phenotype tables;
- figure-generation code;
- manuscript-style interpretation;
- concise figure captions;
- explicit distinction between direct result, interpretation, biological hypothesis, and proposed future analysis.

The notebook should be organized so that figures and tables can be regenerated without manually using Streamlit.

## 2.2 HTML Report

Export the completed notebook to HTML. The HTML should preserve figures, tables, headings, interpretation, and be suitable for sharing with supervisors/collaborators without requiring execution.

## 2.3 PowerPoint Presentation

Create a polished scientific PowerPoint based on the notebook.

Target approximately **18–24 main scientific slides**, plus backup/supplementary slides.

Every results slide should have:

- a **conclusion-style title**;
- preferably one major figure or one coherent multi-panel figure;
- no more than 1–3 short supporting statements;
- a clear bottom-line interpretation;
- speaker notes explaining what is shown, what can be concluded, what cannot be concluded, and the main caveat.

The presentation should look like the first internal presentation of a future high-impact manuscript, not like a software demo.

---

# 3. Critical Execution Constraint: Do Not Rerun Analyses

## 3.1 Do NOT rerun or refit existing pipelines

At this stage, do **not** recompute:

- LIONESS networks;
- BONOBO networks;
- module detection;
- SE2 clustering;
- MDC;
- differential-edge testing;
- KEGG enrichment;
- nested prediction models;
- targeted module selection;
- K selection;
- permutation tests;
- score-transformation sensitivity;
- any computationally expensive statistical pipeline.

The task is to **interrogate, integrate, visualize, and interpret results that already exist**.

## 3.2 Allowed operations

You may:

- read existing files;
- merge existing result tables;
- filter and rank existing results;
- compute simple descriptive summaries from existing outputs;
- create new plots from already-computed data;
- reproduce Streamlit plots directly from their underlying data;
- build composite figures not currently available in the app;
- calculate presentation-level summaries such as overlaps, counts, rank comparisons, descriptive medians, and selection-frequency summaries;
- create tables integrating multiple existing analyses.

Do not generate new inferential claims that require model refitting or new hypothesis testing.

## 3.3 Missing analyses

If a scientifically useful analysis has not yet been performed, **do not run it**. Create a clearly labeled section:

### Proposed analysis — not yet performed

Include:

1. scientific question;
2. required data;
3. proposed method;
4. expected figure;
5. competing hypotheses it could distinguish;
6. why it would strengthen the manuscript.

---

# 4. Current Score-Transformation Caveat

The previous targeted-prediction catalog was discovered to use **asinh-transformed LIONESS connectivity**, although it had initially been treated as if it were Raw.

A corrected analysis is being generated separately.

For this task:

- do not silently label legacy targeted results as Raw;
- preserve transformation provenance exactly;
- use only completed and validated transformation results;
- if Raw or RINT results are incomplete, mark them as interim/unavailable;
- do not rerun the transformation analysis as part of this task.

When the corrected catalog is available:

- **Raw should be the primary/default analysis**;
- asinh and RINT should be treated as robustness/sensitivity analyses.

The presentation structure should therefore be designed so that final prediction values can update automatically from the validated result tables.

---

# 5. Overall Manuscript Story

The presentation should follow this logic:

### Step 1
Brain regions contain local molecular networks, but disease may also alter the **coordination between regions**.

### Step 2
AD does not simply cause a global decrease in connectivity. Instead, the existing results suggest:

- some modules lose coordination;
- some gain coordination;
- CT and TS behavior can diverge;
- effects differ by region pair;
- effects differ by biological pathway.

### Step 3
Healthy-reference modules provide a complementary perspective:

> molecular programs that are normally coordinated across regions may become selectively decoupled in AD.

### Step 4
LIONESS/BONOBO allow these network phenomena to be evaluated at the individual-donor level.

### Step 5
The individual-level signal should then be organized into three phenotype branches:

1. AD disease state;
2. cognitive outcomes;
3. motor/Parkinsonian outcomes.

### Step 6
The branches should reconnect through recurring biological themes such as:

- synaptic-vesicle biology;
- neuronal signaling;
- mitochondrial / energy metabolism;
- neurodegeneration / proteostasis;
- glutamatergic signaling;
- axon guidance;
- immune/inflammatory remodeling;
- lipid/peroxisomal/redox metabolism.

---

# 6. Common Foundation: What Is Being Reorganized?

This section should precede all phenotype branches.

## 6.1 Cohort and brain regions

Introduce ROSMAP and the three regions AC, DLPFC, and PCG.

Explain the distinction between:

### TS
Within-region / tissue-specific network organization.

### CT
Cross-region network organization.

Use a simple conceptual diagram.

## 6.2 Two complementary module definitions

Present:

### Full-cohort modules
Modules discovered from the broader cohort.

Interpretation:

> useful for detecting disease-associated network architecture, including both loss and emergence of coordination.

### Control-derived modules
Modules defined from healthy Controls.

Interpretation:

> useful for asking how a healthy-reference molecular architecture is perturbed in disease.

Important:

- the module sets must remain completely separate;
- identically numbered modules from the two definitions are not equivalent.

## 6.3 Module structure and tissue mixing

Use only enough module-composition analysis to demonstrate that the CT framework is meaningful.

Candidate app plots:

- module-size distribution;
- region-composition bars;
- tissue-mixing entropy;
- normalized entropy;
- dominant-region composition.

Do not devote excessive presentation time to this unless it directly supports the narrative.

---

# 7. Cohort-Level AD Rewiring

## 7.1 AD rewiring is heterogeneous

Use the global/all-module MDC view.

Goal:

> establish that AD is not characterized by a simple global collapse of network connectivity.

Show:

- modules with MDC > 1;
- modules with MDC < 1;
- CT versus TS behavior.

Preferred slide title:

> **AD is characterized by both molecular decoupling and hypercoordination**

Avoid generic titles such as “MDC results”.

## 7.2 CT can behave differently from TS

Select representative modules in which CT changes strongly while TS is weaker, preserved, or changes in another direction.

Candidates from existing work may include Alzheimer-related, synaptic-vesicle, oxidative-phosphorylation, and axon-guidance modules. Use the final existing result tables rather than relying on hard-coded examples when better ones exist.

## 7.3 Region-pair specificity

Use resolved MDC for:

- AC–DLPFC;
- AC–PCG;
- DLPFC–PCG.

Goal:

> demonstrate that disease rewiring is often anatomically specific.

Preferred conclusion:

> **The anatomical unit of molecular rewiring is often the region pair, not the pathway alone.**

Use resolved MDC heatmaps, selected-module resolved bars, and pair-specific significance annotations.

## 7.4 Pathway × region-pair atlas

Create or reuse a pathway-level MDC heatmap.

Prioritize pathways such as:

- Alzheimer disease;
- Synaptic vesicle cycle;
- Oxidative phosphorylation;
- Glutamatergic synapse;
- Neuroactive ligand-receptor signaling;
- Axon guidance;
- relevant metabolic programs.

This should be one of the major manuscript-style figures. Its purpose is to move the story from module IDs to interpretable biological programs.

---

# 8. Healthy-Reference Disruption

Use the Control-derived module set.

The main question:

> Which molecular programs are normally coordinated across regions in healthy brains, and how is that organization altered in disease?

Show selected healthy-reference modules with strong CT changes.

Candidate biological themes:

- Alzheimer/neurodegeneration;
- synaptic vesicle;
- glutamatergic signaling;
- oxidative phosphorylation.

Use CT/TS MDC, resolved region-pair MDC, and KEGG context.

Preferred conclusion:

> **AD selectively disrupts normal cross-regional coordination of specific molecular programs.**

---

# 9. Edge-Level Evidence

Use edge-level analyses only when they strengthen the module-level conclusion.

Potential views:

- AD-Control edge volcano;
- Hedges’ g;
- mean difference;
- Global BH;
- Per-module BH;
- held-out edge statistics where already available;
- tissue/pair-coded edge shapes/colors.

Use one or two exemplary modules rather than showing many volcano plots.

Purpose:

> show that module-level changes can be traced back to specific gene-gene relationships.

Do not imply causal interaction from coexpression edges.

---

# 10. Individual-Level Networks

Explain the conceptual role of LIONESS.

Suggested visual flow:

Group/reference network → leave-one-out / add-one perturbation → donor-specific edge values → module-level donor score.

Show how donor-level scores can be summarized as:

- pooled CT;
- pooled TS;
- AC;
- DLPFC;
- PCG;
- AC–DLPFC;
- AC–PCG;
- DLPFC–PCG;
- resolved CT;
- resolved TS.

BONOBO can be shown as an orthogonal/sensitivity individual-network estimator, while LIONESS remains the main donor-level framework.

---

# 11. BRANCH 1 — AD Disease State / AD-vs-Control Prediction

This should be the strongest branch.

Scientific question:

> **Can an individual donor’s cross-regional molecular coordination distinguish Alzheimer’s disease from healthy aging, and which biological programs carry that information?**

## 11.1 Donor-level distributions

Use strong stable modules from the final validated targeted analysis.

At minimum create violin + jitter/box plots for:

- Control;
- MCI;
- AD.

Use MCI even though the primary classifier may be AD versus Control.

For key modules, show **CT vs TS side-by-side** so the audience can visually assess whether CT separates diagnosis more clearly than TS.

Potential candidate modules from current analyses include M732, M905, M682, M1130, M902, and M728. Do not hard-code these if the corrected final ranking changes.

## 11.2 Additional useful donor-level plots

Where useful, derive from existing data:

- univariate ROC curves;
- univariate AUC;
- ridge/density plots;
- effect-size annotations;
- Control → MCI → AD trend plots;
- resolved region-pair violins;
- scatter/association plots.

Only include a plot if it adds a distinct scientific point.

## 11.3 Prediction framework

Explain the fully nested 5×5 design.

Show:

Outer training → LIONESS reference → stability ranking → redundancy pruning → K selection → preprocessing → Elastic Net → untouched outer test.

Key message:

> model selection and performance estimation are separated.

Clearly state that this remains internal validation conditional on upstream module definitions, not external cohort validation.

## 11.4 Main prediction comparisons

Show existing results for:

- Targeted CT + covariates vs covariates alone;
- Targeted CT vs Targeted TS;
- Targeted CT vs All-module CT;
- CT pooled vs resolved CT;
- individual region-pair performance;
- fold robustness;
- OOF classification diagnostics.

Use the corrected transformation provenance.

## 11.5 Which modules drive AD prediction?

Use:

- outer-fold selection frequency;
- stable rank;
- median incremental score;
- K distribution;
- standardized coefficients;
- KEGG annotations;
- CT MDC;
- TS MDC;
- resolved MDC.

Create a manuscript-style integrated table/figure:

| Module | Selection frequency | CT MDC | TS MDC | Dominant pair | KEGG | Coefficient stability |
|---|---:|---:|---:|---|---|---|

Use this to connect donor-level prediction back to cohort-level rewiring.

## 11.6 Key hypothesis to test visually

Investigate whether the strongest predictors exhibit:

> **selective loss of cross-regional coordination despite relatively preserved within-region organization.**

A useful derived figure is paired CT/TS MDC bars for selected modules, annotated with selection frequency and KEGG.

---

# 12. BRANCH 2 — Cognitive Trajectories

Analyze separately:

- `cogng_demog_slope`;
- `cogng_path_slope`.

Where useful also include global cognition and CogDx.

Questions:

1. Does CT coordination predict cognitive decline beyond covariates?
2. Is CT stronger than TS?
3. Are particular region pairs more informative?
4. Do AD-selected modules transfer to cognition?
5. Which outcome-specific modules recur?
6. Does pathology-adjusted decline provide evidence for resilience?

## 12.1 Performance summary

For each cognitive outcome show:

- network + covariates vs covariates;
- CT vs TS;
- pooled vs resolved;
- region-pair patterns where relevant.

Use OOF performance, paired comparisons, and fold robustness.

## 12.2 Regression diagnostics

Where already available:

- observed vs predicted;
- residual/error plot.

Do not overinterpret weak R².

## 12.3 Outcome-specific module selection

Show selection frequency, stable rank, KEGG context, and overlap with AD-selected modules.

Candidate recurring themes to inspect:

- synaptic;
- mitochondrial/OXPHOS;
- glutamatergic;
- axon guidance;
- Alzheimer/neurodegeneration.

## 12.4 Cognitive interpretation

If overall predictive performance is weak, do not hide it.

Use the negative result to refine the model:

> cross-regional network organization may carry stronger information about disease state / accumulated disease burden than about the rate of longitudinal cognitive progression.

For `cogng_path_slope`, do not claim resilience unless the existing data support incremental predictive or association evidence.

If no evidence exists, explicitly state:

> **Current donor-level network features do not yet provide robust evidence for pathology-adjusted cognitive resilience.**

---

# 13. BRANCH 3 — Motor and Parkinsonian Phenotypes

Analyze:

- `motor10_demog_slope`;
- `sqrt_parksc_demog_slope`;
- `parkinsonism`.

Keep binary Parkinsonism and longitudinal motor slopes as distinct outcomes.

## 13.1 Parkinsonism

Show:

- absolute OOF performance;
- CT + covariates vs covariates;
- CT vs TS;
- CT pooled vs resolved;
- individual region pairs;
- ROC/confusion/calibration diagnostics;
- consensus module selection;
- coefficients.

For the strongest Parkinsonism module(s), perform the same presentation-level deep dive used in the AD branch:

- violin distributions;
- CT vs TS;
- MDC;
- region-pair MDC;
- KEGG;
- selection frequency;
- coefficient stability.

Candidate modules such as M729 may be relevant if they remain stable in the final validated tables.

## 13.2 Motor slopes

Show weak or negative findings transparently. Where predictive performance is not convincing, still inspect whether stable biological themes recur. Avoid converting pathway recurrence into a positive prediction claim.

## 13.3 AD–motor overlap

Create a cross-branch comparison of:

- AD-driving modules;
- Parkinsonism-driving modules;
- motor-slope modules.

Ask which modules/pathways overlap, which are distinct, and whether synaptic and mitochondrial themes recur.

## 13.4 Important future control

If not already available, create a **Proposed analysis — not yet performed** slide/section testing whether Parkinsonism prediction remains after adding:

- AD diagnosis / CogDx;
- Lewy-body pathology;
- nigral pathology;

where available.

Purpose:

> distinguish a genuinely motor/Parkinsonian coordination signal from a signal that simply tracks AD disease status.

Do not run this analysis as part of the current task.

---

# 14. Biological Interpretation Using Existing KEGG Results

Do not rerun enrichment.

Use the existing KEGG tables to generate biologically plausible hypotheses.

For every theme distinguish:

### A. Direct result
What the network/prediction data show.

### B. KEGG support
Which biological program the module is enriched for.

### C. Biological hypothesis
What mechanism could plausibly explain the pattern.

### D. Required validation
What would be needed to test that mechanism.

---

# 15. Biological Theme 1 — Synaptic Vesicle / Neurotransmission

Inspect modules enriched for:

- Synaptic vesicle cycle;
- Neuroactive ligand-receptor interaction;
- Glutamatergic synapse;
- related neuronal signaling.

Working hypothesis:

> AD may disrupt the coordinated regulation of synaptic machinery across brain regions even when local within-region organization remains relatively preserved.

Compare recurrence across AD diagnosis, cognition, Parkinsonism, and motor outcomes.

Important: pathway-level recurrence does **not** imply that the same genes or edges drive every phenotype.

---

# 16. Biological Theme 2 — Mitochondrial / Energy Metabolism

Inspect:

- Oxidative phosphorylation;
- Parkinson disease KEGG;
- thermogenesis;
- mitochondrial/energy pathways.

Working hypothesis:

> distributed neural systems may lose coordinated energetic regulation across regions in neurodegeneration.

Potential connection to synaptic biology:

- synaptic activity is energetically expensive;
- synaptic and mitochondrial coordination may form related but separable vulnerability axes.

Do not claim causality between mitochondrial changes and synaptic changes from cross-sectional coexpression data.

---

# 17. Biological Theme 3 — Alzheimer / Proteostasis / Neurodegeneration

Inspect:

- Alzheimer disease;
- proteasome;
- protein processing;
- other neurodegenerative pathways.

Ask whether the existing results indicate loss of healthy coordination, emergence of disease-associated coordination, or both in different modules.

Full-cohort and Control-derived modules can be used to frame these as complementary phenomena.

---

# 18. Biological Theme 4 — Glutamatergic / Excitatory Programs

Where supported:

> altered cross-regional glutamatergic coordination may indicate dysregulation of distributed excitatory molecular programs.

Do not infer firing rates, functional connectivity, or direct anatomical communication from transcriptomic coexpression alone.

---

# 19. Biological Theme 5 — Axon Guidance / Structural Programs

Where axon-guidance enrichment appears, use the working hypothesis:

> altered coordination may reflect dysregulation of structural maintenance, plasticity, or connectivity-related gene programs.

Do not interpret this as direct evidence of physical axonal degeneration between the sampled regions.

---

# 20. Biological Theme 6 — Immune / Inflammatory Programs

Inspect recurrence of:

- cytokine signaling;
- TNF;
- IL-17;
- NOD-like receptor;
- complement;
- infection-labeled KEGG pathways.

Important: many KEGG infection pathways reflect shared immune/inflammatory genes. Do **not** interpret an enrichment such as “viral infection” as evidence of actual infection.

Possible hypothesis:

> AD may involve coordinated neuroimmune remodeling across regions.

---

# 21. Biological Theme 7 — Lipid / Peroxisomal / Redox Programs

Where supported, connect:

- lipid metabolism;
- peroxisome;
- oxidative stress;
- redox balance;
- mitochondrial function;
- neuronal membrane/synaptic biology.

Keep this at the level of hypothesis generation.

---

# 22. Cross-Phenotype Integration

Create one major integrated figure.

## 22.1 Module/pathway × phenotype matrix

Rows: modules and/or biological pathways.

Columns:

- AD diagnosis;
- CogDx/global cognition;
- cognitive slope;
- pathology-adjusted cognitive slope;
- Parkinsonism;
- motor slope;
- Parkinsonian-score slope.

Cells can encode existing selection frequency, rank, coefficient magnitude, effect direction, or qualitative evidence category.

## 22.2 Pathway × evidence matrix

Suggested rows:

- Synaptic vesicle;
- Oxidative phosphorylation / mitochondrial;
- Alzheimer/neurodegeneration;
- Glutamatergic signaling;
- Neuroactive signaling;
- Axon guidance;
- Immune/inflammatory;
- Lipid/peroxisomal/redox;
- other strongly recurrent pathways.

Suggested columns:

- Full-cohort MDC;
- Healthy-reference MDC;
- CT specificity;
- Region-pair specificity;
- AD individual-level selection;
- Cognitive evidence;
- Parkinsonism/motor evidence.

Do not create new significance tests for this matrix. It is a synthesis of existing evidence.

---

# 23. Proposed Main Presentation Structure

Target approximately 18–24 main slides.

## Slide 1
### Cross-regional molecular coordination in Alzheimer’s disease
Visual: three-region network framework.

## Slide 2
### Regional molecular activity may miss disease-related changes in coordination
Visual: local/TS vs cross-region/CT conceptual diagram.

## Slide 3
### A multi-scale framework links cohort rewiring to donor-level phenotypes
Visual: ROSMAP → modules → MDC → LIONESS/BONOBO → phenotype/prediction.

## Slide 4
### Healthy-reference and full-cohort modules capture complementary disease phenomena
Visual: module-definition schematic + limited composition summary.

## Slide 5
### AD is characterized by both molecular decoupling and hypercoordination
Visual: global MDC overview.

## Slide 6
### Cross-regional and within-region network organization can diverge
Visual: selected module CT vs TS MDC.

## Slide 7
### Molecular rewiring depends on the specific pair of brain regions
Visual: resolved MDC heatmap.

## Slide 8
### Distinct biological pathways are rewired in distinct anatomical connections
Visual: pathway × region-pair MDC heatmap.

## Slide 9
### AD disrupts molecular programs defined by healthy cross-regional architecture
Visual: Control-derived MDC heatmap.

## Slide 10
### Synaptic and metabolic programs can lose CT coordination while TS organization remains relatively preserved
Visual: paired CT/TS MDC bars for selected modules.

## Slide 11
### LIONESS converts cohort network architecture into donor-level coordination scores
Visual: LIONESS conceptual diagram.

### Branch 1

## Slide 12
### Individual CT scores separate Control, MCI, and AD in selected modules
Visual: Control/MCI/AD violin plots.

## Slide 13
### Cross-regional coordination contains individual-level diagnostic information
Visual: nested CV design + performance summary.

## Slide 14
### CT adds diagnostic information beyond demographics/APOE and can outperform TS
Visual: primary paired-comparison forest plot.

## Slide 15
### Stable predictive modules converge on interpretable neuronal and metabolic programs
Visual: selection-frequency plot + KEGG.

## Slide 16
### The strongest predictors link donor-level discrimination to cohort-level CT disruption
Visual: integrated module × MDC × KEGG figure.

### Branch 2

## Slide 17
### Cognitive progression shows a weaker network signal than diagnostic disease state
Visual: cognitive performance forest/heatmap.

## Slide 18
### Cognitive outcome rankings still highlight recurring biological programs
Visual: selection-frequency / pathway summary.

### Branch 3

## Slide 19
### Cross-regional coordination also carries information about Parkinsonism
Visual: Parkinsonism performance comparison.

## Slide 20
### Parkinsonian phenotypes highlight synaptic and energetic coordination programs
Visual: top Parkinsonism module deep dive.

## Slide 21
### Longitudinal motor slopes are weaker than binary Parkinsonian disease state
Visual: motor outcome summary.

### Integration

## Slide 22
### Synaptic and mitochondrial coordination recur across disease and clinical phenotypes
Visual: cross-phenotype pathway matrix.

## Slide 23
### Integrated model: AD reorganizes distributed molecular programs across the brain
Visual: conceptual biological model.

## Slide 24
### Current evidence, limitations, and next validation steps
Two columns: Supported now / Not yet supported.

---

# 24. Plots to Prioritize From the Existing App

Where possible, regenerate these from stored tables rather than using screenshots.

Useful existing plot families include:

- MDC overview;
- module-level MDC;
- resolved MDC heatmap;
- pathway MDC heatmap;
- module entropy;
- module region composition;
- module size distribution;
- donor distributions;
- phenotype associations;
- correlation heatmaps;
- Module Finder;
- edge volcano;
- edge summary;
- targeted OOF performance;
- CT vs TS comparison;
- targeted primary comparison;
- selection frequency;
- fold robustness;
- prediction coefficient plot;
- ROC;
- PR;
- calibration;
- confusion matrix;
- threshold diagnostics;
- observed vs predicted regression plot;
- residual/error diagnostics.

---

# 25. New Figures Allowed Without New Analysis

The following are strongly encouraged if the underlying data already exist:

1. integrated module evidence table: selection frequency + CT MDC + TS MDC + region pair + KEGG;
2. CT-vs-TS paired module plot for selected predictive modules;
3. cross-phenotype module-selection heatmap;
4. cross-phenotype pathway matrix;
5. full-cohort vs healthy-reference biological-theme comparison;
6. Control → MCI → AD donor distribution panels;
7. prediction-module × region-pair heatmap;
8. evidence hierarchy graphic: cohort rewiring → healthy-reference disruption → individual signal → phenotype relevance.

These are descriptive syntheses of existing results, not new inferential analyses.

---

# 26. Statistical Reporting Rules

For every major reported result state, when available:

- module definition;
- network method;
- estimator;
- feature;
- CT/TS/component;
- score transformation;
- edge set;
- score normalization;
- sample size;
- effect/metric;
- confidence interval;
- nominal p-value;
- relevant FDR;
- evidence tier.

Do not mix Global BH, Per-module BH, primary-family FDR, or exploratory within-outcome FDR without explicit labels.

---

# 27. Prediction Reporting Rules

Use fold-specific panels for performance estimation.

The consensus panel is **interpretation/display only**.

Do not treat consensus-selected modules as if the same fixed set generated the OOF performance unless that is explicitly what the stored analysis did.

Clearly distinguish:

- Primary;
- Secondary;
- Exploratory;
- Sensitivity.

Do not describe internal nested CV as external validation.

---

# 28. Biological Claim Language

Use appropriate evidence-level wording.

Preferred:

- “consistent with”;
- “suggests”;
- “supports the hypothesis that”;
- “is enriched for genes involved in”;
- “cross-regional coordination of this program is altered”;
- “is associated with”;
- “carries predictive information”.

Avoid unsupported statements such as:

- “AD causes mitochondrial failure between regions”;
- “these regions communicate through this pathway”;
- “synaptic dysfunction causes cognitive decline”;
- “infection drives this module”;
- “this transcriptomic network represents anatomical connectivity”.

---

# 29. Explicit Limitations to Preserve

The final presentation should mention:

- bulk transcriptomic coexpression does not demonstrate direct communication;
- no causal inference from cross-sectional network rewiring;
- module definitions are upstream/fixed;
- nested CV is internal validation;
- external multi-region replication is still required;
- cell-type composition remains an important sensitivity/validation issue;
- diagnosis and CogDx are related outcomes, not independent replications;
- pathway enrichment is interpretive;
- module-level recurrence does not imply identical genes/edges across phenotypes;
- some cognitive and motor longitudinal outcomes may be weak or negative;
- transformation provenance must be correct.

---

# 30. Highest-Priority Future Analyses — Present as Proposed Only

Do not run these now.

Potential future validation slides/notes:

1. cell-composition-corrected network reconstruction/sensitivity;
2. held-out Control-derived module validation;
3. ROS vs MAP replication where possible;
4. external multi-region AD cohort replication;
5. orthogonal proteomic/metabolomic validation;
6. pathology-adjusted Parkinsonism analysis;
7. stronger cognitive resilience analysis;
8. direct pathway × pair testing if not already sufficiently covered;
9. LIONESS–BONOBO convergence;
10. edge-level replication/stability.

---

# 31. Jupyter Notebook Structure

Recommended notebook sections:

```text
00. Title and scientific question
01. Data provenance and available result files
02. Cohort and module definitions
03. Module composition and CT/TS architecture
04. Full-cohort MDC
05. Resolved region-pair MDC
06. Pathway × region-pair synthesis
07. Control-derived healthy-reference disruption
08. Edge-level examples
09. LIONESS/BONOBO donor-level framework
10. AD branch — distributions
11. AD branch — prediction
12. AD branch — module drivers
13. Cognitive branch
14. Motor/Parkinsonian branch
15. Cross-phenotype module overlap
16. Biological pathway synthesis
17. Integrated model
18. Limitations
19. Proposed analyses
20. Main manuscript figure plan
21. Slide-generation assets
```

At the end of each section include:

### Result
One or two sentences describing what the existing data show.

### Interpretation
Scientific interpretation.

### Caveat
Main limitation.

### Manuscript relevance
How it contributes to the proposed paper.

---

# 32. Figure Export Requirements

For each major figure:

- export PNG at high resolution;
- preferably also SVG and/or PDF for publication;
- use clear English labels;
- use DLPFC consistently;
- preserve the same directionality conventions throughout;
- include n where appropriate;
- label transformation explicitly;
- avoid tiny text;
- avoid excessive legends;
- keep colors consistent across the deck.

Suggested directory structure:

```text
manuscript_story/
├── notebook/
│   ├── ROSMAP_LIONESS_story.ipynb
│   └── ROSMAP_LIONESS_story.html
├── figures/
│   ├── main/
│   ├── supplementary/
│   └── conceptual/
├── tables/
│   ├── integrated_module_evidence.tsv
│   ├── pathway_evidence_matrix.tsv
│   └── phenotype_module_overlap.tsv
├── presentation/
│   └── ROSMAP_LIONESS_manuscript_story.pptx
└── README.md
```

---

# 33. PowerPoint Design Rules

Use:

- 16:9 layout;
- large readable titles;
- conclusion-style slide titles;
- minimal prose;
- one central visual message per slide;
- consistent typography;
- consistent CT/TS terminology;
- consistent Control/MCI/AD group order.

Avoid:

- screenshots when direct data plots are available;
- dense tables in the main deck;
- long methodological paragraphs;
- unexplained module IDs;
- mixing statistical and biological conclusions on the same visual without hierarchy.

Use module IDs together with biological annotations, for example:

> **M905 — Synaptic vesicle cycle**

rather than only “M905”.

---

# 34. Speaker Notes

For every main slide provide:

### What to say
2–5 concise sentences.

### Main result
The result that should remain with the audience.

### Caveat
The most important limitation.

### Transition
One sentence connecting to the next slide.

---

# 35. Final Manuscript Synthesis

The final presentation should test whether the existing results support a manuscript story approximately of the form:

> **Alzheimer’s disease is associated with pathway- and region-pair-specific reorganization of molecular coordination across the brain. Healthy-reference analyses suggest selective loss of normal cross-regional architecture in synaptic, energetic, and neurodegenerative programs, while full-cohort analyses also identify disease-associated hypercoordination. These network alterations can be quantified at the individual level, where cross-regional coordination carries substantial information about AD disease state and selected Parkinsonian phenotypes, while longitudinal cognitive and motor progression signals appear weaker.**

This is a working synthesis. Do not force the data to match it. If the final existing results support a different or more precise narrative, revise the story accordingly.

---

# 36. Final Outputs to Include

The completed package should end with:

- proposed manuscript title options;
- one-paragraph manuscript story;
- proposed main figures, ideally Figure 1–6;
- strongest current claims;
- claims currently unsupported;
- most important limitations;
- highest-priority validation steps;
- recommended next analysis only after the current presentation package is complete.

---

# 37. Final Quality-Control Checklist

Before finalizing:

- confirm every major numerical result against the stored source table;
- confirm module definition;
- confirm estimator/method;
- confirm CT/TS/component;
- confirm score transformation;
- confirm edge set;
- confirm FDR type;
- confirm group order;
- confirm sample counts;
- confirm no plot is mislabeled;
- confirm old asinh results are never labeled Raw;
- confirm no proposed analysis is presented as completed;
- confirm KEGG was not used for module selection unless explicitly documented;
- inspect every figure manually for clipping, axis direction, legends, title accuracy, and overlapping text;
- inspect every PowerPoint slide for readability at presentation scale;
- confirm notebook values, HTML values, and slide annotations agree.

---

# 38. Final Instruction to the Agent

> **Do not generate additional scientific results. Use the complete set of results that already exist to discover, organize, visualize, and biologically interpret the strongest manuscript story currently supported by the ROSMAP LIONESS/BONOBO project. Build the notebook first, use it as the source of truth for the HTML and PowerPoint, and keep a strict separation between demonstrated results, biological interpretation, and future hypotheses.**
