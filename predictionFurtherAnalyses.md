I want to extend the current leakage-reduced LIONESS prediction analysis with TARGETED MODULE PREDICTION, without replacing the existing all-module benchmark.

Main idea: the predictive signal may be concentrated in a small subset of modules and diluted when all 154/186 modules are used together. All module selection must therefore be done using DEVELOPMENT DATA ONLY; held-out donors must never influence module ranking, panel size, thresholds, or tuning.

Please plan and implement several module-selection approaches:
1. AD-rewiring panel: select stable modules that best distinguish AD vs Control in development, preferably emphasizing strong CT effects, CT>TS specificity, adequate edge support, and biological interpretability.
2. Stability-selected panel: repeatedly resample development donors and prioritize modules that are consistently selected.
3. Outcome-specific nested selection as an exploratory analysis, with selection fully inside inner CV.
4. Optional hypothesis-driven/custom panels, clearly labeled exploratory.
Keep full-cohort and control-derived modules completely separate.

Test sparse panel sizes such as top 3/5/10/15/20 versus all modules, choosing K only inside development CV. Consider pruning highly correlated/redundant modules.

For each targeted panel compare:
- targeted network vs all-module network;
- demographics/APOE + targeted network vs demographics/APOE alone;
- CT vs TS;
- CT pooled vs resolved CT pairs;
- individual pairs (AC-DLPFC, AC-PCG, DLPFC-PCG).
Use all edges as the primary analysis and differential-edge masks as sensitivity analyses.

Also add repeated-split robustness so we can see whether the same modules and tissue pairs recur across different train/test splits.

For Streamlit, extend the existing Prediction page with a “Targeted modules” mode. Make it easy to inspect:
- how and why modules were selected (development statistics, stability, CT/TS effects, KEGG);
- selected panel and K;
- held-out performance vs all modules and covariates;
- CT-vs-TS and tissue-pair comparisons;
- development vs held-out distributions for selected modules;
- model coefficients/importance;
- robustness across edge masks, module definitions, and repeated splits.

The UI should clearly distinguish DEVELOPMENT-BASED SELECTION from HELD-OUT VALIDATION, and held-out results must never be usable to automatically select modules.