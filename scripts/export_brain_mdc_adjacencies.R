#!/usr/bin/env Rscript
# Export only observed, paired module networks. No permutations, preservation,
# LIONESS recomputation, or source-result modification is performed.
options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: export_brain_mdc_adjacencies.R <repository_root> <private_cache>")
root <- normalizePath(args[[1]], mustWork = TRUE)
cache <- args[[2]]
dir.create(cache, recursive = TRUE, showWarnings = FALSE)

# The historical MDC file contains executable plotting code at its bottom.
# Evaluate only these unchanged function definitions, not source() side effects.
helper_names <- c(".extract_donor", "read_tissue_expr", "aggregate_by_donor",
                  "assemble_cohort_matrix", ".apply_corr_transform",
                  ".apply_block_power", "compute_module_adjacency")
helper_env <- new.env(parent = globalenv())
ast <- parse(file.path(root, "scripts/R/MDC_signedAlt_rosmap_with_preservation.R"))
for (node in ast) {
  if (is.call(node) && identical(node[[1]], as.name("<-")) &&
      is.symbol(node[[2]]) && as.character(node[[2]]) %in% helper_names) {
    eval(node, envir = helper_env)
  }
}
if (!all(vapply(helper_names, exists, logical(1), envir = helper_env, inherits = FALSE))) {
  stop("Historical MDC helper definitions could not be resolved")
}

configs <- list(
  full_cohort = list(
    n = 154L,
    assignments = file.path(root, "data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_table_filtered_4.csv"),
    stats = file.path(root, "apps/lioness_streamlit/data/mdc_ad_vs_control_summary.tsv"),
    rds = file.path(root, "MDC_Preservation_signedAlt_AD_Control_reverse2/AD_vs_Control/AD_vs_Control_block_preservation_full.rds")
  ),
  control_derived = list(
    n = 186L,
    assignments = file.path(root, "data/proccessed/se2_signed_alt_control_filtered/se2_rosmap_control_signed-alt/speakeasy_clusters_table_level_4_filtered.csv"),
    stats = file.path(root, "apps/lioness_streamlit/data/control_derived/mdc_ad_vs_control_summary.tsv")
  )
)

ratio_scopes <- function(ad, control, tissues) {
  upper <- upper.tri(ad)
  same <- outer(tissues, tissues, "==")
  masks <- list(total = upper, ts = upper & same, ct = upper & !same)
  vapply(masks, function(mask) {
    if (!any(mask)) return(NA_real_)
    denominator <- mean(abs(control[mask]))
    if (!is.finite(denominator) || denominator < 1e-10) return(NA_real_)
    mean(abs(ad[mask])) / denominator
  }, numeric(1))
}

write_pair <- function(key, mid, ad, control, tissues, genes, expected, stats, source_kind) {
  n <- nrow(ad)
  stopifnot(identical(dim(ad), dim(control)), n >= 2L,
            length(tissues) == n, length(genes) == n,
            !anyDuplicated(genes), setequal(genes, expected),
            all(is.finite(ad)), all(is.finite(control)),
            max(abs(ad - t(ad))) < 1e-10,
            max(abs(control - t(control))) < 1e-10,
            all(diag(ad) == 0), all(diag(control) == 0),
            max(abs(ad)) <= 1 + 1e-10, max(abs(control)) <= 1 + 1e-10)
  observed <- ratio_scopes(ad, control, tissues)
  row <- stats[stats$module == mid, , drop = FALSE]
  stopifnot(nrow(row) == 1L, n == row$module_size_mapped)
  error <- 0
  for (scope in names(observed)) {
    stored <- row[[paste0("mdc_", scope)]]
    stopifnot(identical(is.na(observed[[scope]]), is.na(stored)))
    if (is.finite(stored)) {
      difference <- abs(observed[[scope]] - stored)
      if (difference > 1e-5 * max(1, abs(stored))) {
        stop(sprintf("%s M%d %s does not reproduce source MDC: %.12g vs %.12g", key, mid, scope, observed[[scope]], stored))
      }
      error <- max(error, difference)
    }
  }
  directory <- file.path(cache, key)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  # R writes column-major doubles; Python explicitly reads order='F'. Gene
  # identifiers are validated above but are never exported to figure metadata.
  for (group in c("AD", "Control")) {
    path <- file.path(directory, sprintf("M%d__%s.f64", mid, group))
    temporary <- paste0(path, ".partial")
    writeBin(as.double(if (group == "AD") ad else control), temporary, size = 8L, endian = "little")
    stopifnot(file.rename(temporary, path))
  }
  tissue_codes <- match(tissues, c("AC", "MFBA9BA46", "PCGBA23")) - 1L
  stopifnot(!anyNA(tissue_codes))
  writeBin(as.integer(tissue_codes), file.path(directory, sprintf("M%d__tissues.i32", mid)), size = 4L, endian = "little")
  data.frame(module_set = key, module = mid, n_nodes = n,
             n_genes_ac = sum(tissue_codes == 0L), n_genes_dlpfc = sum(tissue_codes == 1L),
             n_genes_pcg = sum(tissue_codes == 2L),
             mdc_total = observed[["total"]], mdc_ts = observed[["ts"]], mdc_ct = observed[["ct"]],
             max_mdc_reproduction_error = error, adjacency_source = source_kind,
             upper_triangle = "AD", lower_triangle = "Control")
}

for (key in names(configs)) {
  config <- configs[[key]]
  assignments <- read.csv(config$assignments, check.names = FALSE)
  assignments$feature <- paste0(assignments$Tissue, "_", assignments$`Gene Symbol`)
  conflicts <- tapply(assignments$`Cluster ID`, assignments$feature, function(v) length(unique(v)))
  stopifnot(!any(conflicts > 1L))
  stats <- read.delim(config$stats, check.names = FALSE)
  modules <- sort(unique(stats$module))
  stopifnot(length(modules) == config$n, setequal(modules, unique(assignments$`Cluster ID`)))
  if (key == "full_cohort") {
    message("Loading the saved full-cohort MDC adjacencies ...")
    payload <- readRDS(config$rds)
    stopifnot(payload$parameters$beta_TS == 3, payload$parameters$beta_CT == 2,
              payload$parameters$corr_type == "signedAlt",
              setequal(as.integer(names(payload$module_adjacencies)), modules))
  } else {
    message("Recovering observed Control-derived adjacencies with the original MDC functional ...")
    paths <- function(group) setNames(vapply(c("AC", "MFBA9BA46", "PCGBA23"), function(tissue) {
      file.path(root, "data/external/ROSMAP_partitioned", paste0("ROSMAP_fixed_", group, tissue, ".csv"))
    }, character(1)), c("AC", "MFBA9BA46", "PCGBA23"))
    ad_expr <- helper_env$assemble_cohort_matrix(paths("AD"))
    control_expr <- helper_env$assemble_cohort_matrix(paths("Control"))
    common <- intersect(colnames(ad_expr), colnames(control_expr))
    ad_expr <- ad_expr[, common, drop = FALSE]
    control_expr <- control_expr[, common, drop = FALSE]
    stopifnot(nrow(ad_expr) == 517L, nrow(control_expr) == 408L)
    gene_to_module <- setNames(as.integer(assignments$`Cluster ID`), assignments$feature)
    module_by_gene <- gene_to_module[common]
  }
  records <- vector("list", length(modules))
  for (index in seq_along(modules)) {
    mid <- modules[[index]]
    if (key == "full_cohort") {
      value <- payload$module_adjacencies[[as.character(mid)]]
      ad <- value$adj_ref; control <- value$adj_target
      genes <- value$genes; tissues <- value$tissues
      stopifnot(identical(colnames(ad), colnames(control)))
      source_kind <- "Saved original MDC adjacencies"
    } else {
      positions <- which(module_by_gene == mid)
      a <- helper_env$compute_module_adjacency(ad_expr, positions, 3, 2, "signedAlt")
      c <- helper_env$compute_module_adjacency(control_expr, positions, 3, 2, "signedAlt")
      ad <- a$adj; control <- c$adj; tissues <- a$tissues; genes <- common[positions]
      source_kind <- "Observed adjacencies recovered with unchanged MDC helper"
    }
    expected <- unique(assignments$feature[assignments$`Cluster ID` == mid])
    records[[index]] <- write_pair(key, mid, ad, control, tissues, genes, expected, stats, source_kind)
    if (index %% 20L == 0L || index == length(modules)) message(sprintf("%s: %d/%d validated paired networks", key, index, length(modules)))
  }
  write.table(do.call(rbind, records), file.path(cache, key, "manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "")
  if (key == "full_cohort") rm(payload)
  gc()
}
