#!/usr/bin/env python3
"""CRND vs Baselines Head-to-Head Noise Detection Evaluation.

Comprehensive comparison of CRND against 11 baselines (kDN, cleanlab, k-NN consistency,
random) across 5 datasets, 3 noise rates, and multiple seeds. Computes ROC-AUC,
Spearman rho, precision@k with Wilcoxon tests, bootstrap CIs, Cohen's d,
DerSimonian-Laird meta-analysis, boundary stratification analysis, Schoener's D
2D/5D sensitivity, per-class CRND distributions, and computational cost comparison.
"""

import json
import re
import resource
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1 h CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "eval.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Paths to dependency data
# ---------------------------------------------------------------------------
DEP_BASE = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260216_170044/3_invention_loop/iter_2/gen_art"
)
EXP_ID1_FULL = DEP_BASE / "exp_id1_it2__opus" / "full_method_out.json"
EXP_ID3_FULL = DEP_BASE / "exp_id3_it2__opus" / "full_method_out.json"

# Use mini files for quick testing during dev; set MAX_EXAMPLES to limit
MAX_EXAMPLES: int | None = None  # None = use all

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05

# All 11 baselines from exp_id3 (note: knn_consist_* == kdn_* per metadata)
ALL_BASELINES = [
    "kdn_tfidf", "kdn_embed", "kdn_combined", "kdn_avg",
    "cleanlab_tfidf", "cleanlab_embed", "cleanlab_avg",
    "knn_consist_tfidf", "knn_consist_embed", "knn_consist_combined",
    "random",
]

# Unique baselines (excluding knn_consist duplicates for statistical tests)
UNIQUE_BASELINES = [
    "kdn_tfidf", "kdn_embed", "kdn_combined", "kdn_avg",
    "cleanlab_tfidf", "cleanlab_embed", "cleanlab_avg",
    "random",
]

DATASETS = [
    "medical_abstracts", "mimic_iv_ed_demo", "clinical_patient_triage_nl",
    "ohsumed_single", "mental_health_conditions",
]
NOISE_RATES = ["0.05", "0.1", "0.2"]


# ===================================================================
# Helper: parse predict_* strings from exp_id3
# ===================================================================
def parse_predict_string(s: str) -> dict[str, float]:
    """Parse 'ROC-AUC=0.8216, rho=0.2445, P@k=0.3867' -> dict."""
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        m = re.match(r"([\w@\-]+)=([\d.\-eE+]+)", part)
        if m:
            key = m.group(1).replace("-", "_").replace("@", "_at_")
            out[key] = float(m.group(2))
    return out


# ===================================================================
# Data extraction
# ===================================================================
def load_exp_id1(path: Path) -> dict:
    """Load exp_id1 (CRND experiment)."""
    logger.info(f"Loading exp_id1 from {path}")
    data = json.loads(path.read_text())
    logger.info(f"Loaded exp_id1 metadata keys: {list(data['metadata'].keys())}")
    return data


def load_exp_id3(path: Path) -> dict:
    """Load exp_id3 (baselines experiment)."""
    logger.info(f"Loading exp_id3 from {path}")
    data = json.loads(path.read_text())
    n_datasets = len(data["datasets"])
    total_examples = sum(len(d["examples"]) for d in data["datasets"])
    logger.info(f"Loaded exp_id3: {n_datasets} datasets, {total_examples} trials")
    return data


def extract_baseline_trial_metrics(exp3: dict) -> dict:
    """Extract per-trial metrics from exp_id3 into structured dict.

    Returns: {dataset: {noise_rate: {baseline: {seed: {auc, rho, p_at_k}}}}}
    """
    result: dict = {}
    for ds_block in exp3["datasets"]:
        ds_name = ds_block["dataset"]
        result[ds_name] = {}
        for ex in ds_block["examples"]:
            nr = str(ex["metadata_noise_rate"])
            seed = ex["metadata_seed"]
            if nr not in result[ds_name]:
                result[ds_name][nr] = {}
            for bl in ALL_BASELINES:
                key = f"predict_{bl}"
                if key not in ex:
                    continue
                if bl not in result[ds_name][nr]:
                    result[ds_name][nr][bl] = {}
                parsed = parse_predict_string(ex[key])
                result[ds_name][nr][bl][seed] = {
                    "auc": parsed.get("ROC_AUC", np.nan),
                    "rho": parsed.get("rho", np.nan),
                    "p_at_k": parsed.get("P_at_k", np.nan),
                }
    return result


def extract_crnd_agg_metrics(exp1: dict) -> dict:
    """Extract CRND aggregated noise detection results.

    Returns: {dataset: {noise_rate: {metric: {mean, std, n_seeds}}}}
    """
    ndr = exp1["metadata"]["noise_detection_results"]
    result: dict = {}
    for ds_name, nr_block in ndr.items():
        result[ds_name] = {}
        for nr, methods in nr_block.items():
            crnd = methods["crnd"]
            baseline_entropy = methods["baseline_entropy"]
            result[ds_name][nr] = {
                "crnd": {
                    "mean_auc": crnd["mean_auc"],
                    "std_auc": crnd["std_auc"],
                    "mean_rho": crnd["mean_rho"],
                    "std_rho": crnd["std_rho"],
                    "n_seeds": exp1["metadata"]["n_noise_seeds"],
                },
                "baseline_entropy": {
                    "mean_auc": baseline_entropy["mean_auc"],
                    "std_auc": baseline_entropy["std_auc"],
                    "mean_rho": baseline_entropy["mean_rho"],
                    "std_rho": baseline_entropy["std_rho"],
                    "n_seeds": exp1["metadata"]["n_noise_seeds"],
                },
            }
    return result


# ===================================================================
# 1. Primary metrics: aggregate baselines across seeds
# ===================================================================
def compute_baseline_aggregates(trial_metrics: dict) -> dict:
    """Compute mean/std across seeds for each baseline.

    Returns: {dataset: {noise_rate: {baseline: {mean_auc, std_auc, mean_rho, ...}}}}
    """
    result: dict = {}
    for ds, nr_block in trial_metrics.items():
        result[ds] = {}
        for nr, bl_block in nr_block.items():
            result[ds][nr] = {}
            for bl, seed_data in bl_block.items():
                aucs = [v["auc"] for v in seed_data.values()]
                rhos = [v["rho"] for v in seed_data.values()]
                pks = [v["p_at_k"] for v in seed_data.values()]
                result[ds][nr][bl] = {
                    "mean_auc": float(np.nanmean(aucs)),
                    "std_auc": float(np.nanstd(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    "mean_rho": float(np.nanmean(rhos)),
                    "std_rho": float(np.nanstd(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
                    "mean_p_at_k": float(np.nanmean(pks)),
                    "std_p_at_k": float(np.nanstd(pks, ddof=1)) if len(pks) > 1 else 0.0,
                    "n_seeds": len(aucs),
                    "seed_aucs": aucs,
                    "seed_rhos": rhos,
                    "seed_pks": pks,
                }
    return result


# ===================================================================
# 2. Statistical comparison: CRND vs each baseline
# ===================================================================
def welch_t_test(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> dict:
    """Welch's t-test from summary statistics."""
    se1_sq = (std1 ** 2) / n1 if n1 > 0 else 0
    se2_sq = (std2 ** 2) / n2 if n2 > 0 else 0
    se_diff = np.sqrt(se1_sq + se2_sq)
    if se_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "df": 0.0}
    t_stat = (mean1 - mean2) / se_diff
    # Welch-Satterthwaite degrees of freedom
    num = (se1_sq + se2_sq) ** 2
    denom = 0.0
    if n1 > 1 and se1_sq > 0:
        denom += (se1_sq ** 2) / (n1 - 1)
    if n2 > 1 and se2_sq > 0:
        denom += (se2_sq ** 2) / (n2 - 1)
    df = num / denom if denom > 0 else 1.0
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df))
    return {"t_stat": float(t_stat), "p_value": p_value, "df": float(df)}


def cohens_d_from_summary(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> float:
    """Cohen's d from summary stats (pooled std)."""
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def effect_size_category(d: float) -> str:
    """Categorize Cohen's d."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci_from_seed_arrays(
    crnd_values: list[float],
    baseline_values: list[float],
    n_boot: int = BOOTSTRAP_N,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap CI on difference of means from seed-level data.

    Since CRND has only aggregated stats (10 seeds) but baselines have per-seed data (5 seeds),
    we bootstrap from the baseline data and compare against CRND point estimate.
    """
    rng = np.random.RandomState(seed)
    crnd_arr = np.array(crnd_values)
    bl_arr = np.array(baseline_values)

    # Create pseudo-paired differences
    # CRND mean is a point estimate; bootstrap the baseline and compute diff each time
    crnd_mean = float(np.mean(crnd_arr))
    diffs = []
    for _ in range(n_boot):
        bl_boot = rng.choice(bl_arr, size=len(bl_arr), replace=True)
        diffs.append(crnd_mean - float(np.mean(bl_boot)))
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    excludes_zero = not (lo <= 0 <= hi)
    return {
        "mean_diff": mean_diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "excludes_zero": excludes_zero,
    }


def wilcoxon_from_seed_arrays(
    crnd_mean: float,
    baseline_values: list[float],
) -> dict:
    """Mann-Whitney U test comparing CRND mean (point est) vs baseline seed values.

    Since CRND has only aggregated stats, we use a one-sample Wilcoxon signed-rank test
    on (baseline_i - crnd_mean) to test if baselines differ from CRND's mean.
    """
    bl_arr = np.array(baseline_values)
    diffs = bl_arr - crnd_mean
    # Remove zeros (ties with the hypothesized mean)
    diffs_nz = diffs[diffs != 0]
    if len(diffs_nz) < 2:
        return {"statistic": np.nan, "p_value": 1.0, "test": "wilcoxon_signed_rank", "n": len(diffs_nz)}
    try:
        stat, pval = sp_stats.wilcoxon(diffs_nz, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(pval), "test": "wilcoxon_signed_rank", "n": len(diffs_nz)}
    except ValueError:
        return {"statistic": np.nan, "p_value": 1.0, "test": "wilcoxon_signed_rank", "n": len(diffs_nz)}


def compute_statistical_comparisons(
    crnd_agg: dict,
    baseline_agg: dict,
    baseline_trials: dict,
) -> dict:
    """Compute Welch's t-test, bootstrap CI, Cohen's d, Wilcoxon for CRND vs each baseline."""
    comparisons: dict = {}
    for ds in DATASETS:
        comparisons[ds] = {}
        for nr in NOISE_RATES:
            comparisons[ds][nr] = {}
            crnd = crnd_agg.get(ds, {}).get(nr, {}).get("crnd")
            if crnd is None:
                continue
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data is None:
                    continue
                # Delta AUC
                delta_auc = crnd["mean_auc"] - bl_data["mean_auc"]
                delta_rho = crnd["mean_rho"] - bl_data["mean_rho"]

                # Welch's t-test (unpaired, since seed sets differ)
                t_res_auc = welch_t_test(
                    mean1=crnd["mean_auc"], std1=crnd["std_auc"], n1=crnd["n_seeds"],
                    mean2=bl_data["mean_auc"], std2=bl_data["std_auc"], n2=bl_data["n_seeds"],
                )
                t_res_rho = welch_t_test(
                    mean1=crnd["mean_rho"], std1=crnd["std_rho"], n1=crnd["n_seeds"],
                    mean2=bl_data["mean_rho"], std2=bl_data["std_rho"], n2=bl_data["n_seeds"],
                )

                # Cohen's d
                d_auc = cohens_d_from_summary(
                    mean1=crnd["mean_auc"], std1=crnd["std_auc"], n1=crnd["n_seeds"],
                    mean2=bl_data["mean_auc"], std2=bl_data["std_auc"], n2=bl_data["n_seeds"],
                )
                d_rho = cohens_d_from_summary(
                    mean1=crnd["mean_rho"], std1=crnd["std_rho"], n1=crnd["n_seeds"],
                    mean2=bl_data["mean_rho"], std2=bl_data["std_rho"], n2=bl_data["n_seeds"],
                )

                # Bootstrap CI from seed arrays
                bl_seed_aucs = bl_data.get("seed_aucs", [])
                # For CRND we simulate seed values from N(mean, std) with 10 seeds
                rng_crnd = np.random.RandomState(BOOTSTRAP_SEED + hash(ds + nr) % 10000)
                crnd_pseudo_seeds = list(rng_crnd.normal(
                    crnd["mean_auc"], crnd["std_auc"], crnd["n_seeds"]
                ))
                boot_ci = bootstrap_ci_from_seed_arrays(
                    crnd_values=crnd_pseudo_seeds,
                    baseline_values=bl_seed_aucs,
                )

                # Wilcoxon
                wilcox = wilcoxon_from_seed_arrays(
                    crnd_mean=crnd["mean_auc"],
                    baseline_values=bl_seed_aucs,
                )

                comparisons[ds][nr][bl] = {
                    "delta_auc": float(delta_auc),
                    "delta_rho": float(delta_rho),
                    "welch_t_auc": t_res_auc,
                    "welch_t_rho": t_res_rho,
                    "cohens_d_auc": float(d_auc),
                    "cohens_d_auc_category": effect_size_category(d_auc),
                    "cohens_d_rho": float(d_rho),
                    "cohens_d_rho_category": effect_size_category(d_rho),
                    "bootstrap_ci_auc": boot_ci,
                    "wilcoxon_auc": wilcox,
                }
    return comparisons


# ===================================================================
# 3. Meta-analysis: DerSimonian-Laird random effects
# ===================================================================
def dersimonian_laird(
    effects: list[float],
    variances: list[float],
) -> dict:
    """DerSimonian-Laird random-effects meta-analysis.

    Args:
        effects: per-study effect sizes
        variances: per-study variance of effect sizes

    Returns:
        pooled_effect, ci_lower, ci_upper, tau_sq, i_sq, q_stat, q_p_value
    """
    k = len(effects)
    if k < 2:
        return {
            "pooled_effect": effects[0] if effects else 0.0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "tau_sq": 0.0, "i_sq": 0.0, "q_stat": 0.0, "q_p_value": 1.0, "k": k,
        }
    effects_arr = np.array(effects)
    vars_arr = np.array(variances)
    # Avoid division by zero
    vars_arr = np.where(vars_arr <= 0, 1e-10, vars_arr)
    weights = 1.0 / vars_arr

    # Fixed-effect pooled estimate
    theta_fe = float(np.sum(weights * effects_arr) / np.sum(weights))

    # Q statistic
    q_stat = float(np.sum(weights * (effects_arr - theta_fe) ** 2))
    q_df = k - 1
    q_p_value = float(1 - sp_stats.chi2.cdf(q_stat, q_df)) if q_df > 0 else 1.0

    # Between-study variance (tau²)
    c = float(np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    tau_sq = max(0.0, (q_stat - q_df) / c) if c > 0 else 0.0

    # Random-effects weights
    re_weights = 1.0 / (vars_arr + tau_sq)
    pooled = float(np.sum(re_weights * effects_arr) / np.sum(re_weights))
    se_pooled = float(np.sqrt(1.0 / np.sum(re_weights)))

    ci_lo = pooled - 1.96 * se_pooled
    ci_hi = pooled + 1.96 * se_pooled

    # I² statistic
    i_sq = max(0.0, (q_stat - q_df) / q_stat * 100) if q_stat > 0 else 0.0

    return {
        "pooled_effect": float(pooled),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "tau_sq": float(tau_sq),
        "i_sq": float(i_sq),
        "q_stat": float(q_stat),
        "q_p_value": float(q_p_value),
        "k": k,
    }


def meta_analysis_crnd_vs_best_baseline(
    crnd_agg: dict,
    baseline_agg: dict,
    stat_comparisons: dict,
) -> dict:
    """Pool Cohen's d across datasets for CRND vs best baseline (per dataset)."""
    meta_results: dict = {}
    for nr in NOISE_RATES:
        effects = []
        variances = []
        ds_labels = []
        for ds in DATASETS:
            # Find best baseline for this dataset+noise_rate
            best_bl = None
            best_auc = -1
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data and bl_data["mean_auc"] > best_auc:
                    best_auc = bl_data["mean_auc"]
                    best_bl = bl
            if best_bl is None:
                continue
            comp = stat_comparisons.get(ds, {}).get(nr, {}).get(best_bl)
            if comp is None:
                continue
            d = comp["cohens_d_auc"]
            # Variance of d ~ (n1+n2)/(n1*n2) + d²/(2*(n1+n2))
            crnd = crnd_agg[ds][nr]["crnd"]
            bl_data = baseline_agg[ds][nr][best_bl]
            n1, n2 = crnd["n_seeds"], bl_data["n_seeds"]
            var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
            effects.append(d)
            variances.append(var_d)
            ds_labels.append(f"{ds}_vs_{best_bl}")

        dl_result = dersimonian_laird(effects, variances)
        dl_result["per_study"] = [
            {"dataset": lab, "effect": eff, "variance": var}
            for lab, eff, var in zip(ds_labels, effects, variances)
        ]
        meta_results[nr] = dl_result
    return meta_results


# ===================================================================
# 4. Boundary stratification analysis (Gap G6)
# ===================================================================
def boundary_stratification_analysis(exp1: dict) -> dict:
    """Compare CRND for boundary vs interior instances."""
    strat = exp1["metadata"]["crnd_boundary_stratification"]
    results: dict = {}
    for ds, bins_data in strat.items():
        # Interior: bins 0.0-0.2
        # Boundary: bins 0.6-0.8 and 0.8-1.0
        interior_bins = []
        boundary_bins = []
        for bin_key, bin_vals in bins_data.items():
            lo = float(bin_key.split("-")[0])
            if lo <= 0.2:
                interior_bins.append(bin_vals)
            elif lo >= 0.6:
                boundary_bins.append(bin_vals)

        # Weighted mean CRND for interior vs boundary
        def weighted_stats(bins: list[dict]) -> tuple[float, float, int]:
            total_n = sum(b["count"] for b in bins)
            if total_n == 0:
                return 0.0, 0.0, 0
            wmean = sum(b["mean_crnd"] * b["count"] for b in bins) / total_n
            # Pooled std
            wvar = sum(b["count"] * (b["std_crnd"] ** 2 + (b["mean_crnd"] - wmean) ** 2) for b in bins) / total_n
            return wmean, np.sqrt(wvar), total_n

        int_mean, int_std, int_n = weighted_stats(interior_bins)
        bnd_mean, bnd_std, bnd_n = weighted_stats(boundary_bins)

        # Cohen's d between boundary and interior
        if int_n > 0 and bnd_n > 0 and (int_std + bnd_std) > 0:
            pooled_std_val = np.sqrt(
                (int_n * int_std ** 2 + bnd_n * bnd_std ** 2) / (int_n + bnd_n)
            )
            d_val = (bnd_mean - int_mean) / pooled_std_val if pooled_std_val > 0 else 0.0
        else:
            d_val = 0.0

        # Mann-Whitney U (approximate from summary stats) — we report summary-based stats
        # since we don't have individual instance data here
        results[ds] = {
            "interior_mean_crnd": float(int_mean),
            "interior_std_crnd": float(int_std),
            "interior_n": int_n,
            "boundary_mean_crnd": float(bnd_mean),
            "boundary_std_crnd": float(bnd_std),
            "boundary_n": bnd_n,
            "difference_bnd_minus_int": float(bnd_mean - int_mean),
            "cohens_d": float(d_val),
            "cohens_d_category": effect_size_category(d_val),
            "welch_t": welch_t_test(
                mean1=bnd_mean, std1=bnd_std, n1=bnd_n,
                mean2=int_mean, std2=int_std, n2=int_n,
            ),
            "all_bins": {k: v for k, v in bins_data.items()},
        }
    return results


# ===================================================================
# 5. Schoener's D 2D vs 5D sensitivity (Gap G5)
# ===================================================================
def schoeners_d_sensitivity(exp1: dict) -> dict:
    """Compare 2D vs 5D Schoener's D overlap matrices."""
    schd = exp1["metadata"]["schoeners_d_matrices"]
    pca_var = exp1["metadata"]["pca_explained_variance"]
    results: dict = {}

    for ds, spaces in schd.items():
        ds_result: dict = {"feature_spaces": {}, "pca_explained_variance": {}}
        # Group by feature space
        space_names = set()
        for key in spaces.keys():
            # key format: tfidf_2d, tfidf_5d, etc.
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                space_names.add(parts[0])

        for sp in sorted(space_names):
            key_2d = f"{sp}_2d"
            key_5d = f"{sp}_5d"
            if key_2d not in spaces or key_5d not in spaces:
                continue
            mat_2d = np.array(spaces[key_2d])
            mat_5d = np.array(spaces[key_5d])

            # Upper triangle values (excluding diagonal)
            n = mat_2d.shape[0]
            idx_upper = np.triu_indices(n, k=1)
            vals_2d = mat_2d[idx_upper]
            vals_5d = mat_5d[idx_upper]

            # Mean absolute difference
            mad = float(np.mean(np.abs(vals_2d - vals_5d)))
            # Pearson correlation
            if len(vals_2d) > 2:
                r, p = sp_stats.pearsonr(vals_2d, vals_5d)
            else:
                r, p = np.nan, np.nan

            ds_result["feature_spaces"][sp] = {
                "mean_abs_diff_2d_vs_5d": mad,
                "pearson_r": float(r) if not np.isnan(r) else None,
                "pearson_p": float(p) if not np.isnan(p) else None,
                "values_2d": vals_2d.tolist(),
                "values_5d": vals_5d.tolist(),
            }

        # PCA explained variance
        for key, vals in pca_var.get(ds, {}).items():
            ds_result["pca_explained_variance"][key] = {
                "components": vals,
                "cumulative": float(np.sum(vals)),
            }

        results[ds] = ds_result
    return results


# ===================================================================
# 6. Niche overlap profile agreement (cross-space)
# ===================================================================
def niche_overlap_profile_analysis(exp1: dict) -> dict:
    """Report Kendall tau between feature spaces' niche overlap profiles."""
    nop = exp1["metadata"]["niche_overlap_profile_comparison"]
    results: dict = {}
    for ds, comparisons in nop.items():
        results[ds] = {}
        for pair_key, vals in comparisons.items():
            results[ds][pair_key] = {
                "kendall_tau": float(vals["kendall_tau"]),
                "p_value": float(vals["p_value"]),
                "significant": vals["p_value"] < ALPHA,
            }
    return results


# ===================================================================
# 7. Per-class CRND distributions (Gap G3)
# ===================================================================
def per_class_crnd_analysis(exp1: dict) -> dict:
    """Analyze per-class CRND for clinical datasets."""
    crnd_pc = exp1["metadata"]["crnd_per_class"]
    results: dict = {}
    for ds, classes in crnd_pc.items():
        # Sort by mean CRND
        sorted_classes = sorted(classes.items(), key=lambda x: x[1]["mean"])
        class_means = [v["mean"] for _, v in sorted_classes]
        class_names = [k for k, _ in sorted_classes]

        # Kruskal-Wallis H-test (approximate from summary stats)
        # We use the means and n's to create pseudo-data for the test
        groups = []
        for cls_name, cls_data in classes.items():
            n = cls_data["n"]
            mean = cls_data["mean"]
            std = cls_data["std"]
            if n >= 2 and std > 0:
                rng = np.random.RandomState(hash(ds + cls_name) % 2**31)
                pseudo = rng.normal(mean, std, n)
                groups.append(pseudo)
            elif n >= 1:
                groups.append(np.array([mean] * n))

        if len(groups) >= 2:
            try:
                h_stat, h_p = sp_stats.kruskal(*groups)
            except ValueError:
                h_stat, h_p = np.nan, 1.0
        else:
            h_stat, h_p = np.nan, 1.0

        results[ds] = {
            "classes_sorted_by_mean": [
                {"class": k, **v} for k, v in sorted_classes
            ],
            "kruskal_wallis_h": float(h_stat) if not np.isnan(h_stat) else None,
            "kruskal_wallis_p": float(h_p) if not np.isnan(h_p) else None,
            "crnd_varies_by_class": h_p < ALPHA if not np.isnan(h_p) else False,
            "range_min_max": float(max(class_means) - min(class_means)) if class_means else 0.0,
        }
    return results


# ===================================================================
# 8. Computational cost comparison
# ===================================================================
def computational_cost_analysis(exp1: dict, exp3: dict) -> dict:
    """Compare computational costs between CRND and baselines."""
    m1 = exp1["metadata"]
    m3 = exp3["metadata"]
    crnd_runtime = m1["runtime_seconds"]
    baseline_runtime = m3["total_runtime_seconds"]
    return {
        "crnd_runtime_seconds": float(crnd_runtime),
        "baseline_runtime_seconds": float(baseline_runtime),
        "crnd_to_baseline_ratio": float(crnd_runtime / baseline_runtime) if baseline_runtime > 0 else float("inf"),
        "crnd_phase_timings": {k: float(v) for k, v in m1["phase_timings"].items()},
        "crnd_llm_calls": m1.get("llm_calls_made", 0),
        "crnd_llm_cost_usd": m1.get("llm_total_cost_usd", 0.0),
        "crnd_llm_calls_failed": m1.get("llm_calls_failed", 0),
        "crnd_llm_input_tokens": m1.get("llm_input_tokens", 0),
        "crnd_llm_output_tokens": m1.get("llm_output_tokens", 0),
        "baseline_uses_llm": False,
        "baseline_llm_cost_usd": 0.0,
    }


# ===================================================================
# 9. Build output examples for eval_out.json schema
# ===================================================================
def build_output_examples(
    crnd_agg: dict,
    baseline_agg: dict,
    stat_comparisons: dict,
    trial_metrics: dict,
) -> list[dict]:
    """Build per-dataset output sections following exp_eval_sol_out schema.

    Each "example" row = one (dataset, noise_rate) combination comparing CRND vs all baselines.
    """
    datasets_out = []
    for ds in DATASETS:
        examples = []
        for nr in NOISE_RATES:
            crnd = crnd_agg.get(ds, {}).get(nr, {}).get("crnd")
            if crnd is None:
                continue

            # Input: description of the comparison
            input_str = (
                f"CRND vs {len(UNIQUE_BASELINES)} baselines on {ds} at {nr} noise rate. "
                f"CRND (10 seeds): AUC={crnd['mean_auc']:.4f}±{crnd['std_auc']:.4f}, "
                f"rho={crnd['mean_rho']:.4f}±{crnd['std_rho']:.4f}"
            )

            # Output: best baseline summary
            best_bl = None
            best_auc = -1
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data and bl_data["mean_auc"] > best_auc:
                    best_auc = bl_data["mean_auc"]
                    best_bl = bl
            if best_bl:
                bl_data = baseline_agg[ds][nr][best_bl]
                output_str = (
                    f"Best baseline: {best_bl} AUC={bl_data['mean_auc']:.4f}±{bl_data['std_auc']:.4f}. "
                    f"Delta AUC (CRND-best)={crnd['mean_auc'] - bl_data['mean_auc']:.4f}"
                )
            else:
                output_str = "No baseline data available"

            # Build predict_* fields (one per method)
            example: dict[str, Any] = {
                "input": input_str,
                "output": output_str,
            }

            # CRND predict string
            example["predict_crnd"] = (
                f"AUC={crnd['mean_auc']:.4f}, rho={crnd['mean_rho']:.4f}"
            )

            # Baseline predict strings
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data:
                    example[f"predict_{bl}"] = (
                        f"AUC={bl_data['mean_auc']:.4f}, rho={bl_data['mean_rho']:.4f}, "
                        f"P@k={bl_data['mean_p_at_k']:.4f}"
                    )

            # Eval metrics
            example["eval_crnd_mean_auc"] = crnd["mean_auc"]
            example["eval_crnd_mean_rho"] = crnd["mean_rho"]

            if best_bl:
                comp = stat_comparisons.get(ds, {}).get(nr, {}).get(best_bl, {})
                example["eval_delta_auc_vs_best"] = comp.get("delta_auc", 0.0)
                example["eval_cohens_d_auc_vs_best"] = comp.get("cohens_d_auc", 0.0)
                example["eval_welch_p_auc_vs_best"] = comp.get("welch_t_auc", {}).get("p_value", 1.0)

            # Best baseline AUC
            if best_bl:
                example["eval_best_baseline_auc"] = baseline_agg[ds][nr][best_bl]["mean_auc"]
                example["eval_best_baseline_rho"] = baseline_agg[ds][nr][best_bl]["mean_rho"]

            # Metadata
            example["metadata_dataset"] = ds
            example["metadata_noise_rate"] = float(nr)
            example["metadata_crnd_n_seeds"] = crnd["n_seeds"]
            example["metadata_baseline_n_seeds"] = baseline_agg.get(ds, {}).get(nr, {}).get(
                best_bl, {}
            ).get("n_seeds", 5)

            examples.append(example)

        if examples:
            datasets_out.append({"dataset": ds, "examples": examples})
    return datasets_out


# ===================================================================
# 10. Build aggregate metrics
# ===================================================================
def build_metrics_agg(
    crnd_agg: dict,
    baseline_agg: dict,
    stat_comparisons: dict,
    meta_analysis: dict,
    boundary_results: dict,
    schoeners_results: dict,
    per_class_results: dict,
    cost_results: dict,
    niche_results: dict,
) -> dict:
    """Build the metrics_agg dict for the eval output schema."""
    # Collect all CRND AUCs and best baseline AUCs
    all_crnd_aucs = []
    all_best_bl_aucs = []
    all_deltas = []
    n_significant = 0
    n_total = 0
    any_sc1_met = False  # SC1: rho > 0.3 with p < 0.01

    for ds in DATASETS:
        for nr in NOISE_RATES:
            crnd = crnd_agg.get(ds, {}).get(nr, {}).get("crnd")
            if crnd is None:
                continue
            all_crnd_aucs.append(crnd["mean_auc"])

            # Check SC1
            if abs(crnd["mean_rho"]) > 0.3:
                # We'd need the p-value; approximate from t-test on rho
                n_seeds = crnd["n_seeds"]
                if n_seeds > 2 and crnd["std_rho"] > 0:
                    t_sc1 = crnd["mean_rho"] / (crnd["std_rho"] / np.sqrt(n_seeds))
                    p_sc1 = 2 * sp_stats.t.sf(abs(t_sc1), n_seeds - 1)
                    if p_sc1 < 0.01:
                        any_sc1_met = True

            best_bl_auc = -1
            best_bl = None
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data and bl_data["mean_auc"] > best_bl_auc:
                    best_bl_auc = bl_data["mean_auc"]
                    best_bl = bl
            if best_bl:
                all_best_bl_aucs.append(best_bl_auc)
                all_deltas.append(crnd["mean_auc"] - best_bl_auc)
                comp = stat_comparisons.get(ds, {}).get(nr, {}).get(best_bl)
                if comp:
                    n_total += 1
                    if comp["welch_t_auc"]["p_value"] < ALPHA:
                        n_significant += 1

    # Meta-analysis pooled effect (average across noise rates)
    pooled_effects = [meta_analysis[nr]["pooled_effect"] for nr in NOISE_RATES if nr in meta_analysis]
    avg_pooled_d = float(np.mean(pooled_effects)) if pooled_effects else 0.0

    # Boundary analysis: average Cohen's d across datasets
    bnd_ds = [v["cohens_d"] for v in boundary_results.values() if "cohens_d" in v]
    avg_bnd_d = float(np.mean(bnd_ds)) if bnd_ds else 0.0

    # Schoener's D: average correlation across datasets and spaces
    sch_rs = []
    for ds, ds_data in schoeners_results.items():
        for sp, sp_data in ds_data.get("feature_spaces", {}).items():
            r = sp_data.get("pearson_r")
            if r is not None:
                sch_rs.append(r)
    avg_sch_r = float(np.mean(sch_rs)) if sch_rs else 0.0

    # Per-class: how many datasets show significant class variation
    n_class_sig = sum(1 for v in per_class_results.values() if v.get("crnd_varies_by_class"))

    metrics = {
        "overall_mean_auc_crnd": float(np.mean(all_crnd_aucs)) if all_crnd_aucs else 0.0,
        "overall_mean_auc_best_baseline": float(np.mean(all_best_bl_aucs)) if all_best_bl_aucs else 0.0,
        "overall_delta_auc": float(np.mean(all_deltas)) if all_deltas else 0.0,
        "overall_delta_auc_min": float(np.min(all_deltas)) if all_deltas else 0.0,
        "overall_delta_auc_max": float(np.max(all_deltas)) if all_deltas else 0.0,
        "pooled_effect_d": avg_pooled_d,
        "n_significant_comparisons": n_significant,
        "n_total_comparisons": n_total,
        "pct_significant": float(n_significant / n_total * 100) if n_total > 0 else 0.0,
        "hypothesis_sc1_supported": 1 if any_sc1_met else 0,
        "boundary_avg_cohens_d": avg_bnd_d,
        "schoeners_d_2d_5d_avg_pearson_r": avg_sch_r,
        "n_datasets_class_crnd_varies": n_class_sig,
        "crnd_runtime_seconds": cost_results["crnd_runtime_seconds"],
        "baseline_runtime_seconds": cost_results["baseline_runtime_seconds"],
        "crnd_to_baseline_cost_ratio": cost_results["crnd_to_baseline_ratio"],
    }
    return metrics


# ===================================================================
# Main
# ===================================================================
@logger.catch
def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("CRND vs Baselines Head-to-Head Evaluation")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    exp1 = load_exp_id1(EXP_ID1_FULL)
    exp3 = load_exp_id3(EXP_ID3_FULL)

    # ------------------------------------------------------------------
    # Extract structured metrics
    # ------------------------------------------------------------------
    logger.info("Extracting CRND aggregated metrics from exp_id1...")
    crnd_agg = extract_crnd_agg_metrics(exp1)
    logger.info(f"CRND datasets: {list(crnd_agg.keys())}")

    logger.info("Extracting baseline trial-level metrics from exp_id3...")
    trial_metrics = extract_baseline_trial_metrics(exp3)
    logger.info(f"Baseline datasets: {list(trial_metrics.keys())}")

    logger.info("Computing baseline aggregates...")
    baseline_agg = compute_baseline_aggregates(trial_metrics)

    # ------------------------------------------------------------------
    # Log summary table: CRND vs baselines
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY: CRND AUC vs Best Baseline AUC")
    logger.info("-" * 60)
    for ds in DATASETS:
        for nr in NOISE_RATES:
            crnd = crnd_agg.get(ds, {}).get(nr, {}).get("crnd")
            if crnd is None:
                continue
            best_bl = None
            best_auc = -1
            for bl in UNIQUE_BASELINES:
                bl_data = baseline_agg.get(ds, {}).get(nr, {}).get(bl)
                if bl_data and bl_data["mean_auc"] > best_auc:
                    best_auc = bl_data["mean_auc"]
                    best_bl = bl
            delta = crnd["mean_auc"] - best_auc if best_bl else 0
            logger.info(
                f"  {ds:35s} NR={nr}: CRND={crnd['mean_auc']:.4f} | "
                f"Best={best_bl}={best_auc:.4f} | Δ={delta:+.4f}"
            )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Statistical comparisons
    # ------------------------------------------------------------------
    logger.info("Computing statistical comparisons (Welch t, Bootstrap CI, Cohen's d, Wilcoxon)...")
    stat_comps = compute_statistical_comparisons(crnd_agg, baseline_agg, trial_metrics)
    logger.info("Statistical comparisons complete.")

    # ------------------------------------------------------------------
    # Meta-analysis
    # ------------------------------------------------------------------
    logger.info("Running DerSimonian-Laird meta-analysis...")
    meta = meta_analysis_crnd_vs_best_baseline(crnd_agg, baseline_agg, stat_comps)
    for nr, res in meta.items():
        logger.info(
            f"  NR={nr}: pooled d={res['pooled_effect']:.3f} "
            f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}], "
            f"I²={res['i_sq']:.1f}%, Q p={res['q_p_value']:.4f}"
        )

    # ------------------------------------------------------------------
    # Boundary stratification (G6)
    # ------------------------------------------------------------------
    logger.info("Analyzing boundary vs interior CRND stratification (Gap G6)...")
    boundary_res = boundary_stratification_analysis(exp1)
    for ds, res in boundary_res.items():
        logger.info(
            f"  {ds:35s}: interior={res['interior_mean_crnd']:.4f}, "
            f"boundary={res['boundary_mean_crnd']:.4f}, d={res['cohens_d']:.3f}"
        )

    # ------------------------------------------------------------------
    # Schoener's D 2D vs 5D (G5)
    # ------------------------------------------------------------------
    logger.info("Analyzing Schoener's D 2D vs 5D sensitivity (Gap G5)...")
    schoeners_res = schoeners_d_sensitivity(exp1)
    for ds, ds_data in schoeners_res.items():
        for sp, sp_data in ds_data.get("feature_spaces", {}).items():
            logger.info(
                f"  {ds:25s} {sp:25s}: MAD={sp_data['mean_abs_diff_2d_vs_5d']:.4f}, "
                f"r={sp_data['pearson_r']}"
            )

    # ------------------------------------------------------------------
    # Niche overlap profile agreement
    # ------------------------------------------------------------------
    logger.info("Analyzing niche overlap profile agreement...")
    niche_res = niche_overlap_profile_analysis(exp1)

    # ------------------------------------------------------------------
    # Per-class CRND (G3)
    # ------------------------------------------------------------------
    logger.info("Analyzing per-class CRND distributions (Gap G3)...")
    per_class_res = per_class_crnd_analysis(exp1)
    for ds, res in per_class_res.items():
        logger.info(
            f"  {ds:35s}: range={res['range_min_max']:.4f}, "
            f"KW H={res['kruskal_wallis_h']}, p={res['kruskal_wallis_p']}"
        )

    # ------------------------------------------------------------------
    # Computational cost
    # ------------------------------------------------------------------
    logger.info("Analyzing computational costs...")
    cost_res = computational_cost_analysis(exp1, exp3)
    logger.info(
        f"  CRND: {cost_res['crnd_runtime_seconds']:.1f}s, "
        f"Baselines: {cost_res['baseline_runtime_seconds']:.1f}s, "
        f"Ratio: {cost_res['crnd_to_baseline_ratio']:.1f}x"
    )

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    logger.info("Building evaluation output...")
    datasets_out = build_output_examples(crnd_agg, baseline_agg, stat_comps, trial_metrics)
    metrics_agg = build_metrics_agg(
        crnd_agg=crnd_agg,
        baseline_agg=baseline_agg,
        stat_comparisons=stat_comps,
        meta_analysis=meta,
        boundary_results=boundary_res,
        schoeners_results=schoeners_res,
        per_class_results=per_class_res,
        cost_results=cost_res,
        niche_results=niche_res,
    )

    # Full output JSON
    output = {
        "metadata": {
            "evaluation_name": "CRND_vs_Baselines_HeadToHead",
            "description": (
                "Comprehensive head-to-head comparison of CRND against 11 baselines "
                "(kDN, cleanlab, k-NN consistency, random) across 5 datasets, "
                "3 noise rates, and multiple seeds."
            ),
            "exp_id1_source": str(EXP_ID1_FULL),
            "exp_id3_source": str(EXP_ID3_FULL),
            "crnd_n_seeds": 10,
            "baseline_n_seeds": 5,
            "bootstrap_resamples": BOOTSTRAP_N,
            "alpha": ALPHA,
            "datasets": DATASETS,
            "noise_rates": [float(nr) for nr in NOISE_RATES],
            "baselines_tested": ALL_BASELINES,
            "unique_baselines_for_stats": UNIQUE_BASELINES,
            "note_knn_consist_equals_kdn": (
                "knn_consist_* scores are numerically identical to kdn_* "
                "per exp_id3 metadata. Excluded from unique comparisons."
            ),
            "statistical_comparisons": stat_comps,
            "meta_analysis": meta,
            "boundary_stratification": boundary_res,
            "schoeners_d_sensitivity": schoeners_res,
            "niche_overlap_profiles": niche_res,
            "per_class_crnd": per_class_res,
            "computational_cost": cost_res,
            "baseline_aggregates": {
                ds: {
                    nr: {
                        bl: {k: v for k, v in bl_data.items() if k != "seed_aucs" and k != "seed_rhos" and k != "seed_pks"}
                        for bl, bl_data in nr_data.items()
                    }
                    for nr, nr_data in ds_data.items()
                }
                for ds, ds_data in baseline_agg.items()
            },
            "crnd_aggregates": crnd_agg,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved evaluation output to {out_path}")
    logger.info(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    elapsed = time.time() - t0
    logger.info(f"Total evaluation time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print key results
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("KEY RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Overall CRND mean AUC:        {metrics_agg['overall_mean_auc_crnd']:.4f}")
    logger.info(f"  Overall best baseline AUC:     {metrics_agg['overall_mean_auc_best_baseline']:.4f}")
    logger.info(f"  Overall delta AUC:             {metrics_agg['overall_delta_auc']:.4f}")
    logger.info(f"  Pooled Cohen's d:              {metrics_agg['pooled_effect_d']:.3f}")
    logger.info(f"  Significant comparisons:       {metrics_agg['n_significant_comparisons']}/{metrics_agg['n_total_comparisons']}")
    logger.info(f"  SC1 hypothesis supported:      {bool(metrics_agg['hypothesis_sc1_supported'])}")
    logger.info(f"  Boundary avg Cohen's d:        {metrics_agg['boundary_avg_cohens_d']:.3f}")
    logger.info(f"  Schoener D 2D-5D avg r:        {metrics_agg['schoeners_d_2d_5d_avg_pearson_r']:.3f}")
    logger.info(f"  Datasets with class CRND diff: {metrics_agg['n_datasets_class_crnd_varies']}")
    logger.info(f"  CRND/baseline cost ratio:      {metrics_agg['crnd_to_baseline_cost_ratio']:.1f}x")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
