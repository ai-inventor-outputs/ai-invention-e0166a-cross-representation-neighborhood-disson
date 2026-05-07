#!/usr/bin/env python3
"""
Cross-Representation Class Characterization Evaluation.

Synthesizes 4 dependency experiments into a class-level representation suitability map:
- Phase 1: Data merging from 4 experiments
- Phase 2: Representation Suitability Map (D-gap, uniformly-high-D, best-space, CRND stats)
- Phase 3: Disagreement Topology (boundary CRND graph, D similarity graph, Mantel test)
- Phase 4: Predictive Value (Kendall tau D-gap→F1-gap, CRND improvement, high-overlap precision)
- Phase 5: Clinical Interpretability Profiles
- Phase 6: Ecological vs. ML Overlap Comparison (Fisher F1, Bhattacharyya, Spearman)
"""

import json
import sys
import time
import resource
import warnings
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "eval.log"), rotation="30 MB", level="DEBUG")

# ── Dependency paths ─────────────────────────────────────────────────────────
BASE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044/3_invention_loop")
EXP1_PATH = BASE / "iter_2/gen_art/exp_id1_it2__opus/full_method_out.json"
EXP2_PATH = BASE / "iter_2/gen_art/exp_id2_it2__opus/full_method_out.json"
EXP3_IT3_PATH = BASE / "iter_3/gen_art/exp_id3_it3__opus/full_method_out.json"
EXP3_IT2_PATH = BASE / "iter_2/gen_art/exp_id3_it2__opus/full_method_out.json"

# Feature space name mappings
# exp_id1 uses: tfidf, sentence_transformer, llm_zeroshot
# exp_id2 uses: tfidf_word, sent_embed, tfidf_char
# We use exp_id1 names as canonical for D matrices
CANONICAL_SPACES = ["tfidf", "sentence_transformer", "llm_zeroshot"]
EXP2_SPACES = ["tfidf_word", "sent_embed", "tfidf_char"]

# For D matrices, we use 2D PCA variant as default
D_PCA_DIM = "2d"

DATASETS = [
    "medical_abstracts",
    "mimic_iv_ed_demo",
    "clinical_patient_triage_nl",
    "ohsumed_single",
    "mental_health_conditions",
]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Data Loading and Merging
# ═══════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    """Load a JSON file."""
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    data = json.loads(path.read_text())
    logger.info(f"  Loaded successfully")
    return data


def get_class_names(d_matrix_size: int, dataset_name: str, crnd_per_class: dict) -> list[str]:
    """Get ordered class names for a dataset from CRND per-class data."""
    if dataset_name in crnd_per_class:
        classes = sorted(crnd_per_class[dataset_name].keys())
        if len(classes) == d_matrix_size:
            return classes
    return [f"class_{i}" for i in range(d_matrix_size)]


def extract_d_matrices(exp1_meta: dict) -> dict[str, dict[str, np.ndarray]]:
    """Extract Schoener's D matrices per dataset per feature space (2D PCA)."""
    d_matrices = {}
    raw = exp1_meta["schoeners_d_matrices"]
    for ds_name in DATASETS:
        if ds_name not in raw:
            continue
        d_matrices[ds_name] = {}
        for space in CANONICAL_SPACES:
            key = f"{space}_{D_PCA_DIM}"
            if key in raw[ds_name]:
                mat = np.array(raw[ds_name][key], dtype=float)
                d_matrices[ds_name][space] = mat
    return d_matrices


def extract_classifier_f1(exp2_data: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Extract per-class-pair, per-feature-space best classifier F1 from exp_id2.

    Returns: {dataset: {class_pair: {feature_space: best_f1}}}
    """
    result = defaultdict(lambda: defaultdict(dict))
    for ds_entry in exp2_data["datasets"]:
        ds_name = ds_entry["dataset"]
        for ex in ds_entry["examples"]:
            pair = ex["metadata_class_pair"]
            space = ex["metadata_feature_space"]
            f1 = ex["metadata_best_f1"]
            # Keep max F1 across classifiers for this (pair, space)
            if space not in result[ds_name][pair] or f1 > result[ds_name][pair][space]:
                result[ds_name][pair][space] = f1
    return dict(result)


def extract_per_instance_crnd(exp1_data: dict) -> dict[str, pd.DataFrame]:
    """Extract per-instance CRND and class labels from exp_id1."""
    result = {}
    for ds_entry in exp1_data["datasets"]:
        ds_name = ds_entry["dataset"]
        rows = []
        for ex in ds_entry["examples"]:
            rows.append({
                "class": ex["output"],
                "crnd_k10": ex.get("metadata_crnd_k10", np.nan),
                "crnd_k20": ex.get("metadata_crnd_k20", np.nan),
                "boundary_proximity": ex.get("metadata_boundary_proximity", np.nan),
                "row_index": ex.get("metadata_row_index", -1),
            })
        result[ds_name] = pd.DataFrame(rows)
    return result


def extract_per_instance_crnd_ablation(exp3_it3_data: dict) -> dict[str, pd.DataFrame]:
    """Extract per-instance CRND pairwise decomposition from exp_id3_it3."""
    result = {}
    for ds_entry in exp3_it3_data["datasets"]:
        ds_name = ds_entry["dataset"]
        rows = []
        for ex in ds_entry["examples"]:
            row = {"class": ex["output"], "row_index": ex.get("metadata_row_index", -1)}
            for key in ["metadata_crnd_pairwise_tfidf_vs_sentence_transformer",
                        "metadata_crnd_pairwise_tfidf_vs_llm_zeroshot",
                        "metadata_crnd_pairwise_sentence_transformer_vs_llm_zeroshot"]:
                if key in ex:
                    short_key = key.replace("metadata_crnd_pairwise_", "crnd_pw_")
                    row[short_key] = ex[key]
            rows.append(row)
        if rows:
            result[ds_name] = pd.DataFrame(rows)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Representation Suitability Map
# ═══════════════════════════════════════════════════════════════════════════

def compute_d_gap(d_matrices: dict[str, dict[str, np.ndarray]],
                  class_names_map: dict[str, list[str]]) -> dict:
    """Compute D-gap for each class pair in each dataset.

    D_gap(i,j) = max(D_space1, D_space2, D_space3) - min(D_space1, D_space2, D_space3)
    """
    results = {}
    for ds_name, space_mats in d_matrices.items():
        classes = class_names_map[ds_name]
        n = len(classes)
        pair_results = {}
        for i, j in combinations(range(n), 2):
            pair_name = f"{classes[i]}__vs__{classes[j]}"
            d_values = {}
            for space, mat in space_mats.items():
                d_values[space] = mat[i, j]
            vals = list(d_values.values())
            if len(vals) < 2:
                continue
            d_gap = max(vals) - min(vals)
            best_space = min(d_values, key=d_values.get)  # lowest D = best separation
            worst_space = max(d_values, key=d_values.get)
            min_d = min(vals)
            max_d = max(vals)
            mean_d = np.mean(vals)
            pair_results[pair_name] = {
                "d_gap": d_gap,
                "d_values": d_values,
                "best_space": best_space,
                "worst_space": worst_space,
                "min_d": min_d,
                "max_d": max_d,
                "mean_d": mean_d,
            }
        mean_d_gap = np.mean([v["d_gap"] for v in pair_results.values()]) if pair_results else 0.0
        max_d_gap = max([v["d_gap"] for v in pair_results.values()]) if pair_results else 0.0
        results[ds_name] = {
            "pairs": pair_results,
            "mean_d_gap": mean_d_gap,
            "max_d_gap": max_d_gap,
        }
    return results


def compute_uniformly_high_d(d_gap_results: dict,
                             thresholds: list[float] = [0.4, 0.5, 0.6, 0.7]) -> dict:
    """Identify class pairs where min(D) > threshold across all spaces."""
    results = {}
    for ds_name, ds_data in d_gap_results.items():
        ds_results = {}
        for threshold in thresholds:
            high_pairs = []
            for pair_name, pair_data in ds_data["pairs"].items():
                if pair_data["min_d"] > threshold:
                    high_pairs.append(pair_name)
            total = len(ds_data["pairs"])
            ds_results[f"threshold_{threshold}"] = {
                "count": len(high_pairs),
                "fraction": len(high_pairs) / total if total > 0 else 0.0,
                "pairs": high_pairs,
            }
        results[ds_name] = ds_results
    return results


def compute_per_class_crnd_stats(exp1_meta: dict) -> dict:
    """Extract per-class CRND mean and variance from exp_id1 metadata."""
    results = {}
    crnd_per_class = exp1_meta.get("crnd_per_class", {})
    for ds_name in DATASETS:
        if ds_name not in crnd_per_class:
            continue
        ds_results = {}
        for class_name, stats_dict in crnd_per_class[ds_name].items():
            ds_results[class_name] = {
                "mean_crnd": stats_dict["mean"],
                "std_crnd": stats_dict["std"],
                "var_crnd": stats_dict["std"] ** 2,
                "n": stats_dict["n"],
            }
        results[ds_name] = ds_results
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Disagreement Topology
# ═══════════════════════════════════════════════════════════════════════════

def build_boundary_crnd_graph(per_instance: dict[str, pd.DataFrame],
                              class_names_map: dict[str, list[str]],
                              k: int = 10) -> dict:
    """Build boundary CRND graph: nodes=classes, edge weight = mean CRND at boundary.

    A boundary instance is one whose k-NN set contains instances of another class.
    Since we don't have raw features, we approximate using available CRND and class data.
    We compute mean CRND for instances of class i that have boundary_proximity > 0.
    """
    results = {}
    for ds_name, df in per_instance.items():
        if ds_name not in class_names_map:
            continue
        classes = class_names_map[ds_name]
        n = len(classes)
        # Build a boundary CRND matrix
        boundary_matrix = np.zeros((n, n))
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i == j:
                    continue
                # Instances of class i that are near class j boundary
                # Use boundary_proximity as a proxy
                mask_i = df["class"] == ci
                if mask_i.sum() == 0:
                    boundary_matrix[i, j] = np.nan
                    continue
                instances_i = df[mask_i]
                # Use instances with boundary_proximity > 0.3 as "boundary instances"
                boundary_instances = instances_i[instances_i["boundary_proximity"] > 0.3]
                if len(boundary_instances) > 0:
                    boundary_matrix[i, j] = boundary_instances["crnd_k10"].mean()
                else:
                    boundary_matrix[i, j] = instances_i["crnd_k10"].mean()
        # Symmetrize: average (i,j) and (j,i)
        sym_matrix = (boundary_matrix + boundary_matrix.T) / 2
        np.fill_diagonal(sym_matrix, 0.0)
        results[ds_name] = {
            "matrix": sym_matrix,
            "classes": classes,
        }
    return results


def build_d_similarity_graph(d_matrices: dict[str, dict[str, np.ndarray]],
                             class_names_map: dict[str, list[str]]) -> dict:
    """Build Schoener's D similarity graph: mean D across feature spaces."""
    results = {}
    for ds_name, space_mats in d_matrices.items():
        if ds_name not in class_names_map:
            continue
        classes = class_names_map[ds_name]
        n = len(classes)
        mean_d_matrix = np.zeros((n, n))
        count = 0
        for space, mat in space_mats.items():
            mean_d_matrix += mat
            count += 1
        if count > 0:
            mean_d_matrix /= count
        np.fill_diagonal(mean_d_matrix, 0.0)
        results[ds_name] = {
            "matrix": mean_d_matrix,
            "classes": classes,
        }
    return results


def mantel_test(dist_matrix_1: np.ndarray, dist_matrix_2: np.ndarray,
                n_permutations: int = 9999) -> dict:
    """Compute Mantel test between two distance matrices.

    Uses Pearson correlation on condensed distance vectors.
    """
    n = dist_matrix_1.shape[0]
    if n < 3:
        return {"mantel_r": np.nan, "p_value": np.nan, "n_permutations": 0}

    # Extract upper triangle (condensed form)
    idx = np.triu_indices(n, k=1)
    vec1 = dist_matrix_1[idx]
    vec2 = dist_matrix_2[idx]

    # Remove NaN pairs
    valid = ~(np.isnan(vec1) | np.isnan(vec2))
    vec1 = vec1[valid]
    vec2 = vec2[valid]

    if len(vec1) < 3:
        return {"mantel_r": np.nan, "p_value": np.nan, "n_permutations": 0}

    # Observed correlation
    r_obs, _ = stats.pearsonr(vec1, vec2)

    # Permutation test
    count_ge = 0
    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_matrix = dist_matrix_1[np.ix_(perm, perm)]
        perm_vec = perm_matrix[idx]
        perm_vec = perm_vec[valid]
        r_perm, _ = stats.pearsonr(perm_vec, vec2)
        if r_perm >= r_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return {
        "mantel_r": float(r_obs),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


def compute_disagreement_topology(boundary_graph: dict, d_graph: dict) -> dict:
    """Compute Mantel test between boundary CRND and D similarity graphs."""
    results = {}
    for ds_name in DATASETS:
        if ds_name not in boundary_graph or ds_name not in d_graph:
            continue
        crnd_mat = boundary_graph[ds_name]["matrix"]
        d_mat = d_graph[ds_name]["matrix"]
        # Convert D to distance: 1 - D
        d_dist = 1.0 - d_mat
        np.fill_diagonal(d_dist, 0.0)
        # CRND is already a dissimilarity-like measure (higher = more dissonance)
        # But for Mantel test we need distance matrices
        # Use CRND directly as distance (higher CRND = more boundary disagreement)
        mantel_result = mantel_test(crnd_mat, d_dist, n_permutations=9999)
        results[ds_name] = mantel_result
        logger.info(f"  Mantel test {ds_name}: r={mantel_result['mantel_r']:.4f}, p={mantel_result['p_value']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Predictive Value of the Suitability Map
# ═══════════════════════════════════════════════════════════════════════════

def compute_kendall_tau_d_gap_vs_f1_gap(
    d_gap_results: dict,
    classifier_f1: dict[str, dict[str, dict[str, float]]],
) -> dict:
    """Compute Kendall's tau between D-gap ranking and classifier performance gap ranking.

    For each class pair, D-gap = max(D) - min(D) across spaces.
    Classifier gap = max(best_F1) - min(best_F1) across spaces.
    """
    results = {}
    pooled_d_gaps = []
    pooled_f1_gaps = []

    for ds_name in DATASETS:
        if ds_name not in d_gap_results or ds_name not in classifier_f1:
            continue
        d_gaps = []
        f1_gaps = []
        pairs_used = []

        for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
            if pair_name not in classifier_f1[ds_name]:
                continue
            f1_per_space = classifier_f1[ds_name][pair_name]
            if len(f1_per_space) < 2:
                continue
            f1_vals = list(f1_per_space.values())
            f1_gap = max(f1_vals) - min(f1_vals)
            d_gaps.append(pair_data["d_gap"])
            f1_gaps.append(f1_gap)
            pairs_used.append(pair_name)
            pooled_d_gaps.append(pair_data["d_gap"])
            pooled_f1_gaps.append(f1_gap)

        if len(d_gaps) >= 3:
            tau, p_val = stats.kendalltau(d_gaps, f1_gaps)
            # Bootstrap CI
            n_boot = 1000
            rng = np.random.RandomState(42)
            boot_taus = []
            for _ in range(n_boot):
                idx = rng.choice(len(d_gaps), size=len(d_gaps), replace=True)
                bt, _ = stats.kendalltau(np.array(d_gaps)[idx], np.array(f1_gaps)[idx])
                if not np.isnan(bt):
                    boot_taus.append(bt)
            ci_low = np.percentile(boot_taus, 2.5) if boot_taus else np.nan
            ci_high = np.percentile(boot_taus, 97.5) if boot_taus else np.nan
        else:
            tau, p_val = np.nan, np.nan
            ci_low, ci_high = np.nan, np.nan

        results[ds_name] = {
            "kendall_tau": float(tau) if not np.isnan(tau) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "ci_low": float(ci_low) if not np.isnan(ci_low) else None,
            "ci_high": float(ci_high) if not np.isnan(ci_high) else None,
            "n_pairs": len(d_gaps),
        }

    # Pooled
    if len(pooled_d_gaps) >= 3:
        tau_p, pval_p = stats.kendalltau(pooled_d_gaps, pooled_f1_gaps)
    else:
        tau_p, pval_p = np.nan, np.nan

    results["pooled"] = {
        "kendall_tau": float(tau_p) if not np.isnan(tau_p) else None,
        "p_value": float(pval_p) if not np.isnan(pval_p) else None,
        "n_pairs": len(pooled_d_gaps),
    }
    return results


def compute_crnd_prediction_improvement(
    d_gap_results: dict,
    classifier_f1: dict,
    crnd_stats: dict,
    class_names_map: dict,
) -> dict:
    """Test if adding CRND improves prediction of best space beyond D-gap alone.

    Model A: predict best-space label from D-gap alone
    Model B: predict best-space label from D-gap + mean class CRND
    Compare LOO-CV accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    # Collect features and labels
    X_dgap = []
    X_dgap_crnd = []
    y_labels = []

    for ds_name in DATASETS:
        if ds_name not in d_gap_results or ds_name not in classifier_f1:
            continue
        classes = class_names_map.get(ds_name, [])

        for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
            if pair_name not in classifier_f1[ds_name]:
                continue
            f1_per_space = classifier_f1[ds_name][pair_name]
            if len(f1_per_space) < 2:
                continue

            # Best space by classifier = space with highest F1
            best_space_by_f1 = max(f1_per_space, key=f1_per_space.get)

            d_gap = pair_data["d_gap"]
            X_dgap.append([d_gap])

            # Get mean CRND for the two classes in this pair
            parts = pair_name.split("__vs__")
            crnd_vals = []
            for part in parts:
                if ds_name in crnd_stats and part in crnd_stats[ds_name]:
                    crnd_vals.append(crnd_stats[ds_name][part]["mean_crnd"])
            mean_crnd = np.mean(crnd_vals) if crnd_vals else 0.5
            X_dgap_crnd.append([d_gap, mean_crnd])
            y_labels.append(best_space_by_f1)

    if len(X_dgap) < 5:
        return {"delta_accuracy": None, "model_a_accuracy": None, "model_b_accuracy": None, "n_samples": len(X_dgap)}

    X_a = np.array(X_dgap)
    X_b = np.array(X_dgap_crnd)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # LOO-CV
    correct_a = 0
    correct_b = 0
    n = len(y)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        test_idx = [i]
        try:
            clf_a = LogisticRegression(max_iter=1000, random_state=42)
            clf_a.fit(X_a[train_idx], y[train_idx])
            if clf_a.predict(X_a[test_idx])[0] == y[i]:
                correct_a += 1
        except Exception:
            pass
        try:
            clf_b = LogisticRegression(max_iter=1000, random_state=42)
            clf_b.fit(X_b[train_idx], y[train_idx])
            if clf_b.predict(X_b[test_idx])[0] == y[i]:
                correct_b += 1
        except Exception:
            pass

    acc_a = correct_a / n
    acc_b = correct_b / n
    return {
        "delta_accuracy": acc_b - acc_a,
        "model_a_accuracy": acc_a,
        "model_b_accuracy": acc_b,
        "n_samples": n,
    }


def compute_high_overlap_precision(
    d_gap_results: dict,
    classifier_f1: dict,
    thresholds: list[float] = [0.4, 0.5, 0.6, 0.7],
) -> dict:
    """Compute precision/recall/F1 for identifying unsolvable class pairs.

    True high-overlap = best F1 across all spaces < 0.6.
    Predictor: min(D) > T.
    """
    # Collect all (min_d, best_f1) pairs
    min_ds = []
    best_f1s = []

    for ds_name in DATASETS:
        if ds_name not in d_gap_results or ds_name not in classifier_f1:
            continue
        for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
            if pair_name not in classifier_f1[ds_name]:
                continue
            f1_per_space = classifier_f1[ds_name][pair_name]
            if not f1_per_space:
                continue
            best_f1 = max(f1_per_space.values())
            min_ds.append(pair_data["min_d"])
            best_f1s.append(best_f1)

    if not min_ds:
        return {"best_threshold": None, "best_precision": None, "best_recall": None, "best_f1": None}

    min_ds = np.array(min_ds)
    best_f1s = np.array(best_f1s)
    true_hard = best_f1s < 0.6  # unsolvable pairs

    results_per_threshold = {}
    best_f1_score = -1
    best_t = None

    for t in thresholds:
        predicted_hard = min_ds > t
        tp = np.sum(predicted_hard & true_hard)
        fp = np.sum(predicted_hard & ~true_hard)
        fn = np.sum(~predicted_hard & true_hard)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results_per_threshold[f"threshold_{t}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        if f1 > best_f1_score:
            best_f1_score = f1
            best_t = t

    return {
        "per_threshold": results_per_threshold,
        "best_threshold": best_t,
        "best_precision": float(results_per_threshold[f"threshold_{best_t}"]["precision"]) if best_t else None,
        "best_recall": float(results_per_threshold[f"threshold_{best_t}"]["recall"]) if best_t else None,
        "best_f1": float(best_f1_score),
        "n_true_hard": int(np.sum(true_hard)),
        "n_total_pairs": len(min_ds),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Clinical Interpretability Profiles
# ═══════════════════════════════════════════════════════════════════════════

def generate_clinical_profiles(
    d_gap_results: dict,
    crnd_stats: dict,
    class_names_map: dict,
    d_matrices: dict,
) -> dict:
    """Generate structured clinical interpretability profiles for medical datasets."""
    clinical_datasets = ["medical_abstracts", "mental_health_conditions"]
    results = {}

    for ds_name in clinical_datasets:
        if ds_name not in d_gap_results or ds_name not in crnd_stats:
            continue
        classes = class_names_map.get(ds_name, [])
        profiles = {}
        for cls in classes:
            if cls not in crnd_stats[ds_name]:
                continue
            cls_crnd = crnd_stats[ds_name][cls]

            # Find best and worst separated pairs
            best_sep = []
            worst_sep = []
            for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
                parts = pair_name.split("__vs__")
                if cls not in parts:
                    continue
                other = parts[0] if parts[1] == cls else parts[1]
                min_d = pair_data["min_d"]
                max_d = pair_data["max_d"]
                best_space = pair_data["best_space"]
                if min_d < 0.3:
                    best_sep.append((other, best_space, min_d))
                if max_d > 0.6:
                    worst_sep.append((other, pair_data["worst_space"], max_d))

            # Build profile string
            stability = "stable" if cls_crnd["std_crnd"] < 0.06 else "ambiguous"
            profile_parts = [f"Class {cls}:"]
            if best_sep:
                sep_str = ", ".join([f"{o} in {s} (D={d:.2f})" for o, s, d in best_sep[:3]])
                profile_parts.append(f"best separated from [{sep_str}]")
            if worst_sep:
                ov_str = ", ".join([f"{o} in {s} (D={d:.2f})" for o, s, d in worst_sep[:3]])
                profile_parts.append(f"overlaps most with [{ov_str}]")
            profile_parts.append(
                f"mean CRND={cls_crnd['mean_crnd']:.2f} (+-{cls_crnd['std_crnd']:.2f}), "
                f"indicating {stability} cross-representation behavior."
            )
            profiles[cls] = " ".join(profile_parts)

        results[ds_name] = profiles
    return results


def identify_human_deferral_pairs(
    d_gap_results: dict,
    boundary_graph: dict,
    d_threshold: float = 0.5,
    crnd_threshold: float = 0.85,
) -> tuple[dict, int]:
    """Identify pairs where both uniformly high D AND high boundary CRND."""
    results = {}
    total_deferral = 0
    for ds_name in DATASETS:
        if ds_name not in d_gap_results or ds_name not in boundary_graph:
            continue
        classes = boundary_graph[ds_name]["classes"]
        crnd_mat = boundary_graph[ds_name]["matrix"]
        deferral_pairs = []

        for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
            if pair_data["min_d"] <= d_threshold:
                continue
            # Check boundary CRND
            parts = pair_name.split("__vs__")
            if len(parts) != 2:
                continue
            try:
                i = classes.index(parts[0])
                j = classes.index(parts[1])
            except ValueError:
                continue
            boundary_crnd = crnd_mat[i, j]
            if not np.isnan(boundary_crnd) and boundary_crnd > crnd_threshold:
                deferral_pairs.append({
                    "pair": pair_name,
                    "min_d": pair_data["min_d"],
                    "boundary_crnd": float(boundary_crnd),
                })

        results[ds_name] = {
            "count": len(deferral_pairs),
            "pairs": deferral_pairs,
        }
        total_deferral += len(deferral_pairs)
    return results, total_deferral


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Ecological vs. ML Overlap Comparison
# ═══════════════════════════════════════════════════════════════════════════

def compute_fisher_discriminant_ratio(
    per_instance: dict[str, pd.DataFrame],
    class_names_map: dict[str, list[str]],
    d_matrices: dict,
) -> dict:
    """Compute Fisher's Discriminant Ratio (F1) for each class pair.

    Since we don't have raw features, we use CRND values as proxy features.
    F1(i,j) = max_k [(mu_ik - mu_jk)^2 / (sigma^2_ik + sigma^2_jk)]

    We use available numeric fields per instance as proxy features:
    crnd_k10, crnd_k20, boundary_proximity
    """
    results = {}
    for ds_name, df in per_instance.items():
        if ds_name not in class_names_map:
            continue
        classes = class_names_map[ds_name]
        feature_cols = [c for c in ["crnd_k10", "crnd_k20", "boundary_proximity"] if c in df.columns]
        if not feature_cols:
            continue

        pair_fisher = {}
        for ci, cj in combinations(classes, 2):
            pair_name = f"{ci}__vs__{cj}"
            mask_i = df["class"] == ci
            mask_j = df["class"] == cj
            if mask_i.sum() < 2 or mask_j.sum() < 2:
                continue

            max_f1 = 0.0
            for col in feature_cols:
                vals_i = df.loc[mask_i, col].dropna().values
                vals_j = df.loc[mask_j, col].dropna().values
                if len(vals_i) < 2 or len(vals_j) < 2:
                    continue
                mu_i, mu_j = np.mean(vals_i), np.mean(vals_j)
                var_i, var_j = np.var(vals_i, ddof=1), np.var(vals_j, ddof=1)
                denom = var_i + var_j
                if denom < 1e-12:
                    continue
                f1_val = (mu_i - mu_j) ** 2 / denom
                max_f1 = max(max_f1, f1_val)
            pair_fisher[pair_name] = float(max_f1)
        results[ds_name] = pair_fisher
    return results


def compute_bhattacharyya_coefficient(
    per_instance: dict[str, pd.DataFrame],
    class_names_map: dict[str, list[str]],
) -> dict:
    """Compute Bhattacharyya coefficient for each class pair.

    Project onto top-2 features, compute KDE, then BC = integral sqrt(p_i * p_j) dx.
    Uses crnd_k10 as the primary feature for 1D KDE.
    """
    results = {}
    for ds_name, df in per_instance.items():
        if ds_name not in class_names_map:
            continue
        classes = class_names_map[ds_name]
        pair_bc = {}

        for ci, cj in combinations(classes, 2):
            pair_name = f"{ci}__vs__{cj}"
            mask_i = df["class"] == ci
            mask_j = df["class"] == cj

            vals_i = df.loc[mask_i, "crnd_k10"].dropna().values
            vals_j = df.loc[mask_j, "crnd_k10"].dropna().values

            if len(vals_i) < 3 or len(vals_j) < 3:
                pair_bc[pair_name] = np.nan
                continue

            # 1D KDE-based Bhattacharyya coefficient
            try:
                x_min = min(vals_i.min(), vals_j.min()) - 0.1
                x_max = max(vals_i.max(), vals_j.max()) + 0.1
                x_grid = np.linspace(x_min, x_max, 200)

                kde_i = stats.gaussian_kde(vals_i)
                kde_j = stats.gaussian_kde(vals_j)

                p_i = kde_i(x_grid)
                p_j = kde_j(x_grid)

                # BC = integral sqrt(p_i * p_j) dx
                dx = x_grid[1] - x_grid[0]
                bc = np.sum(np.sqrt(p_i * p_j)) * dx
                bc = min(bc, 1.0)  # Clip to [0, 1]
                pair_bc[pair_name] = float(bc)
            except Exception:
                pair_bc[pair_name] = np.nan

        results[ds_name] = pair_bc
    return results


def compute_ecological_vs_ml_correlations(
    d_gap_results: dict,
    fisher_results: dict,
    bhattacharyya_results: dict,
    classifier_f1: dict,
) -> dict:
    """Compute Spearman correlations between ecological and ML overlap measures."""
    # Pool all (D, Fisher, BC, F1_error) tuples
    d_vals = []
    fisher_vals = []
    bc_vals = []
    f1_error_vals = []

    for ds_name in DATASETS:
        if ds_name not in d_gap_results:
            continue
        for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
            mean_d = pair_data["mean_d"]
            fisher_f1 = fisher_results.get(ds_name, {}).get(pair_name, np.nan)
            bc = bhattacharyya_results.get(ds_name, {}).get(pair_name, np.nan)

            # Get best F1 error from classifier
            if ds_name in classifier_f1 and pair_name in classifier_f1[ds_name]:
                best_f1 = max(classifier_f1[ds_name][pair_name].values())
                f1_error = 1.0 - best_f1
            else:
                f1_error = np.nan

            if not np.isnan(mean_d):
                d_vals.append(mean_d)
                fisher_vals.append(fisher_f1 if not np.isnan(fisher_f1) else 0.0)
                bc_vals.append(bc if not np.isnan(bc) else 0.5)
                f1_error_vals.append(f1_error if not np.isnan(f1_error) else 0.5)

    d_vals = np.array(d_vals)
    fisher_vals = np.array(fisher_vals)
    bc_vals = np.array(bc_vals)
    f1_error_vals = np.array(f1_error_vals)

    # Spearman: D vs 1/F1 (inverted Fisher)
    inv_fisher = np.where(fisher_vals > 1e-10, 1.0 / fisher_vals, np.nan)
    valid_fisher = ~np.isnan(inv_fisher)
    if np.sum(valid_fisher) >= 3:
        rho_d_fisher, p_d_fisher = stats.spearmanr(d_vals[valid_fisher], inv_fisher[valid_fisher])
    else:
        rho_d_fisher, p_d_fisher = np.nan, np.nan

    # Spearman: D vs Bhattacharyya
    valid_bc = ~np.isnan(bc_vals)
    if np.sum(valid_bc) >= 3:
        rho_d_bc, p_d_bc = stats.spearmanr(d_vals[valid_bc], bc_vals[valid_bc])
    else:
        rho_d_bc, p_d_bc = np.nan, np.nan

    # Partial Spearman: D -> F1_error controlling for Fisher
    valid_all = valid_fisher & ~np.isnan(f1_error_vals)
    if np.sum(valid_all) >= 5:
        partial_d = _partial_spearman(
            d_vals[valid_all], f1_error_vals[valid_all], fisher_vals[valid_all]
        )
        partial_fisher = _partial_spearman(
            fisher_vals[valid_all], f1_error_vals[valid_all], d_vals[valid_all]
        )
    else:
        partial_d = {"rho": np.nan, "p_value": np.nan}
        partial_fisher = {"rho": np.nan, "p_value": np.nan}

    return {
        "spearman_d_vs_inv_fisher": {
            "rho": float(rho_d_fisher) if not np.isnan(rho_d_fisher) else None,
            "p_value": float(p_d_fisher) if not np.isnan(p_d_fisher) else None,
        },
        "spearman_d_vs_bhattacharyya": {
            "rho": float(rho_d_bc) if not np.isnan(rho_d_bc) else None,
            "p_value": float(p_d_bc) if not np.isnan(p_d_bc) else None,
        },
        "partial_spearman_d_controlling_fisher": {
            "rho": float(partial_d["rho"]) if partial_d["rho"] is not None and not np.isnan(partial_d["rho"]) else None,
            "p_value": float(partial_d["p_value"]) if partial_d["p_value"] is not None and not np.isnan(partial_d["p_value"]) else None,
        },
        "partial_spearman_fisher_controlling_d": {
            "rho": float(partial_fisher["rho"]) if partial_fisher["rho"] is not None and not np.isnan(partial_fisher["rho"]) else None,
            "p_value": float(partial_fisher["p_value"]) if partial_fisher["p_value"] is not None and not np.isnan(partial_fisher["p_value"]) else None,
        },
        "n_pairs_pooled": len(d_vals),
    }


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Compute partial Spearman correlation of x and y, controlling for z."""
    try:
        # Rank transform
        x_rank = stats.rankdata(x)
        y_rank = stats.rankdata(y)
        z_rank = stats.rankdata(z)

        # Residualize x and y on z using linear regression
        coef_xz = np.polyfit(z_rank, x_rank, 1)
        res_x = x_rank - np.polyval(coef_xz, z_rank)

        coef_yz = np.polyfit(z_rank, y_rank, 1)
        res_y = y_rank - np.polyval(coef_yz, z_rank)

        rho, p = stats.pearsonr(res_x, res_y)
        return {"rho": rho, "p_value": p}
    except Exception:
        return {"rho": np.nan, "p_value": np.nan}


# ═══════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════

def safe_float(val) -> float:
    """Convert to float, replacing None/NaN with 0.0 for schema compliance."""
    if val is None:
        return 0.0
    if isinstance(val, float) and np.isnan(val):
        return 0.0
    return float(val)


def build_output(
    d_gap_results: dict,
    uniformly_high_d: dict,
    crnd_stats: dict,
    mantel_results: dict,
    kendall_results: dict,
    crnd_improvement: dict,
    high_overlap: dict,
    clinical_profiles: dict,
    deferral_pairs: dict,
    total_deferral: int,
    eco_vs_ml: dict,
    exp1_data: dict,
    classifier_f1: dict,
    class_names_map: dict,
) -> dict:
    """Build the final output JSON conforming to exp_eval_sol_out schema."""

    # ── metrics_agg ──────────────────────────────────────────────────────
    mean_d_gaps = [d_gap_results[ds]["mean_d_gap"] for ds in DATASETS if ds in d_gap_results]
    mean_d_gap_across = np.mean(mean_d_gaps) if mean_d_gaps else 0.0

    # Fraction uniformly high D pairs (threshold 0.5)
    total_high = 0
    total_pairs = 0
    for ds in DATASETS:
        if ds in uniformly_high_d and "threshold_0.5" in uniformly_high_d[ds]:
            total_high += uniformly_high_d[ds]["threshold_0.5"]["count"]
        if ds in d_gap_results:
            total_pairs += len(d_gap_results[ds]["pairs"])
    frac_high = total_high / total_pairs if total_pairs > 0 else 0.0

    mantel_rs = [v["mantel_r"] for v in mantel_results.values()
                 if isinstance(v.get("mantel_r"), (int, float)) and not np.isnan(v.get("mantel_r", np.nan))]
    mantel_ps = [v["p_value"] for v in mantel_results.values()
                 if isinstance(v.get("p_value"), (int, float)) and not np.isnan(v.get("p_value", np.nan))]

    metrics_agg = {
        "mean_d_gap_across_datasets": safe_float(mean_d_gap_across),
        "frac_uniformly_high_d_pairs": safe_float(frac_high),
        "mean_mantel_r": safe_float(np.mean(mantel_rs) if mantel_rs else np.nan),
        "mean_mantel_p": safe_float(np.mean(mantel_ps) if mantel_ps else np.nan),
        "kendall_tau_d_gap_vs_f1_gap_pooled": safe_float(
            kendall_results.get("pooled", {}).get("kendall_tau")),
        "crnd_improves_prediction_delta": safe_float(
            crnd_improvement.get("delta_accuracy")),
        "high_overlap_precision_at_best_threshold": safe_float(
            high_overlap.get("best_precision")),
        "spearman_schoener_vs_fisher_pooled": safe_float(
            eco_vs_ml.get("spearman_d_vs_inv_fisher", {}).get("rho")),
        "spearman_schoener_vs_bhattacharyya_pooled": safe_float(
            eco_vs_ml.get("spearman_d_vs_bhattacharyya", {}).get("rho")),
        "schoener_unique_partial_spearman": safe_float(
            eco_vs_ml.get("partial_spearman_d_controlling_fisher", {}).get("rho")),
        "num_human_deferral_pairs_total": safe_float(total_deferral),
        "num_datasets": safe_float(len(DATASETS)),
        "num_class_pairs_total": safe_float(total_pairs),
    }

    # ── datasets (per-example output) ────────────────────────────────────
    datasets_out = []
    for ds_entry in exp1_data["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in d_gap_results:
            continue

        examples_out = []
        for ex in ds_entry["examples"]:
            cls = ex["output"]
            crnd_k10 = ex.get("metadata_crnd_k10", 0.0)

            # Get class-level D-gap info
            cls_d_gap_min = None
            cls_d_gap_max = None
            cls_best_space = None
            pair_count = 0
            for pair_name, pair_data in d_gap_results[ds_name]["pairs"].items():
                parts = pair_name.split("__vs__")
                if cls in parts:
                    pair_count += 1
                    dg = pair_data["d_gap"]
                    if cls_d_gap_min is None or dg < cls_d_gap_min:
                        cls_d_gap_min = dg
                    if cls_d_gap_max is None or dg > cls_d_gap_max:
                        cls_d_gap_max = dg
                    if cls_best_space is None:
                        cls_best_space = pair_data["best_space"]

            # Get class CRND stats
            cls_crnd_mean = crnd_stats.get(ds_name, {}).get(cls, {}).get("mean_crnd", 0.0)
            cls_crnd_var = crnd_stats.get(ds_name, {}).get(cls, {}).get("var_crnd", 0.0)

            example = {
                "input": ex["input"][:500],
                "output": cls,
                "predict_d_gap_profile": (
                    f"Class {cls}: D-gap range [{cls_d_gap_min:.3f}, {cls_d_gap_max:.3f}] "
                    f"across {pair_count} pairs. Best separation space: {cls_best_space}. "
                    f"Mean class CRND: {cls_crnd_mean:.3f} (var={cls_crnd_var:.4f}). "
                    f"Instance CRND_k10: {crnd_k10:.3f}."
                ) if cls_d_gap_min is not None else f"Class {cls}: No D-gap data available.",
                "predict_suitability_label": (
                    cls_best_space if cls_best_space else "unknown"
                ),
                "eval_instance_crnd_k10": safe_float(crnd_k10),
                "eval_class_mean_crnd": safe_float(cls_crnd_mean),
                "eval_class_crnd_variance": safe_float(cls_crnd_var),
                "eval_min_d_gap_for_class": safe_float(cls_d_gap_min if cls_d_gap_min is not None else 0.0),
                "eval_max_d_gap_for_class": safe_float(cls_d_gap_max if cls_d_gap_max is not None else 0.0),
                "metadata_dataset": ds_name,
                "metadata_row_index": ex.get("metadata_row_index", -1),
                "metadata_boundary_proximity": ex.get("metadata_boundary_proximity", 0.0),
            }
            examples_out.append(example)

        datasets_out.append({
            "dataset": ds_name,
            "examples": examples_out,
        })

    return {
        "metadata": {
            "evaluation_name": "Cross-Representation Class Characterization",
            "description": (
                "Synthesizes 4 dependency experiments into a class-level representation suitability map "
                "using Schoener's D gap analysis, boundary CRND disagreement topology with Mantel test, "
                "predictive validation (Kendall tau), ecological vs. ML overlap comparison "
                "(Fisher F1, Bhattacharyya coefficient), and clinical interpretability profiles."
            ),
            "d_gap_results_per_dataset": {
                ds: {
                    "mean_d_gap": d_gap_results[ds]["mean_d_gap"],
                    "max_d_gap": d_gap_results[ds]["max_d_gap"],
                    "n_pairs": len(d_gap_results[ds]["pairs"]),
                    "pair_details": {
                        pair: {
                            "d_gap": v["d_gap"],
                            "d_values": {k: round(dv, 4) for k, dv in v["d_values"].items()},
                            "best_space": v["best_space"],
                            "min_d": round(v["min_d"], 4),
                            "max_d": round(v["max_d"], 4),
                        }
                        for pair, v in d_gap_results[ds]["pairs"].items()
                    },
                }
                for ds in DATASETS if ds in d_gap_results
            },
            "uniformly_high_d_pairs": uniformly_high_d,
            "crnd_per_class_stats": crnd_stats,
            "mantel_test_results": mantel_results,
            "kendall_tau_d_gap_vs_f1_gap": kendall_results,
            "crnd_prediction_improvement": crnd_improvement,
            "high_overlap_identification": high_overlap,
            "clinical_profiles": clinical_profiles,
            "human_deferral_pairs": deferral_pairs,
            "ecological_vs_ml_comparison": eco_vs_ml,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Cross-Representation Class Characterization Evaluation")
    logger.info("=" * 70)

    # ── Phase 1: Load data ───────────────────────────────────────────────
    logger.info("Phase 1: Loading dependency data...")
    exp1_data = load_json(EXP1_PATH)
    exp2_data = load_json(EXP2_PATH)
    exp3_it3_data = load_json(EXP3_IT3_PATH)
    exp3_it2_data = load_json(EXP3_IT2_PATH)

    exp1_meta = exp1_data["metadata"]

    # Extract structures
    d_matrices = extract_d_matrices(exp1_meta)
    logger.info(f"  D matrices: {list(d_matrices.keys())}")

    class_names_map = {}
    for ds_name in DATASETS:
        if ds_name in d_matrices:
            n = d_matrices[ds_name][CANONICAL_SPACES[0]].shape[0]
            class_names_map[ds_name] = get_class_names(
                n, ds_name, exp1_meta.get("crnd_per_class", {})
            )
            logger.info(f"  {ds_name}: {len(class_names_map[ds_name])} classes")

    classifier_f1 = extract_classifier_f1(exp2_data)
    logger.info(f"  Classifier F1 data: {list(classifier_f1.keys())}")

    per_instance = extract_per_instance_crnd(exp1_data)
    logger.info(f"  Per-instance CRND: {', '.join(f'{k}={len(v)}' for k, v in per_instance.items())}")

    per_instance_ablation = extract_per_instance_crnd_ablation(exp3_it3_data)
    logger.info(f"  Per-instance ablation: {list(per_instance_ablation.keys())}")

    t1 = time.time()
    logger.info(f"Phase 1 complete in {t1-t0:.1f}s")

    # ── Phase 2: Representation Suitability Map ──────────────────────────
    logger.info("Phase 2: Computing Representation Suitability Map...")
    d_gap_results = compute_d_gap(d_matrices, class_names_map)
    for ds, res in d_gap_results.items():
        logger.info(f"  {ds}: mean_d_gap={res['mean_d_gap']:.4f}, max_d_gap={res['max_d_gap']:.4f}, n_pairs={len(res['pairs'])}")

    uniformly_high_d = compute_uniformly_high_d(d_gap_results)
    for ds in DATASETS:
        if ds in uniformly_high_d and "threshold_0.5" in uniformly_high_d[ds]:
            logger.info(f"  {ds} uniformly-high-D (>0.5): {uniformly_high_d[ds]['threshold_0.5']['count']}/{len(d_gap_results.get(ds,{}).get('pairs',{}))}")

    crnd_stats = compute_per_class_crnd_stats(exp1_meta)
    logger.info(f"  CRND stats computed for {len(crnd_stats)} datasets")

    t2 = time.time()
    logger.info(f"Phase 2 complete in {t2-t1:.1f}s")

    # ── Phase 3: Disagreement Topology ───────────────────────────────────
    logger.info("Phase 3: Computing Disagreement Topology...")
    boundary_graph = build_boundary_crnd_graph(per_instance, class_names_map)
    d_graph = build_d_similarity_graph(d_matrices, class_names_map)
    mantel_results = compute_disagreement_topology(boundary_graph, d_graph)

    t3 = time.time()
    logger.info(f"Phase 3 complete in {t3-t2:.1f}s")

    # ── Phase 4: Predictive Value ────────────────────────────────────────
    logger.info("Phase 4: Computing Predictive Value of Suitability Map...")
    kendall_results = compute_kendall_tau_d_gap_vs_f1_gap(d_gap_results, classifier_f1)
    logger.info(f"  Pooled Kendall tau: {kendall_results.get('pooled', {}).get('kendall_tau')}")

    crnd_improvement = compute_crnd_prediction_improvement(
        d_gap_results, classifier_f1, crnd_stats, class_names_map
    )
    logger.info(f"  CRND improvement delta: {crnd_improvement.get('delta_accuracy')}")

    high_overlap = compute_high_overlap_precision(d_gap_results, classifier_f1)
    logger.info(f"  High-overlap best precision: {high_overlap.get('best_precision')} at threshold={high_overlap.get('best_threshold')}")

    t4 = time.time()
    logger.info(f"Phase 4 complete in {t4-t3:.1f}s")

    # ── Phase 5: Clinical Interpretability ───────────────────────────────
    logger.info("Phase 5: Generating Clinical Interpretability Profiles...")
    clinical_profiles = generate_clinical_profiles(
        d_gap_results, crnd_stats, class_names_map, d_matrices
    )
    for ds, profiles in clinical_profiles.items():
        logger.info(f"  {ds}: {len(profiles)} class profiles generated")

    deferral_pairs, total_deferral = identify_human_deferral_pairs(
        d_gap_results, boundary_graph
    )
    logger.info(f"  Total human deferral pairs: {total_deferral}")

    t5 = time.time()
    logger.info(f"Phase 5 complete in {t5-t4:.1f}s")

    # ── Phase 6: Ecological vs. ML Overlap ───────────────────────────────
    logger.info("Phase 6: Computing Ecological vs. ML Overlap Comparison...")
    fisher_results = compute_fisher_discriminant_ratio(
        per_instance, class_names_map, d_matrices
    )
    logger.info(f"  Fisher F1 computed for {len(fisher_results)} datasets")

    bhattacharyya_results = compute_bhattacharyya_coefficient(
        per_instance, class_names_map
    )
    logger.info(f"  Bhattacharyya coefficients computed for {len(bhattacharyya_results)} datasets")

    eco_vs_ml = compute_ecological_vs_ml_correlations(
        d_gap_results, fisher_results, bhattacharyya_results, classifier_f1
    )
    logger.info(f"  Spearman D vs inv-Fisher: rho={eco_vs_ml.get('spearman_d_vs_inv_fisher', {}).get('rho')}")
    logger.info(f"  Spearman D vs Bhattacharyya: rho={eco_vs_ml.get('spearman_d_vs_bhattacharyya', {}).get('rho')}")
    logger.info(f"  Partial Spearman (D controlling Fisher): rho={eco_vs_ml.get('partial_spearman_d_controlling_fisher', {}).get('rho')}")

    t6 = time.time()
    logger.info(f"Phase 6 complete in {t6-t5:.1f}s")

    # ── Build and save output ────────────────────────────────────────────
    logger.info("Building final output...")
    output = build_output(
        d_gap_results=d_gap_results,
        uniformly_high_d=uniformly_high_d,
        crnd_stats=crnd_stats,
        mantel_results=mantel_results,
        kendall_results=kendall_results,
        crnd_improvement=crnd_improvement,
        high_overlap=high_overlap,
        clinical_profiles=clinical_profiles,
        deferral_pairs=deferral_pairs,
        total_deferral=total_deferral,
        eco_vs_ml=eco_vs_ml,
        exp1_data=exp1_data,
        classifier_f1=classifier_f1,
        class_names_map=class_names_map,
    )

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    for k, v in output["metrics_agg"].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Total runtime: {total_time:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
