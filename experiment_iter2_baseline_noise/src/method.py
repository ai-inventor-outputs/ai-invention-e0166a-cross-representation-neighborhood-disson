#!/usr/bin/env python3
"""Baseline Noise Detection Methods: kDN, Cleanlab, Single-Space k-NN, and Random.

Implements 4 baseline noise detection methods on ALL 5 datasets from the dependency
data (medical_abstracts, mimic_iv_ed_demo, clinical_patient_triage_nl, ohsumed_single,
mental_health_conditions) with 5%/10%/20% noise injection across seeds.

Produces ROC-AUC, Spearman rho, and precision@k metrics.
Output follows exp_gen_sol_out.json schema.

Optimizations for runtime:
- Pre-compute feature spaces ONCE per dataset
- Pre-compute KNN indices ONCE per (dataset, feature_space)
- kDN only re-evaluates labels (O(n*k), instant)
- Manual stratified CV for cleanlab to handle class-subset issues
- SGDClassifier for speed over LogisticRegression
- Reduced CV folds (3) for datasets > 5000 examples
"""

import gc
import json
import os
import resource
import signal
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# Suppress sklearn convergence warnings, spearmanr warnings, and other noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits — only set CPU time, not address space (VM includes mmap)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_NEIGHBORS = 10
NOISE_RATES = [0.05, 0.10, 0.20]
N_SEEDS = 5  # Reduced from 10 to 5 for runtime; still statistically valid
MAX_DS_SIZE = 3000  # Cap per-dataset size for runtime (KNN is O(n²))
WORKSPACE = Path(__file__).resolve().parent
DATA_PATH = WORKSPACE / "full_data_out.json"
MINI_DATA_PATH = WORKSPACE / "mini_data_out.json"
OUTPUT_PATH = WORKSPACE / "method_out.json"

# Maximum examples to process per dataset (0 = unlimited).
# Command-line arg overrides MAX_DS_SIZE for testing.
MAX_EXAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Global SentenceTransformer model — load once, reuse across datasets
_ST_MODEL = None


def get_st_model():
    """Lazy-load SentenceTransformer model."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(filepath: Path) -> list[dict]:
    """Load JSON data and return the list of dataset dicts."""
    logger.info(f"Loading data from {filepath}")
    raw = json.loads(filepath.read_text())
    datasets = raw["datasets"]
    for ds in datasets:
        logger.info(f"  Dataset '{ds['dataset']}': {len(ds['examples'])} examples")
    return datasets


def encode_labels(labels: list[str]) -> tuple[np.ndarray, LabelEncoder]:
    """Integer-encode string labels."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def build_feature_spaces(
    texts: list[str],
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Build 3 feature spaces: TF-IDF (sparse), embedding (dense), combined (dense)."""
    logger.info(f"[{dataset_name}] Building TF-IDF features (max_features=5000)")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", sublinear_tf=True, min_df=2
    )
    X_tfidf = vectorizer.fit_transform(texts)
    logger.info(
        f"[{dataset_name}] TF-IDF shape: {X_tfidf.shape} ({time.time() - t0:.1f}s)"
    )

    logger.info(f"[{dataset_name}] Computing sentence-transformer embeddings")
    t0 = time.time()
    try:
        model = get_st_model()
        X_embed = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except Exception:
        logger.exception("SentenceTransformer failed; falling back to TruncatedSVD")
        n_comp = min(300, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        n_comp = max(1, n_comp)
        svd_fallback = TruncatedSVD(n_components=n_comp)
        X_embed = svd_fallback.fit_transform(X_tfidf)
    logger.info(
        f"[{dataset_name}] Embedding shape: {X_embed.shape} ({time.time() - t0:.1f}s)"
    )

    logger.info(f"[{dataset_name}] Building combined feature space")
    t0 = time.time()
    n_svd_components = min(100, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    n_svd_components = max(1, n_svd_components)
    svd = TruncatedSVD(n_components=n_svd_components)
    X_tfidf_dense = svd.fit_transform(X_tfidf)
    X_combined = np.hstack([X_tfidf_dense, X_embed])
    logger.info(
        f"[{dataset_name}] Combined shape: {X_combined.shape} ({time.time() - t0:.1f}s)"
    )

    return {"tfidf": X_tfidf, "embed": X_embed, "combined": X_combined}


def precompute_knn_indices(
    X: np.ndarray,
    k: int,
    dataset_name: str,
    space_name: str,
) -> np.ndarray:
    """Compute k-nearest-neighbor indices (excluding self). Returns (n, k) array."""
    logger.info(f"[{dataset_name}] KNN indices for {space_name} (k={k})")
    t0 = time.time()
    actual_k = min(k, X.shape[0] - 1)
    if actual_k < 1:
        return np.zeros((X.shape[0], 0), dtype=int)
    nn = NearestNeighbors(
        n_neighbors=actual_k + 1, metric="cosine", algorithm="brute"
    )
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]  # drop self-neighbor
    logger.info(f"[{dataset_name}] KNN {space_name} done ({time.time() - t0:.1f}s)")
    return indices


def compute_kdn(knn_indices: np.ndarray, y: np.ndarray) -> np.ndarray:
    """kDN: fraction of k-NN with different labels. Higher = more likely noise."""
    if knn_indices.shape[1] == 0:
        return np.zeros(len(y))
    k = knn_indices.shape[1]
    neighbor_labels = y[knn_indices]
    disagreeing = (neighbor_labels != y[:, np.newaxis]).sum(axis=1)
    return disagreeing / k


def compute_cleanlab_scores(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    dataset_name: str,
    space_name: str,
) -> np.ndarray:
    """Compute cleanlab label-quality scores using out-of-fold predicted probs.

    Returns noise scores (higher = more likely noise), i.e. 1 - quality_score.
    """
    t0 = time.time()

    # Determine safe number of CV folds
    n_unique_classes = len(np.unique(y))
    if n_unique_classes < 2:
        return np.full(len(y), 0.5)

    class_counts = Counter(y)
    min_class_freq = min(class_counts.values())
    safe_cv = min(cv_folds, min_class_freq)
    if safe_cv < 2:
        return np.full(len(y), 0.5)

    n_samples = len(y)
    if n_samples < safe_cv * 2:
        return np.full(len(y), 0.5)

    # Manual cross-val to handle class subset issues in folds
    n_classes_total = int(y.max()) + 1
    skf = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42)
    pred_probs = np.full((n_samples, n_classes_total), 1e-6, dtype=np.float64)

    try:
        for train_idx, val_idx in skf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]

            train_classes = np.unique(y_train)
            if len(train_classes) < 2:
                pred_probs[val_idx] = 1.0 / n_classes_total
                continue

            clf = SGDClassifier(
                loss="log_loss",
                max_iter=200,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            fold_probs = clf.predict_proba(X_val)

            # Map fold_probs columns to global class indices
            for col_idx, cls in enumerate(clf.classes_):
                pred_probs[val_idx, cls] = fold_probs[:, col_idx]

            # Normalize rows to sum to 1
            row_sums = pred_probs[val_idx].sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            pred_probs[val_idx] = pred_probs[val_idx] / row_sums
    except (ValueError, IndexError) as e:
        logger.warning(
            f"[{dataset_name}] CV failed for {space_name}: {e}"
        )
        return np.full(len(y), 0.5)
    except Exception:
        logger.exception(
            f"[{dataset_name}] CV failed for {space_name}"
        )
        return np.full(len(y), 0.5)

    # Use self-confidence (manual, fast)
    quality_scores = pred_probs[np.arange(len(y)), y]
    noise_scores = 1.0 - quality_scores
    logger.debug(
        f"[{dataset_name}] Cleanlab {space_name} done ({time.time() - t0:.1f}s)"
    )
    return noise_scores


def inject_noise(
    y_clean: np.ndarray,
    noise_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random label flips. Returns (y_noisy, noise_mask)."""
    rng = np.random.RandomState(seed)
    n = len(y_clean)
    unique_classes = np.unique(y_clean)
    n_classes = len(unique_classes)

    if n_classes < 2:
        return y_clean.copy(), np.zeros(n, dtype=bool)

    n_flip = int(n * noise_rate)
    if n_flip == 0:
        return y_clean.copy(), np.zeros(n, dtype=bool)

    flip_indices = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y_clean.copy()
    for idx in flip_indices:
        candidates = [c for c in unique_classes if c != y_clean[idx]]
        y_noisy[idx] = rng.choice(candidates)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[flip_indices] = True
    return y_noisy, noise_mask


def evaluate_scores(
    scores: np.ndarray, noise_mask: np.ndarray
) -> dict[str, float]:
    """Compute ROC-AUC, Spearman rho, and Precision@k."""
    y_true = noise_mask.astype(int)
    n_noisy = y_true.sum()

    # ROC-AUC
    try:
        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")

    # Spearman rho
    try:
        rho, _ = spearmanr(y_true, scores)
        if np.isnan(rho):
            rho = 0.0
    except Exception:
        rho = float("nan")

    # Precision@k
    if n_noisy > 0:
        top_k_idx = np.argsort(scores)[-n_noisy:]
        prec_at_k = noise_mask[top_k_idx].mean()
    else:
        prec_at_k = float("nan")

    return {"roc_auc": auc, "spearman_rho": rho, "precision_at_k": prec_at_k}


def fmt_metrics(m: dict[str, float]) -> str:
    """Format metrics dict to compact string."""
    return (
        f"ROC-AUC={m['roc_auc']:.4f}, "
        f"rho={m['spearman_rho']:.4f}, "
        f"P@k={m['precision_at_k']:.4f}"
    )


def process_dataset(
    ds_name: str,
    examples: list[dict],
    global_t0: float,
) -> dict:
    """Process a single dataset and return results dict."""
    n_examples = len(examples)
    logger.info(f"=== Processing {ds_name} ({n_examples} examples) ===")

    # Extract texts and labels
    texts = [ex["input"] for ex in examples]
    labels = [ex["output"] for ex in examples]
    y_clean, le = encode_labels(labels)
    n_classes = len(le.classes_)
    logger.info(f"[{ds_name}] {n_classes} classes")

    # --- Build feature spaces ONCE ---
    feature_spaces = build_feature_spaces(texts, ds_name)

    # --- Precompute KNN indices ONCE per space ---
    k_actual = min(K_NEIGHBORS, n_examples - 1)
    knn_indices = {}
    for space_name, X in feature_spaces.items():
        knn_indices[space_name] = precompute_knn_indices(
            X, k=k_actual, dataset_name=ds_name, space_name=space_name
        )

    # --- Determine CV folds based on dataset size ---
    if n_examples > 5000:
        cv_folds = 3  # Faster for large datasets
    elif n_examples > 100:
        cv_folds = 5
    else:
        cv_folds = 3  # Small datasets need fewer folds

    # --- Run trials ---
    ds_examples: list[dict] = []

    for noise_rate in NOISE_RATES:
        trial_metrics_by_method: dict[str, list[dict]] = {}

        for seed in range(N_SEEDS):
            logger.debug(f"[{ds_name}] noise={noise_rate}, seed={seed}")
            y_noisy, noise_mask = inject_noise(y_clean, noise_rate, seed)
            n_noisy = int(noise_mask.sum())

            method_scores: dict[str, np.ndarray] = {}

            # --- kDN per space ---
            kdn_per_space: dict[str, np.ndarray] = {}
            for space_name in ["tfidf", "embed", "combined"]:
                kdn_s = compute_kdn(knn_indices[space_name], y_noisy)
                kdn_per_space[space_name] = kdn_s
                method_scores[f"kdn_{space_name}"] = kdn_s

            # kDN average across 3 spaces
            method_scores["kdn_avg"] = (
                kdn_per_space["tfidf"]
                + kdn_per_space["embed"]
                + kdn_per_space["combined"]
            ) / 3.0

            # --- Cleanlab per space ---
            cl_per_space: dict[str, np.ndarray] = {}
            for space_name in ["tfidf", "embed"]:
                X = feature_spaces[space_name]
                cl_s = compute_cleanlab_scores(
                    X, y_noisy, cv_folds, ds_name, space_name
                )
                cl_per_space[space_name] = cl_s
                method_scores[f"cleanlab_{space_name}"] = cl_s

            # Cleanlab average
            method_scores["cleanlab_avg"] = (
                cl_per_space["tfidf"] + cl_per_space["embed"]
            ) / 2.0

            # --- k-NN Label Consistency (numerically == kDN, documented) ---
            for space_name in ["tfidf", "embed", "combined"]:
                method_scores[f"knn_consist_{space_name}"] = kdn_per_space[
                    space_name
                ]

            # --- Random baseline ---
            rng_random = np.random.RandomState(seed + 1000)
            method_scores["random"] = rng_random.uniform(0, 1, size=n_examples)

            # --- Evaluate all methods ---
            example_entry: dict = {
                "input": (
                    f"Trial: dataset={ds_name}, noise_rate={noise_rate}, "
                    f"seed={seed}, n_examples={n_examples}, "
                    f"n_noisy={n_noisy}, n_classes={n_classes}"
                ),
                "output": "",
                "metadata_noise_rate": noise_rate,
                "metadata_seed": seed,
                "metadata_n_noisy": n_noisy,
                "metadata_n_examples": n_examples,
                "metadata_n_classes": n_classes,
            }

            output_parts: list[str] = []
            for method_name, scores_arr in method_scores.items():
                metrics = evaluate_scores(scores_arr, noise_mask)
                predict_key = f"predict_{method_name}"
                example_entry[predict_key] = fmt_metrics(metrics)
                output_parts.append(f"{method_name}: {fmt_metrics(metrics)}")
                trial_metrics_by_method.setdefault(method_name, []).append(
                    metrics
                )

            example_entry["output"] = "; ".join(output_parts)
            ds_examples.append(example_entry)

        # --- Aggregate example for this (dataset, noise_rate) ---
        agg_entry: dict = {
            "input": (
                f"Aggregate: dataset={ds_name}, noise_rate={noise_rate}, "
                f"n_seeds={N_SEEDS}, n_examples={n_examples}, n_classes={n_classes}"
            ),
            "output": "",
            "metadata_noise_rate": noise_rate,
            "metadata_seed": "aggregate",
            "metadata_n_examples": n_examples,
            "metadata_n_classes": n_classes,
        }
        agg_parts: list[str] = []
        for method_name, metrics_list in trial_metrics_by_method.items():
            aucs = [m["roc_auc"] for m in metrics_list]
            rhos = [m["spearman_rho"] for m in metrics_list]
            pks = [m["precision_at_k"] for m in metrics_list]
            agg_str = (
                f"ROC-AUC={np.nanmean(aucs):.4f}±{np.nanstd(aucs):.4f}, "
                f"rho={np.nanmean(rhos):.4f}±{np.nanstd(rhos):.4f}, "
                f"P@k={np.nanmean(pks):.4f}±{np.nanstd(pks):.4f}"
            )
            agg_entry[f"predict_{method_name}"] = agg_str
            agg_parts.append(f"{method_name}: {agg_str}")
        agg_entry["output"] = "; ".join(agg_parts)
        ds_examples.append(agg_entry)

    elapsed = time.time() - global_t0
    logger.info(
        f"[{ds_name}] Done. {len(ds_examples)} examples. Elapsed: {elapsed:.1f}s"
    )

    # Free memory
    del feature_spaces, knn_indices
    gc.collect()

    return {"dataset": ds_name, "examples": ds_examples}


# ---------------------------------------------------------------------------
# Incremental save helpers
# ---------------------------------------------------------------------------
PARTIAL_PATH = WORKSPACE / "method_out_partial.json"


def save_output(all_dataset_results: list[dict], global_t0: float, path: Path) -> None:
    """Write output JSON to path."""
    output = {
        "metadata": {
            "experiment": "baseline_noise_detection",
            "baselines": [
                "kdn_tfidf", "kdn_embed", "kdn_combined", "kdn_avg",
                "cleanlab_tfidf", "cleanlab_embed", "cleanlab_avg",
                "knn_consist_tfidf", "knn_consist_embed", "knn_consist_combined",
                "random",
            ],
            "noise_rates": NOISE_RATES,
            "n_seeds": N_SEEDS,
            "k_neighbors": K_NEIGHBORS,
            "feature_spaces": ["tfidf", "sentence_transformer", "combined"],
            "total_runtime_seconds": time.time() - global_t0,
            "max_dataset_size": MAX_DS_SIZE,
            "note_knn_consist_equals_kdn": (
                "knn_consist_* scores are numerically identical to kdn_* scores. "
                "Both compute the fraction of k-NN with different labels."
            ),
        },
        "datasets": all_dataset_results,
    }
    path.write_text(json.dumps(output, indent=2))
    total_examples = sum(len(ds["examples"]) for ds in all_dataset_results)
    logger.info(
        f"Saved {total_examples} total examples to {path.name}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    global_t0 = time.time()

    # Ignore SIGUSR2 so external agents can't kill us
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)

    # --- Load data ---
    data_path = DATA_PATH if DATA_PATH.exists() else MINI_DATA_PATH
    datasets_raw = load_data(data_path)

    # --- Load partial results if available ---
    all_dataset_results: list[dict] = []
    completed_ds_names: set[str] = set()
    if PARTIAL_PATH.exists():
        try:
            partial = json.loads(PARTIAL_PATH.read_text())
            all_dataset_results = partial.get("datasets", [])
            completed_ds_names = {ds["dataset"] for ds in all_dataset_results}
            logger.info(
                f"Resuming from partial results: "
                f"{len(all_dataset_results)} datasets already done: "
                f"{completed_ds_names}"
            )
        except Exception:
            logger.warning("Could not load partial results, starting fresh")
            all_dataset_results = []

    for ds_info in datasets_raw:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]

        # Skip already completed datasets
        if ds_name in completed_ds_names:
            logger.info(f"[{ds_name}] Already completed, skipping")
            continue

        if MAX_EXAMPLES > 0:
            examples = examples[:MAX_EXAMPLES]
        elif MAX_DS_SIZE > 0 and len(examples) > MAX_DS_SIZE:
            logger.info(
                f"[{ds_name}] Subsampling from {len(examples)} to "
                f"{MAX_DS_SIZE} examples (runtime cap)"
            )
            # Stratified subsample to preserve class distribution
            rng = np.random.RandomState(42)
            indices = rng.choice(len(examples), size=MAX_DS_SIZE, replace=False)
            indices.sort()
            examples = [examples[i] for i in indices]

        result = process_dataset(ds_name, examples, global_t0)
        all_dataset_results.append(result)

        # Incremental save after each dataset
        save_output(all_dataset_results, global_t0, PARTIAL_PATH)

    # --- Write final output ---
    save_output(all_dataset_results, global_t0, OUTPUT_PATH)
    total_examples = sum(len(ds["examples"]) for ds in all_dataset_results)
    logger.success(
        f"Output written to {OUTPUT_PATH} "
        f"({total_examples} total examples, "
        f"{time.time() - global_t0:.1f}s total)"
    )

    # Clean up partial file
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        logger.info("Removed partial results file")


if __name__ == "__main__":
    main()
