#!/usr/bin/env python3
"""Cross-Representation Method Selection Validation with True LLM Features.

Re-runs method selection validation with three genuine representation families
(TF-IDF, sentence-transformer, LLM zero-shot via OpenRouter) to test whether
Schoener's D niche overlap profiles predict classifier rank-ordering better
than iteration 2's Kendall's tau=0.24.

Key innovation: extract LLM features via OpenRouter /chat/completions endpoint
with logprobs for probability vectors over class labels.
"""

import json
import math
import os
import resource
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger
from scipy.stats import gaussian_kde, kendalltau
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, label_binarize, normalize
from sklearn.svm import LinearSVC

# ──────────────────────────────────────────────────────────────
# Resource limits (14 GB RAM, 1h CPU)
# ──────────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
load_dotenv("/home/adrian/projects/ai-inventor/.env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"

DATA_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)

# ──── Scaling control ────
# Set via env var or default. Use MAX_EXAMPLES_PER_DATASET=0 for full dataset.
MAX_EXAMPLES_PER_DATASET = int(os.environ.get("MAX_EXAMPLES", "0"))

K_VALUES = [10, 20]
N_BOOTSTRAP = 1000
BATCH_DELAY = 0.15  # seconds between LLM calls

# Budget tracking
LLM_CALL_COUNT = 0
LLM_TOTAL_COST = 0.0
LLM_COST_LIMIT = 9.50  # stop before $10
LLM_CALL_LIMIT = 9500  # stop before 10k


# ══════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & SAMPLING
# ══════════════════════════════════════════════════════════════

def load_and_sample_data(
    data_path: Path,
    max_per_dataset: int = 0,
) -> dict:
    """Load full_data_out.json and stratified-sample each dataset."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets_out = {}

    for ds in raw["datasets"]:
        ds_name = ds["dataset"]
        examples = ds["examples"]
        texts = [ex["input"] for ex in examples]
        labels = np.array([ex["output"] for ex in examples])
        unique_labels = sorted(set(labels))
        n_total = len(texts)

        if max_per_dataset > 0 and n_total > max_per_dataset:
            # Stratified sample
            rng = np.random.default_rng(42)
            indices = []
            for lbl in unique_labels:
                lbl_idx = np.where(labels == lbl)[0]
                n_take = max(1, int(max_per_dataset * len(lbl_idx) / n_total))
                n_take = min(n_take, len(lbl_idx))
                chosen = rng.choice(lbl_idx, size=n_take, replace=False)
                indices.extend(chosen.tolist())
            indices = sorted(indices)[:max_per_dataset]
        else:
            indices = list(range(n_total))

        sampled_texts = [texts[i] for i in indices]
        sampled_labels = labels[indices]

        datasets_out[ds_name] = {
            "texts": sampled_texts,
            "labels": sampled_labels,
            "unique_labels": sorted(set(sampled_labels)),
            "examples": [examples[i] for i in indices],
            "indices": indices,
        }
        logger.info(
            f"  {ds_name}: {len(sampled_texts)} examples, "
            f"{len(datasets_out[ds_name]['unique_labels'])} classes"
        )

    return datasets_out


# ══════════════════════════════════════════════════════════════
# PHASE 2A: TF-IDF Features
# ══════════════════════════════════════════════════════════════

def build_tfidf_features(texts: list[str]) -> np.ndarray:
    """Build TF-IDF feature matrix (lexical representation)."""
    logger.info(f"Building TF-IDF features for {len(texts)} texts")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    logger.info(f"  TF-IDF shape: {X.shape}")
    return X


# ══════════════════════════════════════════════════════════════
# PHASE 2B: Sentence-Transformer Features
# ══════════════════════════════════════════════════════════════

def build_sbert_features(texts: list[str]) -> np.ndarray:
    """Build sentence-transformer embeddings (semantic representation)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Building SBERT features for {len(texts)} texts")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, batch_size=64, show_progress_bar=True)
    logger.info(f"  SBERT shape: {X.shape}")
    return X


# ══════════════════════════════════════════════════════════════
# PHASE 2C: LLM Zero-Shot Probability Features
# ══════════════════════════════════════════════════════════════

def fallback_one_hot(
    response_text: str,
    class_labels: list[str],
) -> np.ndarray:
    """Create smoothed one-hot vector from text response."""
    prob_vector = np.ones(len(class_labels)) * 0.01
    response_lower = response_text.lower().strip()
    matched = False
    for idx, label in enumerate(class_labels):
        if label.lower() in response_lower or response_lower in label.lower():
            prob_vector[idx] = 1.0
            matched = True
            break
    if not matched:
        # Try partial matching
        for idx, label in enumerate(class_labels):
            label_parts = label.lower().replace("_", " ").split()
            for part in label_parts:
                if len(part) > 3 and part in response_lower:
                    prob_vector[idx] = 1.0
                    matched = True
                    break
            if matched:
                break
    prob_vector /= prob_vector.sum()
    return prob_vector


def get_llm_features(
    text: str,
    class_labels: list[str],
    max_retries: int = 3,
) -> tuple[np.ndarray, bool, float]:
    """Get LLM zero-shot classification probability vector.

    Returns: (prob_vector, logprobs_available, cost_usd)
    """
    global LLM_CALL_COUNT, LLM_TOTAL_COST

    # Budget check
    if LLM_CALL_COUNT >= LLM_CALL_LIMIT:
        logger.warning("LLM call limit reached, returning uniform")
        return np.ones(len(class_labels)) / len(class_labels), False, 0.0
    if LLM_TOTAL_COST >= LLM_COST_LIMIT:
        logger.warning("LLM cost limit reached, returning uniform")
        return np.ones(len(class_labels)) / len(class_labels), False, 0.0

    labels_str = ", ".join([f'"{c}"' for c in class_labels])
    prompt = (
        f"Classify the following text into exactly one of these categories: "
        f"{labels_str}.\n"
        f"Respond with ONLY the category name, nothing else.\n\n"
        f"Text: {text[:2000]}\n\n"
        f"Category:"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }

    for attempt in range(max_retries):
        try:
            LLM_CALL_COUNT += 1
            resp = requests.post(
                CHAT_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                wait_time = 2 ** (attempt + 1)
                logger.warning(f"Rate limited (429), waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Estimate cost (input + output tokens)
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # Llama 3.1 8B pricing: ~$0.06/M input, ~$0.06/M output
            call_cost = (input_tokens + output_tokens) * 0.06 / 1_000_000
            LLM_TOTAL_COST += call_cost

            choice = data["choices"][0]
            response_text = choice["message"]["content"].strip()

            # Try to extract logprobs
            logprobs_data = choice.get("logprobs")
            if logprobs_data and isinstance(logprobs_data, dict):
                content_logprobs = logprobs_data.get("content", [])
            else:
                content_logprobs = []

            if content_logprobs:
                first_token_logprobs = content_logprobs[0].get("top_logprobs", [])
                if first_token_logprobs:
                    prob_vector = np.zeros(len(class_labels))
                    for tlp in first_token_logprobs:
                        token = tlp["token"].strip().lower()
                        logprob = tlp["logprob"]
                        prob = math.exp(logprob)
                        for idx, label in enumerate(class_labels):
                            label_lower = label.lower()
                            label_parts = label_lower.replace("_", " ").split()
                            if (
                                token in label_lower
                                or label_lower.startswith(token)
                                or any(
                                    part.startswith(token)
                                    for part in label_parts
                                    if len(token) > 1
                                )
                            ):
                                prob_vector[idx] += prob
                                break

                    if prob_vector.sum() > 0.01:
                        prob_vector /= prob_vector.sum() + 1e-10
                        return prob_vector, True, call_cost
                    else:
                        return fallback_one_hot(response_text, class_labels), False, call_cost
                else:
                    return fallback_one_hot(response_text, class_labels), False, call_cost
            else:
                return fallback_one_hot(response_text, class_labels), False, call_cost

        except requests.exceptions.Timeout:
            logger.warning(f"LLM call timeout (attempt {attempt + 1})")
            time.sleep(2 ** (attempt + 1))
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** (attempt + 1))
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning(f"LLM response parse error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** (attempt + 1))

    # Final fallback: uniform distribution
    return np.ones(len(class_labels)) / len(class_labels), False, 0.0


def build_llm_features(
    texts: list[str],
    class_labels: list[str],
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, bool]:
    """Build LLM zero-shot probability features for all texts.

    Returns: (X_llm, logprobs_available)
    """
    n = len(texts)
    n_classes = len(class_labels)
    X_llm = np.zeros((n, n_classes))
    logprobs_count = 0

    # Try to resume from checkpoint
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = np.load(str(checkpoint_path))
            ckpt_X = ckpt["X_llm"]
            ckpt_last = int(ckpt["last_idx"])
            # Only resume if checkpoint matches current dimensions
            if ckpt_X.shape[0] == n and ckpt_X.shape[1] == n_classes:
                X_llm = ckpt_X
                start_idx = ckpt_last + 1
                logprobs_count = int(ckpt.get("logprobs_count", 0))
                logger.info(f"Resuming LLM features from index {start_idx}")
            else:
                logger.warning(
                    f"Checkpoint shape {ckpt_X.shape} doesn't match "
                    f"expected ({n}, {n_classes}), starting fresh"
                )
                start_idx = 0
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            start_idx = 0

    logger.info(
        f"Building LLM features for {n} texts "
        f"(starting at {start_idx}, {n_classes} classes)"
    )

    for i in range(start_idx, n):
        X_llm[i], lp_avail, _ = get_llm_features(texts[i], class_labels)
        if lp_avail:
            logprobs_count += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"LLM features: {i + 1}/{n} done | "
                f"logprobs: {logprobs_count}/{i + 1} | "
                f"cost: ${LLM_TOTAL_COST:.4f} | "
                f"calls: {LLM_CALL_COUNT}"
            )
            # Save checkpoint
            if checkpoint_path:
                np.savez(
                    str(checkpoint_path),
                    X_llm=X_llm,
                    last_idx=i,
                    logprobs_count=logprobs_count,
                )

        time.sleep(BATCH_DELAY)

    # Final checkpoint save
    if checkpoint_path:
        np.savez(
            str(checkpoint_path),
            X_llm=X_llm,
            last_idx=n - 1,
            logprobs_count=logprobs_count,
        )

    logprobs_available = logprobs_count > (n * 0.1)  # >10% had logprobs
    logger.info(
        f"LLM features complete: {n} texts, "
        f"logprobs available: {logprobs_available} "
        f"({logprobs_count}/{n})"
    )
    return X_llm, logprobs_available


# ══════════════════════════════════════════════════════════════
# PHASE 3: k-NN COMPUTATION AND CRND
# ══════════════════════════════════════════════════════════════

def compute_knn_and_crnd(
    feature_spaces: dict[str, np.ndarray],
    k_values: list[int],
) -> dict:
    """Compute k-NN neighbors and CRND for each feature space.

    Returns dict with:
      - neighbors: {k: {space_name: array of neighbor indices}}
      - crnd: {k: array of CRND values per instance}
    """
    results = {"neighbors": {}, "crnd": {}}
    n_instances = next(iter(feature_spaces.values())).shape[0]
    space_names = list(feature_spaces.keys())

    for k in k_values:
        effective_k = min(k, n_instances - 1)
        if effective_k < 2:
            logger.warning(f"k={k} too large for {n_instances} instances, skipping")
            results["neighbors"][k] = {}
            results["crnd"][k] = np.zeros(n_instances)
            continue

        logger.info(f"Computing k-NN with k={effective_k}")
        neighbors_k = {}
        for space_name, X in feature_spaces.items():
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            X_norm = normalize(X_dense)
            nn = NearestNeighbors(
                n_neighbors=effective_k + 1,
                metric="cosine",
            )
            nn.fit(X_norm)
            _, indices = nn.kneighbors(X_norm)
            neighbors_k[space_name] = indices[:, 1:]  # exclude self
            logger.info(f"  {space_name}: neighbors shape {neighbors_k[space_name].shape}")

        results["neighbors"][k] = neighbors_k

        # Compute CRND (1 - avg pairwise Jaccard)
        space_pairs = list(combinations(space_names, 2))
        crnd = np.zeros(n_instances)
        for i in range(n_instances):
            jaccard_sum = 0.0
            for s1, s2 in space_pairs:
                set1 = set(neighbors_k[s1][i].tolist())
                set2 = set(neighbors_k[s2][i].tolist())
                union_size = len(set1 | set2)
                if union_size > 0:
                    jaccard = len(set1 & set2) / union_size
                else:
                    jaccard = 1.0
                jaccard_sum += jaccard
            crnd[i] = 1.0 - jaccard_sum / max(len(space_pairs), 1)

        results["crnd"][k] = crnd
        logger.info(
            f"  CRND k={effective_k}: mean={crnd.mean():.4f}, "
            f"std={crnd.std():.4f}"
        )

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 4: ECOLOGICAL NICHE OVERLAP (SCHOENER'S D)
# ══════════════════════════════════════════════════════════════

def compute_schoeners_d(
    X: np.ndarray,
    labels: np.ndarray,
    class_a: str,
    class_b: str,
    grid_size: int = 100,
) -> float:
    """Compute Schoener's D between two classes in a feature space.

    Uses PCA-env framework: project to 2D PCA, then KDE on a grid.
    D = 1 - 0.5 * sum(|p1 - p2|) where p1, p2 are normalized density grids.
    """
    mask_a = labels == class_a
    mask_b = labels == class_b
    X_a = X[mask_a]
    X_b = X[mask_b]

    if len(X_a) < 5 or len(X_b) < 5:
        return float("nan")

    # PCA to 2D
    pca = PCA(n_components=min(2, X_a.shape[1], X_b.shape[1]))
    X_all = np.vstack([X_a, X_b])
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()

    try:
        X_pca = pca.fit_transform(X_all)
    except Exception:
        return float("nan")

    if X_pca.shape[1] < 2:
        # Only 1D available, use 1D KDE
        X_a_pca = X_pca[: len(X_a)].ravel()
        X_b_pca = X_pca[len(X_a) :].ravel()
        try:
            kde_a = gaussian_kde(X_a_pca, bw_method="scott")
            kde_b = gaussian_kde(X_b_pca, bw_method="scott")
        except np.linalg.LinAlgError:
            return float("nan")
        x_min = min(X_a_pca.min(), X_b_pca.min()) - 1
        x_max = max(X_a_pca.max(), X_b_pca.max()) + 1
        grid = np.linspace(x_min, x_max, grid_size)
        density_a = kde_a(grid)
        density_b = kde_b(grid)
        density_a /= density_a.sum() + 1e-10
        density_b /= density_b.sum() + 1e-10
        D = 1.0 - 0.5 * np.sum(np.abs(density_a - density_b))
        return float(np.clip(D, 0.0, 1.0))

    X_a_pca = X_pca[: len(X_a)]
    X_b_pca = X_pca[len(X_a) :]

    # Add jitter to avoid singular matrices
    jitter = np.random.default_rng(42).normal(0, 1e-6, X_a_pca.shape)
    X_a_pca = X_a_pca + jitter[: len(X_a_pca)]
    jitter_b = np.random.default_rng(43).normal(0, 1e-6, X_b_pca.shape)
    X_b_pca = X_b_pca + jitter_b[: len(X_b_pca)]

    x_min = X_pca[:, 0].min() - 1
    x_max = X_pca[:, 0].max() + 1
    y_min = X_pca[:, 1].min() - 1
    y_max = X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kde_a = gaussian_kde(X_a_pca.T, bw_method="scott")
        kde_b = gaussian_kde(X_b_pca.T, bw_method="scott")
    except np.linalg.LinAlgError:
        # Fallback: histogram-based overlap
        return _histogram_overlap(X_a_pca, X_b_pca, bins=50)

    density_a = kde_a(grid_points)
    density_b = kde_b(grid_points)

    density_a /= density_a.sum() + 1e-10
    density_b /= density_b.sum() + 1e-10

    D = 1.0 - 0.5 * np.sum(np.abs(density_a - density_b))
    return float(np.clip(D, 0.0, 1.0))


def _histogram_overlap(
    X_a: np.ndarray,
    X_b: np.ndarray,
    bins: int = 50,
) -> float:
    """Fallback: histogram-based overlap when KDE fails."""
    X_all = np.vstack([X_a, X_b])
    x_range = (X_all[:, 0].min() - 1, X_all[:, 0].max() + 1)
    y_range = (X_all[:, 1].min() - 1, X_all[:, 1].max() + 1)

    hist_a, _, _ = np.histogram2d(
        X_a[:, 0], X_a[:, 1], bins=bins, range=[x_range, y_range]
    )
    hist_b, _, _ = np.histogram2d(
        X_b[:, 0], X_b[:, 1], bins=bins, range=[x_range, y_range]
    )

    hist_a = hist_a / (hist_a.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)

    D = 1.0 - 0.5 * np.sum(np.abs(hist_a - hist_b))
    return float(np.clip(D, 0.0, 1.0))


def compute_niche_overlap_all(
    feature_spaces: dict[str, np.ndarray],
    labels: np.ndarray,
    class_pairs: list[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute Schoener's D for all class pairs in each feature space."""
    niche_overlap = {}
    for space_name, X in feature_spaces.items():
        logger.info(f"Computing niche overlap for {space_name} ({len(class_pairs)} pairs)")
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        space_d = {}
        for ca, cb in class_pairs:
            D = compute_schoeners_d(X_dense, labels, ca, cb)
            space_d[(ca, cb)] = D
        niche_overlap[space_name] = space_d
        valid = [v for v in space_d.values() if not np.isnan(v)]
        if valid:
            logger.info(
                f"  {space_name}: mean D={np.mean(valid):.4f}, "
                f"valid pairs={len(valid)}/{len(class_pairs)}"
            )
    return niche_overlap


# ══════════════════════════════════════════════════════════════
# PHASE 5: CLASSIFIER TRAINING (per feature space)
# ══════════════════════════════════════════════════════════════

def train_classifiers_ovo(
    feature_spaces: dict[str, np.ndarray],
    labels: np.ndarray,
    class_pairs: list[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], float]]:
    """For each feature space × class pair, compute best OvO binary F1."""
    classifiers = {
        "logreg": LogisticRegression(
            max_iter=1000, C=1.0, solver="saga",
        ),
        "svm": LinearSVC(max_iter=5000, C=1.0, dual=False),
        "gb": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    results = {}

    for space_name, X in feature_spaces.items():
        logger.info(f"Training classifiers for {space_name}")
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        space_f1s = {}

        for ca, cb in class_pairs:
            mask = (labels == ca) | (labels == cb)
            X_pair = X_dense[mask]
            y_pair = (labels[mask] == ca).astype(int)

            if len(X_pair) < 10:
                space_f1s[(ca, cb)] = float("nan")
                continue

            n_splits = min(5, min(np.sum(y_pair == 0), np.sum(y_pair == 1)))
            if n_splits < 2:
                space_f1s[(ca, cb)] = float("nan")
                continue

            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42
            )

            best_f1_for_pair = 0.0
            for clf_name, clf in classifiers.items():
                fold_f1s = []
                for train_idx, test_idx in skf.split(X_pair, y_pair):
                    try:
                        clf_copy = clone(clf)
                        clf_copy.fit(X_pair[train_idx], y_pair[train_idx])
                        y_pred = clf_copy.predict(X_pair[test_idx])
                        fold_f1s.append(
                            f1_score(y_pair[test_idx], y_pred, average="binary")
                        )
                    except Exception:
                        fold_f1s.append(0.0)
                mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.0
                best_f1_for_pair = max(best_f1_for_pair, mean_f1)

            space_f1s[(ca, cb)] = best_f1_for_pair

        results[space_name] = space_f1s
        valid = [v for v in space_f1s.values() if not np.isnan(v)]
        if valid:
            logger.info(
                f"  {space_name}: mean best F1={np.mean(valid):.4f}, "
                f"valid pairs={len(valid)}/{len(class_pairs)}"
            )

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 5b: BASELINE — Multi-class F1 per space (no niche overlap)
# ══════════════════════════════════════════════════════════════

def train_baseline_multiclass(
    feature_spaces: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Baseline: standard multi-class classification per space.

    Returns {space_name: {"macro_f1": float, "micro_f1": float}}
    This is a simpler baseline that just picks the best-performing space
    overall (no per-pair niche overlap analysis).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    baseline_results = {}
    for space_name, X in feature_spaces.items():
        logger.info(f"Baseline multi-class classification for {space_name}")
        X_dense = X.toarray() if hasattr(X, "toarray") else X

        n_splits = min(5, min(np.bincount(y)))
        if n_splits < 2:
            baseline_results[space_name] = {
                "macro_f1": float("nan"),
                "micro_f1": float("nan"),
            }
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="saga")

        fold_macro = []
        fold_micro = []
        for train_idx, test_idx in skf.split(X_dense, y):
            try:
                clf_copy = clone(clf)
                clf_copy.fit(X_dense[train_idx], y[train_idx])
                y_pred = clf_copy.predict(X_dense[test_idx])
                fold_macro.append(f1_score(y[test_idx], y_pred, average="macro"))
                fold_micro.append(f1_score(y[test_idx], y_pred, average="micro"))
            except Exception:
                pass

        baseline_results[space_name] = {
            "macro_f1": float(np.mean(fold_macro)) if fold_macro else float("nan"),
            "micro_f1": float(np.mean(fold_micro)) if fold_micro else float("nan"),
        }
        logger.info(
            f"  {space_name}: macro_f1={baseline_results[space_name]['macro_f1']:.4f}"
        )

    return baseline_results


# ══════════════════════════════════════════════════════════════
# PHASE 6: METHOD SELECTION TEST (Kendall's τ)
# ══════════════════════════════════════════════════════════════

def compute_kendall_tau(
    niche_overlap: dict[str, dict[tuple[str, str], float]],
    classifier_results: dict[str, dict[tuple[str, str], float]],
    class_pairs: list[tuple[str, str]],
    space_names: list[str],
) -> dict:
    """Compute Kendall's τ between predicted (niche overlap) and actual (F1) rankings."""
    actual_ranks = []
    predicted_ranks = []
    valid_pairs = []

    for ca, cb in class_pairs:
        f1_values = [
            classifier_results[s].get((ca, cb), float("nan"))
            for s in space_names
        ]
        d_values = [
            niche_overlap[s].get((ca, cb), float("nan"))
            for s in space_names
        ]

        if any(np.isnan(v) for v in f1_values + d_values):
            continue

        # Rank by F1 (higher = better → argsort descending)
        actual_rank = np.argsort(-np.array(f1_values)).tolist()
        # Rank by D (lower overlap = predicted better → argsort ascending)
        predicted_rank = np.argsort(np.array(d_values)).tolist()

        actual_ranks.append(actual_rank)
        predicted_ranks.append(predicted_rank)
        valid_pairs.append((ca, cb))

    if len(actual_ranks) < 3:
        logger.warning(f"Too few valid pairs ({len(actual_ranks)}) for Kendall's τ")
        return {
            "tau": float("nan"),
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_pairs": len(actual_ranks),
        }

    # Flatten for Kendall's tau
    actual_flat = [r for ranks in actual_ranks for r in ranks]
    predicted_flat = [r for ranks in predicted_ranks for r in ranks]

    tau, p_value = kendalltau(actual_flat, predicted_flat)

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    n_pairs = len(actual_ranks)
    tau_samples = []
    for _ in range(N_BOOTSTRAP):
        boot_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        a_flat = [r for i in boot_idx for r in actual_ranks[i]]
        p_flat = [r for i in boot_idx for r in predicted_ranks[i]]
        bt, _ = kendalltau(a_flat, p_flat)
        if not np.isnan(bt):
            tau_samples.append(bt)

    ci_lower = float(np.percentile(tau_samples, 2.5)) if tau_samples else float("nan")
    ci_upper = float(np.percentile(tau_samples, 97.5)) if tau_samples else float("nan")

    logger.info(
        f"Kendall's τ = {tau:.4f} (p={p_value:.6f}), "
        f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
        f"n_pairs = {n_pairs}"
    )

    return {
        "tau": float(tau) if not np.isnan(tau) else float("nan"),
        "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_pairs": n_pairs,
        "valid_pairs": [(ca, cb) for ca, cb in valid_pairs],
    }


# ══════════════════════════════════════════════════════════════
# PHASE 7: META-PREDICTOR (Niche Overlap → Best Space)
# ══════════════════════════════════════════════════════════════

def train_meta_predictor(
    niche_overlap: dict[str, dict[tuple[str, str], float]],
    classifier_results: dict[str, dict[tuple[str, str], float]],
    class_pairs: list[tuple[str, str]],
    space_names: list[str],
) -> dict:
    """Train a meta-predictor: niche overlap features → predict best space."""
    meta_X = []
    meta_y = []

    for ca, cb in class_pairs:
        d_vec = [
            niche_overlap[s].get((ca, cb), float("nan"))
            for s in space_names
        ]
        f1_vec = [
            classifier_results[s].get((ca, cb), float("nan"))
            for s in space_names
        ]

        if any(np.isnan(v) for v in d_vec + f1_vec):
            continue

        meta_X.append(d_vec)
        meta_y.append(int(np.argmax(f1_vec)))

    meta_X = np.array(meta_X) if meta_X else np.empty((0, len(space_names)))
    meta_y = np.array(meta_y) if meta_y else np.empty(0, dtype=int)

    if len(meta_X) < 10:
        logger.warning(f"Too few samples ({len(meta_X)}) for meta-predictor")
        return {"auc": float("nan"), "n_samples": len(meta_X)}

    n_unique = len(np.unique(meta_y))
    if n_unique < 2:
        logger.warning("Only one class in meta-predictor targets")
        return {"auc": float("nan"), "n_samples": len(meta_X)}

    try:
        meta_clf = LogisticRegression(max_iter=1000)
        cv_folds = min(5, len(meta_X), min(np.bincount(meta_y)))
        if cv_folds < 2:
            cv_folds = 2

        meta_probs = cross_val_predict(
            meta_clf,
            meta_X,
            meta_y,
            cv=cv_folds,
            method="predict_proba",
        )

        all_classes = sorted(np.unique(meta_y))
        if len(all_classes) == 2:
            auc = roc_auc_score(meta_y, meta_probs[:, 1])
        else:
            y_bin = label_binarize(meta_y, classes=list(range(len(space_names))))
            # Trim y_bin to match actual predictions
            auc = roc_auc_score(
                y_bin[:, all_classes],
                meta_probs,
                average="macro",
                multi_class="ovr",
            )
    except Exception as e:
        logger.warning(f"Meta-predictor AUC failed: {e}")
        auc = float("nan")

    logger.info(f"Meta-predictor AUC = {auc:.4f}, n_samples = {len(meta_X)}")
    return {"auc": float(auc), "n_samples": len(meta_X)}


# ══════════════════════════════════════════════════════════════
# PHASE 8: ASSEMBLE OUTPUT
# ══════════════════════════════════════════════════════════════

def assemble_output(
    datasets_data: dict,
    knn_results: dict,
    niche_overlap: dict[str, dict[tuple[str, str], float]],
    classifier_results: dict[str, dict[tuple[str, str], float]],
    baseline_results: dict[str, dict[str, float]],
    tau_result: dict,
    meta_result: dict,
    per_dataset_tau: dict[str, dict],
    logprobs_available: bool,
    space_names: list[str],
) -> dict:
    """Assemble the final output JSON matching exp_gen_sol_out schema."""
    output = {"datasets": []}

    for ds_name, ds_data in datasets_data.items():
        labels = ds_data["labels"]
        unique_labels = ds_data["unique_labels"]
        class_pairs = list(combinations(unique_labels, 2))
        n = len(ds_data["texts"])

        ds_examples = []
        for i in range(n):
            # Determine which class pair this instance belongs to
            inst_label = labels[i]
            # Find the best predicted and actual spaces for class pairs involving this label
            relevant_pairs = [
                (ca, cb) for ca, cb in class_pairs if ca == inst_label or cb == inst_label
            ]

            # Aggregate predicted best space
            pred_space_votes = {s: 0.0 for s in space_names}
            actual_space_votes = {s: 0.0 for s in space_names}

            for ca, cb in relevant_pairs:
                d_vals = [
                    niche_overlap.get(s, {}).get((ca, cb), float("nan"))
                    for s in space_names
                ]
                f1_vals = [
                    classifier_results.get(s, {}).get((ca, cb), float("nan"))
                    for s in space_names
                ]
                if not any(np.isnan(v) for v in d_vals):
                    pred_best_idx = int(np.argmin(d_vals))  # lowest D = best predicted
                    pred_space_votes[space_names[pred_best_idx]] += 1
                if not any(np.isnan(v) for v in f1_vals):
                    actual_best_idx = int(np.argmax(f1_vals))
                    actual_space_votes[space_names[actual_best_idx]] += 1

            pred_best = max(pred_space_votes, key=pred_space_votes.get) if any(
                v > 0 for v in pred_space_votes.values()
            ) else "unknown"
            actual_best = max(actual_space_votes, key=actual_space_votes.get) if any(
                v > 0 for v in actual_space_votes.values()
            ) else "unknown"

            example = {
                "input": ds_data["texts"][i][:500],
                "output": str(labels[i]),
            }

            # Add metadata for k-NN neighbors
            for k in K_VALUES:
                neighbors = knn_results.get("neighbors", {}).get(k, {})
                for s in space_names:
                    nn_key = f"metadata_{s}_nn_k{k}"
                    if s in neighbors and i < len(neighbors[s]):
                        example[nn_key] = json.dumps(neighbors[s][i].tolist())
                    else:
                        example[nn_key] = "[]"

            # CRND metadata
            for k in K_VALUES:
                crnd_arr = knn_results.get("crnd", {}).get(k)
                if crnd_arr is not None and i < len(crnd_arr):
                    example[f"metadata_crnd_k{k}"] = str(round(float(crnd_arr[i]), 4))
                else:
                    example[f"metadata_crnd_k{k}"] = "0.0"

            # Predictions
            example["predict_method_selection"] = pred_best
            example["predict_actual_best"] = actual_best

            ds_examples.append(example)

        output["datasets"].append({
            "dataset": ds_name,
            "examples": ds_examples,
        })

    # Metadata
    tau = tau_result.get("tau", float("nan"))
    ci_lo = tau_result.get("ci_lower", float("nan"))
    ci_hi = tau_result.get("ci_upper", float("nan"))
    p_val = tau_result.get("p_value", float("nan"))
    meta_auc = meta_result.get("auc", float("nan"))

    # Per-dataset tau dict for JSON
    per_ds_tau_json = {}
    for ds_name, ds_tau in per_dataset_tau.items():
        per_ds_tau_json[ds_name] = {
            "tau": round(ds_tau.get("tau", float("nan")), 4)
            if not np.isnan(ds_tau.get("tau", float("nan")))
            else "N/A",
            "n_pairs": ds_tau.get("n_pairs", 0),
        }

    # Baseline results for JSON
    baseline_json = {}
    for s, br in baseline_results.items():
        baseline_json[s] = {
            k: round(v, 4) if not np.isnan(v) else "N/A"
            for k, v in br.items()
        }

    output["metadata"] = {
        "method_name": "Cross-Representation Method Selection via Niche Overlap (3-family)",
        "description": (
            "Validates whether Schoener's D niche overlap profiles across "
            "TF-IDF, SBERT, and LLM feature spaces predict classifier rank-ordering"
        ),
        "kendall_tau_pooled": round(tau, 4) if not np.isnan(tau) else "N/A",
        "kendall_tau_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else "N/A",
        "kendall_tau_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else "N/A",
        "kendall_tau_p_value": round(p_val, 6) if not np.isnan(p_val) else "N/A",
        "iteration2_tau_baseline": 0.24,
        "meta_predictor_auc": round(meta_auc, 4) if not np.isnan(meta_auc) else "N/A",
        "n_class_pairs_evaluated": tau_result.get("n_pairs", 0),
        "feature_spaces": [
            "tfidf_5000_unigram",
            "sbert_miniLM_384d",
            "llm_llama31_8b_zeroshot",
        ],
        "classifiers_used": ["logistic_regression", "linear_svc", "gradient_boosting"],
        "k_values": K_VALUES,
        "per_dataset_tau": per_ds_tau_json,
        "baseline_multiclass_f1": baseline_json,
        "llm_model": LLM_MODEL,
        "llm_logprobs_available": logprobs_available,
        "llm_total_calls": LLM_CALL_COUNT,
        "llm_total_cost_usd": round(LLM_TOTAL_COST, 4),
        "n_bootstrap": N_BOOTSTRAP,
    }

    return output


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Cross-Representation Method Selection Validation (Iter 3)")
    logger.info("=" * 60)

    # ── Load data ──
    max_examples = MAX_EXAMPLES_PER_DATASET
    if max_examples > 0:
        logger.info(f"MAX_EXAMPLES_PER_DATASET = {max_examples}")
    else:
        logger.info("Running on ALL available data (no limit)")
        max_examples = 1500  # Cap at 1500 as per plan for LLM budget

    datasets_data = load_and_sample_data(
        data_path=DATA_PATH,
        max_per_dataset=max_examples,
    )

    space_names = ["tfidf", "sbert", "llm"]

    # ── Process each dataset ──
    all_niche_overlap = {}
    all_classifier_results = {}
    all_baseline_results = {}
    all_knn_results = {}
    per_dataset_tau = {}
    global_logprobs = False

    for ds_name, ds_data in datasets_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {ds_name} ({len(ds_data['texts'])} examples)")
        logger.info(f"{'='*60}")

        texts = ds_data["texts"]
        labels = ds_data["labels"]
        unique_labels = ds_data["unique_labels"]
        class_pairs = list(combinations(unique_labels, 2))

        if len(unique_labels) < 2:
            logger.warning(f"Skipping {ds_name}: only {len(unique_labels)} class(es)")
            continue

        logger.info(f"Class pairs: {len(class_pairs)}")

        # ── Phase 2: Build feature spaces ──
        X_tfidf = build_tfidf_features(texts)
        X_sbert = build_sbert_features(texts)

        ckpt_path = WORKSPACE / f"llm_ckpt_{ds_name}.npz"
        X_llm, logprobs_available = build_llm_features(
            texts=texts,
            class_labels=unique_labels,
            checkpoint_path=ckpt_path,
        )
        if logprobs_available:
            global_logprobs = True

        feature_spaces = {
            "tfidf": X_tfidf,
            "sbert": X_sbert,
            "llm": X_llm,
        }

        # ── Phase 3: k-NN and CRND ──
        knn_results = compute_knn_and_crnd(
            feature_spaces=feature_spaces,
            k_values=K_VALUES,
        )
        all_knn_results[ds_name] = knn_results

        # ── Phase 4: Niche overlap ──
        niche_overlap = compute_niche_overlap_all(
            feature_spaces=feature_spaces,
            labels=labels,
            class_pairs=class_pairs,
        )

        # ── Phase 5: Classifier training ──
        classifier_results = train_classifiers_ovo(
            feature_spaces=feature_spaces,
            labels=labels,
            class_pairs=class_pairs,
        )

        # ── Phase 5b: Baseline ──
        baseline_results = train_baseline_multiclass(
            feature_spaces=feature_spaces,
            labels=labels,
        )

        # Store per-dataset results
        for s in space_names:
            if s not in all_niche_overlap:
                all_niche_overlap[s] = {}
            all_niche_overlap[s].update(niche_overlap.get(s, {}))

            if s not in all_classifier_results:
                all_classifier_results[s] = {}
            all_classifier_results[s].update(classifier_results.get(s, {}))

        all_baseline_results[ds_name] = baseline_results

        # Per-dataset Kendall's τ
        if len(class_pairs) >= 3:
            ds_tau = compute_kendall_tau(
                niche_overlap=niche_overlap,
                classifier_results=classifier_results,
                class_pairs=class_pairs,
                space_names=space_names,
            )
            per_dataset_tau[ds_name] = ds_tau
        else:
            per_dataset_tau[ds_name] = {
                "tau": float("nan"),
                "n_pairs": len(class_pairs),
            }

        elapsed = time.time() - t_start
        logger.info(f"Dataset {ds_name} complete. Elapsed: {elapsed:.0f}s")

    # ── Phase 6: Pooled Kendall's τ ──
    all_class_pairs = list(all_niche_overlap.get("tfidf", {}).keys())
    logger.info(f"\nPooled analysis: {len(all_class_pairs)} total class pairs")

    pooled_tau = compute_kendall_tau(
        niche_overlap=all_niche_overlap,
        classifier_results=all_classifier_results,
        class_pairs=all_class_pairs,
        space_names=space_names,
    )

    # ── Phase 7: Meta-predictor ──
    meta_result = train_meta_predictor(
        niche_overlap=all_niche_overlap,
        classifier_results=all_classifier_results,
        class_pairs=all_class_pairs,
        space_names=space_names,
    )

    # ── Phase 8: Assemble per-dataset output ──
    # We need per-dataset knn_results, niche_overlap, classifier_results
    # for the output assembly
    output = {"datasets": [], "metadata": {}}

    for ds_name, ds_data in datasets_data.items():
        labels = ds_data["labels"]
        unique_labels = ds_data["unique_labels"]
        class_pairs_ds = list(combinations(unique_labels, 2))
        n = len(ds_data["texts"])

        knn_res = all_knn_results.get(ds_name, {"neighbors": {}, "crnd": {}})

        ds_examples = []
        for i in range(n):
            inst_label = labels[i]
            relevant_pairs = [
                (ca, cb) for ca, cb in class_pairs_ds
                if ca == inst_label or cb == inst_label
            ]

            pred_space_votes = {s: 0.0 for s in space_names}
            actual_space_votes = {s: 0.0 for s in space_names}

            for ca, cb in relevant_pairs:
                d_vals = [
                    all_niche_overlap.get(s, {}).get((ca, cb), float("nan"))
                    for s in space_names
                ]
                f1_vals = [
                    all_classifier_results.get(s, {}).get((ca, cb), float("nan"))
                    for s in space_names
                ]
                if not any(np.isnan(v) for v in d_vals):
                    pred_best_idx = int(np.argmin(d_vals))
                    pred_space_votes[space_names[pred_best_idx]] += 1
                if not any(np.isnan(v) for v in f1_vals):
                    actual_best_idx = int(np.argmax(f1_vals))
                    actual_space_votes[space_names[actual_best_idx]] += 1

            pred_best = max(pred_space_votes, key=pred_space_votes.get) if any(
                v > 0 for v in pred_space_votes.values()
            ) else "unknown"
            actual_best = max(actual_space_votes, key=actual_space_votes.get) if any(
                v > 0 for v in actual_space_votes.values()
            ) else "unknown"

            example = {
                "input": ds_data["texts"][i][:500],
                "output": str(labels[i]),
            }

            for k in K_VALUES:
                neighbors = knn_res.get("neighbors", {}).get(k, {})
                for s in space_names:
                    nn_key = f"metadata_{s}_nn_k{k}"
                    if s in neighbors and i < len(neighbors[s]):
                        example[nn_key] = json.dumps(neighbors[s][i].tolist())
                    else:
                        example[nn_key] = "[]"

            for k in K_VALUES:
                crnd_arr = knn_res.get("crnd", {}).get(k)
                if crnd_arr is not None and i < len(crnd_arr):
                    example[f"metadata_crnd_k{k}"] = str(round(float(crnd_arr[i]), 4))
                else:
                    example[f"metadata_crnd_k{k}"] = "0.0"

            example["predict_method_selection"] = pred_best
            example["predict_actual_best"] = actual_best

            ds_examples.append(example)

        output["datasets"].append({
            "dataset": ds_name,
            "examples": ds_examples,
        })

    # ── Build metadata ──
    tau = pooled_tau.get("tau", float("nan"))
    ci_lo = pooled_tau.get("ci_lower", float("nan"))
    ci_hi = pooled_tau.get("ci_upper", float("nan"))
    p_val = pooled_tau.get("p_value", float("nan"))
    meta_auc = meta_result.get("auc", float("nan"))

    per_ds_tau_json = {}
    for ds_name, ds_tau in per_dataset_tau.items():
        t_val = ds_tau.get("tau", float("nan"))
        per_ds_tau_json[ds_name] = {
            "tau": round(t_val, 4) if not np.isnan(t_val) else "N/A",
            "n_pairs": ds_tau.get("n_pairs", 0),
        }

    baseline_json = {}
    for ds_name, br in all_baseline_results.items():
        baseline_json[ds_name] = {
            s: {
                k2: round(v, 4) if isinstance(v, float) and not np.isnan(v) else "N/A"
                for k2, v in vals.items()
            }
            for s, vals in br.items()
        }

    output["metadata"] = {
        "method_name": "Cross-Representation Method Selection via Niche Overlap (3-family)",
        "description": (
            "Validates whether Schoener's D niche overlap profiles across "
            "TF-IDF, SBERT, and LLM feature spaces predict classifier rank-ordering. "
            "Compares against iteration 2 baseline of Kendall's tau=0.24."
        ),
        "kendall_tau_pooled": round(tau, 4) if not np.isnan(tau) else "N/A",
        "kendall_tau_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else "N/A",
        "kendall_tau_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else "N/A",
        "kendall_tau_p_value": round(p_val, 6) if not np.isnan(p_val) else "N/A",
        "iteration2_tau_baseline": 0.24,
        "meta_predictor_auc": round(meta_auc, 4) if not np.isnan(meta_auc) else "N/A",
        "n_class_pairs_evaluated": pooled_tau.get("n_pairs", 0),
        "feature_spaces": [
            "tfidf_5000_unigram",
            "sbert_miniLM_384d",
            "llm_llama31_8b_zeroshot",
        ],
        "classifiers_used": ["logistic_regression", "linear_svc", "gradient_boosting"],
        "k_values": K_VALUES,
        "per_dataset_tau": per_ds_tau_json,
        "baseline_multiclass_f1": baseline_json,
        "llm_model": LLM_MODEL,
        "llm_logprobs_available": global_logprobs,
        "llm_total_calls": LLM_CALL_COUNT,
        "llm_total_cost_usd": round(LLM_TOTAL_COST, 4),
        "n_bootstrap": N_BOOTSTRAP,
    }

    # ── Save output ──
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    total_examples = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"Total examples in output: {total_examples}")
    logger.info(f"Total LLM calls: {LLM_CALL_COUNT}, cost: ${LLM_TOTAL_COST:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)
    logger.info("DONE")


if __name__ == "__main__":
    main()
