#!/usr/bin/env python3
"""
CRND Hybrid Experiment: Cross-Representation Neighborhood Dissonance
with Ecological Niche Overlap and 4-Quadrant Subpopulation Analysis.

Computes per-instance CRND + 6 hybrid noise detection metrics across
clinical text datasets in 3 feature spaces (TF-IDF, sentence embeddings,
LLM zero-shot), with noise injection evaluation, Schoener's D niche
overlap, and subpopulation analysis.
"""

import gc
import json
import os
import re
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, kendalltau, kruskal, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Resource limits (16 GB system → cap at 14 GB; 1-hour CPU)
# ---------------------------------------------------------------------------
try:
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
except ValueError:
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except ValueError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{name:>12.12}}{END}.{CYAN}{{function:<22.22}}{END}:{CYAN}{{line:<4}}{END}| {{message}}",
    colorize=False,
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_NEIGHBORS = 10
NOISE_RATES = [0.05, 0.10, 0.20]
N_SEEDS = 3
MAX_SAMPLE = 500  # max per dataset for TF-IDF / embedding
LLM_SUBSET_SIZE = 100  # max for LLM features (reduced for budget)
MAX_LLM_CALLS = 10000  # hard API call limit
LLM_BUDGET_USD = 10.0  # hard budget limit

WORKSPACE = Path(__file__).resolve().parent
DEP1_PATH = WORKSPACE / "full_data_out.json"
DEP2_PATH = WORKSPACE / "full_data_out_dep2.json"

SKILL_DIR = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii_openrouter_llms")
OR_PY = SKILL_DIR / "scripts/.venv/bin/python"
OR_SCRIPT = SKILL_DIR / "scripts/aii_or_call_llms.py"

# Track global LLM usage
llm_calls_total = 0
llm_cost_total = 0.0


# ===================================================================
# PHASE 0: DATA LOADING
# ===================================================================
def stratified_sample_indices(
    labels: np.ndarray | list,
    n: int,
    seed: int = 42,
) -> list[int]:
    """Return stratified random sample of *n* indices preserving label proportions."""
    labels_arr = np.asarray(labels)
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(labels_arr, return_counts=True)
    proportions = counts / counts.sum()
    per_class = np.maximum((proportions * n).astype(int), 1)

    # adjust if we over/under-shot
    diff = n - per_class.sum()
    if diff > 0:
        for _ in range(diff):
            per_class[rng.choice(len(unique))] += 1
    elif diff < 0:
        for _ in range(-diff):
            candidates = np.where(per_class > 1)[0]
            if len(candidates) == 0:
                break
            per_class[rng.choice(candidates)] -= 1

    indices: list[int] = []
    for cls, cnt in zip(unique, per_class):
        cls_idx = np.where(labels_arr == cls)[0]
        chosen = rng.choice(cls_idx, size=min(cnt, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return indices[:n]


@logger.catch
def load_and_prepare_datasets(
    dep1_path: Path,
    dep2_path: Path,
    max_sample: int = MAX_SAMPLE,
    limit_per_dataset: int | None = None,
) -> dict[str, tuple[list[str], list[str], list[dict]]]:
    """Load from both dependency files, merge, deduplicate, sample."""
    all_datasets: dict[str, tuple[list[str], list[str], list[dict]]] = {}

    for path in [dep1_path, dep2_path]:
        if not path.exists():
            logger.warning(f"Dependency file not found: {path}")
            continue
        data = json.loads(path.read_text())
        for ds_entry in data["datasets"]:
            ds_name = ds_entry["dataset"]
            if ds_name == "clinical_patient_triage_nl":
                logger.info(f"Skipping {ds_name} (N=31, too small for k-NN)")
                continue
            if ds_name in all_datasets:
                logger.warning(f"Duplicate dataset '{ds_name}' — keeping first occurrence")
                continue
            examples = ds_entry["examples"]
            texts = [ex["input"] for ex in examples]
            labels = [str(ex["output"]) for ex in examples]
            all_datasets[ds_name] = (texts, labels, examples)

    # Apply optional per-dataset limit (for gradual scaling) — use stratified sampling
    if limit_per_dataset is not None:
        trimmed: dict[str, tuple[list[str], list[str], list[dict]]] = {}
        for name, (texts, labels, examples) in all_datasets.items():
            n = min(limit_per_dataset, len(texts))
            if n < len(texts):
                indices = stratified_sample_indices(labels, n=n, seed=42)
                trimmed[name] = (
                    [texts[i] for i in indices],
                    [labels[i] for i in indices],
                    [examples[i] for i in indices],
                )
            else:
                trimmed[name] = (texts, labels, examples)
            logger.info(f"{name}: trimmed to {n}/{len(texts)}")
        all_datasets = trimmed

    # Stratified sampling for large datasets
    sampled: dict[str, tuple[list[str], list[str], list[dict]]] = {}
    for name, (texts, labels, examples) in all_datasets.items():
        if len(texts) > max_sample:
            indices = stratified_sample_indices(labels, n=max_sample, seed=42)
            sampled[name] = (
                [texts[i] for i in indices],
                [labels[i] for i in indices],
                [examples[i] for i in indices],
            )
            logger.info(f"{name}: sampled {max_sample}/{len(texts)}")
        else:
            sampled[name] = (texts, labels, examples)
            logger.info(f"{name}: using all {len(texts)}")

    return sampled


# ===================================================================
# PHASE 1: FEATURE SPACE CONSTRUCTION
# ===================================================================
def build_tfidf_features(texts: list[str]) -> np.ndarray:
    """TF-IDF with clinical preprocessing, max 5000 features."""
    max_feats = min(5000, max(100, len(texts) * 2))
    vectorizer = TfidfVectorizer(
        max_features=max_feats,
        stop_words="english",
        min_df=max(2, int(len(texts) * 0.005)),
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts).toarray()
    logger.debug(f"TF-IDF shape: {X.shape}")
    return X


def build_sentence_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """all-MiniLM-L6-v2 sentence embeddings, 384-dim. CPU-only."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    logger.debug(f"Sentence embeddings shape: {embeddings.shape}")
    return embeddings


def build_llm_features(
    texts: list[str],
    labels_unique: list[str],
    dataset_name: str,
) -> np.ndarray:
    """
    LLM zero-shot classification via OpenRouter.
    Uses meta-llama/llama-3.1-8b-instruct for probability vectors.
    Falls back to uniform distribution on parse failure.
    """
    global llm_calls_total, llm_cost_total

    K = len(labels_unique)
    label_list_str = ", ".join(labels_unique)
    features = np.zeros((len(texts), K))

    if not OR_PY.exists() or not OR_SCRIPT.exists():
        logger.warning("OpenRouter skill not found — using uniform LLM features")
        features[:] = 1.0 / K
        return features

    # Batch texts for efficiency (5 per call)
    batch_size = 5
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        # Check limits
        if llm_calls_total >= MAX_LLM_CALLS:
            logger.warning(f"LLM call limit ({MAX_LLM_CALLS}) reached — filling rest with uniform")
            start_i = batch_idx * batch_size
            features[start_i:] = 1.0 / K
            break
        if llm_cost_total >= LLM_BUDGET_USD * 0.95:
            logger.warning(f"LLM budget limit (${LLM_BUDGET_USD}) approaching — filling rest with uniform")
            start_i = batch_idx * batch_size
            features[start_i:] = 1.0 / K
            break

        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, len(texts))
        batch_texts = texts[start_i:end_i]

        # Build batch prompt
        text_block = ""
        for j, t in enumerate(batch_texts):
            truncated = t[:400]
            text_block += f"\n[Text {j+1}]: {truncated}\n"

        prompt = (
            f"Classify each text below into exactly one of these categories: [{label_list_str}]. "
            f"For each text, respond with ONLY a JSON array of objects, one per text. "
            f"Each object maps every category name to a probability (0.0-1.0), summing to 1.0. "
            f"Example for 2 texts with 3 categories: "
            f'[{{"cat_A": 0.7, "cat_B": 0.2, "cat_C": 0.1}}, {{"cat_A": 0.1, "cat_B": 0.8, "cat_C": 0.1}}]'
            f"\n\nTexts:{text_block}\n\nRespond with ONLY the JSON array, nothing else."
        )

        try:
            result = subprocess.run(
                [
                    str(OR_PY),
                    str(OR_SCRIPT),
                    "--model", "meta-llama/llama-3.1-8b-instruct",
                    "--input", prompt,
                    "--temperature", "0.0",
                    "--max-tokens", "500",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            llm_calls_total += 1
            # Estimate cost: ~0.001 per call for llama-3.1-8b
            llm_cost_total += 0.001

            response_text = result.stdout.strip()
            # Try to parse the JSON from the response
            batch_probs = _parse_batch_llm_response(response_text, labels_unique, len(batch_texts))

            for j in range(len(batch_texts)):
                features[start_i + j] = batch_probs[j]

        except subprocess.TimeoutExpired:
            logger.warning(f"LLM batch {batch_idx} timed out — using uniform")
            for j in range(len(batch_texts)):
                features[start_i + j] = np.ones(K) / K
        except Exception:
            logger.exception(f"LLM batch {batch_idx} failed")
            for j in range(len(batch_texts)):
                features[start_i + j] = np.ones(K) / K

        if batch_idx % 20 == 0 and batch_idx > 0:
            logger.info(
                f"  LLM progress: {batch_idx}/{n_batches} batches, "
                f"calls={llm_calls_total}, cost=${llm_cost_total:.3f}"
            )

    return features


def _parse_batch_llm_response(
    response: str,
    labels_unique: list[str],
    n_texts: int,
) -> np.ndarray:
    """Parse LLM batch response into probability matrix."""
    K = len(labels_unique)
    uniform = np.ones((n_texts, K)) / K

    # Find JSON array in response
    # Look for text after "Response:" if present
    if "Response:" in response:
        response = response.split("Response:", 1)[1].strip()

    # Try to extract JSON array
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if not json_match:
        return uniform

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return uniform

    if not isinstance(parsed, list):
        return uniform

    result = np.ones((n_texts, K)) / K
    for j in range(min(len(parsed), n_texts)):
        if isinstance(parsed[j], dict):
            probs = np.zeros(K)
            for idx, label in enumerate(labels_unique):
                # Try exact match and case-insensitive match
                val = parsed[j].get(label, parsed[j].get(label.lower(), 0.0))
                try:
                    probs[idx] = float(val)
                except (TypeError, ValueError):
                    probs[idx] = 0.0
            # Normalize
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                probs = np.ones(K) / K
            result[j] = probs

    return result


def build_doc2vec_features(texts: list[str], vector_size: int = 100) -> np.ndarray:
    """
    Doc2Vec fallback feature space (used if LLM features are unavailable).
    Uses simple TF-IDF with char n-grams as a distinct representation.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=min(3000, max(100, len(texts) * 2)),
        min_df=max(2, int(len(texts) * 0.005)),
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts).toarray()
    # PCA down to vector_size dims for k-NN efficiency
    if X.shape[1] > vector_size:
        pca = PCA(n_components=vector_size, random_state=42)
        X = pca.fit_transform(X)
    logger.debug(f"Char n-gram features shape: {X.shape}")
    return X


# ===================================================================
# PHASE 2: SCORE COMPUTATION
# ===================================================================
def jaccard_index(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A n B| / |A u B|."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_crnd(
    X_space1: np.ndarray,
    X_space2: np.ndarray,
    X_space3: np.ndarray,
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-Representation Neighborhood Dissonance.
    Returns: crnd_scores, pair_12, pair_13, pair_23 (all as dissonance = 1 - Jaccard).
    """
    effective_k = min(k, X_space1.shape[0] - 1)
    if effective_k < 1:
        N = X_space1.shape[0]
        return np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    nn1 = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine").fit(X_space1)
    nn2 = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine").fit(X_space2)
    nn3 = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine").fit(X_space3)

    _, idx1 = nn1.kneighbors(X_space1)
    _, idx2 = nn2.kneighbors(X_space2)
    _, idx3 = nn3.kneighbors(X_space3)

    idx1 = idx1[:, 1:]
    idx2 = idx2[:, 1:]
    idx3 = idx3[:, 1:]

    N = X_space1.shape[0]
    crnd = np.zeros(N)
    pair_12 = np.zeros(N)
    pair_13 = np.zeros(N)
    pair_23 = np.zeros(N)

    for i in range(N):
        s_12 = jaccard_index(set(idx1[i]), set(idx2[i]))
        s_13 = jaccard_index(set(idx1[i]), set(idx3[i]))
        s_23 = jaccard_index(set(idx2[i]), set(idx3[i]))

        pair_12[i] = 1.0 - s_12
        pair_13[i] = 1.0 - s_13
        pair_23[i] = 1.0 - s_23
        crnd[i] = 1.0 - np.mean([s_12, s_13, s_23])

    return crnd, pair_12, pair_13, pair_23


def compute_kdn(X: np.ndarray, labels: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """k-Disagreeing Neighbors: fraction of k-NN with different label."""
    effective_k = min(k, X.shape[0] - 1)
    if effective_k < 1:
        return np.zeros(len(labels))

    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine").fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]

    labels_arr = np.asarray(labels)
    kdn = np.zeros(len(labels))
    for i in range(len(labels)):
        neighbor_labels = labels_arr[indices[i]]
        kdn[i] = np.mean(neighbor_labels != labels_arr[i])
    return kdn


def compute_cleanlab_scores(
    X: np.ndarray,
    labels_encoded: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Cleanlab self-confidence scores via cross-validated predicted probabilities.
    Returns 1 - self_confidence (higher = more suspicious).
    Falls back to manual implementation if cleanlab unavailable.
    """
    n_classes = len(np.unique(labels_encoded))
    actual_splits = min(n_splits, min(np.bincount(labels_encoded)))
    actual_splits = max(2, actual_splits)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    try:
        pred_probs = cross_val_predict(clf, X, labels_encoded, cv=cv, method="predict_proba")
    except ValueError:
        logger.warning("cross_val_predict failed — returning uniform scores")
        return np.full(len(labels_encoded), 0.5)

    try:
        from cleanlab.rank import get_self_confidence_for_each_label

        self_conf = get_self_confidence_for_each_label(
            labels=labels_encoded, pred_probs=pred_probs
        )
    except (ImportError, Exception):
        logger.warning("Cleanlab unavailable — using manual self-confidence")
        self_conf = np.array(
            [pred_probs[i, labels_encoded[i]] for i in range(len(labels_encoded))]
        )

    return 1.0 - self_conf


def compute_wann_adapted(
    X_embed: np.ndarray,
    labels_encoded: np.ndarray,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    WANN-adapted: distance-weighted k-NN label agreement in embedding space.
    Score = 1 - weighted_agreement (higher = more suspicious).
    """
    effective_k = min(k, X_embed.shape[0] - 1)
    if effective_k < 1:
        return np.zeros(len(labels_encoded))

    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine").fit(X_embed)
    distances, indices = nn.kneighbors(X_embed)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / distances

    labels_arr = np.asarray(labels_encoded)
    wann_scores = np.zeros(len(labels_encoded))

    for i in range(len(labels_encoded)):
        neighbor_labels = labels_arr[indices[i]]
        same_label = (neighbor_labels == labels_arr[i]).astype(float)
        weighted_agreement = np.sum(same_label * weights[i]) / np.sum(weights[i])
        wann_scores[i] = 1.0 - weighted_agreement

    return wann_scores


def compute_cartography_proxy(
    X: np.ndarray,
    labels_encoded: np.ndarray,
    n_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dataset Cartography proxy using multiple random-seed CV rounds.
    Returns: (cartography_score, confidence, variability).
    """
    N = len(labels_encoded)
    n_classes = len(np.unique(labels_encoded))
    min_class_count = min(np.bincount(labels_encoded))
    n_folds = min(5, min_class_count)
    n_folds = max(2, n_folds)

    all_probs = np.zeros((N, n_rounds))

    for seed_idx in range(n_rounds):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + seed_idx)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
        try:
            pred_probs = cross_val_predict(
                clf, X, labels_encoded, cv=cv, method="predict_proba"
            )
            for i in range(N):
                all_probs[i, seed_idx] = pred_probs[i, labels_encoded[i]]
        except ValueError:
            logger.warning(f"Cartography CV round {seed_idx} failed")
            all_probs[:, seed_idx] = 0.5

    confidence = np.mean(all_probs, axis=1)
    variability = np.std(all_probs, axis=1)
    cartography_score = variability / (confidence + 1e-8)

    return cartography_score, confidence, variability


def compute_label_entropy(
    X_space1: np.ndarray,
    X_space2: np.ndarray,
    X_space3: np.ndarray,
    labels_encoded: np.ndarray,
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average kDN across 3 feature spaces."""
    kdn1 = compute_kdn(X_space1, labels_encoded, k)
    kdn2 = compute_kdn(X_space2, labels_encoded, k)
    kdn3 = compute_kdn(X_space3, labels_encoded, k)
    label_entropy = (kdn1 + kdn2 + kdn3) / 3.0
    return label_entropy, kdn1, kdn2, kdn3


# ===================================================================
# PHASE 3: HYBRID CRND-L FORMULATIONS
# ===================================================================
def compute_hybrid_scores(
    crnd: np.ndarray,
    pair_12: np.ndarray,
    pair_13: np.ndarray,
    pair_23: np.ndarray,
    kdn_avg: np.ndarray,
    label_entropy: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Six hybrid formulations (H1-H4 unsupervised, H5/H6 computed separately).
    H1: CRND x label_entropy (multiplicative)
    H2: CRND x kdn_avg (multiplicative)
    H3: rank_CRND + rank_kDN (rank fusion)
    H4: CRND_tfidf_vs_llm x kdn_avg (best pair, tfidf<->llm most informative)
    """
    N = len(crnd)
    h1 = crnd * label_entropy
    h2 = crnd * kdn_avg

    rank_crnd = rankdata(crnd) / N
    rank_kdn = rankdata(kdn_avg) / N
    h3 = rank_crnd + rank_kdn

    h4 = pair_13 * kdn_avg  # space1 (tfidf) vs space3 (llm/char-ngram)

    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


# ===================================================================
# PHASE 4: NOISE INJECTION + EVALUATION
# ===================================================================
def inject_noise(
    labels_encoded: np.ndarray,
    noise_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip noise_rate fraction of labels to random different class."""
    rng = np.random.RandomState(seed)
    N = len(labels_encoded)
    n_flip = int(N * noise_rate)
    if n_flip == 0:
        return labels_encoded.copy(), np.zeros(N, dtype=bool)

    flip_indices = rng.choice(N, size=n_flip, replace=False)
    n_classes = len(np.unique(labels_encoded))

    noisy_labels = labels_encoded.copy()
    is_flipped = np.zeros(N, dtype=bool)

    for idx in flip_indices:
        old_label = noisy_labels[idx]
        candidates = [c for c in range(n_classes) if c != old_label]
        if candidates:
            noisy_labels[idx] = rng.choice(candidates)
        is_flipped[idx] = True

    return noisy_labels, is_flipped


def evaluate_noise_detection(
    scores: np.ndarray,
    is_flipped: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC, Spearman rho, Precision@K for noise detection scores."""
    flipped_int = is_flipped.astype(int)

    # ROC-AUC
    try:
        if len(np.unique(flipped_int)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(flipped_int, scores)
    except ValueError:
        auc = 0.5

    # Spearman
    try:
        rho, p_val = spearmanr(scores, flipped_int.astype(float))
        if np.isnan(rho):
            rho = 0.0
            p_val = 1.0
    except Exception:
        rho = 0.0
        p_val = 1.0

    # Precision@K
    K = is_flipped.sum()
    if K > 0:
        top_k_indices = np.argsort(scores)[-K:]
        precision_at_k = float(is_flipped[top_k_indices].mean())
    else:
        precision_at_k = 0.0

    return {
        "roc_auc": float(auc),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "precision_at_k": float(precision_at_k),
    }


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    is_flipped: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap significance test for ROC-AUC difference."""
    rng = np.random.RandomState(seed)
    N = len(is_flipped)
    flipped_int = is_flipped.astype(int)

    if len(np.unique(flipped_int)) < 2:
        return 1.0

    try:
        observed_diff = roc_auc_score(flipped_int, scores_a) - roc_auc_score(
            flipped_int, scores_b
        )
    except ValueError:
        return 1.0

    count_ge = 0
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(N, size=N, replace=True)
        try:
            auc_a = roc_auc_score(flipped_int[boot_idx], scores_a[boot_idx])
            auc_b = roc_auc_score(flipped_int[boot_idx], scores_b[boot_idx])
            if (auc_a - auc_b) >= observed_diff:
                count_ge += 1
        except ValueError:
            continue

    return count_ge / max(1, n_bootstrap)


def aggregate_seed_results(
    seed_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Aggregate metric dictionaries across seeds: mean +/- std."""
    if not seed_results:
        return {}

    method_names = list(seed_results[0].keys())
    agg: dict[str, dict[str, float]] = {}

    for method in method_names:
        metrics_across_seeds: dict[str, list[float]] = {}
        for sr in seed_results:
            if method not in sr:
                continue
            for metric_name, val in sr[method].items():
                metrics_across_seeds.setdefault(metric_name, []).append(val)

        agg[method] = {}
        for metric_name, vals in metrics_across_seeds.items():
            agg[method][f"{metric_name}_mean"] = float(np.mean(vals))
            agg[method][f"{metric_name}_std"] = float(np.std(vals))

    return agg


def find_best_method(
    methods: dict[str, np.ndarray],
    is_flipped: np.ndarray,
) -> tuple[str, np.ndarray]:
    """Find the method with highest ROC-AUC."""
    best_name = ""
    best_auc = -1.0
    best_scores = np.zeros(len(is_flipped))

    for name, scores in methods.items():
        try:
            auc = roc_auc_score(is_flipped.astype(int), scores)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_scores = scores

    return best_name, best_scores


# ===================================================================
# PHASE 5: QUADRANT SUBPOPULATION ANALYSIS
# ===================================================================
def quadrant_analysis(
    crnd: np.ndarray,
    label_entropy: np.ndarray,
    is_flipped: np.ndarray,
    labels_encoded: np.ndarray,
    texts: list[str],
) -> dict[str, Any]:
    """Partition instances by median CRND x median label_entropy into 4 quadrants."""
    med_crnd = float(np.median(crnd))
    med_entropy = float(np.median(label_entropy))

    quadrants = {
        "Q1_easy": (crnd <= med_crnd) & (label_entropy <= med_entropy),
        "Q2_repr_ambiguous": (crnd > med_crnd) & (label_entropy <= med_entropy),
        "Q3_standard_noise": (crnd <= med_crnd) & (label_entropy > med_entropy),
        "Q4_genuinely_ambiguous": (crnd > med_crnd) & (label_entropy > med_entropy),
    }

    results: dict[str, Any] = {"median_crnd": med_crnd, "median_label_entropy": med_entropy}

    for q_name, mask in quadrants.items():
        q_size = int(mask.sum())
        q_noise_rate = float(is_flipped[mask].mean()) if q_size > 0 else 0.0
        q_text_lengths = [len(texts[i]) for i in range(len(texts)) if mask[i]]

        q_labels = labels_encoded[mask]
        if len(q_labels) > 0:
            unique, counts = np.unique(q_labels, return_counts=True)
            class_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        else:
            class_dist = {}

        results[q_name] = {
            "count": q_size,
            "fraction": float(q_size / len(crnd)) if len(crnd) > 0 else 0.0,
            "noise_rate": q_noise_rate,
            "text_length_mean": float(np.mean(q_text_lengths)) if q_text_lengths else 0.0,
            "text_length_std": float(np.std(q_text_lengths)) if q_text_lengths else 0.0,
            "class_distribution": class_dist,
        }

    return results


# ===================================================================
# PHASE 6: SCHOENER'S D NICHE OVERLAP
# ===================================================================
def compute_schoeners_d_from_2d(
    X_2d: np.ndarray,
    labels_encoded: np.ndarray,
    class_a: int,
    class_b: int,
    grid_points: np.ndarray,
) -> float:
    """Schoener's D overlap from pre-computed PCA-2D + shared grid."""
    mask_a = labels_encoded == class_a
    mask_b = labels_encoded == class_b

    if mask_a.sum() < 3 or mask_b.sum() < 3:
        return float("nan")

    X_a = X_2d[mask_a].T
    X_b = X_2d[mask_b].T

    try:
        kde_a = gaussian_kde(X_a)
        kde_b = gaussian_kde(X_b)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")

    z_a = kde_a(grid_points)
    z_b = kde_b(grid_points)

    z_a = z_a / z_a.sum() if z_a.sum() > 0 else z_a
    z_b = z_b / z_b.sum() if z_b.sum() > 0 else z_b

    D = 1.0 - 0.5 * np.sum(np.abs(z_a - z_b))
    return float(D)


def compute_niche_overlap_matrix(
    X: np.ndarray,
    labels_encoded: np.ndarray,
    unique_classes: list[int],
    max_pairs: int = 100,
) -> np.ndarray:
    """Compute pairwise Schoener's D. Pre-compute PCA and grid once. Cap at max_pairs."""
    n_classes = len(unique_classes)
    D_matrix = np.eye(n_classes)

    # Pre-compute PCA to 2D once
    pca = PCA(n_components=min(2, X.shape[1]), random_state=42)
    X_2d = pca.fit_transform(X)

    # Pre-compute grid once
    n_grid = 80
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min = X_2d[:, 1].min() - 1 if X_2d.shape[1] > 1 else -1
    y_max = X_2d[:, 1].max() + 1 if X_2d.shape[1] > 1 else 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid))
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    # Collect all pairs, cap if too many
    pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
    if len(pairs) > max_pairs:
        rng = np.random.RandomState(42)
        selected = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[s] for s in selected]
        logger.info(f"  Niche overlap: sampling {max_pairs}/{n_classes*(n_classes-1)//2} pairs")

    for i, j in pairs:
        d_val = compute_schoeners_d_from_2d(
            X_2d, labels_encoded, unique_classes[i], unique_classes[j], grid_points
        )
        D_matrix[i, j] = d_val
        D_matrix[j, i] = d_val

    return D_matrix


def per_class_crnd_analysis(
    crnd: np.ndarray,
    labels_encoded: np.ndarray,
    unique_classes: list[int],
) -> dict[str, Any]:
    """Kruskal-Wallis test + per-class CRND stats."""
    groups = [crnd[labels_encoded == c] for c in unique_classes]
    # Filter out empty groups
    non_empty = [g for g in groups if len(g) > 0]

    if len(non_empty) < 2:
        stat, p_val = 0.0, 1.0
    else:
        try:
            stat, p_val = kruskal(*non_empty)
        except ValueError:
            stat, p_val = 0.0, 1.0

    per_class: dict[str, Any] = {}
    for c, g in zip(unique_classes, groups):
        if len(g) > 0:
            per_class[str(c)] = {
                "mean": float(np.mean(g)),
                "median": float(np.median(g)),
                "std": float(np.std(g)),
                "n": int(len(g)),
            }
        else:
            per_class[str(c)] = {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}

    return {
        "kruskal_wallis_stat": float(stat),
        "kruskal_wallis_p": float(p_val),
        "per_class": per_class,
    }


def niche_predicts_classifier(
    X_space1: np.ndarray,
    X_space2: np.ndarray,
    X_space3: np.ndarray,
    labels_encoded: np.ndarray,
    niche_s1: np.ndarray,
    niche_s2: np.ndarray,
    niche_s3: np.ndarray,
    unique_classes: list[int],
) -> dict[str, Any]:
    """
    Test whether niche overlap predicts pairwise classification difficulty.
    For each feature space, compute per-class-pair confusion rate vs niche D → Kendall tau.
    """
    n_classes = len(unique_classes)
    if n_classes < 2:
        return {"tau_space1": 0.0, "tau_space2": 0.0, "tau_space3": 0.0}

    results: dict[str, Any] = {}

    for space_name, X, niche_matrix in [
        ("space1_tfidf", X_space1, niche_s1),
        ("space2_embed", X_space2, niche_s2),
        ("space3_llm", X_space3, niche_s3),
    ]:
        # Train a simple classifier, get confusion
        n_folds = min(5, min(np.bincount(labels_encoded)))
        n_folds = max(2, n_folds)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        try:
            preds = cross_val_predict(clf, X, labels_encoded, cv=cv)
        except ValueError:
            results[f"tau_{space_name}"] = 0.0
            results[f"p_{space_name}"] = 1.0
            continue

        # Compute per-pair confusion rate
        confusion_rates = []
        niche_overlaps = []

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                mask_ij = (labels_encoded == unique_classes[i]) | (
                    labels_encoded == unique_classes[j]
                )
                if mask_ij.sum() < 5:
                    continue

                preds_ij = preds[mask_ij]
                labels_ij = labels_encoded[mask_ij]

                # Confusion rate = fraction misclassified between these two classes
                n_confused = np.sum(
                    ((labels_ij == unique_classes[i]) & (preds_ij == unique_classes[j]))
                    | ((labels_ij == unique_classes[j]) & (preds_ij == unique_classes[i]))
                )
                conf_rate = n_confused / mask_ij.sum()

                niche_val = niche_matrix[i, j]
                if not np.isnan(niche_val):
                    confusion_rates.append(conf_rate)
                    niche_overlaps.append(niche_val)

        if len(confusion_rates) >= 3:
            try:
                tau, p_val = kendalltau(niche_overlaps, confusion_rates)
                results[f"tau_{space_name}"] = float(tau) if not np.isnan(tau) else 0.0
                results[f"p_{space_name}"] = float(p_val) if not np.isnan(p_val) else 1.0
            except Exception:
                results[f"tau_{space_name}"] = 0.0
                results[f"p_{space_name}"] = 1.0
        else:
            results[f"tau_{space_name}"] = 0.0
            results[f"p_{space_name}"] = 1.0

    return results


# ===================================================================
# PHASE 7: OUTPUT FORMATTING
# ===================================================================
def format_output(
    all_results: dict[str, Any],
) -> dict[str, Any]:
    """Ensure all numpy types are JSON-serializable."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(all_results)


# ===================================================================
# MAIN EXECUTION
# ===================================================================
@logger.catch
def main() -> None:
    t_start = time.time()

    # ---------------------------------------------------------------
    # 0. Copy dependency data if not present
    # ---------------------------------------------------------------
    dep1_src = Path(
        "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044"
        "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
    )
    dep2_src = Path(
        "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_220558"
        "/3_invention_loop/iter_4/gen_art/data_id3_it4__opus/full_data_out.json"
    )

    if not DEP1_PATH.exists() and dep1_src.exists():
        import shutil
        shutil.copy2(dep1_src, DEP1_PATH)
        logger.info(f"Copied dep1 → {DEP1_PATH}")

    if not DEP2_PATH.exists() and dep2_src.exists():
        import shutil
        shutil.copy2(dep2_src, DEP2_PATH)
        logger.info(f"Copied dep2 → {DEP2_PATH}")

    # Check for limit_per_dataset env var (for gradual scaling)
    limit_str = os.environ.get("LIMIT_PER_DATASET")
    limit_per_dataset = int(limit_str) if limit_str else None

    # Check for LLM disable flag
    use_llm = os.environ.get("USE_LLM", "1") == "1"

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    datasets = load_and_prepare_datasets(
        dep1_path=DEP1_PATH,
        dep2_path=DEP2_PATH,
        max_sample=MAX_SAMPLE,
        limit_per_dataset=limit_per_dataset,
    )

    if not datasets:
        logger.error("No datasets loaded!")
        return

    logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    all_results: dict[str, Any] = {"metadata": {}, "datasets": []}
    aggregate: dict[str, Any] = {}

    # Load sentence transformer once (reuse across datasets)
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    for ds_idx, (ds_name, (texts, labels, examples)) in enumerate(datasets.items()):
        t_ds_start = time.time()
        logger.info(f"=== [{ds_idx+1}/{len(datasets)}] Processing {ds_name} (N={len(texts)}) ===")

        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        unique_classes = list(range(len(le.classes_)))
        logger.info(f"  Classes ({len(le.classes_)}): {le.classes_.tolist()[:10]}")

        # ---------------------------------------------------------------
        # 2. Build feature spaces
        # ---------------------------------------------------------------
        logger.info("  Building TF-IDF features...")
        X_tfidf = build_tfidf_features(texts)

        logger.info("  Building sentence embeddings...")
        X_embed = st_model.encode(texts, batch_size=64, show_progress_bar=False)
        logger.debug(f"  Embeddings shape: {X_embed.shape}")

        # Third feature space: LLM or char-ngram fallback
        llm_subset_size = min(LLM_SUBSET_SIZE, len(texts))

        if use_llm and llm_subset_size >= 10:
            llm_subset_idx = stratified_sample_indices(labels_encoded, n=llm_subset_size, seed=42)
            logger.info(f"  Building LLM features (subset={llm_subset_size})...")
            X_llm_subset = build_llm_features(
                texts=[texts[i] for i in llm_subset_idx],
                labels_unique=le.classes_.tolist(),
                dataset_name=ds_name,
            )
            # Check if LLM returned meaningful features (not all uniform)
            llm_variance = np.var(X_llm_subset)
            if llm_variance < 1e-8:
                logger.warning("  LLM features are all uniform — falling back to char n-gram")
                use_llm_for_this_ds = False
            else:
                use_llm_for_this_ds = True
        else:
            use_llm_for_this_ds = False
            llm_subset_idx = list(range(len(texts)))

        if not use_llm_for_this_ds:
            logger.info("  Building char n-gram features (fallback for LLM)...")
            llm_subset_idx = list(range(len(texts)))
            X_llm_subset = build_doc2vec_features(texts, vector_size=100)
            llm_subset_size = len(texts)

        # Work on the LLM subset for cross-representation analysis
        if use_llm_for_this_ds:
            X_tfidf_sub = X_tfidf[llm_subset_idx]
            X_embed_sub = X_embed[llm_subset_idx]
            X_llm_sub = X_llm_subset
            labels_sub = labels_encoded[llm_subset_idx]
            texts_sub = [texts[i] for i in llm_subset_idx]
            examples_sub = [examples[i] for i in llm_subset_idx]
        else:
            # All three feature spaces are on full data
            X_tfidf_sub = X_tfidf
            X_embed_sub = X_embed
            X_llm_sub = X_llm_subset
            labels_sub = labels_encoded
            texts_sub = texts
            examples_sub = examples
            llm_subset_idx = list(range(len(texts)))

        N_sub = len(labels_sub)
        logger.info(f"  Working subset size: {N_sub}")

        # ---------------------------------------------------------------
        # 3. Compute all scores
        # ---------------------------------------------------------------
        logger.info("  Computing CRND scores...")
        crnd, pair_te, pair_tl, pair_el = compute_crnd(X_tfidf_sub, X_embed_sub, X_llm_sub)

        logger.info("  Computing label entropy (kDN per space)...")
        label_entropy, kdn_tfidf, kdn_embed, kdn_llm = compute_label_entropy(
            X_tfidf_sub, X_embed_sub, X_llm_sub, labels_sub
        )
        kdn_avg = (kdn_tfidf + kdn_embed + kdn_llm) / 3.0

        logger.info("  Computing cleanlab scores...")
        cleanlab_tfidf = compute_cleanlab_scores(X_tfidf_sub, labels_sub)
        cleanlab_embed = compute_cleanlab_scores(X_embed_sub, labels_sub)
        cleanlab_avg = (cleanlab_tfidf + cleanlab_embed) / 2.0

        logger.info("  Computing WANN-adapted scores...")
        wann = compute_wann_adapted(X_embed_sub, labels_sub)

        logger.info("  Computing cartography proxy scores...")
        carto_score, carto_conf, carto_var = compute_cartography_proxy(X_tfidf_sub, labels_sub)

        logger.info("  Computing hybrid scores (H1-H4)...")
        hybrids = compute_hybrid_scores(crnd, pair_te, pair_tl, pair_el, kdn_avg, label_entropy)

        # ---------------------------------------------------------------
        # 4. Noise injection evaluation
        # ---------------------------------------------------------------
        logger.info("  Running noise injection evaluation...")
        noise_results: dict[str, Any] = {}

        for noise_rate in NOISE_RATES:
            seed_results_list: list[dict[str, dict[str, float]]] = []

            for seed in range(N_SEEDS):
                noise_seed = seed * 100 + int(noise_rate * 100)
                noisy_labels, is_flipped = inject_noise(labels_sub, noise_rate, seed=noise_seed)

                if is_flipped.sum() == 0:
                    continue

                # All methods to evaluate
                methods: dict[str, np.ndarray] = {
                    "crnd": crnd,
                    "kdn_tfidf": kdn_tfidf,
                    "kdn_embed": kdn_embed,
                    "kdn_llm": kdn_llm,
                    "kdn_avg": kdn_avg,
                    "cleanlab_tfidf": cleanlab_tfidf,
                    "cleanlab_embed": cleanlab_embed,
                    "cleanlab_avg": cleanlab_avg,
                    "wann_adapted": wann,
                    "cartography": carto_score,
                    "label_entropy": label_entropy,
                    "random": np.random.RandomState(seed).rand(N_sub),
                }
                methods.update(hybrids)

                eval_results: dict[str, dict[str, float]] = {}
                for method_name, scores in methods.items():
                    eval_results[method_name] = evaluate_noise_detection(scores, is_flipped)

                seed_results_list.append(eval_results)

            if seed_results_list:
                noise_results[str(noise_rate)] = aggregate_seed_results(seed_results_list)

        # ---------------------------------------------------------------
        # 5. Bootstrap significance tests (at 10% noise, seed 0)
        # ---------------------------------------------------------------
        logger.info("  Running significance tests...")
        _, is_flipped_10 = inject_noise(labels_sub, 0.10, seed=10)
        sig_tests: dict[str, float] = {}

        if is_flipped_10.sum() > 0:
            best_hybrid_name, best_hybrid_scores = find_best_method(hybrids, is_flipped_10)
            sig_tests["best_hybrid"] = best_hybrid_name
            sig_tests["best_hybrid_vs_crnd_p"] = paired_bootstrap_test(
                best_hybrid_scores, crnd, is_flipped_10
            )
            sig_tests["best_hybrid_vs_cleanlab_p"] = paired_bootstrap_test(
                best_hybrid_scores, cleanlab_avg, is_flipped_10
            )
            sig_tests["crnd_vs_kdn_p"] = paired_bootstrap_test(crnd, kdn_avg, is_flipped_10)

        # ---------------------------------------------------------------
        # 6. Quadrant analysis
        # ---------------------------------------------------------------
        logger.info("  Running quadrant analysis...")
        quadrant_results = quadrant_analysis(
            crnd=crnd,
            label_entropy=label_entropy,
            is_flipped=is_flipped_10,
            labels_encoded=labels_sub,
            texts=texts_sub,
        )

        # ---------------------------------------------------------------
        # 7. Schoener's D niche overlap
        # ---------------------------------------------------------------
        logger.info("  Computing Schoener's D niche overlap...")
        niche_tfidf = compute_niche_overlap_matrix(X_tfidf_sub, labels_sub, unique_classes)
        niche_embed = compute_niche_overlap_matrix(X_embed_sub, labels_sub, unique_classes)
        niche_llm = compute_niche_overlap_matrix(X_llm_sub, labels_sub, unique_classes)

        # ---------------------------------------------------------------
        # 8. Per-class CRND analysis
        # ---------------------------------------------------------------
        logger.info("  Per-class CRND analysis...")
        class_crnd = per_class_crnd_analysis(crnd, labels_sub, unique_classes)

        # ---------------------------------------------------------------
        # 9. Niche overlap -> classifier performance prediction
        # ---------------------------------------------------------------
        logger.info("  Testing niche overlap → classifier prediction...")
        kendall_results = niche_predicts_classifier(
            X_space1=X_tfidf_sub,
            X_space2=X_embed_sub,
            X_space3=X_llm_sub,
            labels_encoded=labels_sub,
            niche_s1=niche_tfidf,
            niche_s2=niche_embed,
            niche_s3=niche_llm,
            unique_classes=unique_classes,
        )

        # ---------------------------------------------------------------
        # Store aggregate results for this dataset
        # ---------------------------------------------------------------
        aggregate[ds_name] = {
            "n_instances": N_sub,
            "n_classes": len(le.classes_),
            "class_names": le.classes_.tolist(),
            "noise_detection": noise_results,
            "significance_tests": sig_tests,
            "quadrant_analysis": quadrant_results,
            "niche_overlap": {
                "tfidf": niche_tfidf.tolist(),
                "embedding": niche_embed.tolist(),
                "llm": niche_llm.tolist(),
                "class_names": le.classes_.tolist(),
            },
            "per_class_crnd": class_crnd,
            "kendall_tau_niche_vs_classifier": kendall_results,
            "score_statistics": {
                "crnd": {"mean": float(np.mean(crnd)), "std": float(np.std(crnd)), "median": float(np.median(crnd))},
                "kdn_avg": {"mean": float(np.mean(kdn_avg)), "std": float(np.std(kdn_avg)), "median": float(np.median(kdn_avg))},
                "cleanlab_avg": {"mean": float(np.mean(cleanlab_avg)), "std": float(np.std(cleanlab_avg)), "median": float(np.median(cleanlab_avg))},
                "label_entropy": {"mean": float(np.mean(label_entropy)), "std": float(np.std(label_entropy)), "median": float(np.median(label_entropy))},
            },
        }

        # ---------------------------------------------------------------
        # 10. Build per-instance output examples
        # ---------------------------------------------------------------
        logger.info("  Building per-instance output...")
        ds_examples: list[dict[str, Any]] = []

        med_c = float(np.median(crnd))
        med_e = float(np.median(label_entropy))
        p33 = float(np.percentile(crnd, 33.3))
        p66 = float(np.percentile(crnd, 66.7))

        for idx_in_sub in range(N_sub):
            orig_idx = llm_subset_idx[idx_in_sub]
            ex: dict[str, Any] = {
                "input": examples_sub[idx_in_sub]["input"],
                "output": str(examples_sub[idx_in_sub]["output"]),
            }

            # Copy existing metadata_ fields
            for key, val in examples_sub[idx_in_sub].items():
                if key.startswith("metadata_") and key not in ex:
                    ex[key] = val

            # Add computed scores
            ex["metadata_crnd"] = float(crnd[idx_in_sub])
            ex["metadata_crnd_tfidf_embed"] = float(pair_te[idx_in_sub])
            ex["metadata_crnd_tfidf_llm"] = float(pair_tl[idx_in_sub])
            ex["metadata_crnd_embed_llm"] = float(pair_el[idx_in_sub])
            ex["metadata_kdn_tfidf"] = float(kdn_tfidf[idx_in_sub])
            ex["metadata_kdn_embed"] = float(kdn_embed[idx_in_sub])
            ex["metadata_kdn_llm"] = float(kdn_llm[idx_in_sub])
            ex["metadata_kdn_avg"] = float(kdn_avg[idx_in_sub])
            ex["metadata_cleanlab_tfidf"] = float(cleanlab_tfidf[idx_in_sub])
            ex["metadata_cleanlab_embed"] = float(cleanlab_embed[idx_in_sub])
            ex["metadata_cleanlab_avg"] = float(cleanlab_avg[idx_in_sub])
            ex["metadata_wann_adapted"] = float(wann[idx_in_sub])
            ex["metadata_cartography_score"] = float(carto_score[idx_in_sub])
            ex["metadata_cartography_confidence"] = float(carto_conf[idx_in_sub])
            ex["metadata_cartography_variability"] = float(carto_var[idx_in_sub])
            ex["metadata_label_entropy"] = float(label_entropy[idx_in_sub])
            ex["metadata_h1_score"] = float(hybrids["H1"][idx_in_sub])
            ex["metadata_h2_score"] = float(hybrids["H2"][idx_in_sub])
            ex["metadata_h3_score"] = float(hybrids["H3"][idx_in_sub])
            ex["metadata_h4_score"] = float(hybrids["H4"][idx_in_sub])

            # Quadrant assignment
            c_val = crnd[idx_in_sub]
            e_val = label_entropy[idx_in_sub]
            if c_val <= med_c and e_val <= med_e:
                quad = "Q1_easy"
            elif c_val > med_c and e_val <= med_e:
                quad = "Q2_repr_ambiguous"
            elif c_val <= med_c and e_val > med_e:
                quad = "Q3_standard_noise"
            else:
                quad = "Q4_genuinely_ambiguous"
            ex["metadata_quadrant"] = quad

            # predict_ fields (required by schema, must be strings)
            tercile = "high" if c_val > p66 else ("mid" if c_val > p33 else "low")
            ex["predict_crnd_noise_rank"] = tercile

            best_h = max(["H1", "H2", "H3", "H4"], key=lambda h: hybrids[h][idx_in_sub])
            ex["predict_best_hybrid"] = best_h

            ds_examples.append(ex)

        all_results["datasets"].append({"dataset": ds_name, "examples": ds_examples})

        ds_elapsed = time.time() - t_ds_start
        logger.info(f"  {ds_name} completed in {ds_elapsed:.0f}s ({len(ds_examples)} examples)")

        # Free memory between datasets
        del X_tfidf, X_embed, X_llm_sub, X_tfidf_sub, X_embed_sub
        gc.collect()

    # ---------------------------------------------------------------
    # Store metadata
    # ---------------------------------------------------------------
    total_examples = sum(len(ds["examples"]) for ds in all_results["datasets"])

    all_results["metadata"] = {
        "method_name": "CRND_Hybrid_Experiment",
        "description": (
            "Cross-Representation Neighborhood Dissonance with 4 hybrid formulations, "
            "noise detection evaluation, subpopulation analysis, and Schoener's D niche overlap. "
            "Feature spaces: TF-IDF (5000 features), sentence embeddings (384-dim, all-MiniLM-L6-v2), "
            "and LLM zero-shot / char n-gram fallback."
        ),
        "k_neighbors": K_NEIGHBORS,
        "noise_rates": NOISE_RATES,
        "n_seeds": N_SEEDS,
        "feature_spaces": ["tfidf_5000", "sentence_embedding_384", "llm_zero_shot_or_char_ngram"],
        "hybrid_variants": ["H1_mult_entropy", "H2_mult_kdn", "H3_rank_fusion", "H4_best_pair"],
        "datasets_processed": list(datasets.keys()),
        "total_examples": total_examples,
        "llm_calls_total": llm_calls_total,
        "llm_cost_usd": llm_cost_total,
        "aggregate_results": aggregate,
    }

    # ---------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------
    output = format_output(all_results)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))

    elapsed = time.time() - t_start
    logger.success(
        f"Saved method_out.json ({total_examples} examples across "
        f"{len(datasets)} datasets) in {elapsed:.0f}s"
    )
    logger.info(f"LLM calls: {llm_calls_total}, cost: ${llm_cost_total:.3f}")


if __name__ == "__main__":
    main()
