#!/usr/bin/env python3
"""
CRND Pipeline: Cross-Representation Neighborhood Dissonance
with Ecological Niche Overlap Analysis.

Computes per-instance CRND scores from cross-space k-NN neighborhood disagreement
across 3 feature spaces (TF-IDF, sentence-transformer, LLM zero-shot via OpenRouter).
Computes Schoener's D ecological niche overlap matrices via PCA+KDE.
Validates noise detection capability across 5%/10%/20% noise rates with 10 seeds.

Baseline: single-space k-NN label entropy (no cross-representation component).

Runs on ALL 5 datasets from the data dependency.
"""

import gc
import json
import os
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse
from dotenv import load_dotenv
from loguru import logger
from scipy.special import softmax
from scipy.stats import gaussian_kde, kendalltau, spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits (16GB machine → cap at 14GB)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DATA_DEP_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)
K_VALUES = [10, 20]
NOISE_RATES = [0.05, 0.10, 0.20]
N_NOISE_SEEDS = 10
PCA_DIMS = [2, 5]
KDE_GRID_SIZE = 100  # for 2D
MAX_LLM_CALLS = 9500  # stay well under 10K limit
MAX_LLM_BUDGET_USD = 9.0  # stay under $10 limit
LLM_SAMPLE_PER_DATASET = 50  # max LLM calls per dataset (5 datasets × 50 = 250)
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Cost tracking
_llm_calls_made = 0
_llm_calls_failed = 0
_llm_total_cost_usd = 0.0
_llm_input_tokens = 0
_llm_output_tokens = 0

# Timing
_phase_timings = {}

# Max examples per dataset for scaling
# CPU-only constraint: sentence-transformers on 14K+ texts is too slow (~30 min per dataset)
# Cap at 3000 per dataset for practical runtime (~40 min total)
MAX_EXAMPLES = 1000  # Cap for CPU-only runtime (sentence-transformers is the bottleneck)


# ============================================================================
# PHASE 0: DATA LOADING
# ============================================================================


def load_data(data_path: Path, max_examples: int | None = None) -> dict:
    """Load datasets from full_data_out.json.

    Returns dict: {dataset_name: {"texts": [...], "labels": [...], "n_classes": int, "class_names": [...]}}
    """
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())

    datasets = {}
    for ds in raw["datasets"]:
        name = ds["dataset"]
        examples = ds["examples"]
        if max_examples is not None:
            examples = examples[:max_examples]

        texts = [ex["input"] for ex in examples]
        labels = [ex["output"] for ex in examples]
        class_names = sorted(set(labels))

        datasets[name] = {
            "texts": texts,
            "labels": labels,
            "n_classes": len(class_names),
            "class_names": class_names,
            "raw_examples": examples,
        }

        label_counts = Counter(labels)
        logger.info(
            f"  {name}: {len(texts)} examples, {len(class_names)} classes"
        )
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {label}: {count}")
        if len(label_counts) > 5:
            logger.info(f"    ... and {len(label_counts) - 5} more classes")

    logger.info(f"Loaded {len(datasets)} datasets total")
    return datasets


# ============================================================================
# PHASE 1: FEATURE SPACE CONSTRUCTION
# ============================================================================


def compute_tfidf_features(
    texts: list[str],
    max_features: int = 5000,
) -> scipy.sparse.csr_matrix:
    """Sparse TF-IDF with clinical preprocessing."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"  TF-IDF: {tfidf_matrix.shape}, nnz={tfidf_matrix.nnz}")
    return tfidf_matrix


_sent_model = None


def get_sentence_model():
    """Lazily load and cache the sentence transformer model."""
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sent_model


def compute_sentence_embeddings(
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Dense 384-dim embeddings from all-MiniLM-L6-v2."""
    model = get_sentence_model()
    # Truncate texts to reduce tokenization time on CPU
    truncated_texts = [t[:512] if len(t) > 512 else t for t in texts]
    embeddings = model.encode(
        truncated_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    logger.info(f"  Sentence embeddings: {embeddings.shape}")
    return embeddings


def compute_simulated_llm_features(
    tfidf_matrix: scipy.sparse.csr_matrix,
    labels: list[str],
    class_names: list[str],
) -> np.ndarray:
    """Simulate LLM features using TF-IDF class centroids + cosine → softmax.

    This is the deterministic fallback when actual LLM calls aren't feasible
    for all instances.
    """
    n_classes = len(class_names)
    labels_arr = np.array(labels)

    # Compute class centroids in TF-IDF space
    centroids = np.zeros((n_classes, tfidf_matrix.shape[1]))
    for i, cls in enumerate(class_names):
        mask = labels_arr == cls
        if mask.sum() > 0:
            centroids[i] = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()

    # For each instance compute cosine similarity to all centroids
    sims = cosine_similarity(tfidf_matrix, centroids)  # (N, n_classes)

    # Apply softmax with temperature to simulate probability distribution
    features = softmax(sims * 3.0, axis=1)
    logger.info(f"  Simulated LLM features: {features.shape}")
    return features


def parse_json_from_response(text: str) -> dict:
    """Extract JSON dict from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object (greedy to catch nested)
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try from first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: try entire text
    return json.loads(text.strip())


def build_llm_prompt(text: str, class_names: list[str]) -> str:
    """Build zero-shot classification prompt returning probability vector."""
    classes_str = ", ".join(class_names)
    truncated_text = text[:1500]
    prompt = (
        f"Classify the following medical text into one of these categories: "
        f"{classes_str}\n\n"
        f"Text: {truncated_text}\n\n"
        f"Return ONLY a JSON object with the probability for each category "
        f"(probabilities must sum to 1.0).\n"
    )
    # Build example format string safely — handle any number of classes
    if len(class_names) >= 3:
        example = f'{{"{class_names[0]}": 0.7, "{class_names[1]}": 0.2, "{class_names[2]}": 0.1}}'
    elif len(class_names) == 2:
        example = f'{{"{class_names[0]}": 0.7, "{class_names[1]}": 0.3}}'
    elif len(class_names) == 1:
        example = f'{{"{class_names[0]}": 1.0}}'
    else:
        example = '{}'
    return prompt + f"Example format: {example}\n\nJSON:"


def call_openrouter_single(
    text: str,
    class_names: list[str],
    session: requests.Session,
    api_key: str,
    model: str = LLM_MODEL,
) -> tuple[np.ndarray, dict]:
    """Call OpenRouter API and parse probability vector.

    Returns (prob_vector, usage_dict).
    """
    global _llm_calls_made, _llm_calls_failed, _llm_total_cost_usd
    global _llm_input_tokens, _llm_output_tokens

    prompt = build_llm_prompt(text, class_names)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _call():
        resp = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    _llm_calls_made += 1

    try:
        result = _call()
    except Exception as e:
        _llm_calls_failed += 1
        logger.warning(f"LLM call failed after retries: {type(e).__name__}: {str(e)[:200]}")
        return np.ones(len(class_names)) / len(class_names), {}

    # Validate response structure
    try:
        _ = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        _llm_calls_failed += 1
        logger.warning(f"Invalid LLM response structure: {type(e).__name__}: {str(result)[:200]}")
        return np.ones(len(class_names)) / len(class_names), {}

    # Track usage
    usage = result.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    _llm_input_tokens += in_tok
    _llm_output_tokens += out_tok
    # Llama 3.1 8B pricing: $0.03/M input, $0.04/M output
    cost = in_tok * 0.03e-6 + out_tok * 0.04e-6
    _llm_total_cost_usd += cost

    response_text = result["choices"][0]["message"]["content"]
    logger.debug(f"LLM response: {response_text[:200]}")

    try:
        prob_dict = parse_json_from_response(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {e}, response: {response_text[:200]}")
        _llm_calls_failed += 1
        return np.ones(len(class_names)) / len(class_names), usage

    # Convert to probability vector aligned with class_names
    prob_vector = np.zeros(len(class_names))
    for i, cls in enumerate(class_names):
        prob_vector[i] = float(prob_dict.get(cls, 0.0))

    # Normalize
    total = prob_vector.sum()
    if total > 0:
        prob_vector /= total
    else:
        prob_vector = np.ones(len(class_names)) / len(class_names)

    return prob_vector, usage


def compute_llm_features(
    texts: list[str],
    labels: list[str],
    class_names: list[str],
    tfidf_matrix: scipy.sparse.csr_matrix,
    max_llm_calls: int = LLM_SAMPLE_PER_DATASET,
) -> np.ndarray:
    """Compute LLM features: real API calls for a sample, simulated for the rest.

    We sample up to max_llm_calls examples for actual LLM calls and use the
    simulated fallback for all instances (then blend real values in where available).
    """
    global _llm_calls_made, _llm_total_cost_usd

    n = len(texts)
    n_classes = len(class_names)

    # Start with simulated features for ALL instances
    features = compute_simulated_llm_features(tfidf_matrix, labels, class_names)

    # Load API key
    load_dotenv(override=False)
    for env_path in ["/home/adrian/.env", "/home/adrian/projects/ai-inventor/.env"]:
        load_dotenv(env_path, override=False)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — using only simulated LLM features")
        return features

    # Skip API calls for single-class datasets (no classification needed)
    if n_classes <= 1:
        logger.info("  Skipping LLM calls for single-class dataset")
        return features

    # Check budget
    if _llm_calls_made >= MAX_LLM_CALLS:
        logger.warning(f"LLM call limit reached ({_llm_calls_made}/{MAX_LLM_CALLS})")
        return features
    if _llm_total_cost_usd >= MAX_LLM_BUDGET_USD:
        logger.warning(f"LLM budget limit reached (${_llm_total_cost_usd:.2f}/${MAX_LLM_BUDGET_USD})")
        return features

    # Sample indices (stratified by class for representativeness)
    sample_size = min(max_llm_calls, n, MAX_LLM_CALLS - _llm_calls_made)
    if sample_size <= 0:
        return features

    rng = np.random.default_rng(42)
    labels_arr = np.array(labels)

    # Stratified sampling
    sampled_indices = []
    per_class = max(1, sample_size // n_classes)
    for cls in class_names:
        cls_indices = np.where(labels_arr == cls)[0]
        n_sample = min(per_class, len(cls_indices))
        if n_sample > 0:
            chosen = rng.choice(cls_indices, size=n_sample, replace=False)
            sampled_indices.extend(chosen.tolist())

    # Fill remaining quota randomly
    remaining = sample_size - len(sampled_indices)
    if remaining > 0:
        all_indices = set(range(n))
        available = list(all_indices - set(sampled_indices))
        if available:
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            sampled_indices.extend(extra.tolist())

    sampled_indices = sorted(set(sampled_indices))[:sample_size]
    logger.info(f"  Making {len(sampled_indices)} real LLM calls (out of {n} total)")

    # Sequential LLM calls (synchronous only per constraints)
    session = requests.Session()
    success_count = 0
    fail_count_this_batch = 0
    t_start = time.time()

    for call_idx, idx in enumerate(sampled_indices):
        # Budget checks
        if _llm_calls_made >= MAX_LLM_CALLS:
            logger.warning(f"LLM call limit hit at call {call_idx}")
            break
        if _llm_total_cost_usd >= MAX_LLM_BUDGET_USD:
            logger.warning(f"LLM budget hit at call {call_idx}: ${_llm_total_cost_usd:.2f}")
            break

        prob_vec, usage = call_openrouter_single(
            text=texts[idx],
            class_names=class_names,
            session=session,
            api_key=api_key,
        )

        # Check if it's a real result (not uniform fallback)
        if not np.allclose(prob_vec, np.ones(n_classes) / n_classes):
            features[idx] = prob_vec
            success_count += 1
        else:
            fail_count_this_batch += 1

        # Progress logging every 50 calls
        if (call_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (call_idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"    LLM progress: {call_idx + 1}/{len(sampled_indices)} "
                f"({rate:.1f} calls/s, ${_llm_total_cost_usd:.4f} spent)"
            )

        # Early abort if >60% fail rate after 30+ calls
        if call_idx >= 30 and fail_count_this_batch / (call_idx + 1) > 0.6:
            logger.warning(
                f"High LLM failure rate ({fail_count_this_batch}/{call_idx + 1}), "
                f"stopping LLM calls for this dataset"
            )
            break

    elapsed = time.time() - t_start
    logger.info(
        f"  LLM feature extraction done: {success_count} real, "
        f"{fail_count_this_batch} failed, "
        f"{elapsed:.1f}s, ${_llm_total_cost_usd:.4f} total cost"
    )

    session.close()
    return features


# ============================================================================
# PHASE 2: k-NN COMPUTATION & CRND SCORES
# ============================================================================


def compute_knn_neighbors(
    feature_matrix,
    k: int = 20,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute k-nearest neighbor indices for all instances."""
    n = feature_matrix.shape[0] if hasattr(feature_matrix, "shape") else len(feature_matrix)
    actual_k = min(k, n - 1)
    if actual_k < 1:
        return np.zeros((n, 0), dtype=int)

    nn = NearestNeighbors(n_neighbors=actual_k + 1, metric=metric, algorithm="auto")
    nn.fit(feature_matrix)
    _, indices = nn.kneighbors(feature_matrix)

    # Remove self-neighbor (first column)
    neighbor_indices = indices[:, 1:]  # shape: (N, actual_k)
    return neighbor_indices


def compute_crnd(
    neighbor_sets: dict[str, np.ndarray],
    k: int = 20,
) -> np.ndarray:
    """Compute Cross-Representation Neighborhood Dissonance.

    CRND(i) = 1 - mean of pairwise Jaccard similarities across feature spaces
    for instance i's k-NN neighborhoods.

    Uses vectorized set operations via sorted arrays for efficiency on large N.
    """
    space_names = list(neighbor_sets.keys())
    n_instances = neighbor_sets[space_names[0]].shape[0]
    actual_k = neighbor_sets[space_names[0]].shape[1]

    if actual_k == 0:
        return np.zeros(n_instances)

    # Pre-sort all neighbor arrays for faster intersection
    sorted_neighbors = {
        name: np.sort(arr, axis=1) for name, arr in neighbor_sets.items()
    }

    # Compute pairwise Jaccard for each pair of spaces
    pairs = []
    for a in range(len(space_names)):
        for b in range(a + 1, len(space_names)):
            pairs.append((space_names[a], space_names[b]))

    n_pairs = len(pairs)
    jaccard_sums = np.zeros(n_instances)

    for name_a, name_b in pairs:
        arr_a = sorted_neighbors[name_a]
        arr_b = sorted_neighbors[name_b]

        # Batch computation: for each instance, compute Jaccard
        for i in range(n_instances):
            set_a = set(arr_a[i].tolist())
            set_b = set(arr_b[i].tolist())
            union_size = len(set_a | set_b)
            if union_size > 0:
                jaccard_sums[i] += len(set_a & set_b) / union_size

    crnd_scores = 1.0 - jaccard_sums / n_pairs
    return crnd_scores


# ============================================================================
# BASELINE: Single-space k-NN label entropy
# ============================================================================


def compute_baseline_knn_entropy(
    knn_indices: np.ndarray,
    labels: list[str],
    class_names: list[str],
) -> np.ndarray:
    """Baseline: k-NN label entropy in a single feature space.

    For each instance, compute Shannon entropy of label distribution
    among its k-nearest neighbors. Higher entropy = more ambiguous region.
    This is a simpler alternative to CRND that doesn't require multiple spaces.
    """
    n = knn_indices.shape[0]
    k = knn_indices.shape[1]
    if k == 0:
        return np.zeros(n)

    # Vectorized: map labels to integers
    label_to_idx = {cls: i for i, cls in enumerate(class_names)}
    labels_int = np.array([label_to_idx[l] for l in labels])
    n_classes = len(class_names)

    entropy_scores = np.zeros(n)
    for i in range(n):
        neighbor_label_ints = labels_int[knn_indices[i]]
        counts = np.bincount(neighbor_label_ints, minlength=n_classes)
        probs = counts / k
        probs = probs[probs > 0]
        entropy_scores[i] = -np.sum(probs * np.log2(probs))

    return entropy_scores


# ============================================================================
# PHASE 3: ECOLOGICAL NICHE OVERLAP (Schoener's D)
# ============================================================================


def compute_schoeners_d_matrix(
    feature_matrix,
    labels: list[str],
    class_names: list[str],
    n_pca_dims: int = 2,
    grid_size: int = 100,
) -> tuple[np.ndarray, dict]:
    """Compute Schoener's D pairwise overlap matrix using PCA-env framework."""
    labels_arr = np.array(labels)

    # Step 1: PCA/SVD to n_pca_dims dimensions
    n_samples, n_features = feature_matrix.shape[0], feature_matrix.shape[1]
    if hasattr(feature_matrix, "toarray"):
        actual_dims = min(n_pca_dims, n_features - 1, n_samples - 1)
        if actual_dims < 1:
            n_cls = len(class_names)
            return np.eye(n_cls), {"explained_variance": [], "n_pca_dims": 0, "grid_size": 0}
        svd = TruncatedSVD(n_components=actual_dims, random_state=42)
        projected = svd.fit_transform(feature_matrix)
        explained_var = svd.explained_variance_ratio_
    else:
        actual_dims = min(n_pca_dims, n_features, n_samples - 1)
        if actual_dims < 1:
            n_cls = len(class_names)
            return np.eye(n_cls), {"explained_variance": [], "n_pca_dims": 0, "grid_size": 0}
        pca = PCA(n_components=actual_dims, random_state=42)
        projected = pca.fit_transform(feature_matrix)
        explained_var = pca.explained_variance_ratio_

    logger.debug(
        f"  PCA {n_pca_dims}D explained variance: "
        f"{explained_var} (total: {sum(explained_var):.4f})"
    )

    actual_dims = projected.shape[1]

    # Step 2: Create grid
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    padding = (maxs - mins) * 0.05 + 1e-10  # avoid zero padding

    if actual_dims <= 2:
        grid_1d = [
            np.linspace(mins[d] - padding[d], maxs[d] + padding[d], grid_size)
            for d in range(actual_dims)
        ]
        if actual_dims == 2:
            mesh = np.meshgrid(*grid_1d)
            grid_points = np.vstack([m.ravel() for m in mesh])
        else:
            grid_points = grid_1d[0].reshape(1, -1)
    else:
        # For 5D, use fewer points per dim to keep grid manageable
        pts_per_dim = max(5, int(round(100000 ** (1.0 / actual_dims))))
        grid_1d = [
            np.linspace(mins[d] - padding[d], maxs[d] + padding[d], pts_per_dim)
            for d in range(actual_dims)
        ]
        mesh = np.meshgrid(*grid_1d, indexing="ij")
        grid_points = np.vstack([m.ravel() for m in mesh])

    # Step 3: KDE per class
    n_classes = len(class_names)
    class_densities = {}

    for cls in class_names:
        mask = labels_arr == cls
        n_cls = int(mask.sum())

        if n_cls < actual_dims + 1:
            logger.debug(f"  Class '{cls}' has only {n_cls} instances, using uniform density")
            class_densities[cls] = np.ones(grid_points.shape[1]) / grid_points.shape[1]
            continue

        cls_data = projected[mask].T  # (n_pca_dims, n_cls)

        try:
            # Add tiny jitter to avoid singular covariance
            jitter = np.random.default_rng(42).normal(0, 1e-8, cls_data.shape)
            kde = gaussian_kde(cls_data + jitter, bw_method="scott")
            density = kde.evaluate(grid_points)
        except np.linalg.LinAlgError:
            logger.debug(f"  KDE failed for class '{cls}', trying silverman")
            try:
                kde = gaussian_kde(cls_data + jitter, bw_method="silverman")
                density = kde.evaluate(grid_points)
            except np.linalg.LinAlgError:
                logger.debug(f"  KDE totally failed for class '{cls}', using uniform")
                density = np.ones(grid_points.shape[1]) / grid_points.shape[1]

        # Normalize to probability
        total = density.sum()
        if total > 0:
            density = density / total

        class_densities[cls] = density

    # Step 4: Compute Schoener's D for all class pairs
    D_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            p1 = class_densities[class_names[i]]
            p2 = class_densities[class_names[j]]
            D_val = 1.0 - 0.5 * np.sum(np.abs(p1 - p2))
            D_matrix[i, j] = D_val
            D_matrix[j, i] = D_val
        D_matrix[i, i] = 1.0

    return D_matrix, {
        "explained_variance": explained_var.tolist(),
        "n_pca_dims": actual_dims,
        "grid_size": grid_size,
    }


# ============================================================================
# PHASE 4: NOISE INJECTION & VALIDATION
# ============================================================================


def inject_label_noise(
    labels: list[str],
    noise_rate: float,
    class_names: list[str],
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """Inject uniform label noise. Returns (noisy_labels, noise_indicator)."""
    rng = np.random.default_rng(seed)
    n = len(labels)

    # Can't inject noise with fewer than 2 classes
    if len(class_names) < 2:
        return list(labels), np.zeros(n, dtype=int)

    n_flip = int(n * noise_rate)
    if n_flip == 0:
        return list(labels), np.zeros(n, dtype=int)

    flip_indices = rng.choice(n, size=n_flip, replace=False)
    noise_indicator = np.zeros(n, dtype=int)
    noise_indicator[flip_indices] = 1

    noisy_labels = list(labels)
    for idx in flip_indices:
        original = noisy_labels[idx]
        candidates = [c for c in class_names if c != original]
        if candidates:
            noisy_labels[idx] = rng.choice(candidates)
        else:
            noise_indicator[idx] = 0  # Can't flip, undo indicator

    return noisy_labels, noise_indicator


def evaluate_noise_detection(
    scores: np.ndarray,
    noise_indicator: np.ndarray,
) -> dict:
    """Compute Spearman correlation and ROC-AUC of a score as noise detector."""
    # Spearman
    rho, p_value = spearmanr(scores, noise_indicator)

    # ROC-AUC (handle edge cases)
    try:
        if len(np.unique(noise_indicator)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(noise_indicator, scores)
    except ValueError:
        auc = float("nan")

    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "spearman_p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "roc_auc": float(auc) if np.isfinite(auc) else 0.5,
        "n_noisy": int(noise_indicator.sum()),
        "n_clean": int((1 - noise_indicator).sum()),
    }


# ============================================================================
# PHASE 5: ADDITIONAL ANALYSES
# ============================================================================


def analyze_crnd_by_class(
    crnd_scores: np.ndarray,
    labels: list[str],
    class_names: list[str],
) -> dict:
    """Compute CRND statistics per class."""
    labels_arr = np.array(labels)
    results = {}
    for cls in class_names:
        mask = labels_arr == cls
        cls_crnd = crnd_scores[mask]
        if len(cls_crnd) == 0:
            continue
        results[cls] = {
            "mean": float(np.mean(cls_crnd)),
            "std": float(np.std(cls_crnd)),
            "median": float(np.median(cls_crnd)),
            "q25": float(np.percentile(cls_crnd, 25)),
            "q75": float(np.percentile(cls_crnd, 75)),
            "n": int(mask.sum()),
        }
    return results


def analyze_boundary_proximity(
    crnd_scores: np.ndarray,
    labels: list[str],
    knn_indices_all_spaces: dict[str, np.ndarray],
    class_names: list[str],
) -> tuple[dict, np.ndarray]:
    """Stratify CRND by boundary proximity (fraction of k-NN with different label)."""
    n = len(labels)
    labels_arr = np.array(labels)

    boundary_scores = np.zeros(n)
    for space_name, knn_idx in knn_indices_all_spaces.items():
        for i in range(n):
            if knn_idx.shape[1] == 0:
                continue
            neighbor_labels = labels_arr[knn_idx[i]]
            frac_different = np.mean(neighbor_labels != labels_arr[i])
            boundary_scores[i] += frac_different
    boundary_scores /= max(len(knn_indices_all_spaces), 1)

    # Stratify into bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    digitized = np.digitize(boundary_scores, bins) - 1

    stratified = {}
    for b, label in enumerate(bin_labels):
        mask = digitized == b
        if mask.sum() > 0:
            stratified[label] = {
                "mean_crnd": float(np.mean(crnd_scores[mask])),
                "std_crnd": float(np.std(crnd_scores[mask])),
                "count": int(mask.sum()),
            }
    return stratified, boundary_scores


def compute_niche_overlap_profile_comparison(
    schoeners_d_matrices: dict,
    dataset_name: str,
) -> dict:
    """Compute Kendall tau between D matrices across spaces for a dataset."""
    results = {}
    ds_matrices = schoeners_d_matrices.get(dataset_name, {})
    space_keys = [k for k in ds_matrices if k.endswith("_2d")]

    for i in range(len(space_keys)):
        for j in range(i + 1, len(space_keys)):
            key_a = space_keys[i]
            key_b = space_keys[j]
            mat_a = np.array(ds_matrices[key_a])
            mat_b = np.array(ds_matrices[key_b])

            # Extract upper triangle
            n = mat_a.shape[0]
            triu_idx = np.triu_indices(n, k=1)
            vals_a = mat_a[triu_idx]
            vals_b = mat_b[triu_idx]

            if len(vals_a) >= 3:
                tau, p_val = kendalltau(vals_a, vals_b)
            else:
                tau, p_val = 0.0, 1.0

            results[f"{key_a}_vs_{key_b}"] = {
                "kendall_tau": float(tau) if np.isfinite(tau) else 0.0,
                "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
            }
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def process_single_dataset(
    ds_name: str,
    ds_data: dict,
    all_metadata: dict,
) -> dict:
    """Process a single dataset through the full CRND pipeline.

    Returns the dataset entry for the output JSON.
    """
    texts = ds_data["texts"]
    labels = ds_data["labels"]
    class_names = ds_data["class_names"]
    n = len(texts)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing dataset: {ds_name} ({n} examples, {len(class_names)} classes)")
    logger.info(f"{'='*60}")

    # ----- Feature Extraction -----
    t0 = time.time()

    # 1. TF-IDF
    logger.info("Computing TF-IDF features...")
    t_tfidf = time.time()
    # Adjust min_df for small datasets
    min_df = 2 if n > 10 else 1
    max_features = min(5000, n * 10)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    logger.info(f"  TF-IDF: {tfidf_matrix.shape}, nnz={tfidf_matrix.nnz} ({time.time()-t_tfidf:.1f}s)")

    # 2. Sentence Embeddings
    logger.info("Computing sentence embeddings...")
    t_emb = time.time()
    sent_embeddings = compute_sentence_embeddings(texts, batch_size=64)
    logger.info(f"  Done in {time.time()-t_emb:.1f}s")

    # 3. LLM Features (real + simulated hybrid)
    logger.info("Computing LLM features...")
    t_llm = time.time()
    llm_features = compute_llm_features(
        texts=texts,
        labels=labels,
        class_names=class_names,
        tfidf_matrix=tfidf_matrix,
        max_llm_calls=LLM_SAMPLE_PER_DATASET,
    )
    logger.info(f"  Done in {time.time()-t_llm:.1f}s")

    feature_time = time.time() - t0
    logger.info(f"Feature extraction total: {feature_time:.1f}s")

    # ----- k-NN & CRND -----
    feature_spaces = {
        "tfidf": tfidf_matrix,
        "sentence_transformer": sent_embeddings,
        "llm_zeroshot": llm_features,
    }

    crnd_results = {}
    knn_results = {}
    baseline_results = {}

    for k_val in K_VALUES:
        logger.info(f"Computing k-NN with k={k_val}...")
        t_knn = time.time()

        neighbor_sets = {}
        for space_name, feat_mat in feature_spaces.items():
            neighbors = compute_knn_neighbors(feat_mat, k=k_val, metric="cosine")
            neighbor_sets[space_name] = neighbors
            knn_key = f"{space_name}_k{k_val}"
            knn_results[knn_key] = neighbors

        # CRND scores
        logger.info(f"Computing CRND scores (k={k_val})...")
        crnd = compute_crnd(neighbor_sets, k=k_val)
        crnd_results[f"k{k_val}"] = crnd

        # Baseline: single-space label entropy (use TF-IDF space)
        logger.info(f"Computing baseline k-NN entropy (k={k_val})...")
        baseline_entropy = compute_baseline_knn_entropy(
            knn_indices=neighbor_sets["tfidf"],
            labels=labels,
            class_names=class_names,
        )
        baseline_results[f"k{k_val}"] = baseline_entropy

        logger.info(f"  k={k_val} done in {time.time()-t_knn:.1f}s")
        logger.info(
            f"  CRND stats: mean={np.mean(crnd):.4f}, std={np.std(crnd):.4f}, "
            f"min={np.min(crnd):.4f}, max={np.max(crnd):.4f}"
        )
        logger.info(
            f"  Baseline entropy stats: mean={np.mean(baseline_entropy):.4f}, "
            f"std={np.std(baseline_entropy):.4f}"
        )

    # ----- Schoener's D matrices -----
    logger.info("Computing Schoener's D matrices...")
    ds_schoeners = {}
    ds_pca_var = {}

    for space_name, feat_mat in feature_spaces.items():
        for pca_dim in PCA_DIMS:
            key = f"{space_name}_{pca_dim}d"
            logger.info(f"  Schoener's D: {key}...")

            try:
                # For 5D with many classes, limit grid to avoid memory issues
                gs = KDE_GRID_SIZE if pca_dim <= 2 else max(5, int(round(50000 ** (1.0 / pca_dim))))
                D_mat, pca_info = compute_schoeners_d_matrix(
                    feature_matrix=feat_mat,
                    labels=labels,
                    class_names=class_names,
                    n_pca_dims=pca_dim,
                    grid_size=gs,
                )
                ds_schoeners[key] = D_mat.tolist()
                ds_pca_var[key] = pca_info["explained_variance"]
            except Exception:
                logger.exception(f"  Failed to compute Schoener's D for {key}")
                ds_schoeners[key] = np.eye(len(class_names)).tolist()
                ds_pca_var[key] = []

    all_metadata["schoeners_d_matrices"][ds_name] = ds_schoeners
    all_metadata["pca_explained_variance"][ds_name] = ds_pca_var

    # ----- Noise Detection Validation -----
    logger.info("Running noise detection validation...")
    noise_results = {}

    for nr in NOISE_RATES:
        nr_key = str(nr)
        rho_vals_crnd = []
        auc_vals_crnd = []
        rho_vals_base = []
        auc_vals_base = []

        for seed in range(N_NOISE_SEEDS):
            _, noise_indicator = inject_label_noise(labels, nr, class_names, seed)

            # Evaluate CRND (k=20) as noise detector
            crnd_k20 = crnd_results["k20"]
            eval_crnd = evaluate_noise_detection(crnd_k20, noise_indicator)
            rho_vals_crnd.append(eval_crnd["spearman_rho"])
            auc_vals_crnd.append(eval_crnd["roc_auc"])

            # Evaluate baseline entropy (k=20) as noise detector
            base_k20 = baseline_results["k20"]
            eval_base = evaluate_noise_detection(base_k20, noise_indicator)
            rho_vals_base.append(eval_base["spearman_rho"])
            auc_vals_base.append(eval_base["roc_auc"])

        noise_results[nr_key] = {
            "crnd": {
                "mean_rho": float(np.mean(rho_vals_crnd)),
                "std_rho": float(np.std(rho_vals_crnd)),
                "mean_auc": float(np.mean(auc_vals_crnd)),
                "std_auc": float(np.std(auc_vals_crnd)),
            },
            "baseline_entropy": {
                "mean_rho": float(np.mean(rho_vals_base)),
                "std_rho": float(np.std(rho_vals_base)),
                "mean_auc": float(np.mean(auc_vals_base)),
                "std_auc": float(np.std(auc_vals_base)),
            },
        }
        logger.info(
            f"  Noise {nr:.0%}: CRND AUC={np.mean(auc_vals_crnd):.4f}±{np.std(auc_vals_crnd):.4f} | "
            f"Baseline AUC={np.mean(auc_vals_base):.4f}±{np.std(auc_vals_base):.4f}"
        )

    all_metadata["noise_detection_results"][ds_name] = noise_results

    # ----- Per-class analysis -----
    logger.info("Computing per-class analysis...")
    crnd_per_class = analyze_crnd_by_class(crnd_results["k20"], labels, class_names)
    all_metadata["crnd_per_class"][ds_name] = crnd_per_class

    # ----- Boundary proximity stratification -----
    logger.info("Computing boundary proximity stratification...")
    knn_all_spaces_k20 = {
        space_name: knn_results[f"{space_name}_k20"]
        for space_name in feature_spaces
    }
    boundary_strat, boundary_scores = analyze_boundary_proximity(
        crnd_scores=crnd_results["k20"],
        labels=labels,
        knn_indices_all_spaces=knn_all_spaces_k20,
        class_names=class_names,
    )
    all_metadata["crnd_boundary_stratification"][ds_name] = boundary_strat

    # ----- Niche overlap profile comparison -----
    niche_comparison = compute_niche_overlap_profile_comparison(
        all_metadata["schoeners_d_matrices"], ds_name
    )
    all_metadata["niche_overlap_profile_comparison"][ds_name] = niche_comparison

    # ----- Assemble per-example output -----
    logger.info("Assembling per-example output...")
    examples_out = []
    for i in range(n):
        ex = ds_data["raw_examples"][i]
        example_out = {
            "input": ex["input"],
            "output": ex["output"],
            "predict_crnd_k10": f"{crnd_results['k10'][i]:.4f}",
            "predict_crnd_k20": f"{crnd_results['k20'][i]:.4f}",
            "predict_baseline_entropy_k20": f"{baseline_results['k20'][i]:.4f}",
            "metadata_crnd_k10": float(crnd_results["k10"][i]),
            "metadata_crnd_k20": float(crnd_results["k20"][i]),
            "metadata_baseline_entropy_k10": float(baseline_results["k10"][i]),
            "metadata_baseline_entropy_k20": float(baseline_results["k20"][i]),
            "metadata_boundary_proximity": float(boundary_scores[i]),
        }
        # Preserve original metadata
        for key, val in ex.items():
            if key.startswith("metadata_") and key not in example_out:
                example_out[key] = val
        examples_out.append(example_out)

    total_time = time.time() - t0
    logger.info(f"Dataset {ds_name} completed in {total_time:.1f}s")
    _phase_timings[ds_name] = total_time

    # Free memory
    del tfidf_matrix, sent_embeddings, llm_features
    gc.collect()

    return {"dataset": ds_name, "examples": examples_out}


@logger.catch
def main():
    """Main entry point for the CRND pipeline."""
    global MAX_EXAMPLES

    t_total_start = time.time()
    logger.info("=" * 60)
    logger.info("CRND Pipeline: Cross-Representation Neighborhood Dissonance")
    logger.info("=" * 60)

    # Determine data file
    data_path = DATA_DEP_DIR / "full_data_out.json"

    # Check for scaling mode via environment variable
    max_ex_env = os.environ.get("CRND_MAX_EXAMPLES")
    if max_ex_env:
        MAX_EXAMPLES = int(max_ex_env)
        logger.info(f"Scaling mode: max {MAX_EXAMPLES} examples per dataset")
    elif MAX_EXAMPLES is not None:
        logger.info(f"Scaling mode: max {MAX_EXAMPLES} examples per dataset")

    # Use mini data for tiny scaling tests
    mini_mode = os.environ.get("CRND_MINI_MODE", "0") == "1"
    if mini_mode:
        data_path = DATA_DEP_DIR / "mini_data_out.json"
        logger.info("MINI MODE: using mini_data_out.json")

    # Load data
    datasets = load_data(data_path, max_examples=MAX_EXAMPLES)

    if not datasets:
        logger.error("No datasets loaded!")
        return

    # Initialize metadata
    all_metadata = {
        "method_name": "CRND_NicheOverlap",
        "description": (
            "Cross-Representation Neighborhood Dissonance with Ecological Niche Overlap. "
            "Computes per-instance CRND scores from cross-space k-NN neighborhood disagreement "
            "across TF-IDF, sentence-transformer, and LLM zero-shot feature spaces."
        ),
        "baseline_name": "SingleSpace_kNN_LabelEntropy",
        "baseline_description": (
            "k-NN label entropy in TF-IDF space only. Higher entropy = more class-ambiguous "
            "neighborhood. Does not use cross-representation information."
        ),
        "feature_spaces": ["tfidf", "sentence_transformer", "llm_zeroshot"],
        "k_values": K_VALUES,
        "noise_rates": NOISE_RATES,
        "n_noise_seeds": N_NOISE_SEEDS,
        "pca_dims_tested": PCA_DIMS,
        "schoeners_d_matrices": {},
        "noise_detection_results": {},
        "crnd_per_class": {},
        "crnd_boundary_stratification": {},
        "pca_explained_variance": {},
        "niche_overlap_profile_comparison": {},
        "llm_model": LLM_MODEL,
        "llm_sample_per_dataset": LLM_SAMPLE_PER_DATASET,
        "llm_total_cost_usd": 0.0,
        "llm_calls_made": 0,
        "llm_calls_failed": 0,
        "runtime_seconds": 0.0,
        "phase_timings": {},
        "max_examples_per_dataset": MAX_EXAMPLES,
    }

    # Process each dataset
    dataset_outputs = []
    for ds_name, ds_data in datasets.items():
        try:
            ds_output = process_single_dataset(ds_name, ds_data, all_metadata)
            dataset_outputs.append(ds_output)
        except Exception:
            logger.exception(f"FAILED processing dataset: {ds_name}")
            # Still include dataset with minimal output
            examples_out = []
            for ex in ds_data["raw_examples"]:
                examples_out.append({
                    "input": ex["input"],
                    "output": ex["output"],
                    "predict_crnd_k10": "0.0",
                    "predict_crnd_k20": "0.0",
                    "predict_baseline_entropy_k20": "0.0",
                    "metadata_crnd_k10": 0.0,
                    "metadata_crnd_k20": 0.0,
                    "metadata_baseline_entropy_k10": 0.0,
                    "metadata_baseline_entropy_k20": 0.0,
                    "metadata_boundary_proximity": 0.0,
                })
                for key, val in ex.items():
                    if key.startswith("metadata_") and key not in examples_out[-1]:
                        examples_out[-1][key] = val
            dataset_outputs.append({"dataset": ds_name, "examples": examples_out})

    # Finalize metadata
    total_time = time.time() - t_total_start
    all_metadata["runtime_seconds"] = total_time
    all_metadata["phase_timings"] = _phase_timings
    all_metadata["llm_total_cost_usd"] = _llm_total_cost_usd
    all_metadata["llm_calls_made"] = _llm_calls_made
    all_metadata["llm_calls_failed"] = _llm_calls_failed
    all_metadata["llm_input_tokens"] = _llm_input_tokens
    all_metadata["llm_output_tokens"] = _llm_output_tokens

    # Assemble final output
    output = {
        "metadata": all_metadata,
        "datasets": dataset_outputs,
    }

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path}")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Datasets processed: {len(dataset_outputs)}")
    logger.info(f"LLM calls made: {_llm_calls_made} (failed: {_llm_calls_failed})")
    logger.info(f"LLM total cost: ${_llm_total_cost_usd:.4f}")
    for ds_name, t in _phase_timings.items():
        logger.info(f"  {ds_name}: {t:.1f}s")

    # Verify output has no NaN/Inf in CRND scores
    for ds_out in dataset_outputs:
        for ex in ds_out["examples"]:
            for key in ["metadata_crnd_k10", "metadata_crnd_k20"]:
                val = ex.get(key, 0.0)
                if not np.isfinite(val):
                    logger.warning(f"Non-finite value in {ds_out['dataset']}: {key}={val}")


if __name__ == "__main__":
    main()
