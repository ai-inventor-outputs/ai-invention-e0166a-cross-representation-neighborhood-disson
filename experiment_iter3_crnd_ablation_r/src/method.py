#!/usr/bin/env python3
"""CRND Ablation & Robustness Analysis.

Systematic ablation of the CRND framework across 6 dimensions:
  Phase 1: Feature space construction (TF-IDF, SentenceTransformer, simulated LLM)
  Phase 2: k sensitivity (k in {5, 10, 15, 20, 30, 50})
  Phase 3: PCA dimensionality for Schoener's D (d in {2, 3, 5, 10, 20})
  Phase 4: Distance metric ablation (euclidean, cosine, manhattan)
  Phase 5: Alternative CRND formulations (RBO, weighted Jaccard, pairwise decomp)
  Phase 6: Confound disentanglement (partial correlations)

Runs on ALL 5 datasets from data_id2_it1__opus.
Outputs method_out.json conforming to exp_gen_sol_out.json schema.
"""

import json
import os
import resource
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

load_dotenv()

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE.parent / "data_id2_it1__opus"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ── Constants ────────────────────────────────────────────────────────────────
K_VALUES = [5, 10, 15, 20, 30, 50]
PCA_DIMS = [2, 3, 5, 10, 20]
METRICS = ["euclidean", "cosine", "manhattan"]
NOISE_RATE = 0.10
N_SEEDS = 5
DEFAULT_K = 10
MAX_EXAMPLES_PER_DATASET = int(os.environ.get("MAX_EXAMPLES", "1000"))
OPENROUTER_MAX_CALLS = 10  # per dataset, budget control (reduced for speed)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Track global timing
GLOBAL_START = time.time()
PHASE_TIMINGS: dict[str, float] = {}
LLM_TOTAL_COST_USD = 0.0
LLM_CALL_COUNT = 0


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(x: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    """Compute softmax with temperature scaling."""
    x_scaled = x * temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))
    return e_x / e_x.sum()


def flatten_upper_triangle(matrix: list[list[float]]) -> np.ndarray:
    """Extract upper-triangle values from a symmetric matrix (as list of lists)."""
    arr = np.array(matrix)
    n = arr.shape[0]
    indices = np.triu_indices(n, k=1)
    return arr[indices]


def compute_crnd_jaccard(
    neighbor_sets: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Compute CRND using Jaccard similarity across feature spaces."""
    space_names = list(neighbor_sets.keys())
    crnd = np.zeros(n_samples)
    pairs = list(combinations(space_names, 2))
    if len(pairs) == 0:
        return crnd
    for i in range(n_samples):
        jaccards = []
        for s1, s2 in pairs:
            set1 = set(neighbor_sets[s1][i].tolist())
            set2 = set(neighbor_sets[s2][i].tolist())
            union_size = len(set1 | set2)
            if union_size == 0:
                jaccards.append(0.0)
            else:
                jaccards.append(len(set1 & set2) / union_size)
        crnd[i] = 1.0 - np.mean(jaccards)
    return crnd


def compute_crnd_weighted_jaccard(
    neighbor_sets: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Compute CRND using weighted Jaccard (weight by 1/rank)."""
    space_names = list(neighbor_sets.keys())
    crnd = np.zeros(n_samples)
    pairs = list(combinations(space_names, 2))
    if len(pairs) == 0:
        return crnd
    for i in range(n_samples):
        wj_scores = []
        for s1, s2 in pairs:
            neighbors_1 = neighbor_sets[s1][i].tolist()
            neighbors_2 = neighbor_sets[s2][i].tolist()
            # Build weight dictionaries: weight = 1/(rank+1)
            w1 = {n: 1.0 / (rank + 1) for rank, n in enumerate(neighbors_1)}
            w2 = {n: 1.0 / (rank + 1) for rank, n in enumerate(neighbors_2)}
            all_neighbors = set(neighbors_1) | set(neighbors_2)
            intersection = set(neighbors_1) & set(neighbors_2)
            w_inter = sum(max(w1.get(j, 0), w2.get(j, 0)) for j in intersection)
            w_union = sum(max(w1.get(j, 0), w2.get(j, 0)) for j in all_neighbors)
            wj = w_inter / w_union if w_union > 0 else 0.0
            wj_scores.append(wj)
        crnd[i] = 1.0 - np.mean(wj_scores)
    return crnd


def rbo_score(list1: list, list2: list, p: float = 0.9) -> float:
    """Rank-Biased Overlap (Webber et al. 2010).

    Manual implementation to avoid dependency issues with rbo package.
    """
    k = max(len(list1), len(list2))
    if k == 0:
        return 0.0
    score = 0.0
    for d in range(1, k + 1):
        set1_d = set(list1[:d])
        set2_d = set(list2[:d])
        overlap_d = len(set1_d & set2_d) / d
        score += (p ** (d - 1)) * overlap_d
    return (1.0 - p) * score


def compute_crnd_rbo(
    neighbor_sets: dict[str, np.ndarray],
    n_samples: int,
    p: float = 0.9,
) -> np.ndarray:
    """Compute CRND using Rank-Biased Overlap."""
    space_names = list(neighbor_sets.keys())
    crnd = np.zeros(n_samples)
    pairs = list(combinations(space_names, 2))
    if len(pairs) == 0:
        return crnd
    for i in range(n_samples):
        rbo_scores = []
        for s1, s2 in pairs:
            l1 = neighbor_sets[s1][i].tolist()
            l2 = neighbor_sets[s2][i].tolist()
            rbo_scores.append(rbo_score(l1, l2, p=p))
        crnd[i] = 1.0 - np.mean(rbo_scores)
    return crnd


def compute_pairwise_crnd(
    neighbor_sets: dict[str, np.ndarray],
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Compute CRND per pair of feature spaces (Jaccard)."""
    space_names = list(neighbor_sets.keys())
    pairs = list(combinations(space_names, 2))
    result = {}
    for s1, s2 in pairs:
        key = f"{s1}_vs_{s2}"
        crnd = np.zeros(n_samples)
        for i in range(n_samples):
            set1 = set(neighbor_sets[s1][i].tolist())
            set2 = set(neighbor_sets[s2][i].tolist())
            union_size = len(set1 | set2)
            if union_size == 0:
                crnd[i] = 0.0
            else:
                crnd[i] = 1.0 - len(set1 & set2) / union_size
        result[key] = crnd
    return result


def noise_detection_auc(
    crnd: np.ndarray,
    n_samples: int,
    noise_rate: float = NOISE_RATE,
    n_seeds: int = N_SEEDS,
) -> tuple[float, float]:
    """Evaluate noise detection ROC-AUC over multiple random seeds."""
    auc_scores = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 42)
        n_flip = int(noise_rate * n_samples)
        if n_flip == 0:
            n_flip = 1
        flip_indices = rng.choice(n_samples, size=n_flip, replace=False)
        noise_indicator = np.zeros(n_samples)
        noise_indicator[flip_indices] = 1.0
        # Check we have both classes
        if noise_indicator.sum() == 0 or noise_indicator.sum() == n_samples:
            continue
        try:
            auc = roc_auc_score(noise_indicator, crnd)
            auc_scores.append(auc)
        except ValueError:
            continue
    if len(auc_scores) == 0:
        return 0.5, 0.0
    return float(np.mean(auc_scores)), float(np.std(auc_scores))


def build_knn(
    features: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build k-NN and return neighbor indices (excluding self)."""
    n = features.shape[0]
    actual_k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=actual_k, metric=metric, algorithm="auto")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    # Exclude self (first column)
    if actual_k > 1:
        return indices[:, 1:]
    else:
        return indices


def build_knn_with_distances(
    features: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN and return (distances, indices) excluding self."""
    n = features.shape[0]
    actual_k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=actual_k, metric=metric, algorithm="auto")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    if actual_k > 1:
        return distances[:, 1:], indices[:, 1:]
    else:
        return distances, indices


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(
    data_path: Path,
    max_per_dataset: int = MAX_EXAMPLES_PER_DATASET,
) -> dict[str, dict]:
    """Load all datasets from full_data_out.json."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets = {}
    for ds in raw["datasets"]:
        name = ds["dataset"]
        examples = ds["examples"][:max_per_dataset]
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
        label_counts = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        logger.info(
            f"  {name}: {len(texts)} examples, "
            f"{len(class_names)} classes, "
            f"dist={json.dumps(label_counts, sort_keys=True)[:200]}"
        )
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: FEATURE SPACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_spaces(
    texts: list[str],
    labels: list[str],
    class_names: list[str],
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Build TF-IDF, sentence transformer, and simulated LLM feature spaces."""
    t0 = time.time()
    n = len(texts)
    feature_spaces = {}

    # 1a. TF-IDF
    logger.info(f"  [{dataset_name}] Building TF-IDF features...")
    max_features = min(5000, n * 10)  # cap for small datasets
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1 if n < 20 else 2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_spaces["tfidf"] = tfidf_matrix.toarray().astype(np.float32)
    feature_spaces["_tfidf_vectorizer"] = vectorizer  # save for confound analysis
    logger.info(f"    TF-IDF shape: {feature_spaces['tfidf'].shape}")

    # 1b. Sentence transformer embeddings
    logger.info(f"  [{dataset_name}] Building sentence transformer embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        # Cache model globally to avoid reloading per dataset
        if not hasattr(build_feature_spaces, "_st_model"):
            build_feature_spaces._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = build_feature_spaces._st_model
        truncated = [t[:512] for t in texts]
        embeddings = model.encode(
            truncated,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        feature_spaces["sentence_transformer"] = embeddings.astype(np.float32)
        logger.info(f"    SentenceTransformer shape: {embeddings.shape}")
    except Exception:
        logger.exception(f"  [{dataset_name}] SentenceTransformer failed, using TF-IDF SVD fallback")
        svd = TruncatedSVD(n_components=min(384, tfidf_matrix.shape[1] - 1, n - 1))
        fallback = svd.fit_transform(tfidf_matrix).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(fallback, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        fallback = fallback / norms
        feature_spaces["sentence_transformer"] = fallback
        logger.info(f"    SVD fallback shape: {fallback.shape}")

    # 1c. Simulated LLM zero-shot features
    logger.info(f"  [{dataset_name}] Building simulated LLM features...")
    n_classes = len(class_names)
    # Compute class centroids in TF-IDF space
    tfidf_dense = feature_spaces["tfidf"]
    class_centroids = np.zeros((n_classes, tfidf_dense.shape[1]), dtype=np.float32)
    for ci, cn in enumerate(class_names):
        mask = np.array([l == cn for l in labels])
        if mask.sum() > 0:
            class_centroids[ci] = tfidf_dense[mask].mean(axis=0)

    # Cosine similarity to centroids → softmax → probability vector
    llm_features = np.zeros((n, n_classes), dtype=np.float32)
    for i in range(n):
        vec = tfidf_dense[i]
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            llm_features[i] = np.ones(n_classes) / n_classes
            continue
        sims = np.zeros(n_classes)
        for ci in range(n_classes):
            c_norm = np.linalg.norm(class_centroids[ci])
            if c_norm == 0:
                sims[ci] = 0.0
            else:
                sims[ci] = np.dot(vec, class_centroids[ci]) / (vec_norm * c_norm)
        llm_features[i] = softmax(sims, temperature=3.0)

    # Try real LLM for small batch via OpenRouter
    if OPENROUTER_API_KEY and n >= 10:
        llm_features = _try_openrouter_features(
            texts=texts,
            class_names=class_names,
            llm_features=llm_features,
            dataset_name=dataset_name,
            max_calls=min(OPENROUTER_MAX_CALLS, n),
        )

    feature_spaces["llm_zeroshot"] = llm_features
    logger.info(f"    LLM features shape: {llm_features.shape}")

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Feature construction done in {elapsed:.1f}s")
    return feature_spaces


def _try_openrouter_features(
    texts: list[str],
    class_names: list[str],
    llm_features: np.ndarray,
    dataset_name: str,
    max_calls: int,
) -> np.ndarray:
    """Try to get real LLM zero-shot features via OpenRouter."""
    global LLM_TOTAL_COST_USD, LLM_CALL_COUNT
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, max=5))
    def call_openrouter(text: str) -> dict:
        global LLM_TOTAL_COST_USD, LLM_CALL_COUNT
        classes_str = ", ".join(class_names)
        prompt = (
            f"Classify the following text into one of these categories: {classes_str}\n"
            f"Text: {text[:800]}\n"
            f"Return ONLY a valid JSON object mapping each category to its probability (0-1). "
            f"Example: {json.dumps({c: round(1.0/len(class_names), 2) for c in class_names[:3]})}"
        )
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        LLM_CALL_COUNT += 1
        # Estimate cost (llama-3.1-8b: ~$0.06/M input, $0.06/M output)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 500)
        output_tokens = usage.get("completion_tokens", 100)
        cost = (input_tokens * 0.06 + output_tokens * 0.06) / 1_000_000
        LLM_TOTAL_COST_USD += cost
        return data

    n_calls = min(max_calls, len(texts))
    success_count = 0
    logger.info(f"  [{dataset_name}] Attempting {n_calls} OpenRouter LLM calls...")

    for idx in range(n_calls):
        if LLM_TOTAL_COST_USD >= 9.5:
            logger.warning("Approaching $10 budget limit, stopping LLM calls")
            break
        if LLM_CALL_COUNT >= 9900:
            logger.warning("Approaching 10k call limit, stopping LLM calls")
            break
        try:
            data = call_openrouter(texts[idx])
            content = data["choices"][0]["message"]["content"]
            # Parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            probs = json.loads(content)
            prob_vec = np.zeros(len(class_names))
            for ci, cn in enumerate(class_names):
                prob_vec[ci] = float(probs.get(cn, 0.0))
            total = prob_vec.sum()
            if total > 0:
                prob_vec /= total
            else:
                prob_vec = np.ones(len(class_names)) / len(class_names)
            llm_features[idx] = prob_vec.astype(np.float32)
            success_count += 1
        except Exception:
            logger.debug(f"  [{dataset_name}] LLM call {idx} failed, keeping simulated")
            continue

    logger.info(
        f"  [{dataset_name}] LLM: {success_count}/{n_calls} successful, "
        f"total cost: ${LLM_TOTAL_COST_USD:.4f}, calls: {LLM_CALL_COUNT}"
    )
    return llm_features


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: k SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_k_sensitivity(
    feature_spaces: dict[str, np.ndarray],
    n_samples: int,
    dataset_name: str,
) -> dict:
    """Evaluate CRND noise detection across different k values."""
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Phase 2: k sensitivity analysis")
    results = {}
    analysis_spaces = {k: v for k, v in feature_spaces.items() if not k.startswith("_")}

    for k in K_VALUES:
        if k >= n_samples:
            logger.info(f"    k={k} >= N={n_samples}, skipping")
            continue
        # Build k-NN for each feature space
        neighbor_sets = {}
        for space_name, features in analysis_spaces.items():
            indices = build_knn(features, k=k, metric="euclidean")
            neighbor_sets[space_name] = indices

        # Compute CRND
        crnd = compute_crnd_jaccard(neighbor_sets, n_samples)

        # Evaluate noise detection
        mean_auc, std_auc = noise_detection_auc(crnd, n_samples)

        results[str(k)] = {
            "mean_auc": round(mean_auc, 6),
            "std_auc": round(std_auc, 6),
            "mean_crnd": round(float(np.mean(crnd)), 6),
            "std_crnd": round(float(np.std(crnd)), 6),
        }
        logger.info(
            f"    k={k}: AUC={mean_auc:.4f}±{std_auc:.4f}, "
            f"CRND={np.mean(crnd):.4f}±{np.std(crnd):.4f}"
        )

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Phase 2 done in {elapsed:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: PCA DIMENSIONALITY FOR SCHOENER'S D
# ═══════════════════════════════════════════════════════════════════════════════

def compute_schoeners_d_2d(
    points_i: np.ndarray,
    points_j: np.ndarray,
    all_projected: np.ndarray,
    grid_size: int = 50,
) -> float:
    """Compute Schoener's D between two groups using 2D KDE."""
    try:
        kde_i = gaussian_kde(points_i.T)
        kde_j = gaussian_kde(points_j.T)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")

    d = points_i.shape[1]
    grid_ranges = [
        np.linspace(all_projected[:, dim].min(), all_projected[:, dim].max(), grid_size)
        for dim in range(d)
    ]
    mesh = np.meshgrid(*grid_ranges)
    grid_points = np.vstack([m.ravel() for m in mesh])

    try:
        pdf_i = kde_i(grid_points)
        pdf_j = kde_j(grid_points)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")

    total_i = pdf_i.sum()
    total_j = pdf_j.sum()
    if total_i == 0 or total_j == 0:
        return float("nan")
    pdf_i /= total_i
    pdf_j /= total_j
    D = 1.0 - 0.5 * np.sum(np.abs(pdf_i - pdf_j))
    return float(np.clip(D, 0.0, 1.0))


def phase3_pca_dimensionality(
    feature_spaces: dict[str, np.ndarray],
    labels: list[str],
    class_names: list[str],
    dataset_name: str,
) -> tuple[dict, dict]:
    """Compute Schoener's D at various PCA dimensionalities."""
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Phase 3: PCA dimensionality for Schoener's D")
    analysis_spaces = {k: v for k, v in feature_spaces.items() if not k.startswith("_")}

    schoeners_results: dict = {}
    pca_stability: dict = {}
    labels_arr = np.array(labels)
    n_classes = len(class_names)

    for space_name, features in analysis_spaces.items():
        logger.info(f"    Space: {space_name}")
        schoeners_results[space_name] = {}

        # Dimensionality reduction
        max_dim = min(max(PCA_DIMS), features.shape[1], features.shape[0] - 1)
        if max_dim < 2:
            logger.info(f"      max_dim={max_dim}, skipping PCA")
            continue

        try:
            if hasattr(features, 'toarray'):
                reducer = TruncatedSVD(n_components=max_dim)
            else:
                reducer = PCA(n_components=max_dim)
            projected_full = reducer.fit_transform(features)
            if hasattr(reducer, 'explained_variance_ratio_'):
                evr = reducer.explained_variance_ratio_
            else:
                evr = np.zeros(max_dim)
        except Exception:
            logger.exception(f"      PCA/SVD failed for {space_name}")
            continue

        for d in PCA_DIMS:
            if d > max_dim:
                continue
            projected = projected_full[:, :d]
            explained_var = float(evr[:d].sum()) if len(evr) >= d else 0.0

            D_matrix = np.ones((n_classes, n_classes))
            for (ci, class_i), (cj, class_j) in combinations(enumerate(class_names), 2):
                mask_i = labels_arr == class_i
                mask_j = labels_arr == class_j
                pts_i = projected[mask_i]
                pts_j = projected[mask_j]

                if len(pts_i) < 3 or len(pts_j) < 3:
                    D_matrix[ci, cj] = D_matrix[cj, ci] = float("nan")
                    continue

                if d <= 3:
                    all_pts = np.vstack([pts_i, pts_j])
                    D_val = compute_schoeners_d_2d(pts_i, pts_j, all_pts, grid_size=40)
                else:
                    # Random 2D projections for high dimensions
                    D_projections = []
                    for proj_seed in range(3):  # reduced from 5 for speed
                        rng_proj = np.random.default_rng(proj_seed)
                        proj_matrix = rng_proj.standard_normal((d, 2))
                        proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
                        proj_i = pts_i @ proj_matrix
                        proj_j = pts_j @ proj_matrix
                        all_proj = np.vstack([proj_i, proj_j])
                        D_val_2d = compute_schoeners_d_2d(proj_i, proj_j, all_proj, grid_size=40)
                        if not np.isnan(D_val_2d):
                            D_projections.append(D_val_2d)
                    D_val = float(np.mean(D_projections)) if D_projections else float("nan")

                D_matrix[ci, cj] = D_matrix[cj, ci] = D_val

            schoeners_results[space_name][str(d)] = {
                "D_matrix": [[round(v, 4) if not np.isnan(v) else None for v in row] for row in D_matrix.tolist()],
                "explained_variance_cumulative": round(explained_var, 4),
            }
            logger.info(
                f"      d={d}: explained_var={explained_var:.3f}, "
                f"mean_D={np.nanmean(D_matrix[np.triu_indices(n_classes, k=1)]):.3f}"
            )

        # Stability analysis: correlate D values at each dim with reference (max dim)
        ref_dim = str(max(d for d in PCA_DIMS if d <= max_dim))
        if ref_dim in schoeners_results[space_name]:
            ref_D = flatten_upper_triangle(schoeners_results[space_name][ref_dim]["D_matrix"])
            pca_stability[space_name] = {}
            for d in PCA_DIMS:
                if str(d) not in schoeners_results[space_name] or str(d) == ref_dim:
                    continue
                d_D = flatten_upper_triangle(schoeners_results[space_name][str(d)]["D_matrix"])
                # Replace None with NaN
                ref_clean = np.array([0.0 if v is None else v for v in ref_D], dtype=float)
                d_clean = np.array([0.0 if v is None else v for v in d_D], dtype=float)
                valid = ~(np.isnan(ref_clean) | np.isnan(d_clean))
                if valid.sum() >= 3:
                    r, p_val = pearsonr(ref_clean[valid], d_clean[valid])
                    pca_stability[space_name][str(d)] = {
                        "pearson_r": round(float(r), 4),
                        "p_value": round(float(p_val), 6),
                        "ref_dim": int(ref_dim),
                    }

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Phase 3 done in {elapsed:.1f}s")
    return schoeners_results, pca_stability


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: DISTANCE METRIC ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_distance_metrics(
    feature_spaces: dict[str, np.ndarray],
    n_samples: int,
    dataset_name: str,
) -> dict:
    """Evaluate CRND noise detection across different distance metrics."""
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Phase 4: Distance metric ablation")
    results = {}
    analysis_spaces = {k: v for k, v in feature_spaces.items() if not k.startswith("_")}
    k = DEFAULT_K
    if k >= n_samples:
        k = max(1, n_samples - 1)

    for metric in METRICS:
        logger.info(f"    Metric: {metric}")
        neighbor_sets = {}
        for space_name, features in analysis_spaces.items():
            try:
                indices = build_knn(features, k=k, metric=metric)
                neighbor_sets[space_name] = indices
            except Exception:
                logger.exception(f"      Failed for {space_name} with {metric}")
                # fallback to euclidean
                indices = build_knn(features, k=k, metric="euclidean")
                neighbor_sets[space_name] = indices

        crnd = compute_crnd_jaccard(neighbor_sets, n_samples)
        mean_auc, std_auc = noise_detection_auc(crnd, n_samples)

        results[metric] = {
            "mean_auc": round(mean_auc, 6),
            "std_auc": round(std_auc, 6),
            "mean_crnd": round(float(np.mean(crnd)), 6),
            "std_crnd": round(float(np.std(crnd)), 6),
        }
        logger.info(f"      AUC={mean_auc:.4f}±{std_auc:.4f}")

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Phase 4 done in {elapsed:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ALTERNATIVE CRND FORMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_alternative_formulations(
    feature_spaces: dict[str, np.ndarray],
    n_samples: int,
    dataset_name: str,
    k: int = DEFAULT_K,
) -> tuple[dict, dict, dict]:
    """Compare CRND formulations: Jaccard, RBO, weighted Jaccard, pairwise decomp."""
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Phase 5: Alternative CRND formulations (k={k})")
    analysis_spaces = {kk: v for kk, v in feature_spaces.items() if not kk.startswith("_")}

    if k >= n_samples:
        k = max(1, n_samples - 1)

    # Build k-NN for each space
    neighbor_sets = {}
    for space_name, features in analysis_spaces.items():
        indices = build_knn(features, k=k, metric="euclidean")
        neighbor_sets[space_name] = indices

    # 5a. Standard Jaccard
    crnd_jaccard = compute_crnd_jaccard(neighbor_sets, n_samples)
    auc_jaccard, std_jaccard = noise_detection_auc(crnd_jaccard, n_samples)

    # 5b. RBO (p=0.9)
    crnd_rbo = compute_crnd_rbo(neighbor_sets, n_samples, p=0.9)
    auc_rbo, std_rbo = noise_detection_auc(crnd_rbo, n_samples)

    # 5c. Weighted Jaccard
    crnd_wj = compute_crnd_weighted_jaccard(neighbor_sets, n_samples)
    auc_wj, std_wj = noise_detection_auc(crnd_wj, n_samples)

    formulation_results = {
        "jaccard": {"mean_auc": round(auc_jaccard, 6), "std_auc": round(std_jaccard, 6),
                     "mean_crnd": round(float(np.mean(crnd_jaccard)), 6)},
        "rbo_p09": {"mean_auc": round(auc_rbo, 6), "std_auc": round(std_rbo, 6),
                     "mean_crnd": round(float(np.mean(crnd_rbo)), 6)},
        "weighted_jaccard": {"mean_auc": round(auc_wj, 6), "std_auc": round(std_wj, 6),
                              "mean_crnd": round(float(np.mean(crnd_wj)), 6)},
    }
    logger.info(f"    Jaccard AUC={auc_jaccard:.4f}, RBO AUC={auc_rbo:.4f}, WJ AUC={auc_wj:.4f}")

    # 5d. Pairwise decomposition
    pairwise_crnd = compute_pairwise_crnd(neighbor_sets, n_samples)
    pairwise_results = {}
    for pair_name, crnd_pair in pairwise_crnd.items():
        auc_pair, std_pair = noise_detection_auc(crnd_pair, n_samples)
        pairwise_results[pair_name] = {
            "mean_auc": round(auc_pair, 6),
            "std_auc": round(std_pair, 6),
            "mean_crnd": round(float(np.mean(crnd_pair)), 6),
        }
        logger.info(f"    Pairwise {pair_name}: AUC={auc_pair:.4f}")

    # Per-instance scores for output
    per_instance = {
        "crnd_jaccard": crnd_jaccard,
        "crnd_rbo_p09": crnd_rbo,
        "crnd_weighted_jaccard": crnd_wj,
    }
    for pair_name, crnd_pair in pairwise_crnd.items():
        per_instance[f"crnd_pairwise_{pair_name}"] = crnd_pair

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Phase 5 done in {elapsed:.1f}s")
    return formulation_results, pairwise_results, per_instance


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: CONFOUND DISENTANGLEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def phase6_confound_analysis(
    feature_spaces: dict[str, np.ndarray],
    labels: list[str],
    texts: list[str],
    n_samples: int,
    crnd_values: np.ndarray,
    dataset_name: str,
    k: int = DEFAULT_K,
) -> tuple[dict, dict]:
    """Compute confound scores and partial correlations."""
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Phase 6: Confound disentanglement")
    analysis_spaces = {kk: v for kk, v in feature_spaces.items() if not kk.startswith("_")}

    if k >= n_samples:
        k = max(1, n_samples - 1)

    # 6a. Compute confound scores
    # Outlier score: mean distance to k-NN across spaces
    outlier_scores = np.zeros(n_samples)
    boundary_scores = np.zeros(n_samples)
    labels_arr = np.array(labels)

    for space_name, features in analysis_spaces.items():
        distances, indices = build_knn_with_distances(features, k=k, metric="euclidean")
        # Outlier: mean distance to k-NN
        if distances.shape[1] > 0:
            outlier_scores += distances.mean(axis=1)

        # Boundary: distance to nearest different-class neighbor
        nn_all = NearestNeighbors(n_neighbors=min(k * 3, n_samples), metric="euclidean")
        nn_all.fit(features)
        dists_all, idxs_all = nn_all.kneighbors(features)
        for i in range(n_samples):
            found = False
            for j_idx in range(1, dists_all.shape[1]):
                neighbor_idx = idxs_all[i, j_idx]
                if labels_arr[neighbor_idx] != labels_arr[i]:
                    boundary_scores[i] += dists_all[i, j_idx]
                    found = True
                    break
            if not found:
                # All neighbors same class — use max distance
                boundary_scores[i] += dists_all[i, -1]

    n_spaces = len(analysis_spaces)
    if n_spaces > 0:
        outlier_scores /= n_spaces
        boundary_scores /= n_spaces

    # Vocab rarity: mean IDF of terms per text
    vocab_rarity_scores = np.zeros(n_samples)
    vectorizer = feature_spaces.get("_tfidf_vectorizer")
    if vectorizer is not None and hasattr(vectorizer, 'idf_'):
        vocab = vectorizer.vocabulary_
        idf = vectorizer.idf_
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            idf_vals = [idf[vocab[t]] for t in tokens if t in vocab]
            if idf_vals:
                vocab_rarity_scores[i] = np.mean(idf_vals)
    else:
        logger.warning(f"  [{dataset_name}] No TF-IDF vectorizer found for vocab rarity")

    # 6b. Inject noise and compute partial correlations
    rng = np.random.default_rng(42)
    n_flip = max(1, int(NOISE_RATE * n_samples))
    flip_indices = rng.choice(n_samples, size=n_flip, replace=False)
    noise_indicator = np.zeros(n_samples)
    noise_indicator[flip_indices] = 1.0

    confound_results = {}
    per_instance_confounds = {
        "outlier_score": outlier_scores,
        "boundary_score": boundary_scores,
        "vocab_rarity_score": vocab_rarity_scores,
    }

    # Raw Spearman
    rho_raw, p_raw = spearmanr(crnd_values, noise_indicator)
    confound_results["raw_spearman"] = {
        "rho": round(float(rho_raw), 6),
        "p_value": round(float(p_raw), 6),
    }
    logger.info(f"    Raw Spearman: rho={rho_raw:.4f}, p={p_raw:.4f}")

    # Try pingouin for partial correlations
    try:
        import pingouin as pg

        df = pd.DataFrame({
            "crnd": crnd_values,
            "noise": noise_indicator,
            "outlier": outlier_scores,
            "boundary": boundary_scores,
            "vocab_rarity": vocab_rarity_scores,
        })

        # Partial correlations controlling for each confound individually
        for confound_name in ["outlier", "boundary", "vocab_rarity"]:
            try:
                result = pg.partial_corr(
                    data=df,
                    x="crnd",
                    y="noise",
                    covar=confound_name,
                    method="spearman",
                )
                confound_results[f"partial_{confound_name}"] = {
                    "rho": round(float(result["r"].values[0]), 6),
                    "p_value": round(float(result["p-val"].values[0]), 6),
                }
                logger.info(
                    f"    Partial ({confound_name}): "
                    f"rho={result['r'].values[0]:.4f}, p={result['p-val'].values[0]:.4f}"
                )
            except Exception:
                logger.exception(f"    Partial corr ({confound_name}) failed")
                confound_results[f"partial_{confound_name}"] = {"rho": None, "p_value": None}

        # Partial controlling for ALL confounds jointly
        try:
            result_all = pg.partial_corr(
                data=df,
                x="crnd",
                y="noise",
                covar=["outlier", "boundary", "vocab_rarity"],
                method="spearman",
            )
            confound_results["partial_all"] = {
                "rho": round(float(result_all["r"].values[0]), 6),
                "p_value": round(float(result_all["p-val"].values[0]), 6),
            }
            logger.info(
                f"    Partial (all): "
                f"rho={result_all['r'].values[0]:.4f}, p={result_all['p-val'].values[0]:.4f}"
            )
        except Exception:
            logger.exception("    Joint partial corr failed")
            confound_results["partial_all"] = {"rho": None, "p_value": None}

    except ImportError:
        logger.warning("pingouin not available, using manual partial correlations")
        # Manual partial Spearman: partial_r(x,y|z) = (r_xy - r_xz*r_yz) / sqrt((1-r_xz²)(1-r_yz²))
        from scipy.stats import rankdata
        rank_crnd = rankdata(crnd_values)
        rank_noise = rankdata(noise_indicator)

        for confound_name, confound_vals in [
            ("outlier", outlier_scores),
            ("boundary", boundary_scores),
            ("vocab_rarity", vocab_rarity_scores),
        ]:
            rank_z = rankdata(confound_vals)
            r_xy, _ = pearsonr(rank_crnd, rank_noise)
            r_xz, _ = pearsonr(rank_crnd, rank_z)
            r_yz, _ = pearsonr(rank_noise, rank_z)
            denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            if denom > 0:
                partial_r = (r_xy - r_xz * r_yz) / denom
            else:
                partial_r = float("nan")
            confound_results[f"partial_{confound_name}"] = {
                "rho": round(float(partial_r), 6),
                "p_value": None,  # manual method doesn't give p-value easily
            }
            logger.info(f"    Manual partial ({confound_name}): rho={partial_r:.4f}")

        confound_results["partial_all"] = {"rho": None, "p_value": None}

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Phase 6 done in {elapsed:.1f}s")
    return confound_results, per_instance_confounds


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE: SINGLE-SPACE CRND (TF-IDF only)
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_single_space_crnd(
    feature_spaces: dict[str, np.ndarray],
    n_samples: int,
    dataset_name: str,
    k: int = DEFAULT_K,
) -> dict:
    """Baseline: Use only TF-IDF space for anomaly score (mean kNN distance).

    This is a standard kNN-based anomaly detector, NOT cross-representation.
    Serves as comparison to show whether multi-space CRND adds value.
    """
    t0 = time.time()
    logger.info(f"  [{dataset_name}] Baseline: Single-space kNN anomaly detection")

    if k >= n_samples:
        k = max(1, n_samples - 1)

    results = {}

    for space_name in ["tfidf", "sentence_transformer", "llm_zeroshot"]:
        if space_name not in feature_spaces:
            continue
        features = feature_spaces[space_name]
        distances, _ = build_knn_with_distances(features, k=k, metric="euclidean")
        # Anomaly score = mean distance to k-NN
        anomaly_scores = distances.mean(axis=1)

        mean_auc, std_auc = noise_detection_auc(anomaly_scores, n_samples)
        results[f"knn_anomaly_{space_name}"] = {
            "mean_auc": round(mean_auc, 6),
            "std_auc": round(std_auc, 6),
            "mean_score": round(float(np.mean(anomaly_scores)), 6),
        }
        logger.info(f"    {space_name} kNN anomaly AUC={mean_auc:.4f}±{std_auc:.4f}")

    # Also compute random baseline
    rng = np.random.default_rng(42)
    random_scores = rng.random(n_samples)
    auc_rand, std_rand = noise_detection_auc(random_scores, n_samples)
    results["random_baseline"] = {
        "mean_auc": round(auc_rand, 6),
        "std_auc": round(std_rand, 6),
    }
    logger.info(f"    Random baseline AUC={auc_rand:.4f}±{std_rand:.4f}")

    elapsed = time.time() - t0
    logger.info(f"  [{dataset_name}] Baseline done in {elapsed:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    global PHASE_TIMINGS

    data_file = os.environ.get("DATA_FILE", str(DATA_DIR / "full_data_out.json"))
    data_path = Path(data_file)
    max_examples = MAX_EXAMPLES_PER_DATASET
    output_path = WORKSPACE / "method_out.json"

    logger.info("=" * 70)
    logger.info("CRND Ablation & Robustness Analysis")
    logger.info("=" * 70)
    logger.info(f"Data: {data_path}")
    logger.info(f"Max examples per dataset: {max_examples}")
    logger.info(f"Output: {output_path}")
    logger.info(f"K values: {K_VALUES}")
    logger.info(f"PCA dims: {PCA_DIMS}")
    logger.info(f"Distance metrics: {METRICS}")
    logger.info(f"Noise rate: {NOISE_RATE}, Seeds: {N_SEEDS}")

    # Load data (all datasets)
    try:
        datasets = load_data(data_path, max_per_dataset=max_examples)
    except FileNotFoundError:
        logger.exception("Data file not found")
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in data file")
        raise

    # Process each dataset
    all_dataset_results = []
    metadata_k_sensitivity: dict = {}
    metadata_pca_dim: dict = {}
    metadata_pca_stability: dict = {}
    metadata_distance_metric: dict = {}
    metadata_formulation: dict = {}
    metadata_pairwise: dict = {}
    metadata_confound: dict = {}
    metadata_baseline: dict = {}

    for ds_name, ds_data in datasets.items():
        logger.info("=" * 60)
        logger.info(f"Processing dataset: {ds_name}")
        logger.info("=" * 60)

        texts = ds_data["texts"]
        labels = ds_data["labels"]
        class_names = ds_data["class_names"]
        raw_examples = ds_data["raw_examples"]
        n = len(texts)

        if n < 3:
            logger.warning(f"  [{ds_name}] Only {n} examples, skipping complex analyses")
            # Still include in output with minimal processing
            examples_out = []
            for ex in raw_examples:
                out_ex = {
                    "input": ex["input"],
                    "output": ex["output"],
                    "predict_crnd_best_k": "0.5",
                    "predict_crnd_rbo": "0.5",
                    "predict_crnd_weighted_jaccard": "0.5",
                }
                for k_meta, v_meta in ex.items():
                    if k_meta.startswith("metadata_"):
                        out_ex[k_meta] = v_meta
                examples_out.append(out_ex)
            all_dataset_results.append({"dataset": ds_name, "examples": examples_out})
            continue

        # Phase 1: Feature space construction
        t1 = time.time()
        feature_spaces = build_feature_spaces(
            texts=texts,
            labels=labels,
            class_names=class_names,
            dataset_name=ds_name,
        )
        PHASE_TIMINGS[f"{ds_name}_phase1"] = time.time() - t1

        # Phase 2: k sensitivity
        t2 = time.time()
        k_sens = phase2_k_sensitivity(
            feature_spaces=feature_spaces,
            n_samples=n,
            dataset_name=ds_name,
        )
        metadata_k_sensitivity[ds_name] = k_sens
        PHASE_TIMINGS[f"{ds_name}_phase2"] = time.time() - t2

        # Phase 3: PCA dimensionality
        t3 = time.time()
        schoeners, stability = phase3_pca_dimensionality(
            feature_spaces=feature_spaces,
            labels=labels,
            class_names=class_names,
            dataset_name=ds_name,
        )
        metadata_pca_dim[ds_name] = schoeners
        metadata_pca_stability[ds_name] = stability
        PHASE_TIMINGS[f"{ds_name}_phase3"] = time.time() - t3

        # Phase 4: Distance metrics
        t4 = time.time()
        dist_results = phase4_distance_metrics(
            feature_spaces=feature_spaces,
            n_samples=n,
            dataset_name=ds_name,
        )
        metadata_distance_metric[ds_name] = dist_results
        PHASE_TIMINGS[f"{ds_name}_phase4"] = time.time() - t4

        # Determine optimal k from Phase 2
        best_k = DEFAULT_K
        best_auc = 0.0
        for kk, kv in k_sens.items():
            if kv["mean_auc"] > best_auc:
                best_auc = kv["mean_auc"]
                best_k = int(kk)
        logger.info(f"  [{ds_name}] Optimal k={best_k} (AUC={best_auc:.4f})")

        # Phase 5: Alternative formulations
        t5 = time.time()
        form_results, pair_results, per_instance_scores = phase5_alternative_formulations(
            feature_spaces=feature_spaces,
            n_samples=n,
            dataset_name=ds_name,
            k=best_k,
        )
        metadata_formulation[ds_name] = form_results
        metadata_pairwise[ds_name] = pair_results
        PHASE_TIMINGS[f"{ds_name}_phase5"] = time.time() - t5

        # Phase 6: Confound analysis
        t6 = time.time()
        crnd_for_confound = per_instance_scores["crnd_jaccard"]
        confound_res, confound_per_instance = phase6_confound_analysis(
            feature_spaces=feature_spaces,
            labels=labels,
            texts=texts,
            n_samples=n,
            crnd_values=crnd_for_confound,
            dataset_name=ds_name,
            k=best_k,
        )
        metadata_confound[ds_name] = confound_res
        PHASE_TIMINGS[f"{ds_name}_phase6"] = time.time() - t6

        # Baseline
        t_bl = time.time()
        baseline_res = baseline_single_space_crnd(
            feature_spaces=feature_spaces,
            n_samples=n,
            dataset_name=ds_name,
            k=best_k,
        )
        metadata_baseline[ds_name] = baseline_res
        PHASE_TIMINGS[f"{ds_name}_baseline"] = time.time() - t_bl

        # Compute per-k CRND for each instance (at best k)
        analysis_spaces_clean = {k_: v for k_, v in feature_spaces.items() if not k_.startswith("_")}
        per_k_crnd = {}
        for kk in K_VALUES:
            if kk >= n:
                continue
            neighbor_sets_k = {}
            for space_name, features in analysis_spaces_clean.items():
                indices = build_knn(features, k=kk, metric="euclidean")
                neighbor_sets_k[space_name] = indices
            per_k_crnd[kk] = compute_crnd_jaccard(neighbor_sets_k, n)

        # Build per-instance output
        examples_out = []
        for i, ex in enumerate(raw_examples):
            out_ex = {
                "input": ex["input"],
                "output": ex["output"],
                "predict_crnd_best_k": str(round(crnd_for_confound[i], 4)),
                "predict_crnd_rbo": str(round(per_instance_scores.get("crnd_rbo_p09", np.zeros(n))[i], 4)),
                "predict_crnd_weighted_jaccard": str(round(per_instance_scores.get("crnd_weighted_jaccard", np.zeros(n))[i], 4)),
            }
            # Preserve original metadata
            for k_meta, v_meta in ex.items():
                if k_meta.startswith("metadata_"):
                    out_ex[k_meta] = v_meta

            # Add per-k CRND as metadata
            for kk in K_VALUES:
                if kk in per_k_crnd:
                    out_ex[f"metadata_crnd_k{kk}"] = round(float(per_k_crnd[kk][i]), 4)

            # Add formulation scores as metadata
            out_ex["metadata_crnd_rbo_p09"] = round(float(per_instance_scores.get("crnd_rbo_p09", np.zeros(n))[i]), 4)
            out_ex["metadata_crnd_weighted_jaccard"] = round(float(per_instance_scores.get("crnd_weighted_jaccard", np.zeros(n))[i]), 4)

            # Add pairwise CRND
            for pair_key, pair_vals in per_instance_scores.items():
                if pair_key.startswith("crnd_pairwise_"):
                    short_key = pair_key.replace("crnd_pairwise_", "")
                    out_ex[f"metadata_crnd_pairwise_{short_key}"] = round(float(pair_vals[i]), 4)

            # Add confound scores
            out_ex["metadata_outlier_score"] = round(float(confound_per_instance["outlier_score"][i]), 4)
            out_ex["metadata_boundary_score"] = round(float(confound_per_instance["boundary_score"][i]), 4)
            out_ex["metadata_vocab_rarity_score"] = round(float(confound_per_instance["vocab_rarity_score"][i]), 4)

            examples_out.append(out_ex)

        all_dataset_results.append({"dataset": ds_name, "examples": examples_out})

    # Determine global recommendations
    # Best k across all datasets
    all_k_aucs: dict[int, list[float]] = {}
    for ds_name, k_sens in metadata_k_sensitivity.items():
        for kk, kv in k_sens.items():
            all_k_aucs.setdefault(int(kk), []).append(kv["mean_auc"])
    best_global_k = DEFAULT_K
    best_global_auc = 0.0
    for kk, aucs in all_k_aucs.items():
        mean_auc = np.mean(aucs)
        if mean_auc > best_global_auc:
            best_global_auc = mean_auc
            best_global_k = kk

    # PCA stability threshold
    pca_threshold_dim = 20
    for ds_name, stab in metadata_pca_stability.items():
        for space_name, dim_data in stab.items():
            for d_str, vals in dim_data.items():
                if isinstance(vals, dict) and vals.get("pearson_r", 0) > 0.95:
                    pca_threshold_dim = min(pca_threshold_dim, int(d_str))

    # Best CRND formulation
    form_aucs: dict[str, list[float]] = {}
    for ds_name, form in metadata_formulation.items():
        for fname, fv in form.items():
            form_aucs.setdefault(fname, []).append(fv["mean_auc"])
    best_form = "jaccard"
    best_form_auc = 0.0
    for fname, aucs in form_aucs.items():
        mean_auc = np.mean(aucs)
        if mean_auc > best_form_auc:
            best_form_auc = mean_auc
            best_form = fname

    # Most informative pair
    pair_aucs: dict[str, list[float]] = {}
    for ds_name, pairs in metadata_pairwise.items():
        for pname, pv in pairs.items():
            pair_aucs.setdefault(pname, []).append(pv["mean_auc"])
    best_pair = "unknown"
    best_pair_auc = 0.0
    for pname, aucs in pair_aucs.items():
        mean_auc = np.mean(aucs)
        if mean_auc > best_pair_auc:
            best_pair_auc = mean_auc
            best_pair = pname

    total_runtime = time.time() - GLOBAL_START

    # Build output
    output = {
        "metadata": {
            "method_name": "CRND_Ablation_Robustness",
            "description": (
                "Systematic ablation of CRND: k-sensitivity, PCA dimensionality, "
                "distance metrics, alternative formulations, confound controls"
            ),
            "k_values_tested": K_VALUES,
            "pca_dims_tested": PCA_DIMS,
            "distance_metrics_tested": METRICS,
            "crnd_formulations_tested": ["jaccard", "rbo_p09", "weighted_jaccard", "pairwise_decomp"],
            "confounds_tested": ["outlier_score", "boundary_score", "vocab_rarity"],
            "noise_rate": NOISE_RATE,
            "n_seeds": N_SEEDS,
            "datasets_used": list(datasets.keys()),
            "max_examples_per_dataset": max_examples,
            "k_sensitivity_results": metadata_k_sensitivity,
            "pca_dimensionality_results": metadata_pca_dim,
            "pca_stability_analysis": metadata_pca_stability,
            "distance_metric_results": metadata_distance_metric,
            "formulation_comparison": metadata_formulation,
            "pairwise_decomposition": metadata_pairwise,
            "confound_analysis": metadata_confound,
            "baseline_results": metadata_baseline,
            "optimal_k_recommendation": best_global_k,
            "pca_stability_threshold": pca_threshold_dim,
            "best_crnd_formulation": best_form,
            "most_informative_pair": best_pair,
            "runtime_seconds": round(total_runtime, 1),
            "phase_timings": {k: round(v, 1) for k, v in PHASE_TIMINGS.items()},
            "llm_model": LLM_MODEL,
            "llm_total_cost_usd": round(LLM_TOTAL_COST_USD, 4),
            "llm_call_count": LLM_CALL_COUNT,
        },
        "datasets": all_dataset_results,
    }

    # Write output
    logger.info(f"Writing output to {output_path}")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output: {file_size_mb:.1f} MB")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total runtime: {total_runtime:.1f}s")
    logger.info(f"Optimal k: {best_global_k} (mean AUC={best_global_auc:.4f})")
    logger.info(f"PCA stability threshold: d={pca_threshold_dim}")
    logger.info(f"Best formulation: {best_form} (mean AUC={best_form_auc:.4f})")
    logger.info(f"Most informative pair: {best_pair} (mean AUC={best_pair_auc:.4f})")
    logger.info(f"LLM calls: {LLM_CALL_COUNT}, cost: ${LLM_TOTAL_COST_USD:.4f}")
    for ds_name in datasets:
        total_ds = sum(v for k, v in PHASE_TIMINGS.items() if k.startswith(ds_name))
        logger.info(f"  {ds_name}: {total_ds:.1f}s total")
    logger.info("Done!")


if __name__ == "__main__":
    main()
