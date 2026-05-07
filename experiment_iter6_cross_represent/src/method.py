#!/usr/bin/env python3
"""
Cross-Representation Neighborhood Dissonance (CRND) + Schoener's D
Ecological Niche Overlap across 6 Clinical Datasets.

Computes three feature spaces (TF-IDF, sentence-transformer, LLM zero-shot),
per-instance CRND scores, ecological niche overlap matrices, per-class CRND
analysis with statistical tests, noise detection benchmarks, and cross-dataset
generalization analysis.
"""

import json
import os
import re
import resource
import sys
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from scipy.stats import entropy, gaussian_kde, kruskal, mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DEP1_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
DEP2_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_220558"
    "/3_invention_loop/iter_4/gen_art/data_id3_it4__opus/full_data_out.json"
)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Load env ─────────────────────────────────────────────────────────────────
load_dotenv(Path("/home/adrian/projects/ai-inventor/.env"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ── Global config ────────────────────────────────────────────────────────────
MAX_SAMPLE_N = int(os.environ.get("CRND_MAX_SAMPLE", "300"))
LLM_PER_DATASET = int(os.environ.get("CRND_LLM_PER_DS", "150"))  # max LLM calls per dataset
NOISE_TIME_LIMIT = 600  # 10 min max per dataset for noise detection
K_VALUES = [5, 10, 20]
NOISE_RATES = [0.05, 0.10, 0.20]
N_NOISE_SEEDS = 3
LLM_MAX_CALLS = 5000  # hard cap across all datasets
LLM_MAX_COST_USD = 9.0  # leave $1 buffer
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
LLM_COST_PER_1K_INPUT = 0.00005  # ~$0.05/M input tokens for llama-3.1-8b
LLM_COST_PER_1K_OUTPUT = 0.00005

# Track LLM usage globally
_llm_calls = 0
_llm_cost = 0.0

# ── Datasets to use (drop clinical_patient_triage_nl, N=31) ──────────────────
DATASETS_CONFIG = {
    # dep1 datasets
    "medical_abstracts": {"source": "dep1", "sample_n": MAX_SAMPLE_N},
    "mimic_iv_ed_demo": {"source": "dep1", "sample_n": None},  # use all 207
    "ohsumed_single": {"source": "dep1", "sample_n": MAX_SAMPLE_N},
    "mental_health_conditions": {"source": "dep1", "sample_n": MAX_SAMPLE_N},
    # dep2 datasets (medical_abstracts from dep2 renamed to avoid collision)
    "medical_abstracts_v2": {"source": "dep2", "dataset_key": "medical_abstracts", "sample_n": MAX_SAMPLE_N},
    "medical_transcriptions": {"source": "dep2", "sample_n": MAX_SAMPLE_N},
}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_datasets() -> dict:
    """Load and sample datasets from dependency files."""
    logger.info("Loading dependency data files...")
    dep1_data = json.loads(DEP1_PATH.read_text())
    dep2_data = json.loads(DEP2_PATH.read_text())

    dep_lookup = {}
    for ds in dep1_data["datasets"]:
        dep_lookup[("dep1", ds["dataset"])] = ds["examples"]
    for ds in dep2_data["datasets"]:
        dep_lookup[("dep2", ds["dataset"])] = ds["examples"]

    datasets = {}
    for ds_name, cfg in DATASETS_CONFIG.items():
        source = cfg["source"]
        dataset_key = cfg.get("dataset_key", ds_name)
        key = (source, dataset_key)

        if key not in dep_lookup:
            logger.warning(f"Dataset {ds_name} ({key}) not found in dependencies, skipping")
            continue

        examples = dep_lookup[key]
        texts = [ex["input"] for ex in examples]
        labels = [ex["output"] for ex in examples]
        raw_examples = examples

        # Sample if needed
        sample_n = cfg["sample_n"]
        if sample_n is not None and len(texts) > sample_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(texts), sample_n, replace=False)
            idx.sort()
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]
            raw_examples = [raw_examples[i] for i in idx]

        unique_labels = sorted(set(labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        labels_int = np.array([label_to_int[l] for l in labels])

        n = len(texts)
        n_classes = len(unique_labels)
        dist = Counter(labels)

        if n < 50 or n_classes < 2:
            logger.warning(f"Skipping {ds_name}: N={n}, classes={n_classes} (too small)")
            continue

        datasets[ds_name] = {
            "texts": texts,
            "labels": labels,
            "labels_int": labels_int,
            "label_to_int": label_to_int,
            "class_names": unique_labels,
            "n_classes": n_classes,
            "n": n,
            "distribution": dict(dist),
            "raw_examples": raw_examples,
        }
        logger.info(f"  {ds_name}: N={n}, classes={n_classes}, dist={dict(dist)}")

    logger.info(f"Loaded {len(datasets)} datasets total")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE SPACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_tfidf_features(texts: list[str]) -> np.ndarray:
    """Build TF-IDF feature matrix."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    return X.toarray()  # dense for uniform k-NN interface


def build_embedding_features(texts: list[str], model) -> np.ndarray:
    """Build sentence-transformer embedding features."""
    X = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return X


def build_llm_features(
    texts: list[str],
    class_names: list[str],
    dataset_name: str,
    labels_int: np.ndarray | None = None,
    n_classes: int | None = None,
) -> np.ndarray:
    """Build LLM zero-shot classification features via OpenRouter.

    Falls back to cross-val predicted probabilities if LLM unavailable.
    """
    global _llm_calls, _llm_cost

    n = len(texts)
    k = len(class_names)
    X_llm = np.full((n, k), 1.0 / k)  # default: uniform

    if not OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY — using fallback proxy LLM features")
        if labels_int is not None and n_classes is not None:
            return _build_proxy_llm_features_with_labels(texts, labels_int, n_classes)
        return np.full((n, k), 1.0 / k)

    # Check budget
    remaining_calls = LLM_MAX_CALLS - _llm_calls
    remaining_budget = LLM_MAX_COST_USD - _llm_cost
    if remaining_calls <= 0 or remaining_budget <= 0.5:
        logger.warning(f"LLM budget exhausted (calls={_llm_calls}, cost=${_llm_cost:.2f}). Using proxy.")
        if labels_int is not None and n_classes is not None:
            return _build_proxy_llm_features_with_labels(texts, labels_int, n_classes)
        return np.full((n, k), 1.0 / k)

    # Limit calls for this dataset
    max_for_dataset = min(n, remaining_calls, LLM_PER_DATASET)

    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # For datasets with many classes (>10), use abbreviated class names to avoid
    # exceeding token limits and reduce LLM parsing failures
    if k > 10:
        # Shorten class names: replace underscores with spaces, truncate
        short_names = [cn.replace("_", " ")[:40] for cn in class_names]
        classes_json = json.dumps(short_names)
        name_to_idx = {short: i for i, short in enumerate(short_names)}
        # Also keep original mapping for fallback matching
        orig_to_idx = {cn: i for i, cn in enumerate(class_names)}
        max_tokens_resp = min(50 + k * 25, 1500)  # Scale with class count
    else:
        classes_json = json.dumps(class_names)
        name_to_idx = {cn: i for i, cn in enumerate(class_names)}
        orig_to_idx = name_to_idx
        max_tokens_resp = 300

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def _call_llm(text_snippet: str) -> dict:
        global _llm_calls, _llm_cost
        prompt = (
            f"You are a medical text classifier. Given a clinical text, output a JSON object "
            f"mapping each category to a probability (0-1, sum=1).\n\n"
            f"Categories: {classes_json}\n\n"
            f"IMPORTANT: Output ONLY the raw JSON object, nothing else. No explanation.\n\n"
            f"Text: {text_snippet}"
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens_resp,
            response_format={"type": "json_object"},
            timeout=30,
        )
        _llm_calls += 1
        # Estimate cost
        usage = resp.usage
        if usage:
            _llm_cost += (usage.prompt_tokens / 1000) * LLM_COST_PER_1K_INPUT
            _llm_cost += (usage.completion_tokens / 1000) * LLM_COST_PER_1K_OUTPUT
        content = resp.choices[0].message.content.strip()
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: extract JSON with regex (handle nested braces)
            match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse LLM response: {content[:200]}")

    success_count = 0
    fail_count = 0
    indices_to_process = list(range(min(n, max_for_dataset)))
    llm_start_time = time.time()
    LLM_TIME_LIMIT_PER_DS = 300  # 5 minutes max per dataset

    for idx in tqdm(indices_to_process, desc=f"LLM [{dataset_name}]", leave=False):
        # Budget check every 20 calls
        if idx % 20 == 0 and idx > 0:
            elapsed = time.time() - llm_start_time
            if _llm_cost >= LLM_MAX_COST_USD or _llm_calls >= LLM_MAX_CALLS:
                logger.warning(f"LLM budget reached mid-dataset at idx={idx}")
                break
            if elapsed > LLM_TIME_LIMIT_PER_DS:
                logger.warning(f"LLM time limit ({LLM_TIME_LIMIT_PER_DS}s) reached at idx={idx}")
                break

        text_snippet = texts[idx][:1500]
        try:
            result = _call_llm(text_snippet)
            probs = np.zeros(k)
            # Try matching both short and original class names
            for key, val in result.items():
                idx_match = name_to_idx.get(key)
                if idx_match is None:
                    idx_match = orig_to_idx.get(key)
                if idx_match is None:
                    # Fuzzy: case-insensitive + strip
                    key_lower = key.strip().lower()
                    for cn_i, cn in enumerate(class_names):
                        if cn.lower() == key_lower or cn.replace("_", " ").lower() == key_lower:
                            idx_match = cn_i
                            break
                if idx_match is not None:
                    try:
                        probs[idx_match] = float(val) if val else 0.0
                    except (TypeError, ValueError):
                        pass
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.full(k, 1.0 / k)
            X_llm[idx] = probs
            success_count += 1
        except Exception:
            fail_count += 1
            # Keep uniform default

    logger.info(
        f"LLM features [{dataset_name}]: {success_count} success, {fail_count} fail, "
        f"total_calls={_llm_calls}, cost=${_llm_cost:.4f}"
    )

    # If too many failures, fall back to proxy
    if fail_count > 0.3 * len(indices_to_process) and success_count < 10:
        logger.warning(f"High LLM failure rate ({fail_count}/{len(indices_to_process)}). Using proxy.")
        if labels_int is not None and n_classes is not None:
            return _build_proxy_llm_features_with_labels(texts, labels_int, n_classes)
        return np.full((n, k), 1.0 / k)

    return X_llm


def _build_proxy_llm_features_with_labels(
    texts: list[str],
    labels_int: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Proxy LLM features using cross-validated predicted probabilities."""
    vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True, stop_words="english")
    X_tfidf = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    n_per_class = np.bincount(labels_int, minlength=n_classes)
    min_class_size = n_per_class[n_per_class > 0].min()
    cv_folds = min(5, min_class_size)
    if cv_folds < 2:
        cv_folds = 2
    try:
        pred_probs = cross_val_predict(clf, X_tfidf, labels_int, cv=cv_folds, method="predict_proba")
    except Exception:
        logger.warning("Cross-val proxy failed, returning uniform")
        pred_probs = np.full((len(texts), n_classes), 1.0 / n_classes)
    return pred_probs


def build_all_feature_spaces(datasets: dict) -> dict:
    """Build all 3 feature spaces for all datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Building feature spaces")
    logger.info("=" * 60)

    # Load sentence-transformer model ONCE
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    logger.info("Sentence-transformer loaded")

    feature_spaces = {}
    for ds_name, ds in datasets.items():
        logger.info(f"Building features for {ds_name} (N={ds['n']})...")
        t0 = time.time()

        # TF-IDF
        X_tfidf = build_tfidf_features(ds["texts"])
        logger.info(f"  TF-IDF: shape {X_tfidf.shape}")

        # Embeddings
        X_embed = build_embedding_features(ds["texts"], st_model)
        logger.info(f"  Embed: shape {X_embed.shape}")

        # LLM zero-shot
        X_llm = build_llm_features(
            texts=ds["texts"],
            class_names=ds["class_names"],
            dataset_name=ds_name,
            labels_int=ds["labels_int"],
            n_classes=ds["n_classes"],
        )
        logger.info(f"  LLM: shape {X_llm.shape}")

        # If LLM features are all uniform (proxy failed), use label-based proxy
        # Check variance: if all rows are truly identical uniform vectors
        row_variance = np.var(X_llm, axis=0).sum()
        logger.info(f"  LLM feature variance: {row_variance:.6f}")
        if row_variance < 1e-8:
            logger.info("  LLM features truly uniform — building label-based proxy")
            X_llm = _build_proxy_llm_features_with_labels(
                ds["texts"], ds["labels_int"], ds["n_classes"]
            )
            logger.info(f"  LLM proxy: shape {X_llm.shape}")

        feature_spaces[ds_name] = {
            "tfidf": X_tfidf,
            "embed": X_embed,
            "llm": X_llm,
        }
        logger.info(f"  Features built in {time.time()-t0:.1f}s")

    # Free model memory
    del st_model
    return feature_spaces


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CRND COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_knn(X: np.ndarray, k_max: int = 20, metric: str = "cosine") -> np.ndarray:
    """Compute k-NN indices (excluding self). Returns shape (N, k_max)."""
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric=metric, algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices[:, 1:]  # drop self


def compute_crnd_scores(
    knn_sets: dict[str, np.ndarray],
    n: int,
    k_values: list[int],
) -> dict:
    """Compute CRND scores for all k values."""
    space_pairs = [("tfidf", "embed"), ("tfidf", "llm"), ("embed", "llm")]
    results = {}

    for k in k_values:
        crnd = np.zeros(n)
        crnd_components = {f"{s1}_{s2}": np.zeros(n) for s1, s2 in space_pairs}

        for i in range(n):
            pairwise_jaccards = []
            for s1, s2 in space_pairs:
                set1 = set(knn_sets[s1][i, :k].tolist())
                set2 = set(knn_sets[s2][i, :k].tolist())
                jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
                pairwise_jaccards.append(jaccard)
                crnd_components[f"{s1}_{s2}"][i] = 1.0 - jaccard

            crnd[i] = 1.0 - np.mean(pairwise_jaccards)

        results[k] = {
            "crnd": crnd,
            **{key: val for key, val in crnd_components.items()},
        }

    return results


def run_phase2(datasets: dict, feature_spaces: dict) -> tuple[dict, dict]:
    """Phase 2: Compute k-NN and CRND for all datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Computing CRND scores")
    logger.info("=" * 60)

    all_knn_sets = {}
    all_crnd_scores = {}

    for ds_name, ds in datasets.items():
        logger.info(f"Computing k-NN for {ds_name}...")
        t0 = time.time()

        knn_sets = {}
        for space_name, X in feature_spaces[ds_name].items():
            metric = "cosine" if space_name in ("tfidf", "embed") else "euclidean"
            knn_sets[space_name] = compute_knn(X, k_max=20, metric=metric)

        all_knn_sets[ds_name] = knn_sets

        crnd_scores = compute_crnd_scores(knn_sets, ds["n"], K_VALUES)
        all_crnd_scores[ds_name] = crnd_scores

        # Log summary
        for k in K_VALUES:
            vals = crnd_scores[k]["crnd"]
            logger.info(
                f"  k={k}: mean_CRND={vals.mean():.4f}, "
                f"median={np.median(vals):.4f}, std={vals.std():.4f}"
            )

        logger.info(f"  Done in {time.time()-t0:.1f}s")

    return all_knn_sets, all_crnd_scores


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SCHOENER'S D ECOLOGICAL NICHE OVERLAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_schoeners_d_2d(
    X_proj: np.ndarray,
    labels_int: np.ndarray,
    n_classes: int,
    grid_size: int = 80,
) -> np.ndarray:
    """Compute Schoener's D matrix using 2D KDE on a regular grid."""
    D_matrix = np.full((n_classes, n_classes), np.nan)
    np.fill_diagonal(D_matrix, 1.0)

    for ci, cj in combinations(range(n_classes), 2):
        X_ci = X_proj[labels_int == ci]
        X_cj = X_proj[labels_int == cj]

        if len(X_ci) < 5 or len(X_cj) < 5:
            continue

        try:
            kde_ci = gaussian_kde(X_ci.T)
            kde_cj = gaussian_kde(X_cj.T)
        except np.linalg.LinAlgError:
            # Singular covariance — add jitter
            try:
                X_ci_j = X_ci + np.random.normal(0, 1e-6, X_ci.shape)
                X_cj_j = X_cj + np.random.normal(0, 1e-6, X_cj.shape)
                kde_ci = gaussian_kde(X_ci_j.T)
                kde_cj = gaussian_kde(X_cj_j.T)
            except Exception:
                continue

        x_min = X_proj[:, 0].min() - 1
        x_max = X_proj[:, 0].max() + 1
        y_min = X_proj[:, 1].min() - 1
        y_max = X_proj[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size),
        )
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

        try:
            f_ci = kde_ci(grid_points)
            f_cj = kde_cj(grid_points)
        except Exception:
            continue

        # Normalize to sum to 1
        f_ci_sum = f_ci.sum()
        f_cj_sum = f_cj.sum()
        if f_ci_sum > 0 and f_cj_sum > 0:
            f_ci = f_ci / f_ci_sum
            f_cj = f_cj / f_cj_sum
            D = 1.0 - 0.5 * np.sum(np.abs(f_ci - f_cj))
            D = float(np.clip(D, 0.0, 1.0))
        else:
            D = np.nan

        D_matrix[ci, cj] = D
        D_matrix[cj, ci] = D

    return D_matrix


def compute_schoeners_d_5d(
    X_proj: np.ndarray,
    labels_int: np.ndarray,
    n_classes: int,
    n_eval: int = 3000,
) -> np.ndarray:
    """Compute Schoener's D matrix using 5D KDE with Monte Carlo evaluation."""
    D_matrix = np.full((n_classes, n_classes), np.nan)
    np.fill_diagonal(D_matrix, 1.0)

    for ci, cj in combinations(range(n_classes), 2):
        X_ci = X_proj[labels_int == ci]
        X_cj = X_proj[labels_int == cj]

        if len(X_ci) < 5 or len(X_cj) < 5:
            continue

        try:
            kde_ci = gaussian_kde(X_ci.T)
            kde_cj = gaussian_kde(X_cj.T)
        except np.linalg.LinAlgError:
            try:
                X_ci_j = X_ci + np.random.normal(0, 1e-6, X_ci.shape)
                X_cj_j = X_cj + np.random.normal(0, 1e-6, X_cj.shape)
                kde_ci = gaussian_kde(X_ci_j.T)
                kde_cj = gaussian_kde(X_cj_j.T)
            except Exception:
                continue

        eval_points = np.vstack([X_ci, X_cj])
        if len(eval_points) > n_eval:
            idx = np.random.choice(len(eval_points), n_eval, replace=False)
            eval_points = eval_points[idx]

        try:
            f_ci = kde_ci(eval_points.T)
            f_cj = kde_cj(eval_points.T)
        except Exception:
            continue

        f_ci_sum = f_ci.sum()
        f_cj_sum = f_cj.sum()
        if f_ci_sum > 0 and f_cj_sum > 0:
            f_ci = f_ci / f_ci_sum
            f_cj = f_cj / f_cj_sum
            D = 1.0 - 0.5 * np.sum(np.abs(f_ci - f_cj))
            D = float(np.clip(D, 0.0, 1.0))
        else:
            D = np.nan

        D_matrix[ci, cj] = D
        D_matrix[cj, ci] = D

    return D_matrix


def run_phase3(
    datasets: dict,
    feature_spaces: dict,
) -> dict:
    """Phase 3: Schoener's D ecological niche overlap."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Computing Schoener's D niche overlap")
    logger.info("=" * 60)

    niche_overlap = {}

    for ds_name, ds in datasets.items():
        logger.info(f"Schoener's D for {ds_name} (classes={ds['n_classes']})...")
        t0 = time.time()
        niche_overlap[ds_name] = {}

        for space_name in ["tfidf", "embed", "llm"]:
            X = feature_spaces[ds_name][space_name]
            niche_overlap[ds_name][space_name] = {}

            for n_comp_target, label in [(2, "pca_2d"), (5, "pca_5d")]:
                n_comp = min(n_comp_target, X.shape[1], X.shape[0])
                if n_comp < 2:
                    n_comp = 2

                pca = PCA(n_components=n_comp)
                X_proj = pca.fit_transform(X)
                explained_var = float(pca.explained_variance_ratio_.sum())

                if n_comp_target == 2 or n_comp <= 2:
                    D_matrix = compute_schoeners_d_2d(
                        X_proj[:, :2], ds["labels_int"], ds["n_classes"]
                    )
                else:
                    D_matrix = compute_schoeners_d_5d(
                        X_proj, ds["labels_int"], ds["n_classes"]
                    )

                niche_overlap[ds_name][space_name][label] = {
                    "D_matrix": D_matrix,
                    "explained_variance_ratio": explained_var,
                }
                logger.debug(f"  {space_name}/{label}: explained_var={explained_var:.3f}")

        # D-gap computation
        for n_comp_label in ["pca_2d", "pca_5d"]:
            D_matrices = []
            for space_name in ["tfidf", "embed", "llm"]:
                if n_comp_label in niche_overlap[ds_name][space_name]:
                    D_matrices.append(
                        niche_overlap[ds_name][space_name][n_comp_label]["D_matrix"]
                    )

            if len(D_matrices) >= 2:
                D_stack = np.stack(D_matrices, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    D_gap = np.nanmax(D_stack, axis=0) - np.nanmin(D_stack, axis=0)
            else:
                D_gap = np.zeros((ds["n_classes"], ds["n_classes"]))

            niche_overlap[ds_name][f"D_gap_{n_comp_label}"] = D_gap

        logger.info(f"  Done in {time.time()-t0:.1f}s")

    return niche_overlap


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — PER-CLASS CRND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase4(
    datasets: dict,
    all_crnd_scores: dict,
) -> dict:
    """Phase 4: Per-class CRND analysis with statistical tests."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Per-class CRND analysis")
    logger.info("=" * 60)

    per_class_analysis = {}

    for ds_name, ds in datasets.items():
        logger.info(f"Per-class analysis for {ds_name}...")
        crnd_values = all_crnd_scores[ds_name][10]["crnd"]  # k=10 primary
        labels_int = ds["labels_int"]
        class_names = ds["class_names"]
        n_classes = ds["n_classes"]

        # 4a: Per-class CRND distributions
        per_class_crnd = {}
        groups = []
        for ci, cn in enumerate(class_names):
            mask = labels_int == ci
            vals = crnd_values[mask]
            groups.append(vals)
            per_class_crnd[cn] = {
                "mean": float(np.mean(vals)) if len(vals) > 0 else 0.0,
                "median": float(np.median(vals)) if len(vals) > 0 else 0.0,
                "std": float(np.std(vals)) if len(vals) > 0 else 0.0,
                "n": int(len(vals)),
            }

        # 4b: Kruskal-Wallis test
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) >= 2:
            try:
                H_stat, kw_pvalue = kruskal(*valid_groups)
                H_stat = float(H_stat)
                kw_pvalue = float(kw_pvalue)
            except Exception:
                H_stat, kw_pvalue = 0.0, 1.0
        else:
            H_stat, kw_pvalue = 0.0, 1.0

        # 4c: Post-hoc pairwise Mann-Whitney U with Bonferroni correction
        dunn_results = {}
        n_pairs = n_classes * (n_classes - 1) // 2
        for ci, cj in combinations(range(n_classes), 2):
            g1 = groups[ci]
            g2 = groups[cj]
            if len(g1) >= 2 and len(g2) >= 2:
                try:
                    stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                    pval_corrected = min(pval * n_pairs, 1.0)  # Bonferroni
                except Exception:
                    pval_corrected = 1.0
            else:
                pval_corrected = 1.0
            key = f"{class_names[ci]}_vs_{class_names[cj]}"
            dunn_results[key] = float(pval_corrected)

        # 4d: Cohen's d effect sizes
        cohens_d_matrix = np.zeros((n_classes, n_classes))
        for ci, cj in combinations(range(n_classes), 2):
            x = groups[ci]
            y = groups[cj]
            if len(x) >= 2 and len(y) >= 2:
                var_x = np.var(x, ddof=1)
                var_y = np.var(y, ddof=1)
                pooled_std = np.sqrt((var_x + var_y) / 2)
                if pooled_std > 0:
                    d = float((np.mean(x) - np.mean(y)) / pooled_std)
                else:
                    d = 0.0
            else:
                d = 0.0
            cohens_d_matrix[ci, cj] = d
            cohens_d_matrix[cj, ci] = -d

        per_class_analysis[ds_name] = {
            "per_class_crnd": per_class_crnd,
            "kruskal_wallis": {"H_statistic": H_stat, "p_value": kw_pvalue},
            "pairwise_tests": dunn_results,
            "cohens_d_matrix": cohens_d_matrix.tolist(),
            "class_names": class_names,
        }

        logger.info(
            f"  Kruskal-Wallis: H={H_stat:.2f}, p={kw_pvalue:.4f}"
        )

    return per_class_analysis


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — NOISE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase5(
    datasets: dict,
    feature_spaces: dict,
    all_knn_sets: dict,
    all_crnd_scores: dict,
) -> dict:
    """Phase 5: Well-calibrated noise detection benchmark."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Noise detection benchmarks")
    logger.info("=" * 60)

    noise_detection = {}

    for ds_name, ds in datasets.items():
        logger.info(f"Noise detection for {ds_name} (N={ds['n']})...")
        t0 = time.time()
        noise_detection[ds_name] = {}
        labels_int = ds["labels_int"]
        n = ds["n"]
        n_classes = ds["n_classes"]
        unique_classes = list(range(n_classes))

        noise_t0 = time.time()
        for noise_rate in NOISE_RATES:
            if time.time() - noise_t0 > NOISE_TIME_LIMIT:
                logger.warning(f"  Noise detection time limit reached for {ds_name}")
                break
            noise_detection[ds_name][str(noise_rate)] = {}
            seed_results = {}  # score_name -> list of dicts

            for seed in range(N_NOISE_SEEDS):
                rng = np.random.RandomState(seed)
                n_flip = int(n * noise_rate)
                if n_flip < 1:
                    n_flip = 1
                flip_indices = rng.choice(n, n_flip, replace=False)
                noisy_labels = labels_int.copy()
                for idx in flip_indices:
                    other = [c for c in unique_classes if c != labels_int[idx]]
                    noisy_labels[idx] = rng.choice(other)

                is_flipped = np.zeros(n, dtype=bool)
                is_flipped[flip_indices] = True

                # ── Compute scores ──
                all_scores = {}

                # Score 1: CRND (precomputed, k=10)
                all_scores["crnd"] = all_crnd_scores[ds_name][10]["crnd"]

                # Score 2: kDN per space and averaged
                for space in ["tfidf", "embed", "llm"]:
                    knn_idx = all_knn_sets[ds_name][space][:, :10]
                    neighbor_labels = noisy_labels[knn_idx]
                    kdn = np.mean(neighbor_labels != noisy_labels[:, None], axis=1)
                    all_scores[f"kdn_{space}"] = kdn
                all_scores["kdn_avg"] = np.mean(
                    [all_scores[f"kdn_{s}"] for s in ["tfidf", "embed", "llm"]], axis=0
                )

                # Score 3: Cleanlab-like self-confidence (manual implementation)
                for space in ["tfidf", "embed", "llm"]:
                    X = feature_spaces[ds_name][space]
                    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                    n_per_class = np.bincount(noisy_labels, minlength=n_classes)
                    min_class_size = n_per_class[n_per_class > 0].min()
                    cv_folds = min(3, min_class_size)
                    if cv_folds < 2:
                        cv_folds = 2
                    try:
                        pred_probs = cross_val_predict(
                            clf, X, noisy_labels, cv=cv_folds, method="predict_proba"
                        )
                        # self-confidence: P(given_label | x)
                        sc = np.array([pred_probs[i, noisy_labels[i]] for i in range(n)])
                        all_scores[f"cleanlab_{space}"] = 1.0 - sc  # invert: higher = more suspicious
                    except Exception:
                        all_scores[f"cleanlab_{space}"] = np.full(n, 0.5)

                all_scores["cleanlab_avg"] = np.mean(
                    [all_scores[f"cleanlab_{s}"] for s in ["tfidf", "embed", "llm"]], axis=0
                )

                # Score 4: kNN label entropy
                for space in ["tfidf", "embed", "llm"]:
                    knn_idx = all_knn_sets[ds_name][space][:, :10]
                    neighbor_labels = noisy_labels[knn_idx]
                    label_ent = np.zeros(n)
                    for i in range(n):
                        counts = np.bincount(neighbor_labels[i], minlength=n_classes)
                        probs = counts / counts.sum()
                        label_ent[i] = entropy(probs)
                    all_scores[f"knn_entropy_{space}"] = label_ent

                # Score 5: Random baseline
                all_scores["random"] = rng.rand(n)

                # ── Compute detection metrics ──
                for score_name, scores in all_scores.items():
                    if score_name not in seed_results:
                        seed_results[score_name] = []

                    metrics = {}

                    # ROC-AUC
                    try:
                        auc = float(roc_auc_score(is_flipped.astype(int), scores))
                    except ValueError:
                        auc = 0.5
                    metrics["roc_auc"] = auc

                    # Spearman rho
                    try:
                        rho, rho_p = spearmanr(scores, is_flipped.astype(int))
                        metrics["spearman_rho"] = float(rho) if not np.isnan(rho) else 0.0
                    except Exception:
                        metrics["spearman_rho"] = 0.0

                    # Precision@k
                    top_k_idx = np.argsort(scores)[-n_flip:]
                    prec_at_k = float(np.mean(is_flipped[top_k_idx]))
                    metrics["precision_at_k"] = prec_at_k

                    seed_results[score_name].append(metrics)

            # Aggregate across seeds
            for score_name, results_list in seed_results.items():
                aucs = [r["roc_auc"] for r in results_list]
                rhos = [r["spearman_rho"] for r in results_list]
                precs = [r["precision_at_k"] for r in results_list]

                noise_detection[ds_name][str(noise_rate)][score_name] = {
                    "roc_auc_mean": float(np.mean(aucs)),
                    "roc_auc_std": float(np.std(aucs)),
                    "spearman_rho_mean": float(np.mean(rhos)),
                    "spearman_rho_std": float(np.std(rhos)),
                    "precision_at_k_mean": float(np.mean(precs)),
                    "precision_at_k_std": float(np.std(precs)),
                }

        logger.info(f"  Done in {time.time()-t0:.1f}s")

        # Log summary for k=10%
        nd_10 = noise_detection[ds_name].get("0.1", {})
        if nd_10:
            crnd_auc = nd_10.get("crnd", {}).get("roc_auc_mean", -1)
            kdn_auc = nd_10.get("kdn_avg", {}).get("roc_auc_mean", -1)
            cl_auc = nd_10.get("cleanlab_avg", {}).get("roc_auc_mean", -1)
            rand_auc = nd_10.get("random", {}).get("roc_auc_mean", -1)
            logger.info(
                f"  @10% noise: CRND_AUC={crnd_auc:.3f}, kDN_AUC={kdn_auc:.3f}, "
                f"CL_AUC={cl_auc:.3f}, Random_AUC={rand_auc:.3f}"
            )

    return noise_detection


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — CROSS-DATASET GENERALIZATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase6(
    datasets: dict,
    all_crnd_scores: dict,
    niche_overlap: dict,
) -> dict:
    """Phase 6: Cross-dataset generalization analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 6: Cross-dataset generalization analysis")
    logger.info("=" * 60)

    cross_dataset = {}

    # 6a: D topology comparison — summary stats per feature space
    d_topology = {}
    for ds_name, ds in datasets.items():
        d_topology[ds_name] = {}
        for space in ["tfidf", "embed", "llm"]:
            for pc_label in ["pca_2d", "pca_5d"]:
                D_mat = niche_overlap[ds_name][space][pc_label]["D_matrix"]
                # Upper triangular values (excluding diagonal and NaN)
                mask = np.triu(np.ones_like(D_mat, dtype=bool), k=1)
                vals = D_mat[mask]
                valid = vals[~np.isnan(vals)]
                d_topology[ds_name][f"{space}_{pc_label}"] = {
                    "mean_D": float(np.mean(valid)) if len(valid) > 0 else None,
                    "std_D": float(np.std(valid)) if len(valid) > 0 else None,
                    "frac_high_overlap": float(np.mean(valid > 0.5)) if len(valid) > 0 else None,
                    "n_valid_pairs": int(len(valid)),
                }

    # Compare medical_abstracts (dep1) vs medical_abstracts_v2 (dep2) — same source
    d_corr = {}
    if "medical_abstracts" in niche_overlap and "medical_abstracts_v2" in niche_overlap:
        for space in ["tfidf", "embed", "llm"]:
            for pc_label in ["pca_2d", "pca_5d"]:
                # Both have 5 classes but potentially different orderings
                D1 = niche_overlap["medical_abstracts"][space][pc_label]["D_matrix"]
                D2 = niche_overlap["medical_abstracts_v2"][space][pc_label]["D_matrix"]
                if D1.shape == D2.shape:
                    mask = np.triu(np.ones_like(D1, dtype=bool), k=1)
                    v1 = D1[mask]
                    v2 = D2[mask]
                    valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
                    if valid_mask.sum() >= 3:
                        rho, p = spearmanr(v1[valid_mask], v2[valid_mask])
                        d_corr[f"{space}_{pc_label}"] = {
                            "spearman_rho": float(rho) if not np.isnan(rho) else None,
                            "p_value": float(p) if not np.isnan(p) else None,
                        }

    cross_dataset["d_topology_comparison"] = d_topology
    cross_dataset["d_cross_version_correlation"] = d_corr

    # 6b: D-gap consistency
    d_gap_consistency = {}
    for ds_name in datasets:
        for pc_label in ["pca_2d", "pca_5d"]:
            D_gap = niche_overlap[ds_name][f"D_gap_{pc_label}"]
            mask = np.triu(np.ones_like(D_gap, dtype=bool), k=1)
            vals = D_gap[mask]
            valid = vals[~np.isnan(vals)]
            d_gap_consistency[f"{ds_name}_{pc_label}"] = {
                "mean_D_gap": float(np.mean(valid)) if len(valid) > 0 else None,
                "max_D_gap": float(np.max(valid)) if len(valid) > 0 else None,
                "std_D_gap": float(np.std(valid)) if len(valid) > 0 else None,
            }
    cross_dataset["d_gap_consistency"] = d_gap_consistency

    # 6c: CRND distribution comparison
    crnd_dist_comparison = {}
    for ds_name, ds in datasets.items():
        crnd_vals = all_crnd_scores[ds_name][10]["crnd"]
        crnd_dist_comparison[ds_name] = {
            "mean_crnd": float(np.mean(crnd_vals)),
            "median_crnd": float(np.median(crnd_vals)),
            "std_crnd": float(np.std(crnd_vals)),
            "skewness": float(
                np.mean(((crnd_vals - np.mean(crnd_vals)) / (np.std(crnd_vals) + 1e-10)) ** 3)
            ),
            "n_classes": ds["n_classes"],
            "n_instances": ds["n"],
        }

    # Test: correlation between n_classes and mean CRND
    n_cls = [crnd_dist_comparison[d]["n_classes"] for d in crnd_dist_comparison]
    mean_crnd = [crnd_dist_comparison[d]["mean_crnd"] for d in crnd_dist_comparison]
    if len(n_cls) >= 3:
        rho, p = spearmanr(n_cls, mean_crnd)
        crnd_dist_comparison["n_classes_vs_crnd_correlation"] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p) if not np.isnan(p) else None,
        }

    cross_dataset["crnd_distribution_comparison"] = crnd_dist_comparison

    # 6d: Which feature space has lowest mean D across datasets?
    space_rank = {}
    for ds_name in datasets:
        ranks = {}
        for pc_label in ["pca_2d"]:  # use 2D for consistency
            means = {}
            for space in ["tfidf", "embed", "llm"]:
                D_mat = niche_overlap[ds_name][space][pc_label]["D_matrix"]
                mask_ut = np.triu(np.ones_like(D_mat, dtype=bool), k=1)
                valid = D_mat[mask_ut]
                valid = valid[~np.isnan(valid)]
                means[space] = float(np.mean(valid)) if len(valid) > 0 else 1.0
            ranks[pc_label] = means
        space_rank[ds_name] = ranks

    cross_dataset["feature_space_overlap_ranking"] = space_rank

    logger.info("Phase 6 complete")
    return cross_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — OUTPUT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def _make_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, float) and np.isinf(obj):
        return None
    else:
        return obj


def assemble_output(
    datasets: dict,
    feature_spaces: dict,
    all_crnd_scores: dict,
    niche_overlap: dict,
    per_class_analysis: dict,
    noise_detection: dict,
    cross_dataset_analysis: dict,
    runtime_seconds: float,
) -> dict:
    """Assemble full method_out.json output."""
    logger.info("Assembling output...")

    output = {
        "metadata": {
            "experiment_id": "experiment_iter6_dir1",
            "hypothesis": "Cross-Representation Neighborhood Dissonance + Schoener's D Ecological Niche Overlap",
            "n_datasets": len(datasets),
            "datasets_used": list(datasets.keys()),
            "feature_spaces": ["tfidf", "embed", "llm"],
            "k_values": K_VALUES,
            "noise_rates": NOISE_RATES,
            "n_noise_seeds": N_NOISE_SEEDS,
            "max_sample_n": MAX_SAMPLE_N,
            "llm_model": LLM_MODEL,
            "llm_total_calls": _llm_calls,
            "llm_total_cost_usd": round(_llm_cost, 4),
            "runtime_seconds": round(runtime_seconds, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "datasets": [],
    }

    for ds_name, ds in datasets.items():
        ds_result = {
            "dataset": ds_name,
            "examples": [],
        }

        # Build per-example results
        for i in range(ds["n"]):
            ex = ds["raw_examples"][i]
            example_entry = {
                "input": ex["input"],
                "output": ex["output"],
                "metadata_dataset": ds_name,
                "metadata_n_classes": ds["n_classes"],
                "metadata_n_instances": ds["n"],
            }

            # CRND scores at k=10
            crnd_k10 = all_crnd_scores[ds_name][10]
            example_entry["metadata_crnd_k10"] = round(float(crnd_k10["crnd"][i]), 6)
            example_entry["metadata_crnd_tfidf_embed_k10"] = round(float(crnd_k10["tfidf_embed"][i]), 6)
            example_entry["metadata_crnd_tfidf_llm_k10"] = round(float(crnd_k10["tfidf_llm"][i]), 6)
            example_entry["metadata_crnd_embed_llm_k10"] = round(float(crnd_k10["embed_llm"][i]), 6)

            # CRND at k=5 and k=20
            for k in [5, 20]:
                example_entry[f"metadata_crnd_k{k}"] = round(
                    float(all_crnd_scores[ds_name][k]["crnd"][i]), 6
                )

            # Predict fields: CRND as characterization score
            example_entry["predict_crnd_score"] = str(round(float(crnd_k10["crnd"][i]), 4))
            example_entry["predict_crnd_category"] = (
                "high_dissonance" if crnd_k10["crnd"][i] > 0.7
                else "medium_dissonance" if crnd_k10["crnd"][i] > 0.4
                else "low_dissonance"
            )

            ds_result["examples"].append(example_entry)

        output["datasets"].append(ds_result)

    # Add aggregate results as metadata
    output["metadata"]["per_dataset_aggregate"] = {}
    for ds_name, ds in datasets.items():
        agg = {
            "dataset_info": {
                "n": ds["n"],
                "n_classes": ds["n_classes"],
                "class_distribution": ds["distribution"],
                "class_names": ds["class_names"],
            },
            "feature_space_info": {
                "tfidf": {"shape": list(feature_spaces[ds_name]["tfidf"].shape)},
                "embed": {"shape": list(feature_spaces[ds_name]["embed"].shape)},
                "llm": {"shape": list(feature_spaces[ds_name]["llm"].shape)},
            },
            "crnd_summary": {},
            "niche_overlap_summary": {},
        }

        # CRND summary
        for k in K_VALUES:
            vals = all_crnd_scores[ds_name][k]["crnd"]
            agg["crnd_summary"][f"k{k}"] = {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        # Niche overlap summary
        for space in ["tfidf", "embed", "llm"]:
            for pc in ["pca_2d", "pca_5d"]:
                D_mat = niche_overlap[ds_name][space][pc]["D_matrix"]
                mask_ut = np.triu(np.ones_like(D_mat, dtype=bool), k=1)
                valid = D_mat[mask_ut]
                valid = valid[~np.isnan(valid)]
                agg["niche_overlap_summary"][f"{space}_{pc}"] = {
                    "mean_D": round(float(np.mean(valid)), 4) if len(valid) > 0 else None,
                    "std_D": round(float(np.std(valid)), 4) if len(valid) > 0 else None,
                    "D_matrix": _make_serializable(D_mat),
                    "explained_var": round(
                        niche_overlap[ds_name][space][pc]["explained_variance_ratio"], 4
                    ),
                }

        # D-gap
        for pc in ["pca_2d", "pca_5d"]:
            D_gap = niche_overlap[ds_name][f"D_gap_{pc}"]
            agg["niche_overlap_summary"][f"D_gap_{pc}"] = _make_serializable(D_gap)

        # Per-class analysis
        agg["per_class_analysis"] = _make_serializable(per_class_analysis.get(ds_name, {}))

        # Noise detection
        agg["noise_detection"] = _make_serializable(noise_detection.get(ds_name, {}))

        output["metadata"]["per_dataset_aggregate"][ds_name] = agg

    # Cross-dataset analysis
    output["metadata"]["cross_dataset_analysis"] = _make_serializable(cross_dataset_analysis)

    # Success criteria evaluation
    output["metadata"]["success_criteria_evaluation"] = _evaluate_success_criteria(
        datasets, all_crnd_scores, noise_detection, niche_overlap, per_class_analysis
    )

    return output


def _evaluate_success_criteria(
    datasets: dict,
    all_crnd_scores: dict,
    noise_detection: dict,
    niche_overlap: dict,
    per_class_analysis: dict,
) -> dict:
    """Evaluate the method against predefined success criteria."""
    criteria = {}

    # Criterion 1: Spearman rho > 0.3 between CRND and injected noise
    c1_results = {}
    c1_met_count = 0
    for ds_name in datasets:
        nd = noise_detection.get(ds_name, {})
        best_rho = 0.0
        for nr in ["0.05", "0.1", "0.2"]:
            if nr in nd and "crnd" in nd[nr]:
                rho = nd[nr]["crnd"]["spearman_rho_mean"]
                if abs(rho) > abs(best_rho):
                    best_rho = rho
        c1_results[ds_name] = {"best_spearman_rho": best_rho}
        if abs(best_rho) > 0.3:
            c1_met_count += 1

    criteria["criterion_1_noise_correlation"] = {
        "description": "Spearman rho > 0.3 between CRND and injected noise",
        "results_per_dataset": c1_results,
        "met_count": c1_met_count,
        "total": len(datasets),
        "met": c1_met_count >= len(datasets) // 2,
    }

    # Criterion 2: Niche overlap D-gap provides meaningful signal
    c2_results = {}
    for ds_name in datasets:
        d_gap = niche_overlap[ds_name].get("D_gap_pca_2d")
        if d_gap is not None:
            mask = np.triu(np.ones_like(d_gap, dtype=bool), k=1)
            vals = d_gap[mask]
            valid = vals[~np.isnan(vals)]
            c2_results[ds_name] = {
                "mean_D_gap": round(float(np.mean(valid)), 4) if len(valid) > 0 else None,
                "max_D_gap": round(float(np.max(valid)), 4) if len(valid) > 0 else None,
                "nonzero_fraction": round(float(np.mean(valid > 0.05)), 4) if len(valid) > 0 else None,
            }

    criteria["criterion_2_niche_overlap_d_gap"] = {
        "description": "D-gap reveals representation-sensitive class boundaries (nonzero fraction > 0.5)",
        "results_per_dataset": c2_results,
        "met": sum(
            1 for d in c2_results.values()
            if d.get("nonzero_fraction") is not None and d["nonzero_fraction"] > 0.5
        ) >= len(datasets) // 2,
    }

    # Criterion 3: Interpretable CRND structure (significant Kruskal-Wallis)
    c3_results = {}
    c3_met_count = 0
    for ds_name in datasets:
        pca = per_class_analysis.get(ds_name, {})
        kw = pca.get("kruskal_wallis", {})
        p = kw.get("p_value", 1.0)
        c3_results[ds_name] = {
            "kruskal_wallis_p": p,
            "significant": p < 0.05,
        }
        if p < 0.05:
            c3_met_count += 1

    criteria["criterion_3_interpretable_crnd_structure"] = {
        "description": "CRND varies significantly across classes (Kruskal-Wallis p < 0.05)",
        "results_per_dataset": c3_results,
        "met_count": c3_met_count,
        "total": len(datasets),
        "met": c3_met_count >= 3,
    }

    return criteria


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("CRND + Schoener's D Experiment — Starting")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"MAX_SAMPLE_N: {MAX_SAMPLE_N}")
    logger.info("=" * 60)

    # Phase 0: Load data
    datasets = load_all_datasets()
    if len(datasets) < 3:
        logger.error(f"Only {len(datasets)} datasets loaded — need at least 3")
        return

    # Phase 1: Build feature spaces
    feature_spaces = build_all_feature_spaces(datasets)

    # Phase 2: Compute CRND
    all_knn_sets, all_crnd_scores = run_phase2(datasets, feature_spaces)

    # Phase 3: Schoener's D
    niche_overlap = run_phase3(datasets, feature_spaces)

    # Phase 4: Per-class CRND analysis
    per_class_analysis = run_phase4(datasets, all_crnd_scores)

    # Phase 5: Noise detection
    noise_detection = run_phase5(datasets, feature_spaces, all_knn_sets, all_crnd_scores)

    # Phase 6: Cross-dataset generalization
    cross_dataset_analysis = run_phase6(datasets, all_crnd_scores, niche_overlap)

    # Phase 7: Assemble output
    runtime = time.time() - t_start
    output = assemble_output(
        datasets=datasets,
        feature_spaces=feature_spaces,
        all_crnd_scores=all_crnd_scores,
        niche_overlap=niche_overlap,
        per_class_analysis=per_class_analysis,
        noise_detection=noise_detection,
        cross_dataset_analysis=cross_dataset_analysis,
        runtime_seconds=runtime,
    )

    # Write output
    output_serializable = _make_serializable(output)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output_serializable, indent=2))
    logger.info(f"Output written to {out_path}")

    # File size check
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output size: {size_mb:.1f} MB")

    # Validate JSON
    try:
        json.loads(out_path.read_text())
        logger.info("JSON validation passed")
    except json.JSONDecodeError:
        logger.exception("JSON validation FAILED")

    logger.info(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    logger.info(f"LLM calls: {_llm_calls}, LLM cost: ${_llm_cost:.4f}")
    logger.success("Pipeline complete!")


if __name__ == "__main__":
    main()
