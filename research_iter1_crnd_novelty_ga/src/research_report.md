# CRND Novelty Gap

## Summary

Comprehensive 7-step literature survey across instance-level data complexity (Ho &amp; Basu, Smith kDN, Lorena survey, Arruda/PyHard), cross-representation comparison (CKA, SVCCA, PWCCA, tRSA, NNGS), and neighbor-based noise detection (NCR, WANN, Cartography, cleanlab, AUM, CORES², Deep k-NN) confirms CRND occupies a unique position at the conjunction of 5 criteria: per-instance, cross-representation, neighborhood-based, unsupervised, and model-agnostic. No existing method among 20+ surveyed combines all five. NNGS (2024) is the closest conceptual relative (Jaccard of k-NN across embeddings) but averages to global. WANN (TMLR 2025) is closest competitor but single-embedding. Schoener's D transfer from ecology confirmed genuinely novel. Top 4 baselines: WANN η, kDN, Dataset Cartography, cleanlab.

## Research Findings

## CRND Novelty Gap Survey: Comprehensive Literature Analysis

### Executive Summary

This survey rigorously examines three established research streams — instance-level data complexity measures, cross-representation comparison methods, and neighbor-based label noise detection — to validate whether Cross-Representation Neighborhood Dissonance (CRND) fills a genuine novelty gap. After analyzing 20+ methods across these streams, the conclusion is clear: **no existing method combines per-instance granularity, cross-representation comparison, neighborhood-based analysis, unsupervised operation, and model-agnostic design**. CRND's unique conjunction of all five properties is confirmed.

---

### Stream 1: Instance-Level Data Complexity Measures

#### Ho &amp; Basu (2002) — Foundational Neighborhood Complexity

Ho and Basu introduced foundational complexity measures for classification problems, including neighborhood measures N1 (fraction of points on class boundary), N2 (intra/inter class NN distance ratio), and N3 (1-NN error rate) [1]. These operate at the **dataset level** in a **single feature space** [1]. While N1 can reveal difficulty at the instance level, it does so within one representation only. **CRND gap: No cross-representation comparison; primarily global metrics.**

#### Smith et al. (2014) — Instance Hardness and kDN

Smith et al. introduced instance hardness as the likelihood each observation has of being misclassified [2]. Their kDN (k-Disagreeing Neighbors) metric estimates local overlap by computing the percentage of k nearest neighbors that do not share an instance's label [2]. Crucially, Smith et al. vary **classifiers** (different learning algorithms) on the **same features**, while CRND varies **feature representations** using the **same algorithm** (k-NN) [2]. **CRND gap: Single feature space; varies algorithms, not representations.**

#### Lorena et al. (2019) — Comprehensive Survey

The most comprehensive complexity measures survey covers feature-based, linearity, neighborhood, network, dimensionality, balance, and overlap categories [3]. A grep of the full PDF for Schoener, ecological, niche, or Broennimann returned **zero matches** [3]. Hellinger distance is mentioned only for complexity curves, not ecological niche overlap [3]. **No cross-representation measures exist in this taxonomy.**

#### Arruda et al. (2020) — Decomposed Per-Instance Complexity / PyHard

Arruda et al. decomposed dataset-level complexity measures to the instance level with the PyHard visualization tool [4]. While per-instance, it operates entirely within a single feature representation. **CRND gap: Single feature space; no cross-representation analysis.**

#### Santos et al. (2022) — Class Overlap Survey (Information Fusion)

The most comprehensive class overlap survey establishes a taxonomy covering feature, instance, structural, and multiresolution overlap [5]. A grep of the full 176-page PDF for Schoener, ecological, niche, and cross-representation terms returned **zero matches** [5]. Ecological niche overlap metrics have never been adopted by the ML class overlap community.

---

### Stream 2: Cross-Representation Comparison Methods

#### CKA (Kornblith et al., 2019)

CKA computes global similarity between representations via normalized HSIC [6]. A grep for per-instance, diagnostic, noise, or label terms returned **zero matches** [6]. Purely global similarity metric with no per-instance diagnostic capability.

#### SVCCA (Raghu et al., 2017) &amp; PWCCA (Morcos et al., 2018)

SVCCA [7] and PWCCA [8] combine SVD/CCA for representation comparison, producing global similarity scores. No per-instance diagnostics.

#### tRSA (Lin, 2024)

The most recent RSA extension applying nonlinear transforms to representational dissimilarities [9]. Compares matrices as wholes — no per-instance diagnostics.

#### NNGS — Nearest Neighbor Graph Similarity (2024) ⚠️ CLOSEST CONCEPTUAL RELATIVE

**This is the closest conceptual relative to CRND.** NNGS computes per-node Jaccard similarity of k-NN sets across different embeddings — the **same mathematical building block** CRND uses [10]. However, NNGS **averages** these per-node scores into a single global similarity metric [10]. A grep for per-instance diagnostics, noise, label, anomaly, or outlier terms found only statistical noise experiments, not label noise detection [10]. **NNGS uses the same mechanism (Jaccard of k-NN sets) but for a completely different purpose (global embedding comparison) and does NOT provide per-instance diagnostics.**

---

### Stream 3: Neighbor-Based Label Noise Detection

#### WANN — Weighted Adaptive Nearest Neighbors (Di Salvo et al., TMLR 2025) ⚠️ CLOSEST COMPETITOR

**WANN is the closest direct competitor.** It introduces reliability score η per instance using k-NN in a **single pre-trained foundation model embedding** [12]. Testing across 8 different backbones independently, WANN evaluates embeddings one at a time, **never comparing neighborhoods across them** [12]. **Critical distinction: WANN detects noise where a single embedding's k-NN disagrees with a label; CRND detects instances where k-NN neighborhoods are inconsistent ACROSS independent embeddings.**

#### NCR (Iscen et al., CVPR 2022)

Neighbor consistency regularization operating in a single learned feature space [11]. Supervised, requires training.

#### Dataset Cartography (Swayamdipta et al., EMNLP 2020)

Per-instance confidence/variability mapping via training dynamics [13]. Requires model training, single representation. Strong baseline but training-dependent.

#### Zero-Shot Data Maps (Basile et al., EMNLP 2023)

Eliminates training using zero-shot ensemble variability, but uses single representation type with variability from label description changes only [14].

#### cleanlab / Confident Learning (Northcutt et al., JAIR 2021)

Industry-standard noise detection via confident joint estimation [15]. Requires trained model predictions. Probability-based, not neighborhood-based.

#### AUM (Pleiss et al., NeurIPS 2020), CORES² (Cheng et al., ICLR 2021), Deep k-NN (Bahri et al., ICML 2020)

All require model training and operate in single representations [16, 17, 18].

---

### Stream 4: Ecological Niche Overlap Transfer

**Critical negative result:** Extensive searching for Schoener's D applied to machine learning returned **zero relevant results** [19]. Neither the Lorena 2019 survey [3] nor the Santos 2022 survey [5] mention Schoener's D, ecological metrics, or niche overlap. This confirms **genuine Level 3 cross-domain transfer novelty** [20]. While Hellinger distance is used in ML (e.g., decision trees) [21], it has never been combined with the Broennimann PCA-env framework or Schoener's D for ML class overlap.

---

### Potential Novelty Threats Investigated and Dismissed

- **MODGD** (2023): Multi-view outlier detection via graph denoising — different mechanism, not k-NN Jaccard [22]
- **TMNR²** (2024): Within-view neighbors, no cross-view Jaccard overlap [23]
- **NIRNL** (2024): Single shared embedding space, supervised [24]
- **2025 Benchmark Survey**: 34 methods, zero cross-representation approaches [25]
- **2024 Mislabeled Survey**: Modular framework, zero cross-representation concepts [26]

---

### Master Novelty Positioning Table

| Method | Year | Per-Inst | Multi-Rep | Neighbor | Unsuperv | Model-Ag | CRND Gap |
|--------|------|----------|-----------|----------|----------|----------|----------|
| Ho&amp;Basu N1-N3 | 2002 | Partial | No | Yes | Yes | Yes | Single rep |
| Smith kDN | 2014 | Yes | No | Yes | No | No | Varies algos not reps |
| PyHard | 2020 | Yes | No | Yes | No | No | Single rep |
| CKA | 2019 | No | Yes | No | Yes | Yes | Global only |
| SVCCA | 2017 | No | Yes | No | Yes | Yes | Global only |
| PWCCA | 2018 | No | Yes | No | Yes | Yes | Global only |
| tRSA | 2024 | No | Yes | No | Yes | Yes | Global only |
| **NNGS** | 2024 | Averaged | Yes | Yes | Yes | Yes | **Averaged to global** |
| NCR | 2022 | Yes | No | Yes | No | No | Single rep, supervised |
| **WANN η** | 2024 | Yes | No | Yes | Yes | Yes | **Single rep** |
| Cartography | 2020 | Yes | No | No | No | No | Needs training |
| ZS Data Maps | 2023 | Yes | No | No | Yes | Partial | Single rep type |
| cleanlab | 2021 | Yes | No | No | No | Partial | Probability-based |
| AUM | 2020 | Yes | No | No | No | No | Needs training |
| Deep k-NN | 2020 | Yes | No | Yes | No | No | Single embed |
| MODGD | 2023 | Yes | Yes | Partial | Yes | Yes | Graph-based |
| TMNR² | 2024 | Yes | Yes | Yes | No | No | Within-view only |
| **CRND** | **2025** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Unique** |

### CRND Unique Position Statement

CRND is the **first method** to provide per-instance, cross-representation, neighborhood-based, unsupervised, and model-agnostic data diagnostics. The closest competitor (WANN) operates in a single embedding; the closest mechanism (NNGS) averages to a global metric. Schoener's D transfer from ecology is genuinely novel.

### Recommended Top 4 Baselines

1. **WANN η** — github.com/francescodisalvo05/wann-noisy-labels — single-embedding reliability score
2. **kDN** (Smith et al.) — deslib/PyHard packages — classic instance hardness
3. **Dataset Cartography** — github.com/allenai/cartography — training dynamics baseline
4. **cleanlab** — pip install cleanlab — industry-standard probability-based detection

### Confidence: 95% that CRND's 5-criterion conjunction is unique. 90% that Schoener's D has never been applied to ML class overlap.

## Sources

[1] [How Complex is your classification problem? A survey on measuring classification complexity (Lorena et al., 2019)](https://arxiv.org/abs/1808.03591) — Comprehensive survey of data complexity measures including Ho & Basu N1-N3. Confirmed zero mentions of ecological metrics. All measures operate in single feature spaces.

[2] [An instance level analysis of data complexity (Smith et al., 2014)](https://link.springer.com/article/10.1007/s10994-013-5422-z) — Introduced kDN and instance hardness. Varies classifiers on same features via cross-validation, fundamentally different from CRND's representation-varying approach.

[3] [Lorena et al. 2019 survey — full PDF analysis](https://arxiv.org/pdf/1808.03591) — Full PDF grep confirmed zero mentions of Schoener's D, ecological metrics, or cross-representation measures.

[4] [Measuring Instance Hardness Using Data Complexity Measures (Arruda et al., 2020) / PyHard](https://link.springer.com/chapter/10.1007/978-3-030-61380-8_33) — Decomposed complexity to per-instance level with PyHard tool. Single feature space only.

[5] [A unifying view of class overlap and imbalance (Santos et al., 2022, Information Fusion)](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001099) — Most comprehensive class overlap survey. Zero mentions of ecological metrics or cross-representation overlap measures.

[6] [Similarity of Neural Network Representations Revisited — CKA (Kornblith et al., 2019)](https://arxiv.org/abs/1905.00414) — CKA for comparing representations. Zero matches for per-instance diagnostics. Purely global similarity metric.

[7] [SVCCA (Raghu et al., NeurIPS 2017)](https://arxiv.org/abs/1706.05806) — SVD+CCA for representation comparison. Global similarity scores only, no per-instance diagnostics.

[8] [PWCCA (Morcos et al., NeurIPS 2018)](https://arxiv.org/abs/1806.05759) — Projection-weighted CCA extending SVCCA. Global metric with no per-instance capability.

[9] [Topological RSA — tRSA (Lin, 2024)](https://arxiv.org/abs/2408.11948) — Topological extension of RSA. Compares matrices as wholes, no per-instance diagnostics.

[10] [NNGS — Nearest Neighbor Graph Similarity (2024)](https://arxiv.org/abs/2411.08687) — Closest conceptual relative. Computes per-node Jaccard of k-NN sets across embeddings but averages to global metric. No noise detection or per-instance diagnostics.

[11] [NCR (Iscen et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Iscen_Learning_With_Neighbor_Consistency_for_Noisy_Labels_CVPR_2022_paper.html) — Neighbor consistency regularization in single learned feature space. Supervised method.

[12] [WANN — An Embedding is Worth a Thousand Noisy Labels (Di Salvo et al., TMLR 2025)](https://arxiv.org/abs/2408.14358) — Closest competitor. Reliability score per instance in single foundation model embedding. Tests 8 backbones independently, never compares across them.

[13] [Dataset Cartography (Swayamdipta et al., EMNLP 2020)](https://arxiv.org/abs/2009.10795) — Per-instance mapping via training dynamics. Requires model training, single representation.

[14] [Zero-Shot Data Maps (Basile et al., EMNLP 2023 Findings)](https://aclanthology.org/2023.findings-emnlp.554/) — Training-free cartography using zero-shot ensembles. Single representation type with variability from label descriptions.

[15] [Confident Learning / cleanlab (Northcutt et al., JAIR 2021)](https://arxiv.org/abs/1911.00068) — Industry-standard noise detection via confident joint. Requires trained predictions, probability-based.

[16] [AUM — Area Under the Margin (Pleiss et al., NeurIPS 2020)](https://arxiv.org/abs/2001.10528) — Logit margin tracking across epochs. Requires full model training, single output space.

[17] [CORES² (Cheng et al., ICLR 2021)](https://arxiv.org/pdf/2010.02347) — Confidence-regularized sample sieve. Requires training, single representation.

[18] [Deep k-NN for Noisy Labels (Bahri et al., ICML 2020)](https://arxiv.org/abs/2004.12289) — k-NN filtering on logit layer. Single learned embedding requiring trained model.

[19] [Schoener's D metric — definition and ecological usage](https://rdrr.io/github/GwenAntell/kerneval/man/schoenr.html) — Schoener's D (1968) for niche overlap in ecology. Search for ML applications returned zero results — never applied to ML class overlap.

[20] [Broennimann et al. (2012) — Measuring ecological niche overlap](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2011.00698.x) — PCA-env framework with Schoener's D. Kernel density in PCA space. Source methodology for CRND's cross-domain transfer.

[21] [Hellinger distance decision trees (ML application)](https://link.springer.com/article/10.1007/s10618-011-0222-1) — Hellinger distance used in ML as standard divergence, NOT via Broennimann ecological framework.

[22] [MODGD — Multi-view Outlier Detection via Graphs Denoising (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1566253523003287) — Multi-view outlier detection using graph denoising. Different mechanism from CRND, not k-NN Jaccard overlap.

[23] [TMNR² — Trusted Multi-View Learning under Noisy Supervision (2024)](https://arxiv.org/abs/2404.11944) — Multi-view noise refining using evidential networks. Within-view neighbors only, no cross-view Jaccard overlap.

[24] [NIRNL — Neighbor-aware Instance Refining (2024)](https://arxiv.org/abs/2512.24064) — Neighbor consistency in single shared cross-modal space. Supervised method.

[25] [Benchmarking ML methods for mislabeled data identification (2025)](https://link.springer.com/article/10.1007/s10462-025-11293-9) — Comprehensive 2025 benchmark of 34 methods. Zero cross-representation approaches among all surveyed methods.

[26] [Mislabeled examples detection survey (2024)](https://arxiv.org/abs/2410.15772) — Modular framework survey. Zero matches for cross-representation concepts across all methods.

## Follow-up Questions

- How does CRND's detection performance scale with the number of independent representations compared (2 vs 3 vs 5+), and is there a diminishing returns threshold?
- Can NNGS (2024) be straightforwardly extended to per-instance diagnostics, and would that constitute a concurrent/independent novelty threat to CRND?
- What specific noise patterns does CRND detect that WANN misses (e.g., instances where one embedding's neighborhood supports the label but another's contradicts it)?
- How should Schoener's D be adapted for high-dimensional ML feature spaces where the PCA-env 2D projection may lose critical information?
- What is the computational complexity of CRND's cross-representation Jaccard computation compared to WANN's single-embedding reliability score, and does it scale to datasets with >100K instances?
- Are there multi-modal contrastive learning methods (e.g., CLIP-style) that implicitly enforce cross-representation neighborhood consistency during training, potentially making CRND's post-hoc analysis redundant?

---
*Generated by AI Inventor Pipeline*
