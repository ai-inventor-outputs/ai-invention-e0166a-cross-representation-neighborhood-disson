# CRND Lit Survey

## Summary

Comprehensive literature survey across three areas critical for the CRND paper's novelty claims: (1) multi-representation disagreement diagnostics, (2) hybrid noise detection methods, and (3) ecological niche overlap metrics in ML. Key findings: CRND's ecological metric transfer (Schoener's D) is the strongest novelty claim with zero prior ML applications across 15 searches. SPEC (ICML 2025) and DRES (EMNLP 2025) are the closest Area 1 competitors but differ fundamentally. TMNR (IJCAI 2024) and WANN (TMLR 2025) are closest in Area 2 but use single spaces or require training.

## Research Findings

## Literature Survey: Multi-Representation Disagreement, Hybrid Noise Detection, and Ecological Metrics in ML

### Executive Summary

This survey examines recent (2023–2025) literature across three areas critical for positioning the CRND (Cross-Representation Neighborhood Disagreement) framework. The central finding is that CRND occupies a **unique intersection**: no existing method combines (a) per-instance cross-representation neighborhood disagreement diagnostics with (b) training-free noise detection and (c) ecological niche overlap metrics for class characterization.

---

### Area 1: Multi-Representation Disagreement Diagnostics

The closest competitor is **SPEC** (ICML 2025) [1], which compares embeddings via eigendecomposition of difference kernel matrices to detect sample clusters captured differently by two embeddings. However, SPEC operates at the **cluster/global level**, not per-instance, and uses kernel-based comparison rather than neighborhood Jaccard overlap. It has not been applied to noise detection.

A critical discovery is **DRES** (EMNLP 2025) [2], which computes kDN (k-Disagreeing Neighbors) across **14 text representations** including TF-IDF, Word2Vec, GloVe, FastText, BERT, RoBERTa, LLaMA3, and Mistral. However, DRES computes kDN **independently within each representation** and selects the representation with lowest hardness per instance — it does NOT measure cross-representation neighborhood disagreement or Jaccard overlap between neighborhoods in different spaces [2]. This is a key distinction from CRND.

**Cook et al.** (NAACL 2025) [3] examine instance-level complexity metrics for NLP classification, finding that training loss provides similar complexity rankings as more expensive techniques. They focus on metric redundancy, not cross-representation comparison.

**CKA extensions** (Patch-CKA, RCKA) [4] extend Centered Kernel Alignment to patch-level and relation-level comparison but operate within the same model architecture for knowledge distillation. **DDN** (Neurocomputing 2025) [14] improves kDN with dynamic neighborhoods but remains within a single feature space.

**Novelty assessment**: No paper computes per-instance Jaccard overlap of kNN sets across fundamentally different representation families for noise/difficulty diagnostics. CRND's approach is genuinely novel.

---

### Area 2: Hybrid Noise Detection

**TMNR/TMNR²** (IJCAI 2024) [7] is the closest multi-view competitor, using evidential deep neural networks across views to detect noise via view-specific noise correlation matrices and Dempster-Shafer fusion. Key difference: TMNR **requires training** evidential networks, while CRND is training-free and representation-agnostic.

**WANN** (TMLR 2025) [5] uses foundation model embeddings with weighted adaptive kNN for noise robustness, introducing a per-instance reliability score η. It operates in a **single embedding space** — CRND extends the kNN philosophy to cross-representation comparison [5].

**DeFT** (NeurIPS 2024) [6] leverages VLM text-visual alignment for noise detection — multi-**modal**, not multi-**representation** in the CRND sense. **CoDC** (2024) [8] uses feature-level disagreement between co-trained networks of the same architecture. **DynaCor** (CVPR 2024) [9] uses training dynamics with intentional corruption — requires training. **DSCL** (IEEE TCSVT 2024) [10] combines semantic and feature spaces but for robust training, not diagnostics.

The **NLP noisy labels survey** (2025) [11] classifies methods into five categories (feature vector, transition matrix, prediction confidence, loss improvement, data weighting) — CRND's cross-representation approach doesn't fit any, suggesting genuine taxonomic novelty. **AlleNoise** (AISTATS 2025) [12] demonstrates that synthetic noise methods fail on real-world noise, supporting CRND's training-free diagnostic motivation. The **geometry-aware framework** by Bozkurt & Ortega (2025) [13] offers training-free, NNK-based reliability in single foundation model space — complementary to CRND.

---

### Area 3: Ecological Metrics in ML

Across **15 targeted searches** combining Schoener's D, Warren's I, niche overlap, Broennimann (2012), and ML/classification terms, **ZERO papers** were found applying ecological niche overlap metrics to ML class distributions or feature spaces [15, 16, 17]. Hellinger distance is widely used in ML for class imbalance and feature selection [18], but the specific ecological framework — Schoener's D (1968) and the Broennimann et al. (2012) KDE approach adapted to class distributions — has **no precedent** in ML literature.

No ecology-to-ML crossover workshops were found at NeurIPS, ICML, or ICLR. The comprehensive class overlap survey by Santos et al. (2022) [19] does not reference ecological niche overlap metrics. Schoener (1968) has zero citations in CS/ML venues.

This constitutes the paper's **strongest and most defensible novelty claim**. The mathematical form of Schoener's D (1 − 0.5 × Σ|pᵢ − qᵢ|) is equivalent to 1 minus half the L1 distance between discretized distributions — but no ML paper applies this under the ecological niche overlap interpretation with KDE-based class density estimation.

---

### Updated Positioning Statement

CRND should be positioned as the first per-instance, training-free diagnostic that measures cross-representation neighborhood disagreement. Cite SPEC for global comparison (differentiate: per-instance vs. cluster-level), DRES for multi-representation usage without cross-comparison, WANN for single-space kNN noise robustness, and TMNR for multi-view noise detection requiring training. The ecological metric transfer (Schoener's D) should be presented as a genuinely novel conceptual contribution, explicitly distinguished from standard Hellinger distance usage in ML.

## Sources

[1] [SPEC: Towards an Explainable Comparison and Alignment of Feature Embeddings (ICML 2025)](https://arxiv.org/abs/2506.06231) — Cluster-level embedding comparison via spectral decomposition of difference kernel matrices. Closest Area 1 competitor but global/cluster-level, not per-instance.

[2] [DRES: Fake news detection by dynamic representation and ensemble selection (EMNLP 2025)](https://arxiv.org/abs/2509.16893) — Uses kDN across 14 text representations independently — selects best representation per instance, does NOT measure cross-representation neighborhood disagreement.

[3] [No Simple Answer to Data Complexity: Instance-Level Complexity Metrics (NAACL 2025)](https://aclanthology.org/2025.naacl-long.129/) — Examines complexity metric relationships, finds training loss provides similar rankings. Focuses on metric redundancy, not cross-representation comparison.

[4] [Rethinking CKA in Knowledge Distillation / Patch-CKA (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0628.pdf) — CKA extensions to patch/relation level for knowledge distillation. Operates within same model architecture, not across representation families.

[5] [WANN: An Embedding is Worth a Thousand Noisy Labels (TMLR 2025)](https://arxiv.org/abs/2408.14358) — Foundation model embeddings + weighted adaptive kNN for noise robustness. Single embedding space, per-instance reliability score.

[6] [DeFT: Vision-Language Models are Strong Noisy Label Detectors (NeurIPS 2024)](https://arxiv.org/abs/2409.19696) — Uses VLM text-visual alignment with learnable prompts for noise detection. Multi-modal (text+image), not multi-representation.

[7] [TMNR/TMNR²: Trusted Multi-view Learning with Label Noise (IJCAI 2024)](https://arxiv.org/abs/2404.11944) — Evidential DNNs across views with noise correlation matrices. Closest multi-view noise detection competitor but requires training.

[8] [CoDC: Accurate Learning with Noisy Labels via Disagreement and Consistency (2024)](https://www.mdpi.com/2313-7673/9/2/92) — Co-teaching with feature-level disagreement between same-architecture networks. Not cross-representation.

[9] [DynaCor: Learning Discriminative Dynamics with Label Corruption (CVPR 2024)](https://arxiv.org/abs/2405.19902) — Training dynamics with intentional label corruption for noise detection. Requires model training, single architecture.

[10] [DSCL: Learning With Noisy Labels by Semantic and Feature Space Collaboration (IEEE TCSVT 2024)](https://ieeexplore.ieee.org/document/10454029/) — Dual-space collaborative learning with global prototypes. Requires end-to-end training.

[11] [Survey on Learning with Noisy Labels in NLP (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0952197625001575) — Comprehensive NLP noisy labels taxonomy with five method categories. CRND doesn't fit existing taxonomy.

[12] [AlleNoise: Large-scale Text Classification Benchmark with Real-world Label Noise (AISTATS 2025)](https://arxiv.org/abs/2407.10992) — 500K+ examples showing synthetic noise methods fail on real-world noise. Supports CRND's training-free diagnostic motivation.

[13] [Geometry-Aware Reliability Framework for Foundation Models (CAMSAP 2025)](https://arxiv.org/abs/2508.00202) — Training-free NNK-based reliability estimation in single foundation model embedding. Complementary to CRND.

[14] [Dynamic Disagreeing Neighbors: A Deep Dive into Complexity Estimation (Neurocomputing 2025)](https://www.sciencedirect.com/science/article/pii/S0925231225027481) — Improved kDN with dynamic density-aware neighborhoods, but single feature space only.

[15] [Broennimann et al. (2012): Measuring Ecological Niche Overlap from Occurrence Data](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2011.00698.x) — Source ecological framework for KDE-based niche overlap. No ML citations found.

[16] [Schoener's D and Study Extent (Blog)](https://plantarum.ca/2021/12/02/schoenersd/) — Explanation of Schoener's D metric properties and sensitivity to extent.

[17] [modOverlap: Schoener's D, Hellinger distance and Warren's I](https://modtools.wordpress.com/2015/10/30/modoverlap/) — Ecological overlap metrics implementation and comparison.

[18] [Hellinger Distance-based Feature Selection for Imbalanced Data (BMC Bioinformatics 2020)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3411-3) — Hellinger distance in ML context — distinguishing from ecological niche overlap framework.

[19] [A Unifying View of Class Overlap and Imbalance (Information Fusion 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001099) — Comprehensive class overlap taxonomy. Does NOT include ecological niche overlap metrics.

[20] [Reliable Conflictive Multi-View Learning (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29546) — Multi-view learning addressing conflictive instances with reliability estimation.

[21] [SPEC GitHub Repository](https://github.com/mjalali/embedding-comparison) — Official implementation of SPEC method for embedding comparison.

[22] [Advances in Label-Noise Learning (Curated List)](https://github.com/weijiaheng/Advances-in-Label-Noise-Learning) — Comprehensive tracking of recent noisy label papers across venues.

## Follow-up Questions

- Does DRES (EMNLP 2025) report any analysis of cross-representation neighborhood overlap or disagreement, even as secondary analysis? The full paper should be checked for any Jaccard-style comparison.
- Has SPEC been applied to noise detection in any follow-up work since ICML 2025? Check citing papers.
- Are there ecology-ML crossover papers at domain-specific workshops (AI4Science, ML4Ecology) that might apply niche overlap metrics to classification?
- Could Schoener's D have been independently discovered under a different name in ML? It equals 1 minus half the L1 distance between discretized distributions — verify no ML paper uses this formulation under an ecological interpretation.
- Does TMNR use fundamentally different feature extractors for its views, or just different neural architectures on the same input modality?

---
*Generated by AI Inventor Pipeline*
