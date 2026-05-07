# Niche→ML Adapt

## Summary

Comprehensive survey of ecological niche overlap metrics (Schoener's D, Hellinger distance, Warren's I) and their adaptation from low-dimensional ecology (2D PCA grids) to high-dimensional ML feature spaces (50-768D). Covers exact formulas verified from ecospat source code, the Broennimann PCA-env 5-step framework, KDE curse of dimensionality with MISE convergence rate O(n^{-4/(d+4)}), dimensionality reduction strategy (2D PCA primary, 5D sensitivity, UMAP/t-SNE rejected), bandwidth selection (Scott's rule), computational feasibility (<10 min for 10K instances, 5 classes, 3 feature spaces), comparison with ML class overlap measures (pycol F1-F3, Bhattacharyya coefficient), and novelty assessment confirming no prior cross-domain application of ecological niche metrics to ML class distributions.

## Research Findings

## Adapting Ecological Niche Overlap Metrics to High-Dimensional ML Feature Spaces

### 1. Mathematical Definitions (Verified from Source Code)

**Schoener's D** was created by Schoener (1968) for quantifying prey item overlap in anoles and is defined as D(p_X, p_Y) = 1 - (1/2) × Σ_i |p_{X,i} - p_{Y,i}| [1]. The metric ranges from 0 (no overlap) to 1 (identical distributions). The continuous/KDE-based form replaces the sum with an integral over density functions [1]. In the ecospat R package, this is implemented as `SchoenerD <- 1 - 0.5 * sum(abs(p1 - p2))` where p1 and p2 are normalized probability vectors (sum = 1) [2, 5].

**Hellinger distance** H(p_X, p_Y) = √(Σ_i (√p_{X,i} - √p_{Y,i})²) was applied to niche overlap by Warren et al. (2008) [6]. **Warren's I** is derived from Hellinger as I = 1 - (H²)/2, also ranging from 0 to 1 [6]. The modOverlap R function implements all three in ~5 lines: `HellingerDist <- sqrt(sum((sqrt(p1) - sqrt(p2))^2))` and `WarrenI <- 1 - ((HellingerDist^2)/2)` [2]. Warren noted that despite theoretical advantages of I over D, both give qualitatively similar results in practice [1, 6].

**Key mathematical relationship**: Schoener's D is equivalent to 1 minus the Total Variation distance. The Bhattacharyya coefficient BC = Σ√(p_i × q_i) is related to Hellinger: H² = 2(1 - BC), meaning Warren's I = BC [17]. This means Warren's I IS the Bhattacharyya coefficient — an important connection between ecology and statistics.

### 2. Broennimann PCA-env Framework (2012)

The standard ecological pipeline for niche overlap measurement follows a 5-step procedure [3, 4]:

1. **PCA Calibration**: PCA on ALL environmental variables from the combined background of both species using `dudi.pca()` from ade4. Retain first 2 axes [4].
2. **Score Projection**: Project each species' occurrences as supplementary rows onto PCA axes using `suprow()` [4].
3. **Grid Construction**: Build a 100×100 grid (R=100) spanning PCA-space extent bounded by background environment min/max values [4, 19].
4. **Kernel Density Estimation**: Apply KDE per species on the 2D grid using `ecospat.grid.clim.dyn()`, which delegates to `kernelUD()` from adehabitatHR or `kde()` from ks package [4, 5].
5. **Overlap Computation**: Normalize density grids so each sums to 1, then compute D = 1 - 0.5 × Σ|p1 - p2| [5].

The framework produces two key outputs: **z.uncor** (raw density normalized to [0,1]) and **z.cor** (density corrected for environmental prevalence: z/Z where Z is background density) [4]. This prevalence correction has no direct ML analogue but could correspond to correcting for the prior distribution of features.

**Critical insight**: The framework ALWAYS reduces to 2D before KDE. This is not arbitrary but reflects the fundamental impossibility of grid-based KDE beyond d=3-4 [3, 4].

### 3. Alternative Methods

**nicheROVER** (Swanson et al. 2015) works in arbitrary dimensions by assuming X ~ N(μ, Σ) and defining niche regions as α-probability regions of the multivariate normal [7, 8]. Overlap = P(individual from species A falls in niche region of B). Advantages: no grid needed, Bayesian uncertainty quantification, works in any dimension. Limitation: the multivariate normality assumption may not hold for complex ML class distributions, especially multimodal ones [7].

**Hypervolume methods** construct n-dimensional bounding volumes but suffer from the curse of dimensionality: 5% per-dimension sampling error compounds to 40% total error at d=10 [20].

**Consensus**: PCA to 2D + grid-KDE (Broennimann) is the standard approach in ecology [3, 4, 20].

### 4. KDE Curse of Dimensionality

The optimal MISE convergence rate for multivariate KDE is O(n^{-4/(d+4)}) [9]. Any reasonable bandwidth selector has H = O(n^{-2/(d+4)}) [9]. This means:

| Dimensions | MISE exponent (n=10K) | Quality |
|---|---|---|
| d=2 | -0.667 | Good |
| d=5 | -0.444 | Acceptable |
| d=10 | -0.286 | Marginal |
| d=50 | -0.074 | Unusable |
| d=768 | -0.005 | Hopeless |

**Grid infeasibility**: With R=100, a d-dimensional grid has 100^d cells. At d=5 this is 10^10 cells (infeasible). Grid-based KDE is limited to d≤3 [9, 10].

Wang & Scott (2019) define "high-dimensional" for density estimation as 3 < d ≤ 50 and note the ideal density estimator should handle d = 4 to 50, but acknowledge this remains extremely challenging [10].

### 5. Dimensionality Reduction Strategies

**Primary recommendation: 2D PCA** — Directly faithful to Broennimann's PCA-env framework. 100×100 grid perfectly feasible. No curse of dimensionality. PCA preserves Euclidean distances (required for meaningful KDE with Gaussian kernels) [3, 4]. Limitation: may explain < 20% variance for 768-d BERT embeddings.

**Sensitivity analysis: 5D PCA** — Retains more variance (typically >80-90% for 50-d features). KDE still converges at acceptable rate for n=10,000 [9]. Use sampling-based Schoener's D: fit per-class KDEs in 5D, evaluate at union of data points, normalize, compute D [11].

**UMAP/t-SNE are NOT appropriate**: UMAP does not preserve density well due to its uniform density assumption [13]. The UMAP author (McInnes) confirmed that distances in the embedding are not precisely interpretable, axes have no meaning, and "the scale of UMAP output is somewhat arbitrary" [14]. This makes UMAP unsuitable for KDE-based overlap computation where both distances and densities must be meaningful.

**Separate vs. shared PCA**: Apply PCA separately per feature space (analogous to ecospat PCA-env where PCA is calibrated on the combined background within each environmental context) [3, 4].

### 6. Bandwidth Selection

**Scott's rule**: h = n^{-1/(d+4)} × σ_i per dimension [9, 11]. In scipy: `scotts_factor = n**(-1./(d+4))` [11]. For n=10,000: h ≈ 0.215σ at d=2 (good resolution), h ≈ 0.374σ at d=5 (acceptable), h ≈ 0.464σ at d=10 (severe oversmoothing) [9, 11].

**Silverman's rule**: h = (n(d+2)/4)^{-1/(d+4)} × σ_i, gives slightly smaller bandwidth [11].

scipy.stats.gaussian_kde explicitly recommends: "for data that lies in a lower-dimensional subspace, consider performing PCA / dimensionality reduction and using gaussian_kde with the transformed data" [11].

**Recommendation**: Scott's rule default for d≤5. Leave-one-out CV bandwidth as robustness check. For d>5, oversmoothing is severe [9, 11].

### 7. Computational Feasibility

For the target scenario (N=10K, 5 classes, 3 feature spaces, 10 class pairs per space = 30 computations):

- **2D PCA + grid** (ecospat-style): < 1 minute total. 100×100 grid = trivial [4, 5].
- **5D PCA + sampling**: 2-5 minutes total. n_class ≈ 2000, m ≈ 10000 evaluation points → ~20M ops per pair → seconds [11, 12].
- **10D PCA + sampling**: < 10 minutes total, but results less reliable [9, 12].

**Python libraries**: scipy.stats.gaussian_kde (Scott/Silverman bandwidth, simple API, O(nm) evaluation) [11]; sklearn KernelDensity (tree-based acceleration, ball_tree for moderate d) [12].

All approaches are well within a 1-hour budget.

### 8. Novelty Assessment and ML Comparison

**No prior work found** applying Schoener's D, Hellinger distance, or Warren's I to ML class distributions in feature spaces [extensive search]. The ecological niche overlap literature remains entirely within ecology, while ML class overlap literature (pycol, Ho & Basu measures) uses entirely different metrics [15, 16, 22].

**Critical distinction**: ML complexity measures (F1-F4 feature overlap, N1-N4 neighborhood measures) all work within ONE feature space to characterize classification difficulty [15, 16]. The ecological approach + CRND compares overlap ACROSS representations — a fundamentally different and novel question.

**Mathematical equivalence note**: Schoener's D equals 1 - Total Variation distance, and Warren's I equals the Bhattacharyya coefficient [1, 17]. These are well-known in statistics. The novelty is NOT the metric itself but: (1) applying it to class distributions rather than species distributions, (2) the PCA-env framework adaptation, and (3) the cross-representation comparison paradigm.

### 9. Failure Contingencies

- If 5D KDE fails: fall back to 2D PCA (always works) [3, 4]
- If 2D variance too low: report limitation, use nicheROVER parametric overlap as complementary metric [7, 8]
- If distributions strongly multimodal: Scott's rule oversmooths — try CV bandwidth or mixture-model-based overlap [9, 11]

### Contradicting Evidence and Limitations

- The 2D PCA reduction may lose substantial information for high-dimensional feature spaces (768-d BERT embeddings → 2D could explain &lt;20% variance), potentially making overlap measurements misleading
- The z.cor prevalence correction in ecospat has no clear ML analogue — using z.uncor (raw density) is more appropriate for ML but loses the ecological framework's robustness to environmental frequency bias [4]
- nicheROVER's multivariate normal assumption, while restrictive, could actually be appropriate for certain ML feature spaces (e.g., penultimate layer features often approximately Gaussian after batch normalization) [7]

## Sources

[1] [Schoener's D and Study Extent - plantarum.ca](https://plantarum.ca/2021/12/02/schoenersd/) — Comprehensive walkthrough of Schoener's D formula, origin (Schoener 1968), Warren's I, Hellinger distance, with R code and worked examples.

[2] [Assess model overlap with Schoener's D, Hellinger distance and Warren's I - modTools](https://modtools.wordpress.com/2015/10/30/modoverlap/) — Complete R implementation (modOverlap function) showing exact code for computing all three metrics from two prediction vectors.

[3] [Broennimann et al. (2012) - Measuring ecological niche overlap](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2011.00698.x) — Primary reference for PCA-env framework introducing kernel density estimation on 2D PCA grids for niche overlap.

[4] [Ecospat Niche Overlap Analysis - plantarum.ca](https://plantarum.ca/notebooks/ecospat/) — Step-by-step tutorial of ecospat framework showing PCA calibration, grid.clim.dyn, R=100, z.uncor vs z.cor.

[5] [ecospat source code - ecospat.niche.overlap](https://raw.githubusercontent.com/cran/ecospat/master/R/ecospat.nicheoverlap.R) — R source code showing D = 1 - (0.5 * sum(abs(p1 - p2))) and I = 1 - (H^2)/2.

[6] [Warren et al. (2008) - Environmental Niche Equivalency vs Conservatism](https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.2008.00482.x) — Original paper proposing Warren's I based on Hellinger distance for niche overlap.

[7] [Swanson et al. (2015) - nicheROVER: n-dimensional niche overlap](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/14-0235.1) — Introduces nicheROVER for n-dimensional overlap using multivariate normal and Bayesian framework.

[8] [nicheROVER R package documentation](https://cran.r-project.org/web/packages/nicheROVER/nicheROVER.pdf) — Package docs for Normal-Inverse-Wishart prior, alpha-probability niche regions in arbitrary dimensions.

[9] [Multivariate kernel density estimation - Wikipedia](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation) — MISE convergence rate O(n^{-4/(d+4)}), bandwidth parametrizations, Scott's and Silverman's rules.

[10] [Wang & Scott (2019) - Nonparametric Density Estimation for High-Dimensional Data](https://arxiv.org/pdf/1904.00176) — Review of density estimation algorithms for high-dimensional data (3 < d <= 50).

[11] [scipy.stats.gaussian_kde documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) — Python KDE with scotts_factor = n**(-1./(d+4)), recommends PCA for lower-dimensional subspace data.

[12] [sklearn.neighbors.KernelDensity documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) — Tree-based KDE; KD tree fast for d<20, ball tree for higher d but still limited.

[13] [UMAP FAQ - density preservation](https://umap-learn.readthedocs.io/en/latest/faq.html) — UMAP author confirms density not well preserved due to uniform density assumption.

[14] [UMAP Issue #92 - Are euclidean distances interpretable?](https://github.com/lmcinnes/umap/issues/92) — McInnes states distances not precisely interpretable, scale is 'somewhat arbitrary'.

[15] [Lorena et al. (2019) - Survey of Data Complexity Measures](https://arxiv.org/pdf/1808.03591) — Comprehensive survey of ML complexity measures: F1-F4, N1-N4. Extends Ho & Basu (2002).

[16] [pycol - Python Class Overlap Library](https://github.com/miriamspsantos/pycol) — Python library for class overlap complexity measures, all within single feature space.

[17] [Bhattacharyya distance - Wikipedia](https://en.wikipedia.org/wiki/Bhattacharyya_distance) — BC = sum(sqrt(p*q)), DB = -ln(BC). H^2 = 2(1-BC), so Warren's I = BC.

[18] [Distribution-Free Overlapping Index (2019)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.01089/full) — Distribution-free overlap linked to KL divergence and Bhattacharyya distance.

[19] [ecospat.grid.clim.dyn documentation](https://rdrr.io/cran/ecospat/man/ecospat.grid.clim.dyn.html) — Function docs for dynamic occurrence density grid with R=100 default and kernel methods.

[20] [Hypervolume concepts in niche ecology](https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.03187) — Curse of dimensionality in hypervolumes: 5% per-dim error = 40% at d=10.

[21] [Lecture 7: Density Estimation - UW](https://faculty.washington.edu/yenchic/17Sp_403/Lec7-density.pdf) — Formal treatment of KDE convergence rates and curse of dimensionality.

[22] [A unifying view of class overlap and imbalance (2022)](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001099) — Class overlap more harmful than imbalance; taxonomy of overlap measures.

[23] [Multivariate KDE - Nonparametric Statistics Notes](https://bookdown.org/egarpor/NP-UC3M/kde-ii-mult.html) — Textbook treatment of multivariate KDE bandwidth selection and convergence.

## Follow-up Questions

- How does the explained variance ratio of 2D PCA differ across feature space types (penultimate layer vs logits vs embeddings) for common ML architectures, and at what threshold should overlap results be considered unreliable?
- Could a mixture-model-based overlap metric (e.g., using Gaussian Mixture Models instead of KDE) provide better handling of multimodal class distributions in higher dimensions while avoiding the curse of dimensionality?
- What is the sensitivity of Schoener's D to the choice of grid resolution R (e.g., 50 vs 100 vs 200) and to the prevalence correction (z.cor vs z.uncor) in the ML context where 'environmental prevalence' has no direct analogue?

---
*Generated by AI Inventor Pipeline*
