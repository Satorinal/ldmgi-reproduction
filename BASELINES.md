# Executed baselines and cluster visualization

This extension adds independent baseline experiments without modifying `ldmgi.py`, existing LDMGI predictions, sample indices or preprocessing. Every method uses the same complete feature matrix and samples as the corresponding saved LDMGI experiment. All code, reports and figures use English.

## What is implemented

| Method | Implementation | Parameter | Starts |
|---|---|---|---:|
| K-means | scikit-learn KMeans | None | 20 |
| DisKmeans | Exact discriminative-kernel feature transformation, then sklearn KMeans | gamma | 20 per grid point |
| NCut | Gaussian union-kNN affinity; normalized Laplacian; SciPy eigensolver; Yu-Shi discretization | sigma (bandwidth) | 20 per usable grid point |
| LLC-L | Linear-kernel local ridge prediction; prediction-error matrix; SciPy eigensolver; Yu-Shi discretization | eta | 20 per grid point |
| LDMGI | Existing independent implementation and saved runs | lambda | 20 per grid point |

LLC means **local learning-based clustering**, not Locally Linear Embedding (LLE). The paper reports linear and Gaussian versions; this extension implements **LLC-L**, making the kernel choice explicit and avoiding an undocumented second Gaussian-bandwidth selection. LLC-G is not claimed to have been run. DisKmeans is the paper's discriminative K-means, not an ordinary K-means run with a renamed label.

Standard library routines are used where they solve the required problem: KMeans, nearest neighbors, linear solves, sparse eigendecomposition, PCA and t-SNE. The paper-specific matrix construction still needs a small custom implementation. Calling sklearn SpectralClustering with its default affinity or label assignment would not automatically reproduce the paper's NCut configuration.

## Definitions and implementation checks

Inputs use rows for samples. Let A be the globally centered n-by-d feature matrix, with thin SVD A = U S V^T.

### DisKmeans

The paper's Eq.28 is equivalent to kernel K-means with kernel:

```latex
K = A(A^{\mathsf T}A + \gamma I)^{-1}A^{\mathsf T}.
Z = U\,\operatorname{diag}\!\left(\frac{s_j}{\sqrt{s_j^2+\gamma}}\right),
\qquad ZZ^{\mathsf T}=K.
```

Ordinary K-means on Z therefore optimizes the same kernel K-means objective without constructing a dense n-by-n kernel. No truncated PCA approximation is used in this transformation. Minimum within-cluster inertia selects the representative run. `test_baselines.py` verifies the transformed Gram matrix against the direct matrix formula.

### NCut

Five Euclidean nearest neighbors excluding self are used; an undirected edge is present if either endpoint selected the other. No self-loops are added. This union-graph convention is explicit because the paper does not fully specify symmetrization.

```latex
W_{ij}=\exp\!\left(-\frac{\lVert x_i-x_j\rVert^2}{\sigma^2}\right)
\quad\text{on neighbor edges},
\qquad D_{ii}=\sum_j W_{ij},
\qquad L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}.
```

The c smallest eigenvectors, including the trivial direction, are row-normalized and discretized with the existing Yu-Shi routine. The representative minimizes the discrete normalized-cut objective, sum of cut/volume, not the unnormalized count-based LDMGI trace. A test verifies the matrix expression against direct edge cuts.

Gaussian normalization is now computed directly in log space with extended-precision intermediates. Row maxima and log sums are kept separate to avoid cancellation at extreme bandwidths. The discrete normalized-cut objective uses the same stable degree representation. No self-loops, weight floors or changed bandwidths are introduced. Tiny normalized edges can still vanish, representing an effectively disconnected graph; zero embedding rows and invalid starts remain explicitly reported. A computable result at an extreme bandwidth is not necessarily meaningful.

The previous version rejected raw Gaussian underflow. Version 4 replaces that computation and reruns NCut only. Other algorithms and their predictions are unchanged.

### LLC-L

For each sample, N_i contains five neighbors excluding the sample. With the linear kernel K(x,z)=x^T z:

```latex
\alpha_i = (K_i+\eta I)^{-1}k_i,
\qquad A_{i,N_i}=\alpha_i^{\mathsf T},
\qquad L_{\mathrm{LLC}}=(I-A)^{\mathsf T}(I-A).
```

The c smallest eigenvectors are row-normalized and discretized. Representative runs minimize the discrete trace using count-normalized cluster indicators. Tests verify the assembled matrix and its quadratic form against explicit prediction errors.

For large eta, L approaches I. The implementation solves for the smallest eigenvectors of the scaled matrix Q = L-I = A^T A-A-A^T, avoiding loss of the informative small terms. Identity shifts and positive scaling preserve eigenvectors. For smaller eta it uses shift-invert on L. Solver residuals are saved.

**Spectral degeneracy:** for some COIL LLC settings and MNIST NCut at sigma=1, the retained eigenspace gives rows with norm below 1e-12. Disconnected components or numerically weak connections can cause this. These rows are set to zero, rather than dividing by numerical noise. Subsequent argmax ties select cluster 0 deterministically. The number of affected rows is reported in JSON, RESULTS.md and visualization metadata. Such results are degenerate and should not be interpreted as an exact reproduction of the authors' scores. The LDMGI implementation's own zero-row behavior is unchanged.

## Experimental protocol and limitations

- Full datasets: COIL-20 1440, USPS 9298, and the existing balanced MNIST subset of 5000. Input hashes, true labels and sample order are checked before any baseline runs.
- Same grid: 1e-8 through 1e8 in factors of 100; 20 starts with seeds 0-19 at every computable setting. K-means has no grid and runs only 20 times.
- K-means and DisKmeans use sklearn k-means++ initialization, n_init=1 per seed, max_iter=300 and tol=1e-4. This is an explicit implementation choice rather than a claim that the authors used those initializations.
- NCut and LLC use five neighbors excluding self. LDMGI's existing local clique contains self plus four neighbors. These follow the respective local-set definitions, rather than silently forcing them to be identical.
- Each method selects a representative by its own objective within a parameter. Cross-parameter best ACC/NMI and mean ACC/NMI are oracle benchmark selections using labels. Objective scales are never compared across methods or parameters.
- Standard deviations use ddof=0. All generated starts contribute to means; validity and convergence flags are retained. Iterations reaching the KMeans cap are conservatively marked not converged.
- Earlier limitations remain: exact author MNIST indices, interpolation, normalization and seeds are unverified. The reported tables contain this repository's experiments, not copied original baseline scores.
- The four main tables have five algorithm columns, all executed. The older original-paper comparison is preserved separately as LDMGI_REFERENCE_RESULTS.md.

## Run and regenerate

```bash
pip install -r requirements.txt
python -m unittest -v

python run_baselines.py --dataset coil20
python run_baselines.py --dataset usps
python run_baselines.py --dataset mnist

python make_comparison_results.py
python visualize_clusters.py --dataset coil20
python visualize_clusters.py --dataset usps
python visualize_clusters.py --dataset mnist
python audit_baselines.py
python audit_results.py
```

To resume already recorded settings, add `--resume`. Recorded failures are skipped too; omit `--resume` to rerun a method, using `--methods ncut` or `--methods llc_l` to restrict scope. Reports are checkpointed after each parameter. Raw data must be present or downloadable only when actually running models or fitting new projections. Table and curve generation reads existing reports and does not download data.

New outputs are in `results/DATASET/baselines/METHOD/`, `results/comparison/`, and `results/DATASET/visualizations/`. Each baseline keeps per-start predictions and per-parameter cluster statistics, sample CSVs and image galleries. The separate earlier `results/DATASET/kmeans/` remains unchanged; use the new baseline directory for the new independently seeded K-means comparison.

## PCA and t-SNE: what the figures mean

For each dataset, PCA and t-SNE are fitted once without labels on the same preprocessed input features. The coordinates are then reused for every method and parameter. This keeps image locations fixed so the effect of different predicted partitions is visible. These are **input-feature projections colored by clustering results**, not projections of different learned embeddings, and not clustering performed in two dimensions.

- Main figures: fixed numeric parameters **1 and 100**, ground truth plus the four baselines and LDMGI; both PCA and t-SNE. K-means is repeated unchanged because it has no regularizer.
- Additional figures: **LDMGI only at lambda=1e-6, 1e-2, 1e4**, alongside ground truth, for both PCA and t-SNE.
- Total: six figures per dataset, eighteen figures across all three datasets, exported in PNG and SVG.
- PCA uses full-SVD PCA; the first two scores are plotted. t-SNE uses the first 50 PCA scores, perplexity 30, learning_rate='auto', PCA initialization, Barnes-Hut angle=0.5, seed=0 and 1000 iterations.
- All images are shown; there is no visualization-only subsampling. Source order, coordinates and configuration are saved in coordinates.csv, coordinates.npz and metadata.json.
- Colors are aligned by the global Hungarian matching for each predicted partition. The ground-truth panel uses the same class palette. Original cluster IDs remain available in prediction files and per-image reports.
- Runs are chosen by objective, never by how attractive the projection looks. PCA overlap does not prove high-dimensional clustering failed; t-SNE separation does not prove the labels are correct. t-SNE between-group distances and apparent group sizes are not calibrated measurements.

## Sources

- Yang et al., *Image Clustering Using Local Discriminant Models and Global Integration*, IEEE TIP, 2010, Eqs.28 and 31-35. DOI: 10.1109/TIP.2010.2049235.
- Ye, Zhao and Wu, *Discriminative K-means for Clustering*, NeurIPS 2007: https://papers.nips.cc/paper_files/paper/2007/file/a5cdd4aa0048b187f7182f1b9ce7a6a7-Paper.pdf
- Wu and Scholkopf, *A Local Learning Approach for Clustering*, NeurIPS 2006: https://papers.nips.cc/paper_files/paper/2006/file/366f0bc7bd1d4bf414073cabbadfdfcd-Paper.pdf
- Yu and Shi, *Multiclass Spectral Clustering*, ICCV 2003.
- sklearn spectral clustering documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
- sklearn t-SNE documentation: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
