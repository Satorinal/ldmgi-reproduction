# LDMGI: independent Python reproduction

An independent reference implementation of **Image Clustering Using Local Discriminant Models and Global Integration**, Yang et al., IEEE TIP 19(10), 2010, DOI: 10.1109/TIP.2010.2049235. This is not the authors' official code, and exact reproduction of their published scores is not claimed.

The repository includes executed experiments on MNIST, COIL-20 and USPS, four paper-style performance tables comparing five executed methods, combined parameter curves, PCA/t-SNE visualizations, and per-cluster/per-image diagnostics. All repository documentation, code comments and generated reports are in English.

## Quick start

Use Python 3.10 or newer. From the extracted repository directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest -v

# Generate tables and figures from the included results; no downloads or training.
python summarize_results.py
python make_comparison_results.py
python audit_results.py
python audit_baselines.py
```

Open [RESULTS.md](RESULTS.md) for the four tables comparing K-means, DisKmeans, NCut, LLC-L and LDMGI, combined parameter curves and visualization links. See [BASELINES.md](BASELINES.md) for definitions, numerical limitations and all run commands. The previous published-versus-reproduced LDMGI tables and fixed-lambda cluster statistics remain in [LDMGI_REFERENCE_RESULTS.md](LDMGI_REFERENCE_RESULTS.md).

## Run the experiments

```bash
# Single-parameter runs, each with 20 random starts.
python run_evaluation.py --dataset coil20 --reg 1 --output results/coil20_reg1
python run_evaluation.py --dataset usps --reg 1 --output results/usps_reg1
python run_evaluation.py --dataset mnist --samples 5000 --selection balanced --reg 1 --output results/mnist_reg1

# Complete paper grid: nine parameters times 20 starts per dataset.
python run_evaluation.py --dataset coil20 --paper-grid --output results/coil20
python run_evaluation.py --dataset usps --paper-grid --output results/usps
python run_evaluation.py --dataset mnist --paper-grid --output results/mnist
python summarize_results.py
python make_comparison_results.py
python audit_results.py
python audit_baselines.py
```

The saved LDMGI benchmark contains 540 starts; additional baseline runs are stored separately. Coverage and numerical degeneracies are listed in RESULTS.md. NCut now uses log-domain Gaussian normalization. Inputs are downloaded once and cached in `data/`. Raw datasets are excluded from the source archive. To work offline, download the named files from the sources below and place them in `data/`.

Use a distinct `--output` directory to preserve earlier experiments. `report.json` is checkpointed after each parameter; automatic resume is not implemented. One eigensolve is shared by all initializations at each parameter. `--samples` and `--selection` apply only to MNIST; COIL-20 and USPS always use their complete datasets. `--dataset digits` is an offline smoke benchmark using sklearn's 8x8 digits, not MNIST and not a paper benchmark.

## Datasets and preprocessing

| Dataset | Samples x features | Classes | Processing |
|---|---:|---:|---|
| MNIST subset | 5000 x 784 | 10 | 500 randomly sampled test images per digit, seed=0; 28x28 pixels divided by 255 |
| COIL-20 | 1440 x 1024 | 20 | Processed archive; 72 views per object; Pillow bilinear resize to 32x32; divide by 255 |
| USPS | 9298 x 256 | 10 | Concatenate 7291 training and 2007 test samples; 16x16 pixels; map source [-1,1] to [0,1] |

Sources:

- MNIST: https://storage.googleapis.com/cvdf-datasets/mnist/ ; files `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`.
- COIL-20: https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php ; processed `coil-20-proc.zip`. Images are read inside the archive without extracting arbitrary paths.
- USPS: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps ; `usps.bz2` and `usps.t.bz2`.

USPS combines both splits for unsupervised clustering, not supervised train/test evaluation. Labels are used only for evaluation and the known number of classes, as in the paper. Source USPS IDs 1-10 are preserved without treating an ID as the digit glyph. COIL class IDs 1-20 come from object filenames.

**Reproduction limitations:** the paper describes MNIST-T as a first part of the test set with 500 images per class, but exact sample indices have not been verified. Our deterministic balanced subset is not asserted identical. COIL interpolation, pixel scaling and source processing have not all been confirmed against the authors' implementation. Scaling changes the interpretation of regularization. Matching sample counts and the parameter grid does not establish exact reproduction of the paper's tables. These choices and input SHA256 hashes are recorded in each `report.json`.

## All-method tables, curves and PCA/t-SNE

`make_comparison_results.py` reads saved results and generates four tables: best ACC, best NMI, mean ACC with standard deviation, and mean NMI with standard deviation. All five method columns are actual executions on the same samples. Tables and combined parameter curves are in `results/comparison/`, exported as PNG/SVG and numeric CSV; tables also include Markdown and LaTeX versions. K-means is a parameter-free horizontal reference in the curves. NCut uses stable log-domain normalization; any unavailable results remain gaps.

The baseline runner uses standard sklearn KMeans and SciPy primitives with paper-specific matrix construction. LLC is the linear-kernel LLC-L version, not LLE or LLC-G. DisKmeans uses its discriminative kernel through an exact feature transformation. See BASELINES.md before interpreting degeneracies or comparing parameter values with different meanings.

```bash
python run_baselines.py --dataset coil20
python run_baselines.py --dataset usps
python run_baselines.py --dataset mnist
python make_comparison_results.py
python visualize_clusters.py --dataset coil20
python visualize_clusters.py --dataset usps
python visualize_clusters.py --dataset mnist
python audit_baselines.py
```

PCA/t-SNE figures use shared coordinates for all methods, colored by their already-computed predictions. Main comparisons use numeric parameters 1 and 100; additional LDMGI comparisons use 1e-6, 1e-2 and 1e4. Each dataset has six figures under `results/DATASET/visualizations/`, with coordinates and configuration included. Projection fitting never uses true labels and does not change cluster predictions. All eighteen figures include a ground-truth reference.

The earlier original-paper comparison remains available: `python make_paper_results.py` now writes `LDMGI_REFERENCE_RESULTS.md` and the unchanged `results/paper_style/` exports, avoiding overwriting the new all-method RESULTS.md.

### Selection rules

- Neighborhood size k=5: the sample itself plus four Euclidean nearest neighbors. The number of clusters is supplied.
- Grid: 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8; seeds 0-19 by default.
- At a fixed parameter, choose the valid initialization minimizing the original discrete `Tr(G.T @ L @ G)`. Ground truth does not select that initialization.
- Best ACC and best NMI independently choose the best parameter among those objective-selected representatives. They are **not** maxima over every seed's metric.
- Mean tables independently choose the parameter with the highest mean ACC or mean NMI over its starts. Standard deviations use `ddof=0`; all trials contribute, with validity and convergence recorded. Ties use the smallest parameter.
- Cross-parameter choices using ACC/NMI are explicitly **oracle benchmark selection using labels**, not a label-free deployment procedure. ACC and NMI can select different parameters or runs.
- Never compare raw objective magnitudes across regularization values. Higher reproduced scores do not establish improvement over the original algorithm. No significance test is claimed.
- Primary curves use the objective-selected representative at every parameter, matching the paper's reporting convention. Supplementary curves show means and variability; standard deviation bars are not confidence intervals.

## Cluster and image-level reporting

Every dataset and parameter has a `selected_regVALUE/` directory for the objective-selected run:

| File | Content |
|---|---|
| `clusters.csv` | Cluster size, matched class, class total, correct/wrong counts, missed class members, precision, recall, purity, and full true-class composition |
| `clusters.json` | The same statistics plus the Hungarian mapping and contingency matrix |
| `contingency.csv` | True classes in rows and predicted clusters in columns |
| `samples.csv` | Source filename/index, true class, cluster, matched class and correctness for every image |
| `cluster_examples.png` | Up to 12 evenly spaced members per cluster; green borders mean correct after matching, red mean incorrect |

Examples are deterministic previews, not cherry-picked successes. CSV files cover every image. All initializations retain their predictions in `labels_reg..._seed....npy`, and their cluster sizes in `report.json`. True labels are saved in `true_labels.npy`, indices in `indices.npy`. K-means has the same diagnostics under `kmeans/` and predictions in `labels_kmeans.npy`.

Cluster IDs have no intrinsic class meaning. Hungarian matching maximizes the total correct count subject to a one-to-one correspondence. `matched_precision = matched_correct / size`; `matched_recall = matched_correct / matched_class_total`. The sum of matched correct counts divided by the sample count equals ACC. Purity independently chooses each cluster's majority class, so it is not ACC. A pure split cluster can have zero matched precision when its majority class has already been assigned to another cluster. Matching can have tied solutions. NMI does not need label matching.

## Implementation map

| File | Responsibility |
|---|---|
| `ldmgi.py` | Local discriminant matrices, global assembly, eigensolve and Yu-Shi discretization; accepts no true labels |
| `datasets.py` | COIL/USPS download, parsing, validation and preprocessing |
| `run_evaluation.py` | MNIST IDX loading, evaluation orchestration, repeated starts and checkpoints |
| `reporting.py` | Post-hoc cluster statistics, matching, image previews and CSV exports |
| `summarize_results.py` | Cross-dataset numeric index |
| `make_paper_results.py` | Original-paper comparison and LDMGI_REFERENCE_RESULTS.md |
| `baselines.py`, `run_baselines.py` | Four baseline implementations and experiment runner |
| `make_comparison_results.py` | Four all-method tables, combined curves and RESULTS.md |
| `visualize_clusters.py` | Shared-coordinate PCA/t-SNE figures and provenance |
| `audit_baselines.py` | Recompute baseline metrics and verify table selections |
| `audit_results.py` | Recompute metrics from stored predictions and verify diagnostics |
| `test_*.py` | Numerical, reporting and table-selection checks |

Code inputs use `X[n,d]`, the transpose of the paper's `X[d,n]`. For each local set, center its features as A and form `H = I - 11.T/k`. Solve `Li = H (A A.T + lambda I)^-1 H` through linear systems rather than explicitly inverting the matrix (Eq.15). Scatter-add local blocks to global sparse L (Eq.18).

Default `--mode paper` excludes the constant direction and takes c smallest directions in its orthogonal complement (Eq.22). A Householder basis removes exactly the constant vector even when zero eigenvalues repeat; other null directions remain. The executed COIL graph has 13 connected components, so discarding all zero eigenvectors would change the algorithm. Zero embedding rows raise an explicit error. `--mode standard` includes the constant direction as a comparison, not as the paper procedure.

Row normalization follows Eq.23. Yu-Shi discretization alternates row argmax and orthogonal Procrustes SVD (Eq.24). Initialization selects approximately orthogonal sample rows; selecting distinct rows is an explicit implementation choice. Nonconvex discretization does not guarantee a global optimum. Yu-Shi retains the constant direction, whereas the LDMGI procedure excludes it; this distinction is explicit.

ACC uses Hungarian matching. NMI uses geometric normalization (`average_method='geometric'`) to match the paper, not sklearn's arithmetic default. Empty-cluster runs are flagged rather than repaired using labels. The earlier K-means reference uses `n_init=20`; it remains intact. The new baseline comparison instead saves 20 independent `n_init=1` runs with seeds 0-19, allowing means and standard deviations to be reported.

## Resources and validation

The sparse global matrix is converted to dense for the reference eigensolve: O(n^2) storage and worst-case cubic time. One 9298x9298 float64 USPS matrix is approximately 692 MB; several matrices coexist, so allow several GB of free RAM. Do not directly apply this solver to all 70,000 MNIST images. BLAS threads are limited to one in the runner. Timing depends on your hardware; saved timing excludes downloading and some report-generation work.

Fixed versions, hashes, sample order and seeds improve reproducibility, but numerical libraries, repeated eigenvalues and tied neighbors can change results. Input hashes identify the executed data; they are not an official signature verification. `requirements-tested.txt` records the tested dependency versions.

Tests cover local inverse equivalence, symmetry, zero row sums, positive semidefiniteness, eigen residuals, disconnected graphs, decreasing rotation loss, permutation-invariant ACC, empty/fewer/more clusters, sample report consistency, and correct table selection. The GitHub Actions workflow runs unit tests without downloading the full benchmarks.

## References

Raw datasets and paper PDFs are not redistributed. Sample previews support research result inspection; follow each dataset's terms and cite its source.

- Yang et al., *Image Clustering Using Local Discriminant Models and Global Integration*, IEEE TIP 19(10), 2010. DOI: 10.1109/TIP.2010.2049235.
- S. X. Yu and J. Shi, *Multiclass Spectral Clustering*, ICCV 2003, pp.313-319.
