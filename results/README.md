# Executed experiments

For the four paper-style tables and regularization figures, see [RESULTS.md](../RESULTS.md). Run `python make_comparison_results.py` for all-method comparisons under `comparison/`. The earlier LDMGI-only comparison can be regenerated with `python make_paper_results.py` under `paper_style/`.

Each parameter uses 20 random starts unless report.json says otherwise. ACC/NMI are fractions in CSV and percentages below.

Within each parameter, the saved representative is selected by minimum discrete LDMGI objective, without labels. The best-mean choices below are explicitly label-based (oracle); ACC and NMI may select different parameters.

| Dataset | n | parameters completed | best mean ACC ± std (%) | lambda | best mean NMI ± std (%) | lambda |
|---|---:|---:|---:|---:|---:|---:|
| mnist | 5000 | 9 | 77.76 ± 2.05 | 1 | 76.50 ± 0.95 | 1 |
| coil20 | 1440 | 9 | 81.95 ± 1.89 | 1 | 94.00 ± 0.66 | 1 |
| usps | 9298 | 9 | 85.50 ± 5.75 | 1 | 87.20 ± 1.73 | 1 |

## Where to inspect individual clusters

For every dataset and parameter, open `DATASET/selected_regVALUE/clusters.csv` (counts, composition, matched precision/recall) and `cluster_examples.png`. `samples.csv` identifies every sample, true class, cluster, matched class and correctness. `contingency.csv` has true classes in rows and predicted clusters in columns. `clusters.json` also records the matching.

A red image border means wrong after global Hungarian matching; green means correct. Images are evenly spaced members in deterministic source order, not best examples. The gallery is a preview; the CSV covers every image.

The representative is not necessarily the maximum-ACC run. Purity uses a separate per-cluster majority and can be higher than matched precision. Empty clusters are retained in diagnostics; unmatched clusters have no matched class.

All trial labels are saved as NPY, all true labels as true_labels.npy; report.json includes every run’s cluster sizes. Ground-truth labels never enter LDMGI. USPS reports source class IDs 1–10 without assuming a digit-glyph mapping.

See README.md for differences from the original paper; these are independent reproduction results, not the authors’ published scores.
