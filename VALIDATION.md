# Validation of the baseline extension

- Twelve unit tests passed, including discriminative-kernel equivalence, local prediction energy and normalized-cut energy.
- All 540 original LDMGI prediction arrays and their metrics were audited.
- All saved baseline prediction arrays were audited against ACC, geometric NMI and cluster sizes; representative sample reports and comparison tables agree.
- Version 4 replaces raw NCut affinity computation with log-domain normalization and stable degree-weighted objective evaluation. NCut is rerun; numerical graph degeneracy remains explicitly reported.
- Empty-cluster and zero-spectral-row flags remain in reports; invalid starts are excluded from objective-based representative selection and retained in means.
- All baseline parameter grids have nine attempted settings; K-means has one parameter-free group of twenty starts.
- Eighteen PCA/t-SNE figures were generated. Every plotted representative was checked against its final saved report.
- Original ldmgi.py, run_evaluation.py, datasets.py, reporting.py, previous prediction arrays and JSON reports were checked byte-for-byte against the prior delivered archive and remain unchanged.
- Repository text and filenames were scanned for Chinese characters; none were found.
- ZIP integrity is verified before delivery. Raw input data is not bundled.
