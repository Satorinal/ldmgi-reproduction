# NCut numerical fix (version 4)

The normalized Gaussian graph is computed in log space using extended-precision intermediates. Maxima and log sums are stored separately to avoid cancellation at extreme bandwidths. The normalized-cut objective uses stable degree ratios too. No self-loops, edge floors, extra eigenvector or second spectral-clustering stage is introduced.

All nine bandwidths on all three datasets now complete: 540 NCut starts. Small-bandwidth graphs can still effectively disconnect and produce zero embedding rows or empty clusters. These flags and actual poor scores remain visible. Successful numerical execution is not evidence of good clustering or exact reproduction of the original authors' experiment.

The submitted alternative code builds a complete graph including diagonal ones, uses a Gaussian denominator 2*sigma^2, selects c+1 eigenvectors, then fits a second nearest-neighbor spectral clustering model on those eigenvectors. It therefore changes both the graph and the discretization relative to the intended paper baseline. Diagonal ones prevent zero degrees, but can leave an almost identity affinity and a nearly zero Laplacian at small bandwidths.

Only NCut predictions and dependent reports/figures were regenerated. All non-NCut prediction arrays were verified unchanged against version 3. Twelve unit tests passed; 1680 saved baseline prediction arrays and table statistics were audited. Previous LDMGI predictions remain intact.
