"""LDMGI equations 15,18,22-24. Input X: samples x features (n,d).
Independent implementation, not official author code. No ground-truth labels.
Dense reference solver: O(n^2) storage. Intended for benchmark-sized datasets.
"""
import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def local_matrix(Xi, reg):
    if not np.isfinite(reg) or reg <= 0:
        raise ValueError('reg must be finite and positive')
    k = len(Xi)
    H = np.eye(k) - np.ones((k,k))/k
    A = Xi - Xi.mean(axis=0, keepdims=True)  # k x d: transpose of paper convention
    Li = H @ solve(A @ A.T + reg*np.eye(k), H, assume_a='pos')
    return (Li + Li.T)/2


def build_laplacian(X, k=5, reg=1.):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all() or not 2 <= k <= len(X):
        raise ValueError('finite n x d input and 2 <= k <= n required')
    n = len(X)
    candidates = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X).kneighbors(X, return_distance=False)
    rows, cols, vals = [], [], []
    for i, neighbors in enumerate(candidates):
        idx = np.r_[i, neighbors[neighbors != i][:k-1]]
        Li = local_matrix(X[idx], reg)
        rows.extend(np.repeat(idx,k)); cols.extend(np.tile(idx,k)); vals.extend(Li.ravel())
    L = coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()
    graph = coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    return L, int(components)


def embed(L, c, mode='paper'):
    """paper: remove ONLY constant direction explicitly, retain other null directions.
    standard: include constant direction (comparison, not paper Eq.22).
    Householder transformation removes the known constant vector without assuming
    the first numerical eigenvector equals it when zero eigenvalues are repeated.
    """
    n = L.shape[0]
    if mode not in ('paper','standard') or not 1 < c < n:
        raise ValueError('mode paper/standard; 1 < c < n required')
    M = L.toarray() if hasattr(L,'toarray') else np.array(L,copy=True)
    if mode == 'paper':
        v = np.ones(n)/np.sqrt(n); v[0] -= 1; v /= np.linalg.norm(v)
        b = M @ v; a = v @ b
        # Q M Q, Q=I-2vv^T; Q e_0 is normalized constant vector.
        M -= 2*np.outer(v,b) + 2*np.outer(b,v) - 4*a*np.outer(v,v)
        values, U = eigh(M[1:,1:], subset_by_index=(0,c-1), driver='evr')
        G = np.vstack([np.zeros((1,c)), U]); G -= 2*np.outer(v, v @ G)
    else:
        values,G = eigh(M, subset_by_index=(0,c-1), driver='evr')
    norms = np.linalg.norm(G,axis=1,keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero embedding row: change neighborhood or inspect disconnected graph; not silently normalized')
    residual = np.linalg.norm(L@G-G*values)/max(np.linalg.norm(L@G),1e-15)
    return G/norms, values, float(residual)


def discretize(F, seed=0, max_iter=300, tol=1e-10):
    """Yu-Shi Sec.2.3 initialization; row argmax then SVD.
    Initialization need not be orthogonal; all subsequent R updates are.
    Empty-cluster runs are returned as invalid, never repaired using labels.
    """
    n,c = F.shape
    rng=np.random.default_rng(seed)
    R=np.zeros((c,c)); chosen=[int(rng.integers(n))]; R[:,0]=F[chosen[0]]
    scores=np.zeros(n)
    for j in range(1,c):
        scores += np.abs(F @ R[:,j-1]); scores[chosen]=np.inf
        idx=int(np.argmin(scores)); chosen.append(idx); R[:,j]=F[idx]
    history=[]; previous=None; converged=False
    for it in range(max_iter):
        labels=np.argmax(F@R,axis=1); Y=np.eye(c)[labels]
        U,s,Vt=np.linalg.svd(Y.T@F,full_matrices=False)
        R=Vt.T@U.T
        loss=float(np.sum((Y-F@R)**2)); history.append(loss)
        if previous is not None and abs(previous-loss) < tol:
            converged=True; break
        previous=loss
    return dict(labels=labels, loss=loss, history=history, converged=converged,
                valid=bool(np.unique(labels).size==c), iterations=it+1)


def objective(L, labels, c):
    Y=np.eye(c)[labels]; counts=Y.sum(axis=0)
    if np.any(counts==0): return float('inf')
    G=Y/np.sqrt(counts)
    return float(np.sum(G*(L@G)))
