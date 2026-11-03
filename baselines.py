"""Paper-defined baselines using SciPy/sklearn; inputs are samples x features.
NCut: Eqs.31-32. LLC-L: Eqs.33-35. DisKmeans: Eq.28. No true labels.
"""
import numpy as np
from scipy.linalg import svd,solve,eigh
from scipy.sparse import coo_matrix,eye,diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from ldmgi import discretize,objective

def neighbors_excluding_self(X,k=5):
    ids=NearestNeighbors(n_neighbors=k+1,algorithm='brute').fit(X).kneighbors(X,return_distance=False)
    return np.array([a[a!=i][:k] for i,a in enumerate(ids)])

def discriminant_svd(X):
    U,s,_=svd(X-X.mean(0,keepdims=True),full_matrices=False,check_finite=False)
    return U,s

def discriminant_features(U,s,reg):
    if reg<=0:raise ValueError('Positive regularization required')
    return U*(s/np.sqrt(s*s+reg))

def run_kmeans(X,c,seed):
    model=KMeans(n_clusters=c,n_init=1,init='k-means++',random_state=seed,max_iter=300,tol=1e-4).fit(X)
    return dict(labels=model.labels_,objective=float(model.inertia_),iterations=int(model.n_iter_),
                converged=bool(model.n_iter_<300),valid=bool(len(np.unique(model.labels_))==c))

def ncut_matrix(X,neighbors,sigma):
    if sigma<=0:raise ValueError('Positive bandwidth required')
    n,k=neighbors.shape;distances=np.sum((X[:,None,:]-X[neighbors])**2,axis=2)
    weights=np.exp(-distances/(sigma*sigma))
    A=coo_matrix((weights.ravel(),(np.repeat(np.arange(n),k),neighbors.ravel())),shape=(n,n)).tocsr()
    A=A.maximum(A.T);A.eliminate_zeros()
    degree=np.asarray(A.sum(1)).ravel()
    if np.any(degree<=np.finfo(float).tiny):
        raise ValueError(f'Gaussian affinity underflow: {int(np.sum(degree<=np.finfo(float).tiny))} isolated/underflowed vertices; bandwidth not changed')
    inv=diags(1/np.sqrt(degree));L=eye(n,format='csr')-inv@A@inv
    return L,degree,dict(components=int(connected_components(A,directed=False,return_labels=False)),affinity_nonzeros=int(A.nnz),degree_min=float(degree.min()))

def llc_linear_matrix(X,neighbors,reg):
    if reg<=0:raise ValueError('Positive regularization required')
    n,k=neighbors.shape;weights=[]
    for i,ids in enumerate(neighbors):
        Z=X[ids];weights.extend(solve(Z@Z.T+reg*np.eye(k),Z@X[i],assume_a='pos'))
    A=coo_matrix((weights,(np.repeat(np.arange(n),k),neighbors.ravel())),shape=(n,n)).tocsr()
    Q=(A.T@A-A-A.T).tocsr()  # Q=L-I avoids cancellation for large reg
    return eye(n,format='csr')+Q,Q

def spectral_rows(L,c,Q=None,reg=None):
    """c smallest directions; do not discard a direction for NCut/LLC.
    Scaling and identity shifts preserve eigenvectors. Return row-normalized G.
    """
    n=L.shape[0];v0=np.random.default_rng(0).normal(size=n)
    if Q is not None and reg>=1e4:
        scale=float(np.max(np.asarray(abs(Q).sum(1))))
        if not scale>0:raise ValueError('LLC matrix is numerically identity')
        B=Q/scale
        if n<=128:values,G=eigh(B.toarray(),subset_by_index=(0,c-1))
        else:values,G=eigsh(B,k=c,which='SA',tol=1e-9,maxiter=20000,v0=v0)
        order=np.argsort(values);values=values[order];G=G[:,order]
        err=np.linalg.norm(B@G-G*values)/max(1.,np.linalg.norm(B@G));actual=1+scale*values
        solver='eigsh SA on scaled (L-I)'
    else:
        if n<=128:values,G=eigh(L.toarray(),subset_by_index=(0,c-1))
        else:values,G=eigsh(L,k=c,sigma=-1e-7,which='LM',tol=1e-9,maxiter=20000,v0=v0)
        order=np.argsort(values);values=values[order];G=G[:,order]
        err=np.linalg.norm(L@G-G*values)/max(1.,np.linalg.norm(L@G));actual=values
        solver='eigsh shift-invert, sigma=-1e-7'
    if not np.isfinite(err) or err>1e-6:raise ValueError(f'Unacceptable eigen residual {err}')
    norms=np.linalg.norm(G,axis=1,keepdims=True)
    zero_rows=norms[:,0]<1e-12
    # Disconnected or numerically weak graphs can omit samples from the retained subspace.
    # Preserve zero rows; argmax ties then deterministically select cluster 0.
    # Record this degeneracy rather than amplify roundoff by division.
    G[zero_rows]=0;norms[zero_rows]=1
    return G/norms,dict(eigenvalues=actual.tolist(),eigen_residual=float(err),solver=solver,zero_embedding_rows=int(zero_rows.sum()),zero_row_policy='keep zero; deterministic argmax tie')

def ncut_objective(L,degree,labels,c):
    Y=np.eye(c)[labels];vol=degree@Y
    if np.any(vol<=0):return float('inf')
    G=np.sqrt(degree[:,None])*Y/np.sqrt(vol)
    return float(np.sum(G*(L@G)))

def run_spectral(F,L,c,seed,degree=None):
    r=discretize(F,seed);r.pop('history');r['objective']=None
    if r['valid']:r['objective']=ncut_objective(L,degree,r['labels'],c) if degree is not None else objective(L,r['labels'],c)
    return r
