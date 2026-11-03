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
    """Normalize Gaussian weights in log space, without self-loops or floors.
    Extended precision reduces cancellation for extreme bandwidths.
    Tiny normalized edges may still vanish: this reflects effective disconnection.
    """
    if sigma<=0:raise ValueError('Positive bandwidth required')
    n,k=neighbors.shape
    edges=set()
    for i,js in enumerate(neighbors):
        for j in js:
            if i!=j:edges.add((min(i,int(j)),max(i,int(j))))
    pairs=np.array(sorted(edges));u,v=pairs.T
    dist=np.sum((X[u]-X[v])**2,axis=1,dtype=np.longdouble)
    rows=np.r_[u,v];cols=np.r_[v,u]
    logw=np.r_[-dist,-dist]/np.longdouble(sigma)**2
    maxima=np.full(n,-np.inf,dtype=np.longdouble)
    np.maximum.at(maxima,rows,logw)
    if not np.isfinite(maxima).all():raise ValueError('Neighbor graph has isolated vertices')
    sums=np.zeros(n,dtype=np.longdouble)
    np.add.at(sums,rows,np.exp(logw-maxima[rows]))
    logdegree=maxima+np.log(sums)
    logs=np.log(sums)
    lognormalized=((logw-maxima[rows])+(logw-maxima[cols]))/2-(logs[rows]+logs[cols])/2
    normalized=np.exp(lognormalized).astype(float)
    S=coo_matrix((normalized,(rows,cols)),shape=(n,n)).tocsr();S.eliminate_zeros()
    L=eye(n,format='csr')-S
    return L,{'log_degree':logdegree,'log_weight_max':maxima,'log_weight_sum':logs},dict(
        components=int(connected_components(S,directed=False,return_labels=False)),
        affinity_nonzeros=int(S.nnz),log_degree_min=float(logdegree.min()),
        normalized_edges_underflowed=int(np.sum(normalized==0)),
        affinity_computation='log-domain normalization; longdouble intermediates; no weight floor')

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
    Y=np.eye(c)[labels]
    if np.any(Y.sum(0)==0):return float('inf')
    if isinstance(degree,dict):
        ld=degree['log_degree'];G=np.zeros_like(Y)
        for j in range(c):
            mask=labels==j
            if 'log_weight_max' in degree:
                a=degree['log_weight_max'][mask];relative=a-a.max()+degree['log_weight_sum'][mask]
            else:relative=ld[mask]-ld[mask].max()
            logsum=np.log(np.exp(relative).sum())
            G[mask,j]=np.exp((relative-logsum)/2).astype(float)
    else:
        vol=degree@Y
        if np.any(vol<=0):return float('inf')
        G=np.sqrt(degree[:,None])*Y/np.sqrt(vol)
    return float(np.sum(G*(L@G)))

def run_spectral(F,L,c,seed,degree=None):
    r=discretize(F,seed);r.pop('history');r['objective']=None
    if r['valid']:r['objective']=ncut_objective(L,degree,r['labels'],c) if degree is not None else objective(L,r['labels'],c)
    return r
