"""MNIST / COIL-20 / USPS -> features -> label-free LDMGI -> ACC and geometric NMI.
Optionally use --dataset digits for an offline smoke benchmark (NOT MNIST).
"""
import argparse, gzip, hashlib, json, platform, time, urllib.request
from pathlib import Path
import numpy as np
import scipy, sklearn, PIL
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from ldmgi import build_laplacian, embed, discretize, objective
from datasets import load_extra
from reporting import cluster_summary, write_diagnostics


def acc(y, pred):
    _,y=np.unique(y,return_inverse=True); _,pred=np.unique(pred,return_inverse=True)
    C=np.zeros((y.max()+1,pred.max()+1),dtype=int); np.add.at(C,(y,pred),1)
    r,c=linear_sum_assignment(-C)
    return float(C[r,c].sum()/len(y))


def load_mnist(folder, selection, limit, seed):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    arrays=[]; hashes={}
    for name,magic,header in [('t10k-images-idx3-ubyte.gz',2051,16),('t10k-labels-idx1-ubyte.gz',2049,8)]:
        p=folder/name
        if not p.exists():
            tmp=p.with_suffix('.part')
            urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/'+name,tmp)
            tmp.replace(p)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        raw=gzip.decompress(p.read_bytes()); h=np.frombuffer(raw[:header],dtype='>u4')
        if int(h[0]) != magic: raise ValueError('Invalid IDX magic')
        data=np.frombuffer(raw[header:],dtype=np.uint8)
        if data.size != int(np.prod(h[1:])): raise ValueError('Invalid IDX dimensions')
        arrays.append(data.reshape(tuple(map(int,h[1:]))))
    images,y=arrays
    if len(images)!=len(y): raise ValueError('Image/label count mismatch')
    if selection=='balanced':
        if limit%10: raise ValueError('Balanced sample count must be divisible by 10')
        rng=np.random.default_rng(seed)
        idx=np.concatenate([rng.choice(np.flatnonzero(y==j),limit//10,replace=False) for j in range(10)])
        idx.sort()
    else: idx=np.arange(limit)
    return images[idx].reshape(len(idx),-1).astype(float)/255., y[idx], idx, hashes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mnist','coil20','usps','digits'],default='mnist')
    p.add_argument('--data-dir',default='data'); p.add_argument('--samples',type=int,default=5000)
    p.add_argument('--selection',choices=['balanced','first'],default='balanced')
    p.add_argument('--reg',type=float,nargs='+',default=[1.])
    p.add_argument('--paper-grid',action='store_true',help='Use nine regularizers 10^-8,...,10^8')
    p.add_argument('--k',type=int,default=5); p.add_argument('--runs',type=int,default=20)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--mode',choices=['paper','standard'],default='paper')
    p.add_argument('--output',default=None); args=p.parse_args()
    if args.paper_grid: args.reg=[10.**i for i in range(-8,9,2)]
    if args.output is None: args.output='results/'+args.dataset
    if args.runs<1 or not 10<=args.samples<=10000: p.error('runs>=1; 10<=samples<=10000')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.dataset=='mnist':
        X,y,idx,hashes=load_mnist(args.data_dir,args.selection,args.samples,args.seed)
        metadata={'image_shape':(28,28),'source_ids':['t10k:'+str(i) for i in idx],
                  'preprocessing':'Raw 28x28 /255; '+args.selection+' subset, seed '+str(args.seed)+'; not verified as the authors exact MNIST-T subset.'}
    elif args.dataset in ('coil20','usps'):
        X,y,idx,hashes,metadata=load_extra(args.dataset,args.data_dir)
    else:
        from sklearn.datasets import load_digits
        data=load_digits(); X=data.data/16.; y=data.target; idx=np.arange(len(y)); hashes={}
        metadata={'image_shape':(8,8),'source_ids':idx.tolist(),'preprocessing':'sklearn digits smoke test; not a paper dataset'}
    c=len(np.unique(y)); np.save(out/'indices.npy',idx); np.save(out/'true_labels.npy',y)
    report={'config':vars(args),'shape':list(X.shape),'class_counts':np.bincount(y).tolist(),
            'class_distribution':{str(int(a)):int(np.sum(y==a)) for a in np.unique(y)},
            'sha256':hashes,'dataset_metadata':metadata,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'Pillow':PIL.__version__},'results':[]}
    with threadpool_limits(limits=1):
        for reg in args.reg:
            start=time.perf_counter(); L,components=build_laplacian(X,args.k,reg)
            F,values,residual=embed(L,c,args.mode); trials=[]
            for run in range(args.runs):
                r=discretize(F,args.seed+run); labels=r.pop('labels'); r.pop('history')
                r.update(seed=args.seed+run, objective=objective(L,labels,c) if r['valid'] else None,
                         acc=acc(y,labels),nmi=float(normalized_mutual_info_score(y,labels,average_method='geometric')))
                r['cluster_sizes']=np.bincount(labels,minlength=c).tolist()
                np.save(out/f'labels_reg{reg:g}_seed{args.seed+run}.npy',labels); trials.append(r)
            valid=[r for r in trials if r['valid']]
            best=min(valid,key=lambda r:r['objective']) if valid else None
            row={'reg':reg,'components':components,'eigenvalues':values.tolist(),'eigen_residual':residual,'seconds':time.perf_counter()-start,
                 'trials':trials,'selected_by_objective':best,
                 'mean_acc':float(np.mean([r['acc'] for r in trials])), 'std_acc':float(np.std([r['acc'] for r in trials])),
                 'mean_nmi':float(np.mean([r['nmi'] for r in trials])), 'std_nmi':float(np.std([r['nmi'] for r in trials]))}
            if best is not None:
                labels=np.load(out/f"labels_reg{reg:g}_seed{best['seed']}.npy")
                row['diagnostics']=write_diagnostics(out/f'selected_reg{reg:g}',y,labels,X,metadata['image_shape'],metadata['source_ids'],c)
            report['results'].append(row)
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print({k:v for k,v in row.items() if k not in ('trials','eigenvalues')},flush=True)
        pred=KMeans(n_clusters=c,n_init=20,random_state=args.seed).fit_predict(X)
        np.save(out/'labels_kmeans.npy',pred)
        write_diagnostics(out/'kmeans',y,pred,X,metadata['image_shape'],metadata['source_ids'],c)
        report['kmeans']={'acc':acc(y,pred),'nmi':float(normalized_mutual_info_score(y,pred,average_method='geometric'))}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',out/'report.json', 'K-means:',report['kmeans'])
if __name__=='__main__': main()
