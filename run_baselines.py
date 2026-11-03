"""Run four baselines on exact saved LDMGI samples, in separate directories."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import scipy,sklearn
from threadpoolctl import threadpool_limits
from sklearn.metrics import normalized_mutual_info_score
from run_evaluation import load_mnist,acc
from datasets import load_extra
from reporting import write_diagnostics
from baselines import (neighbors_excluding_self,discriminant_svd,discriminant_features,
    run_kmeans,ncut_matrix,llc_linear_matrix,spectral_rows,run_spectral)
METHODS=('kmeans','diskmeans','ncut','llc_l')
GRID=[10.**i for i in range(-8,9,2)]

def load_saved_dataset(name,root='results',data_dir='data'):
    source=json.loads((Path(root)/name/'report.json').read_text());a=source['config']
    if name=='mnist':X,y,idx,hashes=load_mnist(data_dir,a['selection'],a['samples'],a['seed'])
    else:X,y,idx,hashes,_=load_extra(name,data_dir)
    if hashes!=source['sha256']:raise ValueError('Input hashes differ from saved LDMGI experiment')
    np.testing.assert_array_equal(y,np.load(Path(root)/name/'true_labels.npy'))
    np.testing.assert_array_equal(idx,np.load(Path(root)/name/'indices.npy'))
    return X,y,source

def tag(reg):return 'none' if reg is None else f'{reg:g}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['mnist','coil20','usps'])
    p.add_argument('--root',default='results');p.add_argument('--data-dir',default='data')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--reg',type=float,nargs='+',default=GRID);p.add_argument('--runs',type=int,default=20)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.runs<1:p.error('runs must be positive')
    X,y,source=load_saved_dataset(a.dataset,a.root,a.data_dir);c=len(np.unique(y));k=5
    out=Path(a.root)/a.dataset/'baselines';out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        neighbors=neighbors_excluding_self(X,k);U=s=None
        for method in a.methods:
            folder=out/method;folder.mkdir(exist_ok=True);path=folder/'report.json'
            report=dict(method=method,dataset=a.dataset,shape=list(X.shape),runs=a.runs,k=k,
              sha256=source['sha256'],seeds=list(range(a.runs)),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
              implementation='LLC-L linear kernel' if method=='llc_l' else method,results=[])
            if a.resume and path.exists():
                report=json.loads(path.read_text())
                if report['runs']!=a.runs or report['sha256']!=source['sha256']:raise ValueError('Resume configuration mismatch')
            params=[None] if method=='kmeans' else a.reg
            if method=='diskmeans' and U is None:U,s=discriminant_svd(X)
            for reg in params:
                if any(row['reg']==reg for row in report['results']):continue
                start=time.perf_counter();row={'reg':reg,'status':'ok'}
                try:
                    if method=='kmeans':Z=X
                    elif method=='diskmeans':Z=discriminant_features(U,s,reg)
                    elif method=='ncut':
                        L,degree,diag=ncut_matrix(X,neighbors,reg);F,diag2=spectral_rows(L,c);row.update(diag);row.update(diag2)
                    else:
                        L,Q=llc_linear_matrix(X,neighbors,reg);F,diag=spectral_rows(L,c,Q,reg);row.update(diag)
                    trials=[]
                    for seed in range(a.runs):
                        r=run_kmeans(Z,c,seed) if method in ('kmeans','diskmeans') else run_spectral(F,L,c,seed,degree if method=='ncut' else None)
                        pred=r.pop('labels');r.update(seed=seed,acc=acc(y,pred),nmi=float(normalized_mutual_info_score(y,pred,average_method='geometric')),cluster_sizes=np.bincount(pred,minlength=c).tolist())
                        if not r['valid']:r['objective']=None
                        np.save(folder/f'labels_reg{tag(reg)}_seed{seed}.npy',pred);trials.append(r)
                    valid=[t for t in trials if t['valid']]
                    row.update(trials=trials,selected_by_objective=min(valid,key=lambda t:t['objective']) if valid else None)
                    for metric in ('acc','nmi'):
                        row['mean_'+metric]=float(np.mean([t[metric] for t in trials]));row['std_'+metric]=float(np.std([t[metric] for t in trials]))
                    best=row['selected_by_objective']
                    if best:
                        pred=np.load(folder/f"labels_reg{tag(reg)}_seed{best['seed']}.npy")
                        m=source['dataset_metadata'];write_diagnostics(folder/f'selected_reg{tag(reg)}',y,pred,X,m['image_shape'],m['source_ids'],c)
                except (ValueError,np.linalg.LinAlgError,RuntimeError) as e:
                    row=dict(reg=reg,status='failed',error=str(e),trials=[],selected_by_objective=None)
                row['seconds']=time.perf_counter()-start;report['results'].append(row)
                path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                best=row.get('selected_by_objective');print(a.dataset,method,reg,row['status'],f"ACC={best['acc']:.4f} NMI={best['nmi']:.4f}" if best else row.get('error','no valid run'),f"{row['seconds']:.1f}s",flush=True)
if __name__=='__main__':main()
