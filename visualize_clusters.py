"""Shared PCA/t-SNE coordinates, colored by already-computed cluster labels.
Projections are for inspection only, never inputs to clustering or evaluation.
"""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from run_baselines import load_saved_dataset,tag
from reporting import cluster_summary

METHODS=('kmeans','diskmeans','ncut','llc_l','ldmgi')
NAMES={'kmeans':'K-means','diskmeans':'DisKmeans','ncut':'NCut','llc_l':'LLC-L','ldmgi':'LDMGI'}
FIXED=(1.,100.)
EXTRA=(1e-6,1e-2,1e4)

def prediction(root,dataset,method,reg):
    folder=Path(root)/dataset
    if method!='ldmgi':folder=folder/'baselines'/method
    r=json.loads((folder/'report.json').read_text())
    param=None if method=='kmeans' else reg
    row=next(a for a in r['results'] if a['reg']==param)
    b=row['selected_by_objective']
    if b is None:return None,{'status':row.get('status','no_valid_run'),'error':row.get('error','No valid representative')}
    pred=np.load(folder/f"labels_reg{tag(param)}_seed{b['seed']}.npy")
    return pred,dict(seed=b['seed'],reg=param,acc=b['acc'],nmi=b['nmi'],status='ok',zero_embedding_rows=row.get('zero_embedding_rows',0))

def matched_colors(y,pred):
    d=cluster_summary(y,pred);classes=np.unique(y);lookup={int(a):i for i,a in enumerate(classes)}
    return np.array([lookup.get(d['matching'].get(int(a)),-1) for a in pred])

def plot_panels(path,coords,y,panels,title,method):
    c=len(np.unique(y));palette=plt.get_cmap('tab20',c)
    rows=2 if len(panels)>4 else 1;cols=3 if rows==2 else len(panels)
    fig,axes=plt.subplots(rows,cols,figsize=(4.0*cols,3.55*rows),squeeze=False,sharex=True,sharey=True)
    lo=coords.min(0);hi=coords.max(0);pad=(hi-lo)*.04
    for ax,(label,pred,meta) in zip(axes.ravel(),panels):
        if pred is None:
            ax.text(.5,.5,'Unavailable\n'+meta.get('error',''),ha='center',va='center',transform=ax.transAxes,wrap=True,fontsize=8)
        else:
            colors=np.searchsorted(np.unique(y),y) if label=='Ground truth' else matched_colors(y,pred)
            for j in range(c):
                idx=colors==j;ax.scatter(coords[idx,0],coords[idx,1],s=3,alpha=.6,color=palette(j),linewidths=0,rasterized=True)
            if np.any(colors<0):ax.scatter(coords[colors<0,0],coords[colors<0,1],s=3,c='gray',rasterized=True)
        suffix='' if label=='Ground truth' or pred is None else f"\nACC {100*meta['acc']:.1f}% | NMI {100*meta['nmi']:.1f}% | seed {meta['seed']}"
        if meta.get('zero_embedding_rows',0):suffix+=f"\nZero spectral rows: {meta['zero_embedding_rows']}"
        ax.set_title(label+suffix,fontsize=10);ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_xticks([]);ax.set_yticks([])
        ax.set_xlabel(method+' dimension 1');ax.set_ylabel(method+' dimension 2')
    handles=[plt.Line2D([],[],marker='o',linestyle='',color=palette(i),markersize=5,label=str(a)) for i,a in enumerate(np.unique(y))]
    fig.legend(handles=handles,loc='lower center',ncol=min(c,20),title='True class / globally matched predicted class',fontsize=8,title_fontsize=9)
    fig.suptitle(title,fontsize=13)
    fig.tight_layout(rect=(0,.12 if rows==1 else .09,1,.92 if rows==1 else .94))
    for ext in ('png','svg'):fig.savefig(str(path)+'.'+ext,dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mnist','coil20','usps'],required=True)
    p.add_argument('--root',default='results');p.add_argument('--data-dir',default='data');p.add_argument('--coordinates-only',action='store_true')
    a=p.parse_args();out=Path(a.root)/a.dataset/'visualizations';out.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(limits=1):
        X,y,source=load_saved_dataset(a.dataset,a.root,a.data_dir)
        feature_hash=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
        if not (out/'coordinates.npz').exists():
            pca=PCA(n_components=50,svd_solver='full');P=pca.fit_transform(X)
            tsne=TSNE(n_components=2,perplexity=30,learning_rate='auto',init='pca',max_iter=1000,random_state=0,method='barnes_hut',angle=.5,n_jobs=1)
            T=tsne.fit_transform(P)
            np.savez_compressed(out/'coordinates.npz',pca=P[:,:2],tsne=T)
            meta=dict(dataset=a.dataset,n_samples=len(y),sample_order='identical to saved LDMGI indices',
                sha256=source['sha256'],input_feature_sha256=feature_hash,pca_components_for_tsne=50,pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                tsne=dict(perplexity=30,seed=0,initialization='pca',learning_rate='auto',max_iter=1000,method='barnes_hut',angle=.5,kl_divergence=float(tsne.kl_divergence_)),
                comparison_parameters=list(FIXED),ldmgi_extra_parameters=list(EXTRA),coordinates_shared_across_all_methods=True,
                note='Labels are used only for colors, never for projection fitting. Embeddings show input-feature geometry, not a new clustering fit.')
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            with (out/'coordinates.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['row','source_id','true_class','pca_1','pca_2','tsne_1','tsne_2'])
                for i in range(len(y)):w.writerow([i,source['dataset_metadata']['source_ids'][i],int(y[i]),*P[i,:2],*T[i]])
        else:
            meta=json.loads((out/'metadata.json').read_text())
            if meta['sha256']!=source['sha256'] or meta['n_samples']!=len(y) or meta.get('input_feature_sha256')!=feature_hash:raise ValueError('Projection cache does not match current input')
        if a.coordinates_only:print('Saved coordinates',a.dataset,flush=True);return
        coords=np.load(out/'coordinates.npz');manifest={}
        for reg in FIXED:
            panels=[('Ground truth',y,{})]
            for m in METHODS:
                pred,info=prediction(a.root,a.dataset,m,reg);panels.append((NAMES[m],pred,info))
                manifest[f'{m}:{reg:g}']=info
            for projection in ('pca','tsne'):
                title=f'{a.dataset.upper()} | shared {projection.upper()} | parameter = {reg:g} (K-means: no parameter)'
                plot_panels(out/f'{projection}_all_methods_param{reg:g}',coords[projection],y,panels,title,projection.upper())
        panels=[('Ground truth',y,{})]
        for reg in EXTRA:
            pred,info=prediction(a.root,a.dataset,'ldmgi',reg);panels.append((f'LDMGI lambda={reg:g}',pred,info));manifest[f'ldmgi:{reg:g}']=info
        for projection in ('pca','tsne'):
            plot_panels(out/f'{projection}_ldmgi_extra',coords[projection],y,panels,f'{a.dataset.upper()} | shared {projection.upper()} | LDMGI parameter comparison',projection.upper())
        (out/'plotted_runs.json').write_text(json.dumps(manifest,indent=2))
        lines=['# Shared-coordinate cluster visualizations','',
          'PCA and t-SNE are fitted once on the dataset features without labels. Every method and parameter uses the same coordinates and axis limits. Only colors change. All samples are shown.','',
          'The two main comparisons fix the numeric parameter at 1 and 100; this means lambda for LDMGI, gamma for DisKmeans, eta for LLC-L, and bandwidth sigma for NCut. These parameters have different meanings. K-means has no corresponding parameter and is repeated unchanged.','',
          'The additional LDMGI comparisons use lambda=1e-6, 1e-2 and 1e4. Runs are selected by the method objective, not by the projection or plotting quality. Colors use global Hungarian matching for visual comparison to true classes.','',
          't-SNE uses PCA-50 input, perplexity 30, PCA initialization, seed 0 and 1000 iterations. Distances between t-SNE groups and their sizes should not be read as quantitative evidence of clustering quality. Neither projection changes clustering or ACC/NMI.']
        for projection in ('pca','tsne'):
            for reg in FIXED:lines+=['',f'![{projection} all methods parameter {reg:g}]({projection}_all_methods_param{reg:g}.png)']
            lines+=['',f'![{projection} LDMGI extra parameters]({projection}_ldmgi_extra.png)']
        (out/'README.md').write_text('\n'.join(lines)+'\n');print('Saved six comparison figures',a.dataset,flush=True)
if __name__=='__main__':main()
