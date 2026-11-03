"""Four all-method tables and parameter curves from executed experiment files."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_baselines import GRID
from visualize_clusters import METHODS,NAMES
DATASETS=('coil20','usps','mnist')
DNAMES={'coil20':'COIL-20','usps':'USPS','mnist':'MNIST subset'}

def load_reports(root,dataset):
    root=Path(root)/dataset
    return {m:json.loads(((root if m=='ldmgi' else root/'baselines'/m)/'report.json').read_text()) for m in METHODS}

def selected(rows,metric,mean):
    eligible=[r for r in rows if r.get('status','ok')=='ok' and r.get('selected_by_objective') is not None]
    if not eligible:return None
    eligible.sort(key=lambda r:-1 if r['reg'] is None else r['reg'])
    return max(eligible,key=lambda r:r['mean_'+metric] if mean else r['selected_by_objective'][metric])

def table_export(out,key,title,values,raw):
    headers=['Dataset']+[NAMES[m] for m in METHODS]
    md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']+['---:']*5)+' |\n'
    md+='\n'.join('| '+' | '.join(v)+' |' for v in values)+'\n'
    (out/(key+'.md')).write_text('## '+title+' (%)\n\n'+md)
    with (out/(key+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    tex=[r'% Requires \usepackage{booktabs}',r'\begin{table}[ht]',r'\centering',r'\caption{'+title+' (percent). All methods executed in this repository.}',r'\begin{tabular}{lrrrrr}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    tex+=[' & '.join(v.replace('±',r'$\pm$') for v in row)+r' \\' for row in values]
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(out/(key+'.tex')).write_text('\n'.join(tex)+'\n')
    fig,ax=plt.subplots(figsize=(11.8,2.7));ax.axis('off');ax.set_title(title+' (%)',fontsize=14,fontweight='bold',pad=15)
    tab=ax.table(cellText=values,colLabels=headers,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.8)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        if i==0:cell.set_facecolor('#e2e8f0');cell.set_text_props(weight='bold')
        elif j==5:cell.set_facecolor('#eff6ff')
    fig.text(.5,.025,'LLC = LLC-L (linear kernel). NCut underflowed bandwidths are excluded and explicitly listed in the report.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ('png','svg'):fig.savefig(out/(key+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig);return md

def generate(root='results',document='RESULTS.md'):
    root=Path(root);out=root/'comparison';out.mkdir(exist_ok=True)
    reports={d:load_reports(root,d) for d in DATASETS};records=[];selections={}
    for d in DATASETS:
        for m,r in reports[d].items():
            for row in r['results']:
                b=row.get('selected_by_objective');records.append(dict(dataset=d,method=m,parameter=row['reg'],status=row.get('status','ok'),
                 error=row.get('error',''),starts=len(row.get('trials',[])),valid_starts=sum(t['valid'] for t in row.get('trials',[])),
                 selected_seed=b['seed'] if b else None,selected_acc=b['acc'] if b else None,selected_nmi=b['nmi'] if b else None,
                 mean_acc=row.get('mean_acc'),std_acc=row.get('std_acc'),mean_nmi=row.get('mean_nmi'),std_nmi=row.get('std_nmi'),
                 zero_embedding_rows=row.get('zero_embedding_rows',0),seconds=row.get('seconds')))
    with (out/'all_runs_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# Executed baseline and LDMGI comparison','',
      'Every algorithm column below is computed from runs in this repository on identical samples and features. No published score is substituted for a missing experiment. The earlier published-versus-reproduced LDMGI comparison is preserved in [LDMGI_REFERENCE_RESULTS.md](LDMGI_REFERENCE_RESULTS.md).','',
      '## Protocol','',
      '- K-means: 20 independent sklearn starts, n_init=1 each; minimum inertia selects the representative. No regularization parameter.',
      '- DisKmeans: exact regularized discriminative-kernel feature transform, followed by sklearn K-means; 9 parameters x 20 starts.',
      '- NCut: symmetric union 5-neighbor Gaussian graph, normalized Laplacian, Yu-Shi discretization; sigma is the bandwidth, not a ridge coefficient.',
      '- LLC-L: linear-kernel local ridge prediction, five neighbors excluding the sample; spectral discretization of (I-A).T @ (I-A). This is not LLE and not LLC-G.',
      '- LDMGI: the unchanged implementation and previously executed 9 x 20 starts.',
      '- Best ACC/NMI: select minimum-objective initialization at each usable parameter, then independently select the best parameter for each metric. Mean tables select the best parameter by that metric\'s mean and report its standard deviation (ddof=0). This is oracle benchmark selection with labels, not a deployable unsupervised parameter rule.',
      '- All methods use the same 0-19 seed set. NCut/LLC use five neighbors excluding self; LDMGI uses self plus four neighbors, following the local-set definitions. sklearn k-means++ initialization is an explicit implementation choice.',
      '- The parameter grid is 1e-8 through 1e8 in factors of 100. Underflow/degenerate runs are reported below, not scored as zero or silently fixed. No claim of exact published scores or statistical significance is made.','']
    for key,title in [('best_acc','Table II: Best ACC'),('best_nmi','Table III: Best NMI'),('mean_acc','Table IV: Mean ACC and standard deviation'),('mean_nmi','Table V: Mean NMI and standard deviation')]:
        metric=key.split('_')[1];mean=key.startswith('mean');values=[];raw=[];selections[key]={}
        for d in DATASETS:
            rowvalues=[DNAMES[d]];selections[key][d]={}
            for m in METHODS:
                r=selected(reports[d][m]['results'],metric,mean)
                if r is None:rowvalues.append('N/A');continue
                b=r['selected_by_objective'];value=r['mean_'+metric] if mean else b[metric];std=r['std_'+metric] if mean else None
                rowvalues.append(f'{100*value:.2f} ± {100*std:.2f}' if mean else f'{100*value:.2f}')
                item=dict(dataset=d,method=m,value_percent=100*value,std_percent=100*std if mean else None,selected_parameter=r['reg'],selected_seed=None if mean else b['seed'])
                raw.append(item);selections[key][d][m]=item
            values.append(rowvalues)
        lines+=['## '+title+' (%)','',table_export(out,key,title,values,raw),'']
    (out/'selections.json').write_text(json.dumps(selections,indent=2))
    colors=dict(zip(METHODS,['#64748b','#d97706','#7c3aed','#059669','#dc2626']))
    for mean in (False,True):
        for metric in ('acc','nmi'):
            key=('mean_' if mean else '')+metric+'_vs_parameter';fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
            for ax,d in zip(axes,DATASETS):
                for m in METHODS:
                    rows=reports[d][m]['results'];ys=[];errs=[]
                    for param in GRID:
                        r=next((v for v in rows if v['reg']==(None if m=='kmeans' else param)),None)
                        ok=r and r.get('status','ok')=='ok' and r.get('selected_by_objective') is not None
                        ys.append(100*(r['mean_'+metric] if mean else r['selected_by_objective'][metric]) if ok else np.nan)
                        errs.append(100*r['std_'+metric] if mean and ok else np.nan)
                    ax.plot(GRID,ys,marker=None if m=='kmeans' else 'o',markersize=3,linestyle='--' if m=='kmeans' else '-',color=colors[m],label=NAMES[m],linewidth=1.6)
                    if mean and m!='kmeans':ax.errorbar(GRID,ys,yerr=errs,fmt='none',ecolor=colors[m],alpha=.35,capsize=2)
                ax.set_xscale('log');ax.set_xticks([1e-8,1e-4,1,1e4,1e8]);ax.set_ylim(0,103);ax.grid(alpha=.25);ax.set_title(DNAMES[d]);ax.set_xlabel('Method parameter (log scale)')
            axes[0].set_ylabel(metric.upper()+' (%)');fig.suptitle(('Mean over random starts' if mean else 'Objective-selected initialization')+' - '+metric.upper(),fontsize=13)
            h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.025))
            fig.tight_layout(rect=(0,.13,1,.94))
            for ext in ('png','svg'):fig.savefig(out/(key+'.'+ext),dpi=180,bbox_inches='tight')
            plt.close(fig)
    lines+=['## Parameter curves','',
      'The horizontal parameter denotes lambda (LDMGI), gamma (DisKmeans), eta (LLC-L), or sigma (NCut). Equal numeric values do not imply equivalent regularization. K-means is a dashed horizontal reference, not rerun for each grid point. Gaps mean failed or unavailable parameters; no interpolation fills them.','']
    for key in ('acc_vs_parameter','nmi_vs_parameter','mean_acc_vs_parameter','mean_nmi_vs_parameter'):
        lines+=[f'![{key}](results/comparison/{key}.png)','']
    lines+=['## Numerical status and coverage','',
      '| Dataset | Method | Attempted parameters | Parameters with a valid representative | Saved starts | Valid starts |',
      '|---|---|---:|---:|---:|---:|']
    for d in DATASETS:
        for m in METHODS:
            rows=reports[d][m]['results'];lines.append(f"| {DNAMES[d]} | {NAMES[m]} | {len(rows)} | {sum(r.get('selected_by_objective') is not None for r in rows)} | {sum(len(r.get('trials',[])) for r in rows)} | {sum(t['valid'] for r in rows for t in r.get('trials',[]))} |")
    failures=[r for r in records if r['status']!='ok']
    if failures:
        lines+=['','### Explicitly failed parameter settings','']
        for r in failures:lines.append(f"- {DNAMES[r['dataset']]} / {NAMES[r['method']]} / parameter={r['parameter']:g}: {r['error']}")
    zeros=[r for r in records if r['zero_embedding_rows']]
    if zeros:
        lines+=['','### Zero spectral rows','',
          'Sparse spectral models can produce retained eigenspaces that omit some samples, particularly on disconnected or numerically weakly connected graphs. Rows with norm below 1e-12 are kept at zero rather than normalizing numerical noise. Argmax ties deterministically choose cluster 0. These results are numerically degenerate, retained for transparency, and not evidence of exact reproduction.']
        for r in zeros:lines.append(f"- {DNAMES[r['dataset']]} / {NAMES[r['method']]} / parameter={r['parameter']:g}: {r['zero_embedding_rows']} zero rows.")
    lines+=['','## PCA and t-SNE inspection','',
      'For each dataset, all methods share a label-free PCA projection and a label-free t-SNE projection of the same input features. Colors show existing predicted clusters after global matching; projection coordinates never enter clustering or evaluation. Every image is included.','',
      '- All-method comparisons: numeric parameters 1 and 100, with ground truth plus all five algorithms.',
      '- Additional LDMGI comparisons: lambda=1e-6, 1e-2 and 1e4, with a ground-truth reference.',
      '- t-SNE: PCA-50 preprocessing, perplexity 30, PCA initialization, seed 0, 1000 iterations. Distances between t-SNE groups should not be interpreted as calibrated distances.','']
    for d in DATASETS:lines.append(f"- [{DNAMES[d]}: six PCA/t-SNE figures](results/{d}/visualizations/README.md)")
    lines+=['','## Files and reproducibility','',
      '`results/comparison/` contains four tables in Markdown/CSV/LaTeX/PNG/SVG, four parameter plots in PNG/SVG, the numeric all_runs_summary.csv and selections.json. Baseline per-start labels, diagnostics and reports are under results/DATASET/baselines/METHOD/. Earlier LDMGI outputs remain intact.','',
      'Run `python make_comparison_results.py` to rebuild these tables and curves from saved data. Run `python audit_baselines.py` to independently recompute baseline metrics, check input hashes and verify table selections. Source details and implementation limitations are documented in BASELINES.md.']
    Path(document).write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Saved comparison tables, curves and',document)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results');p.add_argument('--document',default='RESULTS.md');a=p.parse_args();generate(a.root,a.document)
