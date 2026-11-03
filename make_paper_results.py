"""Generate four paper-style tables and regularization curves from saved runs.
No training, downloading, or changes to predictions are performed.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=('coil20','usps','mnist')
NAMES={'coil20':'COIL-20','usps':'USPS','mnist':'MNIST subset'}
# Published LDMGI columns, Tables II-V, Yang et al., TIP 2010.
# MNIST published values refer to MNIST-T; our exact subset differs.
PUBLISHED={
 'coil20':{'best_acc':82.4,'best_nmi':91.1,'mean_acc':75.3,'std_acc':4.9,'mean_nmi':90.0,'std_nmi':0.7},
 'usps':{'best_acc':81.5,'best_nmi':86.5,'mean_acc':80.5,'std_acc':5.6,'mean_nmi':86.1,'std_nmi':2.3},
 'mnist':{'best_acc':72.8,'best_nmi':69.2,'mean_acc':71.5,'std_acc':3.5,'mean_nmi':68.9,'std_nmi':1.5}}

def select_result(rows,metric,mean=False):
    """Ties: smallest lambda; representative seeds already chosen by objective."""
    rows=sorted(rows,key=lambda r:r['reg'])
    eligible=rows if mean else [r for r in rows if r['selected_by_objective'] is not None]
    if not eligible:raise ValueError('No valid representative runs')
    return max(eligible,key=lambda r:r['mean_'+metric] if mean else r['selected_by_objective'][metric])

def save_table(folder,key,title,headers,values,records):
    md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |\n'
    md+='\n'.join('| '+' | '.join(row)+' |' for row in values)+'\n'
    (folder/(key+'.md')).write_text('## '+title+'\n\n'+md,encoding='utf-8')
    with (folder/(key+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    tex=['% Requires \\usepackage{booktabs}','\\begin{table}[ht]','\\centering',
         '\\caption{'+title+'. Published and reproduced LDMGI results; percentages.}',
         '\\begin{tabular}{l'+'r'*(len(headers)-1)+'}','\\toprule']
    for row in [headers]+values:
        tex.append(' & '.join(s.replace('±',r'$\pm$').replace('%',r'\%') for s in row)+r' \\')
        if row is headers:tex.append(r'\midrule')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (folder/(key+'.tex')).write_text('\n'.join(tex)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,2.65));ax.axis('off')
    ax.set_title(title+' (%)',fontsize=14,fontweight='bold',pad=18)
    t=ax.table(cellText=values,colLabels=headers,cellLoc='center',loc='center')
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.8)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        if i==0:cell.set_facecolor('#e2e8f0');cell.set_text_props(weight='bold')
        elif j==2:cell.set_facecolor('#eff6ff')
    fig.text(.5,.025,'Published MNIST values: MNIST-T. Reproduction: a different balanced subset. No significance test is implied.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1))
    for ext in ('png','svg'):fig.savefig(folder/(key+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    return md

def generate(root,document):
    root=Path(root);folder=root/'paper_style';folder.mkdir(parents=True,exist_ok=True)
    reports={name:json.loads((root/name/'report.json').read_text()) for name in ORDER}
    intro=['# LDMGI reproduction results','',
      'These are independently reproduced experiments on three datasets. The four tables follow the reporting structure of Tables II-V in Yang et al. (2010), restricted to LDMGI. Published baseline methods have not been rerun and are not presented as reproduced results.','',
      '## Selection rules','',
      '- Best ACC/NMI: at each regularization value, use the valid run with the smallest discrete LDMGI objective; then select the parameter with the highest ACC or NMI, respectively. This is not the maximum metric over all individual seeds.',
      '- Mean ACC/NMI: compute the mean over all saved starts at each parameter, then select the parameter with the highest mean of that metric. Report its standard deviation (ddof=0).',
      '- Parameter selection by ACC/NMI uses ground truth for oracle benchmark reporting. It is not a label-free deployment rule. ACC and NMI are selected independently; ties use the smallest lambda.',
      '- Do not compare raw objective values between regularization parameters. Published values are transcribed from the LDMGI columns of the original paper, not computed by this repository.',
      '- The MNIST subset indices are not verified as the authors\' MNIST-T subset. Image scaling and COIL interpolation are explicit implementation choices. Higher reproduced scores do not establish an algorithmic improvement or an exact replication.','']
    lines=intro[:];selection={}
    for key,title in [('best_acc','Table II: Best ACC'),('best_nmi','Table III: Best NMI'),('mean_acc','Table IV: Mean ACC and standard deviation'),('mean_nmi','Table V: Mean NMI and standard deviation')]:
        mean=key.startswith('mean');metric=key.split('_')[1];values=[];records=[];selection[key]={}
        headers=['Dataset','Published LDMGI','Reproduced LDMGI','Selected lambda','Starts' if mean else 'Selected seed']
        for name,r in reports.items():
            chosen=select_result(r['results'],metric,mean)
            pub=PUBLISHED[name];rep=chosen['selected_by_objective']
            value=chosen['mean_'+metric] if mean else rep[metric]
            std=chosen['std_'+metric] if mean else None
            values.append([NAMES[name],f"{pub[key]:.1f} ± {pub['std_'+metric]:.1f}" if mean else f'{pub[key]:.1f}',
                           f'{100*value:.2f} ± {100*std:.2f}' if mean else f'{100*value:.2f}',
                           f"{chosen['reg']:g}",str(len(chosen['trials']) if mean else rep['seed'])])
            rec={'dataset':name,'published_percent':pub[key],'reproduced_percent':100*value,
                 'selected_reg':chosen['reg'],'starts':len(chosen['trials']),
                 'valid_starts':sum(t['valid'] for t in chosen['trials'])}
            if mean:rec.update(published_std_percent=pub['std_'+metric],reproduced_std_percent=100*std)
            else:rec['selected_seed']=rep['seed']
            records.append(rec);selection[key][name]=rec
        md=save_table(folder,key,title,headers,values,records)
        lines+=['## '+title+' (%)','',md,'']
    (folder/'selections.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    curve_records=[]
    for name,r in reports.items():
        for row in sorted(r['results'],key=lambda a:a['reg']):
            b=row['selected_by_objective']
            curve_records.append({'dataset':name,'reg':row['reg'],'selected_seed':b['seed'] if b else None,
              **{'selected_'+m+'_percent':100*b[m] if b else None for m in ('acc','nmi')},
              **{k+'_percent':100*row[k] for k in ('mean_acc','std_acc','mean_nmi','std_nmi')}})
    with (folder/'regularization_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curve_records[0]));w.writeheader();w.writerows(curve_records)
    for mean in (False,True):
        for metric in ('acc','nmi'):
            key=('mean_' if mean else '')+metric+'_vs_regularization'
            fig,axes=plt.subplots(1,3,figsize=(12.6,3.8),sharey=True)
            for ax,(name,r) in zip(axes,reports.items()):
                rows=sorted(r['results'],key=lambda a:a['reg']);x=np.array([a['reg'] for a in rows])
                y=np.array([100*a['mean_'+metric] if mean else (100*a['selected_by_objective'][metric] if a['selected_by_objective'] else np.nan) for a in rows])
                if mean:
                    err=np.array([100*a['std_'+metric] for a in rows]);ax.errorbar(x,y,yerr=err,fmt='o-',color='#2563eb',capsize=3,label='Mean ± 1 SD')
                else:ax.plot(x,y,'o-',color='#2563eb',label='Objective-selected run')
                best=select_result(rows,metric,mean);j=next(i for i,a in enumerate(rows) if a['reg']==best['reg'])
                ax.scatter(x[j],y[j],s=130,marker='*',color='#dc2626',zorder=5,label='Best parameter (oracle)')
                ax.set_xscale('log');ax.set_xticks([1e-8,1e-4,1,1e4,1e8]);ax.set_ylim(0,102)
                ax.set_title(NAMES[name]);ax.set_xlabel(r'Regularization $\lambda$');ax.grid(alpha=.25)
            axes[0].set_ylabel(metric.upper()+' (%)')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
            fig.suptitle(('Mean performance over random starts' if mean else 'Performance of the objective-selected initialization')+' - '+metric.upper(),fontsize=13)
            fig.tight_layout(rect=(0,.13,1,.94))
            for ext in ('png','svg'):fig.savefig(folder/(key+'.'+ext),dpi=200,bbox_inches='tight')
            plt.close(fig)
    prefix=Path(__import__('os').path.relpath(root,Path(document).parent)).as_posix()
    lines+=['## Regularization curves','',
      'The two primary figures follow the paper\'s objective-selected-initialization convention. Each panel shows one dataset. The horizontal axis is logarithmic; stars mark label-based best parameters, not automatically deployable settings. No curves have been invented for unimplemented baseline algorithms.','',
      f'![ACC versus regularization]({prefix}/paper_style/acc_vs_regularization.png)','',
      f'![NMI versus regularization]({prefix}/paper_style/nmi_vs_regularization.png)','',
      'Supplementary mean curves show the mean and one standard deviation at every parameter. These error bars describe variation across initializations, not confidence intervals.','',
      f'![Mean ACC versus regularization]({prefix}/paper_style/mean_acc_vs_regularization.png)','',
      f'![Mean NMI versus regularization]({prefix}/paper_style/mean_nmi_vs_regularization.png)','',
      'All four tables are exported as Markdown, CSV, LaTeX (booktabs), PNG and SVG. Figures are exported as PNG and SVG, with their numeric data in `regularization_curves.csv`. `selections.json` records the exact parameters and seeds used in the tables.','',
      '## Cluster diagnostics at fixed lambda = 1','',
      'The following representative runs use the smallest discrete objective among starts at lambda=1. Cluster IDs have no intrinsic class meaning. Correctness uses global one-to-one Hungarian matching.']
    for name,r in reports.items():
        row=next((a for a in r['results'] if a['reg']==1),None)
        if row is None or row['selected_by_objective'] is None:continue
        b=row['selected_by_objective'];d=json.loads((root/name/'selected_reg1/clusters.json').read_text())
        lines+=['','### '+NAMES[name],'',f"Samples: {r['shape'][0]}; features: {r['shape'][1]}; seed: {b['seed']}; ACC: {100*b['acc']:.2f}%; NMI: {100*b['nmi']:.2f}%.",'',
          '| Cluster | Size | Matched class | Class total | Correct | Wrong | Precision (%) | Recall (%) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in d['cluster_rows']:
            lines.append(f"| {a['cluster']} | {a['size']} | {a['matched_class']} | {a['matched_class_total']} | {a['matched_correct']} | {a['matched_wrong']} | {(a['matched_precision'] or 0)*100:.1f} | {(a['matched_recall'] or 0)*100:.1f} |")
        lines+=['',f'[Image gallery]({prefix}/{name}/selected_reg1/cluster_examples.png) · [Cluster statistics]({prefix}/{name}/selected_reg1/clusters.csv) · [Per-image predictions]({prefix}/{name}/selected_reg1/samples.csv)']
    total=sum(len(row['trials']) for r in reports.values() for row in r['results'])
    lines+=['','## Validation and provenance','',f'The saved experiments contain {total} LDMGI starts. Run `python audit_results.py` to recompute metrics from saved predictions and verify cluster/sample reports. Run `python -m unittest -v` for numerical, reporting and table-selection tests.','',
      'USPS uses source class IDs 1-10; COIL uses object IDs 1-20. A pure cluster can have zero matched precision if its majority class is assigned to another cluster by the global one-to-one mapping. Purity and matched precision are intentionally separate.','',
      'Source: Yang et al., Image Clustering Using Local Discriminant Models and Global Integration, IEEE TIP 19(10), 2010, Tables II-V, DOI: 10.1109/TIP.2010.2049235. The published MNIST row is MNIST-T. Data hashes, software versions and all run-level scores are preserved in each dataset report.json.']
    Path(document).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved four tables, four figures and',document)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results');p.add_argument('--document',default='LDMGI_REFERENCE_RESULTS.md');a=p.parse_args();generate(a.root,a.document)
