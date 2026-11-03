"""Recompute baseline metrics and verify comparison table selection."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from reporting import cluster_summary
from run_baselines import tag
from make_comparison_results import DATASETS,load_reports,METHODS

def audit(root):
    root=Path(root);count=0;failures=0
    for d in DATASETS:
        y=np.load(root/d/'true_labels.npy');c=len(np.unique(y));source=json.loads((root/d/'report.json').read_text())
        reports=load_reports(root,d)
        for m in METHODS:
            r=reports[m]
            assert r['sha256']==source['sha256']
            if m=='ldmgi':continue
            folder=root/d/'baselines'/m
            for row in r['results']:
                if row.get('status')=='failed':
                    assert not row['trials'] and row['selected_by_objective'] is None;failures+=1;continue
                trials=row['trials'];param=tag(row['reg'])
                for t in trials:
                    pred=np.load(folder/f"labels_reg{param}_seed{t['seed']}.npy")
                    assert len(pred)==len(y)
                    assert abs(cluster_summary(y,pred,c)['acc']-t['acc'])<1e-12
                    assert abs(normalized_mutual_info_score(y,pred,average_method='geometric')-t['nmi'])<1e-12
                    assert np.bincount(pred,minlength=c).tolist()==t['cluster_sizes'];count+=1
                for metric in ('acc','nmi'):
                    vals=[t[metric] for t in trials]
                    assert abs(np.mean(vals)-row['mean_'+metric])<1e-12
                    assert abs(np.std(vals)-row['std_'+metric])<1e-12
                b=row['selected_by_objective']
                if b:
                    assert b['seed']==min((t for t in trials if t['valid']),key=lambda t:t['objective'])['seed']
                    with (folder/f'selected_reg{param}/samples.csv').open() as f:samples=list(csv.DictReader(f))
                    assert len(samples)==len(y)
                    assert sum(int(s['correct']) for s in samples)==round(b['acc']*len(y))
                    pred=np.load(folder/f"labels_reg{param}_seed{b['seed']}.npy")
                    assert all(int(s['true_class'])==a and int(s['cluster'])==z for s,a,z in zip(samples,y,pred))
        for metric in ('acc','nmi'):
            for mean in (False,True):
                key=('mean_' if mean else 'best_')+metric
                with (root/'comparison'/f'{key}.csv').open() as f:table={row['method']:row for row in csv.DictReader(f) if row['dataset']==d}
                for m,r in reports.items():
                    rows=[row for row in r['results'] if row.get('selected_by_objective') is not None]
                    expected=max(np.mean([t[metric] for t in row['trials']]) if mean else min((t for t in row['trials'] if t['valid']),key=lambda t:t['objective'])[metric] for row in rows)
                    assert abs(float(table[m]['value_percent'])-100*expected)<1e-10
        print('PASS baseline data and tables:',d)
    print(f'PASS {count} baseline prediction files; {failures} explicitly failed parameter settings.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results');a=p.parse_args();audit(a.root)
