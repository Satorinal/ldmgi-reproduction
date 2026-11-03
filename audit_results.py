"""Audit stored predictions and reports without re-running eigenproblems."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from reporting import cluster_summary

def audit(root):
    count=0
    for path in sorted(Path(root).glob('*/report.json')):
        report=json.loads(path.read_text());folder=path.parent
        y=np.load(folder/'true_labels.npy');c=len(np.unique(y))
        assert len(y)==report['shape'][0],path
        for row in report['results']:
            reg=row['reg']
            for trial in row['trials']:
                pred=np.load(folder/f"labels_reg{reg:g}_seed{trial['seed']}.npy")
                d=cluster_summary(y,pred,c)
                assert abs(d['acc']-trial['acc'])<1e-12,(path,reg,trial['seed'],'ACC')
                nmi=normalized_mutual_info_score(y,pred,average_method='geometric')
                assert abs(nmi-trial['nmi'])<1e-12,(path,reg,'NMI')
                assert np.bincount(pred,minlength=c).tolist()==trial['cluster_sizes'],path
                count+=1
            best=row['selected_by_objective']
            if best is None:continue
            assert best['objective']==min(t['objective'] for t in row['trials'] if t['valid'])
            dest=folder/f'selected_reg{reg:g}'
            d=json.loads((dest/'clusters.json').read_text())
            assert abs(d['acc']-best['acc'])<1e-12
            assert sum(x['size'] for x in d['cluster_rows'])==len(y)
            assert sum(x['matched_correct'] for x in d['cluster_rows'])==round(len(y)*best['acc'])
            with (dest/'samples.csv').open() as f: samples=list(csv.DictReader(f))
            assert len(samples)==len(y)
            assert sum(int(s['correct']) for s in samples)==round(len(y)*best['acc'])
            for s,a,b in zip(samples,y,np.load(folder/f"labels_reg{reg:g}_seed{best['seed']}.npy")):
                assert int(s['true_class'])==a and int(s['cluster'])==b
        print(f"PASS {folder.name}: {len(report['results'])} parameters")
    if count==0:raise ValueError('No stored predictions found')
    print(f'PASS: {count} saved trial predictions; metrics, cluster sizes, representative runs and sample diagnostics agree.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results');args=p.parse_args();audit(args.root)
