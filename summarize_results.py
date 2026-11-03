"""Create a cross-dataset index. Never compare objectives across lambdas.
Oracle choices below use true labels for benchmark comparison, not deployment.
"""
import csv,json,argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results');args=ap.parse_args()
    root=Path(args.root); rows=[]; lines=['# Executed experiments','',
      'For the four paper-style tables and regularization figures, see [RESULTS.md](../RESULTS.md). Run `python make_comparison_results.py` for all-method comparisons under `comparison/`. The earlier LDMGI-only comparison can be regenerated with `python make_paper_results.py` under `paper_style/`.','',
      'Each parameter uses 20 random starts unless report.json says otherwise. ACC/NMI are fractions in CSV and percentages below.','',
      'Within each parameter, the saved representative is selected by minimum discrete LDMGI objective, without labels. The best-mean choices below are explicitly label-based (oracle); ACC and NMI may select different parameters.','',
      '| Dataset | n | parameters completed | best mean ACC ± std (%) | lambda | best mean NMI ± std (%) | lambda |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('mnist','coil20','usps'):
        path=root/name/'report.json'
        if not path.exists():continue
        r=json.loads(path.read_text()); data=r['results']
        for item in data:
            best=item['selected_by_objective']
            rows.append({'dataset':name,'n':r['shape'][0],'reg':item['reg'],
                'runs':len(item['trials']),'valid_runs':sum(t['valid'] for t in item['trials']),
                'converged_runs':sum(t['converged'] for t in item['trials']),
                **{k:item[k] for k in ('mean_acc','std_acc','mean_nmi','std_nmi','eigen_residual','seconds')},
                'representative_seed':best['seed'] if best else None,
                'representative_acc':best['acc'] if best else None,
                'representative_nmi':best['nmi'] if best else None})
        a=max(data,key=lambda x:x['mean_acc']);b=max(data,key=lambda x:x['mean_nmi'])
        lines.append(f"| {name} | {r['shape'][0]} | {len(data)} | {100*a['mean_acc']:.2f} ± {100*a['std_acc']:.2f} | {a['reg']:g} | {100*b['mean_nmi']:.2f} ± {100*b['std_nmi']:.2f} | {b['reg']:g} |")
    lines+=['','## Where to inspect individual clusters','',
      'For every dataset and parameter, open `DATASET/selected_regVALUE/clusters.csv` (counts, composition, matched precision/recall) and `cluster_examples.png`. `samples.csv` identifies every sample, true class, cluster, matched class and correctness. `contingency.csv` has true classes in rows and predicted clusters in columns. `clusters.json` also records the matching.','',
      'A red image border means wrong after global Hungarian matching; green means correct. Images are evenly spaced members in deterministic source order, not best examples. The gallery is a preview; the CSV covers every image.','',
      'The representative is not necessarily the maximum-ACC run. Purity uses a separate per-cluster majority and can be higher than matched precision. Empty clusters are retained in diagnostics; unmatched clusters have no matched class.','',
      'All trial labels are saved as NPY, all true labels as true_labels.npy; report.json includes every run’s cluster sizes. Ground-truth labels never enter LDMGI. USPS reports source class IDs 1–10 without assuming a digit-glyph mapping.','',
      'See README.md for differences from the original paper; these are independent reproduction results, not the authors’ published scores.']
    if rows:
        with (root/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
