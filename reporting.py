"""Post-hoc diagnostics only: no ground-truth feedback to the algorithm."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

def cluster_summary(y,pred,n_clusters=None):
    y=np.asarray(y); pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or len(y)==0: raise ValueError('Nonempty equal-length labels required')
    classes,yi=np.unique(y,return_inverse=True)
    clusters=np.arange(n_clusters) if n_clusters is not None else np.unique(pred)
    if not np.all(np.isin(pred,clusters)): raise ValueError('Unknown predicted cluster')
    pi=np.searchsorted(clusters,pred)
    C=np.zeros((len(classes),len(clusters)),dtype=int); np.add.at(C,(yi,pi),1)
    r,c=linear_sum_assignment(-C)
    mapping={int(clusters[j]):int(classes[i]) for i,j in zip(r,c)}
    rows=[]
    for j,cluster in enumerate(clusters):
        counts=C[:,j]; size=int(counts.sum()); matched=mapping.get(int(cluster))
        correct=int(counts[np.where(classes==matched)[0][0]]) if matched is not None else 0
        majority=int(classes[np.argmax(counts)]) if size else None
        total=int(np.sum(y==matched)) if matched is not None else 0
        rows.append({'cluster':int(cluster),'size':size,'matched_class':matched,
                     'matched_class_total':total,'matched_correct':correct,'matched_wrong':size-correct,
                     'matched_class_elsewhere':total-correct,
                     'matched_precision':correct/size if size else None,
                     'matched_recall':correct/total if total else None,
                     'majority_class':majority,'purity':float(counts.max()/size) if size else None,
                     'composition':{str(int(a)):int(v) for a,v in zip(classes,counts)}})
    return {'acc':int(C[r,c].sum())/len(y),'classes':classes.tolist(),
            'clusters':clusters.tolist(),'contingency_true_by_cluster':C.tolist(),
            'matching':mapping,'cluster_rows':rows}

def write_diagnostics(folder,y,pred,X,image_shape,source_ids,n_clusters):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    d=cluster_summary(y,pred,n_clusters)
    (folder/'clusters.json').write_text(json.dumps(d,indent=2))
    fields=[k for k in d['cluster_rows'][0] if k!='composition']
    with (folder/'clusters.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+['true_'+str(a) for a in d['classes']]);w.writeheader()
        for row in d['cluster_rows']:
            w.writerow({**{k:row[k] for k in fields},**{'true_'+k:v for k,v in row['composition'].items()}})
    with (folder/'contingency.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['true_class']+['cluster_'+str(j) for j in d['clusters']])
        w.writerows([[a]+r for a,r in zip(d['classes'],d['contingency_true_by_cluster'])])
    with (folder/'samples.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row','source_id','true_class','cluster','matched_class','correct'])
        for i,(a,b) in enumerate(zip(y,pred)):
            matched=d['matching'].get(int(b));w.writerow([i,source_ids[i],int(a),int(b),matched,int(a==matched)])
    # Deterministic representative sample, not cherry-picked: evenly spaced members.
    canvas=Image.new('RGB',(1000,100*n_clusters),'white');draw=ImageDraw.Draw(canvas)
    for row in d['cluster_rows']:
        j=row['cluster']; top=100*j
        draw.text((8,top+4),f"cluster {j} | n={row['size']} | matched class={row['matched_class']} | correct={row['matched_correct']} / {row['size']}",fill='black')
        members=np.flatnonzero(pred==j)
        chosen=members[np.linspace(0,len(members)-1,min(12,len(members)),dtype=int)] if len(members) else []
        for col,i in enumerate(chosen):
            im=Image.fromarray(np.uint8(np.clip(X[i].reshape(image_shape),0,1)*255)).resize((52,52))
            x=8+col*82; canvas.paste(im,(x,top+23))
            ok=int(y[i])==row['matched_class']
            draw.rectangle((x-1,top+22,x+52,top+75),outline='green' if ok else 'red',width=2)
            draw.text((x,top+79),f'y={y[i]} i={i}',fill='black')
    canvas.save(folder/'cluster_examples.png')
    return str(folder)
