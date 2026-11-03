"""Download paper datasets; explicitly record implementation choices.
Labels are returned separately and never passed to LDMGI.
"""
import hashlib, io, re, urllib.request, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file

URLS = {
 'coil-20-proc.zip': 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip',
 'usps.bz2': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2',
 'usps.t.bz2': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2',
}

def download(folder, name):
    p=Path(folder)/name; p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        tmp=p.with_suffix(p.suffix+'.part')
        urllib.request.urlretrieve(URLS[name],tmp); tmp.replace(p)
    return p

def load_extra(name, folder):
    hashes={}
    def fetch(filename):
        p=download(folder,filename)
        hashes[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    if name=='coil20':
        with zipfile.ZipFile(fetch('coil-20-proc.zip')) as z:
            entries=[]
            for path in z.namelist():
                m=re.search(r'obj(\d+)__(\d+)\.png$',path)
                if m: entries.append((int(m[1]),int(m[2]),path))
            entries.sort()
            if len(entries)!=1440 or len({(a,b) for a,b,_ in entries})!=1440:
                raise ValueError('COIL-20 must contain 1440 distinct views')
            X=[]; y=[]; ids=[]
            for obj,view,path in entries:
                im=Image.open(io.BytesIO(z.read(path))).convert('L')
                im=im.resize((32,32),Image.Resampling.BILINEAR)
                X.append(np.asarray(im,dtype=float).ravel()/255.)
                y.append(obj); ids.append(path)
        if any(y.count(obj)!=72 for obj in range(1,21)):
            raise ValueError('COIL-20 must have 72 images per object')
        shape=(32,32)
        preprocessing='Processed COIL-20; grayscale; Pillow bilinear resize to 32x32; divide by 255. Interpolation/scaling are explicit reproduction choices.'
    elif name=='usps':
        xs=[]; ys=[]; ids=[]
        for filename,expected in [('usps.bz2',7291),('usps.t.bz2',2007)]:
            a,b=load_svmlight_file(str(fetch(filename)),n_features=256)
            if a.shape!=(expected,256): raise ValueError('Unexpected USPS split shape')
            xs.append(a.toarray()); ys.append(b.astype(int))
            ids.extend(f'{filename}:row{i}' for i in range(expected))
        X=(np.vstack(xs)+1)/2; y=np.concatenate(ys); shape=(16,16)
        if X.min() < -1e-6 or X.max()>1+1e-6: raise ValueError('USPS range is not [-1,1]')
        preprocessing='LIBSVM USPS train7291 + test2007; source [-1,1] mapped to [0,1]; preserve source class IDs 1..10 (do not interpret IDs as digit glyphs).'
    else: raise ValueError(name)
    X=np.asarray(X); y=np.asarray(y)
    if len(np.unique(y)) != (20 if name=='coil20' else 10): raise ValueError('Wrong number of classes')
    return X,y,np.arange(len(y)),hashes,{'image_shape':shape,'source_ids':ids,'preprocessing':preprocessing,'urls':{k:URLS[k] for k in hashes}}
