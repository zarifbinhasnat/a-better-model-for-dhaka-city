"""Confusion matrix and CNG precision/recall of the crop classifier on val (1700-1800 s) and locked test (>=1950 s)."""
import sys
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
m=YOLO(sys.argv[1]);names=m.names
for split in (sys.argv[3].split(',') if len(sys.argv)>3 else ('val','test')):
    conf=Counter();n=0
    for img in sorted(Path(sys.argv[2] if len(sys.argv)>2 else 'dataset',split).glob('*/*.jpg')):
        true=img.parent.name;p=m.predict(str(img),imgsz=224,device='mps',verbose=False)[0];pred=names[int(p.probs.top1)]
        conf[(true,pred)]+=1;n+=1
    acc=sum(v for (t,p),v in conf.items() if t==p)/n
    tp=conf[('cng','cng')];fn=sum(v for (t,p),v in conf.items() if t=='cng' and p!='cng');fp=sum(v for (t,p),v in conf.items() if t!='cng' and p=='cng')
    print(f'== {split}: n={n} accuracy {acc:.3f} | CNG precision {tp/max(1,tp+fp):.3f} recall {tp/max(1,tp+fn):.3f} (TP {tp} FP {fp} FN {fn})')
    for (t,p),v in sorted(conf.items()):
        if t!=p:print(f'   {t} -> {p}: {v}')
