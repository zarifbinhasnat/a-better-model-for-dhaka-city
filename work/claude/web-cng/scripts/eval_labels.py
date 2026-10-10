"""Classifier + colour check on a web label file. Usage: eval_labels.py CLASSIFIER.pt LABELS.json"""
import json, sys, cv2
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'outputs/claude/pipeline'))
import cascade
from ultralytics import YOLO
C=YOLO(sys.argv[1]);P=YOLO(str(ROOT/'work/claude/models/plate_B_yolo26s_run1.pt'));gt=json.load(open(sys.argv[2]))
typ=Counter();col=Counter()
for f,objs in gt.items():
    if f.startswith('_'):continue
    im=cv2.imread(str(HERE/'raw'/f));H,W=im.shape[:2]
    for o in objs:
        x1,y1,x2,y2=o['box'];px,py=(x2-x1)*.05,(y2-y1)*.05;c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
        crop=im[c[1]:c[3],c[0]:c[2]];pred=C.names[int(C.predict(crop,imgsz=224,device='mps',verbose=False)[0].probs.top1)];typ[(o['label'],pred)]+=1
        if o['label']=='cng' and o.get('colour') in('green','silver'):
            ds=P.predict(crop,imgsz=640,conf=.25,device='mps',verbose=False)[0].boxes.data.cpu().numpy()
            col[(o['colour'],cascade.cng_colour(crop,ds[ds[:,4].argmax(),:4] if len(ds) else None))]+=1
tp=typ[('cng','cng')];fn=sum(v for (t,p),v in typ.items() if t=='cng' and p!='cng');fp=sum(v for (t,p),v in typ.items() if t!='cng' and p=='cng')
print(f'CNG recall {tp}/{tp+fn}  precision {tp}/{tp+fp}');print('errors:',{k:v for k,v in typ.items() if k[0]!=k[1]});print('colour:',dict(sorted(col.items())))
