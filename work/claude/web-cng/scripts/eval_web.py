"""Out-of-domain check of a vehicle classifier + CNG colour rule on Claude's marking of the downloaded photos.
Usage: eval_web.py CLASSIFIER.pt [--bottom FRACTION]  (images-9 GT = proposals classified CNG, all green)"""
import json, sys, cv2, numpy as np
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/claude/pipeline'))
import cascade
from ultralytics import YOLO
C=YOLO(sys.argv[1]);P=YOLO(str(ROOT/'work/claude/models/plate_B_yolo26s_run1.pt'))
gt=json.load(open(HERE/'web_labels_manual.json'));gt.pop('_note')
seen=[]
for o in json.load(open(HERE/'proposals.json')):
    if o['file']=='images-9.jpeg' and o['cls']=='cng':
        if all(cascade.iou(o['box'],s)<.5 for s in seen):seen.append(o['box'])
gt['images-9.jpeg']=[dict(box=b,label='cng',colour='green') for b in seen]
gt=json.load(open(HERE/'web_labels.json')) if (HERE/'web_labels.json').exists() else gt   # keep manual edits (occlusion notes)
json.dump(gt,open(HERE/'web_labels.json','w'),indent=1)
typ=Counter();col=Counter()
for f,objs in gt.items():
    im=cv2.imread(str(HERE/'raw'/f));H,W=im.shape[:2]
    for o in objs:
        x1,y1,x2,y2=o['box'];px,py=(x2-x1)*.05,(y2-y1)*.05;c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
        crop=im[c[1]:c[3],c[0]:c[2]];pr=C.predict(crop,imgsz=224,device='mps',verbose=False)[0].probs;pred=C.names[int(pr.top1)]
        typ[(o['label'],pred)]+=1
        if o['label']=='cng' and o.get('colour') in('green','silver'):
            ds=P.predict(crop,imgsz=640,conf=.25,device='mps',verbose=False)[0].boxes.data.cpu().numpy()
            pc=cascade.cng_colour(crop,ds[ds[:,4].argmax(),:4] if len(ds) else None);col[(o['colour'],pc)]+=1
n=sum(v for (t,p),v in typ.items() if t=='cng');tp=typ[('cng','cng')]
print('type (true,pred):',dict(typ));print(f'CNG recall {tp}/{n}');print('colour (true,pred):',dict(col))
