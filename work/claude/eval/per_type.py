"""Per-vehicle-type recall of a plate model in the cascade on dev12, using Claude's visual type tags
from work/claude/vehicle-audit/tags.json (plate order = annotations.json order). Conf 0.35, IoU 0.5."""
import json, sys, cv2, numpy as np
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot';sys.path.insert(0,str(P))
from run_detection import Pipeline, nms, iou
from ultralytics import YOLO
ann=json.load(open(P/'annotations.json'));tags=json.load(open(ROOT/'work/claude/vehicle-audit/tags.json'))
def typ(k):
    for n in('cng','motorcycle','microbus_pickup_truck'):
        if k in tags[n]:return n
    return 'car'
pipe=Pipeline('mps');m=YOLO(sys.argv[1]);k=0;S=defaultdict(lambda:[0,0]);fps=[]
for f in ann['frames']:
    im=cv2.imread(str(P/'frames'/f['file']));h,w=im.shape[:2];ds=[]
    r=pipe.vehicle.predict(im,imgsz=1280,rect=False,conf=.2,iou=.5,classes=[1,2,3,5,7],device='mps',verbose=False)[0]
    for b,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy()):
        if pipe.vehicle.names[int(c)]=='bicycle':continue
        x1,y1,x2,y2=b;px=(x2-x1)*.05;py=(y2-y1)*.05;cb=[max(0,int(x1-px)),max(0,int(y1-py)),min(w,int(x2+px)),min(h,int(y2+py))]
        if cb[2]-cb[0]<32 or cb[3]-cb[1]<32:continue
        for d in m.predict(im[cb[1]:cb[3],cb[0]:cb[2]],imgsz=640,conf=.35,device='mps',verbose=False)[0].boxes.data.cpu().numpy():
            ds.append(dict(box=(d[:4]+np.array(cb[:2]*2)).tolist(),confidence=float(d[4])))
    ds=nms(ds)
    for g in f['boxes_xyxy']:
        t=typ(k);S[t][0]+=1;S[t][1]+=any(iou(d['box'],g)>=.5 for d in ds);k+=1
for t,(n,hit) in S.items():print(f'{t:24s} {hit}/{n}')
