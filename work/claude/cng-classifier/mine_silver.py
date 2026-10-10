"""Mine silver (private) CNG candidates: non-green vehicle proposals (green<0.03) that the classifier
gives P(cng)>=0.05, across the 128 labelling frames and the 66 pilot frames. Claude verifies visually."""
import json, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from green import green_frac
ROOT=Path(__file__).resolve().parents[3];LAB=ROOT/'work/claude/labeling'
m=YOLO(str(ROOT/'work/claude/models/vehicle_cls_run4b.pt'));ci=[k for k,v in m.names.items() if v=='cng'][0]
veh=YOLO(str(ROOT/'work/detection-pilot/yolo26s.pt'))
pilot=json.loads(open(ROOT/'outputs/detection-pilot/predictions.jsonl').readline()) and [json.loads(l) for l in open(ROOT/'outputs/detection-pilot/predictions.jsonl')]
jobs=[(LAB/'frames'/f'{p.stem}.jpg',[(v['box'],v['coco_class'],v['confidence']) for v in json.load(open(p))['vehicles']],'labelling') for p in sorted((LAB/'suggestions').glob('*.json'))]
jobs+=[(ROOT/'outputs/detection-pilot/frames'/r['file'],[(v['box'],v['class'],v['confidence']) for v in r['vehicles']],'pilot') for r in pilot]
cands=[];tiles=[]
for path,vs,src in jobs:
    im=cv2.imread(str(path));H,W=im.shape[:2]
    for vi,(b,cls,conf) in enumerate(vs):
        if cls in('motorcycle','bicycle') or conf<.25:continue
        x1,y1,x2,y2=b;px,py=(x2-x1)*.05,(y2-y1)*.05;c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
        if c[2]-c[0]<32 or c[3]-c[1]<32:continue
        crop=im[c[1]:c[3],c[0]:c[2]]
        if green_frac(crop)>=.03:continue
        p=float(m.predict(crop,imgsz=224,device='mps',verbose=False)[0].probs.data[ci])
        if p<.05:continue
        cands.append(dict(frame=path.name,frame_dir=src,vi=vi,box=b,coco=cls,p_cng=round(p,3)))
        s=150/max(crop.shape[:2]);t=cv2.resize(crop,(int(crop.shape[1]*s),int(crop.shape[0]*s)));tile=np.zeros((168,150,3),np.uint8);tile[18:18+t.shape[0],:t.shape[1]]=t
        cv2.putText(tile,f"{len(cands)-1} {src[0]} p{p:.2f}",(2,13),0,.4,(255,255,255),1);tiles.append(tile)
json.dump(cands,open('silver_candidates.json','w'))
tiles+=[np.zeros((168,150,3),np.uint8)]*((-len(tiles))%10)
for s0 in range(0,len(tiles),60):
    ch=tiles[s0:s0+60];cv2.imwrite(f'silver_{s0//60}.jpg',np.vstack([np.hstack(ch[i:i+10]) for i in range(0,len(ch),10)]))
print(len(cands),'candidates')
