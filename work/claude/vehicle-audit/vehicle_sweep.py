"""Re-run the COCO vehicle detector on the 12 dev frames at lower confidence / larger input.
Question: do missed motorcycles appear as proposals without fine-tuning? Writes only to this folder."""
import json, sys
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot'
ann=json.load(open(P/'annotations.json'));cov=json.load(open('coverage.json'));tags=json.load(open('tags.json'))
moto={k for k in tags['motorcycle']}
m=YOLO(str(ROOT/'work/detection-pilot/yolo26s.pt'))
def inside(p,c):
    iw=max(0,min(p[2],c[2])-max(p[0],c[0]));ih=max(0,min(p[3],c[3])-max(p[1],c[1]));return iw*ih/((p[2]-p[0])*(p[3]-p[1]))
res={}
for imgsz in (1280,1920,2560):
    out={}
    for f in ann['frames']:
        r=m.predict(str(P/'frames'/f['file']),imgsz=imgsz,conf=.05,iou=.5,classes=[1,2,3,5,7],device='mps',verbose=False)[0]
        out[f['file']]=[dict(box=b.tolist(),conf=float(c),cls=m.names[int(k)]) for b,c,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy())]
    res[imgsz]=out
    for th in (.2,.1,.05):
        allc=0;motoc=0;mc_any=0
        for k,o in enumerate(cov):
            vs=[v for v in out[o['file']] if v['conf']>=th]
            def pad(b):w,h=b[2]-b[0],b[3]-b[1];return[b[0]-.05*w,b[1]-.05*h,b[2]+.05*w,b[3]+.05*h]
            c=[v for v in vs if inside(o['box'],pad(v['box']))>=.99]
            allc+=bool(c)
            if k in moto: motoc+=any(v['cls']=='motorcycle' for v in c); mc_any+=bool(c)
        n=sum(len([v for v in out[f] if v['conf']>=th]) for f in out)
        print(f'imgsz {imgsz} conf>={th}: plates covered {allc}/102; motorcycle plates inside a MOTORCYCLE box {motoc}/25 (any box {mc_any}/25); proposals {n}')
json.dump(res,open('vehicle_sweep.json','w'))
