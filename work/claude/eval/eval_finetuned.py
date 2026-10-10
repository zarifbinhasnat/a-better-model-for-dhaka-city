"""Compare fine-tuned plate detectors against the frozen class-switch hybrid, in the full cascade.
Sets: (1) 12 development frames (provisional agent labels; never trained on, but same session and
stopped vehicles can recur near training times), (2) locked-test frames >=1950 s that have repaired labels.
Matching IoU 0.5, ROI [0,500,3600,2824], rickshaw (bicycle) crops skipped. Reports several confidence
thresholds for transparency; 0.35 is the frozen pilot threshold. Usage: eval_finetuned.py A.pt [B.pt ...]"""
import json, sys, cv2, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot';LAB=ROOT/'work/claude/labeling'
sys.path.insert(0,str(P))
from run_detection import Pipeline, nms
from evaluate import evaluate
from ultralytics import YOLO
ROI=[0,500,3600,2824];THS=(.25,.35,.5)
ann=json.load(open(P/'annotations.json'))
sets={'dev12':[(P/'frames'/f['file'],f['boxes_xyxy'],f['ignore_xyxy']) for f in ann['frames']]}
lt=[]
for f in sorted((LAB/'labels_repaired').glob('frame_*.json')):
    d=json.load(open(f))
    if d['split']=='locked_test' and d['status']=='done':
        lt.append((LAB/'frames'/d['file'],[p['box'] for p in d['plates']],[g['box'] for g in d['ignore']]))
sets['locked_test']=lt
pipe=Pipeline('mps');models={Path(m).stem:YOLO(m) for m in sys.argv[1:]}
def crops(im):
    r=pipe.vehicle.predict(im,imgsz=1280,rect=False,conf=.2,iou=.5,classes=[1,2,3,5,7],device='mps',verbose=False)[0];h,w=im.shape[:2];out=[]
    for b,c,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
        x1,y1,x2,y2=b;px=(x2-x1)*.05;py=(y2-y1)*.05;cls=pipe.vehicle.names[int(k)]
        cb=[max(0,int(x1-px)),max(0,int(y1-py)),min(w,int(x2+px)),min(h,int(y2+py))]
        if cb[2]-cb[0]>=32 and cb[3]-cb[1]>=32:out.append((cb,cls))
    return out
res={}
for sname,frames in sets.items():
    tot={}
    for path,gt,ig in frames:
        im=cv2.imread(str(path));cs=crops(im);preds={k:[] for k in ['hybrid_frozen',*models]};raw={}
        for cb,cls in cs:
            sub=im[cb[1]:cb[3],cb[0]:cb[2]];off=np.array(cb[:2]*2)
            if cls=='bicycle':continue                       # rickshaw: no registration plate
            small=cls=='motorcycle'
            for nm,fn in (('bangla',pipe.bangla_predict),('intl',pipe.international.predict)):
                raw.setdefault(nm,[]).extend(dict(box=(np.array(d['box'])+off).tolist(),confidence=d['confidence'],small=small) for d in fn(sub))
            for mn,m in models.items():
                r=m.predict(sub,imgsz=640,conf=.1,iou=.5,device='mps',verbose=False)[0]
                preds[mn]+=[dict(box=(b[:4]+off).tolist(),confidence=float(b[4])) for b in r.boxes.data.cpu().numpy()]
        # frozen ordering: NMS per plate model -> route by source vehicle class -> NMS again
        preds['hybrid_frozen']=[d for d in nms(raw.get('intl',[])) if not d['small']]+[d for d in nms(raw.get('bangla',[])) if d['small']]
        for mn,ds in preds.items():
            ds=nms(ds)
            for th in THS:
                e=evaluate(ds,gt,ig,ROI,conf=th);t=tot.setdefault((mn,th),dict(tp=0,fp=0,fn=0))
                for k in t:t[k]+=e[k]
    res[sname]={f'{mn}@{th}':dict(**t,P=round(t['tp']/max(1,t['tp']+t['fp']),3),R=round(t['tp']/max(1,t['tp']+t['fn']),3)) for (mn,th),t in tot.items()}
    res[sname]['_frames']=len(frames);res[sname]['_plates']=sum(len(g) for _,g,_ in frames)
json.dump(res,open(Path(__file__).parent/'results.json','w'),indent=1)
for s,r in res.items():
    print(f'== {s}: {r["_frames"]} frames, {r["_plates"]} labelled plates')
    for k,v in r.items():
        if not k.startswith('_'):print(f'  {k:28s} TP {v["tp"]:3d} FP {v["fp"]:3d} FN {v["fn"]:3d}  P {v["P"]:.3f}  R {v["R"]:.3f}')
