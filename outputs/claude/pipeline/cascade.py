"""Current best detection cascade (2026-10-10, development-stage; see outputs/claude/plate-finetune-run1-2026-10-10.md):
  1. YOLO26s COCO vehicle proposals (imgsz 1280, conf 0.20; --moto-conf for motorcycles)
  2. vehicle crop classifier (car / cng / motorcycle / rickshaw / bus_truck) on car, truck, bus, bicycle crops
  3. rickshaw -> reported, no registration expected, no plate search
  4. fine-tuned single-class plate detector (run 1, model B) on every other crop, conf 0.35; global NMS
  5. plate -> owning vehicle; allowed BRTA class letters from plate_rules (constraint for future OCR only)
Detection only: no OCR, no tracking. Usage: cascade.py IMAGE [IMAGE ...] [--out DIR] [--moto-conf 0.2]"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'outputs/claude/plate_rules'))
from bd_plate_rules import allowed_letters
from ultralytics import YOLO
VEH=ROOT/'work/detection-pilot/yolo26s.pt';CLS=ROOT/'work/claude/models/vehicle_cls_run2.pt';PLATE=ROOT/'work/claude/models/plate_B_yolo26s_run1.pt'
def cng_colour(crop,plate_box=None):
    """green (commercial, থ) / silver (private, দ) / unknown, from body and plate colour.
    green : lower body >= 3% green pixels, or a saturated (green) plate
    silver: pale plate (median HSV saturation < 30) AND bright, green-free lower body (median V >= 90)
    unknown otherwise (keeps both letters). Tuned on this video: 291 green / 37 silver crops (silver =
    1-2 physical vehicles) -> 263 green, 34 silver correct, 0 cross errors, 31 unknown."""
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);body=hsv[hsv.shape[0]//2:]
    if ((body[...,0]>=40)&(body[...,0]<=90)&(body[...,1]>=70)).mean()>=.03:return 'green'
    if plate_box is None:return 'unknown'
    x1,y1,x2,y2=[int(v) for v in plate_box];pl=hsv[max(0,y1):y2,max(0,x1):x2]
    if not pl.size:return 'unknown'
    if np.median(pl[...,1])>=30:return 'green'
    return 'silver' if np.median(body[...,2])>=90 else 'unknown'
COLORS={'car':(80,200,80),'cng':(0,160,255),'motorcycle':(255,150,60),'rickshaw':(200,100,255),'bus_truck':(220,210,0)}
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
class Cascade:
    def __init__(self,device='mps',moto_conf=.2,vehicle_conf=.2,plate_conf=.35):
        self.v,self.c,self.p=YOLO(str(VEH)),YOLO(str(CLS)),YOLO(str(PLATE))
        self.dev,self.moto_conf,self.vconf,self.pconf=device,moto_conf,vehicle_conf,plate_conf
    def __call__(self,im):
        H,W=im.shape[:2]
        r=self.v.predict(im,imgsz=1280,rect=False,conf=min(self.vconf,self.moto_conf),iou=.5,classes=[1,2,3,5,7],device=self.dev,verbose=False)[0]
        vehicles=[];plates=[]
        for b,s,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
            coco=self.v.names[int(k)]
            if s<(self.moto_conf if coco=='motorcycle' else self.vconf):continue
            x1,y1,x2,y2=b;px,py=(x2-x1)*.05,(y2-y1)*.05
            cb=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
            if cb[2]-cb[0]<32 or cb[3]-cb[1]<32:continue
            crop=im[cb[1]:cb[3],cb[0]:cb[2]]
            if coco=='motorcycle':vt,vc='motorcycle',float(s)
            else:
                pr=self.c.predict(crop,imgsz=224,device=self.dev,verbose=False)[0].probs;vt,vc=self.c.names[int(pr.top1)],float(pr.top1conf)
            reg=vt!='rickshaw'
            vehicles.append(dict(index=len(vehicles),box=[round(float(x),1) for x in b],detector_class=coco,detector_conf=round(float(s),3),
                                 vehicle_type=vt,type_conf=round(vc,3),registration_expected=reg,
                                 allowed_plate_letters=sorted(allowed_letters(vt,source='classifier') or []) if reg else [],
                                 note=None if reg else 'rickshaw: no BRTA registration plate'))
            if not reg:continue
            ds=self.p.predict(crop,imgsz=640,conf=self.pconf,device=self.dev,verbose=False)[0].boxes.data.cpu().numpy()
            for d in ds:
                plates.append(dict(box=[round(float(v+o),1) for v,o in zip(d[:4],cb[:2]*2)],confidence=round(float(d[4]),3),vehicle_index=vehicles[-1]['index']))
            if vt=='cng':
                col=cng_colour(crop,ds[ds[:,4].argmax(),:4] if len(ds) else None);v=vehicles[-1]
                v['cng_colour']=col;v['registration_class']={'green':'commercial (hire)','silver':'private'}.get(col,'unknown')
                v['allowed_plate_letters']=sorted(allowed_letters({'green':'cng_green','silver':'cng_silver'}.get(col,'cng')) or [])
        kept=[]
        for d in sorted(plates,key=lambda d:-d['confidence']):
            if all(iou(d['box'],k['box'])<.5 for k in kept):kept.append(d)
        for d in kept:   # owner = smallest vehicle box containing the plate centre (crop of origin as fallback)
            cx,cy=(d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2
            own=[v for v in vehicles if v['registration_expected'] and v['box'][0]<=cx<=v['box'][2] and v['box'][1]<=cy<=v['box'][3]]
            if own:d['vehicle_index']=min(own,key=lambda v:(v['box'][2]-v['box'][0])*(v['box'][3]-v['box'][1]))['index']
            d['vehicle_type']=vehicles[d['vehicle_index']]['vehicle_type']
        return dict(vehicles=vehicles,plates=kept)
def draw(im,res):
    out=im.copy()
    for v in res['vehicles']:
        b=[int(x) for x in v['box']];c=COLORS.get(v['vehicle_type'],(255,255,255))
        cv2.rectangle(out,b[:2],b[2:],c,2);cv2.putText(out,v['vehicle_type']+(f" {v['cng_colour']}" if v.get('cng_colour') else '')+('' if v['registration_expected'] else ' (no reg)'),(b[0],b[1]-6),0,.9,c,2)
    for p in res['plates']:
        b=[int(x) for x in p['box']];cv2.rectangle(out,b[:2],b[2:],(0,0,255),3);cv2.putText(out,f"plate {p['confidence']:.2f}",(b[0],b[3]+22),0,.7,(0,0,255),2)
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('images',nargs='+');ap.add_argument('--out',default=str(ROOT/'outputs/claude/pipeline/demo'))
    ap.add_argument('--moto-conf',type=float,default=.2);ap.add_argument('--device',default='mps');a=ap.parse_args()
    cas=Cascade(a.device,a.moto_conf);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for f in a.images:
        im=cv2.imread(f);res=cas(im);res['file']=Path(f).name
        (out/(Path(f).stem+'.json')).write_text(json.dumps(res,indent=1,ensure_ascii=False))
        cv2.imwrite(str(out/(Path(f).stem+'_annotated.jpg')),draw(im,res),[cv2.IMWRITE_JPEG_QUALITY,88])
        from collections import Counter
        print(Path(f).name,dict(Counter(v['vehicle_type'] for v in res['vehicles'])),'plates',len(res['plates']))
