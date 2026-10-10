"""Detection-only comparison. Local inference; no OCR, training, or violation decisions.
YOLOv9 letterbox/output mapping follows ankandrew/open-image-models upstream.
"""
import argparse, json, os, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
WORK=HERE.parent.parent/'work/detection-pilot'

def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def nms(dets,threshold=.5):
    kept=[]
    for d in sorted(dets,key=lambda x:x['confidence'],reverse=True):
        if all(iou(d['box'],k['box'])<threshold for k in kept):kept.append(d)
    return kept

class International:
    def __init__(self,path):
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
        self.name=self.session.get_inputs()[0].name
    def predict(self,img):
        h,w=img.shape[:2];r=min(608/h,608/w);nw,nh=round(w*r),round(h*r)
        dw,dh=(608-nw)/2,(608-nh)/2
        resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
        padded=cv2.copyMakeBorder(resized,round(dh-.1),round(dh+.1),round(dw-.1),round(dw+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        pred=self.session.run(None,{self.name:tensor})[0].reshape(-1,7)
        out=[]
        for row in pred:
            if row[6]<.1:continue
            b=(row[1:5]-np.array([dw,dh,dw,dh]))/r
            b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h)
            if b[2]>b[0] and b[3]>b[1]:out.append({'box':b.tolist(),'confidence':float(row[6])})
        return out

class Pipeline:
    def __init__(self,device='cpu'):
        self.device=device
        torch.set_num_threads(4)
        self.vehicle=YOLO(str(WORK/'yolo26s.pt'))
        self.bangla=YOLO(str(WORK/'bangla-yolov8.pt'))
        self.international=International(WORK/'international-yolov9.onnx')
    def bangla_predict(self,img):
        r=self.bangla.predict(img,imgsz=608,rect=False,conf=.1,iou=.5,device=self.device,verbose=False)[0]
        return [{'box':b[:4].tolist(),'confidence':float(b[4])} for b in r.boxes.data.cpu().numpy()]
    def vehicles(self,img,track=False):
        kw=dict(imgsz=1280,rect=False,conf=.2,iou=.5,classes=[1,2,3,5,7],device=self.device,verbose=False)
        if track:r=self.vehicle.track(img,persist=True,tracker='bytetrack.yaml',**kw)[0]
        else:r=self.vehicle.predict(img,**kw)[0]
        boxes=r.boxes.xyxy.cpu().numpy();scores=r.boxes.conf.cpu().numpy();classes=r.boxes.cls.cpu().numpy()
        ids=r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
        out=[];h,w=img.shape[:2]
        for b,c,k,tid in zip(boxes,scores,classes,ids):
            x1,y1,x2,y2=b;px=(x2-x1)*.05;py=(y2-y1)*.05
            crop=[max(0,int(x1-px)),max(0,int(y1-py)),min(w,int(x2+px)),min(h,int(y2+py))]
            if crop[2]-crop[0]<32 or crop[3]-crop[1]<32:continue
            out.append({'box':b.tolist(),'crop_box':crop,'confidence':float(c),'class':self.vehicle.names[int(k)],'track_id':None if tid is None else int(tid)})
        return out
    def plates(self,img,vehicles,which,cascade=True):
        fn=self.international.predict if which=='international' else self.bangla_predict
        if not cascade:return fn(img)
        dets=[]
        for idx,v in enumerate(vehicles):
            x1,y1,x2,y2=v['crop_box']
            for d in fn(img[y1:y2,x1:x2]):
                d['box']=(np.array(d['box'])+np.array([x1,y1,x1,y1])).tolist()
                d['vehicle_index']=idx;d['track_id']=v['track_id'];dets.append(d)
        return nms(dets)

def synced_time(device):
    if device=='mps':torch.mps.synchronize()
    return time.perf_counter()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--annotations-only',action='store_true');args=ap.parse_args()
    p=Pipeline(args.device)
    files=sorted((HERE/'frames').glob('*.jpg'))
    files=[f for f in files if int(f.stem.split('_')[1])>=900]
    if args.annotations_only:
        wanted={x['file'] for x in json.loads((HERE/'annotations.json').read_text())['frames']};files=[f for f in files if f.name in wanted]
    if args.limit:files=files[:args.limit]
    # Warm-up fixed input shapes; excluded from reported timings.
    warm=np.zeros((700,1000,3),np.uint8);p.vehicles(warm);p.bangla_predict(warm);p.international.predict(warm)
    out=HERE/'predictions.jsonl'
    completed={json.loads(x)['file'] for x in out.read_text().splitlines()} if out.exists() else set()
    with out.open('a') as stream:
        for f in files:
            if f.name in completed:continue
            im=cv2.imread(str(f));start=synced_time(args.device);vs=p.vehicles(im);timings={'vehicle_seconds':synced_time(args.device)-start};pred={}
            for name in ['international','bangla']:
                for mode in ['fullframe','cascade']:
                    start=synced_time(args.device);pred[f'{name}_{mode}']=p.plates(im,vs,name,mode=='cascade');timings[f'{name}_{mode}_seconds']=synced_time(args.device)-start
            row={'file':f.name,'time_seconds':int(f.stem.split('_')[1]),'vehicles':vs,'predictions':pred,'timings':timings,'torch_device':args.device,'international_backend':'ONNX CPU, 4 threads','bangla_backend':f'PyTorch {args.device}, 4 CPU threads','image_size':[im.shape[1],im.shape[0]]}
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(f'{f.name}: {len(vs)} vehicles; '+', '.join(f'{k}={sum(d["confidence"]>=.25 for d in v)}' for k,v in pred.items()),flush=True)
if __name__=='__main__':main()
