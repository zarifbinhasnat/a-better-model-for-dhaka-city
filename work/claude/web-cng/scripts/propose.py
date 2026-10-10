"""Run the current cascade components on the user's downloaded CNG photos (raw/) BEFORE any training on
them: vehicle proposals (conf>=0.15), classifier v2 type, CNG colour rule, best plate box. Writes
proposals.json and an indexed review sheet for Claude's marking."""
import json, sys, cv2, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/claude/pipeline'))
from cascade import cng_colour
from ultralytics import YOLO
V=YOLO(str(ROOT/'work/detection-pilot/yolo26s.pt'));C=YOLO(str(ROOT/'work/claude/models'/(sys.argv[3] if len(sys.argv)>3 else 'vehicle_cls_run2.pt')));P=YOLO(str(ROOT/'work/claude/models/plate_B_yolo26s_run1.pt'))
out=[];tiles=[]
PAT=sys.argv[1] if len(sys.argv)>1 else '*.j*';OUTN=sys.argv[2] if len(sys.argv)>2 else 'proposals'
for f in sorted((HERE/'raw').glob(PAT)):
    im=cv2.imread(str(f));H,W=im.shape[:2];vis=im.copy()
    r=V.predict(im,imgsz=1280,conf=.15,iou=.5,classes=[1,2,3,5,7],agnostic_nms=True,verbose=False,device='mps')[0]
    for b,s,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
        x1,y1,x2,y2=b;px,py=(x2-x1)*.05,(y2-y1)*.05;c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
        crop=im[c[1]:c[3],c[0]:c[2]]
        if crop.shape[0]<16 or crop.shape[1]<16:continue
        pr=C.predict(crop,imgsz=224,device='mps',verbose=False)[0].probs;vt=C.names[int(pr.top1)]
        ds=P.predict(crop,imgsz=640,conf=.25,device='mps',verbose=False)[0].boxes.data.cpu().numpy()
        pb=ds[ds[:,4].argmax()] if len(ds) else None
        col=cng_colour(crop,pb[:4] if pb is not None else None) if vt=='cng' else None
        k_=len(out);out.append(dict(k=k_,file=f.name,box=[round(float(v),1) for v in b],coco=V.names[int(k)],det_conf=round(float(s),3),
            cls=vt,cls_conf=round(float(pr.top1conf),3),cng_colour=col,
            plate=[round(float(v+o),1) for v,o in zip(pb[:4],c[:2]*2)] if pb is not None else None,plate_conf=round(float(pb[4]),3) if pb is not None else None))
        bi=[int(v) for v in b];cv2.rectangle(vis,bi[:2],bi[2:],(0,255,255),2);cv2.putText(vis,f"{k_} {vt}{'/'+col if col else ''}",(bi[0],max(12,bi[1]+14)),0,.45,(0,0,255),2)
        if pb is not None:p=[int(v+o) for v,o in zip(pb[:4],c[:2]*2)];cv2.rectangle(vis,p[:2],p[2:],(255,0,255),2)
    cv2.imwrite(str(HERE/f'{OUTN}_{f.stem}.jpg'),cv2.resize(vis,(int(W*max(1,900/W)),int(H*max(1,900/W)))))
    s=420/H;tiles.append(cv2.resize(vis,(int(W*s),420)))
json.dump(out,open(HERE/f'{OUTN}.json','w'),indent=1)
rows=[np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)];w=max(r.shape[1] for r in rows)
cv2.imwrite(str(HERE/f'{OUTN}_sheet.jpg'),np.vstack([np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]))
for o in out:print(o['k'],o['file'],o['coco'],o['det_conf'],'->',o['cls'],o['cls_conf'],o['cng_colour'] or '',('plate %.2f'%o['plate_conf']) if o['plate'] else '')
