"""Model suggestions to speed up human labelling (NOT labels). Rickshaws (COCO bicycle) are kept as
vehicles with no registration expected; no plate search on them."""
import json, sys, cv2, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/detection-pilot'));sys.path.insert(0,str(ROOT/'outputs/claude/plate_rules'))
from run_detection import Pipeline, nms
from bd_plate_rules import vehicle_record
dev=sys.argv[1] if len(sys.argv)>1 else 'mps'
p=Pipeline(dev);out=HERE/'suggestions';out.mkdir(exist_ok=True)
for f in sorted((HERE/'frames').glob('*.jpg')):
    o=out/(f.stem+'.json')
    if o.exists():continue
    im=cv2.imread(str(f));h,w=im.shape[:2]
    r=p.vehicle.predict(im,imgsz=1280,rect=False,conf=.1,iou=.5,classes=[1,2,3,5,7],device=dev,verbose=False)[0]
    vehicles=[];plates=[]
    for b,c,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
        cls=p.vehicle.names[int(k)];rec=vehicle_record(cls)
        vehicles.append(dict(box=[round(float(x),1) for x in b],confidence=round(float(c),3),coco_class=cls,**rec))
        if not rec['registration_expected']:continue
        x1,y1,x2,y2=b;px=(x2-x1)*.05;py=(y2-y1)*.05
        crop=[max(0,int(x1-px)),max(0,int(y1-py)),min(w,int(x2+px)),min(h,int(y2+py))]
        if crop[2]-crop[0]<32 or crop[3]-crop[1]<32:continue
        sub=im[crop[1]:crop[3],crop[0]:crop[2]];off=np.array(crop[:2]*2)
        bd=[dict(box=(np.array(d['box'])+off).tolist(),confidence=d['confidence'],source='bangla') for d in p.bangla_predict(sub)]
        it=[dict(box=(np.array(d['box'])+off).tolist(),confidence=d['confidence'],source='international') for d in p.international.predict(sub)]
        if cls=='motorcycle':plates+=[d for d in bd if d['confidence']>=.25]
        else:plates+=[d for d in it if d['confidence']>=.25]+[d for d in bd if d['confidence']>=.5]  # Bangla extra pass catches green CNG plates
    plates=[dict(d,box=[round(x,1) for x in d['box']],confidence=round(d['confidence'],3)) for d in nms(plates,.3)]
    json.dump(dict(file=f.name,vehicles=vehicles,plate_suggestions=plates,status='model suggestions, unverified'),open(o,'w'))
    print(f.name,len(vehicles),'vehicles',len(plates),'plate suggestions',flush=True)
