"""End-to-end development diagnostic: does changing the VEHICLE stage (confidence, classes, CNG routing)
improve plate P/R without fine-tuning? Development frames only -> NOT held-out evidence.
Read-only use of shared pilot code/data; writes only to this folder."""
import json, sys, cv2, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot'
sys.path.insert(0,str(P))
from run_detection import Pipeline, nms
from evaluate import evaluate
ann=json.load(open(P/'annotations.json'));tags=json.load(open('tags.json'));cov=json.load(open('coverage.json'))
cache=Path('e2e_cache.json')
if cache.exists():data=json.load(open(cache))
else:
    p=Pipeline('cpu');data={}
    for f in ann['frames']:
        im=cv2.imread(str(P/'frames'/f['file']))
        r=p.vehicle.predict(im,imgsz=1280,rect=False,conf=.05,iou=.5,classes=[1,2,3,5,7],device='cpu',verbose=False)[0]
        vs=[];h,w=im.shape[:2]
        for b,c,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
            x1,y1,x2,y2=b;px=(x2-x1)*.05;py=(y2-y1)*.05
            crop=[max(0,int(x1-px)),max(0,int(y1-py)),min(w,int(x2+px)),min(h,int(y2+py))]
            if crop[2]-crop[0]<32 or crop[3]-crop[1]<32:continue
            sub=im[crop[1]:crop[3],crop[0]:crop[2]];off=np.array(crop[:2]*2)
            dets={'international':[{'box':(np.array(d['box'])+off).tolist(),'confidence':d['confidence']} for d in p.international.predict(sub)],
                  'bangla':[{'box':(np.array(d['box'])+off).tolist(),'confidence':d['confidence']} for d in p.bangla_predict(sub)]}
            vs.append({'box':b.tolist(),'crop_box':crop,'confidence':float(c),'class':p.vehicle.names[int(k)],'dets':dets})
        data[f['file']]=vs;print(f['file'],len(vs),flush=True)
    json.dump(data,open(cache,'w'))
# CNG oracle: smallest proposal whose box contains the centre of a CNG-tagged plate
cng_plates=[(cov[k]['file'],cov[k]['box']) for k in tags['cng']]
def is_cng(file,v):return any(fl==file and v['box'][0]<=(b[0]+b[2])/2<=v['box'][2] and v['box'][1]<=(b[1]+b[3])/2<=v['box'][3] for fl,b in cng_plates)
def run(vconf,drop_bicycle=False,cng_oracle=False,route='hybrid'):
    T=dict(tp=0,fp=0,fn=0)
    for f in ann['frames']:
        vs=[v for v in data[f['file']] if v['confidence']>=vconf and not(drop_bicycle and v['class']=='bicycle')]
        if cng_oracle:
            for v in vs:v['_cng']=False
            for fl,b in cng_plates:
                if fl!=f['file']:continue
                c=[v for v in vs if v['box'][0]<=(b[0]+b[2])/2<=v['box'][2] and v['box'][1]<=(b[1]+b[3])/2<=v['box'][3]]
                if c:min(c,key=lambda v:(v['box'][2]-v['box'][0])*(v['box'][3]-v['box'][1]))['_cng']=True
        per={m:nms([dict(d,vi=i) for i,v in enumerate(vs) for d in v['dets'][m]]) for m in('international','bangla')}
        def small(i):return vs[i]['class'] in('motorcycle','bicycle') or (cng_oracle and vs[i].get('_cng'))
        if route=='hybrid':sel=nms([d for d in per['international'] if not small(d['vi'])]+[d for d in per['bangla'] if small(d['vi'])])
        else:sel=per[route]
        r=evaluate(sel,f['boxes_xyxy'],f['ignore_xyxy'],ann['region_xyxy'],conf=.35)
        for k in T:T[k]+=r[k]
    n=sum(len(v) for f in ann['frames'] for v in [[x for x in data[f['file']] if x['confidence']>=vconf and not(drop_bicycle and x['class']=='bicycle')]])
    tp,fp,fn=T['tp'],T['fp'],T['fn'];return dict(**T,P=round(tp/(tp+fp),3),R=round(tp/(tp+fn),3),proposals=n)
rows=[]
for vconf in(.2,.1,.05):
    for db in(False,True):
        for co in(False,True):
            r=run(vconf,db,co);rows.append(dict(vehicle_conf=vconf,drop_bicycle=db,cng_oracle=co,**r));print(rows[-1])
json.dump(rows,open('e2e_results.json','w'),indent=1)
