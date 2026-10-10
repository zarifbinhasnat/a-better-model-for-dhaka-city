"""Label repair: the user's labels mix plate boxes and whole-vehicle boxes. For each vehicle box with
no plate inside (except rickshaws), propose a plate with both plate models; Claude reviews visually."""
import json, glob, sys, cv2, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];LAB=ROOT/'work/claude/labeling'
sys.path.insert(0,str(ROOT/'outputs/detection-pilot'))
from run_detection import Pipeline
def cat(b):
    w,h=b[2]-b[0],b[3]-b[1]
    if min(w,h)<5:return 'degenerate'
    return 'plate' if w<=150 and h<=100 and .6<=w/h<=4.5 else 'vehicle'
def ctr_in(q,b):return b[0]<=(q[0]+q[2])/2<=b[2] and b[1]<=(q[1]+q[3])/2<=b[3]
p=Pipeline('mps');items=[];tiles=[]
for f in sorted(glob.glob(str(LAB/'labels/*.json'))):
    d=json.load(open(f));im=cv2.imread(str(LAB/'frames'/d['file']));H,W=im.shape[:2]
    plates=[q['box'] for q in d['plates'] if cat(q['box'])=='plate']
    for i,q in enumerate(d['plates']):
        b=q['box']
        if cat(b)!='vehicle' or q['vehicle']=='other' or any(ctr_in(pl,b) for pl in plates):continue
        px,py=int((b[2]-b[0])*.05),int((b[3]-b[1])*.05);c=[max(0,b[0]-px),max(0,b[1]-py),min(W,b[2]+px),min(H,b[3]+py)]
        sub=im[c[1]:c[3],c[0]:c[2]]
        cands=[dict(d2,src='bangla') for d2 in p.bangla_predict(sub)]+[dict(d2,src='intl') for d2 in p.international.predict(sub)]
        cands=sorted([x for x in cands if x['confidence']>=.1],key=lambda x:-x['confidence'])
        best=cands[0] if cands else None
        k=len(items)
        items.append(dict(k=k,file=d['file'],label_index=i,vehicle=q['vehicle'],vehicle_box=b,
              candidate=None if not best else dict(box=[int(round(v+o)) for v,o in zip(best['box'],[c[0],c[1],c[0],c[1]])],conf=round(best['confidence'],2),src=best['src'])))
        t=sub.copy()
        if best:x1,y1,x2,y2=map(int,best['box']);cv2.rectangle(t,(x1,y1),(x2,y2),(0,255,0),2)
        s=260/max(t.shape[:2]);t=cv2.resize(t,(int(t.shape[1]*s),int(t.shape[0]*s)));canvas=np.zeros((280,260,3),np.uint8);canvas[20:20+t.shape[0],:t.shape[1]]=t
        cv2.putText(canvas,f"{k} {q['vehicle']} {best['src']+' '+str(round(best['confidence'],2)) if best else 'NO CAND'}",(3,14),0,.42,(255,255,255),1)
        tiles.append(canvas)
json.dump(items,open(HERE/'proposals.json','w'),indent=1)
for s0 in range(0,len(tiles),24):
    ch=tiles[s0:s0+24];ch+=[np.zeros_like(tiles[0])]*(24-len(ch))
    cv2.imwrite(str(HERE/f'review_{s0//24}.jpg'),np.vstack([np.hstack(ch[i:i+6]) for i in range(0,24,6)]))
print(len(items),'vehicle boxes without plates;',sum(i['candidate'] is not None for i in items),'with a candidate')
