import json, cv2, numpy as np
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot'
ann=json.load(open(P/'annotations.json'));R=ann['region_xyxy']
rows={json.loads(l)['file']:json.loads(l) for l in open(P/'predictions.jsonl')}
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def nms(ds,t=.5):
    k=[]
    for d in sorted(ds,key=lambda x:-x['confidence']):
        if all(iou(d['box'],x['box'])<t for x in k):k.append(d)
    return k
fpcls=Counter();tpcls=Counter();fps=[]
for f in ann['frames']:
    r=rows[f['file']];V=r['vehicles']
    hyb=nms([d for d in r['predictions']['international_cascade'] if V[d['vehicle_index']]['class'] not in('motorcycle','bicycle')]+
            [d for d in r['predictions']['bangla_cascade'] if V[d['vehicle_index']]['class'] in('motorcycle','bicycle')])
    hyb=[d for d in hyb if d['confidence']>=.35]
    hyb=[d for d in hyb if R[0]<=(d['box'][0]+d['box'][2])/2<=R[2] and R[1]<=(d['box'][1]+d['box'][3])/2<=R[3]]
    used=set()
    for d in hyb:
        c=V[d['vehicle_index']]['class']
        m=max(((iou(d['box'],g),i) for i,g in enumerate(f['boxes_xyxy']) if i not in used),default=(0,None))
        if m[0]>=.5: used.add(m[1]);tpcls[c]+=1;continue
        cx,cy=(d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2
        if any(g[0]<=cx<=g[2] and g[1]<=cy<=g[3] for g in f['ignore_xyxy']):continue
        fpcls[c]+=1;fps.append((f['file'],d,c))
print('hybrid TP by source vehicle class',dict(tpcls));print('hybrid FP by source vehicle class',dict(fpcls))
print('all vehicle proposals by class',Counter(v['class'] for f in ann['frames'] for v in rows[f['file']]['vehicles']))
json.dump([dict(file=a,box=d['box'],conf=d['confidence'],cls=c,vehicle=rows[a]['vehicles'][d['vehicle_index']]['box']) for a,d,c in fps],open('hybrid_fps.json','w'),indent=1)
# contact sheet: vehicle context around each labelled plate
cov=json.load(open('coverage.json'));tiles=[]
cache={}
def tile(img,b,label,color):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2;w=max(260,(b[2]-b[0])*7);h=w*.75
    x1,y1=int(max(0,cx-w/2)),int(max(0,cy-h*.7));x2,y2=int(min(img.shape[1],x1+w)),int(min(img.shape[0],y1+h))
    t=img[y1:y2,x1:x2].copy();cv2.rectangle(t,(int(b[0]-x1),int(b[1]-y1)),(int(b[2]-x1),int(b[3]-y1)),color,2)
    t=cv2.resize(t,(320,240));cv2.rectangle(t,(0,0),(320,22),(0,0,0),-1)
    cv2.putText(t,label,(3,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);return t
for k,o in enumerate(cov):
    img=cache.setdefault(o['file'],cv2.imread(str(P/'frames'/o['file'])))
    own=o['owners'][0][1] if o['owners'] else 'NONE'
    tiles.append(tile(img,o['box'],f"G{k} {own} h{o['h']} hyb{o['hybrid']}",(0,255,0)))
for k,fp in enumerate(json.load(open('hybrid_fps.json'))):
    img=cache.setdefault(fp['file'],cv2.imread(str(P/'frames'/fp['file'])))
    tiles.append(tile(img,fp['box'],f"F{k} {fp['cls']} c{fp['conf']:.2f}",(0,0,255)))
for s in range(0,len(tiles),24):
    ch=tiles[s:s+24];ch+= [np.zeros_like(tiles[0])]*(24-len(ch))
    sheet=np.vstack([np.hstack(ch[i:i+6]) for i in range(0,24,6)])
    cv2.imwrite(f'sheet_{s//24}.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,88])
print('tiles',len(tiles))
