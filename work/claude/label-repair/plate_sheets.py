import json,glob,cv2,numpy as np
from pathlib import Path
LAB=Path('../labeling')
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
items=[];tiles=[]
for f in sorted(glob.glob(str(LAB/'labels/*.json'))):
    d=json.load(open(f));s=json.load(open(LAB/'suggestions'/Path(f).name));im=cv2.imread(str(LAB/'frames'/d['file']))
    for i,p in enumerate(d['plates']):
        if not p['from_suggestion']:continue
        b=p['box'];m=max(s['plate_suggestions'],key=lambda q:iou(b,q['box']))
        k=len(items);items.append(dict(k=k,file=d['file'],label_index=i,box=b,vehicle=p['vehicle'],conf=m['confidence'],src=m['source'],iou_to_sugg=round(iou(b,m['box']),2),sugg_box=m['box']))
        w=b[2]-b[0];h=b[3]-b[1];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2;S=max(w*3.5,h*5,90)
        x1,y1=int(max(0,cx-S/2)),int(max(0,cy-S*.375));t=im[y1:int(y1+S*.75),x1:int(x1+S)].copy()
        cv2.rectangle(t,(int(b[0]-x1),int(b[1]-y1)),(int(b[2]-x1),int(b[3]-y1)),(0,255,0),1)
        if items[-1]['iou_to_sugg']<.95:sb=m['box'];cv2.rectangle(t,(int(sb[0]-x1),int(sb[1]-y1)),(int(sb[2]-x1),int(sb[3]-y1)),(0,0,255),1)
        t=cv2.resize(t,(220,165));c=np.zeros((180,220,3),np.uint8);c[15:]=t
        cv2.putText(c,f"{k} {p['vehicle'][:4]} {m['source'][:4]} {m['confidence']:.2f}",(2,11),0,.38,(255,255,255),1);tiles.append(c)
json.dump(items,open('plate_items.json','w'),indent=0)
for s0 in range(0,len(tiles),42):
    ch=tiles[s0:s0+42];ch+=[np.zeros_like(tiles[0])]*(42-len(ch))
    cv2.imwrite(f'plates_{s0//42}.jpg',np.vstack([np.hstack(ch[i:i+7]) for i in range(0,42,7)]),[cv2.IMWRITE_JPEG_QUALITY,85])
print(len(items))
