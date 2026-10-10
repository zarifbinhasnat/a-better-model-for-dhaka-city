"""Step 1: label vehicle-proposal crops from the 17 user-labelled frames.
Priority: user vehicle box (IoU>=0.5) > proposal containing a CNG-tagged plate > COCO class (conf>=0.4).
Classes: car (incl. SUV/microbus/pickup), bus_truck, cng, motorcycle, rickshaw. No leguna on this route."""
import json, cv2
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;LAB=HERE.parent/'labeling'
USER={'car':'car','motorcycle':'motorcycle','bus_truck':'bus_truck','cng':'cng','rickshaw':'rickshaw','rickshaw_or_nonmotorised':'rickshaw'}
COCO={'car':'car','motorcycle':'motorcycle','bus':'bus_truck','truck':'bus_truck','bicycle':'rickshaw'}
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
srcs={}
for d in ('labels','labels_repaired'):
    for f in sorted((LAB/d).glob('frame_*.json')):
        j=json.loads(f.read_text())
        if f.name not in srcs or j.get('schema')==2 or (d=='labels_repaired' and srcs[f.name].get('schema')!=2):srcs[f.name]=j
rows=[];c=Counter()
for name,L in sorted(srcs.items()):
    if L['status']!='done':continue
    S=json.loads((LAB/'suggestions'/name).read_text());im=None
    users=[(v['box'],USER.get(v['vehicle'])) for v in L.get('vehicles',[]) if USER.get(v['vehicle'])]
    cngp=[p['box'] for p in L['plates'] if p['vehicle']=='cng']
    for vi,v in enumerate(S['vehicles']):
        b=v['box'];lab=src=None
        m=max(users,key=lambda u:iou(b,u[0]),default=None)
        if m and iou(b,m[0])>=.5:lab,src=m[1],'user_box'
        elif any(b[0]<=(p[0]+p[2])/2<=b[2] and b[1]<=(p[1]+p[3])/2<=b[3] for p in cngp) and v['coco_class']!='motorcycle':lab,src='cng','cng_plate'
        elif v['confidence']>=.4 and v['coco_class'] in COCO:lab,src=COCO[v['coco_class']],'coco'
        if not lab:continue
        if (b[2]-b[0])<24 or (b[3]-b[1])<24:continue
        rows.append(dict(frame=L['file'],t=int(name[6:10]),split=L['split'],vi=vi,box=b,coco=v['coco_class'],label=lab,src=src));c[(lab,src)]+=1
json.dump(rows,open(HERE/'labeled_rows.json','w'))
for k,v in sorted(c.items()):print(k,v)
print('total',len(rows))
