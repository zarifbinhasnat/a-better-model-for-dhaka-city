"""Full-frame vehicle dataset with local classes for a CNG/rickshaw-aware vehicle detector.
Labels = user-drawn vehicle boxes (cng, rickshaw, motorcycle, ...) + COCO YOLO26s pseudo-labels
(conf>=0.4) for every other vehicle; user boxes replace overlapping pseudo-labels (IoU>=0.3).
COCO 'bicycle' -> rickshaw (in this footage almost always a cycle-rickshaw; no registration).
Unlabelled CNGs the user missed would remain as pseudo 'car' (label noise; documented).
Splits by time as for plates: val 1700-1800 s, test = locked block >=1950 s."""
import json, shutil, sys
from pathlib import Path
import cv2
HERE=Path(__file__).resolve().parent;LAB=HERE.parent/'labeling'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/'kaggle/dhaka-vehicles-v1'
CLASSES=['car','motorcycle','bus','truck','cng','rickshaw','leguna']
COCO_MAP={'car':'car','motorcycle':'motorcycle','bus':'bus','truck':'truck','bicycle':'rickshaw'}
USER_MAP={'car':'car','motorcycle':'motorcycle','bus_truck':'truck','cng':'cng','rickshaw':'rickshaw',
          'rickshaw_or_nonmotorised':'rickshaw','leguna':'leguna'}
PSEUDO_CONF,REPLACE_IOU=.4,.3
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
shutil.rmtree(OUT/'yolo',ignore_errors=True)
srcs={}
for d in ('labels','labels_repaired'):
    for f in sorted((LAB/d).glob('frame_*.json')):
        j=json.loads(f.read_text())
        if f.name not in srcs or j.get('schema')==2 or (d=='labels_repaired' and srcs[f.name].get('schema')!=2):srcs[f.name]=j
stats={s:{c:0 for c in CLASSES}|{'frames':0,'user_boxes':0} for s in('train','val','test')}
for name,L in sorted(srcs.items()):
    if L['status']!='done':continue
    t=int(name[6:10]);split='test' if L['split']=='locked_test' else 'val' if 1700<=t<1800 else 'train'
    S=json.loads((LAB/'suggestions'/name).read_text());im=cv2.imread(str(LAB/'frames'/L['file']));H,W=im.shape[:2]
    user=[(v['box'],USER_MAP.get(v['vehicle'],'car')) for v in L.get('vehicles',[])]
    pseudo=[(v['box'],COCO_MAP[v['coco_class']]) for v in S['vehicles'] if v['confidence']>=PSEUDO_CONF and v['coco_class'] in COCO_MAP]
    pseudo=[p for p in pseudo if all(iou(p[0],u[0])<REPLACE_IOU for u in user)]
    for g in L['ignore']:   # user ignore regions (wheelchairs, unsure) painted grey
        b=[int(v) for v in g['box']];im[max(0,b[1]):b[3],max(0,b[0]):b[2]]=114
    boxes=user+pseudo
    (OUT/'yolo/images'/split).mkdir(parents=True,exist_ok=True);(OUT/'yolo/labels'/split).mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(OUT/'yolo/images'/split/L['file']),im,[cv2.IMWRITE_JPEG_QUALITY,93])
    (OUT/'yolo/labels'/split/name.replace('.json','.txt')).write_text(''.join(
        f"{CLASSES.index(c)} {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b,c in boxes))
    st=stats[split];st['frames']+=1;st['user_boxes']+=len(user)
    for _,c in boxes:st[c]+=1
(OUT/'yolo/data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {c}\n' for i,c in enumerate(CLASSES)))
(OUT/'weights').mkdir(parents=True,exist_ok=True);shutil.copy(HERE.parents[1]/'detection-pilot/yolo26s.pt',OUT/'weights/yolo26s.pt')
(OUT/'export_stats.json').write_text(json.dumps(stats,indent=1))
(OUT/'dataset-metadata.json').write_text(json.dumps(dict(title='dhaka-vehicles-v1',id='zarifb/dhaka-vehicles-v1',licenses=[dict(name='other')]),indent=1))
print(json.dumps(stats,indent=1))
