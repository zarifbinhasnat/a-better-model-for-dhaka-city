"""Does CNG body colour (green=commercial, silver=private) show in (a) the plate colour and (b) the body?
For CNG crops with a known colour, detect the plate with model B and measure HSV of plate and body."""
import json, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FR={'labelling':ROOT/'work/claude/labeling/frames','pilot':ROOT/'outputs/detection-pilot/frames'}
P=YOLO(str(ROOT/'work/claude/models/plate_B_yolo26s_run1.pt'))
rows=json.load(open(HERE/'final_rows_v2.json'))
silver_v1={(r['frame'],r['vi']) for i,r in enumerate(json.load(open(HERE/'low_green_cng.json'))) if i in (0,1,3,5)}
out=[];cache={}
for r in rows:
    if r['label']!='cng':continue
    col=r.get('cng_colour') or ('silver' if (r['frame'],r['vi']) in silver_v1 else 'green')   # v1 CNGs were green-mined
    key=(r.get('frame_dir','labelling'),r['frame'])
    if key not in cache:cache.clear();cache[key]=cv2.imread(str(FR[key[0]]/r['frame']))
    im=cache[key];b=[int(x) for x in r['box']];crop=im[b[1]:b[3],b[0]:b[2]]
    d=P.predict(crop,imgsz=640,conf=.25,device='mps',verbose=False)[0].boxes.data.cpu().numpy()
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);body=hsv[hsv.shape[0]//2:]
    rec=dict(colour=col,body_val=float(np.median(body[...,2])),body_sat=float(np.median(body[...,1])),body_green=float(((body[...,0]>=40)&(body[...,0]<=90)&(body[...,1]>=70)).mean()))
    if len(d):
        x1,y1,x2,y2=[int(v) for v in d[d[:,4].argmax(),:4]];pl=hsv[y1:y2,x1:x2]
        if pl.size:rec.update(plate_hue=float(np.median(pl[...,0])),plate_sat=float(np.median(pl[...,1])),plate_val=float(np.median(pl[...,2])))
    out.append(rec)
json.dump(out,open(HERE/'colour_study.json','w'))
for c in('green','silver'):
    g=[o for o in out if o['colour']==c];pp=[o for o in g if 'plate_sat' in o]
    f=lambda k,L:f"{np.percentile([o[k] for o in L],10):.0f}/{np.median([o[k] for o in L]):.0f}/{np.percentile([o[k] for o in L],90):.0f}" if L else '-'
    print(f"{c:6s} n={len(g):3d} plate found {len(pp):3d} | plate hue p10/med/p90 {f('plate_hue',pp)} sat {f('plate_sat',pp)} | body sat {f('body_sat',g)} body green {np.median([o['body_green'] for o in g]):.3f}")
