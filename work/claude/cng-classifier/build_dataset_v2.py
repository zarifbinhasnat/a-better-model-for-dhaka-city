"""v2 dataset: v1 rows with two mislabelled white cars fixed, plus Claude-verified hard cases from the
silver-CNG mining review (silver_top_*.jpg): white/silver sedans & microbuses -> car; silver CNGs and
black-canopy (hire) CNGs -> cng. CNG colour is recorded (green/silver) for the colour study, the
classifier itself keeps one 'cng' class. Pilot-frame crops (dev/exploration frames) form a separate
'pilot' evaluation set and are never trained on."""
import json, shutil, cv2
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'dataset_v2'
FR={'labelling':ROOT/'work/claude/labeling/frames','pilot':ROOT/'outputs/detection-pilot/frames'}
SILVER={49,53,56,57,59,63,67,73,78,82,85,88,92,96,103,110,115,123,127,129,133,138,144,150,421,424,427,432,435,440,446,450,455,458,464,576,580}
GREEN={104,557,230,496,227,303,331,234,499,528,392,288,545,376,211,322,553,342,382,569,312,247,351,370,506,563,360,359,282,295,350,253}
CAR={41,58,60,65,70,74,79,83,87,89,93,412,423,426,429,434,208,443,242,246,64,19,99,101,276,281,466,397,113,504,80,75,515,84,500,527,587,21,408,265,287,125,270,518,521,71}
assert not (SILVER&GREEN or SILVER&CAR or GREEN&CAR)
rows=json.load(open(HERE/'final_rows.json'))
bad={(r['frame'],r['vi']) for i,r in enumerate(json.load(open(HERE/'low_green_cng.json'))) if i in (2,4)}
rows=[dict(r,label='car',src=r['src']+'+fixed_white_car') if (r['frame'],r['vi']) in bad else r for r in rows]
for r in rows:r.setdefault('frame_dir','labelling')
cands=json.load(open(HERE/'silver_candidates.json'));have={(r['frame'],r['vi']) for r in rows}
for i,c in enumerate(cands):
    lab='cng' if i in SILVER|GREEN else 'car' if i in CAR else None
    if not lab or (c['frame'],c['vi']) in have:continue
    t=int(c['frame'][6:10]);man=json.load(open(ROOT/'work/claude/labeling/manifest.json'))
    split='locked_test' if t in man['locked_test_times'] else 'train'
    rows.append(dict(frame=c['frame'],frame_dir=c['frame_dir'],t=t,split=split,vi=c['vi'],box=c['box'],coco=c['coco'],label=lab,
                     cng_colour='silver' if i in SILVER else 'green' if i in GREEN else None,src='silver_review'))
shutil.rmtree(OUT,ignore_errors=True);cnt=Counter();cache={}
for r in rows:
    split='pilot' if r['frame_dir']=='pilot' else 'test' if r['split']=='locked_test' else 'val' if 1700<=r['t']<1800 else 'train'
    key=(r['frame_dir'],r['frame'])
    if key not in cache:cache.clear();cache[key]=cv2.imread(str(FR[r['frame_dir']]/r['frame']))
    im=cache[key];x1,y1,x2,y2=r['box'];px,py=(x2-x1)*.05,(y2-y1)*.05;H,W=im.shape[:2]
    c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]
    d=OUT/split/r['label'];d.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(d/f"{r['frame_dir'][0]}_{r['frame'][:-4]}_v{r['vi']:03d}.jpg"),im[c[1]:c[3],c[0]:c[2]],[cv2.IMWRITE_JPEG_QUALITY,95]);cnt[(split,r['label'])]+=1
for s in ('train','val'):          # ultralytics needs every class folder present in val
    for l in ('car','cng','motorcycle','rickshaw','bus_truck'):(OUT/s/l).mkdir(parents=True,exist_ok=True)
json.dump(rows,open(HERE/'final_rows_v2.json','w'))
for s in('train','val','test','pilot'):print(s,{l:cnt[(s,l)] for l in('car','cng','motorcycle','rickshaw','bus_truck')})
