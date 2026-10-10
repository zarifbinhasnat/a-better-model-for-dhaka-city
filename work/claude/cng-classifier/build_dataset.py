"""Step 3: merge labelled-frame crops with Claude-verified mined candidates; write an image-folder
classification dataset. Splits by time: val = 1700-1800 s, test = locked block >= 1950 s, else train."""
import json, shutil, cv2
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;LAB=HERE.parent/'labeling';OUT=HERE/'dataset'
# Claude's visual decisions on cands_*.jpg (default for an unlisted index = cng)
NOT_CNG={'car':{0,16,18,19,26,28,43,45,49,72,77,84,85,88,89,94,102,105,115,118,149,153,156,166,167,169,188,194,203,207,217,
                241,245,246,252,258,260,263,268,269,277,289,293,338,339,340,341},
         'rickshaw':{10,13,140,143,174,184,286,294},
         'motorcycle':{54,74,218,222,224,232,239}}
SKIP={4,11,14,21,23,31,34,47,60,61,62,67,73,75,76,83,86,87,91,128,130,132,133,142,148,152,155,157,162,170,171,
      180,185,186,191,196,197,210,213,226,230,233,234,236,237,249,275,276,287,288,291,298,305,309,314,320,322,332}
cands=json.load(open(HERE/'candidates.json'));rows=json.load(open(HERE/'labeled_rows.json'))
key=lambda r:(r['frame'],r['vi'])
over={}
for k,c in enumerate(cands):
    if k in SKIP:lab=None
    else:lab=next((n for n,s in NOT_CNG.items() if k in s),'cng')
    if c['src']=='labeled_car_check':over[key(c)]=lab      # relabel (or drop) a labelled-frame crop
    elif lab:rows.append(dict(c,label=lab,src='mined_verified'))
final=[]
for r in rows:
    if key(r) in over:
        if over[key(r)] is None:continue
        r=dict(r,label=over[key(r)],src=r['src']+'+claude_check')
    final.append(r)
shutil.rmtree(OUT,ignore_errors=True);cnt=Counter();cache={}
for i,r in enumerate(final):
    split='test' if r['split']=='locked_test' else 'val' if 1700<=r['t']<1800 else 'train'
    im=cache.get(r['frame'])
    if im is None:cache.clear();im=cache.setdefault(r['frame'],cv2.imread(str(LAB/'frames'/r['frame'])))
    x1,y1,x2,y2=r['box'];px,py=(x2-x1)*.05,(y2-y1)*.05;H,W=im.shape[:2]
    c=[max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py))]   # same 5% padding as the cascade
    d=OUT/split/r['label'];d.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(d/f"{r['frame'][:-4]}_v{r['vi']:03d}.jpg"),im[c[1]:c[3],c[0]:c[2]],[cv2.IMWRITE_JPEG_QUALITY,95])
    cnt[(split,r['label'])]+=1
json.dump(final,open(HERE/'final_rows.json','w'))
for s in('train','val','test'):print(s,{l:cnt[(s,l)] for l in('car','cng','motorcycle','rickshaw','bus_truck')})
