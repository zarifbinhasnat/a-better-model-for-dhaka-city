"""Proposal coverage of labelled plates on the 12 development frames (read-only on shared files)."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'outputs/detection-pilot'
ann=json.load(open(P/'annotations.json'))
rows={json.loads(l)['file']:json.loads(l) for l in open(P/'predictions.jsonl')}
def inside(p,c):
    iw=max(0,min(p[2],c[2])-max(p[0],c[0]));ih=max(0,min(p[3],c[3])-max(p[1],c[1]))
    return iw*ih/((p[2]-p[0])*(p[3]-p[1]))
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]));i=iw*ih
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
out=[]
for f in ann['frames']:
    r=rows[f['file']]
    hyb=[d for d in r['predictions']['international_cascade'] if r['vehicles'][d['vehicle_index']]['class'] not in('motorcycle','bicycle')]+\
        [d for d in r['predictions']['bangla_cascade'] if r['vehicles'][d['vehicle_index']]['class'] in('motorcycle','bicycle')]
    for i,b in enumerate(f['boxes_xyxy']):
        cov=[(inside(b,v['crop_box']),j) for j,v in enumerate(r['vehicles'])]
        full=[(j,r['vehicles'][j]['class'],round(r['vehicles'][j]['confidence'],2)) for c,j in cov if c>=.99]
        best=max(cov) if cov else (0,None)
        def hit(dets):return max([d['confidence'] for d in dets if iou(d['box'],b)>=.5],default=0)
        out.append(dict(file=f['file'],idx=i,box=b,h=b[3]-b[1],w=b[2]-b[0],covered=bool(full),
            owners=full,best_cover=round(best[0],2),
            intl=round(hit(r['predictions']['international_cascade']),2),
            bangla=round(hit(r['predictions']['bangla_cascade']),2),
            hybrid=round(hit(hyb),2)))
json.dump(out,open(Path(__file__).parent/'coverage.json','w'),indent=1)
n=len(out);c=sum(o['covered'] for o in out)
print(f'plates {n}; fully inside >=1 vehicle crop: {c} ({c/n:.1%})')
miss=[o for o in out if not o['covered']]
for o in miss:print('UNCOVERED',o['file'],o['box'],'h',o['h'],'best_cover',o['best_cover'])
fn=[o for o in out if o['hybrid']<.35]
print('hybrid FN at 0.35:',len(fn),' of which uncovered:',sum(not o['covered'] for o in fn))
from collections import Counter
print('owner classes (first owner) of covered plates:',Counter(o['owners'][0][1] for o in out if o['covered']))
