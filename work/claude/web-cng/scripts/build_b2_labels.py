"""Batch-2 marking (Claude, 2026-10-10): verified detector proposals (proposals_b2.json) + manual boxes."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
GREEN={12,14,15,47,55,58,59,60,62,68,69,72,74,87,89,92,96,100,102,108,112,113,114,117,122,127,130,136,137,138,139,141,142,145}
SILVER={11}
CAR={43,48,49,53,78,79,95,97,107,20}
BUS={37,83,105,121,77,80,94,82}
MOTO={110}
SKIP={10,17,104,106,111,134,135,133,129,131,150,118}
MANUAL={'b2_04.jpg':[dict(box=[194,118,502,334],label='cng',colour='silver',note='motion blur; প্রাইভেট on canopy')],
        'b2_10.jpg':[dict(box=[297,136,387,212],label='cng',colour='silver'),dict(box=[246,130,302,192],label='cng',colour='silver'),
                     dict(box=[214,125,257,181],label='cng',colour='silver'),dict(box=[148,124,175,165],label='cng',colour='silver'),
                     dict(box=[182,128,227,175],label='cng',colour='green'),dict(box=[167,127,200,170],label='cng',colour='green')]}
out={f:list(v) for f,v in MANUAL.items()}
for p in json.load(open(HERE/'proposals_b2.json')):
    if p['file'] in MANUAL or p['k'] in SKIP:continue
    k=p['k']
    if k in GREEN:o=dict(label='cng',colour='green')
    elif k in SILVER:o=dict(label='cng',colour='silver')
    elif k in CAR:o=dict(label='car')
    elif k in BUS:o=dict(label='bus_truck')
    elif k in MOTO:o=dict(label='motorcycle')
    elif p['cls']=='cng':continue                      # unreviewed CNG call: leave out
    else:o=dict(label=p['cls'])                         # reviewed non-CNG proposals: classifier label confirmed
    out.setdefault(p['file'],[]).append(dict(box=[round(v) for v in p['box']],proposal_k=k,**o))
json.dump(out,open(HERE/'web_labels_b2.json','w'),indent=1)
from collections import Counter
print(Counter((o['label'],o.get('colour')) for v in out.values() for o in v))
