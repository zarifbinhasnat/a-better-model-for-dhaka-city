"""Build repaired labels (labels_repaired/) from the user's labels without modifying them.
Findings: the user kept 270/271 model plate suggestions unreviewed, and every box they drew was a
WHOLE VEHICLE (cng / motorcycle / 'other' = rickshaws & non-motorised). Claude reviewed each kept
suggestion visually (plates_*.jpg) and each plate proposal inside the user's vehicle boxes (review_*.jpg).
Provisional agent review, not independent human adjudication."""
import json, glob
from pathlib import Path
LAB=Path('../labeling');OUT=LAB/'labels_repaired';OUT.mkdir(exist_ok=True)
REJECT={3,7,9,10,15,16,17,26,28,29,30,31,32,46,48,50,52,53,54,56,69,71,72,81,89,94,96,100,101,102,103,105,123,
        138,140,148,156,158,182,191,193,194,196,203,204,205,214,221,223,226,227,230,232,235,237,241,242,243,
        259,262,263,266,267,269}
IGNORE={8,19,33,55,68,78,79,80,82,83,85,86,87,88,93,98,99,117,118,119,120,121,122,132,134,137,141,149,150,151,
        155,157,172,176,181,185,186,187,189,190,192,195,218,220,229,236,265}
TO_CNG={43,67,115,133,153,171,174,188,212,228,231,256,257}
ACCEPT_PROPOSAL={4,5,23,32,33,35,36,37,44,58}
assert not (REJECT&IGNORE)
items={(i['file'],i['label_index']):i for i in json.load(open('plate_items.json'))}
props={(p['file'],p['label_index']):p for p in json.load(open('proposals.json'))}
def ctr_in(q,b):return b[0]<=(q[0]+q[2])/2<=b[2] and b[1]<=(q[1]+q[3])/2<=b[3]
tot=dict(kept=0,rejected=0,ignored=0,to_cng=0,proposals=0,vehicles=0,vehicle_ignore=0,degenerate=0)
for f in sorted(glob.glob(str(LAB/'labels/*.json'))):
    d=json.load(open(f));plates=[];ignore=[dict(box=g['box'],reason='user ignore') for g in d['ignore']];vehicles=[];log=[]
    for i,p in enumerate(d['plates']):
        if p['from_suggestion']:
            k=items[(d['file'],i)]['k']
            if k in REJECT:tot['rejected']+=1;log.append(f'suggestion {k} rejected (not a plate)');continue
            if k in IGNORE:tot['ignored']+=1;ignore.append(dict(box=p['box'],reason=f'agent unsure, suggestion {k}'));continue
            v='cng' if k in TO_CNG else p['vehicle'];tot['to_cng']+=k in TO_CNG;tot['kept']+=1
            plates.append(dict(box=p['box'],vehicle=v,source='model suggestion kept by user; agent-reviewed'))
        else:
            b=p['box']
            if min(b[2]-b[0],b[3]-b[1])<5:tot['degenerate']+=1;log.append(f'dropped degenerate drawn box {b}');continue
            vt={'other':'rickshaw_or_nonmotorised'}.get(p['vehicle'],p['vehicle'])
            vehicles.append(dict(box=b,vehicle=vt,registration_expected=vt!='rickshaw_or_nonmotorised',source='user-drawn vehicle box'));tot['vehicles']+=1
    for i,p in enumerate(d['plates']):
        pr=props.get((d['file'],i))
        if pr and not p['from_suggestion'] and pr['k'] in ACCEPT_PROPOSAL and pr['candidate']:
            plates.append(dict(box=pr['candidate']['box'],vehicle=p['vehicle'],source=f"plate proposal {pr['k']} in user vehicle box; agent-accepted"));tot['proposals']+=1
    for v in vehicles:   # vehicles that should carry a plate but have none labelled -> ignore so the plate isn't a false negative
        if v['registration_expected'] and not any(ctr_in(q['box'],v['box']) for q in plates):
            ignore.append(dict(box=v['box'],reason='user vehicle box without a confirmed plate'));tot['vehicle_ignore']+=1
    json.dump(dict(file=d['file'],split=d['split'],status=d['status'],labeler='user vehicle boxes + agent-reviewed plate suggestions',
                   plates=plates,ignore=ignore,vehicles=vehicles,repair_log=log,notes=d.get('notes','')),open(OUT/Path(f).name,'w'),indent=1,ensure_ascii=False)
print(tot)
