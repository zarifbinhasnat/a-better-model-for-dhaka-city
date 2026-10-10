"""Step 2: mine green CNG candidates (green_frac>=0.03) among vehicle proposals (conf>=0.25, not bicycle)
in frames without user labels, plus suspicious green 'car' crops in labelled frames. Claude verifies visually."""
import json, cv2, numpy as np
from pathlib import Path
from green import green_frac
LAB=Path('../labeling');man=json.load(open(LAB/'manifest.json'))
labeled={r['frame'] for r in json.load(open('labeled_rows.json'))}
cands=[]
for t in man['train_times']+man['locked_test_times']:
    fr=f'frame_{t:04d}.jpg';S=json.load(open(LAB/'suggestions'/fr.replace('.jpg','.json')));im=cv2.imread(str(LAB/'frames'/fr))
    split='locked_test' if t in man['locked_test_times'] else 'train'
    for vi,v in enumerate(S['vehicles']):
        b=[int(x) for x in v['box']]
        if v['coco_class']=='bicycle' or v['confidence']<.25 or b[2]-b[0]<24 or b[3]-b[1]<24:continue
        if fr in labeled:continue
        g=green_frac(im[b[1]:b[3],b[0]:b[2]])
        if g>=.03:cands.append(dict(frame=fr,t=t,split=split,vi=vi,box=v['box'],coco=v['coco_class'],green=round(g,3),src='mined'))
for r in json.load(open('labeled_rows.json')):
    if r['label']=='car':
        im=cv2.imread(str(LAB/'frames'/r['frame']));b=[int(x) for x in r['box']]
        g=green_frac(im[b[1]:b[3],b[0]:b[2]])
        if g>=.03:cands.append(dict(r,green=round(g,3),src='labeled_car_check'))
json.dump(cands,open('candidates.json','w'))
tiles=[];cache={}
for k,c in enumerate(cands):
    im=cache.get(c['frame']) if c['frame'] in cache else cache.setdefault(c['frame'],cv2.imread(str(LAB/'frames'/c['frame'])))
    b=[int(x) for x in c['box']];t=im[b[1]:b[3],b[0]:b[2]];s=150/max(t.shape[:2]);t=cv2.resize(t,(max(1,int(t.shape[1]*s)),max(1,int(t.shape[0]*s))))
    tile=np.zeros((168,150,3),np.uint8);tile[18:18+t.shape[0],:t.shape[1]]=t
    cv2.putText(tile,f"{k} {c['coco'][:4]}{'*' if c['src']!='mined' else ''}",(2,13),0,.4,(255,255,255),1);tiles.append(tile)
    if len(cache)>6:cache.pop(next(iter(cache)))
N=60
for s0 in range(0,len(tiles),N):
    ch=tiles[s0:s0+N];ch+=[np.zeros_like(tiles[0])]*(N-len(ch))
    cv2.imwrite(f'cands_{s0//N}.jpg',np.vstack([np.hstack(ch[i:i+10]) for i in range(0,N,10)]),[cv2.IMWRITE_JPEG_QUALITY,85])
print(len(cands),'candidates;',sum(c['src']!='mined' for c in cands),'labelled-car checks')
