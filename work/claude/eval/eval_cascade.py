"""End-to-end plate P/R of outputs/claude/pipeline/cascade.py on dev12 (+ locked-test frames with repaired
labels), comparing class-aware vs class-agnostic vehicle NMS. Conf 0.35, IoU 0.5, ROI [0,500,3600,2824]."""
import json, sys, cv2
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/detection-pilot';LAB=ROOT/'work/claude/labeling'
sys.path.insert(0,str(P));sys.path.insert(0,str(ROOT/'outputs/claude/pipeline'))
from evaluate import evaluate
from cascade import Cascade
ann=json.load(open(P/'annotations.json'))
sets={'dev12':[(P/'frames'/f['file'],f['boxes_xyxy'],f['ignore_xyxy']) for f in ann['frames']],'locked_test':[]}
for f in sorted((LAB/'labels_repaired').glob('frame_*.json')):
    d=json.load(open(f))
    if d['split']=='locked_test':sets['locked_test'].append((LAB/'frames'/d['file'],[p['box'] for p in d['plates']],[g['box'] for g in d['ignore']]))
for ag in (False,True):
    cas=Cascade(agnostic=ag)
    for sn,fr in sets.items():
        T=dict(tp=0,fp=0,fn=0);nv=0;dup=0
        for path,gt,ig in fr:
            res=cas(cv2.imread(str(path)));nv+=len(res['vehicles'])
            e=evaluate(res['plates'],gt,ig,[0,500,3600,2824],conf=.35)
            for k in T:T[k]+=e[k]
        print(f"agnostic={ag} {sn}: vehicles {nv} | plates TP {T['tp']} FP {T['fp']} FN {T['fn']}  P {T['tp']/max(1,T['tp']+T['fp']):.3f} R {T['tp']/max(1,T['tp']+T['fn']):.3f}",flush=True)
