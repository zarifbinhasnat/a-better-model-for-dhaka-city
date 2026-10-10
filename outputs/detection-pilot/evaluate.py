"""Provisional ROI detection evaluation; annotations are agent-drawn, not certified ground truth."""
import json,csv
from pathlib import Path
from run_detection import iou
ROOT=Path(__file__).resolve().parent

def inside(box,roi):
    x=(box[0]+box[2])/2;y=(box[1]+box[3])/2
    return roi[0]<=x<roi[2] and roi[1]<=y<roi[3]

def evaluate(pred,gt,ignore,roi,conf=.25,match_iou=.5):
    candidates=sorted([d for d in pred if d['confidence']>=conf and inside(d['box'],roi)],key=lambda d:d['confidence'],reverse=True)
    used=set();details=[]
    for d in candidates:
        opts=[(iou(d['box'],b),j) for j,b in enumerate(gt) if j not in used]
        best,j=max(opts,default=(0,-1))
        if best>=match_iou:
            used.add(j);status='TP'
        elif any(inside(d['box'],b) for b in ignore):status='ignored'
        else:status='FP'
        details.append({**d,'status':status,'matched_gt':j if status=='TP' else None,'best_iou':best})
    tp=len(used);fp=sum(d['status']=='FP' for d in details);fn=len(gt)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'ignored':sum(d['status']=='ignored' for d in details),'details':details,'missed_gt_indices':[j for j in range(len(gt)) if j not in used]}

def main():
    ann=json.loads((ROOT/'annotations.json').read_text());rows={r['file']:r for r in map(json.loads,(ROOT/'predictions.jsonl').read_text().splitlines())}
    selected=[f for f in ann['frames'] if f['file'] in rows]
    methods=list(next(iter(rows.values()))['predictions']);summary=[];frame_details=[]
    for conf in [.25,.5]:
        for method in methods:
            totals=dict(tp=0,fp=0,fn=0,ignored=0)
            for f in selected:
                r=evaluate(rows[f['file']]['predictions'][method],f['boxes_xyxy'],f['ignore_xyxy'],ann['region_xyxy'],conf)
                for k in totals:totals[k]+=r[k]
                if conf==.25:frame_details.append({'file':f['file'],'method':method,**r})
            tp,fp,fn=totals['tp'],totals['fp'],totals['fn'];precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
            summary.append({'method':method,'confidence':conf,**totals,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0})
    result={'annotation_status':ann['status'],'evaluated_frames':len(selected),'total_labeled_plates':sum(len(x['boxes_xyxy']) for x in selected),'region_xyxy':ann['region_xyxy'],'iou_threshold':.5,'summary':summary,'frame_details':frame_details}
    (ROOT/'metrics.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({k:v for k,v in result.items() if k!='frame_details'},indent=2))
if __name__=='__main__':main()
