import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('outputs/detection-pilot').resolve()))
from evaluate import evaluate
from run_detection import nms,iou
p=Path('outputs/detection-pilot');ann=json.loads((p/'annotations.json').read_text());rows={x['file']:x for x in map(json.loads,(p/'predictions.jsonl').read_text().splitlines())}
coverage=0;n=0
for a in ann['frames']:
 row=rows[a['file']]
 for b in a['boxes_xyxy']:
  n+=1
  if any(v['crop_box'][0]<=b[0] and v['crop_box'][1]<=b[1] and v['crop_box'][2]>=b[2] and v['crop_box'][3]>=b[3] for v in row['vehicles']):coverage+=1
print('Plate ground truth fully inside vehicle crop:',coverage,'/',n)
for mode in ['union','class-switch','international-plus-confident-bangla','consensus-or-high-confidence']:
 totals=[0,0,0]
 for a in ann['frames']:
  row=rows[a['file']];ps=row['predictions'];ds=[]
  ints=ps['international_cascade'];bns=ps['bangla_cascade']
  if mode=='union':ds=nms([d for d in ints+bns if d['confidence']>=.25])
  elif mode=='class-switch':
   ds=nms([d for d in ints if row['vehicles'][d['vehicle_index']]['class'] not in ['motorcycle','bicycle']]+[d for d in bns if row['vehicles'][d['vehicle_index']]['class'] in ['motorcycle','bicycle']])
  elif mode=='international-plus-confident-bangla':
   ds=nms([d for d in ints if d['confidence']>=.25]+[d for d in bns if d['confidence']>=.6])
  else:
   ds=nms([d for d in ints if d['confidence']>=.7 or (d['confidence']>=.25 and any(b['confidence']>=.25 and iou(d['box'],b['box'])>=.5 for b in bns))]+[d for d in bns if d['confidence']>=.7])
  r=evaluate(ds,a['boxes_xyxy'],a['ignore_xyxy'],ann['region_xyxy'])
  totals=[totals[i]+r[k] for i,k in enumerate(['tp','fp','fn'])]
 tp,fp,fn=totals;print(mode,totals,'P',round(tp/(tp+fp),3),'R',round(tp/(tp+fn),3))
