exec(open('work/detection-pilot/diagnose.py').read().split("coverage=0")[0])
for th in [.25,.35,.4,.5]:
 totals=[0,0,0]
 for a in ann['frames']:
  r=rows[a['file']];ds=nms([d for d in r['predictions']['international_cascade'] if r['vehicles'][d['vehicle_index']]['class'] not in ['motorcycle','bicycle']]+[d for d in r['predictions']['bangla_cascade'] if r['vehicles'][d['vehicle_index']]['class'] in ['motorcycle','bicycle']])
  m=evaluate(ds,a['boxes_xyxy'],a['ignore_xyxy'],ann['region_xyxy'],th)
  totals=[totals[i]+m[k] for i,k in enumerate(['tp','fp','fn'])]
 tp,fp,fn=totals;print(th,totals,'P',round(tp/(tp+fp),3),'R',round(tp/(tp+fn),3))
