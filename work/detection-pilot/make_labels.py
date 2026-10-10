import json
from pathlib import Path
# Visually annotated on unpredicted half-scale crops: original x=2*x, y=1400+2*y.
boxes={
30:[[570,282,601,305],[1564,235,1601,262],[1653,340,1697,375],[1460,435,1509,472],[1174,426,1214,458],[1002,379,1043,409],[636,433,678,467],[248,527,288,566]],
180:[[260,240,283,267],[569,281,601,305],[268,519,308,558],[636,433,679,466],[1006,390,1048,419],[1195,436,1240,474],[1470,373,1514,406],[1643,385,1687,420],[1676,152,1705,187]],
330:[[107,163,131,194],[156,190,175,215],[477,178,502,205],[568,282,600,305],[268,520,308,558],[636,434,678,467],[1003,391,1042,418],[1194,436,1238,473],[1471,379,1515,411],[1645,386,1688,420],[1673,156,1714,188],[1452,76,1486,101]],
480:[[269,169,289,190],[54,516,97,550],[418,434,460,468],[638,436,680,468],[1712,152,1747,179]],
630:[[585,96,637,124],[1218,126,1278,149],[1010,366,1048,393],[1542,622,1618,660],[346,671,422,710]],
780:[[391,79,419,99]],
930:[[433,49,464,74],[657,140,702,166]],
1080:[[591,129,641,156],[1440,347,1500,376]],
1230:[[303,210,355,237],[1200,437,1263,470],[854,3,901,28]],
1380:[[870,128,921,154],[1360,200,1418,228],[277,45,301,65],[465,250,486,272],[36,497,75,522]],
1530:[],
1680:[[850,434,913,464],[1446,70,1493,94],[380,379,420,406]],
}
ignores={30:[[85,504,115,541],[1600,698,1730,712]],180:[[85,504,115,541],[1760,270,1800,320]],330:[[85,504,115,541],[175,0,210,35]],630:[[298,0,362,17]],1080:[[595,178,657,223]],1680:[[947,143,1005,169],[550,177,579,214],[455,425,523,463]]}
def convert(b): return [2*b[0],1400+2*b[1],2*b[2],1400+2*b[3]]
d={'status':'provisional agent visual annotations; not independently human-adjudicated','created_before_model_predictions':True,'region_xyxy':[0,1400,3600,2824],'policy':'Label identifiable physical number plates in the fixed foreground ROI, including unreadable/blank physical plates. Ignore severely occluded, boundary-truncated or ambiguous plates and painted duplicate registration markings. Other text, signs and stickers are negatives. No transcription. All coordinates in original 4096x2824 pixels.','sampling':'12 timestamps fixed before prediction: 30 + 150*k seconds for k=0..11; repeated stopped vehicles are not independent.','frames':[{'file':f'frame_{t:04d}.jpg','time_seconds':t,'boxes_xyxy':[convert(b) for b in bs],'ignore_xyxy':[convert(b) for b in ignores.get(t,[])]} for t,bs in boxes.items()]}
p=Path('outputs/detection-pilot/annotations.json');p.write_text(json.dumps(d,indent=2))
print('Labeled plates:',sum(len(x) for x in boxes.values()),'frames:',len(boxes))
