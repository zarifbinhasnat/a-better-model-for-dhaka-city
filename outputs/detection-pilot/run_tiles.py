"""Resolution-preserving alternative: overlap native 768-pixel tiles; same international model."""
import json,time
from pathlib import Path
import cv2
from run_detection import International,nms,WORK,HERE

def starts(length,size=768,stride=576):
    if length<=size:return [0]
    return sorted(set(list(range(0,length-size+1,stride))+[length-size]))

def predict_tiles(model,img):
    h,w=img.shape[:2];out=[]
    for y in starts(h):
        for x in starts(w):
            tile=img[y:min(y+768,h),x:min(x+768,w)]
            for d in model.predict(tile):
                b=d['box'];d['box']=[b[0]+x,b[1]+y,b[2]+x,b[3]+y];out.append(d)
    return nms(out)

if __name__=='__main__':
    ann=json.loads((HERE/'annotations.json').read_text());files=[f['file'] for f in ann['frames']]
    m=International(WORK/'international-yolov9.onnx');out=HERE/'tile_predictions.jsonl'
    done={json.loads(s)['file'] for s in out.read_text().splitlines()} if out.exists() else set()
    with out.open('a') as f:
        for name in files:
            if name in done:continue
            im=cv2.imread(str(HERE/'frames'/name));t=time.perf_counter();d=predict_tiles(m,im);elapsed=time.perf_counter()-t
            f.write(json.dumps({'file':name,'predictions':d,'seconds':elapsed})+'\n');f.flush()
            print(name,len(d),round(elapsed,2),flush=True)
