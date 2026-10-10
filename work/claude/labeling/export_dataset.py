"""Turn human labels (labels/*.json) into a vehicle-crop plate dataset: YOLO txt + COCO json.
Crops mimic inference: vehicle proposals (conf>=0.1, 5% padding), rickshaws skipped (no registration).
Plates >=90% inside a crop are labelled; partly-inside plates and ignore regions are painted grey so
they are neither positives nor false negatives. Splits are by frame time, never by random crop:
  train  = labelled train frames outside VAL_BLOCK
  val    = labelled train frames inside VAL_BLOCK (threshold/epoch selection only)
  test   = labelled locked-test frames (>=1950 s) - kept for final scoring, never for training
Run: python export_dataset.py [out_dir]"""
import json, sys, shutil
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / 'claude/kaggle/dhaka-plates-v1'
VAL_BLOCK = (1700, 1800)          # seconds; contiguous block, chosen before any training
VEHICLE_CONF, PAD, MIN_CROP, INSIDE = .10, .05, 32, .90

def frac_inside(b, c):
    iw = max(0, min(b[2], c[2]) - max(b[0], c[0])); ih = max(0, min(b[3], c[3]) - max(b[1], c[1]))
    return iw * ih / max(1, (b[2] - b[0]) * (b[3] - b[1]))

def main():
    if OUT.exists():
        shutil.rmtree(OUT / 'yolo', ignore_errors=True); shutil.rmtree(OUT / 'coco', ignore_errors=True)
    coco = {s: dict(images=[], annotations=[], categories=[dict(id=1, name='plate', supercategory='plate')]) for s in ('train', 'valid', 'test')}
    ysplit = {'train': 'train', 'valid': 'val', 'test': 'test'}
    stats = {s: dict(frames=0, crops=0, plates=0, cng=0, negatives=0) for s in coco}
    meta = []
    # per frame: new-UI labels (schema 2) > Claude-repaired labels > original labels
    srcs = {}
    for d in ('labels', 'labels_repaired'):
        for f in sorted((HERE / d).glob('frame_*.json')):
            j = json.loads(f.read_text())
            if f.name not in srcs or j.get('schema') == 2 or (d == 'labels_repaired' and srcs[f.name][1].get('schema') != 2):
                srcs[f.name] = (f, j)
    print('label sources:', {d: sum(1 for f, _ in srcs.values() if f.parent.name == d) for d in ('labels', 'labels_repaired')})
    for name, (lab, L) in sorted(srcs.items()):
        L = dict(L)
        unrev = [p for p in L['plates'] if p.get('reviewed') is False]       # unreviewed suggestions -> ignore, never positives
        L['plates'] = [p for p in L['plates'] if p.get('reviewed') is not False]
        L['ignore'] = list(L['ignore']) + [dict(box=p['box']) for p in unrev]
        if L['status'] != 'done':
            continue
        t = int(lab.stem.split('_')[1])
        split = 'test' if L['split'] == 'locked_test' else 'valid' if VAL_BLOCK[0] <= t < VAL_BLOCK[1] else 'train'
        S = json.loads((HERE / 'suggestions' / lab.name).read_text())
        im = cv2.imread(str(HERE / 'frames' / L['file'])); H, W = im.shape[:2]
        stats[split]['frames'] += 1
        for vi, v in enumerate(S['vehicles']):
            if v['confidence'] < VEHICLE_CONF or not v['registration_expected']:
                continue
            x1, y1, x2, y2 = v['box']; px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            c = [max(0, int(x1 - px)), max(0, int(y1 - py)), min(W, int(x2 + px)), min(H, int(y2 + py))]
            if c[2] - c[0] < MIN_CROP or c[3] - c[1] < MIN_CROP:
                continue
            crop = im[c[1]:c[3], c[0]:c[2]].copy(); cw, ch = c[2] - c[0], c[3] - c[1]
            keep = []
            for p in L['plates']:
                f = frac_inside(p['box'], c)
                b = [p['box'][0] - c[0], p['box'][1] - c[1], p['box'][2] - c[0], p['box'][3] - c[1]]
                if f >= INSIDE:
                    keep.append((p, [max(0, b[0]), max(0, b[1]), min(cw, b[2]), min(ch, b[3])]))
                elif f > 0:
                    crop[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])] = 114
            for g in L['ignore']:
                if frac_inside(g['box'], c) > 0:
                    b = [g['box'][0] - c[0], g['box'][1] - c[1], g['box'][2] - c[0], g['box'][3] - c[1]]
                    crop[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])] = 114
            name = f'{lab.stem}_v{vi:03d}'
            yd = OUT / 'yolo' / 'images' / ysplit[split]; ld = OUT / 'yolo' / 'labels' / ysplit[split]; cd = OUT / 'coco' / split
            for d in (yd, ld, cd):
                d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(yd / f'{name}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(yd / f'{name}.jpg', cd / f'{name}.jpg')
            (ld / f'{name}.txt').write_text(''.join(
                f'0 {(b[0]+b[2])/2/cw:.6f} {(b[1]+b[3])/2/ch:.6f} {(b[2]-b[0])/cw:.6f} {(b[3]-b[1])/ch:.6f}\n' for _, b in keep))
            iid = len(coco[split]['images']) + 1
            coco[split]['images'].append(dict(id=iid, file_name=f'{name}.jpg', width=cw, height=ch))
            for _, b in keep:
                coco[split]['annotations'].append(dict(id=len(coco[split]['annotations']) + 1, image_id=iid, category_id=1,
                                                       bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]],
                                                       area=(b[2] - b[0]) * (b[3] - b[1]), iscrowd=0))
            st = stats[split]; st['crops'] += 1; st['plates'] += len(keep)
            st['cng'] += sum(p['vehicle'] == 'cng' for p, _ in keep); st['negatives'] += not keep
            meta.append(dict(crop=name, split=split, frame=L['file'], crop_box=c, coco_class=v['coco_class'],
                             vehicle_conf=v['confidence'], plates=[dict(vehicle=p['vehicle'], box=b) for p, b in keep]))
    for s in coco:
        if coco[s]['images']:
            (OUT / 'coco' / s / '_annotations.coco.json').write_text(json.dumps(coco[s]))
    (OUT / 'weights').mkdir(exist_ok=True)
    shutil.copy(HERE.parents[1] / 'detection-pilot/bangla-yolov8.pt', OUT / 'weights/bangla-yolov8.pt')  # init for fine-tune variant A
    (OUT / 'yolo' / 'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n')
    (OUT / 'crops.json').write_text(json.dumps(meta))
    (OUT / 'export_stats.json').write_text(json.dumps(dict(stats=stats, val_block_seconds=VAL_BLOCK, vehicle_conf=VEHICLE_CONF,
                                                          note='plate counts are per crop; one plate can appear in overlapping crops'), indent=1))
    (OUT / 'dataset-metadata.json').write_text(json.dumps(dict(title='dhaka-plates-v1', id='zarifb/dhaka-plates-v1',
                                                               licenses=[dict(name='other')]), indent=1))
    print(json.dumps(stats, indent=1))

if __name__ == '__main__':
    main()
