# Vehicle-type crop classifier (CNG), run 1

Prepared 10 October 2026 (Claude). This replaces the failed full-frame local-class detector. The COCO detector already boxes CNGs and rickshaws but names them `car` / `bicycle`. A small classifier on each box assigns the local type. Legunas are omitted: they don't run on this route (user, 2026-10-10).

## Result

YOLO26s-cls (ImageNet init, 224 px, Mac M4 / MPS, about 2 minutes, best epoch 2). Weights: `work/claude/models/vehicle_cls_run1.pt`.

| Held-out split | Crops | Accuracy | CNG precision | CNG recall | Errors |
| --- | --- | --- | --- | --- | --- |
| Validation (1700–1800 s) | 47 | 91.5% | 100% (24/24) | 100% (24/24) | 4 bus/truck → car |
| Locked test (≥1950 s) | 60 | 95.0% | 94.1% (32/34) | 100% (32/32) | 2 microbus (Toyota HiAce) → CNG, 1 bus/truck → car |

Classes: car (sedan / SUV / microbus / pickup), cng, motorcycle, rickshaw, bus_truck. In the cascade, motorcycles keep the detector's label. The classifier runs on car, truck, bus and bicycle boxes.

## Data

- **Training crops:** 526 — 217 car, 213 CNG, 45 motorcycle, 40 rickshaw, 11 bus/truck — using the same 5% padding as the cascade. All from this video, split by time block. The test block is never trained on.
- **Label sources:**
  - In the 17 labelled frames: the user's vehicle boxes take priority, then boxes containing a CNG plate, then the COCO class at confidence ≥0.4.
  - In the other frames: CNG candidates mined by a green-body filter (≥3% green pixels; it keeps 83% of known CNGs and flags 4% of cars), each verified visually by Claude. 342 candidates were reviewed; crowded multi-vehicle boxes were skipped.
  - That check also found 5 CNGs the user hadn't boxed, previously labelled `car` (including a blue CNG). They were relabelled.

## Limits

- **CNG mining was green-filtered,** so silver and blue CNGs are under-represented in both training and test. Recall on non-green CNGs is not measured.
- **Microbuses → CNG** is the known failure: the boxy rear looks like a CNG. It is visible in `outputs/claude/pipeline/demo/`. Fix: add microbus examples, or a microbus class.
- **Bus/truck → car** confusion; only 11 bus/truck training examples.
- Consecutive frames 5 s apart repeat stopped vehicles, so the effective test size is smaller than 60.
- Labels are agent-verified, not independently adjudicated.

## Integration

`outputs/claude/pipeline/cascade.py` runs: vehicle detection → crop classifier → rickshaw reported with no registration and no plate search → fine-tuned plate detector (run 1, model B) → plate assigned to its vehicle with the allowed BRTA class letters (`plate_rules.allowed_letters(type, source='classifier')`). A CNG now constrains a future OCR reading to থ/দ; a classified car excludes the CNG letters.
