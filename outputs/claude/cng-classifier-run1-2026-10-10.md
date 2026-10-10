# Vehicle-type crop classifier (CNG), runs 1–2, and CNG colour

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
  - That check also found 5 CNGs the user hadn't boxed, previously labelled `car`. One looked blue under the light; per the user, CNGs here are only green or silver. They were relabelled.

## Limits

- **CNG mining was green-filtered,** so silver and blue CNGs are under-represented in both training and test. Recall on non-green CNGs is not measured.
- **Microbuses → CNG** is the known failure: the boxy rear looks like a CNG. It is visible in `outputs/claude/pipeline/demo/`. Fix: add microbus examples, or a microbus class.
- **Bus/truck → car** confusion; only 11 bus/truck training examples.
- Consecutive frames 5 s apart repeat stopped vehicles, so the effective test size is smaller than 60.
- Labels are agent-verified, not independently adjudicated.

## Integration

`outputs/claude/pipeline/cascade.py` runs: vehicle detection → crop classifier → rickshaw reported with no registration and no plate search → fine-tuned plate detector (run 1, model B) → plate assigned to its vehicle with the allowed BRTA class letters (`plate_rules.allowed_letters(type, source='classifier')`). A CNG now constrains a future OCR reading to থ/দ; a classified car excludes the CNG letters.

## Run 2 (current): white-car fix and silver CNGs

The user clarified that CNGs come in two colours: **green** (commercial, by far the most common) and **silver** (private, rare). The green-filter mining had under-sampled silver CNGs. Reviewing the classifier's top-scoring non-green crops (`silver_top_*.jpg`) showed:

- **v1 called white/silver sedans and HiAce microbuses CNG.** Part of the cause was two white-car crops mislabelled as CNG in training (fixed).
- **Silver CNGs:** essentially 1–2 physical vehicles, repeated across stopped frames.
- **Green CNGs seen from behind** under the black "ভাড়ায় চালিত" (hire) canopy show almost no green.

v2 (`work/claude/models/vehicle_cls_run2.pt`) adds about 45 verified white-car/microbus hard negatives and about 70 silver or black-canopy CNG crops. Crops from the pilot frames form a separate "pilot" set that is never trained on.

| CNG precision / recall | Validation (49) | Locked test (69) | Pilot hard cases (37) |
| --- | --- | --- | --- |
| v1 | 96% / 100% | 95% / 100% | 57% / 100% (16 white cars → CNG) |
| **v2** | **100% / 100%** | **100% / 100%** | **100% / 100%** |

The pilot set was built from the same hard-case review, and the same silver CNG appears in training, so its 100% is optimistic. Bus/truck vs car remains weak (4 test errors; 11 training examples).

## CNG colour → registration class → plate letter

The colour rule is `cng_colour()` in `outputs/claude/pipeline/cascade.py`:

- **green** (commercial, letter থ): lower body ≥3% green pixels, or a saturated plate.
- **silver** (private, letter দ): pale plate (median saturation <30) **and** a bright, green-free body (median V ≥90).
- **unknown** otherwise; both letters stay allowed.

The rule is deliberately conservative because silver is rare and a wrong single letter would mislead OCR.

| Measured on this video | Plate saturation (median) | Lower-body brightness V (p10–p90) | Lower-body green |
| --- | --- | --- | --- |
| Green CNG (291 crops) | 134 | 58–82 | 20% |
| Silver CNG (37 crops) | 11 | 94–112 | 0.1% |

Rule result: 263 green and 34 silver correct, **0 green↔silver confusions**, 31 unknown (no plate found and no visible green, or a pale plate on a dark body). In the demo, the silver CNG in frame 1185 is called silver, and a dark-canopy green CNG in frame 1970 whose painted "P.C.R…" text was picked up as a plate is **unknown**, not silver.

**Caveat:** silver = 1–2 physical vehicles. The thresholds are not validated on other silver CNGs, other cameras or other lighting. The letter mapping থ = hire CNG, দ = private CNG comes from the blog source plus the user's colour rule, not from a BRTA gazette.
