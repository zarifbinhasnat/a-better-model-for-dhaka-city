# Vehicle-type crop classifier (CNG), runs 1–3, and CNG colour

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

## Run 3 (current) and the user's downloaded CNG photos

The user downloaded 9 CNG photos: stock and news images, several showing private silver CNGs marked "PRIVATE" / "প্রাইভেট". Claude marked them (`work/claude/web-cng/web_labels.json`): 36 CNG boxes (23 green, 10 silver, 3 colour unknown, 2 more set to unknown because a white car occludes the lower body) and 2 cars. The images stay local (git-ignored): they are third-party stock and news photos, used only for local training and evaluation.

**Before training on them** (an out-of-domain test of classifier v2): CNG recall 31/36. The misses were silver CNGs called `car` (3) and `rickshaw` (2); a close-up side view was not detected at all. Colour rule v1: 22/23 green right, but 2 silver called green and 8 silver unknown.

**Pipeline fixes from this check**
- **Duplicate vehicle boxes** (user spotted 3 boxes on one CNG): COCO wavered between truck, bus and car, and class-aware NMS kept one box per class. The cascade now uses class-agnostic NMS. On dev12 that is 414 → 389 vehicle boxes, with plate results unchanged (95/15/7; locked test 16/2/3). Partial boxes with IoU just under 0.5 can still survive (one case in images-8).
- **Colour rule v2** (`cng_colour` in `cascade.py`), using the lower half of the CNG box:
  - green if ≥6% green pixels;
  - silver if <3% green and a pale plate, or, with no plate, a bright (V ≥80), low-saturation (S ≤40) body;
  - otherwise unknown.
  - **Occlusion guard:** if another vehicle in front covers ≥30% of the lower half, the result is unknown.
  - On all available data, with no untouched set: video 261/291 green and 36/37 silver right; photos 21/21 green and 8/10 silver right; **0 green↔silver confusions**. The silver CNG in video frame 1185 now reads unknown, correctly, because a car covers its lower body.
  - Private-CNG plates in the photos look **dark with light text**, not white. Both are low-saturation, which is what the rule uses.
- **Classifier v3** (`work/claude/models/vehicle_cls_run3.pt`): v2 data plus 34 crops from the photos, with images-5 and images-8 held out.

| | Validation (49) | Locked test (69) | Pilot hard cases (37) | Held-out photos (4 CNGs: 3 silver, 1 green) |
| --- | --- | --- | --- | --- |
| v2 | 91.8%, CNG 100/100 | 92.8%, CNG 100/100 | 100% | **2/4** (2 silver → car) |
| **v3** | 91.8%, CNG 100/100 | **97.1%**, CNG 100/100 | 100% | **4/4** |

The held-out photo test is only 4 vehicles: a sanity check, not a measurement. Remaining limits: tiny web thumbnails such as images-4 get no vehicle box at all, and the detector can box only part of a close-up side view (images-7), which leaves the colour unknown.
