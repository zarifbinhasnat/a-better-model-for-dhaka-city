# Do we need to fine-tune the vehicle detector? (CNG, leguna, motorcycle, rickshaw)

Prepared 10 October 2026 (Claude). **Development diagnostic only**: 12 development frames, 102 provisional agent-drawn plate labels, same camera/session, stopped vehicles repeat. Matching IoU ≥0.5, plate confidence ≥0.35, ROI `[0,500,3600,2824]`, the class-switch hybrid from `check_protocol.json`. Nothing here is held-out evidence, and the frozen check settings were not changed.

## Answer

**Not for plate detection, not now.** The COCO vehicle detector (YOLO26s) already puts 96% of labelled plates inside a vehicle crop at its current threshold, and 99–100% at a lower threshold. Changing the vehicle stage recovers at most 5 of the 15 missed plates. The remaining misses and nearly all false positives come from the **plate** detectors. Fine-tune the plate detector first.

A local-class vehicle detector (CNG, leguna, SUV vs sedan, microbus, pickup, rickshaw) becomes worth training **later, for the plate-letter rule** (`plate_rules/`). COCO's single `car` class covers sedans, SUVs, microbuses and CNGs, so it cannot tell OCR which class letter to expect.

## Evidence

Vehicle type was tagged visually by Claude from context tiles (`work/claude/vehicle-audit/tags.json`). Recall is at plate confidence ≥0.35:

| True vehicle type | Plates | Inside a vehicle crop | Hybrid recall | International model | Bangla model | COCO class given |
| --- | --- | --- | --- | --- | --- | --- |
| Car (sedan/SUV/hatchback) | 60 | 60 | 59 | 59 | 50 | car |
| Microbus / pickup / truck | 7 | 7 | 7 | 7 | 5 | car / truck |
| **CNG** | 10 | 10 | **4** | 4 | **8** | car or truck, never a CNG class |
| **Motorcycle** | 25 | 21 | **17** | 8 | 18 | motorcycle; 4 bikes missed entirely, 2 plates only inside a neighbouring car crop |
| Leguna | 0 | — | — | — | — | none appear in these 12 frames |
| Rickshaw (as `bicycle`) | no plates (municipal tags are ignore regions) | | | | | 19 proposals, 0 true plates |

Hybrid false positives (16): 7 are painted numbers on CNG bumpers, 3 are signs or shop text, 3 are double or mislocalised boxes on real plates, and 3 are other.

End-to-end variants of the vehicle stage (`work/claude/vehicle-audit/e2e.py`; the first row reproduces the pilot exactly):

| Vehicle stage | Crops | TP / FP / FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| Current: confidence 0.20 | 414 | 87 / 16 / 15 | 84.5% | 85.3% |
| + perfect CNG→Bangla routing (oracle from tags) | 414 | 89 / 14 / 13 | 86.4% | 87.3% |
| Confidence 0.10 | 603 | 90 / 19 / 12 | 82.6% | 88.2% |
| Confidence 0.10 + CNG oracle | 603 | 92 / 20 / 10 | 82.1% | 90.2% |
| Confidence 0.05 | 849 | 90 / 20 / 12 | 81.8% | 88.2% |
| Drop `bicycle` (rickshaw) class | −5% crops | identical metrics | | |

The vehicle-detector sweep (`vehicle_sweep.py`) puts 24 of 25 motorcycle plates inside a motorcycle box at confidence 0.05 and input 1280, versus 17 at 0.20. A larger input (2560) made coverage worse.

## What this means

1. **Motorcycle:** lower the vehicle threshold for the motorcycle class only (about 0.10). That's +3 TP and +3 FP here, with no training needed. Confirm on new data before adopting it; it was chosen on development frames.
2. **CNG:** a CNG class would help only through routing (+2 TP, −2 FP even with perfect classification). A single fine-tuned plate detector with green CNG plates and painted bumper numbers (as negatives) in its training data removes the need for routing. This is the bigger lever.
3. **Rickshaw:** drop the `bicycle` class. It produced no true plates and only municipal-tag hits that the evaluation ignores. In deployment those hits would be false positives.
4. **Leguna:** none in the development sample. Include legunas deliberately when sampling training frames, and check whether COCO calls them `bus`, `truck` or `car`.
5. **When to fine-tune the vehicle detector:** when OCR starts. The plate-letter rule needs sedan / SUV / microbus / CNG / leguna / motorcycle / pickup / truck / bus / rickshaw. The fine-tuned plate detector and the local-class vehicle detector can share one labelling pass, with a vehicle box plus a plate box per vehicle.

## Limits

- 102 plates; CNG n=10, motorcycle n=25. One or two plates move the percentages by 1–2 points.
- Vehicle types and plate boxes are agent-judged, not human-adjudicated.
- Thresholds above were explored on development frames and are not frozen. The six-frame check must still use the frozen protocol.
