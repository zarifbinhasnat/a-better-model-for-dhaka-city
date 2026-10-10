# Plate detector fine-tune, run 1

Prepared 10 October 2026 (Claude). One fine-tuned single-class plate detector, run inside the existing cascade, compared with the frozen class-switch hybrid on the same frames and settings.

## Result

| Plate confidence 0.35, IoU ≥0.5, ROI `[0,500,3600,2824]` | Development: 12 frames, 102 plates | Locked test: 1 frame (1970 s), 19 plates |
| --- | --- | --- |
| Frozen hybrid (international + Bangla, class switch) | 87 / 16 / 15 → P 84.5%, R 85.3% | 13 / 2 / 6 → P 86.7%, R 68.4% |
| **B: YOLO26s fine-tuned (COCO init)** | **95 / 15 / 7 → P 86.4%, R 93.1%** | **16 / 2 / 3 → P 88.9%, R 84.2%** |
| A: Bangla YOLOv8 fine-tuned | 97 / 42 / 5 → P 69.8%, R 95.1% | 17 / 11 / 2 → P 60.7%, R 89.5% |

Values are TP / FP / FN. Other thresholds are in `work/claude/eval/results.json`. B was not tuned on these frames: 0.35 is the pilot's frozen threshold.

Per vehicle type for B on the development frames (any-match, Claude's visual type tags): car 59/60, microbus/pickup/truck 7/7, **CNG 9/10 (hybrid: 4/10)**, motorcycle 20/25 (hybrid: 17/25). Four of the five remaining motorcycle misses have no vehicle proposal at vehicle confidence 0.20. Lowering the motorcycle threshold to about 0.10 addresses those (see the vehicle-class audit).

**Candidate replacement:** B, a single model on all non-rickshaw vehicle crops. It replaces the two-detector class switch and its NMS-ordering caveat. Rickshaw crops are skipped (no registration).

## Training

- Kaggle T4, private dataset `zarifb/dhaka-plates-v1` and job `zarifb/dhaka-plates-train-yolo26s`, Ultralytics 8.4.174, image size 640, no horizontal flips, seed 0.
- Data: 16 labelled frames from this video, about 600 vehicle crops. 179 plate instances in training, 14 in validation (1700–1800 s block). Locked test ≥1950 s, never trained on.
- Early stopping on validation: A stopped at 31 epochs, B at 83.
- Weights: `work/claude/models/plate_B_yolo26s_run1.pt` and `plate_A_bangla_init_run1.pt`.

## Read before quoting these numbers

- **Labels are provisional.** The development labels are agent-drawn. The training and locked-test labels are the user's labels repaired by Claude: model suggestions that Claude reviewed visually, plus the user's vehicle boxes. Nothing has been independently adjudicated.
- **Same camera and session.** Training frames sit at least 10 s away from every development timestamp, but stopped vehicles can recur across nearby times, which can flatter the fine-tuned model. This is not an independent vehicle or camera test.
- **The locked test is one frame** (19 plates). Its labels started from the baseline models' suggestions, which biases the comparison toward the baseline.
- **Small samples:** CNG n=10 and motorcycle n=25 on development. One plate moves the per-type numbers by 4–10 points.
- This is detection only. It is not OCR accuracy, and it supports no 95% claim.

## Next

1. Label the rest of the locked-test block (8 frames) in labeller v2. Score A and B once there, with frozen settings.
2. Label more training frames in plate mode (reviewing suggestions) and add legunas, then retrain. Repeat run 1's settings so the comparison is controlled.
3. Add the motorcycle-only vehicle threshold of 0.10 as a separate, pre-declared variant.
4. Get human review of the development and test labels before any external claim.
