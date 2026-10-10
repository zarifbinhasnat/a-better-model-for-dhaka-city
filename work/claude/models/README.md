# Retained models (best only, 2026-10-10)

| File | Role in `outputs/claude/pipeline/cascade.py` | Evidence |
| --- | --- | --- |
| `plate_B_yolo26s_run1.pt` | Single-class plate detector on vehicle crops (YOLO26s fine-tuned on this video; Kaggle run 1, variant B) | dev12 95/15/7 (P 86.4%, R 93.1%) vs frozen hybrid 87/16/15; locked-test frame 16/2/3 — `outputs/claude/plate-finetune-run1-2026-10-10.md` |
| `vehicle_cls_run4b.pt` | Vehicle-type crop classifier: car / cng / motorcycle / rickshaw / bus_truck (YOLO26s-cls) | video val/test/pilot CNG P/R 100%; held-out photos CNG recall 9/10, 2/35 false CNG — `outputs/claude/cng-classifier-run1-2026-10-10.md` |

The vehicle detector is the pilot's COCO `work/detection-pilot/yolo26s.pt` (kept there, unchanged). The pilot baselines `bangla-yolov8.pt` and `international-yolov9.onnx` also stay, because the frozen check protocol needs them.

Superseded models (plate A, classifiers v1–v3, the failed full-frame vehicle detector, training-run folders) were moved to the macOS Trash on 2026-10-10. Earlier versions also remain in git history.

Retraining the classifier from scratch needs the ImageNet init `yolo26s-cls.pt`, which Ultralytics downloads automatically. The downloaded web photos were removed at the user's request, so classifier v4b cannot be retrained exactly. `work/claude/cng-classifier/dataset_v2` (footage only) plus `work/claude/web-cng/web_labels*.json` (boxes only) record what was used.
