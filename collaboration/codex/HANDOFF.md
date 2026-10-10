# Technical handoff — 2026-10-10

The authoritative status is ../../progress.md. Read ../../outputs/ARCHITECTURE.md for the full architecture, concepts, data contracts and teammate tasks. The workspace moved to `/Volumes/Projects/A better model for Dhaka City`; relative paths remain unchanged.

## Runtime and models

- Python: `work/detection-pilot/.venv/bin/python`; ultralytics 8.4.174, torch 2.14.1, torchvision 0.29.1, onnxruntime 1.31.0.
- Host: M4 MacBook Air, 16 GB. Pilot used CPU, four threads; MPS available but not used in reported runs.
- ffmpeg: imageio_ffmpeg installed under `work/video-tools`; use `PYTHONPATH=work/video-tools` with the bundled base Python to locate its executable.
- YOLO26s vehicle checkpoint: `work/detection-pilot/yolo26s.pt`.
- International: `work/detection-pilot/international-yolov9.onnx`, https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx . Input [1,3,608,608]; output [N,7] includes embedded NMS. Implementation remaps letterboxed coordinates.
- Bangla: `work/detection-pilot/bangla-yolov8.pt`, https://huggingface.co/Y3454R/bengali-alpr/resolve/main/models/yolo.pt ; upstream https://github.com/renzhamin/bengali-alpr .
- Vehicle classes: bicycle, car, motorcycle, bus, truck. COCO has no CNG/autorickshaw class. Do not assume vehicle checkpoint has plate labels.

## Authoritative files and caveats

- `outputs/detection-pilot/annotations.json`: final reviewed development labels, 102 appearances, 12 frames. Includes review log and ignore rules. Do not overwrite from scratch generators.
- `predictions.jsonl`: 66 frame records containing original-coordinate boxes, source vehicle indices, confidence, timings and backend metadata. No tracking was run.
- `sample_manifest.json`: original 60-frame sample; needs a separate check manifest or explicit six-frame update.
- `check_protocol.json`: frozen hybrid settings. Check annotations/metrics still missing.
- `run_detection.py`: executable comparison pipeline and coordinate/NMS helpers. Optional tracking code exists but was not used or validated.
- `evaluate.py`: greedy confidence-ranked one-to-one matching. Prediction centers must fall within ROI; unmatched predictions inside ignore boxes are ignored. Baseline CSV contains confidence 0.25 and 0.5 results.
- `run_tiles.py`, `tile_predictions.jsonl`: rejected tiling alternative, not selected architecture.
- Raw footage never uploaded. Source path contains a literal dollar sign; quote shell paths safely.

## Fresh check labeling

Grid views: `work/detection-pilot/label-views/check_mid_NNNN.jpg` and `check_near_NNNN.jpg`. Confirm actual filenames before loading.

Mid views: original crop (0,500,3600,1400), displayed 1800×450; multiply display coordinates by 2 and add 500 to y.
Near views: original crop (0,1400,3600,2824), displayed 1800×712; multiply by 2 and add 1400 to y.

Some 975/1155/1335 views were inspected, but no persisted check labels exist. Start fresh visual annotation if necessary; do not treat informal unfinished observations as truth. Finish all six check frames, without model overlays, before calculating check performance. The same stopped vehicles can occur in both development and check frames.

## Development labeling corrections already applied

- 1830/1920 CNG boxes corrected from vent to plate [1582,986,1624,1016].
- 1380 ambiguous green panel moved to ignore.
- 1920 blurred motorcycle box tightened.

Main misses involve small motorcycle plates and green CNG plates. Painted text, signs and poor localization create false positives. Physical unreadable plates can count as detection targets; ambiguous, heavily occluded/truncated plates and municipal bicycle-rickshaw tags are ignored per annotation policy.
