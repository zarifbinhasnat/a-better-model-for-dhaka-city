# Shared progress

Last updated: 2026-10-10, Claude. Read this file before starting; update it before ending every session.

## Current scope

Number-plate **detection only**, aiming toward >85% recall and eventually 95%, while measuring precision too. For the current source video only (`2025-12-31-111331_2025-12-31-131239_1000004$1$0$0.mp4`), exclude its first 900 seconds (protest); this is a per-file exclusion, not a rule for other footage. Rickshaws are tracked as vehicles with no registration plate expected. OCR, red-light/helmet violations, tracking and full training are outside the current pilot. The target is not yet validated as a production result.

Workspace: `/Volumes/Projects/A better model for Dhaka City`

This file is canonical at `outputs/progress.md`; root `progress.md` is a symlink to it.

## Ownership / coordination

| Contributor | Status | Task / owned paths | Next handoff |
| --- | --- | --- | --- |
| Codex | Architecture handoff complete; pilot evaluation pending | `outputs/ARCHITECTURE.md` and existing pilot; documentation updated | Coordinate next owner for six-frame labels, frozen-hybrid evaluation and report/demo |
| Claude (Claude Code) | Plate fine-tune run 1 + CNG crop classifier done; current pipeline `outputs/claude/pipeline/cascade.py` | `outputs/claude/`, `work/claude/` | Next: label 8 locked-test frames; microbus examples for classifier; motorcycle conf variant |
| Teammate / other agent | Available — write your name and task here | `collaboration/teammate/`, `work/teammate/`, `outputs/teammate/` | Record changes, evidence, blockers and files to integrate |

Before taking over the pilot, change the ownership row and record your task. Do not run simultaneous writers on annotations, predictions, metrics or the shared progress file. The entire workspace now lives on the Projects volume. Relative paths are preserved; the old workspace location is a compatibility symlink so existing environment launchers and this task still resolve.

## Completed and verified

- Literature review: `outputs/plate-recovery-literature-review-2026-10-10.md`.
- Source video: `/Users/md.zarifbinhasnat/Downloads/2025-12-31-111331_2025-12-31-131239_1000004$1$0$0.mp4`; 1,074,759,468 bytes, 4096×2824, approximately 33:23.55.
- Downloaded and ran YOLO26s vehicle proposals, international YOLOv9 plate detector (ONNX), and Bangla YOLOv8 plate detector (PyTorch). CPU runs, four threads; backend differences preclude a pure architecture speed comparison.
- Extracted 60 development/exploration frames from after this video's 900-second exclusion plus six fresh-check frames; prediction records exist for all 66.
- Compared each plate detector on full frames and original-resolution vehicle crops. Vehicles use 5% crop padding.
- Visually annotated/reviewed 102 plate appearances in 12 development frames. These are agent-drawn provisional labels; stopped vehicles repeat.
- Baseline results saved in `outputs/detection-pilot/metrics.csv` and `metrics.json`.
- Tried overlapping tiles; additional false positives did not justify selecting this approach.
- Created shared ownership folders, root README, root AGENTS.md and this session handoff. Existing executable paths remain unchanged.
- Plate-detector model selection research (fine-tune vs pretrained; YOLO26 vs RF-DETR vs D-FINE; Bangla datasets): `outputs/claude/plate-detector-model-selection-2026-10-10.md`. Recommendation: fine-tune on this footage's vehicle crops; YOLO26s (local/Kaggle) and RF-DETR-S/M (Kaggle GPU) on the same time-block split.
- Kaggle GPU command-line procedure (account, phone verification, CLI auth, private dataset upload, GPU kernel push/status/output): `outputs/claude/kaggle-gpu-cli-guide.md`. Not yet executed.
- Vehicle-class audit (dev diagnostic): `outputs/claude/vehicle-class-audit-2026-10-10.md`. COCO vehicle proposals cover 96% of labelled plates; vehicle-stage changes recover ≤5 of 15 misses. Plate detector is the bottleneck (CNG 4/10, motorcycle 17/25 recall). Vehicle fine-tune deferred until OCR needs local classes. Suggested (unfrozen): motorcycle vehicle conf ≈0.10, drop `bicycle`.
- Bangladesh plate class-letter rules (vehicle type → allowed letters, parser, validator, OCR constraint that abstains rather than invents): `outputs/claude/plate_rules/`; 9 unit tests pass. Table is from Wikipedia + Bangla explainers; no official BRTA table online — replace with gazette when found. Sedan ≠ ঘ (ঘ = jeep/SUV).
- **User decisions 2026-10-10:** (a) the 900 s skip applies only to the current source video, not to other footage (AGENTS.md, ARCHITECTURE.md, `run_detection.py --skip-seconds` updated); (b) rickshaws stay detected/tracked, reported as `rickshaw` with no registration expected (`plate_rules.vehicle_record`); (c) **uploading footage-derived vehicle crops to a private Kaggle dataset is authorized**; (d) user will label training frames in a local UI.
- Fine-tune data pipeline: 128 new frames from this video (`work/claude/labeling/extract_frames.py`, manifest): 119 train frames every 5 s in 905–1945 s avoiding ±10 s around the 18 dev/check timestamps, plus a locked test block of 9 frames ≥1950 s. Model suggestions (`prelabel.py`, unverified) → local labelling UI (`server.py`, `index.html`, guide `LABELING.md`; human labels in `labels/`). `export_dataset.py` makes vehicle-crop YOLO + COCO datasets (val = 1700–1800 s block; test = locked block). Kaggle job `work/claude/kaggle/train-yolo26s/` (YOLO26s, 640, no flips) launched via `run_training.sh`. Kaggle CLI 2.2.4 installed (pipx), auth OK as `zarifb`. Export dry-run verified on a temporary label (deleted). No training run yet; no human labels yet.
- Workspace is a private Git repository on GitHub (see README). Model weights are committed (user-approved); footage, extracted frames, JPG views and the venv are git-ignored.

## Current results — development sample only

ROI: `[0,500,3600,2824]`; matching IoU ≥0.5. Baseline confidence ≥0.25.

| Method | TP / FP / FN | Precision | Recall |
| --- | --- | --- | --- |
| International, full frame | 33 / 2 / 69 | 94.3% | 32.4% |
| International, vehicle crops | 82 / 15 / 20 | 84.5% | 80.4% |
| Bangla, full frame | 2 / 16 / 100 | 11.1% | 2.0% |
| Bangla, vehicle crops | 83 / 27 / 19 | 75.5% | 81.4% |
| Selected class-switch hybrid, confidence ≥0.35 | 87 / 16 / 15 | 84.5% | 85.3% |

The hybrid result is a development diagnostic, not yet part of the baseline metrics CSV. It takes Bangla predictions from motorcycle/bicycle proposals and international predictions from other vehicle proposals, then global NMS at IoU 0.5. Settings were selected on development frames; this is not an unbiased test result. A naive union raised recall but produced too many false positives.

## Immediate next steps

1. Finish provisional labels for fresh timestamps **975, 1155, 1335, 1515, 1695, 1875 seconds** without viewing their model overlays. Images are in `outputs/detection-pilot/frames/`; enlarged grid views are in `work/detection-pilot/label-views/check_*`. There is **no completed check annotation JSON yet**.
2. Keep the frozen settings in `outputs/detection-pilot/check_protocol.json`: class-switch hybrid, confidence 0.35, matching IoU 0.5, same ROI. Evaluate once labels are ready; do not tune thresholds on these frames.
3. Save check annotations and check metrics separately from development data. Review ambiguous physical plates and document ignores.
4. Add reusable hybrid selection and validate matching/NMS behavior with small meaningful checks.
5. Produce a local comparison report and annotated preview/demo, with precision/recall and honest limits. No independent >85% or 95% claim yet.
6. Update this file with final metrics, output links and the next task before ending the session.

## How to resume

Read `AGENTS.md`, then `outputs/ARCHITECTURE.md` for architecture, concepts, exact hybrid ordering, evaluation and bounded teammate tasks; read `collaboration/codex/HANDOFF.md` for runtime and source details. From the workspace root:

```sh
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python outputs/detection-pilot/run_detection.py --device cpu
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python outputs/detection-pilot/evaluate.py
```

The first command resumes by filename and skips existing predictions. It is not needed unless new frames are added. The second overwrites baseline metrics from current development annotations; it does **not** calculate the hybrid or fresh-check results. Preserve current labels: old scratch annotation-generation scripts predate manual corrections and must not be rerun over the authoritative JSON.

## Teammate updates

Add your session entry here (date, task, owned files, completed work, checks, next steps, blockers). Preserve prior entries and Codex's ownership notes.

## Session log

- 2026-10-10 — Codex: completed the two-model crop/full-frame pilot and development evaluation; six fresh-check inference runs completed, check labeling/evaluation still pending. Set up the requested shared structure and persistent end-of-session handoff instructions. No training, OCR, tracking or violation evaluation completed.

- 2026-10-10 — Codex: moved the entire workspace to `/Volumes/Projects/A better model for Dhaka City` at the user’s request. Verified all file sizes and file symlink targets survived the move; retained a compatibility symlink at the old task path. Updated active workspace/config paths. Source footage remains in Downloads. Pilot evaluation remains pending as recorded above.

- 2026-10-10 — Codex: added `outputs/ARCHITECTURE.md` with implemented-versus-proposed status, pipeline diagrams, exact detector/crop settings, prediction contracts, frozen hybrid pseudocode and NMS-ordering caveat, metrics/label limitations, future tracking/OCR/violation boundaries, and bounded teammate tasks. Cross-checked against run_detection.py, evaluate.py, calibrate.py, manifests and saved metrics. Updated README and technical handoff links. No inference, label or metric files changed. Next: claim and finish the six-frame check; no new blocker.

- 2026-10-10 — Claude: web research on plate-detector choice and fine-tuning; wrote `outputs/claude/plate-detector-model-selection-2026-10-10.md`. User confirmed Kaggle GPU access (no local CUDA) and approved committing model weights. Initialized Git, rewrote `.gitignore` (weights in; footage-derived images, venv, tool installs out), pushed to a private GitHub repo. No inference, labels or metrics changed. Next: six-frame check labels → frozen-hybrid check score → disjoint time-block training set → YOLO26s vs RF-DETR fine-tune. Blocker: uploading footage-derived crops to Kaggle needs explicit user authorization.

- 2026-10-10 — Claude: vehicle-class audit on 12 dev frames (coverage, per-type recall, FP causes, vehicle conf/class/CNG-oracle end-to-end variants; baseline reproduced 87/16/15). Wrote `outputs/claude/vehicle-class-audit-2026-10-10.md`, scripts and caches in `work/claude/vehicle-audit/` (contact-sheet JPGs local only). Added `outputs/claude/plate_rules/` (BD class letters) with tests. No shared annotations, predictions, metrics or frozen protocol changed. Next: six-frame check labels → frozen score → plate-detector training set (prioritise CNG, motorcycles, legunas, painted bumper numbers as negatives).

- 2026-10-10 — Claude: scoped 900 s exclusion to this video; rickshaw = tracked, no registration; built frame extraction, model-suggestion pre-labelling, local labelling UI (verified in browser: render, Tab-zoom, type change, draw, save JSON in original pixels), dataset export (dry-run verified), Kaggle training job and launcher. Next: user labels frames → run `work/claude/kaggle/run_training.sh` → evaluate fine-tuned cascade on dev frames + locked test vs frozen hybrid. Six-frame check labels still pending.

- 2026-10-10 — Claude: user labelled 17 frames (16 train, 1 locked test). Finding: user kept 270/271 plate suggestions unreviewed and drew WHOLE-VEHICLE boxes (39 CNG, 40 motorcycle, 36 rickshaw/non-motorised as 'other'). Claude reviewed every kept suggestion and proposals inside user vehicle boxes (`work/claude/label-repair/`): 159 plates kept (13 relabelled CNG), 64 rejected (signs, banner, painted CNG text, stickers), 47 → ignore, 10 proposals accepted; vehicles without a confirmed plate → ignore. Repaired labels in `work/claude/labeling/labels_repaired/` (originals untouched; agent-reviewed, provisional). Labeller v2: explicit Plate/Vehicle modes, rickshaw vehicle-only, unreviewed suggestions saved as ignore, repaired labels shown for checking (verified in browser). Kaggle (private): dataset `zarifb/dhaka-plates-v1` (739 crops; train 179 / val 14 / test 29 plate instances) → job `dhaka-plates-train-yolo26s` (variant A init Bangla YOLOv8, B init COCO YOLO26s); dataset `zarifb/dhaka-vehicles-v1` (user vehicle boxes + COCO pseudo-labels ≥0.4; classes car/motorcycle/bus/truck/cng/rickshaw/leguna) → job `dhaka-vehicles-train-yolo26s`. Evaluator `work/claude/eval/eval_finetuned.py` reproduces frozen hybrid 87/16/15 on dev12; locked-test frame 1970 baseline 13/2/6 at 0.35 (labels derived from model suggestions → biased toward baseline). Caveats: tiny data (14 train frames), same session, agent-reviewed labels.

- 2026-10-10 — Claude: **plate fine-tune run 1 complete** (`outputs/claude/plate-finetune-run1-2026-10-10.md`). In cascade at frozen conf 0.35: YOLO26s fine-tuned (B) dev12 95/15/7 (P 86.4%, R 93.1%) vs frozen hybrid 87/16/15 (P 84.5%, R 85.3%); locked-test frame 1970 16/2/3 vs 13/2/6. CNG 9/10 vs 4/10; motorcycle 20/25 vs 17/25. Bangla-init (A) higher recall but P ~70%. Weights `work/claude/models/plate_B_yolo26s_run1.pt`. Caveats: provisional labels, same session (possible stopped-vehicle overlap), 1 locked-test frame. Vehicle local-class job queued/running on Kaggle.

- 2026-10-10 — Claude: vehicle local-class detector run 1 **failed** (best epoch 4; CNG val mAP50 0.0; 0 CNG/rickshaw predictions on locked-test frame). Not adopted; details and proposed crop-classifier run 2 appended to `outputs/claude/plate-finetune-run1-2026-10-10.md`. Weights kept locally only (`work/claude/kaggle/results/vehicles-run1/`, git-ignored).

- 2026-10-10 — Claude: user: **no legunas on this route** → leguna removed from labeller, vehicle classes and COCO type unions (BRTA letter table unchanged). Built **vehicle-type crop classifier** (`outputs/claude/cng-classifier-run1-2026-10-10.md`): 526 train crops (213 CNG via user boxes + green-filter mining verified by Claude); YOLO26s-cls on MPS. CNG recall 100% val (24/24) and locked test (32/32); CNG precision 100% / 94.1% (2 Toyota HiAce microbuses → CNG). Added `allowed_letters(..., source='classifier')` (+test, 11 pass). New end-to-end `outputs/claude/pipeline/cascade.py` (vehicles → classifier → rickshaw no-reg → plate model B → plate owner + allowed letters); demo JSON in `outputs/claude/pipeline/demo/` (annotated JPGs local only).
