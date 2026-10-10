# Shared progress

Last updated: 2026-10-10 (end of session), Claude. Read this file before starting; update it before ending every session.

## Current scope

Number-plate **detection only**, aiming toward >85% recall and eventually 95%, while measuring precision too. For the current source video only (`2025-12-31-111331_2025-12-31-131239_1000004$1$0$0.mp4`), exclude its first 900 seconds (protest); this is a per-file exclusion, not a rule for other footage. Rickshaws are tracked as vehicles with no registration plate expected. OCR, red-light/helmet violations, tracking and full training are outside the current pilot. The target is not yet validated as a production result.

Workspace: `/Volumes/Projects/A better model for Dhaka City`

This file is canonical at `outputs/progress.md`; root `progress.md` is a symlink to it.

## Ownership / coordination

| Contributor | Status | Task / owned paths | Next handoff |
| --- | --- | --- | --- |
| Codex | Architecture handoff complete; pilot evaluation pending | `outputs/ARCHITECTURE.md` and existing pilot; documentation updated | Coordinate next owner for six-frame labels, frozen-hybrid evaluation and report/demo |
| Claude (Claude Code) | Session ended 2026-10-10: best pipeline in place; workspace cleaned (only best models kept, downloaded photos removed) | `outputs/claude/`, `work/claude/` | Next owner: see **What to do next** below |
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
- Kaggle GPU command-line procedure (account, phone verification, CLI auth, private dataset upload, GPU kernel push/status/output): `outputs/claude/kaggle-gpu-cli-guide.md`. Used for the plate and vehicle runs on 2026-10-10.
- Vehicle-class audit (dev diagnostic): `outputs/claude/vehicle-class-audit-2026-10-10.md`. COCO vehicle proposals cover 96% of labelled plates; vehicle-stage changes recover ≤5 of 15 misses. Plate detector is the bottleneck (CNG 4/10, motorcycle 17/25 recall). Vehicle fine-tune deferred until OCR needs local classes. Suggested (unfrozen): motorcycle vehicle conf ≈0.10, drop `bicycle`.
- Bangladesh plate class-letter rules (vehicle type → allowed letters, parser, validator, OCR constraint that abstains rather than invents): `outputs/claude/plate_rules/`; 9 unit tests pass. Table is from Wikipedia + Bangla explainers; no official BRTA table online — replace with gazette when found. Sedan ≠ ঘ (ঘ = jeep/SUV).
- **User decisions 2026-10-10:** (a) the 900 s skip applies only to the current source video, not to other footage (AGENTS.md, ARCHITECTURE.md, `run_detection.py --skip-seconds` updated); (b) rickshaws stay detected/tracked, reported as `rickshaw` with no registration expected (`plate_rules.vehicle_record`); (c) **uploading footage-derived vehicle crops to a private Kaggle dataset is authorized**; (d) user will label training frames in a local UI.
- Fine-tune data pipeline: 128 new frames from this video (`work/claude/labeling/extract_frames.py`, manifest): 119 train frames every 5 s in 905–1945 s avoiding ±10 s around the 18 dev/check timestamps, plus a locked test block of 9 frames ≥1950 s. Model suggestions (`prelabel.py`, unverified) → local labelling UI (`server.py`, `index.html`, guide `LABELING.md`; human labels in `labels/`). `export_dataset.py` makes vehicle-crop YOLO + COCO datasets (val = 1700–1800 s block; test = locked block). Kaggle job `work/claude/kaggle/train-yolo26s/` (YOLO26s, 640, no flips) launched via `run_training.sh`. Kaggle CLI 2.2.4 installed (pipx), auth OK as `zarifb`. Export dry-run verified on a temporary label (deleted). Later the same day: user labelled 17 frames, labels repaired, runs completed (see the current pipeline section and the session log).
- Workspace is a private Git repository on GitHub (see README). Model weights are committed (user-approved); footage, extracted frames, JPG views and the venv are git-ignored. On 2026-10-10 the working tree was reduced to the best models only (`work/claude/models/README.md`); superseded weights remain in git history and the macOS Trash.

## Current best pipeline (2026-10-10, Claude) — development-stage

`outputs/claude/pipeline/cascade.py`:
1. COCO YOLO26s vehicle proposals (`work/detection-pilot/yolo26s.pt`, imgsz 1280, conf 0.20, **class-agnostic NMS**).
2. Vehicle-type classifier `work/claude/models/vehicle_cls_run4b.pt` (car / cng / motorcycle / rickshaw / bus_truck). No leguna on this route.
3. Rickshaw → reported as a vehicle with no registration plate, no plate search.
4. Plate detector `work/claude/models/plate_B_yolo26s_run1.pt` (YOLO26s fine-tuned on this video), conf 0.35.
5. CNG colour: green = commercial, letter থ; silver = private, letter দ; unknown keeps both. Each plate gets its vehicle's allowed BRTA letters (`outputs/claude/plate_rules/`, 11 tests pass). Detection only: no OCR, no tracking.

| Plate detection, conf 0.35, IoU 0.5, ROI `[0,500,3600,2824]` | Dev12 (102 plates) | Locked-test frame 1970 (19 plates) |
| --- | --- | --- |
| Frozen class-switch hybrid (pilot) | 87/16/15 — P 84.5%, R 85.3% | 13/2/6 — P 86.7%, R 68.4% |
| **Current pipeline** | **95/15/7 — P 86.4%, R 93.1%** | **16/2/3 — P 88.9%, R 84.2%** |

- **CNG plates on dev12:** 9/10 (hybrid 4/10). Motorcycle plates: 20/25.
- **Vehicle classifier:** CNG P/R 100% on video validation, locked test and pilot hard cases. On held-out downloaded photos: CNG recall 9/10, 2 false CNGs out of 35.
- **CNG colour:** 0 green↔silver confusions on all available data (tuned on it; no untouched set).
- **Caveats:** provisional labels (agent-drawn or agent-repaired); single camera and session, where stopped vehicles can recur near training times; the locked test is one frame. No 85%/95% production claim.
- **Reports:** `outputs/claude/plate-finetune-run1-2026-10-10.md`, `outputs/claude/cng-classifier-run1-2026-10-10.md`, `outputs/claude/vehicle-class-audit-2026-10-10.md`.

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

## What to do next

Ordered by value. Items 1–3 strengthen the evidence; 4–7 improve the models.

1. **Label the 8 remaining locked-test frames** (≥1950 s) in labeller v2 (`python3 work/claude/labeling/server.py`, then http://localhost:8765). Use **plate mode** (`P`), review every dashed suggestion, and box rickshaws in vehicle mode (`V`). Then score with `work/claude/eval/eval_cascade.py`. This turns the one-frame test into a real held-out check.
2. **Pilot frozen check (still pending from Codex):** label the six fresh timestamps 975, 1155, 1335, 1515, 1695, 1875 s without model overlays. Score the frozen hybrid with `outputs/detection-pilot/check_protocol.json` (conf 0.35, IoU 0.5, same ROI; do not tune), and also score the current pipeline on the same frames.
3. **Human review of labels:** dev12 and the locked test are agent-drawn or agent-repaired. Have a person check them before any external claim.
4. **Motorcycle recall:** pre-declare and test vehicle conf 0.10 for motorcycles only. Four of the five dev12 motorcycle misses have no vehicle box at 0.20.
5. **More plate training data:** label more of the 119 training frames in plate mode, then retrain the plate detector with run 1's settings on Kaggle (`work/claude/kaggle/run_training.sh`). Compare against `plate_B_yolo26s_run1.pt` on the same locked test.
6. **Classifier gaps:** microbus (HiAce) vs CNG, bus/truck vs car (few examples), and silver CNGs from this camera. The best source is frames from this footage, not stock aerial photos, which hurt silver recall (v4). The downloaded photos were deleted at the user's request, so v4b cannot be retrained exactly.
7. **Next modules:** tracking (ByteTrack, already optional in `run_detection.py`, unvalidated), then Bangla OCR constrained by `plate_rules` (letters from vehicle type and CNG colour; abstain rather than invent).

## How to resume

Read `AGENTS.md`, then `outputs/ARCHITECTURE.md` for architecture, concepts, exact hybrid ordering, evaluation and bounded teammate tasks; read `collaboration/codex/HANDOFF.md` for runtime and source details. From the workspace root:

```sh
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python outputs/detection-pilot/run_detection.py --device cpu
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python outputs/detection-pilot/evaluate.py
```

Current pipeline on any image(s) (writes JSON and annotated JPGs):

```sh
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python outputs/claude/pipeline/cascade.py path/to/frame.jpg --out work/claude/demo
YOLO_CONFIG_DIR="$PWD/work/detection-pilot/config" work/detection-pilot/.venv/bin/python work/claude/eval/eval_cascade.py
```

The first pilot command resumes by filename and skips existing predictions. It is not needed unless new frames are added. The second overwrites baseline metrics from current development annotations; it does **not** calculate the hybrid or fresh-check results. Preserve current labels: old scratch annotation-generation scripts predate manual corrections and must not be rerun over the authoritative JSON.

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

- 2026-10-10 — Claude: user: **CNGs are green (commercial, common) or silver (private, rare)**. Found v1 classifier called white sedans/HiAce microbuses CNG (2 mislabelled training crops); built v2 (`work/claude/models/vehicle_cls_run2.pt`) with verified hard negatives + silver/black-canopy CNGs: CNG P/R 100%/100% on val, locked test and a pilot hard-case set (optimistic: same physical silver CNG in train). Added CNG colour rule (plate saturation + body green/brightness; conservative 'unknown') → `cng_green`→থ, `cng_silver`→দ in plate_rules (+tests, 11 pass); cascade now uses classifier v2 and reports `cng_colour`, `registration_class`, narrowed letters. Report updated: `outputs/claude/cng-classifier-run1-2026-10-10.md`.

- 2026-10-10 — Claude: marked the user's 9 downloaded CNG photos (`work/claude/web-cng/`, images local/git-ignored: third-party stock/news). Out-of-domain check of classifier v2: CNG recall 31/36 (silver → car/rickshaw). Fixed duplicate vehicle boxes with class-agnostic NMS (user-spotted; dev12 414→389 boxes, plates unchanged 95/15/7). CNG colour rule v2 + occlusion guard: 0 green↔silver confusions on video+photos (tuned on all data). Classifier v3 (`work/claude/models/vehicle_cls_run3.pt`, held-out photos 4/4 vs v2 2/4; locked test 97.1%); cascade now uses v3. Report: `outputs/claude/cng-classifier-run1-2026-10-10.md`.

- 2026-10-10 — Claude: marked photo batch 2 (10 stock images; local only): 42 CNG (6 silver), 88 car, 8 bus/truck, 3 rickshaw. v3 pre-training on batch 2: CNG recall 42/42, precision 42/56 (white/silver cars + buses in aerial shots). Colour green cutoff 6%→10% (private silver CNG with green accent); 0 green↔silver confusions on all data. Classifier v4b (holds out b2_10 + b2_02): held-out photos recall 9/10, false CNG 2/35 vs v3 10/10 and 6/35; video splits unchanged. Cascade uses v4b; end-to-end plates unchanged (95/15/7, 16/2/3).

- 2026-10-10 — Claude (end of session): at the user's request, moved to macOS Trash (`~/.Trash/dhaka-cleanup-20261010-233047`, recoverable until emptied): the 19 downloaded CNG photos (Downloads originals and project copies) plus all grids, sheets and crops made from them; classifier datasets v1/v3/v4/v4b; superseded models (plate A, classifiers v1–v3, failed vehicle detector, run folders, duplicate weight copies). Kept: `plate_B_yolo26s_run1.pt`, `vehicle_cls_run4b.pt`, the pilot `yolo26s.pt` (vehicle detector) and pilot baselines (bangla/international, needed for the frozen check), footage frames, footage-only classifier dataset v2, and all labels/JSON. Scripts repointed to v4b; `work/claude/models/README.md` added. Rewrote **What to do next**; added the **Current best pipeline** section.
