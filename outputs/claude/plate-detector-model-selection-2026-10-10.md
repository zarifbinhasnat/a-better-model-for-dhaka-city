# Plate detector: fine-tune or not, and which model

Prepared 10 October 2026 (Claude). Scope: number-plate **detection** only, for the fixed roadside camera in this workspace. It is a recommendation from web research plus the pilot results in `../progress.md`. No new model was trained or evaluated for this note.

## Short answer

1. **Fine-tune, yes. On our own footage, not only on public Bangla datasets.** The architecture is less important than the data. Public Bangla plate datasets are mostly close-up smartphone photos. Our frames are 4096×2824 elevated CCTV with small, dense plates. The pilot's misses are small motorcycle plates and green CNG plates. Its false positives are painted text and signs. Only labeled examples from this camera domain target those errors.
2. **Keep the cascade** (vehicle detector → original-resolution vehicle crop → plate detector). Full-frame inference collapsed recall in the pilot (32% international, 2% Bangla), while crops reached around 80%. Tiling was already tried and rejected. The vehicle crop fixes the small-object problem better than switching architectures does.
3. **Model choice, in order:**
   - **Primary: YOLO26s or YOLO11s fine-tuned in Ultralytics**, as a single `plate` class on vehicle crops at 640. Ultralytics is already installed (8.4.174), MPS training works on the M4, iteration is fast, and YOLO26 adds STAL small-target label assignment. This replaces the class-switch hybrid of two unrelated pretrained detectors with one model trained for this camera.
   - **Co-primary on Kaggle GPU: RF-DETR-S or RF-DETR-M (Apache-2.0 weights)**. It is state of the art on RF100-VL, the fine-tuning transfer benchmark, at 53.0 / 54.7 COCO AP. Train it on the same split as YOLO. Caveats: it is heavier (32M parameters for S), it needs COCO-JSON labels, and MPS training is not officially documented. The user has Kaggle access (T4/P100), so train it there rather than on the M4. RF-DETR-N, -S, -M and -L are Apache-2.0; Atto, Femto, Pico, XL and 2XL are under the more restrictive PML-1.0.
   - **Optional third: D-FINE or DEIM-D-FINE-S** (49.0 COCO AP at 10.2M parameters), if a lighter DETR-family baseline is wanted.
   - Do **not** choose by COCO mAP. No published benchmark isolates these models on small license plates. A 2026 road-user benchmark found RF-DETR best overall but YOLO11 better on small pedestrians and cyclists. The decision has to come from our own split, with recall broken down by plate pixel height.
4. **The vehicle stage needs attention too.** COCO has no CNG, rickshaw, leguna or easybike class. A missed vehicle proposal is a guaranteed missed plate. Measure proposal coverage of labeled plates first. If it is the bottleneck, fine-tune the vehicle detector on local classes. A 17-class Bangladeshi YOLOv8 vehicle project exists as a reference; its weights and license should be checked before reuse.

## Evidence

| Source | What it shows | Relevance to us |
| --- | --- | --- |
| [RF-DETR repository (ICLR 2026)](https://github.com/roboflow/rf-detr) | N/S/M/L at 384–704 px input; COCO AP 48.4–56.5; RF100-VL AP 57.7–62.2; `pip install rfdetr` | Strong fine-tuning transfer; input ≤704 means crops are still required for 4K frames |
| [RF-DETR fine-tuning cookbook](https://rfdetr.roboflow.com/latest/cookbooks/fine-tune_detection/) | COCO-JSON format; resolution 384–640 recommended, higher helps small objects; hundreds of images can suffice; GPU recommended | Labels must be exported to COCO; plan GPU time |
| [RF-DETR on Apple silicon (forum)](https://discuss.roboflow.com/t/can-rf-detr-run-with-mac-silicon/10580) | Unsupported MPS ops fall back to CPU | Local M4 training of RF-DETR is uncertain; YOLO is the safer local loop |
| [YOLO26 analysis (arXiv 2601.12882)](https://arxiv.org/pdf/2601.12882) | NMS-free end-to-end; YOLO26s 47.2 COCO AP at 9.5M params; RF-DETR-S 53.0, DEIM-D-FINE-S 49.0 | YOLO26s is lightest; accuracy gap on COCO may not hold on plates |
| [Roboflow: best detection models 2026](https://blog.roboflow.com/best-object-detection-models/) | YOLO26 STAL for small targets; RF-DETR leads RF100-VL | Vendor source (Roboflow builds RF-DETR); treat as claims |
| [MicroVision benchmark (arXiv 2603.18192)](https://arxiv.org/pdf/2603.18192) | RF-DETR best overall, YOLO11 better on small road users | Small-object ranking can flip; test ourselves |
| [Cascaded YOLO Bangladeshi LP (PLOS ONE 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13456284/) | YOLOv12 plate detector, 2,140 images, P 0.986 / R 0.965 / mAP50 0.963; mostly daylight smartphone and close-up shots | High numbers on easy close-up data; not comparable to CCTV |
| [YOLO-Transformer hybrid for Bangla LPR (ACM 2024)](https://dl.acm.org/doi/full/10.1145/3704522.3704557) | YOLOv5/v8/v10 around 96.5–96.9% on 40k images plus 60k synthetic; 75% Dhaka | Shows the YOLO family saturates on close-up data; dataset availability unclear |
| [Bangla LPR with YOLO + VLM OCR (arXiv 2603.10267)](https://arxiv.org/abs/2603.10267) | Two-stage adaptive YOLOv8 training; 97.83% accuracy reported | Preprint; dataset details not on abstract page |
| [Real-time Bangla LPR for video (WACV-W 2022)](https://arxiv.org/pdf/2108.08339) | Video-based Bangla detection and OCR | Closest video-domain precedent |
| Public data: [Kaggle sifatkhan69](https://www.kaggle.com/datasets/sifatkhan69/bangladeshi-vehicle-license-plate), [Kaggle syednahinhossain](https://www.kaggle.com/datasets/syednahinhossain/bangladeshi-license-plate-recognition-dataset), [Roboflow kaiser-durden (222 imgs, public domain)](https://universe.roboflow.com/kaiser-durden/bangladeshi-license-plate-fonfq/dataset/1), [Roboflow CSE499 (CC0)](https://universe.roboflow.com/bangla-license-plate-cse499/license-plate-detection-f1qxo), [synthetic plates](https://github.com/topics/bangla-license-plates) | Small, mostly close-up sets; none labels motorcycle or CNG plates specifically | Useful as auxiliary training data and for pre-training, not as our test set |
| [Bangladeshi vehicle detector (17 classes)](https://github.com/abuubidawadi/Bangladeshi_Vehicle_Detection) | YOLOv8 with CNG, rickshaw, leguna, easybike and motorbike classes | Candidate fix for COCO's missing local vehicle classes |

Venue and verification notes: the RF-DETR README and cookbook were read directly. The Kaggle pages did not render their descriptions, so their sizes and licenses are unverified. The PLOS paper's dataset release status is not stated. Vendor-authored comparisons are flagged above.

## Proposed experiment plan

**Step 0 — finish the frozen check first (already pending in `progress.md`).** Label the six fresh timestamps without overlays and score the frozen hybrid. This gives the pretrained baseline that any fine-tuned model must beat.

**Step 1 — build a training set from this footage.**
- Sample frames from time blocks **disjoint** from the 12 development and 6 check frames. Hold out at least one contiguous time block (for example, the last five minutes) as a locked test set. Stopped vehicles repeat, so split by time block, never by random frame.
- Pre-label with the current hybrid at a low threshold, then correct by hand. Mark these as model-assisted labels. Prioritize motorcycle, CNG and green plates, plus hard negatives: painted text, shop signs, bus route boards.
- Store labels as vehicle-crop images with YOLO txt, plus a COCO JSON export for RF-DETR. Keep crop→frame offsets so evaluation stays in original-frame coordinates with the existing `evaluate.py` matcher.
- Rough target: 1,500–3,000 labeled plate instances across at least 300 frames. This is a planning estimate, not a guarantee.

**Step 2 — train on the same split.**
- A: YOLO26s, 640, single class, from COCO weights; then a variant initialized from `bangla-yolov8.pt`.
- B: A plus public Bangla datasets mixed in (≤50% of samples).
- C: RF-DETR-S and RF-DETR-M at 512–640 on the same data, trained on a Kaggle GPU.
- Compute: YOLO quick iterations can run locally on MPS; for the final comparison, train YOLO and RF-DETR on the same Kaggle GPU so runtimes are comparable. Kaggle requires uploading footage-derived crops as a **private** Kaggle dataset. AGENTS.md requires explicit user authorization before footage leaves the machine, so get that confirmation before uploading.
- Tune thresholds on validation only; freeze; report on the locked test block.

**Step 3 — report.** Precision and recall at IoU 0.5 and a fixed confidence threshold, with ROI and sample size, broken down by plate height (<15, 15–30, >30 px), vehicle class and day/lighting. Report vehicle-proposal coverage separately. Compare against the frozen pretrained hybrid.

**Decision rule:** pick the model with the higher recall at an equal precision floor (for example ≥90%) on the locked block, then consider speed. Adopt RF-DETR only if it wins on our data by more than run-to-run variance; otherwise keep the simpler YOLO pipeline.

## Constraints that remain

- Current labels are provisional agent-drawn labels. A fine-tuned model inherits their errors; human review of the test block is needed before any public claim.
- One camera, one session: results will not generalize to other cameras until tested on them.
- No 85% or 95% claim follows from this note. Detection recall is not OCR accuracy.
