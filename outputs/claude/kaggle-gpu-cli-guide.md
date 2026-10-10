# Kaggle GPU from the command line — step by step

Prepared 10 October 2026 (Claude). How this project trains detectors on a free Kaggle GPU without using the browser notebook editor. The laptop (M4, no CUDA) prepares data and code; Kaggle runs the training; results come back with one command.

Status: **procedure, not yet executed.** Command syntax was checked against the official Kaggle CLI docs ([README](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md), [datasets](https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md), [kernels](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md), [kernel metadata](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md)) on this date. Kaggle changes its UI and quotas without notice, so trust what your account settings show over the numbers here.

```text
laptop                                   Kaggle
──────                                   ──────
crops + labels ──datasets create──▶      private Dataset  (/kaggle/input/...)
train.py + kernel-metadata.json
               ──kernels push──────▶     script runs on GPU T4 ×2
               ◀──kernels output───      /kaggle/working/  (weights, metrics, logs)
```

## 0. Before anything: data permission

`AGENTS.md` requires the user's explicit authorization before source footage or anything derived from it leaves this machine. Vehicle crops and labels are derived data. Record the authorization in `progress.md` before step 4. Uploads are **private by default**; never add `--public`.

## 1. Account (one time, in the browser)

1. Create or sign in to an account at <https://www.kaggle.com>.
2. **Verify your phone number.** Open your profile picture → Settings → the phone verification section. Kaggle sends an SMS code. Without this, GPU and internet access stay disabled for notebooks ([reports](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)).
3. In the same Settings page, note your **GPU quota**. It is commonly reported as about 30 GPU-hours per week, and it resets weekly. A single run is limited to about 12 hours. Both numbers come from third-party reports; the official docs don't publish them.

## 2. Install the CLI on the laptop (one time)

The CLI needs Python ≥3.11. Install it as an isolated tool so it doesn't touch the project venv:

```bash
pipx install kaggle
```

If you don't have pipx, either of these works:

```bash
uv tool install kaggle
```

```bash
python3 -m pip install --user kaggle
```

Check that it's on your PATH:

```bash
kaggle --version
```

## 3. Authenticate (one time)

Pick one of these:

- **Browser login (simplest):**
  ```bash
  kaggle auth login
  ```
- **API token:** Settings → API → *Generate New Token*. Save the token to `~/.kaggle/access_token` (this laptop already has that file), or export it for the current shell:
  ```bash
  export KAGGLE_API_TOKEN=paste_token_here
  ```
- **Legacy key:** Settings → API → *Create Legacy API Key* downloads `kaggle.json`. Move it to `~/.kaggle/kaggle.json`.

Lock the file down, and never put a token inside this repository:

```bash
chmod 600 ~/.kaggle/access_token
```

Verify that authentication works (this lists only your own notebooks):

```bash
kaggle kernels list --mine
```

## 4. Upload training data as a private Dataset

Data lives outside git (images are git-ignored). Suggested local layout:

```text
work/claude/kaggle/dhaka-plates-v1/
├── dataset-metadata.json
├── yolo/          images/{train,val,test}, labels/{train,val,test}, data.yaml
└── coco/          train/ valid/ test/  each with _annotations.coco.json (for RF-DETR)
```

Create the metadata file:

```bash
kaggle datasets init -p work/claude/kaggle/dhaka-plates-v1
```

Edit `dataset-metadata.json` so it reads:

```json
{
  "title": "dhaka-plates-v1",
  "id": "YOUR_KAGGLE_USERNAME/dhaka-plates-v1",
  "licenses": [{"name": "other"}]
}
```

Upload it. It is private by default, and `-r zip` keeps the folder structure:

```bash
kaggle datasets create -p work/claude/kaggle/dhaka-plates-v1 -r zip
```

Check processing status:

```bash
kaggle datasets status YOUR_KAGGLE_USERNAME/dhaka-plates-v1
```

After relabeling, upload a new version instead of creating a new dataset:

```bash
kaggle datasets version -p work/claude/kaggle/dhaka-plates-v1 -r zip -m "v2: added CNG and motorcycle labels"
```

Keep the **locked test time-block** in the dataset if needed for final scoring, but don't evaluate on it while tuning (see `progress.md` scientific constraints).

## 5. Write the training job ("kernel")

Code lives in git, under `work/claude/kaggle/train-yolo26s/` (one folder per experiment):

```text
train-yolo26s/
├── kernel-metadata.json
└── train.py
```

`kernel-metadata.json` turns the GPU on:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/dhaka-plates-train-yolo26s",
  "title": "dhaka-plates-train-yolo26s",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["YOUR_KAGGLE_USERNAME/dhaka-plates-v1"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- `enable_gpu: "true"` is required. Without it the job silently runs on CPU.
- `machine_shape: "NvidiaTeslaT4"` = 2× T4 (16 GB each). `enable_internet` is needed for `pip install` and downloading pretrained weights.
- The title must match the slug in `id` (lowercase, dashes).

`train.py` skeleton (YOLO26s). It prints the GPU first, so a CPU fallback is caught immediately:

```python
import glob, subprocess, sys, shutil
subprocess.run(["nvidia-smi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
import torch; assert torch.cuda.is_available(), "No GPU: check enable_gpu / phone verification"

# Mount paths have changed over time; find data.yaml instead of hard-coding it.
data_yaml = glob.glob("/kaggle/input/**/yolo/data.yaml", recursive=True)[0]

from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.train(data=data_yaml, imgsz=640, epochs=100, batch=32, device=[0, 1],
            project="/kaggle/working/runs", name="yolo26s", seed=0)
shutil.copy("/kaggle/working/runs/yolo26s/weights/best.pt", "/kaggle/working/best.pt")
```

The RF-DETR variant (separate folder, same metadata pattern) swaps the training block:

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rfdetr"])
coco_dir = glob.glob("/kaggle/input/**/coco", recursive=True)[0]
from rfdetr import RFDETRSmall
RFDETRSmall().train(dataset_dir=coco_dir, epochs=50, batch_size=8,
                    grad_accum_steps=2, output_dir="/kaggle/working/rfdetr-s")
```

The data directory `/kaggle/input/` is read-only. Only `/kaggle/working/` is saved as output (about 20 GB limit), so write everything there.

## 6. Run it

Push the code. This uploads it and starts the run on the GPU:

```bash
kaggle kernels push -p work/claude/kaggle/train-yolo26s
```

Optionally pass a hard time limit in seconds, e.g. 6 hours:

```bash
kaggle kernels push -p work/claude/kaggle/train-yolo26s -t 21600
```

Check progress (`running`, `complete` or `error`):

```bash
kaggle kernels status YOUR_KAGGLE_USERNAME/dhaka-plates-train-yolo26s
```

Live logs are visible on the kernel page in the browser (*Your Work → Code*). A job pushed from the CLI keeps running after the laptop sleeps.

## 7. Get results back

Download outputs into a per-run folder:

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/dhaka-plates-train-yolo26s -p work/claude/kaggle/results/yolo26s-run1
```

Download only the weights and CSVs:

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/dhaka-plates-train-yolo26s -p work/claude/kaggle/results/yolo26s-run1 --file-pattern ".*(best\.pt|\.csv)$"
```

Then score locally with the project's evaluator on original-frame coordinates, not Kaggle's mAP printout, so the numbers stay comparable with `outputs/detection-pilot/metrics.csv`. Record the kernel slug, dataset version, commit hash and settings in `progress.md` for every run.

## 8. Rules for this project

- **Same split, same GPU** for every model compared (YOLO26s vs RF-DETR-S/M), with a fixed `seed`. Record which Kaggle dataset version each run used.
- Tune thresholds on the validation split only. Freeze them before scoring the locked test block.
- Stop interactive browser sessions when idle; they burn quota. CLI script runs stop by themselves when finished.
- Never commit tokens. Commit code and metadata; the user has approved committing model weights, but footage-derived images stay out of git.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `401 Unauthorized` | Token missing, expired or wrong file; rerun `kaggle auth login` |
| `assert torch.cuda.is_available()` fails | `enable_gpu` not `"true"`, phone not verified, or weekly quota used up |
| `pip install` fails inside the job | `enable_internet` not `"true"` (also requires phone verification) |
| `data.yaml` not found | Dataset still processing, or slug in `dataset_sources` is wrong; run `kaggle datasets status` |
| Push rejected / title mismatch | `title` must slugify to the name in `id` |
| Run stops at about 12 h | Session limit; lower epochs or resume from `last.pt` in a follow-up run |
