"""Kaggle GPU job: fine-tune single-class plate detectors on Dhaka vehicle crops, same split for all variants.
  A: init from the Bangla YOLOv8 plate model (already knows plates)
  B: init from COCO YOLO26s
Selection uses the val time-block only; the locked test split is NOT evaluated here."""
import glob, os, shutil, subprocess, sys, json
subprocess.run(["nvidia-smi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
import torch
assert torch.cuda.is_available(), "No GPU: check enable_gpu / phone verification / quota"

root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/yolo/data.yaml", recursive=True)[0]))
data = "/kaggle/working/data"
shutil.copytree(f"{root}/yolo", data, dirs_exist_ok=True)   # /kaggle/input is read-only; YOLO writes label caches
with open(f"{data}/data.yaml", "w") as f:
    f.write(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")   # no test key: test stays locked

from ultralytics import YOLO
variants = {"A_bangla_init": f"{root}/weights/bangla-yolov8.pt", "B_yolo26s_coco": "yolo26s.pt"}
summary = {}
for name, init in variants.items():
    YOLO(init).train(data=f"{data}/data.yaml", imgsz=640, epochs=120, patience=30, batch=32, device=0, seed=0,
                     deterministic=True, cos_lr=True, close_mosaic=15, fliplr=0.0,  # Bangla text is not mirror-symmetric
                     project="/kaggle/working/runs", name=name, exist_ok=True, plots=True, single_cls=True)
    best = f"/kaggle/working/runs/{name}/weights/best.pt"
    shutil.copy(best, f"/kaggle/working/{name}.pt")
    m = YOLO(best).val(data=f"{data}/data.yaml", split="val", imgsz=640, plots=False)
    summary[name] = {"init": os.path.basename(init), "val_map50": round(float(m.box.map50), 4),
                     "val_map50_95": round(float(m.box.map), 4), "val_precision": round(float(m.box.mp), 4),
                     "val_recall": round(float(m.box.mr), 4)}
    print(name, summary[name], flush=True)
json.dump(summary, open("/kaggle/working/val_summary.json", "w"), indent=1)
shutil.rmtree(data)                                      # don't re-export the images as kernel output
