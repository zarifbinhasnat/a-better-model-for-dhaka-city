"""Kaggle GPU job: fine-tune a single-class plate detector on Dhaka vehicle crops.
Selection uses the val time-block only; the locked test split is NOT evaluated here."""
import glob, os, shutil, subprocess, sys, json
subprocess.run(["nvidia-smi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
import torch
assert torch.cuda.is_available(), "No GPU: check enable_gpu / phone verification / quota"

src = os.path.dirname(glob.glob("/kaggle/input/**/yolo/data.yaml", recursive=True)[0])
data = "/kaggle/working/data"
shutil.copytree(src, data, dirs_exist_ok=True)          # /kaggle/input is read-only; YOLO writes label caches
with open(f"{data}/data.yaml", "w") as f:
    f.write(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")   # no test key: test stays locked

from ultralytics import YOLO
model = YOLO("yolo26s.pt")                               # COCO-pretrained; downloaded via internet
model.train(data=f"{data}/data.yaml", imgsz=640, epochs=120, patience=30, batch=32, device=0, seed=0,
            deterministic=True, cos_lr=True, close_mosaic=15, fliplr=0.0,  # no horizontal flip: Bangla text is not mirror-symmetric
            project="/kaggle/working/runs", name="yolo26s_plate", exist_ok=True, plots=True)
run = "/kaggle/working/runs/yolo26s_plate"
shutil.copy(f"{run}/weights/best.pt", "/kaggle/working/best.pt")
m = YOLO("/kaggle/working/best.pt").val(data=f"{data}/data.yaml", split="val", imgsz=640, conf=0.001, plots=False)
json.dump({"val_map50": float(m.box.map50), "val_map50_95": float(m.box.map), "val_precision": float(m.box.mp),
           "val_recall": float(m.box.mr)}, open("/kaggle/working/val_summary.json", "w"), indent=1)
shutil.rmtree(data)                                      # don't re-export the images as kernel output
