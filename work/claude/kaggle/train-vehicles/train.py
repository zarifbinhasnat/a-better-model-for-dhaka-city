"""Kaggle GPU job: fine-tune YOLO26s (COCO init) into a local-class vehicle detector:
car, motorcycle, bus, truck, cng, rickshaw, leguna. Full frames at 1280. Val = 1700-1800 s block only."""
import glob, os, shutil, subprocess, sys, json
subprocess.run(["nvidia-smi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
import torch
assert torch.cuda.is_available(), "No GPU"
root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/yolo/data.yaml", recursive=True)[0]))
data = "/kaggle/working/data"
shutil.copytree(f"{root}/yolo", data, dirs_exist_ok=True)
names = ["car", "motorcycle", "bus", "truck", "cng", "rickshaw", "leguna"]
with open(f"{data}/data.yaml", "w") as f:
    f.write(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
from ultralytics import YOLO
YOLO(f"{root}/weights/yolo26s.pt").train(data=f"{data}/data.yaml", imgsz=1280, epochs=150, patience=40, batch=8, device=0,
    seed=0, deterministic=True, cos_lr=True, close_mosaic=20, fliplr=0.5, project="/kaggle/working/runs", name="vehicles",
    exist_ok=True, plots=True)
best = "/kaggle/working/runs/vehicles/weights/best.pt"
shutil.copy(best, "/kaggle/working/vehicles_yolo26s_local.pt")
m = YOLO(best).val(data=f"{data}/data.yaml", split="val", imgsz=1280, plots=False)
json.dump({"val_map50": round(float(m.box.map50), 4), "val_map50_95": round(float(m.box.map), 4),
           "per_class_map50": {names[int(c)]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.ap50)}},
          open("/kaggle/working/val_summary.json", "w"), indent=1)
shutil.rmtree(data)
