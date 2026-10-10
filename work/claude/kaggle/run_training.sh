#!/bin/zsh
# Export human labels -> private Kaggle dataset -> GPU training job. Upload of footage-derived crops
# was authorized by the user on 2026-10-10 (see progress.md). Dataset stays PRIVATE (no --public).
set -euo pipefail
ROOT="${0:A:h}/../../.."
K="${KAGGLE:-$HOME/.local/bin/kaggle}"
PY="$ROOT/work/detection-pilot/.venv/bin/python"
DS="$ROOT/work/claude/kaggle/dhaka-plates-v1"
"$PY" "$ROOT/work/claude/labeling/export_dataset.py" "$DS"
if "$K" datasets status zarifb/dhaka-plates-v1 >/dev/null 2>&1; then
  "$K" datasets version -p "$DS" -r zip -m "labels at $(date +%F_%H%M)"
else
  "$K" datasets create -p "$DS" -r zip
fi
echo "Waiting for Kaggle to process the dataset..."
until "$K" datasets status zarifb/dhaka-plates-v1 2>/dev/null | grep -qi ready; do sleep 15; done
"$K" kernels push -p "$ROOT/work/claude/kaggle/train-yolo26s"
echo "Started. Check: $K kernels status zarifb/dhaka-plates-train-yolo26s"
