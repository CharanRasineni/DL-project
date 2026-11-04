"""Check Kaggle kernel status and download outputs."""
import json
import os
import sys
from pathlib import Path
import shutil

CRED = json.loads(Path(r"C:\Users\rshit\.kaggle\kaggle.json").read_text())
os.environ["KAGGLE_API_TOKEN"] = CRED["key"]

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernels = [
    "rscharan/fathomnet-03-fasterrcnn",
    "rscharan/fathomnet-04-yolov8s",
    "rscharan/fathomnet-05-rtdetr",
]

out_base = Path("C:/Users/rshit/OneDrive/Documents/Charan DL project/notebooks/kaggle/_kaggle_out")
out_base.mkdir(parents=True, exist_ok=True)

for ref in kernels:
    try:
        status = api.kernels_status(ref)
        print(f"{ref}: {status.status if hasattr(status, 'status') else status}")
        if hasattr(status, 'status'):
            print(f"  status: {status.status}")
            print(f"  version: {getattr(status, 'versionNumber', '?')}")
            print(f"  message: {getattr(status, 'message', '?')}")
    except Exception as e:
        print(f"{ref}: error checking status: {e}")

# Try to get kernel output log
print()
print("=== Kernel logs ===")
for ref in kernels:
    out_dir = out_base / ref.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(ref, str(out_dir))
        print(f"{ref}: output downloaded to {out_dir}")
    except Exception as e:
        print(f"{ref}: output error: {e}")


