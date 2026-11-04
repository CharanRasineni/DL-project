import json
from pathlib import Path

base = Path(".")
results = {
    "fasterrcnn": "03_fasterrcnn/output/results_fasterrcnn.json",
    "yolov8s_val": "03_fasterrcnn/output/preds_fasterrcnn.json",
    "fathomnet_coco": "03_fasterrcnn/output/val_gt.json",
    "yolov8s_results": "results_yolov8s.json",
    "rtdetr_results": "results_rtdetr.json",
}

for name, path in results.items():
    p = base / path
    if not p.exists():
        print(f"\n{path}: NOT FOUND")
        continue
    with open(p) as f:
        data = json.load(f)
    size = p.stat().st_size
    print(f"\n=== {path} ({size} bytes) ===")
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for k in ("mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
                   "head_mid_tail_AP50-95", "per_class_AP50-95", "n_dets", "n_preds"):
            if k in data:
                val = data[k]
                if isinstance(val, dict) and len(str(val)) > 200:
                    print(f"  {k}: (dict with {len(val)} entries) sample={list(val.items())[:2]}")
                else:
                    print(f"  {k}: {val}")
    elif isinstance(data, list):
        print(f"  List of {len(data)} items")
        if data:
            print(f"  First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"  First item keys: {list(data[0].keys())}")
                print(f"  First item: {data[0]}")
    else:
        print(f"  Type: {type(data)}, value: {data}")
