"""Offline validation harness for the research-grade library in data.py.

Two jobs:
 1. run data.selftest() (pure-python assertions),
 2. check whether the archived Faster R-CNN run used the frozen split at all,
    and if it did, re-derive every metric with FullEvaluator and compare
    against the archived results_fasterrcnn.json.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

VAL_GT = HERE / "03_fasterrcnn" / "output" / "val_gt.json"
SPLIT = HERE / "02_split" / "output" / "split.json"
PREDS = HERE / "03_fasterrcnn" / "output" / "preds_fasterrcnn.json"
ARCH = HERE / "03_fasterrcnn" / "output" / "results_fasterrcnn.json"
TAB = HERE / "02_split" / "output" / "table_split_class_counts.csv"

val_gt = json.loads(VAL_GT.read_text())
split = json.loads(SPLIT.read_text())
arch = json.loads(ARCH.read_text())

val_gt_ids = {int(i["id"]) for i in val_gt["images"]}
split_val_ids = {int(i) for i in split["val_ids"]}
inter = val_gt_ids & split_val_ids
print("=== SPLIT INTEGRITY ===")
print(f"val_gt.json images     : {len(val_gt_ids)}")
print(f"split.json val_ids     : {len(split_val_ids)}")
print(f"intersection           : {len(inter)}")
print(f"val_gt ids IN split    : {len(inter)/max(len(val_gt_ids),1):.1%}")
print(f"split val NOT in val_gt: {len(split_val_ids - val_gt_ids)}")
print(f"val_gt NOT in split val: {len(val_gt_ids - split_val_ids)}")

import data as D  # noqa: E402

import logging  # noqa: E402

D.LOG.setLevel(logging.CRITICAL)  # keep stderr clean for PowerShell capture

print("\n=== data.selftest() ===")
checks = D.selftest(verbose=False)
print("selftest checks passed:", len(checks), sorted(checks))

# ---------------------------------------------------------------- evaluator
# Reproduce the archived Faster R-CNN metrics with FullEvaluator, using the
# archived val_gt.json as the "dataset" (it holds the val images+annotations),
# to prove the new code path is numerically correct.
os.environ["FATHOMNET_DATASET_JSON"] = str(VAL_GT)
os.environ["FATHOMNET_SPLIT_JSON"] = str(SPLIT)

import importlib  # noqa: E402

importlib.reload(D)
D.LOG.setLevel(logging.CRITICAL)

coco_data = json.loads(VAL_GT.read_text())
print("DEBUG DATASET_JSON =", D.DATASET_JSON)
print("DEBUG val_gt keys  =", list(coco_data.keys())[:8])
print("DEBUG loaded keys  =", list(D.load_coco_data().keys())[:8])
print("DEBUG cache keys   =", list(D._COCO_CACHE.keys()))
val_ids = sorted(val_gt_ids)
split_synth = {"train_ids": val_ids, "val_ids": val_ids, "val_fraction": 0.1, "seed": 42}
cat_info = D.get_cat_info(coco_data, split_synth)
preds = json.loads(PREDS.read_text())
print(f"\npreds_fasterrcnn.json: {len(preds)} detections")

ev = D.FullEvaluator(split_synth, cat_info, score_thr=0.01, max_det=300)
res = ev.evaluate(preds, bootstrap=0)

print("\n=== ARCHIVED vs RECOMPUTED (Faster R-CNN) ===")
for k in ("mAP50-95", "mAP50"):
    print(f"{k:10s} archived={arch[k]:.4f}  recomputed={res[k]:.4f}")
print(f"{'n_dets':10s} archived={arch.get('n_dets', 'n/a')}  "
      f"recomputed={res['preds_used']['n_dets']}")

a = arch["per_class_AP50-95"]
b = res["per_class_AP50-95"]
diffs = sorted(((abs(a[k] - b.get(k, 0.0)), k) for k in a), reverse=True)
print("\nper-class AP50-95: max |diff| =", round(diffs[0][0], 5))
print("worst 5:", [(k, round(a[k], 4), round(b.get(k, 0.0), 4)) for _, k in diffs[:5]])
n_exact = sum(1 for d, _ in diffs if d <= 1e-9)
print(f"classes matching exactly: {n_exact}/{len(a)}")
print("size AP (new):", {k: res[k] for k in ("mAP_small", "mAP_medium", "mAP_large")})
print("AP50 per-class available:", len(res["per_class_AP50"]))
print("size AP for urchin:", res["per_class_size_AP50-95"].get("urchin"))
perclass = arch["per_class_AP50-95"]
import csv  # noqa: E402

rows = list(csv.DictReader(TAB.open()))
counts = {r[""]: int(r["train"]) for r in rows}
g_split = D.define_groups(counts, head_k=3, mid_k=4)
print("\n=== GROUPS ===")
print("groups from frozen split.json counts:", {k: sorted(v) for k, v in g_split.items()})
print("head/mid/tail mean with these groups:",
      {k: round(float(sum(perclass[c] for c in v) / len(v)), 4)
       for k, v in g_split.items()})
print("archived results_fasterrcnn.json    :", arch["head_mid_tail_AP50-95"])

# alternative grouping implied by the 03 runtime log
alt_head = ["urchin", "bony fish", "sea fan"]
alt_mid = ["brittle star", "sponge", "anemone", "stony coral"]
alt_tail = [c for c in perclass if c not in alt_head + alt_mid]
print("\nalt head (from 03 log) mean:",
      round(sum(perclass[c] for c in alt_head) / 3, 4))
print("alt mid  mean:", round(sum(perclass[c] for c in alt_mid) / 4, 4))
print("alt tail mean:", round(sum(perclass[c] for c in alt_tail) / len(alt_tail), 4))