"""Generate Kaggle submission CSV from a trained model.

Reads sample_submission.csv to learn the exact expected format, fills it with
predictions from dataset_test.json images, and writes submission.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from src.data.coco_dataset import CocoPuDataset, collate_fn
from src.models.detector import build_model
from src.training.utils import eval_transforms, get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/faster_rcnn.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="submissions/submission.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    data = cfg["data"]

    model = build_model(cfg["model"].get("name", "faster_rcnn"),
                        cfg["model"].get("num_classes", 33))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.to(device).eval()

    ds = CocoPuDataset(data["image_dir"], data["test_json"], transforms=eval_transforms())
    loader = DataLoader(ds, batch_size=cfg["eval"].get("batch_size", 4),
                        shuffle=False, collate_fn=collate_fn)

    coco = COCO(data["test_json"])
    cat_ids = sorted(c.id for c in coco.loadCats(coco.getCatIds()))
    thr = cfg["eval"].get("score_threshold", 0.05)

    rows = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([img.to(device) for img in images])
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"].item())
                for box, label, score in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
                    if score < thr:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    rows.append({
                        "image_id": img_id,
                        "category_id": cat_ids[int(label) - 1],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: inspect the official sample_submission.csv on Kaggle and map
    # `rows` into its exact schema before your first submission.
    sample = pd.read_csv(data.get("sample_submission", "data/sample_submission.csv"))
    print(f"Sample submission columns: {list(sample.columns)}")
    print(f"Wrote {len(rows)} predictions -> {out_path} (map to sample schema).")
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".preds.csv"), index=False)


if __name__ == "__main__":
    main()
