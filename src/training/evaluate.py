"""mAP evaluation replicating the competition's official evaluator.

The Kaggle evaluation uses mean Average Precision on the fully labeled test
set (COCO-style mAP@[0.5:0.95] averaged over the 32 categories). This module
uses pycocotools against dataset_test.json -- the same protocol.
"""
from __future__ import annotations

import argparse

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from src.data.coco_dataset import CocoPuDataset, collate_fn
from src.models.detector import build_model
from src.data.coco_dataset import FATHOMNET_CLASSES  # noqa: F401
from src.training.utils import eval_transforms, get_device, load_config, set_seed



@torch.no_grad()
def predict_to_coco_json(model, loader, device, coco_gt: COCO, score_thr: float = 0.05):
    """Run model over loader and return COCO-format result list."""
    model.eval()
    cat_ids = sorted(c.id for c in coco_gt.loadCats(coco_gt.getCatIds()))
    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            for box, label, score in zip(
                out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
            ):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": cat_ids[int(label) - 1],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return results


def evaluate(model, device, test_json: str, image_dir: str, batch_size: int = 4):
    coco_gt = COCO(test_json)
    ds = CocoPuDataset(image_dir, test_json, transforms=eval_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = predict_to_coco_json(model, loader, device, coco_gt)
    if not results:
        print("No predictions above threshold.")
        return {}
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP@[0.5:0.95]": ev.stats[0], "mAP@0.5": ev.stats[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/faster_rcnn.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    model = build_model(
        cfg["model"].get("name", "faster_rcnn"),
        cfg["model"].get("num_classes", 33),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    data = cfg["data"]
    metrics = evaluate(model, device, data["test_json"], data["image_dir"])
    print(metrics)


if __name__ == "__main__":
    main()
