"""Main training entry point.

Usage:
    python -m src.training.train --config configs/faster_rcnn.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.coco_dataset import CocoPuDataset, collate_fn
from src.models.detector import build_model
from src.training.utils import (
    MetricLogger,
    get_device,
    load_config,
    set_seed,
    train_transforms,
)


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> MetricLogger:
    model.train()
    logger = MetricLogger()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(images, targets)
        loss = sum(v for v in losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        logger.update(**{k: v.item() for k, v in losses.items()}, total_loss=loss.item())
    print(f"Epoch {epoch}: {logger}")
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/faster_rcnn.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Config: {args.config} | Device: {device}")

    data_cfg = cfg["data"]
    train_ds = CocoPuDataset(
        image_dir=data_cfg["image_dir"],
        ann_file=data_cfg["train_json"],
        transforms=train_transforms(),
        pu_keep_prob=cfg.get("pu", {}).get("keep_prob", 1.0),
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_classes = cfg.get("model", {}).get("num_classes", 33)  # 32 + bg
    model = build_model(cfg["model"].get("name", "faster_rcnn"), num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["train"].get("lr", 0.005),
        momentum=0.9,
        weight_decay=cfg["train"].get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["train"].get("lr_milestones", [8, 11]), gamma=0.1
    )

    out_dir = Path(cfg["train"].get("output_dir", "experiments/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "total_loss"])

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        logger = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, logger.metrics.get("total_loss", [0])[-1]])
        if epoch % cfg["train"].get("save_every", 2) == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                out_dir / "checkpoints" / f"model_{epoch:03d}.pth",
            )

    torch.save({"model": model.state_dict()}, out_dir / "checkpoints" / "model_final.pth")
    print(f"Training complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
