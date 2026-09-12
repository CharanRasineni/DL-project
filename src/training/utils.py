"""Shared training utilities: config loading, seeding, transforms, metrics."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as F


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Compose:
    """Minimal transform pipeline for detection (image + target)."""

    def __init__(self, ops):
        self.ops = ops

    def __call__(self, image: Image.Image, target: dict):
        for op in self.ops:
            image, target = op(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = F.hflip(image)
            w = image.shape[-1] if torch.is_tensor(image) else image.size[0]
            boxes = target["boxes"]
            if len(boxes):
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                target["boxes"][:, 0] = w - x2
                target["boxes"][:, 2] = w - x1
        return image, target


def train_transforms():
    return Compose([RandomHorizontalFlip(0.5), ToTensor()])


def eval_transforms():
    return Compose([ToTensor()])


class MetricLogger:
    """Accumulates scalar metrics; .loss attribute-style access."""

    def __init__(self, delimiter: str = "  "):
        self.metrics = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            v = float(v)
            if k not in self.metrics:
                self.metrics[k] = [v]
            else:
                self.metrics[k].append(v)

    @property
    def loss(self):
        return sum(np.mean(v) for k, v in self.metrics.items() if "loss" in k)

    def __str__(self):
        return self.delimiter.join(
            f"{k}: {np.mean(v):.4f}" for k, v in self.metrics.items()
        )
