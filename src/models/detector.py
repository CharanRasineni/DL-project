"""Detection model builders (torchvision reference implementation).

Includes a Faster R-CNN ResNet-50 FPN baseline that trains out of the box on
the 32-class FathomNet data. Swap in DINO/YOLO wrappers here later; the
training loop only expects a model taking (images, targets) and returning a
loss dict in train mode and detections in eval mode (torchvision contract).
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(
    num_classes: int = 33,  # 32 classes + background
    pretrained_backbone: bool = True,
) -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained_backbone else None,
        weights_backbone="DEFAULT" if pretrained_backbone else None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 33) -> torch.nn.Module:
    builders = {"faster_rcnn": build_faster_rcnn}
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Available: {list(builders)}")
    return builders[name](num_classes=num_classes)
