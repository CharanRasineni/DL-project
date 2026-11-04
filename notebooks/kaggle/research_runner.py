"""Staged, auditable detector experiments. Bundled into each Kaggle notebook.

No automatic training retries. Smoke outputs are never paper results. Evaluation
and figures may be rerun from a checkpoint without repeating training.
"""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import data as D


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(coco, split, root, decode=True):
    train, val = split['train_ids'], split['val_ids']
    assert train and val and len(set(train)) == len(train) and len(set(val)) == len(val)
    assert set(train).isdisjoint(val), 'Train/validation overlap'
    images = {i['id']: i for i in coco['images']}
    assert len(images) == len(coco['images']), 'Duplicate image ids'
    assert set(train + val) <= images.keys()
    cats = {c['id'] for c in coco['categories']}
    assert len(cats) == len(coco['categories'])
    filenames = set()
    for iid in train + val:
        info = images[iid]
        name = info['file_name']
        assert name not in filenames, 'Duplicate filenames across split'
        filenames.add(name)
        path = (root / name).resolve()
        assert root.resolve() in path.parents and path.is_file(), str(path)
        if decode:
            with Image.open(path) as image:
                image.load()
                assert image.size == (info['width'], info['height']), 'Image/annotation size mismatch'
    ann_ids = set()
    for a in coco['annotations']:
        assert a['id'] not in ann_ids, 'Duplicate annotation ids'
        ann_ids.add(a['id'])
        assert a['image_id'] in images and a['category_id'] in cats
        x, y, w, h = a['bbox']
        info = images[a['image_id']]
        assert np.isfinite([x, y, w, h]).all() and w > 0 and h > 0
        assert x >= 0 and y >= 0 and x + w <= info['width'] + 1 and y + h <= info['height'] + 1, a
    return {'train_images': len(train), 'val_images': len(val), 'categories': len(cats)}


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, coco, ids, root, train=False):
        self.images = {i['id']: i for i in coco['images']}
        self.anns = defaultdict(list)
        for a in coco['annotations']:
            self.anns[a['image_id']].append(a)
        self.cat_ids = sorted(c['id'] for c in coco['categories'])
        self.labels = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.ids, self.root, self.train = list(ids), Path(root), train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        iid = self.ids[index]
        with Image.open(self.root / self.images[iid]['file_name']) as im:
            rgb = np.array(im.convert('RGB'), copy=True)
        height, width = rgb.shape[:2]
        boxes, labels = [], []
        for a in self.anns[iid]:
            x, y, w, h = a['bbox']
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                labels.append(self.labels[a['category_id']])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        if self.train and random.random() < 0.5:
            tensor = tensor.flip(-1)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return tensor, {'boxes': boxes, 'labels': torch.tensor(labels, dtype=torch.int64),
                        'image_id': torch.tensor([iid])}


def collate(batch):
    return tuple(zip(*batch))


def build_yolo(coco, split, root, out):
    """Unique image-id names prevent nested-path/duplicate-stem label collisions."""
    dataset = DetectionDataset(coco, [], root)
    names = {c['id']: c['name'] for c in coco['categories']}
    for part in ('train', 'val'):
        image_dir, label_dir = out / 'images' / part, out / 'labels' / part
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for iid in split[part + '_ids']:
            info = dataset.images[iid]
            source = root / info['file_name']
            target = image_dir / (str(iid) + source.suffix.lower())
            if not target.exists():
                target.symlink_to(source.resolve())
            w, h = info['width'], info['height']
            lines = []
            for a in dataset.anns[iid]:
                x, y, bw, bh = a['bbox']
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                if x2 > x1 and y2 > y1:
                    values = [(x1 + x2) / (2*w), (y1 + y2) / (2*h), (x2-x1)/w, (y2-y1)/h]
                    lines.append(str(dataset.labels[a['category_id']] - 1) + ' ' + ' '.join(f'{v:.8f}' for v in values))
            (label_dir / f'{iid}.txt').write_text('\n'.join(lines), encoding='utf-8')
    # JSON scalars are valid YAML scalars (safe for quotes in paths/class names).
    text = 'path: ' + json.dumps(out.resolve().as_posix()) + '\ntrain: images/train\nval: images/val\nnames:\n'
    text += ''.join(f'  {i}: {json.dumps(names[cid])}\n' for i, cid in enumerate(dataset.cat_ids))
    path = out / 'dataset.yaml'
    path.write_text(text, encoding='utf-8')
    return path


def make_frcnn(nclasses, size=640, pretrained=True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(
        weights='COCO_V1' if pretrained else None, weights_backbone=None,
        min_size=size, max_size=size, box_score_thresh=0.001, box_detections_per_img=300)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, nclasses + 1)
    return model


def save_checkpoint(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device='cpu'):
    # Only locally created / explicitly attached trusted checkpoints are accepted.
    return torch.load(path, map_location=device, weights_only=False)


def train_frcnn(cfg, coco, split, root, out, device):
    train = DetectionDataset(coco, split['train_ids'], root, train=True)
    loader = torch.utils.data.DataLoader(train, batch_size=cfg['batch'], shuffle=True,
        num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(cfg['seed']))
    model = make_frcnn(len(train.cat_ids), cfg['imgsz']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 27], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
    history, best_ap, start_epoch = [], -1., 0
    resume_path = cfg.get('resume_checkpoint')
    if resume_path:
        # Continue the budget-stopped run: exact optimizer/scheduler/scaler/RNG state.
        resumed = load_checkpoint(resume_path)
        assert resumed['cat_ids'] == train.cat_ids, 'Category mapping mismatch with checkpoint'
        assert resumed['config'].get('split_sha256') == cfg['split_sha256'], 'Checkpoint used a different split'
        assert 0 < resumed['epoch'] < cfg['epochs'], 'Checkpoint epoch outside resume range'
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        scaler.load_state_dict(resumed['scaler'])
        torch.set_rng_state(resumed['torch_rng'].cpu() if resumed['torch_rng'].device.type != 'cpu' else resumed['torch_rng'])
        np.random.set_state(resumed['numpy_rng'])
        random.setstate(resumed['python_rng'])
        if device == 'cuda' and resumed.get('cuda_rng'):
            torch.cuda.set_rng_state_all([s.cpu() for s in resumed['cuda_rng']])
        history = list(resumed['history'])
        start_epoch = resumed['epoch']
        vals = [h.get('val_COCO_AP') for h in history if h.get('val_COCO_AP') is not None]
        best_ap = max(vals) if vals else -1.
        prev_best = Path(resume_path).parent / 'best.pt'
        if prev_best.is_file():
            # Preserve overall-best weights; epoch rows after the checkpoint's own
            # save (e.g. the final validated epoch) are absent from its history.
            save_checkpoint(out / 'best.pt', load_checkpoint(prev_best))
        print(f'RESUMED from epoch {start_epoch}; history rows {len(history)}; prior best AP {best_ap:.4f}', flush=True)
    start = time.monotonic()
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start = time.monotonic()
        model.train()
        losses = []
        for images, targets in loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; no automatic retry')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        # Save before validation. Optimizer/scaler/RNG are retained for recovery.
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'epoch': epoch + 1, 'config': cfg, 'cat_ids': train.cat_ids,
                 'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                 'python_rng': random.getstate(), 'history': history,
                 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else []}
        save_checkpoint(out / 'last.pt', state)
        predictions = predict_frcnn(model, coco, split, root, device)
        evaluator = ResearchEvaluator(split, D.get_cat_info(coco, split), score_thr=cfg['score_thr'], max_det=300)
        metric = evaluator.evaluate(predictions)['mAP50-95']
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                        'val_COCO_AP': metric, 'seconds': time.monotonic() - epoch_start})
        pd.DataFrame(history).to_csv(out / 'training.csv', index=False)
        if metric > best_ap:
            best_ap = metric
            save_checkpoint(out / 'best.pt', state)
        print(history[-1], flush=True)
        elapsed = time.monotonic() - start
        if epoch + 1 < cfg['epochs'] and elapsed + (elapsed / (epoch + 1)) > cfg['max_train_hours'] * 3600:
            atomic_json(out / 'budget_stop.json', {'completed_epochs': epoch + 1, 'requested': cfg['epochs']})
            raise RuntimeError('Budget stop: checkpoints saved; incomplete run, not a matched-budget result')
    del optimizer, scaler, model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out / 'best.pt'


def predict_frcnn(model, coco, split, root, device):
    model.eval()
    dataset = DetectionDataset(coco, split['val_ids'], root)
    predictions = []
    with torch.no_grad():
        for image, target in dataset:
            output = model([image.to(device)])[0]
            for box, label, score in zip(output['boxes'].cpu().tolist(), output['labels'].cpu().tolist(), output['scores'].cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({'image_id': int(target['image_id'][0]),
                    'category_id': dataset.cat_ids[label - 1], 'score': score,
                    'bbox': [x1, y1, x2-x1, y2-y1]})
    return predictions


def ultra_class(family):
    from ultralytics import YOLO, RTDETR
    return YOLO if family == 'yolov8s' else RTDETR


def preserve_resume_checkpoint(trainer):
    """Keep one unstripped recovery checkpoint, not 30 large epoch files."""
    import shutil
    path = Path(trainer.last).parent / 'resume.pt'
    tmp = Path(str(path) + '.tmp')
    shutil.copyfile(trainer.last, tmp)
    os.replace(tmp, path)


def train_ultra(cfg, yaml, out):
    from ultralytics import settings
    # External experiment trackers are not part of this protocol.
    settings.update({key: False for key in ('raytune', 'wandb', 'mlflow',
        'neptune', 'clearml', 'comet', 'dvc', 'tensorboard')})
    from ultralytics import settings
    settings.update({'raytune': False, 'wandb': False, 'mlflow': False, 'comet': False})
    cls = ultra_class(cfg['family'])
    model = cls('yolov8s.pt' if cfg['family'] == 'yolov8s' else 'rtdetr-l.pt')
    # All defaults are archived by Ultralytics in args.yaml, in addition to cfg.
    model.add_callback('on_model_save', preserve_resume_checkpoint)
    model.train(data=str(yaml), epochs=cfg['epochs'], imgsz=cfg['imgsz'], batch=cfg['batch'],
        workers=0, cache=False, device=0, seed=cfg['seed'], deterministic=True,
        amp=True, patience=0, save=True, save_period=-1, plots=False,
        project=str(out / 'runs'), name='fit', exist_ok=False, verbose=False,
        optimizer='AdamW', lr0=0.0001, weight_decay=0.0005, close_mosaic=0,
        mosaic=0.0, mixup=0.0, degrees=0.0, translate=0.0, scale=0.0,
        fliplr=0.5, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    checkpoint = Path(model.trainer.best)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint


def detection_cam(model, family, rgb, size, class_index):
    """Grad-CAM for one declared class, no silent fallback to another class.

    FRCNN: selected post-NMS class scores. YOLO: raw class probabilities over
    anchors. RT-DETR: raw query class probabilities. Not causal explanations.
    The square visualisation transform is recorded and is not the AP input.
    """
    x = torch.from_numpy(np.array(Image.fromarray(rgb).resize((size, size)), copy=True))
    net = model if family == 'fasterrcnn' else model.model
    device = next(net.parameters()).device
    x = (x.permute(2, 0, 1).float() / 255).unsqueeze(0).to(device).requires_grad_(True)
    net.eval().float()
    # Spatial convolution output, not transformer tokens or tuple decoder output.
    layer_name = {'fasterrcnn': 'backbone.body.layer4',
                  'yolov8s': 'model.9.cv2', 'rtdetr': 'model.7'}[family]
    layer = dict(net.named_modules())[layer_name]
    capture = {}
    def hook(module, inputs, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            raise RuntimeError('CAM target must be a spatial BCHW tensor')
        capture['activation'] = output
        output.retain_grad()
    handle = layer.register_forward_hook(hook)
    try:
        net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if family == 'fasterrcnn':
                pred = net([x[0]])[0]
                scores = pred['scores'][pred['labels'] == class_index + 1]
            else:
                raw = net(x)
                pred = raw[0] if isinstance(raw, tuple) else raw
                if family == 'yolov8s':
                    assert pred.ndim == 3 and pred.shape[1] == 4 + net.model[-1].nc
                    scores = pred[0, 4 + class_index, :]
                else:
                    assert pred.ndim == 3 and pred.shape[-1] == 4 + net.model[-1].nc
                    scores = pred[0, :, 4 + class_index]
            if scores.numel() == 0:
                raise ValueError('No prediction for requested class; no CAM fabricated')
            score = scores.topk(min(5, scores.numel())).values.sum()
            score.backward()
            act = capture['activation']
            if act.grad is None:
                raise RuntimeError('Target activation has no gradient')
            heat = torch.relu((act.grad.mean((2, 3), keepdim=True) * act).sum(1))[0]
            heat = heat.detach().cpu().numpy()
            if not np.isfinite(heat).all() or float(heat.max()) <= float(heat.min()):
                raise ValueError('Zero/nonfinite CAM; not exported as an explanation')
            heat = (heat - heat.min()) / (heat.max() - heat.min())
        return heat, {'layer': layer_name, 'class_index': class_index,
                      'objective': 'sum top-5 selected-class scores',
                      'input_transform': f'square RGB resize {size}x{size}; float32 [0,1]'}
    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)


def figures(model, cfg, coco, split, root, predictions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ds = DetectionDataset(coco, split['val_ids'], root)
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p['image_id']].append(p)
    names = {c['id']: c['name'] for c in coco['categories']}
    ids = sorted(random.Random(42).sample(sorted(split['val_ids']), min(6, len(split['val_ids']))))
    manifest = {'selection': 'seed=42 uniform validation images, not selected by model success', 'images': []}
    for iid in ids:
        with Image.open(root / ds.images[iid]['file_name']) as im:
            rgb = np.array(im.convert('RGB'))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(rgb)
        for a in ds.anns[iid]:
            x, y, w, h = a['bbox']
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='lime'))
            ax.text(x, y, names[a['category_id']], fontsize=6, color='lime')
        selected = [p for p in by_image[iid] if p['score'] >= 0.25][:20]
        for p in selected:
            x, y, w, h = p['bbox']
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linestyle='--'))
            ax.text(x, y+h, f"{names[p['category_id']]} {p['score']:.2f}", fontsize=6, color='red')
        ax.set_title(f'{iid}: available annotations (green), predictions >=0.25 (red)')
        ax.axis('off')
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'qualitative_{iid}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        row = {'image_id': iid, 'cam_status': 'no_detection'}
        # Explain the highest-score detection even if below the display threshold.
        candidates = sorted(by_image[iid], key=lambda p: -p['score'])
        if candidates:
            cid = candidates[0]['category_id']
            try:
                heat, metadata = detection_cam(model, cfg['family'], rgb, cfg['imgsz'], ds.cat_ids.index(cid))
                np.save(out / f'gradcam_{iid}.npy', heat)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(rgb)
                axes[1].imshow(rgb)
                resized = np.array(Image.fromarray(heat).resize((rgb.shape[1], rgb.shape[0])))
                axes[1].imshow(resized, cmap='jet', alpha=0.45, vmin=0, vmax=1)
                for axis in axes:
                    axis.axis('off')
                fig.suptitle(f"{iid}: {names[cid]} — class-score Grad-CAM (not causal evidence)")
                for ext in ('png', 'pdf'):
                    fig.savefig(out / f'gradcam_{iid}.{ext}', dpi=200, bbox_inches='tight')
                plt.close(fig)
                row.update(cam_status='ok', category_id=cid, **metadata)
            except Exception as exc:
                row.update(cam_status='failed', error=f'{type(exc).__name__}: {exc}')
        manifest['images'].append(row)
        atomic_json(out / 'figure_manifest.json', manifest)
    return manifest


class ResearchEvaluator(D.FullEvaluator):
    """COCO default AP@100, with explicit unavailable classes/areas."""
    def _eval_once(self, gt_dict, preds):
        import contextlib
        import io
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(gt_dict)
            gt.createIndex()
            if preds:
                dt = gt.loadRes(copy.deepcopy(preds))
            else:
                dt = COCO()
                dt.dataset = {'images': copy.deepcopy(gt_dict['images']),
                              'categories': copy.deepcopy(gt_dict['categories']), 'annotations': []}
                dt.createIndex()
            ev = COCOeval(gt, dt, 'bbox')
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return ev

    def _collect(self, ev, used):
        precision = ev.eval['precision']
        def mean_valid(values):
            values = values[values > -1]
            return float(values.mean()) if values.size else None
        per, per50, size = {}, {}, {}
        for k, cid in enumerate(ev.params.catIds):
            name = self.cat_info['cat_id_to_name'][cid]
            per[name] = mean_valid(precision[:, :, k, 0, -1])
            per50[name] = mean_valid(precision[0, :, k, 0, -1])
            size[name] = {label: mean_valid(precision[:, :, k, i, -1])
                          for i, label in enumerate(ev.params.areaRngLbl) if label != 'all'}
        groups = {}
        for group, names in self.groups.items():
            vals = [per[n] for n in names if per.get(n) is not None]
            groups[group] = float(np.mean(vals)) if vals else None
        result = {key: float(ev.stats[i]) if ev.stats[i] >= 0 else None
                  for i, key in enumerate(['mAP50-95', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large'])}
        result.update({'per_class_AP50-95': per, 'per_class_AP50': per50,
            'per_class_size_AP50-95': size, 'head_mid_tail_AP50-95': groups,
            'groups': {g: sorted(v) for g, v in self.groups.items()},
            'COCO_maxDets': [1, 10, 100], 'n_dets': len(used)})
        return result

    def bootstrap_research(self, predictions, n=200, seed=42):
        rng = np.random.default_rng(seed)
        ids = self.split['val_ids']
        rows = []
        by_image = defaultdict(list)
        for p in predictions:
            by_image[p['image_id']].append(p)
        for _ in range(n):
            pairs = [(int(orig), i+1) for i, orig in enumerate(rng.choice(ids, len(ids), replace=True))]
            draw = [dict(p, image_id=new) for orig, new in pairs for p in by_image[orig]]
            result = self._collect(self._eval_once(self._gt_dict_from_pairs(pairs), draw), draw)
            rows.append({'mAP50-95': result['mAP50-95'], **result['head_mid_tail_AP50-95']})
        intervals = {}
        for key in rows[0]:
            values = [r[key] for r in rows if r[key] is not None]
            intervals[key] = {'n_valid': len(values), 'lo': float(np.percentile(values, 2.5)),
                              'hi': float(np.percentile(values, 97.5))} if values else None
        return {'n': n, 'seed': seed, 'intervals': intervals,
                'method': 'image percentile bootstrap, 95%; same resampling seed across models',
                'limitation': 'Absent classes excluded per draw; rare-class composition varies. Not training-seed uncertainty.'}


def run(cfg):
    cfg = dict(cfg)
    assert cfg['mode'] in ('smoke', 'train', 'evaluate')
    assert cfg['family'] in ('fasterrcnn', 'yolov8s', 'rtdetr')
    assert cfg['batch'] > 0 and cfg['imgsz'] == 640
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('GPU required for real-image run; no silent CPU training')
    D.set_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    ann, root = D.resolve_dataset_paths()
    D.DATASET_JSON, D.IMG_ROOT = ann, root
    coco = D.load_coco_data()
    split = D.load_frozen_split()
    # Exact expected split identity (both train and validation memberships).
    split_hash = hashlib.sha256(json.dumps({k: sorted(split[k]) for k in ('train_ids', 'val_ids')}, sort_keys=True).encode()).hexdigest()
    assert split_hash == cfg['split_sha256'], 'Unexpected frozen split'
    out = Path('/kaggle/working') / (cfg['family'] + '_' + cfg['mode'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'provenance.json').exists():
        raise RuntimeError('Output already exists; use a fresh working session to avoid overwriting a run')
    full_counts = D.get_cat_info(coco, split)['train_counts']
    if cfg['mode'] == 'smoke':
        # Include the largest image-area instances, not only convenient tiny images.
        info = {im['id']: im for im in coco['images']}
        choose = lambda ids: sorted(ids, key=lambda i: -(info[i]['width'] * info[i]['height']))[:4]
        split = dict(split, train_ids=choose(split['train_ids']), val_ids=choose(split['val_ids']))
        cfg['epochs'] = 1
    checks = validate_dataset(coco, split, root, decode=True)
    atomic_json(out / 'config.json', cfg)
    atomic_json(out / 'split.json', split)
    atomic_json(out / 'train_counts.json', full_counts)
    provenance = {'config': cfg, 'system': D.system_info(), 'checks': checks,
                  'annotation_sha256': sha256(ann), 'full_split_sha256': split_hash,
                  'runner_sha256': sha256(__file__), 'data_sha256': sha256(D.__file__),
                  'task': 'positive-unlabeled marine object detection',
                  'scientific_result': cfg['mode'] != 'smoke',
                  'limitations': ['single training seed', 'partial validation annotations',
                    'same 30 epochs and nominal 640px setting is NOT equal compute',
                    'framework resizing, optimizers and checkpoint selection differ']}
    atomic_json(out / 'provenance.json', provenance)
    yaml = None
    if cfg['family'] != 'fasterrcnn':
        yaml = build_yolo(coco, split, root, out / 'dataset')
    if cfg['mode'] == 'evaluate':
        checkpoint = Path(cfg['checkpoint'])
        assert checkpoint.is_file(), 'Explicit attached checkpoint required'
    elif cfg['family'] == 'fasterrcnn':
        checkpoint = train_frcnn(cfg, coco, split, root, out, device)
    else:
        checkpoint = train_ultra(cfg, yaml, out)
    assert checkpoint.is_file()
    provenance['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256(checkpoint)}
    atomic_json(out / 'provenance.json', provenance)
    if cfg['family'] == 'fasterrcnn':
        state = load_checkpoint(checkpoint)
        model = make_frcnn(len(coco['categories']), cfg['imgsz'], pretrained=False).to(device)
        model.load_state_dict(state['model'], strict=True)
        del state
        predictions = predict_frcnn(model, coco, split, root, device)
    else:
        model = ultra_class(cfg['family'])(str(checkpoint))
        cats = sorted(c['id'] for c in coco['categories'])
        predictions = []
        # Folder source + explicit batch=1 avoids all-validation warmup OOM.
        for r in model.predict(source=str(out / 'dataset' / 'images' / 'val'),
                               stream=True, batch=1, imgsz=cfg['imgsz'], conf=cfg['score_thr'],
                               max_det=300, device=0, half=False, verbose=False):
            iid = int(Path(r.path).stem)
            for box, label, score in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({'image_id': iid, 'category_id': cats[int(label)],
                    'score': score, 'bbox': [x1, y1, x2-x1, y2-y1]})
    atomic_json(out / 'preds.json', predictions)
    cat_info = D.get_cat_info(coco, split)
    cat_info['train_counts'] = full_counts
    ev = ResearchEvaluator(split, cat_info, score_thr=cfg['score_thr'], max_det=300)
    result = ev.evaluate(predictions)
    result['config'] = cfg
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'val_gt.json', ev._gt_dict_for(split['val_ids']))
    counts = defaultdict(int)
    for a in coco['annotations']:
        if a['image_id'] in set(split['val_ids']):
            counts[cat_info['cat_id_to_name'][a['category_id']]] += 1
    pd.DataFrame([{'class': name, 'train_instances': full_counts.get(name, 0),
                   'val_instances': counts[name], 'AP50-95': ap,
                   'AP50': result['per_class_AP50'][name],
                   **result['per_class_size_AP50-95'][name]}
                  for name, ap in result['per_class_AP50-95'].items()]).to_csv(out / 'per_class.csv', index=False)
    report = {'core': 'ok', 'mode': cfg['mode']}
    # Same boundary: decoded RGB uint8 -> CPU output, includes transfer/pre/post,
    # excludes image decode and disk I/O. Batch=1, float32, synchronized CUDA.
    iid = sorted(split['val_ids'])[0]
    info = next(i for i in coco['images'] if i['id'] == iid)
    with Image.open(root / info['file_name']) as im:
        rgb = np.array(im.convert('RGB'), copy=True)
    def inference():
        with torch.no_grad():
            if cfg['family'] == 'fasterrcnn':
                x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255
                return {k: v.cpu() for k, v in model([x])[0].items()}
            return model.predict(source=rgb[:, :, ::-1].copy(), imgsz=cfg['imgsz'],
                conf=cfg['score_thr'], max_det=300, device=0, half=False, verbose=False)[0].cpu()
    with D.stage('latency', report):
        model.eval() if cfg['family'] == 'fasterrcnn' else model.model.eval()
        result['latency'] = D.bench_latency(inference, warmup=3, iters=10 if cfg['mode']=='smoke' else 30)
        result['latency']['boundary'] = 'decoded RGB uint8 to CPU detections; batch1 fp32; excludes disk/decode'
        atomic_json(out / 'results.json', result)
    with D.stage('training_curves', report):
        import matplotlib.pyplot as plt
        csv_path = out / 'training.csv' if cfg['family'] == 'fasterrcnn' else out / 'runs/fit/results.csv'
        frame = pd.read_csv(csv_path)
        frame.columns = frame.columns.str.strip()
        frame.to_csv(out / 'training_history.csv', index=False)
        columns = [c for c in frame.columns if 'loss' in c or 'mAP' in c or c == 'val_COCO_AP']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for column in columns:
            axis = axes[0] if 'loss' in column else axes[1]
            axis.plot(frame['epoch'], frame[column], label=column)
        for axis, title in zip(axes, ('Training losses', 'Native validation metrics (checkpoint selection)')):
            axis.set_title(title)
            axis.set_xlabel('Epoch')
            axis.legend(fontsize=6)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / ('training_curves.' + ext), dpi=200)
        plt.close(fig)
    with D.stage('figures', report):
        # Fresh unfused model: predictor fusion/inference tensors must not contaminate autograd.
        if cfg['family'] != 'fasterrcnn':
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = ultra_class(cfg['family'])(str(checkpoint))
            model.model.to(device).float().eval()
        manifest = figures(model, cfg, coco, split, root, predictions, out)
        report['cam_successes'] = sum(r['cam_status'] == 'ok' for r in manifest['images'])
    atomic_json(out / 'stage_report.json', report)
    if cfg['mode'] == 'smoke':
        # Optional full-run plots cannot lose training, but smoke gate demands CAM.
        if report.get('cam_successes', 0) < 1:
            raise RuntimeError('Smoke CAM gate failed; inspect figure_manifest.json; do not launch full run')
        if any(s['status'] != 'ok' for s in report.get('stages', {}).values()):
            raise RuntimeError('Smoke analysis gate failed')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.get('bootstrap', 0):
        with D.stage('bootstrap', report):
            result['bootstrap'] = ev.bootstrap_research(D.filter_preds(predictions, score_thr=cfg['score_thr'], max_det=300), cfg['bootstrap'])
            atomic_json(out / 'results.json', result)
    atomic_json(out / 'stage_report.json', report)
    analysis_complete = (report.get('cam_successes', 0) > 0 and
        all(s['status'] == 'ok' for s in report.get('stages', {}).values()))
    atomic_json(out / 'completed.json', {'status': 'PASS' if analysis_complete else 'CORE_COMPLETE_ANALYSIS_INCOMPLETE',
        'analysis_complete': analysis_complete, 'mode': cfg['mode'],
        'scientific_result': cfg['mode'] != 'smoke', 'checkpoint_sha256': sha256(checkpoint),
        'runner_sha256': sha256(__file__), 'configuration': cfg})
    print('COMPLETE', out, flush=True)
    return out
