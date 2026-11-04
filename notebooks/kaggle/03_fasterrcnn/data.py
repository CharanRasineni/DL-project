"""FathomNet CLEF shared data + COCO evaluation primitives.

Single source of truth for:
* Frozen 90/10 split (seed 42, image-level, coverage-pass) from 02_split
* COCO bbox eval via pycocotools with per-class, head/mid/tail, and
  small/medium/large AP reporting
* Repeat-Factor Sampling (Shen & Mammone 2020) for long-tail interventions
* YOLO-format dataset tree + fathomnet.yaml generator
* Provenance snapshots for every run
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------------------------------------------------------
# Canonical paths (override via env on Kaggle)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else HERE

SPLIT_JSON = Path(
    os.environ.get(
        "FATHOMNET_SPLIT_JSON",
        "/kaggle/input/fathomnet-02-split/split.json",
    )
)
DATASET_JSON = Path(
    os.environ.get(
        "FATHOMNET_DATASET_JSON",
        "/kaggle/input/fathomnet-01-dataset/train_dataset.json",
    )
)
IMG_ROOT = Path(
    os.environ.get(
        "FATHOMNET_IMG_ROOT",
        "/kaggle/input/fathomnet-01-dataset/images",
    )
)


def load_split(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the frozen split JSON produced by 02_split.

    Returns dict with keys: train_ids, val_ids, split_seed, coverage_passed.
    """
    p = path or SPLIT_JSON
    if not p.exists():
        raise FileNotFoundError(f"split.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_coco_data(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the full COCO-format annotation file (train_dataset.json)."""
    p = path or DATASET_JSON
    if not p.exists():
        raise FileNotFoundError(f"train_dataset.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_cat_info(
    coco_data: Dict[str, Any],
    split: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build category index, sorted ids, and optional train counts.

    Returns:
        cat_id_to_name: {cat_id: class_name}
        name_to_cat_id: {class_name: cat_id}
        cat_ids_sorted: sorted list of cat_ids
        train_counts: {class_name: train_instance_count}  (only if split given)
    """
    cats = {c["id"]: c["name"] for c in coco_data["categories"]}
    name_to_id = {v: k for k, v in cats.items()}
    cat_ids_sorted = sorted(cats.keys())

    train_counts: Dict[str, int] = {}
    if split is not None:
        train_set = set(split["train_ids"])
        counts: Dict[str, int] = {}
        for a in coco_data["annotations"]:
            if a["image_id"] in train_set:
                name = cats[a["category_id"]]
                counts[name] = counts.get(name, 0) + 1
        train_counts = counts

    return {
        "cat_id_to_name": cats,
        "name_to_cat_id": name_to_id,
        "cat_ids_sorted": cat_ids_sorted,
        "train_counts": train_counts,
    }


def define_groups(
    train_counts: Dict[str, int],
    head_k: int = 3,
    mid_k: int = 4,
) -> Dict[str, set]:
    """Head/mid/tail groups by train-instance frequency.

    head: top-k most frequent classes
    mid:  next-k most frequent classes
    tail: the remaining classes
    """
    ordered = sorted(train_counts.keys(), key=lambda k: -train_counts[k])
    head = set(ordered[:head_k])
    mid = set(ordered[head_k:head_k + mid_k])
    tail = set(ordered[head_k + mid_k:])
    return {"head": head, "mid": mid, "tail": tail}


def stratify_per_class(
    per_class: Dict[str, float],
    groups: Dict[str, set],
) -> Dict[str, float]:
    """Average per-class AP within each head/mid/tail group."""
    out: Dict[str, float] = {}
    for g, names in groups.items():
        vals = [per_class[n] for n in names if n in per_class]
        out[g] = float(np.mean(vals)) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# COCO bbox evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Single COCO bbox evaluator with per-class and group reporting.

    Parameters
    ----------
    split_json : Path or dict
        The frozen split (used for val image list and group construction).
    cat_info : dict
        Output of get_cat_info().
    score_thr : float
        Score threshold for detections (default 0.01).
    max_det : int
        Maximum detections per image (default 300).
    """

    def __init__(
        self,
        split: Dict[str, Any],
        cat_info: Dict[str, Any],
        *,
        score_thr: float = 0.01,
        max_det: int = 300,
    ) -> None:
        self.split = split
        self.cat_info = cat_info
        self.score_thr = score_thr
        self.max_det = max_det
        # Build COCO ground-truth once
        self.coco_gt = self._build_gt_coco()
        self.groups = define_groups(cat_info["train_counts"])

    def _build_gt_coco(self) -> COCO:
        """Build a COCO object from val annotations only."""
        coco_data = load_coco_data()
        val_set = set(self.split["val_ids"])
        img_by_id = {i["id"]: i for i in coco_data["images"]}
        gt = {
            "images": [img_by_id[i] for i in self.split["val_ids"]],
            "annotations": [
                a for a in coco_data["annotations"] if a["image_id"] in val_set
            ],
            "categories": coco_data["categories"],
        }
        gt_path = WORK / "eval_val_gt.json"
        gt_path.write_text(json.dumps(gt))
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(gt_path))
        return coco

    def evaluate(
        self,
        preds: List[Dict[str, Any]],
        *,
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run full COCO bbox eval and return structured results.

        Parameters
        ----------
        preds : list of COCO detection dicts
            Each dict must have image_id, category_id, bbox [x,y,w,h],
            score.
        out_dir : Path, optional
            If given, write `preds.json` and `results.json` here.

        Returns
        -------
        results : dict
            {mAP50-95, mAP50, mAP75, mAP_small, mAP_medium, mAP_large,
             per_class_AP50-95: {name: ap},
             head_mid_tail_AP50-95: {head, mid, tail},
             n_dets}
        """
        # Write predictions if requested
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            preds_path = out_dir / "preds.json"
            preds_path.write_text(json.dumps(preds))

        dt = self.coco_gt.loadRes(preds)
        e = COCOeval(self.coco_gt, dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
            e.summarize()

        stats = e.stats  # [mAP@.5:.95, mAP@.5, mAP@.75, mAPs, mAPm, mAPl, ...]

        per_class = self._per_class_ap(e)

        group_ap = stratify_per_class(per_class, self.groups)

        results = {
            "mAP50-95": round(float(stats[0]), 4),
            "mAP50": round(float(stats[1]), 4),
            "mAP75": round(float(stats[2]), 4),
            "mAP_small": round(float(stats[3]), 4),
            "mAP_medium": round(float(stats[4]), 4),
            "mAP_large": round(float(stats[5]), 4),
            "per_class_AP50-95": per_class,
            "head_mid_tail_AP50-95": {
                g: round(v, 4) for g, v in group_ap.items()
            },
            "n_dets": len(preds),
        }

        # Write results
        if out_dir is not None:
            res_path = out_dir / "results.json"
            res_path.write_text(json.dumps(results, indent=2))

        return results

    def _per_class_ap(self, e: COCOeval) -> Dict[str, float]:
        """Run per-class COCO eval (mAP@.5:.95 for each class)."""
        per: Dict[str, float] = {}
        name_by_id = self.cat_info["cat_id_to_name"]
        cat_ids = self.cat_info["cat_ids_sorted"]
        saved = e.params.catIds
        for cid in cat_ids:
            e.params.catIds = [cid]
            with contextlib.redirect_stdout(io.StringIO()):
                e.evaluate()
                e.accumulate()
                e.summarize()
            per[name_by_id[cid]] = round(float(e.stats[0]), 4)
        e.params.catIds = saved
        return per


# ---------------------------------------------------------------------------
# Repeat-Factor Sampling (Shen & Mammone 2020)
# ---------------------------------------------------------------------------

def repeat_factors(
    coco_data: Dict[str, Any],
    split: Dict[str, Any],
    *,
    beta: float = 0.005,
) -> Dict[int, float]:
    """Compute per-image repeat factors for long-tail augmentation.

    An image's factor = max over its classes of class_rep[c], where
    class_rep[c] = max(1, beta / image_freq[c]) if image_freq[c] < beta else 1.
    """
    train_set = set(split["train_ids"])
    img_to_cats: Dict[int, set] = {}
    for a in coco_data["annotations"]:
        if a["image_id"] in train_set:
            img_to_cats.setdefault(a["image_id"], set()).add(a["category_id"])

    n_train = len(train_set)
    img_freq: Dict[int, float] = {}
    for cid in coco_data["categories"]:
        n_imgs = sum(1 for im in img_to_cats if cid in img_to_cats[im])
        img_freq[cid] = n_imgs / n_train if n_train > 0 else 0.0

    class_rep: Dict[int, float] = {}
    for cid, freq in img_freq.items():
        if freq > 0 and freq < beta:
            class_rep[cid] = max(1.0, beta / freq)
        else:
            class_rep[cid] = 1.0

    img_rep: Dict[int, float] = {}
    for img_id in split["train_ids"]:
        cats = img_to_cats.get(img_id, set())
        img_rep[img_id] = max((class_rep.get(c, 1.0) for c in cats), default=1.0)

    return img_rep


def build_repeated_train_list(
    split: Dict[str, Any],
    img_rep: Dict[int, float],
    seed: int,
) -> List[int]:
    """Expand train_ids by repeat factors with stochastic fractional copies."""
    rng = np.random.default_rng(seed)
    expanded: List[int] = []
    for img_id in split["train_ids"]:
        f = img_rep.get(img_id, 1.0)
        n_full = int(np.ceil(f))
        expanded.extend([img_id] * n_full)
        if f > 1.0 and (f - int(f)) > rng.random():
            expanded.append(img_id)
    rng.shuffle(expanded)
    return expanded


def summarize_repeat_factors(img_rep: Dict[int, float]) -> Dict[str, Any]:
    """Return summary stats for repeat factors."""
    vals = list(img_rep.values())
    return {
        "n_images": len(vals),
        "min_factor": round(float(np.min(vals)), 4),
        "max_factor": round(float(np.max(vals)), 4),
        "mean_factor": round(float(np.mean(vals)), 4),
        "n_augmented": int(sum(1 for v in vals if v > 1.0)),
        "frac_augmented": round(sum(1 for v in vals if v > 1.0) / len(vals), 4),
    }


# ---------------------------------------------------------------------------
# YOLO-format dataset writer
# ---------------------------------------------------------------------------

def write_yolo_dataset(
    coco_data: Dict[str, Any],
    split: Dict[str, Any],
    img_root: Path,
    out_dir: Path,
    *,
    train_ids: Optional[Sequence[int]] = None,
    val_ids: Optional[Sequence[int]] = None,
    overwrite: bool = True,
) -> Path:
    """Create a YOLO-format dataset tree + fathomnet.yaml.

    Symlinks images from the source image root; writes normalised YOLO txt
    labels (class_idx cx cy w h) per image.
    """
    if out_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(out_dir)

    cats = {c["id"]: c["name"] for c in coco_data["categories"]}
    sorted_ids = sorted(coco_data["categories"], key=lambda c: c["id"])
    id_to_idx = {cid: i for i, cid in enumerate(sorted_ids)}

    train_ids = list(train_ids or split["train_ids"])
    val_ids = list(val_ids or split["val_ids"])
    all_ids = set(train_ids + val_ids)

    img_by_id = {i["id"]: i for i in coco_data["images"] if i["id"] in all_ids}

    anns_by_img: Dict[int, List[dict]] = {i: [] for i in all_ids}
    for a in coco_data["annotations"]:
        if a["image_id"] in all_ids:
            anns_by_img[a["image_id"]].append(a)

    base = out_dir / "fathomnet"
    for sp, ids in (("train", train_ids), ("val", val_ids)):
        img_dir = base / "images" / sp
        lab_dir = base / "labels" / sp
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        for iid in ids:
            info = img_by_id[iid]
            src = img_root / info["file_name"]
            dst = img_dir / info["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)

            if not dst.exists():
                dst.symlink_to(src.resolve())

            try:
                w = float(info.get("width", src.width))
                h = float(info.get("height", src.height))
            except Exception:
                from PIL import Image
                with Image.open(src) as im:
                    w, h = float(im.width), float(im.height)

            lines: List[str] = []
            for a in anns_by_img.get(iid, []):
                x, y, bw, bh = a["bbox"]
                if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
                    continue
                cx = (x + bw / 2.0) / w
                cy = (y + bh / 2.0) / h
                nw = min(bw / w, 1.0)
                nh = min(bh / h, 1.0)
                cid = a["category_id"]
                idx = id_to_idx.get(cid)
                if idx is None:
                    continue
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            lab_path = str(dst).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            Path(lab_path).write_text("\n".join(lines))

    yaml_path = base / "fathomnet.yaml"
    names_block = "\n".join(
        f"  {idx}: {cats[cid]}" for idx, cid in enumerate(sorted_ids)
    )
    yaml_path.write_text(
        "path: /kaggle/working/fathomnet\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(sorted_ids)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Seed / determinism helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy / random / torch / cudnn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_all(seed: int) -> None:
    """Alias for set_seed for backwards compat."""
    set_seed(seed)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance(
    model: str,
    seed: int,
    epochs: int,
    input: str,
    score_thr: float,
    max_det: int,
    *,
    model_family: Optional[str] = None,
    framework: Optional[str] = None,
    framework_version: Optional[str] = None,
    batch_size: Optional[int] = None,
    n_params_million: Optional[float] = None,
    n_train_images: Optional[int] = None,
    val_split: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Snapshot of experimental conditions for a training run.

    Parameters
    ----------
    model : str
        Model identifier, e.g. 'Faster R-CNN R50-FPN'.
    seed : int
        Random seed used for the split and training.
    epochs : int
        Number of training epochs.
    input : str
        Input resolution policy, e.g. 'min800/max1333' or 'imgsz=960'.
    score_thr : float
        Confidence threshold for COCO eval.
    max_det : int
        Max detections per image for COCO eval.
    model_family : str, optional
        'two-stage', 'one-stage', 'transformer', etc.
    framework : str, optional
        Library used, e.g. 'torchvision', 'ultralytics'.
    framework_version : str, optional
        Version string of the framework.
    batch_size : int, optional
        Training batch size.
    n_params_million : float, optional
        Model parameter count in millions.
    n_train_images : int, optional
        Number of training images.
    val_split : float, optional
        Validation fraction.
    extra : dict, optional
        Additional key-value pairs to include.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    base: Dict[str, Any] = {
        "model": model,
        "seed": seed,
        "epochs": epochs,
        "input": input,
        "score_thr": score_thr,
        "max_det": max_det,
        "timestamp": now,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if model_family is not None:
        base["model_family"] = model_family
    if framework is not None:
        base["framework"] = framework
    if framework_version is not None:
        base["framework_version"] = framework_version
    if batch_size is not None:
        base["batch_size"] = batch_size
    if n_params_million is not None:
        base["params_million"] = round(n_params_million, 2)
    if n_train_images is not None:
        base["n_train_images"] = n_train_images
    if val_split is not None:
        base["val_split"] = val_split
    if torch.cuda.is_available():
        try:
            base["cuda_device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            base["cuda_memory_gb"] = round(props.total_mem / 1e9, 1)
        except Exception:
            pass
    if extra:
        base["extra"] = extra
    return base


def write_provenance(
    out_dir: Path,
    proto: Dict[str, Any],
    results: Dict[str, Any],
) -> Path:
    """Write a combined protocol+results JSON to out_dir/provenance.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "protocol": proto,
        "results": results,
    }
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(combined, indent=2))
    return path


def cleanup_symlinks(dir_path: Path) -> None:
    """Remove symlinks in dir_path that no longer point to existing targets."""
    for p in dir_path.rglob("*"):
        if p.is_symlink() and not p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Prediction converters
# ---------------------------------------------------------------------------

def to_coco_preds_yolo(
    results,
    img_id_by_filename: Dict[str, int],
    cat_id_by_idx: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Convert ultralytics Results to COCO-format bbox predictions.

    Parameters
    ----------
    results : iterable of ultralytics.engine.results.Results
    img_id_by_filename : {file_path_str: coco_image_id}
    cat_id_by_idx : {yolo_class_idx: coco_category_id}

    Returns
    -------
    List[Dict[str, Any]]  — COCO-format detections
    """
    preds: List[Dict[str, Any]] = []
    for r in results:
        path = str(r.path)
        img_id = img_id_by_filename.get(path)
        if img_id is None:
            continue
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            preds.append({
                "image_id": int(img_id),
                "category_id": int(cat_id_by_idx[c]),
                "bbox": [
                    float(x1), float(y1),
                    float(x2 - x1), float(y2 - y1),
                ],
                "score": float(s),
            })
    return preds


def to_coco_preds_detr(
    outputs,  # list of dicts from DETR model (post-processor output)
    img_id_by_filename: Dict[str, int],
    cat_id_by_idx: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Convert DETR post-processor output to COCO-format predictions.

    Expects `outputs` to be a list of dicts, one per image, each with
    keys: 'boxes' (Tensor [N,4] xyxy), 'scores' (Tensor [N]),
    'labels' (Tensor [N]).
    """
    preds: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        # Determine image id: assume outputs are in the same order as val
        # image list.  We rely on the caller to pass the right img_id map.
        # We will require the caller to provide img_id_by_filename keyed by
        # path and to iterate in matching order.
        pass  # placeholder — caller should implement per their dataloader
    # The actual implementation depends on the specific DETR wrapper used.
    raise NotImplementedError(
        "Use to_coco_preds_detr_xyxy instead, which takes explicit paths."
    )


def to_coco_preds_detr_xyxy(
    boxes_list,      # list of Tensor [N,4] xyxy per image
    scores_list,     # list of Tensor [N] per image
    labels_list,     # list of Tensor [N] per image
    img_paths,       # list of str: file paths (must align with above)
    img_id_by_filename: Dict[str, int],
    cat_id_by_idx: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Convert DETR model outputs (xyxy boxes) to COCO predictions.

    Caller must provide aligned lists: one entry per val image, in the
    same order as the validation image list used to build img_id_by_filename.
    """
    preds: List[Dict[str, Any]] = []
    for boxes, scores, labels, path in zip(boxes_list, scores_list, labels_list, img_paths):
        img_id = img_id_by_filename.get(path)
        if img_id is None:
            continue
        if boxes.numel() == 0:
            continue
        xyxy = boxes.detach().cpu().numpy()
        sc = scores.detach().cpu().numpy()
        lbl = labels.detach().cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, sc, lbl):
            preds.append({
                "image_id": int(img_id),
                "category_id": int(cat_id_by_idx[c]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
    return preds


def preds_to_json(
    preds: List[Dict[str, Any]],
    out_path: Path,
    *,
    sort: bool = True,
) -> Path:
    """Write COCO-format predictions to disk.

    Parameters
    ----------
    preds : list of COCO detection dicts
    out_path : Path
    sort : bool — sort by image_id then score desc for determinism

    Returns
    -------
    out_path
    """
    if sort:
        preds = sorted(
            preds,
            key=lambda d: (d["image_id"], -d["score"]),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds))
    return out_path


# ---------------------------------------------------------------------------
# Finalise run: save protocol + results together
# ---------------------------------------------------------------------------

def finalize_run(
    out_dir: Path,
    proto: Dict[str, Any],
    results: Dict[str, Any],
    *,
    preds: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write provenance.json (protocol + results + optional preds summary).

    Also optionally writes preds.json if preds are provided.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {
        "protocol": proto,
        "results": results,
    }
    if preds is not None:
        preds_path = out_dir / "preds.json"
        preds_path.write_text(json.dumps(preds))
        doc["n_predictions"] = len(preds)
    (out_dir / "provenance.json").write_text(json.dumps(doc, indent=2))
    return out_dir / "provenance.json"
