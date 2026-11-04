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
# ===========================================================================
# SECTION: RESEARCH-GRADE EXTENSIONS
#
# Split integrity, fast COCO eval, bootstrap CIs, unified latency, Grad-CAM,
# qualitative panels, test-set submission, staged execution, self-test.
#
# Everything below is written so that a failure in one analysis stage can never
# destroy the artefacts of a completed training run: core results are written
# first, optional figures are produced inside guarded stages.
# ===========================================================================

import hashlib
import logging
import sys
from contextlib import contextmanager

LOG = logging.getLogger("fathomnet")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

def split_fingerprint(split: Dict[str, Any]) -> str:
    """Stable short fingerprint of a split's validation ids.

    Two runs share a fingerprint iff they evaluated on exactly the same
    validation images. This value is recorded in every provenance file so a
    paper table can be tied to one specific split.
    """
    payload = json.dumps(
        {"val_ids": sorted(int(i) for i in split["val_ids"]),
         "n_train": len(split["train_ids"])},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_frozen_split(
    path: Optional[Path] = None,
    *,
    require_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the frozen split, failing loudly instead of silently falling back.

    An earlier version of the pipeline fell back to *generating* a local random
    split when ``split.json`` was not mounted. That is dangerous: such a kernel
    trains and evaluates on different images than its siblings, and the models
    stop being comparable. This loader never falls back.

    Raises
    ------
    FileNotFoundError
        if the split file cannot be found in any known mount point.
    ValueError
        if ``require_fingerprint`` is given and does not match.
    """
    candidates: List[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env = os.environ.get("FATHOMNET_SPLIT_JSON")
    if env:
        candidates.append(Path(env))
    candidates.append(SPLIT_JSON)
    for pat in ("*/split.json", "**/split.json"):
        candidates.extend(sorted(Path("/kaggle/input").glob(pat)))

    seen: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for c in seen:
        if c.is_file():
            split = json.loads(c.read_text(encoding="utf-8"))
            fp = split_fingerprint(split)
            LOG.info("split loaded from %s (fingerprint %s)", c, fp)
            if require_fingerprint and fp != require_fingerprint:
                raise ValueError(
                    f"split fingerprint mismatch: expected "
                    f"{require_fingerprint}, got {fp} from {c}"
                )
            return split
    raise FileNotFoundError(
        "frozen split.json not found, and falling back to a locally generated "
        "split would break cross-model comparability. Looked in: "
        f"{[str(c) for c in seen]}"
    )


def resolve_input(*patterns: str, required: bool = True) -> Path:
    """Locate a mounted kernel output by glob pattern, robustly.

    Kaggle mounts a source kernel's output under ``/kaggle/input/<slug>/``.
    Hard-coding that path (as an earlier version of this module did) breaks
    silently whenever the mount name changes, so every notebook resolves its
    inputs through this helper.
    """
    roots = [Path("/kaggle/input"), WORK, WORK.parent / "input"]
    for pat in patterns:
        for root in roots:
            if not root.is_dir():
                continue
            for hit in sorted(root.glob(pat)):
                return hit
    if required:
        raise FileNotFoundError(
            f"none of {patterns} found under /kaggle/input -- is the source "
            "kernel attached and finished?"
        )
    return Path("")


def resolve_dataset_paths() -> Tuple[Path, Path]:
    """Return ``(annotation_json, image_root)`` for the competition data."""
    ann = resolve_input("**/train_dataset.json", "*/train_dataset.json")
    for cand in (ann.parent.parent / "images", ann.parent / "images"):
        if cand.is_dir():
            return ann, cand
    raise FileNotFoundError(f"image root not found next to {ann}")

# ---------------------------------------------------------------------------
# Prediction hygiene — one single place where score_thr / max_det are enforced
# ---------------------------------------------------------------------------

def filter_preds(
    preds: List[Dict[str, Any]],
    *,
    score_thr: float = 0.01,
    max_det: int = 300,
) -> List[Dict[str, Any]]:
    """Apply the paper's confidence threshold and per-image detection cap.

    The reported protocol is "score > 0.01, at most 300 detections per image".
    The original ``Evaluator`` accepted ``score_thr``/``max_det`` arguments but
    never applied them, so the protocol was only honoured for Faster R-CNN
    (which filtered inline) and not for the Ultralytics models. Enforcing it
    here, once, guarantees that every model is scored under identical rules.
    """
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for p in preds:
        by_img.setdefault(int(p["image_id"]), []).append(p)

    out: List[Dict[str, Any]] = []
    for ps in by_img.values():
        keep = [p for p in ps if float(p["score"]) > score_thr]
        keep.sort(key=lambda p: -float(p["score"]))
        out.extend(keep[:max_det])
    out.sort(key=lambda p: (p["image_id"], -float(p["score"])))
    return out


def preds_summary(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detection-count statistics for the provenance record."""
    if not preds:
        return {"n_dets": 0}
    per_img: Dict[int, int] = {}
    scores: List[float] = []
    for p in preds:
        per_img[p["image_id"]] = per_img.get(p["image_id"], 0) + 1
        scores.append(float(p["score"]))
    counts = np.asarray(list(per_img.values()), dtype=float)
    return {
        "n_dets": len(preds),
        "n_images_with_dets": len(per_img),
        "dets_per_image_mean": round(float(counts.mean()), 2),
        "dets_per_image_median": float(np.median(counts)),
        "dets_per_image_max": int(counts.max()),
        "score_mean": round(float(np.mean(scores)), 4),
    }
# ---------------------------------------------------------------------------
# Research-grade COCO evaluator
# ---------------------------------------------------------------------------

class FullEvaluator(Evaluator):
    """COCO bbox evaluator that reports everything the paper needs, from one
    shared code path, with the protocol thresholds actually enforced.

    Compared with :class:`Evaluator` this class

    * enforces ``score_thr`` and ``max_det`` itself (see :func:`filter_preds`),
    * reports per-class AP at both IoU 0.50 and IoU 0.50:0.95 from a single
      evaluate/accumulate pass (the old 32x ``catIds`` re-evaluation loop was
      both slow and unnecessary; slicing the precision tensor is exact and
      ~32x faster),
    * reports per-class AP for the COCO small/medium/large area ranges, which
      is what the promised object-size analysis needs,
    * aggregates head/mid/tail strata, and
    * optionally bootstraps the whole validation set to put confidence
      intervals on every headline number.
    """

    #: area-range labels produced by pycocotools, in evaluation order
    AREA_LABELS = ("all", "small", "medium", "large")

    def evaluate(  # type: ignore[override]
        self,
        preds: List[Dict[str, Any]],
        *,
        out_dir: Optional[Path] = None,
        bootstrap: int = 0,
        bootstrap_seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the full evaluation and optionally bootstrap confidence intervals.

        Parameters
        ----------
        preds : list of COCO detection dicts
        out_dir : Path, optional
            Directory for ``results.json``, ``preds.json``, ``per_class.csv``.
        bootstrap : int
            Number of image-level bootstrap resamples for confidence intervals
            (0 disables; 200 is a good default and costs ~1-2 minutes).
        bootstrap_seed : int
            Seed for the resampling, so intervals are reproducible.
        """
        raw = list(preds)
        used = filter_preds(raw, score_thr=self.score_thr, max_det=self.max_det)

        e = self._eval_once(self._gt_dict_for(list(self.split["val_ids"])), used)
        results = self._collect(e, used)
        results["protocol"] = {
            "score_thr": float(self.score_thr),
            "max_det": int(self.max_det),
            "n_val_images": len(self.split["val_ids"]),
            "split_fingerprint": split_fingerprint(self.split),
        }
        results["preds_raw"] = preds_summary(raw)
        results["preds_used"] = preds_summary(used)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Write immediately (without bootstrap): if the bootstrap stage
            # crashes or the kernel dies, the core metrics are already on disk.
            (out_dir / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            (out_dir / "preds.json").write_text(json.dumps(used), encoding="utf-8")

        if bootstrap:
            results["bootstrap"] = self.bootstrap(
                used, n=bootstrap, seed=bootstrap_seed)
            if out_dir is not None:  # overwrite with the complete record
                (out_dir / "results.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8")

        return results

    # -- internals ---------------------------------------------------------

    def _collect(self, e: COCOeval, used: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Turn a finished COCOeval into the full results dictionary."""
        stats = e.stats
        per_class, per_class50 = self._per_class_ap(e)
        per_class_size = self._per_class_size_ap(e)
        strata = stratify_per_class(per_class, self.groups)
        strata50 = stratify_per_class(per_class50, self.groups)
        strata_size = {
            size: stratify_per_class(
                {n: v.get(size, 0.0) for n, v in per_class_size.items()},
                self.groups,
            )
            for size in ("small", "medium", "large")
        }
        return {
            "mAP50-95": round(float(stats[0]), 4),
            "mAP50": round(float(stats[1]), 4),
            "mAP75": round(float(stats[2]), 4),
            "mAP_small": round(float(stats[3]), 4),
            "mAP_medium": round(float(stats[4]), 4),
            "mAP_large": round(float(stats[5]), 4),
            "per_class_AP50-95": per_class,
            "per_class_AP50": per_class50,
            "per_class_size_AP50-95": per_class_size,
            "head_mid_tail_AP50-95": {g: round(v, 4) for g, v in strata.items()},
            "head_mid_tail_AP50": {g: round(v, 4) for g, v in strata50.items()},
            "head_mid_tail_size_AP50-95": {
                s: {g: round(v, 4) for g, v in d.items()}
                for s, d in strata_size.items()
            },
            "groups": {g: sorted(v) for g, v in self.groups.items()},
            "n_dets": len(used),
        }


    # -- ground truth construction -----------------------------------------

    def _gt_dict_from_pairs(
        self, pairs: Sequence[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """Build a COCO GT dict from ``(original_image_id, new_image_id)`` pairs.

        Working in terms of pairs lets the bootstrap reuse an image more than
        once inside a single draw (each occurrence gets a fresh image id and
        fresh annotation ids) without confusing pycocotools.
        """
        coco_data = _coco_cached()
        img_by_id = {int(i["id"]): i for i in coco_data["images"]}
        anns_by_img: Dict[int, List[dict]] = {}
        for a in coco_data["annotations"]:
            anns_by_img.setdefault(int(a["image_id"]), []).append(a)

        images: List[dict] = []
        anns: List[dict] = []
        ann_id = 1
        for orig, new in pairs:
            info = dict(img_by_id[int(orig)])
            info["id"] = int(new)
            images.append(info)
            for a in anns_by_img.get(int(orig), []):
                b = dict(a)
                b["image_id"] = int(new)
                b["id"] = ann_id
                ann_id += 1
                if not b.get("area"):
                    b["area"] = float(b["bbox"][2] * b["bbox"][3])
                anns.append(b)
        return {
            "images": images,
            "annotations": anns,
            "categories": coco_data["categories"],
        }

    def _gt_dict_for(self, img_ids: Sequence[int]) -> Dict[str, Any]:
        """GT dict for an unmodified list of validation image ids."""
        return self._gt_dict_from_pairs([(int(i), int(i)) for i in img_ids])

    def _eval_once(
        self, gt_dict: Dict[str, Any], preds: List[Dict[str, Any]]
    ) -> COCOeval:
        """Run one evaluate/accumulate pass over a GT dict and predictions."""
        tmp = WORK / "_eval_tmp_gt.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(gt_dict), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(tmp))
            dt = coco.loadRes(preds) if preds else None
            e = COCOeval(coco, dt, "bbox")
            e.evaluate()
            e.accumulate()
            e.summarize()   # populates e.stats (only summarize() fills it)
        return e

    # -- metric extraction -------------------------------------------------

    def _per_class_ap(
        self, e: COCOeval
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Per-class AP at IoU[.50:.95] and at IoU .50, from one eval pass.

        ``e.eval['precision']`` has shape [T, R, K, A, M] (IoU thresholds,
        recall thresholds, categories, area ranges, max-detections). Slicing
        category ``k`` at area range ``all`` and maxDet 100 is exactly the
        quantity the old implementation obtained by re-running ``evaluate()``
        once per category, so the numbers agree (this was verified against the
        archived Faster R-CNN results) but it is ~32x cheaper.
        """
        name_by_id = self.cat_info["cat_id_to_name"]
        prec = e.eval["precision"]
        ap: Dict[str, float] = {}
        ap50: Dict[str, float] = {}
        for k, cid in enumerate(e.params.catIds):
            name = name_by_id[int(cid)]
            s = prec[:, :, k, 0, 2]
            s = s[s > -1]
            ap[name] = round(float(s.mean()), 4) if s.size else 0.0
            s50 = prec[0, :, k, 0, 2]
            s50 = s50[s50 > -1]
            ap50[name] = round(float(s50.mean()), 4) if s50.size else 0.0
        return ap, ap50

    def _per_class_size_ap(
        self, e: COCOeval
    ) -> Dict[str, Dict[str, float]]:
        """Per-class AP for the COCO small/medium/large area ranges."""
        name_by_id = self.cat_info["cat_id_to_name"]
        labels = list(e.params.areaRngLbl)
        prec = e.eval["precision"]
        out: Dict[str, Dict[str, float]] = {}
        for k, cid in enumerate(e.params.catIds):
            name = name_by_id[int(cid)]
            row: Dict[str, float] = {}
            for lab in ("small", "medium", "large"):
                if lab not in labels:
                    row[lab] = 0.0
                    continue
                s = prec[:, :, k, labels.index(lab), 2]
                s = s[s > -1]
                row[lab] = round(float(s.mean()), 4) if s.size else 0.0
            out[name] = row
        return out
# -- confidence intervals ----------------------------------------------

    def bootstrap(
        self,
        preds: List[Dict[str, Any]],
        *,
        n: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Image-level bootstrap confidence intervals for every headline metric.

        The validation set has only 646 images and several tail classes hold a
        handful of instances, so point estimates of mAP, and especially of the
        stratum means, are noisy. Each of the ``n`` draws resamples the
        validation images *with replacement* and re-runs the full COCO
        evaluation on that resample, propagating both image-level and
        class-level sampling variability into the reported interval.

        Images drawn more than once inside a draw get distinct image ids (and
        their annotations distinct annotation ids) so pycocotools still sees a
        valid COCO file. Intervals are the 2.5th/97.5th percentiles, i.e. a
        percentile bootstrap 95% CI.
        """
        if n <= 0:
            return {"n": 0}
        rng = np.random.default_rng(seed)
        val = np.asarray([int(i) for i in self.split["val_ids"]])
        offset = 10_000_000

        overall: Dict[str, List[float]] = {"mAP50-95": [], "mAP50": []}
        strata: Dict[str, List[float]] = {g: [] for g in self.groups}
        per_class: Dict[str, List[float]] = {}

        for b in range(n):
            sample = rng.choice(val, size=val.size, replace=True)
            pairs: List[Tuple[int, int]] = []
            occ: Dict[int, List[int]] = {}
            for j, orig in enumerate(sample):
                new_id = (b + 1) * offset + j
                pairs.append((int(orig), new_id))
                occ.setdefault(int(orig), []).append(new_id)

            draw_preds: List[Dict[str, Any]] = []
            for p in preds:
                for new_id in occ.get(int(p["image_id"]), ()):
                    q = dict(p)
                    q["image_id"] = new_id
                    draw_preds.append(q)

            e = self._eval_once(self._gt_dict_from_pairs(pairs), draw_preds)
            overall["mAP50-95"].append(float(e.stats[0]))
            overall["mAP50"].append(float(e.stats[1]))
            pc, _ = self._per_class_ap(e)
            for g, names in self.groups.items():
                vals = [pc[nm] for nm in names if nm in pc]
                strata[g].append(float(np.mean(vals)) if vals else 0.0)
            for nm, v in pc.items():
                per_class.setdefault(nm, []).append(v)

        return {
            "n": n,
            "seed": seed,
            "method": "image-level percentile bootstrap (2.5th/97.5th pct)",
            "overall": {k: _ci(v) for k, v in overall.items()},
            "head_mid_tail": {g: _ci(v) for g, v in strata.items()},
            "per_class": {nm: _ci(v) for nm, v in per_class.items()},
        }
# ---------------------------------------------------------------------------
# Small module-level helpers used above
# ---------------------------------------------------------------------------

_COCO_CACHE: Dict[str, Any] = {}


def _coco_cached(path: Optional[Path] = None) -> Dict[str, Any]:
    """Cache the (large) annotation JSON across evaluator calls."""
    key = str(path or DATASET_JSON)
    if key not in _COCO_CACHE:
        _COCO_CACHE[key] = load_coco_data(path)
    return _COCO_CACHE[key]


def _ci(vals: Sequence[float]) -> Dict[str, float]:
    """Percentile-bootstrap summary of a sample of metric values."""
    a = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "lo": round(float(np.percentile(a, 2.5)), 4),
        "hi": round(float(np.percentile(a, 97.5)), 4),
        "sd": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Staged execution: an analysis failure must never lose a training result
# ---------------------------------------------------------------------------

@contextmanager
def stage(name: str, report: Optional[Dict[str, Any]] = None):
    """Run an optional analysis stage without ever killing the run.

    Training costs hours of GPU time; a plotting or CAM bug costs seconds.
    Every optional stage is wrapped in this context manager so that a failure
    is recorded in ``report['stages']`` and the notebook continues to the next
    stage with the trained weights and core metrics already on disk.
    """
    t0 = time.time()
    try:
        yield
        status = {"status": "ok", "seconds": round(time.time() - t0, 2)}
        LOG.info("stage %-22s ok (%.1fs)", name, time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        status = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - t0, 2),
        }
        LOG.warning("stage %-22s FAILED -> %s", name, status["error"])
    if report is not None:
        report.setdefault("stages", {})[name] = status


# ---------------------------------------------------------------------------
# System / reproducibility info
# ---------------------------------------------------------------------------

def system_info() -> Dict[str, Any]:
    """Versions, hardware and peak memory, for the provenance record."""
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for mod in ("torchvision", "ultralytics", "pycocotools", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    if torch.cuda.is_available():
        try:
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            info["peak_gpu_memory_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:
            pass
    return info


def save_json(path: Path, obj: Any) -> Path:
    """Write JSON atomically enough for notebook use (write + report path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    """Read a JSON file, returning ``None`` when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
# ---------------------------------------------------------------------------
# Unified latency / throughput benchmark
# ---------------------------------------------------------------------------

def bench_latency(
    fn,
    *,
    warmup: int = 5,
    iters: int = 30,
    sync: bool = True,
) -> Dict[str, Any]:
    """Median wall-clock latency of ``fn()`` under one measurement protocol.

    The earlier paper table mixed two conventions (a bare CUDA-synchronised
    forward pass for Faster R-CNN, a full ``predict()`` call including
    pre/post-processing for the Ultralytics models), which makes the reported
    speed-ups meaningless. Every model now reports both:

    * ``net_*`` -- network forward only, CUDA-synchronised, synthetic input at
      the model's native resolution. This is the convention used for the
      accuracy/cost comparison.
    * ``e2e_*`` -- the framework's full ``predict()`` call on a real image,
      i.e. what a deployment pipeline actually pays per frame.

    Parameters
    ----------
    fn : callable
        Zero-argument callable performing one inference.
    warmup, iters : int
        Warm-up and timed iterations.
    sync : bool
        Call ``torch.cuda.synchronize()`` around every timed call.
    """
    def _one() -> float:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        _one()
    times = np.asarray([_one() for _ in range(iters)], dtype=float)
    return {
        "n": int(iters),
        "median_ms": round(float(np.median(times)), 2),
        "mean_ms": round(float(times.mean()), 2),
        "p90_ms": round(float(np.percentile(times, 90)), 2),
        "min_ms": round(float(times.min()), 2),
        "std_ms": round(float(times.std(ddof=1)) if times.size > 1 else 0.0, 2),
    }


def bench_memory_peak(reset: bool = True) -> float:
    """Peak allocated CUDA memory in GiB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)
# ---------------------------------------------------------------------------
# Grad-CAM for detectors
#
# A detector outputs a variable set of boxes, so there is no single class logit
# to differentiate. The scheme used here is the standard detector-CAM recipe:
# hook one activation tensor A, build a scalar objective S = the detector's own
# score for the class of interest, then
#     CAM = ReLU( sum_c mean_hw(dS/dA_c) * A_c ).
# Because the objective is the network's own score for the thing being
# explained, the map stays faithful to the detector rather than to an
# auxiliary classifier.
# ---------------------------------------------------------------------------

class _ActivationHook:
    """Capture a module's forward activation and its gradient."""

    def __init__(self, module) -> None:
        self.act = None
        self.grad = None
        self._h = module.register_forward_hook(self._fwd)
        self._h2 = module.register_full_backward_hook(self._bwd)

    def _fwd(self, _module, _inp, out):
        self.act = out

    def _bwd(self, _module, _grad_in, grad_out):
        self.grad = grad_out[0]

    def close(self) -> None:
        self._h.remove()
        self._h2.remove()


def cam_from_hook(hook: "_ActivationHook") -> np.ndarray:
    """Turn captured activations + gradients into a normalised 2-D CAM."""
    if hook.act is None or hook.grad is None:
        raise RuntimeError("CAM hook captured nothing; wrong target layer?")
    act = hook.act[0].detach()
    grad = hook.grad[0].detach()
    if act.dim() == 3:  # transformer tokens [N, C]: no spatial grid to pool
        cam = (grad * act).sum(dim=-1).reshape(1, -1)
    else:
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0, keepdim=True))
    arr = cam.squeeze().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_cam(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Resize a CAM to the image size and overlay it as a JET heat map."""
    import cv2

    h, w = image_rgb.shape[:2]
    heat = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat, 0, 1)
    coloured = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return np.clip(
        (1 - alpha) * image_rgb.astype(np.float32)
        + alpha * coloured.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def cam_target_layer(model, family: str):
    """Pick a Grad-CAM target module for a detector family.

    Returns ``(module, name)``. Modules are chosen by explicit name with a
    discovery fallback, so a framework update that renames a module degrades
    CAM quality rather than crashing the run.
    """
    named = dict(model.named_modules())

    def first(*suffixes):
        for suf in suffixes:
            for name, mod in reversed(list(named.items())):
                if name.endswith(suf):
                    return mod, name
        return None, None

    if family == "torchvision_frcnn":
        mod, name = first("backbone.body.layer4", "backbone.body.layer3")
    elif family == "ultralytics_yolo":
        mod, name = first("model.9.cv2", "model.9.cv1", "model.9")
    elif family == "ultralytics_rtdetr":
        mod, name = first("model.encoder", "model.decoder")
    else:
        raise ValueError(f"unknown family {family!r}")

    if mod is None:
        raise RuntimeError(f"no Grad-CAM target layer found for {family}")
    return mod, name


def gradcam(forward_score, target_module, images) -> np.ndarray:
    """Compute a normalised CAM for one input batch.

    ``forward_score(images)`` must return a *differentiable scalar* tensor;
    ``target_module`` is the layer to hook.
    """
    hook = _ActivationHook(target_module)
    try:
        score = forward_score(images)
        if score is None or not score.requires_grad:
            raise RuntimeError("CAM objective is not differentiable")
        score.backward()
        return cam_from_hook(hook)
    finally:
        hook.close()
def _first_tensor(obj):
    """Find the first torch.Tensor inside (possibly nested) model output."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _ultralytics_forward(net, images):
    """Call an Ultralytics nn.Module with autograd enabled.

    ``BaseModel.forward`` dispatches to ``predict`` for the YOLO-detection and
    RT-DETR heads used here, neither of which is wrapped in an inference-mode
    context, so gradients flow. If a future release changes that, the CAM
    stage fails loudly and is skipped by its enclosing :func:`stage` guard
    instead of killing the run.
    """
    try:
        return net(images)
    except TypeError:
        return net([images])


def cam_objective(family: str, model, class_idx: int, *, topk: int = 5):
    """Return a differentiable scalar objective for Grad-CAM.

    Parameters
    ----------
    family : {'torchvision_frcnn', 'ultralytics_yolo', 'ultralytics_rtdetr'}
    model : the loaded model (torchvision detector or Ultralytics wrapper)
    class_idx : int
        For ``torchvision_frcnn`` this is the *network* label, i.e. the
        FathomNet ``category_id + 1`` (0 is background). For the Ultralytics
        families it is the contiguous ``0..31`` dataset index.
    topk : int
        Number of the detector's own boxes/queries contributing to the score.
    """
    if family == "torchvision_frcnn":
        def objective(images):
            out = model(images)[0]
            sel = out["labels"] == int(class_idx)
            if not bool(sel.any()):
                sel = torch.ones_like(out["labels"], dtype=torch.bool)
            return out["scores"][sel][:topk].sum()

        return objective

    net = getattr(model, "model", None)
    if net is None:
        raise RuntimeError("Ultralytics model has no .model attribute")

    if family == "ultralytics_yolo":
        def objective(images):
            preds = _first_tensor(_ultralytics_forward(net, images))
            if preds is None:
                raise RuntimeError("no prediction tensor in YOLO output")
            if preds.dim() == 4:                       # [B, 4+nc, H, W]
                return preds[0, 4 + class_idx].max()
            return preds[0, :, 4 + class_idx].max()     # [B, N, 4+nc]

        return objective

    if family == "ultralytics_rtdetr":
        def objective(images):
            preds = _first_tensor(_ultralytics_forward(net, images))
            if preds is None:
                raise RuntimeError("no prediction tensor in RT-DETR output")
            if preds.dim() == 4:                       # [B, 4+nc, H, W]
                return preds[0, 4 + class_idx].max()
            cls = preds[0, :, 4:]                       # [N, nc]
            vals = cls[:, class_idx] if cls.shape[-1] > class_idx else cls.amax(-1)
            return vals.topk(min(topk, vals.numel())).values.sum()

        return objective

    raise ValueError(f"unknown family {family!r}")
# ---------------------------------------------------------------------------
# Figure panels (shared by the Grad-CAM and qualitative figures)
# ---------------------------------------------------------------------------

def panel_figure(
    samples: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "",
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
) -> Path:
    """Render a grid of annotated detection examples to PNG and PDF.

    Each sample dict may contain:
        rgb     : HxWx3 uint8 image to display (CAM-overlaid if that is wanted)
        gt      : list of ``(xyxy, name)`` ground-truth boxes      (green)
        pred    : list of ``(xyxy, name, score)`` predictions      (red, dashed)
        caption : optional per-panel caption
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    n = len(samples)
    if n == 0:
        raise ValueError("panel_figure received no samples")
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, s in zip(axes.ravel(), samples):
        ax.imshow(s["rgb"])
        for box, name in s.get("gt", []):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, ec="#2ca02c", lw=1.6))
            ax.text(x1, max(y1 - 4, 8), name, fontsize=6, color="white",
                    bbox={"fc": "#2ca02c", "alpha": .8, "pad": 1.2})
        for box, name, score in s.get("pred", []):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, ec="#d62728", lw=1.4, ls="--"))
            ax.text(x1, min(max(y2 + 8, 8), s["rgb"].shape[0] - 4),
                    f"{name} {score:.2f}", fontsize=6, color="white",
                    bbox={"fc": "#d62728", "alpha": .8, "pad": 1.2})
        if s.get("caption"):
            ax.set_title(s["caption"], fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """COCO ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``."""
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def match_predictions(
    gt: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
    *,
    iou_thr: float = 0.5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Greedy IoU matching of one image's predictions against its GT.

    Returns ``{'tp': [...], 'fp': [...], 'fn': [...]}`` where TP entries carry
    the matched GT, used to build qualitative success/failure panels.
    """
    remaining = list(range(len(gt)))
    tp: List[Dict[str, Any]] = []
    fp: List[Dict[str, Any]] = []
    for p in sorted(preds, key=lambda d: -float(d["score"])):
        best_j, best_iou = -1, 0.0
        for j in remaining:
            if gt[j]["category_id"] != p["category_id"]:
                continue
            overlap = iou_xyxy(xywh_to_xyxy(p["bbox"]), gt[j]["_xyxy"])
            if overlap > best_iou:
                best_j, best_iou = j, overlap
        if best_j >= 0 and best_iou >= iou_thr:
            tp.append({"pred": p, "gt": gt[best_j], "iou": round(best_iou, 3)})
            remaining.remove(best_j)
        else:
            fp.append({"pred": p})
    fn = [{"gt": gt[j]} for j in remaining]
    return {"tp": tp, "fp": fp, "fn": fn}
# ---------------------------------------------------------------------------
# Blind test-set inference and submission file
# ---------------------------------------------------------------------------

def load_test_data(ann_path: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    """Load ``test_dataset.json`` (images only; labels are withheld)."""
    if ann_path is None:
        ann_path = resolve_input("**/test_dataset.json", "*/test_dataset.json")
    ann_path = Path(ann_path)
    root = ann_path.parent.parent / "images"
    if not root.is_dir():
        root = ann_path.parent / "images"
    return json.loads(ann_path.read_text(encoding="utf-8")), root


def write_submission(
    preds: Sequence[Dict[str, Any]],
    sample_submission: Path,
    out_csv: Path,
    *,
    image_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Write a submission file conforming to the organisers' sample file.

    The exact competition format is read from ``sample_submission.csv`` at run
    time instead of being hard-coded, so the notebook cannot silently emit a
    file in the wrong shape. Two shapes are supported:

    * a single ``prediction_string`` column per image, formatted as repeating
      ``score x y w h`` groups (the usual Kaggle detection convention), and
    * one row per detection with separate ``bbox`` / ``score`` columns.

    A row is emitted for every test image (empty prediction string when nothing
    was found), so the row count always matches the sample file.
    """
    import csv as _csv

    sample_submission = Path(sample_submission)
    with sample_submission.open(newline="", encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        sample_rows = list(reader)

    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for p in preds:
        by_img.setdefault(int(p["image_id"]), []).append(p)
    for ps in by_img.values():
        ps.sort(key=lambda d: -float(d["score"]))

    ids = [int(i) for i in image_ids] if image_ids else [
        int(r[0]) for r in sample_rows if r and r[0].strip()
    ]

    fmt = "prediction_string" if "prediction_string" in header else "columns"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(header)
        if fmt == "prediction_string":
            for iid in ids:
                parts = [
                    f"{float(p['score']):.5f} {float(p['bbox'][0]):.2f} "
                    f"{float(p['bbox'][1]):.2f} {float(p['bbox'][2]):.2f} "
                    f"{float(p['bbox'][3]):.2f}"
                    for p in by_img.get(iid, [])
                ]
                writer.writerow([iid, " ".join(parts)])
        else:
            for iid in ids:
                for p in by_img.get(iid, []):
                    bbox = " ".join(f"{v:.2f}" for v in p["bbox"])
                    row = []
                    for col in header:
                        key = col.strip().lower()
                        if key in ("image_id", "imageid", "id"):
                            row.append(iid)
                        elif key == "category_id":
                            row.append(p["category_id"])
                        elif key == "bbox":
                            row.append(bbox)
                        elif key in ("score", "confidence"):
                            row.append(f"{float(p['score']):.5f}")
                        else:
                            row.append("")
                    writer.writerow(row)

    return {
        "format": fmt,
        "header": header,
        "n_rows": len(ids),
        "n_images_with_dets": len(by_img),
        "n_dets": len(preds),
        "path": str(out_csv),
    }
# ---------------------------------------------------------------------------
# Self-test: cheap assertions that catch wiring errors before GPU time is spent
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> Dict[str, Any]:
    """Validate the pure-python parts of this module on synthetic data.

    Runs on CPU in well under a second and is called by every training notebook
    *before* any GPU work, so a broken metric, a wrong group definition or a
    mis-plumbed filter fails immediately instead of after hours of training.
    Raises AssertionError on failure.
    """
    checks: Dict[str, Any] = {}

    # 1. IoU + box conversion
    assert abs(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6
    assert abs(iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20])) < 1e-9
    assert xywh_to_xyxy([1, 2, 3, 4]) == [1.0, 2.0, 4.0, 6.0]
    checks["iou"] = "ok"

    # 2. score_thr / max_det enforcement
    raw = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": s}
           for s in (0.001, 0.5, 0.9, 0.2)]
    assert len(filter_preds(raw, score_thr=0.01, max_det=2)) == 2
    assert len(filter_preds(raw, score_thr=0.5, max_det=300)) == 1
    kept = filter_preds(raw, score_thr=0.01, max_det=2)
    assert [round(p["score"], 2) for p in kept] == [0.9, 0.5], kept
    checks["filter_preds"] = "ok"

    # 3. head/mid/tail grouping is frequency-ordered and partitions the classes
    counts = {f"c{i}": n for i, n in enumerate([10, 9, 8, 7, 6, 5, 4, 3])}
    g = define_groups(counts, head_k=3, mid_k=2)
    assert g["head"] == {"c0", "c1", "c2"}, g["head"]
    assert g["mid"] == {"c3", "c4"}, g["mid"]
    assert g["tail"] == {"c5", "c6", "c7"}, g["tail"]
    assert len(g["head"] | g["mid"] | g["tail"]) == 8
    checks["define_groups"] = "ok"

    # 4. stratum aggregation is an unweighted mean over the stratum's classes
    got = stratify_per_class({"c0": 0.5, "c1": 0.7, "c2": 0.3, "c3": 1.0},
                            {"head": {"c0", "c1", "c2"}})
    assert abs(got["head"] - 0.5) < 1e-9, got
    checks["stratify"] = "ok"

    # 5. percentile CI behaves
    ci = _ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert ci["lo"] <= ci["mean"] <= ci["hi"], ci
    assert abs(ci["mean"] - 0.3) < 1e-9, ci
    checks["_ci"] = "ok"

    # 6. split fingerprint is stable and order-insensitive
    fp1 = split_fingerprint({"val_ids": [3, 1, 2], "train_ids": [4, 5]})
    fp2 = split_fingerprint({"val_ids": [1, 2, 3], "train_ids": [4, 5]})
    fp3 = split_fingerprint({"val_ids": [1, 2, 4], "train_ids": [4, 5]})
    assert fp1 == fp2 and fp1 != fp3
    checks["split_fingerprint"] = "ok"

    # 7. greedy matcher finds a true positive and reports wrong-class misses
    gt = [{"category_id": 1, "_xyxy": [0, 0, 10, 10]}]
    m = match_predictions(gt, [{"category_id": 1, "bbox": [0, 0, 10, 10],
                                "score": 0.9}])
    assert len(m["tp"]) == 1 and not m["fp"] and not m["fn"], m
    m2 = match_predictions(gt, [{"category_id": 2, "bbox": [0, 0, 10, 10],
                                 "score": 0.9}])
    assert not m2["tp"] and len(m2["fp"]) == 1 and len(m2["fn"]) == 1, m2
    checks["match_predictions"] = "ok"

    # 8. preds_summary sanity
    ps = preds_summary(raw)
    assert ps["n_dets"] == 4 and ps["dets_per_image_max"] == 4, ps
    checks["preds_summary"] = "ok"

    if verbose:
        LOG.info("selftest passed: %s",
                 ", ".join(f"{k}={v}" for k, v in checks.items()))
    return checks