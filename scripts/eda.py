"""Exploratory data analysis for FathomNetCLEF2026.

Works locally and in Kaggle notebooks:
    python scripts/eda.py --ann-dir data/annotations --image-dir data/images

Produces (in --out-dir, default 'experiments/eda'):
    class_counts.csv        per-class annotation counts
    bbox_stats.csv          width/height/area statistics
    class_counts.png        long-tail bar chart (paper figure)
    sample_grid.png         annotated sample grid (paper figure)
and prints key numbers for the paper's dataset section.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def eda(ann_dir: Path, image_dir: Path, out_dir: Path, n_samples: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_json(ann_dir / "train_dataset.json")
    test = load_json(ann_dir / "test_dataset.json")

    cats = {c["id"]: c["name"] for c in train["categories"]}
    n_imgs, n_anns = len(train["images"]), len(train["annotations"])

    # ---- headline stats -------------------------------------------------
    imgs_per_annot = len({a["image_id"] for a in train["annotations"]})
    boxes_per_img = [len(v) for v in {}]  # computed below
    anns_by_img: dict[int, list] = {}
    for a in train["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    boxes_per_img = [len(anns_by_img.get(i["id"], [])) for i in train["images"]]

    print("=" * 60)
    print("FathomNetCLEF2026 dataset summary")
    print("=" * 60)
    print(f"Train images        : {n_imgs}")
    print(f"Train annotations   : {n_anns}")
    print(f"Images w/ >=1 label : {imgs_per_annot} ({imgs_per_annot / n_imgs:.1%})")
    print(f"Avg boxes / image   : {np.mean(boxes_per_img):.2f}  "
          f"(median {np.median(boxes_per_img):.0f}, max {np.max(boxes_per_img)})")
    print(f"Classes             : {len(cats)}")
    print(f"Test images         : {len(test['images'])} "
          f"({len(test.get('annotations', []))} annotations)")

    # ---- class counts ---------------------------------------------------
    counts = Counter(cats[a["category_id"]] for a in train["annotations"])
    df = pd.DataFrame(counts.most_common(), columns=["class", "count"])
    df.to_csv(out_dir / "class_counts.csv", index=False)
    print(f"\nHead classes: {df.head(5).to_dict('records')}")
    print(f"Tail classes: {df.tail(5).to_dict('records')}")

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(df["class"][::-1], df["count"][::-1], color="#3a7ca5")
    ax.set_xscale("log")
    ax.set_xlabel("annotations (log scale)")
    ax.set_title("FathomNetCLEF2026 class distribution (long-tail)")
    plt.tight_layout()
    fig.savefig(out_dir / "class_counts.png", dpi=200)
    plt.close(fig)

    # ---- bbox stats ------------------------------------------------------
    b = np.array([a["bbox"] for a in train["annotations"]])
    bdf = pd.DataFrame({
        "width": b[:, 2], "height": b[:, 3], "area": b[:, 2] * b[:, 3],
    })
    bdf.describe().to_csv(out_dir / "bbox_stats.csv")
    print("\nBbox stats:\n", bdf.describe().round(1))

    # ---- sample grid -----------------------------------------------------
    have_files = [
        i for i in train["images"]
        if (image_dir / i["file_name"]).exists()
    ][:n_samples]
    if have_files:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for ax, im in zip(axes.ravel(), have_files):
            ax.imshow(Image.open(image_dir / im["file_name"]))
            for a in anns_by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                               ec="red", lw=1.5))
                ax.text(x, max(y - 4, 8), cats[a["category_id"]], color="yellow",
                        fontsize=7, bbox={"fc": "black", "alpha": 0.6, "pad": 1})
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "sample_grid.png", dpi=200)
        plt.close(fig)
        print(f"\nSaved figures to {out_dir}")
    else:
        print(f"\n(No images found in {image_dir} - figures skipped; "
              f"stats and CSVs are complete.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="data/annotations")
    ap.add_argument("--image-dir", default="data/images")
    ap.add_argument("--out-dir", default="experiments/eda")
    args = ap.parse_args()
    eda(Path(args.ann_dir), Path(args.image_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
