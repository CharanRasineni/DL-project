"""Generate publication-quality figures for the FathomNet CLEF 2026 paper:
1. fig_class_imbalance: 32-class long-tail distribution & cumulative concentration curve
2. fig_performance_tradeoff: Head/mid/tail AP degradation & latency vs mAP Pareto trade-off
3. fig_geometry: Image resolution, bounding box scale, aspect ratio, and object density audit
4. fig_gradcam_large: High-resolution, large-format 4-row Grad-CAM visual attribution
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# =========================================================================
# FIGURE 1: Taxonomic Class Distribution & Long-Tail Imbalance
# =========================================================================
def make_fig_class_imbalance():
    csv_path = Path(__file__).parent.parent / "notebooks/kaggle/02_split/output/table_split_class_counts.csv"
    df = pd.read_csv(csv_path, index_col=0)
    df = df.sort_values(by="train", ascending=True)  # ascending for horizontal bar

    strata_colors = []
    head_names = {"urchin", "bony fish", "brittle star"}
    mid_names = {"sea fan", "sponge", "anemone", "stony coral"}
    
    for cls in df.index:
        if cls in head_names:
            strata_colors.append("#1f77b4")  # Head: Navy/Blue
        elif cls in mid_names:
            strata_colors.append("#2ca02c")  # Mid: Green
        else:
            strata_colors.append("#d95f02")  # Tail: Coral/Orange

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 7.5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel A: Horizontal bar plot of instances (log scale)
    y_pos = np.arange(len(df))
    ax1.barh(y_pos, df["train"], color=strata_colors, alpha=0.85, edgecolor="none", height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df.index, fontsize=8.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Instances (log scale)", fontweight="bold")
    ax1.set_title("(a) Taxonomic Class Counts (32 Marine Classes)", pad=10, fontweight="bold")
    ax1.set_xlim(4, 10000)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    # Custom legend for strata
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Head: Top 3 (53.4% of data)"),
        Patch(facecolor="#2ca02c", label="Mid: Next 4 (27.3% of data)"),
        Patch(facecolor="#d95f02", label="Tail: Remaining 25 (19.4% of data)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=8.5)

    # Panel B: Cumulative instance concentration curve
    sorted_counts = df["train"].sort_values(ascending=False).values
    cum_share = np.cumsum(sorted_counts) / sorted_counts.sum() * 100.0
    ranks = np.arange(1, len(sorted_counts) + 1)

    ax2.plot(ranks, cum_share, marker="o", color="#08519c", lw=2.2, ms=4.5, label="Cumulative instance share")
    ax2.axvline(3, color="#1f77b4", linestyle=":", lw=1.5, label="Head cutoff (rank 3: 53.4%)")
    ax2.axvline(7, color="#2ca02c", linestyle=":", lw=1.5, label="Mid cutoff (rank 7: 80.7%)")
    ax2.axhline(53.4, color="#1f77b4", linestyle="--", alpha=0.5, lw=1)
    ax2.axhline(80.7, color="#2ca02c", linestyle="--", alpha=0.5, lw=1)
    
    # Lorenz/equality reference
    ax2.plot([1, 32], [100.0 / 32, 100.0], color="gray", linestyle="-.", lw=1.2, label="Uniform distribution")

    ax2.set_xlim(1, 32)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Class Rank (Ordered by Frequency)", fontweight="bold")
    ax2.set_ylabel("Cumulative Share of Annotations (%)", fontweight="bold")
    ax2.set_title("(b) Concentration of Training Annotations", pad=10, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_class_imbalance.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_class_imbalance.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_class_imbalance")

# =========================================================================
# FIGURE 2: Performance Trade-off & Long-Tail Degradation
# =========================================================================
def make_fig_performance_tradeoff():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A: Stratum degradation
    strata = ["Head (3 classes)", "Mid (4 classes)", "Tail (25 classes)"]
    x = np.arange(len(strata))
    width = 0.25

    frcnn_vals = [0.451, 0.424, 0.305]
    yolo_vals  = [0.436, 0.408, 0.283]
    rtd_vals   = [0.429, 0.345, 0.189]

    ax1.bar(x - width, frcnn_vals, width, label="Faster R-CNN R50-FPN", color="#1f77b4", alpha=0.9)
    ax1.bar(x,         yolo_vals,  width, label="YOLOv8s",              color="#2ca02c", alpha=0.9)
    ax1.bar(x + width, rtd_vals,   width, label="RT-DETR-L",            color="#d95f02", alpha=0.9)

    ax1.set_ylabel("Average Precision (mAP$_{50-95}$)", fontweight="bold")
    ax1.set_title("(a) Accuracy Degradation Across Taxonomic Strata", pad=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strata, fontweight="bold")
    ax1.set_ylim(0, 0.55)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    for i in range(len(strata)):
        ax1.text(x[i] - width, frcnn_vals[i] + 0.012, f"{frcnn_vals[i]:.3f}", ha="center", fontsize=8)
        ax1.text(x[i],         yolo_vals[i]  + 0.012, f"{yolo_vals[i]:.3f}",  ha="center", fontsize=8)
        ax1.text(x[i] + width, rtd_vals[i]   + 0.012, f"{rtd_vals[i]:.3f}",   ha="center", fontsize=8)

    # Panel B: Latency vs Accuracy Pareto trade-off
    models = [
        {"name": "Faster R-CNN\nR50-FPN", "latency": 46.4, "map": 0.333, "params": 41.3, "color": "#1f77b4", "fps": 21.6},
        {"name": "YOLOv8s",              "latency": 24.0, "map": 0.313, "params": 11.1, "color": "#2ca02c", "fps": 41.7},
        {"name": "RT-DETR-L",            "latency": 62.2, "map": 0.231, "params": 32.9, "color": "#d95f02", "fps": 16.1},
    ]

    for m in models:
        # size scaled with parameters
        size = m["params"] * 22
        ax2.scatter(m["latency"], m["map"], s=size, color=m["color"], alpha=0.8, edgecolors="black", lw=1.2)
        offset_y = 0.008 if m["name"] != "YOLOv8s" else -0.018
        offset_x = 0
        ax2.annotate(
            f"{m['name']}\n({m['params']}M params, {m['fps']:.1f} FPS)",
            xy=(m["latency"], m["map"]),
            xytext=(m["latency"] + offset_x, m["map"] + offset_y),
            ha="center", va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=m["color"], alpha=0.85, lw=1)
        )

    # Pareto frontier line between YOLOv8s and Faster R-CNN
    ax2.plot([24.0, 46.4], [0.313, 0.333], "k--", alpha=0.4, label="Pareto-optimal frontier")

    ax2.set_xlabel("Inference Latency per Image (ms, batch=1 on T4)", fontweight="bold")
    ax2.set_ylabel("Overall Accuracy (mAP$_{50-95}$)", fontweight="bold")
    ax2.set_title("(b) Latency vs. Accuracy Trade-Off (bubble size $\\propto$ params)", pad=10, fontweight="bold")
    ax2.set_xlim(15, 75)
    ax2.set_ylim(0.20, 0.37)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_performance_tradeoff.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_performance_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_performance_tradeoff")

# =========================================================================
# FIGURE 3: Image & Object Geometry Audit (Replacing the empty box)
# =========================================================================
def make_fig_geometry():
    gt_path = Path(__file__).parent.parent / "notebooks/kaggle/03_fasterrcnn/output/val_gt.json"
    with open(gt_path) as f:
        gt = json.load(f)

    imgs = pd.DataFrame(gt["images"])
    anns = pd.DataFrame(gt["annotations"])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # Panel A: Image frame resolutions
    res_counts = imgs.groupby(["width", "height"]).size().reset_index(name="count")
    labels = [f"{w}x{h}\n({w/h:.2f}:1 Aspect)" for w, h in zip(res_counts["width"], res_counts["height"])]
    bars = axes[0].bar(labels, res_counts["count"], color=["#3182bd", "#6baed6"], width=0.5, edgecolor="black", lw=0.8)
    axes[0].set_ylabel("Number of Images", fontweight="bold")
    axes[0].set_title("(a) Validation Image Resolutions (N=646)", pad=10, fontweight="bold")
    axes[0].set_ylim(0, 400)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 10, f"{b.get_height()} imgs\n({b.get_height()/len(imgs)*100:.1f}%)", ha="center", fontsize=8.5)

    # Panel B: Bounding Box Size Breakdown (COCO convention)
    anns["area_sqrt"] = np.sqrt(anns["area"])
    small = (anns["area_sqrt"] < 32).sum()
    medium = ((anns["area_sqrt"] >= 32) & (anns["area_sqrt"] < 96)).sum()
    large = (anns["area_sqrt"] >= 96).sum()

    cat_names = ["Small\n(<32 px)", "Medium\n(32-96 px)", "Large\n(>96 px)"]
    cat_counts = [small, medium, large]
    cat_colors = ["#fd8d3c", "#74c476", "#3182bd"]

    bars = axes[1].bar(cat_names, cat_counts, color=cat_colors, width=0.55, edgecolor="black", lw=0.8)
    axes[1].set_ylabel("Bounding Box Instances", fontweight="bold")
    axes[1].set_title("(b) COCO Object Scale Mix (N=2,102)", pad=10, fontweight="bold")
    axes[1].set_ylim(0, 2200)
    for b, c in zip(bars, cat_counts):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 35, f"{c}\n({c/len(anns)*100:.1f}%)", ha="center", fontsize=8.5)

    # Panel C: Bounding box aspect ratios
    bboxes = np.array(anns["bbox"].tolist())
    aspects = bboxes[:, 2] / np.maximum(bboxes[:, 3], 1.0)
    # Clip extreme outliers for clean histogram
    aspects_clipped = np.clip(aspects, 0.1, 5.0)

    axes[2].hist(aspects_clipped, bins=30, color="#756bb1", alpha=0.85, edgecolor="black", lw=0.6)
    axes[2].axvline(1.0, color="red", linestyle="--", lw=1.5, label="Square (1:1)")
    axes[2].axvline(np.median(aspects), color="black", linestyle=":", lw=1.5, label=f"Median ({np.median(aspects):.2f})")
    axes[2].set_xlabel("Bounding Box Aspect Ratio (Width / Height)", fontweight="bold")
    axes[2].set_ylabel("Instance Count", fontweight="bold")
    axes[2].set_title("(c) Bounding Box Aspect Ratio Distribution", pad=10, fontweight="bold")
    axes[2].legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_geometry.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_geometry")

# =========================================================================
# FIGURE 4: Large-format 4-Row Grad-CAM Composite
# =========================================================================
def make_fig_gradcam_large():
    def get_crops(prefix, iid):
        im = Image.open(OUT / f"{prefix}_gradcam_{iid}.png")
        arr = np.array(im)
        sub = arr[200:, :, :3]
        mask = ~((sub[:, :, 0] > 250) & (sub[:, :, 1] > 250) & (sub[:, :, 2] > 250))
        cols = np.where(mask.any(axis=0))[0]
        gaps = np.where(np.diff(cols) > 50)[0]
        rows = np.where(mask.any(axis=1))[0] + 200
        r0, r1 = rows[0], rows[-1]
        c_left0, c_left1 = cols[0], cols[gaps[0]]
        c_right0, c_right1 = cols[gaps[0]+1], cols[-1]
        inp = arr[r0:r1+1, c_left0:c_left1+1]
        cam = arr[r0:r1+1, c_right0:c_right1+1]
        return inp, cam

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 12.5), dpi=300)

    # Row 1: Image 1203 (bony fish, brittle star)
    inp_1203, cam_frcnn_1203 = get_crops("frcnn", 1203)
    _, cam_yolo_1203 = get_crops("yolov8s", 1203)
    _, cam_rtd_1203 = get_crops("rtdetr", 1203)

    # Row 2: Image 1505 (sea fan, stony coral)
    inp_1505, cam_frcnn_1505 = get_crops("frcnn", 1505)
    _, cam_yolo_1505 = get_crops("yolov8s", 1505)
    _, cam_rtd_1505 = get_crops("rtdetr", 1505)

    panel_data = [
        (inp_1203, "(a) Input Scene: Image 1203\n(bony fish, brittle star)"),
        (cam_frcnn_1203, "(b) Faster R-CNN R50-FPN\nResNet-50 Layer 4 Attribution"),
        (cam_yolo_1203, "(c) YOLOv8s\nCSPDarknet C2f Attribution"),
        (cam_rtd_1203, "(d) RT-DETR-L\nHybrid Encoder Attribution"),
        (inp_1505, "(e) Input Scene: Image 1505\n(sea fan, stony coral)"),
        (cam_frcnn_1505, "(f) Faster R-CNN R50-FPN\nResNet-50 Layer 4 Attribution"),
        (cam_yolo_1505, "(g) YOLOv8s\nCSPDarknet C2f Attribution"),
        (cam_rtd_1505, "(h) RT-DETR-L\nHybrid Encoder Attribution"),
    ]

    for idx, (img_arr, title) in enumerate(panel_data):
        ax = axes[idx // 2, idx % 2]
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        ax.axis("off")

    fig.tight_layout(h_pad=1.0, w_pad=0.35)
    fig.savefig(OUT / "fig_gradcam_large.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fig_gradcam_large.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_gradcam_large")

if __name__ == "__main__":
    make_fig_class_imbalance()
    make_fig_performance_tradeoff()
    make_fig_geometry()
    make_fig_gradcam_large()
