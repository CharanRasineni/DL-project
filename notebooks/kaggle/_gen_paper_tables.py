"""Generate LaTeX per-class longtables for YOLOv8s and RT-DETR-L from logs.

Also emits the class-count table rows (head/mid/tail) from 02_split CSV
and the main results table rows.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
PAPER = BASE.parent.parent / "paper"

STRATA = {}
for cls, counts in {
    # head
    "urchin": "head", "bony fish": "head", "brittle star": "head",
    # mid
    "sea fan": "mid", "sponge": "mid", "anemone": "mid", "stony coral": "mid",
}.items():
    STRATA[cls] = counts

TRAIN_VAL = {}
for ln in (BASE / "02_split" / "output" / "table_split_class_counts.csv").read_text().splitlines()[1:]:
    if not ln.strip():
        continue
    cls, tr, va = ln.split(",")
    TRAIN_VAL[cls] = (int(tr), int(va))
    if cls not in STRATA:
        STRATA[cls] = "tail"


def perclass_from_log(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for m in re.finditer(r'\{"stream_name":"(?:stdout|stderr)","time":([\d.]+),"data":"(.*?)"\}', txt):
        chunks.append(m.group(2).encode().decode("unicode_escape", errors="replace"))
    blob = "\n".join(chunks)
    idx = [i for i, ln in enumerate(blob.splitlines()) if re.match(r"\s*all\s+646\s+2102\s", ln)]
    lines = blob.splitlines()
    block = lines[idx[-1] : idx[-1] + 90]
    per = {}
    for ln in block:
        m = re.match(r"\s*(\S.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)\s*$", ln)
        if m:
            cls = m.group(1).strip()
            if cls == "all":
                continue
            ap95 = float(m.group(7))
            ap50 = float(m.group(6))
            per[cls] = (ap95, ap50)
    return per


def longtable(model: str, per: dict, caption: str, source: str) -> str:
    rows = []
    for cls, (ap95, ap50) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        st = STRATA.get(cls, "tail")
        rows.append(f"  {cls} & {st} & {ap95:.4f} & {ap50:.3f} \\\\")
    body = "\n".join(rows)
    return f"""% === Per-class AP for {model} ===
% Source: {source}
\\begin{{longtable}}{{@{{}}l r r r@{{}}}}
  \\caption{{{caption}}}\\\\
  \\toprule
  Class & Stratum & AP50-95 & AP50 \\\\
  \\midrule
  \\endhead
{body}
  \\bottomrule
  \\multicolumn{{4}}{{l}}{{\\small\\textit{{Source:}} {source}.}}
\\end{{longtable}}
"""


per04 = perclass_from_log(BASE / "04_yolov8s" / "output" / "fathomnet-04-yolov8s.log")
per05 = perclass_from_log(BASE / "05_rtdetr" / "output" / "fathomnet-05-rtdetr.log")
print("yolov8s classes:", len(per04), " rtdetr classes:", len(per05))

t04 = longtable(
    "YOLOv8s (30 epochs, imgsz 960)", per04,
    "YOLOv8s per-class AP at IoU[0.50:0.95] (COCO re-evaluation of the best"
    " checkpoint; Ultralytics per-class validation values) and the"
    " head/mid/tail stratum of each class.",
    "notebooks/kaggle/04\\_yolov8s/output/fathomnet-04-yolov8s.log",
)
t05 = longtable(
    "RT-DETR-L (30 epochs, imgsz 640)", per05,
    "RT-DETR-L per-class AP at IoU[0.50:0.95] (COCO re-evaluation of the best"
    " checkpoint; Ultralytics per-class validation values) and the"
    " head/mid/tail stratum of each class.",
    "notebooks/kaggle/05\\_rtdetr/output/fathomnet-05-rtdetr.log",
)

(PAPER / "_perclass_yolov8s.tex").write_text(t04, encoding="utf-8")
(PAPER / "_perclass_rtdetr.tex").write_text(t05, encoding="utf-8")

# class-count table rows
rows = []
for cls, (tr, va) in sorted(TRAIN_VAL.items(), key=lambda kv: -kv[1][0]):
    rows.append(f"    {STRATA[cls]} & {cls} & {tr:,} & {va} \\\\")
(PAPER / "_classcount_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("wrote _perclass_yolov8s.tex, _perclass_rtdetr.tex, _classcount_rows.tex")
