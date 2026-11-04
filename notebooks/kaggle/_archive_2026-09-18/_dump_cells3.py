"""Dump a compact cell map for the three training notebooks + 05b eval."""
import json
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = [
    HERE / "03_fasterrcnn" / "03_fasterrcnn.ipynb",
    HERE / "04_yolov8s" / "04_yolov8s.ipynb",
    HERE / "05_rtdetr" / "05_rtdetr.ipynb",
    HERE / "05_rtdetr" / "05b" / "05b_eval.ipynb",
]
for nb_path in TARGETS:
    print(f"===== {nb_path.name}  ({nb_path.parent.name}) =====")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]).replace("\n", " ")
        print(f"[{i:02d}][{c['cell_type'][:4]}] {src[:130]}")
    print()
