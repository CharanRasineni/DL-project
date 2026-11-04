"""Dump full cell sources of the training notebooks for exact editing."""
import json
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = {
    "03": HERE / "03_fasterrcnn" / "03_fasterrcnn.ipynb",
    "04": HERE / "04_yolov8s" / "04_yolov8s.ipynb",
    "05": HERE / "05_rtdetr" / "05_rtdetr.ipynb",
    "05b": HERE / "05_rtdetr" / "05b" / "05b_eval.ipynb",
}
for tag, p in TARGETS.items():
    nb = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            out.append(f"########## MD CELL {i:02d} ##########")
            out.append("".join(c["source"]))
        else:
            out.append(f"########## CODE CELL {i:02d} ##########")
            out.append("".join(c["source"]))
    (HERE / f"_full_{tag}.txt").write_text("\n".join(out), encoding="utf-8")
    print(tag, p.name, "cells:", len(nb["cells"]))
