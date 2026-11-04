import json

for nb, out in [("04_yolov8s/04_yolov8s.ipynb", "_04src.txt"),
                ("05_rtdetr/05_rtdetr.ipynb", "_05src.txt")]:
    d = json.load(open(nb, encoding="utf-8"))
    with open(out, "w", encoding="utf-8") as f:
        for i, c in enumerate(d["cells"]):
            f.write("\n" + "#" * 30 + f" CELL {i} ({c['cell_type']}) " + "#" * 30 + "\n")
            f.write("".join(c["source"]))
    print("wrote", out)


for nb in ["04_yolov8s/04_yolov8s.ipynb", "05_rtdetr/05_rtdetr.ipynb",
           "03_fasterrcnn.ipynb"]:
    d = json.load(open(nb, encoding="utf-8"))
    print("=" * 20, nb, len(d["cells"]), "cells")
    for i, c in enumerate(d["cells"]):
        src = "".join(c["source"])
        first = src.strip().splitlines()[0] if src.strip() else "(empty)"
        print("  [%d] %-8s %6d ch | %s" % (i, c["cell_type"], len(src), first[:90]))
