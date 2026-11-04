"""Parse saved kernel logs (JSON-lines) and extract final COCO eval numbers."""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
LOGS = {
    "yolov8s": BASE / "04_yolov8s" / "output" / "fathomnet-04-yolov8s.log",
    "rtdetr": BASE / "05_rtdetr" / "output" / "fathomnet-05-rtdetr.log",
}

PAT = re.compile(
    r'"(?:mAP50-95|mAP50|AP50-95|AP50|Average Precision|Average Recall|"all"|\ball\b)'
)

for name, p in LOGS.items():
    print(f"\n===== {name}: {p.name} =====")
    if not p.exists():
        print("  missing!")
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    # log lines are JSON objects with "data" fields
    chunks = []
    for m in re.finditer(r'\{"stream_name":"(?:stdout|stderr)","time":([\d.]+),"data":"(.*?)"\}', txt):
        data = m.group(2).encode().decode("unicode_escape", errors="replace")
        chunks.append((float(m.group(1)), data))
    print(f"  parsed {len(chunks)} stream chunks")
    # final summary blocks: look for the last occurrences of COCO summary keywords
    KEY = ("Average Precision", "Average Recall", '"mAP', "Results saved", " all  ")
    hits = [(t, d) for t, d in chunks if any(k in d for k in KEY) and "Instances" not in d]
    for t, d in hits:
        sys.stdout.buffer.write(d.encode("utf-8", errors="replace"))
