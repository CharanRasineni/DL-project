"""Pull the final 'all' validation row (P, R, mAP50, mAP50-95) from the 04/05 logs."""
import re
from pathlib import Path

B = Path(__file__).parent
for name, p in {"yolov8s": B / "04_yolov8s/output/fathomnet-04-yolov8s.log",
                "rtdetr": B / "05_rtdetr/output/fathomnet-05-rtdetr.log"}.items():
    txt = p.read_text(encoding="utf-8", errors="replace")
    chunks = [m.group(1).encode().decode("unicode_escape", errors="replace")
              for m in re.finditer(r'"data":"(.*?)"\}', txt)]
    lines = "\n".join(chunks).splitlines()
    rows = [(i, l.strip()) for i, l in enumerate(lines)
            if re.match(r"\s*all\s+\d+\s+\d+", l)]
    print(f"===== {name}: {len(rows)} 'all' rows =====")
    for i, l in rows[-3:]:
        print(" ", l)
