"""Measure per-epoch wall time + step counts from the archived 04/05 logs."""
import re
from pathlib import Path

B = Path(__file__).parent

def blob(p):
    txt = p.read_text(encoding="utf-8", errors="replace")
    ch = [m.group(1).encode().decode("unicode_escape", errors="replace")
          for m in re.finditer(r'"data":"(.*?)"\}', txt)]
    return "\n".join(ch)

for name in ("04_yolov8s/output/fathomnet-04-yolov8s.log",
             "05_rtdetr/output/fathomnet-05-rtdetr.log"):
    b = blob(B / name)
    # ultralytics epoch summary lines contain "30 epochs completed in"
    m = re.search(r"(\d+) epochs completed in ([\d.]+) hours", b)
    print(f"=== {name} ===")
    if m:
        ep, hrs = int(m.group(1)), float(m.group(2))
        print(f"  {ep} epochs in {hrs:.2f} h -> {hrs*60/ep:.1f} min/epoch")
    # summation line e.g. "1455/1455" style step totals
    steps = re.findall(r"0/1455|0/1455\b", b)
    print(f"  '1455' step markers: {len(steps)}")
    for pat in ("epochs completed in", "Optimizer stripped", "EarlyStopping",
                "Results saved to", "validating", "Speed:"):
        for mm in re.finditer(re.escape(pat) + r"[^\n]{0,110}", b):
            print(f"  [{pat}] {mm.group(0).strip()[:130]}")
            break