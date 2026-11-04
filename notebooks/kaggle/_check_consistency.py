"""Audit: are the appendix per-class APs consistent with the reported strata?"""
import json
import re
from pathlib import Path

B = Path(__file__).parent
LOGS = {
    "yolov8s": B / "04_yolov8s" / "output" / "fathomnet-04-yolov8s.log",
    "rtdetr": B / "05_rtdetr" / "output" / "fathomnet-05-rtdetr.log",
}

for name, p in LOGS.items():
    txt = p.read_text(encoding="utf-8", errors="replace")
    chunks = [m.group(1).encode().decode("unicode_escape", errors="replace")
              for m in re.finditer(r'\{"stream_name":"(?:stdout|stderr)","time":[\d.]+,"data":"(.*?)"\}', txt)]
    blob = "\n".join(chunks)
    i = blob.find("per_class_AP50-95")
    print(f"===== {name}: 'per_class_AP50-95' at {i} =====")
    if i > 0:
        print(blob[i - 100:i + 1600])
    # look for a full results json dump
    j = blob.rfind('"head_mid_tail_AP50-95"')
    print(f"--- head_mid_tail context ---")
    print(blob[j - 200:j + 400] if j > 0 else "not found")

# compare appendix per-class values (hardcoded from v1.tex) to reported strata
H = {"urchin": 0.6520, "bony fish": 0.6420, "brittle star": 0.4160}      # rtdetr appendix
M = {"stony coral": 0.6440, "sea fan": 0.5120, "sponge": 0.4870, "anemone": 0.3870}
print("\nRT-DETR appendix head mean:", round(sum(H.values()) / 3, 4), "reported 0.6025")
print("RT-DETR appendix mid  mean:", round(sum(M.values()) / 4, 4), "reported 0.4855")
