"""Check which head/mid/tail grouping each run ACTUALLY used, and whether it
matches the current frozen split. Also recover the per-run train counts if printed.
"""
import glob
import re

pat = re.compile(r"groups: head=\[([^\]]*)\]")
for p in sorted(glob.glob("0*/output/*.log")):
    txt = open(p, encoding="utf-8", errors="replace").read()
    hits = pat.findall(txt)
    print(f"{p}\n   head lines: {len(hits)}")
    seen = []
    for h in hits:
        h = h.strip()
        if h not in seen:
            seen.append(h)
    for h in seen[:6]:
        print(f"   -> {h}")
