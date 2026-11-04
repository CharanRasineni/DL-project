"""Extract run facts from the saved 04/05 kernel logs."""
import re
from pathlib import Path

BASE = Path(__file__).parent
LOGS = {
    "yolov8s": BASE / "04_yolov8s" / "output" / "fathomnet-04-yolov8s.log",
    "rtdetr": BASE / "05_rtdetr" / "output" / "fathomnet-05-rtdetr.log",
}

out = []
for name, p in LOGS.items():
    txt = p.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for m in re.finditer(r'\{"stream_name":"(?:stdout|stderr)","time":([\d.]+),"data":"(.*?)"\}', txt):
        data = m.group(2).encode().decode("unicode_escape", errors="replace")
        chunks.append(data)
    blob = "\n".join(chunks)
    out.append(f"===== {name} =====\n")
    # per-class lines: class name followed by 5-ish numeric columns
    cls_lines = [ln for ln in blob.splitlines() if re.match(r"\s*\S.*\s+\d+\s+\d+\s", ln) and "Images" not in ln and "Instances" not in ln]
    # the per-class table appears right after the final 'all' line
    idx = [i for i, ln in enumerate(blob.splitlines()) if re.match(r"\s*all\s+646\s+", ln)]
    if idx:
        last = idx[-1]
        lines = blob.splitlines()
        out.append("-- per-class block after final 'all' --\n")
        out.extend(lines[last : last + 100])
        out.append("\n")
    # model summary
    for ln in blob.splitlines():
        if "Model summary" in ln or "Speed:" in ln:
            out.append(ln.strip())
    # RT-DETR params from model build table
    m = re.search(r"Model summary.*", blob)
    if name == "rtdetr":
        for ln in blob.splitlines():
            if "total," in ln or "Model summary" in ln:
                out.append(ln.strip())

Path(BASE / "_facts.txt").write_text("\n".join(out), encoding="utf-8")
print("written _facts.txt")
