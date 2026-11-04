"""Extract the embedded result-JSON blocks from the saved 04/05 kernel logs."""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
LOGS = {
    "yolov8s": BASE / "04_yolov8s" / "output" / "fathomnet-04-yolov8s.log",
    "rtdetr": BASE / "05_rtdetr" / "output" / "fathomnet-05-rtdetr.log",
}

for name, p in LOGS.items():
    txt = p.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for m in re.finditer(r'\{"stream_name":"(?:stdout|stderr)","time":([\d.]+),"data":"(.*?)"\}', txt):
        chunks.append(m.group(2).encode().decode("unicode_escape", errors="replace"))
    blob = "\n".join(chunks)
    # find the printed JSON block starting at '{' after 'Loading and preparing results'
    i = blob.rfind('"mAP50-95"')
    if i < 0:
        print(f"{name}: no JSON found")
        continue
    start = blob.rfind("{", 0, i)
    # JSON was printed line by line; reconstruct by taking from start to the end brace
    end = blob.find("}", blob.find("}", blob.find("}", i)))  # crude
    # safer: find the line-by-line dump; each chunk is one line; JSON closes with standalone '}'
    lines = blob[start:].splitlines()
    buf = []
    depth = 0
    for ln in lines:
        s = ln.strip()
        if s.endswith("{") or s.startswith("{"):
            pass
        buf.append(s)
        depth += s.count("{") - s.count("}")
        if depth == 0 and s == "}":
            break
    try:
        obj = json.loads("\n".join(buf))
    except Exception as e:
        print(f"{name}: parse failed: {e}")
        obj = None
    if obj:
        out = BASE / f"results_{name}.json"
        out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"{name}: saved {out.name}")
        print(json.dumps({k: v for k, v in obj.items() if k != "per_class_AP50-95"}, indent=2)[:1200])
        print("per-class classes:", len(obj.get("per_class_AP50-95", {})))
