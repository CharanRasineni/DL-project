"""Extract the final results-JSON printed by research_runner from Kaggle kernel logs.

Kernel logs are JSON-lines: {"stream_name": "stdout", "data": "..."} chunks where
each data chunk is one stdout line of a pretty-printed JSON block. The results
block starts with a bare '{' chunk and contains 'latency'. Reassemble stdout,
json-decode each block that starts on such a line, keep the last valid one.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "research_results"
JOBS = {
    "yolov8s": R / "yolov8s_full" / "fathomnet-research-yolov8s.log",
    "rtdetr": R / "rtdetr_full2" / "fathomnet-research-rtdetr.log",
}

for fam, p in JOBS.items():
    out = []
    n_parsed = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("[{"):
            line = line[1:]
        if line.startswith(",{"):
            line = line[1:]
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            n_parsed += 1
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("stream_name") == "stdout":
            out.append(obj["data"])
    joined = "".join(out)
    (R / f"{fam}_stdout_dump.txt").write_text(joined, encoding="utf-8")

    decoder = json.JSONDecoder()
    best = None
    # find block starts: a chunk that is exactly '{'
    idxs = [i for i, c in enumerate(out) if c.strip() == "{"]
    for i in idxs:
        pos = sum(len(c) for c in out[:i])
        try:
            obj, _ = decoder.raw_decode(joined, pos)
        except Exception:
            continue
        if isinstance(obj, dict) and "latency" in obj and "per_class_AP50-95" in obj:
            best = obj
    dest = R / f"{fam}_inline_results.json"
    if best is None:
        dest.write_text("NOT FOUND", encoding="utf-8")
        print(fam, f"NOT FOUND (n_parsed={n_parsed}, block starts={len(idxs)})")
        continue
    dest.write_text(json.dumps(best, indent=1), encoding="utf-8")
    keep = {
        k: best.get(k)
        for k in ("mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium",
                  "mAP_large", "head_mid_tail_AP50-95", "COCO_maxDets",
                  "n_dets", "latency", "protocol")
    }
    print(f"=== {fam} ===")
    print(json.dumps(keep, indent=1))
    zero = [k for k, v in best.get("per_class_AP50-95", {}).items() if v == 0.0]
    print("zero-AP classes:", zero)
    print("saved ->", dest.name)




