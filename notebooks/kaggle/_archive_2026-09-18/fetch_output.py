"""Detached kernel-output downloader: python fetch_output.py <user/slug> <dest>
Uses the kernels/output JSON envelope and downloads each file via its signed URL."""
import json
import os
import sys
import time
from pathlib import Path

import requests

slug = sys.argv[1] if len(sys.argv) > 1 else ""
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
dest.mkdir(parents=True, exist_ok=True)
user, kslug = slug.split("/")
key = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["key"]
H = {"Authorization": f"Bearer {key}"}

r = requests.get("https://www.kaggle.com/api/v1/kernels/output",
                 params={"username": user, "kernelslug": kslug}, headers=H, timeout=120)
j = r.json()
print("status:", r.status_code, "| files:", len(j.get("files", [])), flush=True)

fs = j.get("files", [])
fs = sorted(fs, key=lambda f: 0 if "results" in (f.get("fileNameNullable") or "") else 1)
for f in fs:
    name, url = f["fileNameNullable"], f["urlNullable"]
    tgt = dest / name
    tgt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as d:
        d.raise_for_status()
        tmp = tgt.with_suffix(tgt.suffix + ".part")
        with open(tmp, "wb") as g:
            for chunk in d.iter_content(1 << 20):
                g.write(chunk)
    os.replace(str(tmp), str(tgt))
    print(f"{name}: {tgt.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
