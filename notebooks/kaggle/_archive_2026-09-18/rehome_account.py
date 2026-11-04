"""One-time re-homing: eternalsorcerer/* -> rscharan/* (new Kaggle account).

Rewrites the `id` in all four kernel-metadata.json files and swaps the
`fathomnet-02-split` kernel_source to the new account. Idempotent.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
OLD_USER, NEW_USER = "eternalsorcerer", "rscharan"
SLUGS = ["fathomnet-02-split", "fathomnet-03-fasterrcnn",
         "fathomnet-04-yolov8s", "fathomnet-05-rtdetr"]

for slug in SLUGS:
    d = [x for x in BASE.iterdir()
         if x.is_dir() and (x / "kernel-metadata.json").exists()
         and slug in json.loads((x / "kernel-metadata.json").read_text()).get("id", "")]
    if not d:
        print(f"SKIP {slug}: no matching dir")
        continue
    p = d[0] / "kernel-metadata.json"
    m = json.loads(p.read_text())
    m["id"] = f"{NEW_USER}/{slug}"
    m["title"] = slug
    m["kernel_sources"] = [
        s.replace(f"{OLD_USER}/", f"{NEW_USER}/") for s in m.get("kernel_sources", [])
    ]
    p.write_text(json.dumps(m, indent=2) + "\n")
    print(f"{d[0].name}: id={m['id']} sources={m['kernel_sources']}")
