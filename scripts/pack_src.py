"""Package the src/ code (and configs) into a zip for uploading to Kaggle
as a Dataset that all competition notebooks can import.

Usage:  python scripts/pack_src.py
Output: kaggle/fathomnet_pu_src.zip  -> upload at kaggle.com/datasets (new)
Then in each Kaggle notebook:
    import sys; sys.path.insert(0, '/kaggle/input/<your-dataset-slug>/src')
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "fathomnet_pu_src.zip"
INCLUDE = ["src", "configs"]

OUT.parent.mkdir(exist_ok=True)
if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in INCLUDE:
        base = ROOT / item
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                zf.write(p, p.relative_to(ROOT))

print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
