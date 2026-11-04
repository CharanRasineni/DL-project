"""Dump current project state: dirs, files, key JSON contents."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

print("=== notebooks/kaggle top level ===")
for f in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, f)
    kind = "dir " if os.path.isdir(p) else "file"
    sz = "" if os.path.isdir(p) else f" ({os.path.getsize(p)} B)"
    print(f"  {kind} {f}{sz}")

print()
print("=== kernel dirs ===")
for d in ["03_fasterrcnn", "04_yolov8s", "05_rtdetr"]:
    dd = os.path.join(BASE, d)
    if not os.path.isdir(dd):
        print(f"  {d}: MISSING")
        continue
    print(f"  {d}/")
    for root, dirs, files in os.walk(dd):
        rel = os.path.relpath(root, dd)
        if "__pycache__" in rel:
            continue
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"    {os.path.join(rel, f) if rel != '.' else f} ({os.path.getsize(fp)} B)")

print()
print("=== _kaggle_out ===")
ko = os.path.join(BASE, "_kaggle_out")
if os.path.isdir(ko):
    for d in sorted(os.listdir(ko)):
        dd = os.path.join(ko, d)
        print(f"  {d}/")
        for f in sorted(os.listdir(dd)):
            fp = os.path.join(dd, f)
            print(f"    {f} ({os.path.getsize(fp)} B)")
else:
    print("  (missing)")