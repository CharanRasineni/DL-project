"""One-shot diagnostic: directory state + actual error text from kernel logs."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("1. TOP-LEVEL FILES IN notebooks/kaggle")
print("=" * 70)
for p in sorted(ROOT.iterdir()):
    kind = "DIR " if p.is_dir() else "FILE"
    print(f"  {kind} {p.name}")

print()
print("=" * 70)
print("2. KERNEL DIRS: contents")
print("=" * 70)
for d in ["03_fasterrcnn", "04_yolov8s", "05_rtdetr"]:
    kd = ROOT / d
    if not kd.is_dir():
        print(f"  {d}: MISSING")
        continue
    print(f"  {d}/")
    for p in sorted(kd.iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        size = "" if p.is_dir() else f" ({p.stat().st_size}B)"
        print(f"    {kind} {p.name}{size}")

print()
print("=" * 70)
print("3. _kaggle_out CONTENTS")
print("=" * 70)
ko = ROOT / "_kaggle_out"
if ko.is_dir():
    for p in sorted(ko.rglob("*")):
        rel = p.relative_to(ko)
        kind = "DIR " if p.is_dir() else "FILE"
        size = "" if p.is_dir() else f" ({p.stat().st_size}B)"
        print(f"  {kind} {rel}{size}")
else:
    print("  (missing)")

print()
print("=" * 70)
print("4. ERROR TEXT EXTRACTED FROM LOGS")
print("=" * 70)
if ko.is_dir():
    for log in sorted(ko.rglob("*.log")):
        print()
        print("-" * 70)
        print(f"LOG: {log.relative_to(ko)}")
        print("-" * 70)
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  (unreadable: {e})")
            continue
        lines = text.splitlines()
        print(f"  total lines: {len(lines)}")
        # Find any line mentioning error/traceback/exception
        hits = []
        for i, line in enumerate(lines):
            low = line.lower()
            if any(k in low for k in ("traceback", "error", "exception",
                                      "no module", "not found", "failed",
                                      "cannot", "importerror", "modulenotfound")):
                hits.append((i + 1, line.rstrip()))
        if hits:
            print(f"  {len(hits)} matching lines; showing first 40:")
            for ln, txt in hits[:40]:
                print(f"    L{ln}: {txt[:200]}")
        else:
            print("  (no error-like lines found)")
        # Always show last 25 lines for context
        print("  --- last 25 lines ---")
        for line in lines[-25:]:
            print(f"    {line.rstrip()[:200]}")