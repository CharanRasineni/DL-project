"""Inspect how kaggle 1.7.4.5 builds its HTTP client (auth + base URL)."""
import inspect
from pathlib import Path

import kaggle

base = Path(kaggle.__file__).parent
src = (base / "api" / "kaggle_api.py").read_text(encoding="utf-8", errors="replace")
print("=== imports mentioning client ===")
for line in src.splitlines():
    if "import" in line.lower() and "client" in line.lower():
        print("   ", line.strip())
print("=== KaggleClient class location ===")
try:
    from kaggle.api.kaggle_api import KaggleClient  # noqa
    print("importable from kaggle.api.kaggle_api; file:", inspect.getfile(KaggleClient))
except Exception as e:  # noqa: BLE001
    print("not in kaggle_api:", e)
import pkgutil
import kaggle.api as api_pkg

print("submodules:", [m.name for m in pkgutil.iter_modules(api_pkg.__path__)])
