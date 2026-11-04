"""Find the kernel-session-output client + request model to enable pagination."""
import inspect

import kaggle.api.kaggle_api_extended as kx

src = inspect.getsource(kx)
print("=== module imports ===")
for line in src.splitlines()[:40]:
    if line.startswith(("import", "from")):
        print(line)
