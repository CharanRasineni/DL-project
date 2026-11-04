"""Push a kernel dir using the official SDK path with bearer auth.

kaggle 1.7.4.5's KaggleHttpClient uses KAGGLE_API_TOKEN env var for
BearerAuth (new KGAT_ tokens). The CLI never sets it, so plain
`kaggle kernels push` 401s. Setting it + calling kernels_push() fixes that.

Usage: python push_kernel.py <kernel_dir>
"""
import json
import os
import sys
from pathlib import Path

CRED = json.loads(Path(r"C:\Users\rshit\.kaggle\kaggle.json").read_text())
os.environ["KAGGLE_API_TOKEN"] = CRED["key"]

# kaggle 1.7.4.5's api/kaggle_api.py is an obsolete shim; use the real class.
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()


def main() -> None:
    d = Path(sys.argv[1]).resolve()
    meta = json.loads((d / "kernel-metadata.json").read_text())
    res = api.kernels_push(str(d))
    print(f"push {meta['id']}: version={getattr(res, 'versionNumber', '?')} "
          f"ref={getattr(res, 'ref', '?')}")


if __name__ == "__main__":
    main()
