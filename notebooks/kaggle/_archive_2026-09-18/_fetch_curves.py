"""Fetch remaining kernel output artifacts (04/05 pngs) via direct URLs."""
import json
import os
from pathlib import Path

import requests

CRED = json.loads(Path(r"C:\Users\rshit\.kaggle\kaggle.json").read_text())
os.environ["KAGGLE_API_TOKEN"] = CRED["key"]

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()

BASE = Path(__file__).parent
PAPER_IMAGES = BASE.parent.parent / "paper" / "images"
JOBS = {
    "rscharan/fathomnet-05-rtdetr": (BASE / "05_rtdetr" / "output", {"curve_rtdetr.png", "curve_rtdetr.pdf"}),
    "rscharan/fathomnet-04-yolov8s": (BASE / "04_yolov8s" / "output", {"curve_yolov8s.pdf"}),
}

for slug, (outdir, wanted) in JOBS.items():
    owner, kern = slug.split("/")
    with api.build_kaggle_client() as kaggle:
        from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E501
            ApiListKernelSessionOutputRequest,
        )
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = kern
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    for item in resp.files:
        if item.file_name in wanted:
            r = requests.get(item.url, timeout=180)
            if r.status_code == 200:
                (outdir / item.file_name).write_bytes(r.content)
                print(f"saved {item.file_name} ({len(r.content)/1024:.1f} KB)")
            else:
                print(f"FAILED {item.file_name}: {r.status_code}")
