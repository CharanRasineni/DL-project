
"""List kernel output files; fetch only small results/log files."""
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
SLUGS = {
    "rscharan/fathomnet-04-yolov8s": BASE / "04_yolov8s" / "output",
    "rscharan/fathomnet-05-rtdetr": BASE / "05_rtdetr" / "output",
}

KEEP_EXT = (".log", ".json", ".csv", ".txt", ".yaml", ".yml")

for slug, outdir in SLUGS.items():
    print(f"\n=== {slug} ===", flush=True)
    owner, kern = slug.split("/")
    try:
        status = api.kernels_status(slug)
        print(f"status: {status.status}  msg={status.failure_message!r}", flush=True)
        with api.build_kaggle_client() as kaggle:
            from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E501
                ApiListKernelSessionOutputRequest,
            )
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = kern
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
        outdir.mkdir(parents=True, exist_ok=True)
        for item in resp.files:
            name = item.file_name
            print(f"  {name:60s}", flush=True)
            if not name.lower().endswith(KEEP_EXT):
                continue
            r = requests.get(item.url, timeout=180)
            if r.status_code == 200:
                dst = outdir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(r.content)
                print(f"    -> saved {len(r.content)/1024:.1f} KB", flush=True)
            else:
                print(f"    -> download failed {r.status_code}", flush=True)
        if resp.log:
            (outdir / f"{kern}.log").write_text(resp.log, encoding="utf-8")
            print(f"    -> saved log ({len(resp.log)/1024:.1f} KB)", flush=True)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", flush=True)

print("DONE", flush=True)




