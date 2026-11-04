"""List kernel session output FILE NAMES + sizes only (no downloads)."""
import json
import os
from pathlib import Path

CRED = json.loads(Path.home().joinpath(".kaggle", "kaggle.json").read_text())
os.environ["KAGGLE_API_TOKEN"] = CRED["key"]

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()

SLUGS = [
    "rscharan/fathomnet-02-split",
    "rscharan/fathomnet-03-fasterrcnn",
    "rscharan/fathomnet-04-yolov8s",
    "rscharan/fathomnet-05-rtdetr",
]

from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiListKernelSessionOutputRequest,
)

for slug in SLUGS:
    owner, kern = slug.split("/")
    print(f"\n=== {slug} ===", flush=True)
    try:
        st = api.kernels_status(slug)
        print(f"status: {st.status}  msg={st.failure_message!r}", flush=True)
    except Exception as e:
        print(f"status error: {e}", flush=True)
    try:
        with api.build_kaggle_client() as kaggle:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = kern
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        for item in resp.files:
            print(f"  {item.file_name}", flush=True)
        if resp.log:
            print(f"  [log inline, {len(resp.log)} chars]", flush=True)
    except Exception as e:
        print(f"list error: {type(e).__name__}: {e}", flush=True)
print("\nDONE", flush=True)
