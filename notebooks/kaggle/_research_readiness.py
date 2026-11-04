"""Read-only remote readiness check. Never prints credentials."""
import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

cred = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
os.environ['KAGGLE_API_TOKEN'] = cred['key']
api = KaggleApi()
api.authenticate()
for slug in ['rscharan/00-setup-data-download-fathomnetclef2026',
             'rscharan/fathomnet-02-split', 'rscharan/fathomnet-03-fasterrcnn',
             'rscharan/fathomnet-04-yolov8s', 'rscharan/fathomnet-05-rtdetr']:
    try:
        response = api.kernels_status(slug)
        print(slug, str(response.status), getattr(response, 'failure_message', ''), flush=True)
    except Exception as exc:
        print(slug, type(exc).__name__, str(exc)[:200], flush=True)
