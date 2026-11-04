import json, os
from pathlib import Path
b = Path(r'c:\\Users\\rshit\\OneDrive\\Documents\\Charan DL project\\notebooks\\kaggle')
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
req = ApiListKernelSessionOutputRequest()
req.user_name = 'rscharan'
req.kernel_slug = 'fathomnet-research-rtdetr'
with api.build_kaggle_client() as client:
    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
log = resp.log or ''
(b / 'research_results' / 'rtdetr_inline.log').write_text(log, encoding='utf-8', errors='replace')
print('log chars:', len(log))
