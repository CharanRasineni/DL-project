import json, os, subprocess, sys
from pathlib import Path
b = Path(r'c:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle')
script = '''import json, os
from pathlib import Path
b = Path(%r)
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
dest = b / 'research_results' / 'fasterrcnn_failed'; dest.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_output('rscharan/fathomnet-research-fasterrcnn', str(dest)); print('output downloaded', flush=True)
except Exception as e:
    print('output dl error:', e, flush=True)
req = ApiListKernelSessionOutputRequest(); req.user_name = 'rscharan'; req.kernel_slug = 'fathomnet-research-fasterrcnn'
with api.build_kaggle_client() as c:
    r = c.kernels.kernels_api_client.list_kernel_session_output(req)
(dest / 'inline.log').write_text(r.log or '', encoding='utf-8', errors='replace')
print('log chars:', len(r.log or ''), flush=True)
''' % str(b)
(b / '_dl_frcnn.py').write_text(script, encoding='utf-8')
subprocess.Popen([sys.executable, '-u', str(b / '_dl_frcnn.py')], cwd=str(b),
                 stdout=open(b / 'research_checks' / 'dl_frcnn.log', 'wb'), stderr=subprocess.STDOUT,
                 creationflags=subprocess.DETACHED_PROCESS)
print('download started')
