"""Retry full output download of the budget-stopped FRCNN kernel, UTF-8 safe."""
import json
import os
import subprocess
import sys
from pathlib import Path
b = Path(r'c:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle')
script = '''import json, os
from pathlib import Path
b = Path(%r)
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
dest = b / 'research_results' / 'fasterrcnn_failed'
try:
    api.kernels_output('rscharan/fathomnet-research-fasterrcnn', str(dest))
    print('output downloaded', flush=True)
except Exception as e:
    print('output dl error:', type(e).__name__, e, flush=True)
for name in ('last.pt', 'best.pt', 'budget_stop.json', 'training.csv', 'provenance.json', 'config.json'):
    p = dest / name
    print(name, p.stat().st_size if p.exists() else 'MISSING', flush=True)
''' % str(b)
(b / '_dl_frcnn2.py').write_text(script, encoding='utf-8')
env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(b / '_dl_frcnn2.py')], cwd=str(b),
                 stdout=open(b / 'research_checks' / 'dl_frcnn2.log', 'wb'), stderr=subprocess.STDOUT,
                 env=env, creationflags=subprocess.DETACHED_PROCESS)
print('download started')
