import json, os
from pathlib import Path
b = Path('c:\\Users\\rshit\\OneDrive\\Documents\\Charan DL project\\notebooks\\kaggle')
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
