import json, os
from pathlib import Path
b = Path(r'c:\\Users\\rshit\\OneDrive\\Documents\\Charan DL project\\notebooks\\kaggle')
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
dest = b / 'research_results' / 'rtdetr_full'
dest.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_output('rscharan/fathomnet-research-rtdetr', str(dest))
    print('download ok')
except Exception as e:
    print('download error', type(e).__name__, str(e))
files = [p for p in dest.rglob('*') if p.is_file()]
print('files:', len(files))
for p in sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:20]:
    print(round(p.stat().st_size / 1e6, 1), 'MB', p.relative_to(dest))
