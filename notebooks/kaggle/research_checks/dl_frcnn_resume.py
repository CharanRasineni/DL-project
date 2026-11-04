import json,os
from pathlib import Path
b=Path(r'c:\\Users\\rshit\\OneDrive\\Documents\\Charan DL project\\notebooks\\kaggle')
os.environ['KAGGLE_API_TOKEN']=json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
dest=b/'research_results'/'frcnn_resume_full'
dest.mkdir(parents=True,exist_ok=True)
try:
    api.kernels_output('rscharan/fathomnet-research-frcnn-resume',str(dest)); print('downloaded',flush=True)
except Exception as e: print('download error',type(e).__name__,str(e)[:300],flush=True)
for p in sorted(dest.rglob('*')):
    if p.is_file(): print(p.relative_to(dest), p.stat().st_size,flush=True)
print('DONE',flush=True)
