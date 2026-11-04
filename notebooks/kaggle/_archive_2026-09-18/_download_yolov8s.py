import json,os,sys
from pathlib import Path
b = Path(__file__).resolve().parent
os.environ['KAGGLE_API_TOKEN']=json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
ref='rscharan/fathomnet-research-yolov8s'
dest=b/'research_results'/'yolov8s_full'; dest.mkdir(parents=True,exist_ok=True)
api.kernels_output(ref,str(dest))
print('OK',dest)