# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.kernels_output('rscharan/fathomnet-research-fasterrcnn', r'C:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle\research_results\frcnn_v2_api')
print('DONE', flush=True)
