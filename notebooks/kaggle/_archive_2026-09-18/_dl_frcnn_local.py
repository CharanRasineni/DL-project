"""Detached downloader: fetch the failed FRCNN kernel's output via API to disk."""
import json
import os
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
dest = B / 'research_results' / 'frcnn_v2_api'
dest.mkdir(parents=True, exist_ok=True)
code = f'''# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.kernels_output('rscharan/fathomnet-research-fasterrcnn', r'{dest}')
print('DONE', flush=True)
'''
(B / '_dl_frcnn_child.py').write_text(code, encoding='utf-8')
env = dict(os.environ, PYTHONIOENCODING='utf-8')
log = (B / 'research_checks' / 'dl_frcnn.log').open('ab', buffering=0)
proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(B / '_dl_frcnn_child.py')],
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.DETACHED_PROCESS)
print('downloader PID', proc.pid, '->', dest)
