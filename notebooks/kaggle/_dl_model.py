"""Detached full-output download for one completed kernel; verifies key artifacts."""
import json
import os
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
KEY_FILES = ['results.json', 'per_class.csv', 'stage_report.json', 'figure_manifest.json',
             'provenance.json', 'training_curves.png']

def run(family):
    dest = B / 'research_results' / (family + '_full2')
    dest.mkdir(parents=True, exist_ok=True)
    code = f'''# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.kernels_output('rscharan/fathomnet-research-{family}', r'{dest}')
print('DONE', flush=True)
'''
    child = B / '_dl_child.py'
    child.write_text(code, encoding='utf-8')
    log = (B / 'research_checks' / ('dl_' + family + '.log')).open('ab', buffering=0)
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(child)], stdout=log,
                            stderr=subprocess.STDOUT, creationflags=subprocess.DETACHED_PROCESS)
    print(family, 'downloader PID', proc.pid, '->', dest)

if __name__ == '__main__':
    for family in sys.argv[1:] or ['yolov8s', 'rtdetr']:
        run(family)
