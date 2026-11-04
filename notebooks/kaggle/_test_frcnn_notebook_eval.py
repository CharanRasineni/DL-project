"""Execute the notebook's actual evaluator block on archived predictions (CPU)."""
import csv
import json
import os
from pathlib import Path
import pandas as pd
import nbformat

HERE = Path(__file__).resolve().parent
archive = HERE / '03_fasterrcnn' / 'output'
os.environ['FATHOMNET_DATASET_JSON'] = str(archive / 'val_gt.json')
import data as D

nb = nbformat.read(HERE / '03_fasterrcnn' / '03_fasterrcnn.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, str(HERE / '03_fasterrcnn' / '03_fasterrcnn.ipynb') + ':' + str(i), 'exec')
S = D.load_split(HERE / '02_split' / 'output' / 'split.json')
cat_info = D.get_cat_info(D.load_coco_data())
with (HERE / '02_split' / 'output' / 'table_split_class_counts.csv').open() as f:
    cat_info['train_counts'] = {r['']: int(r['train']) for r in csv.DictReader(f)}
preds = json.loads((archive / 'preds_fasterrcnn.json').read_text())
OUT = HERE / 'research_checks' / 'frcnn_archived_eval'
OUT.mkdir(parents=True, exist_ok=True)
source = nb.cells[4].source
block = source.split('# BEGIN VERIFIED EVALUATION', 1)[1].split('# END VERIFIED EVALUATION', 1)[0]
exec(compile(block, 'notebook_actual_evaluation_block', 'exec'), globals())
arch = json.loads((archive / 'results_fasterrcnn.json').read_text())
saved = json.loads((OUT / 'results_fasterrcnn.json').read_text())
assert saved['mAP50-95'] == arch['mAP50-95'] == 0.308
assert saved['per_class_AP50-95'] == arch['per_class_AP50-95']
assert saved['head_mid_tail_AP50-95'] == {'head': 0.5093, 'mid': 0.4044, 'tail': 0.2684}
assert set(saved['groups']['head']) == {'urchin', 'bony fish', 'brittle star'}
assert len(pd.read_csv(OUT / 'eval_frcnn' / 'per_class.csv')) == 32
assert len(saved['per_class_size_AP50-95']) == 32
print('PASS: actual notebook evaluator, all 32 APs, corrected groups, JSON and CSV')
print('CPU re-evaluation only; no new training or inference:', OUT)
