"""Verify all three completed runs' artifacts and print a summary."""
import json
from pathlib import Path

b = Path(r'c:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle\research_results')
KEYS = ['mAP50-95', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large',
        'head_mid_tail_AP50-95', 'n_dets', 'latency']
for name in ('frcnn_resume_full', 'yolov8s_full_verified', 'rtdetr_full'):
    found = list((b / name).rglob('completed.json'))
    if not found:
        print(name, ': NO completed.json')
        continue
    root = found[0].parent
    print('==', name, '->', root.name)
    print('completed:', json.loads(found[0].read_text(encoding='utf-8')).get('status'))
    res = root / 'results.json'
    if res.is_file():
        j = json.loads(res.read_text())
        for k in KEYS:
            v = j.get(k)
            if k == 'latency' and isinstance(v, dict):
                v = v.get('median_ms')
            print(' ', k, '=', v)
        pcs = j.get('per_class_AP50-95', {})
        print('  classes:', len(pcs), '| zero-AP:', sorted(k for k, v in pcs.items() if v == 0))
    else:
        print('  results.json MISSING')
    sr = root / 'stage_report.json'
    if sr.is_file():
        print('  stages:', json.loads(sr.read_text()))
    fig = root / 'figure_manifest.json'
    if fig.is_file():
        m = json.loads(fig.read_text())
        print('  figures:', len(m.get('images', m if isinstance(m, list) else [])))
