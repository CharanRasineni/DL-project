import json, glob, re
from pathlib import Path

kaggle = Path('notebooks/kaggle')
for d in ['02_split', '03_fasterrcnn', '04_yolov8s', '05_rtdetr']:
    nb = json.load(open(kaggle / d / f'{d}.ipynb'))
    print('=' * 70)
    print(f'### {d}  ({len(nb["cells"])} cells) ###')
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c['source'])
        print(f'\n----- CELL {i} (code, {len(src)} chars) -----')
        print(src[:3000])

for d in ['04_yolov8s', '05_rtdetr']:
    meta = json.load(open(kaggle / d / 'kernel-metadata.json'))
    print('\n' + '=' * 60)
    print(f'### {d}/kernel-metadata.json ###')
    print(json.dumps({kk: vv for kk, vv in meta.items() if kk in ('kernelId', 'id', 'kernel_sources', 'container_image', 'language', 'env')}, indent=2, sort_keys=True))
