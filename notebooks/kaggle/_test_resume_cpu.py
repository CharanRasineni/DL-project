"""CPU test of the new train_frcnn resume path. Synthetic data; production logic."""
import contextlib
import json
import sys
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'research_checks' / 'resume_cpu_test'
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'run.log').open('w', encoding='utf-8', buffering=1) as log, \
        contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    try:
        import numpy as np
        import torch
        from PIL import Image
        import data as D
        import research_runner as R
        torch.set_num_threads(2)
        torch.manual_seed(42)
        images = OUT / 'images'
        images.mkdir(exist_ok=True)
        coco = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'synthetic'}]}
        for iid in range(4):
            Image.fromarray(np.random.default_rng(iid).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(images / f'{iid}.png')
            coco['images'].append(dict(id=iid, file_name=f'{iid}.png', width=64, height=64))
            coco['annotations'].append(dict(id=iid + 1, image_id=iid, category_id=0,
                bbox=[12, 12, 24, 24], area=576, iscrowd=0))
        import hashlib
        split = dict(train_ids=[0, 1], val_ids=[2, 3], seed=42)
        digest = hashlib.sha256(json.dumps({k: sorted(split[k]) for k in ('train_ids', 'val_ids')},
            sort_keys=True).encode()).hexdigest()
        D.DATASET_JSON, D.IMG_ROOT, D.WORK = OUT / 'ann.json', images, OUT
        (OUT / 'ann.json').write_text(json.dumps(coco))
        cfg1 = dict(mode='train', family='fasterrcnn', epochs=1, batch=1, imgsz=64, seed=42,
                    score_thr=0.001, split_sha256=digest)
        first = R.train_frcnn(cfg1, coco, split, images, OUT, 'cpu')
        state1 = R.load_checkpoint(OUT / 'last.pt')
        # Known quirk: the checkpoint is saved before the current epoch's val row is
        # appended, so its history holds epochs 1..N-1 (0 rows after 1 epoch).
        assert state1['epoch'] == 1 and len(state1['history']) == 0, (state1['epoch'], len(state1['history']))
        assert (OUT / 'best.pt').is_file()
        # Resume for the 2nd epoch from the saved state (fresh optimizer objects).
        cfg2 = dict(cfg1, epochs=2, resume_checkpoint=str(OUT / 'last.pt'))
        best = R.train_frcnn(cfg2, coco, split, images, OUT, 'cpu')
        state2 = R.load_checkpoint(OUT / 'last.pt')
        assert state2['epoch'] == 2, state2['epoch']
        assert len(state2['history']) == 2, state2['history']
        assert str(best) == str(OUT / 'best.pt')
        # best.pt must be whichever epoch had the higher val AP; verify consistency.
        aps = [h['val_COCO_AP'] for h in state2['history']]
        best_state = R.load_checkpoint(OUT / 'best.pt')
        assert best_state['epoch'] - 1 == int(max(range(2), key=lambda i: aps[i])), (aps, best_state['epoch'])
        (OUT / 'result.json').write_text(json.dumps({'status': 'PASS', 'epochs': state2['epoch'],
            'history_rows': len(state2['history']), 'val_aps': aps}, indent=2))
        print('RESUME CPU PASS', flush=True)
    except BaseException:
        traceback.print_exc()
        (OUT / 'result.json').write_text(json.dumps({'status': 'FAIL',
            'traceback': traceback.format_exc()}, indent=2))
print((OUT / 'run.log').read_text(encoding='utf-8')[-8000:])
