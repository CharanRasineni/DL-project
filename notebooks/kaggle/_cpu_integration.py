"""Detached CPU end-to-end test of the full production runner (synthetic data).

In-memory overrides: CUDA guard, paths, 64px, random init. Production file unchanged.
Covers train->checkpoint->predict->evaluate->latency->curves->figures->completed.json.
"""
import contextlib
import hashlib
import json
import sys
import traceback
from pathlib import Path

B = Path(__file__).resolve().parent
import tempfile
OUT = Path(tempfile.mkdtemp(prefix='cpu_run_', dir=B / 'research_checks'))
(B / 'research_checks' / 'latest_cpu_run.txt').write_text(str(OUT))
with (OUT / 'execution.log').open('w', encoding='utf-8', buffering=1) as log, \
        contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    try:
        import numpy as np
        import torch
        from PIL import Image
        import data as D
        import research_runner as R
        torch.set_num_threads(2)
        images = OUT / 'images'
        images.mkdir(exist_ok=True)
        coco = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'synthetic'}]}
        for iid in range(4):
            Image.fromarray(np.random.default_rng(iid).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(images / f'{iid}.png')
            coco['images'].append(dict(id=iid, file_name=f'{iid}.png', width=64, height=64))
            coco['annotations'].append(dict(id=iid + 1, image_id=iid, category_id=0,
                bbox=[12, 12, 24, 24], area=576, iscrowd=0))
        ann = OUT / 'annotations.json'
        ann.write_text(json.dumps(coco))
        split = dict(train_ids=[0, 1], val_ids=[2, 3], seed=42)
        digest = hashlib.sha256(json.dumps({k: sorted(split[k]) for k in ('train_ids', 'val_ids')},
            sort_keys=True).encode()).hexdigest()
        D.DATASET_JSON, D.IMG_ROOT, D.WORK = ann, images, OUT
        D.resolve_dataset_paths = lambda: (ann, images)
        D.load_frozen_split = lambda: split
        source = Path(R.__file__).read_text(encoding='utf-8')
        guard = "    if device != 'cuda':\n        raise RuntimeError('GPU required for real-image run; no silent CPU training')"
        assert source.count(guard) == 1
        source = source.replace(guard, "    assert device == 'cpu', 'CPU diagnostic only'")
        source = source.replace("cfg['imgsz'] == 640", "cfg['imgsz'] == 64")
        source = source.replace("Path('/kaggle/working')", 'Path(' + repr(str(OUT)) + ')')
        source = source.replace('def make_frcnn(nclasses, size=640, pretrained=True):',
                                'def make_frcnn(nclasses, size=640, pretrained=False):')
        exec(compile(source, str(Path(R.__file__)) + '[CPU-only in-memory overrides]', 'exec'), R.__dict__)
        config = dict(mode='train', family='fasterrcnn', epochs=1, batch=1, imgsz=64, seed=42,
                      score_thr=0.001, bootstrap=2, max_train_hours=1, split_sha256=digest)
        print('OVERRIDES: CPU guard, synthetic paths, 64px, random init; production file unchanged.', flush=True)
        result_dir = R.run(config)
        state = R.load_checkpoint(result_dir / 'last.pt')
        assert state['epoch'] == 1
        for name in ('best.pt', 'last.pt', 'preds.json', 'val_gt.json', 'results.json', 'per_class.csv',
                     'completed.json', 'stage_report.json', 'training_history.csv', 'provenance.json',
                     'figure_manifest.json'):
            assert (result_dir / name).is_file(), name
        assert 'bootstrap' in json.loads((result_dir / 'results.json').read_text())
        completed = json.loads((result_dir / 'completed.json').read_text())
        assert completed['status'] in ('PASS', 'CORE_COMPLETE_ANALYSIS_INCOMPLETE'), completed
        R.atomic_json(OUT / 'test_result.json', dict(status='PASS', synthetic_only=True,
            scientific_result=False, production_gpu_validated=False, output=str(result_dir),
            completed_status=completed['status'],
            stages=json.loads((result_dir / 'stage_report.json').read_text()),
            overrides=['CPU guard', 'synthetic paths', '64px', 'random initialization']))
        print('END-TO-END CPU PASS', flush=True)
    except BaseException:
        traceback.print_exc()
        (OUT / 'test_result.json').write_text(json.dumps({'status': 'FAIL',
            'traceback': traceback.format_exc()}, indent=2))
print((OUT / 'execution.log').read_text(encoding='utf-8')[-12000:])
