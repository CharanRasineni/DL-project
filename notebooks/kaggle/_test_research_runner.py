"""CPU regression checks for the exact bundled research runner; no GPU spend."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ['FATHOMNET_DATASET_JSON'] = str(HERE / '03_fasterrcnn/output/val_gt.json')
os.environ['MPLBACKEND'] = 'Agg'
import data as D
import research_runner as R

torch.set_num_threads(2)

def main():
    checks = []
    D.selftest()
    gt = D.load_coco_data()
    split = json.loads((HERE / '02_split/output/split.json').read_text())
    cat = D.get_cat_info(gt, split)
    cat['train_counts'] = split['train_counts'] if 'train_counts' in split else {c['name']: 1 for c in gt['categories']}
    preds = json.loads((HERE / '03_fasterrcnn/output/preds_fasterrcnn.json').read_text())
    ev = R.ResearchEvaluator(split, cat, score_thr=0.001)
    result = ev.evaluate(preds)
    assert len(result['per_class_AP50-95']) == 32
    assert result['COCO_maxDets'] == [1, 10, 100]
    assert abs(np.mean(list(result['per_class_AP50-95'].values())) - result['mAP50-95']) < 1e-8
    assert ev.evaluate([])['mAP50-95'] == 0
    tiny_split = dict(split, val_ids=split['val_ids'][:2])
    tiny = R.ResearchEvaluator(tiny_split, cat, score_thr=0.001)
    ci = tiny.bootstrap_research([p for p in preds if p['image_id'] in tiny_split['val_ids']], n=2)
    assert ci['intervals']['mAP50-95']['n_valid'] == 2
    checks.append('COCO full/empty/per-class/size/bootstrap')
    from ultralytics import YOLO, RTDETR
    rgb = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for family, model in [('fasterrcnn', R.make_frcnn(32, 64, pretrained=False)),
                          ('yolov8s', YOLO('yolov8s.yaml')), ('rtdetr', RTDETR('rtdetr-l.yaml'))]:
        print('Testing CAM:', family, flush=True)
        class_index = 0
        if family == 'fasterrcnn':
            model.eval()
            with torch.no_grad():
                output = model([torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255])[0]
            assert len(output['labels'])
            class_index = int(output['labels'][0]) - 1
        heat, meta = R.detection_cam(model, family, rgb, 320 if family == 'rtdetr' else 64, class_index)
        assert heat.ndim == 2 and np.isfinite(heat).all() and heat.max() > heat.min()
        checks.append(family + ' CAM forward/backward')
        del model
    (HERE / 'research_checks').mkdir(exist_ok=True)
    R.atomic_json(HERE / 'research_checks/local_regression.json', {'status': 'PASS', 'checks': checks,
        'runner_sha256': R.sha256(R.__file__), 'limitation': 'CPU synthetic CAM, not full-dataset GPU validation'})
    print('PASS', checks, flush=True)

if __name__ == '__main__':
    main()
