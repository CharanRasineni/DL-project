"""Isolated real Grad-CAM CPU diagnostic; no training, GPU, or timeout.

Imports only detection_cam from the project runner. Synthetic images and random
weights test mechanics, not scientific explanations or detection quality.
"""
import contextlib
import hashlib
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'research_checks' / 'isolated_cam'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'diagnostic.log').open('w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report = {'status': 'RUNNING', 'device': 'cpu', 'synthetic_only': True,
                      'training_validated': False, 'rtdetr': 'SKIPPED; not tested or stubbed',
                      'checks': []}
            try:
                import numpy as np
                import torch
                import ultralytics
                import torchvision
                from PIL import Image
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from ultralytics import YOLO
                from research_runner import detection_cam

                torch.set_num_threads(2)
                torch.manual_seed(42)
                rgb = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
                Image.fromarray(rgb).save(OUT / 'synthetic_input.png')
                report['versions'] = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                                      'ultralytics': ultralytics.__version__}
                for family in ('fasterrcnn', 'yolov8s'):
                    print('START', family, flush=True)
                    if family == 'fasterrcnn':
                        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                            min_size=64, max_size=64, box_score_thresh=0.001,
                            box_detections_per_img=300)
                        model.roi_heads.box_predictor = FastRCNNPredictor(
                            model.roi_heads.box_predictor.cls_score.in_features, 33)
                        model.eval()
                        with torch.no_grad():
                            pred = model([tensor])[0]
                        assert pred['labels'].numel(), 'No detection: cannot choose CAM class'
                        class_index = int(pred['labels'][0]) - 1
                    else:
                        model = YOLO('yolov8s.yaml')
                        class_index = 0
                    heat, metadata = detection_cam(model, family, rgb, 64, class_index)
                    assert heat.ndim == 2 and np.isfinite(heat).all(), 'Invalid heatmap'
                    assert float(heat.max()) > float(heat.min()), 'Constant heatmap'
                    np.save(OUT / (family + '_heat.npy'), heat)
                    Image.fromarray(np.rint(heat * 255).astype(np.uint8)).resize((256, 256)).save(
                        OUT / (family + '_heat.png'))
                    report['checks'].append({'family': family, 'status': 'PASS',
                        'shape': list(heat.shape), 'minimum': float(heat.min()),
                        'maximum': float(heat.max()), 'metadata': metadata})
                    print('PASS', family, metadata, flush=True)
                    del model
                report['status'] = 'PASS'
            except BaseException:
                report['status'] = 'FAIL'
                traceback.print_exc()
            finally:
                report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                report['runner_sha256'] = hashlib.sha256((HERE / 'research_runner.py').read_bytes()).hexdigest()
                (OUT / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                print('FINAL', report['status'], flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
