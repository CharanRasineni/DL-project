"""CPU train/checkpoint/CAM integration; synthetic data, no research claims."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import json
import tempfile
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, settings
import research_runner as R

HERE = Path(__file__).resolve().parent
OUT = Path(tempfile.mkdtemp(prefix='ultra_cpu_', dir=HERE/'research_checks'))
(HERE/'research_checks/latest_ultra_cpu.txt').write_text(str(OUT))

class CPUModel(YOLO):
    def __init__(self, ignored):
        super().__init__('yolov8s.yaml')
    def train(self, **kwargs):
        kwargs.update(device='cpu', amp=False)
        return super().train(**kwargs)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = OUT/'dataset'
    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'synthetic'}]}
    for part, ids in [('train', [0, 1]), ('val', [2, 3])]:
        for sub in ('images', 'labels'):
            (root/sub/part).mkdir(parents=True)
        for iid in ids:
            rgb = np.random.default_rng(iid).integers(0, 256, (64,64,3), dtype=np.uint8)
            path = root/'images'/part/f'{iid}.png'
            Image.fromarray(rgb).save(path)
            (root/'labels'/part/f'{iid}.txt').write_text('0 0.5 0.5 0.375 0.375\n')
            coco['images'].append(dict(id=iid,file_name=f'{part}/{iid}.png',width=64,height=64))
            coco['annotations'].append(dict(id=iid+1,image_id=iid,category_id=1,bbox=[20,20,24,24]))
    yaml = root/'dataset.yaml'
    yaml.write_text('path: '+json.dumps(root.as_posix())+'\ntrain: images/train\nval: images/val\nnames: {0: synthetic}\n')
    original = R.ultra_class
    R.ultra_class = lambda family: CPUModel
    cfg = dict(family='yolov8s', epochs=1, batch=2, imgsz=64, seed=42)
    try:
        checkpoint = R.train_ultra(cfg,yaml,OUT)
    finally:
        R.ultra_class = original
    assert checkpoint.is_file()
    assert (checkpoint.parent/'resume.pt').is_file()
    net = YOLO(str(checkpoint))
    predictions = []
    for iid in (2,3):
        for result in net.predict(str(root/'images/val'/f'{iid}.png'),imgsz=64,conf=1e-8,device='cpu',verbose=False):
            for box,score in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist()):
                x,y,x2,y2=box
                predictions.append(dict(image_id=iid,category_id=1,score=score,bbox=[x,y,x2-x,y2-y]))
    assert predictions, 'Random model produced no predictions for CAM target selection'
    del net
    net=YOLO(str(checkpoint))
    manifest=R.figures(net,cfg,coco,dict(val_ids=[2,3]),root/'images',predictions,OUT)
    passed=[row for row in manifest['images'] if row['cam_status']=='ok']
    assert passed, manifest
    for row in passed:
        heat=np.load(OUT/f"gradcam_{row['image_id']}.npy")
        assert np.isfinite(heat).all() and heat.max()>heat.min()
    R.atomic_json(OUT/'test_result.json',dict(status='PASS',synthetic_only=True,
        overrides=['CPU adapter','64px','random initialization','very low prediction threshold for random model'],
        checkpoint=str(checkpoint),cams=len(passed)))
    print('PASS CPU train/checkpoint/CAM',OUT,flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception:
        R.atomic_json(OUT/'test_result.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise
