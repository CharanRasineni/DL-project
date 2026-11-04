"""CPU integration test of actual notebook cells, using synthetic data only."""
import json
import os
import tempfile
from pathlib import Path
import nbformat
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
nb = nbformat.read(HERE / '03_fasterrcnn' / '03_fasterrcnn.ipynb', as_version=4)
with tempfile.TemporaryDirectory(prefix='frcnn_smoke_') as temp:
    root = Path(temp)
    images, annotations = [], []
    for iid in range(4):
        rgb = np.random.default_rng(iid).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(root / f'{iid}.jpg')
        images.append({'id': iid, 'file_name': f'{iid}.jpg', 'width': 64, 'height': 64})
        annotations.append({'id': iid+1, 'image_id': iid, 'category_id': 0,
                            'bbox': [12, 12, 24, 24], 'area': 576, 'iscrowd': 0})
    coco = {'images': images, 'annotations': annotations, 'categories': [{'id': 0, 'name': 'synthetic'}]}
    split = {'train_ids': [0, 1], 'val_ids': [2, 3], 'val_fraction': 0.5, 'seed': 42}
    (root / 'annotations.json').write_text(json.dumps(coco))
    (root / 'split.json').write_text(json.dumps(split))
    os.environ['FATHOMNET_DATASET_JSON'] = str(root / 'annotations.json')
    os.environ['FATHOMNET_SPLIT_JSON'] = str(root / 'split.json')
    os.environ['FATHOMNET_IMG_ROOT'] = str(root)
    import data as D
    D.DATASET_JSON, D.SPLIT_JSON, D.WORK = root / 'annotations.json', root / 'split.json', root
    scope = {'__name__': '__main__'}
    setup = nb.cells[1].source.replace("Path('/kaggle/working')", 'Path(' + repr(str(root)) + ')')
    exec(compile(setup, 'actual_notebook_setup', 'exec'), scope)
    scope['IMG'] = root
    dataset = nb.cells[2].source.replace("weights='COCO_V1'", "weights=None, weights_backbone=None, min_size=64, max_size=64")
    dataset = dataset.replace('num_workers=2', 'num_workers=0').replace('batch_size=4', 'batch_size=1')
    exec(compile(dataset, 'actual_notebook_dataset', 'exec'), scope)
    train = nb.cells[3].source.replace('EPOCHS, HIST = 8, []', 'EPOCHS, HIST = 1, []')
    exec(compile(train, 'actual_notebook_train', 'exec'), scope)
    evaluation = nb.cells[4].source.replace('torch.rand(3, 800, 1280)', 'torch.rand(3, 64, 64)')
    evaluation = evaluation.replace('range(30)', 'range(2)').replace('range(5)', 'range(1)')
    # A CPU notebook run must not call CUDA unconditionally.
    evaluation = evaluation.replace('torch.cuda.synchronize()', "(torch.cuda.synchronize() if torch.cuda.is_available() else None)")
    exec(compile(evaluation, 'actual_notebook_evaluation', 'exec'), scope)
    state = torch.load(root / 'fasterrcnn_last.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == 1
    for name, value in scope['model'].state_dict().items():
        assert torch.equal(value.cpu(), state['model'][name]), name
    saved = json.loads((root / 'results_fasterrcnn.json').read_text())
    assert 'per_class_size_AP50-95' in saved and len(saved['per_class_AP50']) == 1
    result = {'status': 'PASS', 'synthetic_only': True, 'device': 'cpu',
              'training_steps': 1, 'train_images': 2, 'val_images': 2,
              'checkpoint_roundtrip': True, 'notebook_cells_executed': [1, 2, 3, 4],
              'overrides': ['temporary data/output paths', '64px', 'no pretrained weights',
                            'workers=0', '1 epoch', 'short CPU-safe timing']}
    out = HERE / 'research_checks' / 'cpu_smoke.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print('PASS: actual notebook CPU synthetic training/checkpoint/inference/evaluation', flush=True)
