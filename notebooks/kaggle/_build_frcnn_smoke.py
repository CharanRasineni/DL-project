"""Build isolated bounded GPU smoke notebook from the actual FRCNN notebook."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import nbformat

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=int, choices=[2, 50], default=2)
args = parser.parse_args()
N = args.images
HERE = Path(__file__).resolve().parent
src = HERE / '03_fasterrcnn' / '03_fasterrcnn.ipynb'
nb = nbformat.read(src, as_version=4)
# Resolve the shared module without relying on unshipped sidecar files.
payload = base64.b64encode((HERE / 'data.py').read_bytes()).decode('ascii')
setup = """import base64, hashlib, sys, subprocess
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools==2.0.11'], check=True, timeout=180)
""" + f"Path('/kaggle/working/data.py').write_bytes(base64.b64decode({payload!r}))\n"
nb.cells.insert(1, nbformat.v4.new_code_cell(setup))
s = nb.cells[2].source
s = s.replace('S = D.load_split()', """D.DATASET_JSON, IMG = D.resolve_dataset_paths()
D.IMG_ROOT = IMG
S = D.load_frozen_split()
assert set(S['train_ids']).isdisjoint(S['val_ids'])
assert torch.cuda.is_available(), 'Smoke test requires GPU'
""")
s = s.replace("torch.backends.cudnn.benchmark = True", "torch.backends.cudnn.benchmark = False")
s += """
# BOUNDED SMOKE ONLY: retain full training counts but train/evaluate 2 images.
S = dict(S, train_ids=sorted(S['train_ids'])[:2], val_ids=sorted(S['val_ids'])[:2])
train_ids, val_list = set(S['train_ids']), sorted(S['val_ids'])
val_ids = val_list
for iid in sorted(train_ids) + val_list:
    with Image.open(IMG / img_by_id[iid]['file_name']) as im:
        im.load()
        assert im.size == (img_by_id[iid]['width'], img_by_id[iid]['height'])
print('SMOKE: 2 training + 2 validation images. NOT PAPER RESULTS.')
"""
nb.cells[2].source = s
s = nb.cells[3].source
s = s.replace("weights='COCO_V1'", "weights='COCO_V1', min_size=640, max_size=640, box_score_thresh=0.001, box_detections_per_img=300")
s = s.replace('num_workers=2', 'num_workers=0').replace('batch_size=4', 'batch_size=1')
s = s.replace("boxes = [[0, 0, 1, 1]]; labels = [1]; areas = [1.0]", "boxes = []; labels = []; areas = []")
s = s.replace("torch.as_tensor(boxes, dtype=torch.float32)", "torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)")
nb.cells[3].source = s
nb.cells[4].source = nb.cells[4].source.replace('EPOCHS, HIST = 8, []', 'EPOCHS, HIST = 1, []')
# Keep the notebook inference/evaluation code but make timing bounded.
s = nb.cells[5].source.replace('range(30)', 'range(2)').replace('range(5)', 'range(1)')
s = s.replace('torch.rand(3, 800, 1280)', 'torch.rand(3, 640, 640)')
s += """
# Reload smoke checkpoint and verify parameters survived serialization exactly.
checkpoint = torch.load(OUT / 'fasterrcnn_last.pt', map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 1
for name, value in model.state_dict().items():
    assert torch.equal(value.detach().cpu(), checkpoint['model'][name]), name
D.save_json(OUT / 'smoke_pass.json', {
    'status': 'PASS', 'purpose': 'integration only, not paper results',
    'train_images': 2, 'val_images': 2, 'epochs': 1,
    'checkpoint_roundtrip': True, 'gpu': torch.cuda.get_device_name(0),
    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
    'data_sha256': hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest()})
"""
nb.cells[5].source = s
nb.cells[0].source = '# Faster R-CNN bounded integration smoke test\n\nNOT a scientific result. One batch, checkpoint roundtrip, inference and shared evaluation.'
for c in nb.cells:
    if c.cell_type == 'code':
        c.outputs, c.execution_count = [], None
        compile(c.source, str(src), 'exec')
nbformat.validate(nb)
# Update bounded-run counts and provenance without changing the passed code path.
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("[:2]", f"[:{N}]")
        cell.source = cell.source.replace("'train_images': 2, 'val_images': 2", f"'train_images': {N}, 'val_images': {N}")
        cell.source = cell.source.replace('2 training + 2 validation', f'{N} training + {N} validation')
        cell.source = cell.source.replace('min800/max1333', 'min640/max640')
        cell.source = cell.source.replace('8-epoch pilot; full 30-epoch budget not run for this model', f'{N}-image one-epoch integration test, NOT a paper result')
        cell.source = cell.source.replace("D.write_provenance(OUT / 'provenance_frcnn.json', proto, R)", "D.write_provenance(OUT, proto, R)")
        compile(cell.source, str(src), 'exec')
nb.cells[0].source = f'# Faster R-CNN bounded {N}-image test\n\n{N} train + {N} val images, one epoch. NOT paper results.'
out = HERE / 'research_smoke' / ('fasterrcnn' if N == 2 else 'fasterrcnn50')
out.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out / 'smoke.ipynb')
meta = json.loads((src.parent / 'kernel-metadata.json').read_text())
meta.update(id='rscharan/fathomnet-research-smoke-frcnn', title='FathomNet research smoke FRCNN',
            code_file='smoke.ipynb', enable_gpu=True, is_private=True, enable_internet=True)
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('Built and compiled', out / 'smoke.ipynb')
