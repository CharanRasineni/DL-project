"""Build self-contained full training notebooks; never submits jobs."""
import hashlib
import json
from pathlib import Path
import nbformat
HERE = Path(__file__).resolve().parent
MODELS = [('03_fasterrcnn', 'fasterrcnn', 2), ('04_yolov8s', 'yolov8s', 4), ('05_rtdetr', 'rtdetr', 4)]
PROTOCOL = '''## Protocol and limitations
Standard COCO-pretrained baselines on **positive-unlabeled marine images**.
Frozen 5,817/646 image split; partial validation annotations mean unmatched predictions
are not necessarily biological false positives. No novel PU loss or test-set score is claimed.
30 epochs, seed 42, nominal 640 pixels; horizontal flip only. SGD for Faster R-CNN,
AdamW for Ultralytics. Framework resizing, gradient accumulation and checkpoint selection
differ: NOT equal compute or a definitive detector-family ranking.
FRCNN selects best COCO AP; Ultralytics selects best native fitness. Final metrics all use
pycocotools: score >0.001, candidate cap 300, COCO maxDets [1,10,100], original-pixel area ranges.
Groups: train-frequency top 3 / next 4 / remaining 25. Per-class support counts are archived.

## GPU/failure policy
Fixed conservative batches, workers=0, cache off; batch-1 final inference.
No automatic retries or full retraining for plot failures. Save weights and metrics before
optional figures. Evaluation-only mode accepts an attached trusted checkpoint.
Bootstrap is deferred to CPU after download; no GPU time spent on resampling.
Grad-CAM is qualitative class-score sensitivity, not causal evidence or novelty.
'''

def build():
    manifest = []
    for folder, family, batch in MODELS:
        path = HERE / folder / (folder + '.ipynb')
        backup = HERE / 'research_archive' / folder / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        cfg = dict(mode='train', family=family, epochs=30, imgsz=640, batch=batch,
            seed=42, score_thr=0.001, bootstrap=0, max_train_hours=9,
            split_sha256='232eb5647757077a3019d2b495fb0417c59402dad310845caf15578a0bc3b4f8')
        cells = [nbformat.v4.new_markdown_cell('# ' + folder + '\n\n' + PROTOCOL)]
        cells.append(nbformat.v4.new_code_cell('''import os, sys, subprocess
from pathlib import Path
# Only trusted official/local checkpoints: .pt files may contain pickled objects.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['WANDB_DISABLED'] = 'true'
try:
    import pycocotools  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'],
                   check=True, timeout=300)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.3.40'],
               check=True, timeout=300)
# Never pin numpy here: Kaggle's pandas is compiled against the preinstalled numpy.
WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK))
'''))
        for name in ('data.py', 'research_runner.py'):
            source = (HERE / name).read_text(encoding='utf-8')
            digest = hashlib.sha256(source.encode()).hexdigest()
            cells.append(nbformat.v4.new_markdown_cell(f'## Bundled {name}\nSHA256 `{digest}`; exact source, no stale mounted imports.'))
            cells.append(nbformat.v4.new_code_cell(f'%%writefile /kaggle/working/{name}\n' + source))
        cells.append(nbformat.v4.new_code_cell('CONFIG = ' + repr(cfg) + '\nprint(CONFIG)'))
        cells.append(nbformat.v4.new_markdown_cell('## Full training and evaluation\nArchives checkpoints, predictions, ground truth, per-class/size/group metrics, curves, qualitative images and available Grad-CAMs.'))
        cells.append(nbformat.v4.new_code_cell('''import traceback
import data as D
import research_runner as R
D.selftest()
try:
    OUTPUT = R.run(CONFIG)
except Exception:
    error_dir = WORK / (CONFIG['family'] + '_' + CONFIG['mode'])
    error_dir.mkdir(exist_ok=True)
    R.atomic_json(error_dir / 'failed.json', {'status': 'FAILED', 'config': CONFIG,
        'traceback': traceback.format_exc(), 'automatic_retry': False})
    raise
print((OUTPUT / 'results.json').read_text())
print((OUTPUT / 'stage_report.json').read_text())
'''))
        cells.append(nbformat.v4.new_markdown_cell('''## Paper-use checklist
Verify 30 epochs, 646 validation images, completed.json and stage_report.json.
Use only this run's results.json/per_class.csv, never previous logs/native metrics.
Inspect figure_manifest.json for missing/failed CAMs. Report rare-class support and
image-bootstrap intervals; single-seed results do not establish training stability.
Single-image timing is illustrative, not a dataset-wide benchmark.
These engineering upgrades do not establish novelty or publication readiness.
'''))
        nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
        nbformat.validate(nb)
        for c in cells:
            if c.cell_type == 'code':
                compile(c.source.split('\n', 1)[1] if c.source.startswith('%%writefile') else c.source, str(path), 'exec')
        nbformat.write(nb, path)
        meta = json.loads((path.parent / 'kernel-metadata.json').read_text())
        meta.update(id='rscharan/fathomnet-research-' + family, title='FathomNet research ' + family,
            is_private=True, enable_gpu=True, enable_internet=True)
        (path.parent / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        manifest.append({'path': str(path), 'kernel': meta['id'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'config': cfg})
        print('BUILT + VALIDATED', path)
    (HERE / 'research_checks/build_manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    build()
