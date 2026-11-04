"""Build ONE minimal Kaggle smoke notebook: bundled cells + 1-epoch/4-image run
per family; asserts checkpoint artifacts and PASS completion records. Not results."""
import hashlib
import json
from pathlib import Path
import nbformat
HERE = Path(__file__).resolve().parent
SPLIT_HASH = '232eb5647757077a3019d2b495fb0417c59402dad310845caf15578a0bc3b4f8'

def build():
    cells = [nbformat.v4.new_markdown_cell('''# Minimal integration gate (NOT scientific results)

Runs the exact bundled `data.py` + `research_runner.py` source, then one epoch on the
4 largest training and 4 validation images per detector family. Passes only if each
family writes a `.pt` checkpoint and a `completed.json` with status PASS.
Purpose: verify bundling, data mounts, training, checkpoint saving, evaluation and
figures before spending GPU hours on the full 30-epoch runs.''')]
    cells.append(nbformat.v4.new_code_cell('''import os, sys, subprocess
from pathlib import Path
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['WANDB_DISABLED'] = 'true'
try:
    import pycocotools  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'], check=True, timeout=300)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.3.40'], check=True, timeout=300)
WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK))
'''))
    for name in ('data.py', 'research_runner.py'):
        source = (HERE / name).read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode()).hexdigest()
        cells.append(nbformat.v4.new_markdown_cell(f'## Bundled {name} — SHA256 `{digest}`'))
        cells.append(nbformat.v4.new_code_cell(f'%%writefile /kaggle/working/{name}\n' + source))
    cells.append(nbformat.v4.new_code_cell('''import glob, json, traceback
from pathlib import Path
import data as D
import research_runner as R
D.selftest()
summary = {}
for family in ('fasterrcnn', 'yolov8s', 'rtdetr'):
    cfg = dict(mode='smoke', family=family, epochs=1, batch=2, imgsz=640, seed=42,
               score_thr=0.001, bootstrap=0, max_train_hours=1, split_sha256=''' + repr(SPLIT_HASH) + ''')
    try:
        out_dir = Path(R.run(cfg))
        weights = list(out_dir.rglob('*.pt'))
        completed = json.loads((out_dir / 'completed.json').read_text())
        assert weights, 'no .pt checkpoint artifact was written'
        assert completed['status'] == 'PASS', completed
        (WORK / (family + '_PASS.txt')).write_text('PASS', encoding='utf-8')
        summary[family] = {'status': 'PASS', 'dir': str(out_dir), 'weights': len(weights)}
        print(family, 'PASS', len(weights), 'checkpoint files', flush=True)
    except Exception:
        summary[family] = {'status': 'FAIL', 'error': traceback.format_exc()}
        print(family, 'FAIL', flush=True)
R.atomic_json(WORK / 'smoke_summary.json', summary)
failed = [k for k, v in summary.items() if v['status'] != 'PASS']
if failed:
    raise RuntimeError('Smoke failed for: ' + ', '.join(failed))
print('PASS: 1-epoch/4-image runs produced checkpoints and PASS completion records for all three families')
'''))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nbformat.validate(nb)
    for cell in cells:
        if cell.cell_type == 'code':
            compile(cell.source.split('\n', 1)[1] if cell.source.startswith('%%writefile') else cell.source, 'smoke', 'exec')
    out = HERE / 'research_smoke' / 'minimal'
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out / 'smoke.ipynb')
    meta = {'id': 'rscharan/fathomnet-research-smoke-minimal', 'title': 'FathomNet research smoke minimal',
        'code_file': 'smoke.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'kernel_sources': ['rscharan/00-setup-data-download-fathomnetclef2026', 'rscharan/fathomnet-02-split'],
        'dataset_data': {}, 'competition_sources': [], 'model_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    (HERE / 'research_checks' / 'minimal_smoke_manifest.json').write_text(json.dumps(
        {'path': str(out / 'smoke.ipynb'), 'kernel': meta['id'],
         'sha256': hashlib.sha256((out / 'smoke.ipynb').read_bytes()).hexdigest()}, indent=2))
    print('BUILT + VALIDATED', out / 'smoke.ipynb')

if __name__ == '__main__':
    build()
