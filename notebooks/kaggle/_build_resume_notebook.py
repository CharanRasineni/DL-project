"""Build ONE resume notebook: continues the budget-stopped FRCNN run for epochs 27-30.

Attaches the failed kernel's output as a Kaggle input; finds last.pt at runtime.
Not submitted automatically by this script.
"""
import hashlib
import json
from pathlib import Path
import nbformat
HERE = Path(__file__).resolve().parent
SPLIT_HASH = '232eb5647757077a3019d2b495fb0417c59402dad310845caf15578a0bc3b4f8'

def build():
    cells = [nbformat.v4.new_markdown_cell('''# Faster R-CNN resume: epochs 27-30 after budget stop

The previous run completed 26/30 epochs and stopped via the declared `max_train_hours`
budget check, not a crash. This kernel attaches that run's output as an input and
resumes from `last.pt` (exact model/optimizer/scheduler/scaler/RNG state), then runs the
standard evaluation, figures and Grad-CAM stages. Known bookkeeping gap: the epoch-26
validation row (AP 0.32741, from the stopped kernel's log) is not in the checkpoint's
saved history; resumed history therefore holds 25 + 4 rows. Overall-best semantics are
preserved by seeding `best.pt` from the attached run (epoch 24, AP 0.32905).
Still a single-seed result on partial validation annotations; not equal compute with
the other detectors.''')]
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
    cells.append(nbformat.v4.new_code_cell('''import glob, os
print('/kaggle/input mounts:', sorted(os.listdir('/kaggle/input')), flush=True)
candidates = sorted(glob.glob('/kaggle/input/**/last.pt', recursive=True))
print('last.pt candidates:', candidates, flush=True)
assert len(candidates) == 1, f'expected exactly one attached last.pt, found {candidates}'
RESUME_CKPT = candidates[0]
CONFIG = dict(mode='train', family='fasterrcnn', epochs=30, imgsz=640, batch=2, seed=42,
    score_thr=0.001, bootstrap=0, max_train_hours=4.5,
    resume_checkpoint=RESUME_CKPT,
    split_sha256=''' + repr(SPLIT_HASH) + ''')
import torch
state = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
print('attached checkpoint epoch:', state['epoch'], '| history rows:', len(state['history']))
assert state['epoch'] == 26 and state['config']['split_sha256'] == CONFIG['split_sha256']
print(CONFIG)
'''))
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
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nbformat.validate(nb)
    for cell in cells:
        if cell.cell_type == 'code':
            compile(cell.source.split('\n', 1)[1] if cell.source.startswith('%%writefile') else cell.source, 'resume', 'exec')
    out = HERE / '03_fasterrcnn_resume'
    out.mkdir(exist_ok=True)
    nbformat.write(nb, out / 'resume.ipynb')
    meta = {'id': 'rscharan/fathomnet-research-frcnn-resume', 'title': 'FathomNet research FRCNN resume',
        'code_file': 'resume.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'kernel_sources': ['rscharan/00-setup-data-download-fathomnetclef2026',
                           'rscharan/fathomnet-02-split'],
        'dataset_sources': ['rscharan/fathom-net-fasterrcnn'],
        'dataset_data': {}, 'competition_sources': [], 'model_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    (HERE / 'research_checks' / 'resume_manifest.json').write_text(json.dumps(
        {'path': str(out / 'resume.ipynb'), 'kernel': meta['id'],
         'sha256': hashlib.sha256((out / 'resume.ipynb').read_bytes()).hexdigest()}, indent=2))
    print('BUILT + VALIDATED', out / 'resume.ipynb')

if __name__ == '__main__':
    build()
