import json
p = '05_rtdetr/05_rtdetr.ipynb'
nb = json.load(open(p))

def src(i):
    return ''.join(nb['cells'][i]['source'])

def setsrc(i, t):
    nb['cells'][i]['source'] = [t]

# cell0 markdown: append GPU-safe note
s0 = src(0)
if 'GPU-SAFE' not in s0:
    s0 = s0.rstrip() + ('\n\nGPU-SAFE SETTINGS (T4 15 GB): train batch=4, '
        'workers=1, cache=disk, amp on; val batch=4 streamed from folder. '
        'Same 30-epoch/imgsz-640 protocol, only micro-batch is smaller.')
    setsrc(0, s0)

# cell4 train: batch8->4 workers2->1 cache disk amp
e4 = src(4)
e4 = e4.replace(
    '# --- 3. Train RT-DETR-L (transformer; heavier per step than YOLO at same epochs) ---',
    '# --- 3. Train RT-DETR-L (GPU-safe for T4 15GB) ---\n'
    '# batch=4 + workers=1 + cache=disk keeps peak well under 14.6 GiB.\n'
    '# Same 30-epoch / imgsz-640 protocol, only the micro-batch is smaller.\n'
    'import gc; gc.collect()\n'
    'if torch.cuda.is_available():\n'
    '    torch.cuda.empty_cache()')
e4 = e4.replace(
    'epochs=30, imgsz=640, batch=8, seed=SEED,\n'
    '            workers=2, project=',
    'epochs=30, imgsz=640, batch=4, seed=SEED,\n'
    "            workers=1, cache='disk', amp=True, project=")
setsrc(4, e4)

# cell5 eval: folder source already; add gc pre + batch=4 + empty_cache per iter
s5 = src(5)
if 'gc.collect()' not in s5:
    s5 = s5.replace(
        '# --- 4. Predict on val + unified COCO evaluation ---',
        '# --- 4. Predict on val + unified COCO evaluation ---\n'
        'import gc; gc.collect()\n'
        'if torch.cuda.is_available():\n'
        '    torch.cuda.empty_cache()')
s5 = s5.replace(
    'max_det=300, imgsz=640, stream=True, verbose=False)',
    'max_det=300, imgsz=640, batch=4, stream=True, verbose=False)')
if s5.count('torch.cuda.empty_cache()') < 2:
    s5 = s5.replace(
        "json.dump(preds, open(OUT / 'preds_rtdetr.json', 'w'))",
        "    torch.cuda.empty_cache()\n"
        "json.dump(preds, open(OUT / 'preds_rtdetr.json', 'w'))")
setsrc(5, s5)

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('05 edited OK')
