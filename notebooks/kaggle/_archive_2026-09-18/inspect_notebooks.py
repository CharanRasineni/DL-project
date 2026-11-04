import json

for name, path in [('04_YOLOv8s', '04_yolov8s/04_yolov8s.ipynb'),
                   ('05_RT-DETR', '05_rtdetr/05_rtdetr.ipynb')]:
    nb = json.load(open(path))
    print(f'\n=== {name} NOTEBOOK: {len(nb["cells"])} cells ===')
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = c.get('source') or []
        src = ''.join(src) if isinstance(src, list) else src
        exec_count = c.get('execution_count', '?')
        print(f'\n--- Cell {i} (executed={exec_count}) ---')
        print(src[:1500])