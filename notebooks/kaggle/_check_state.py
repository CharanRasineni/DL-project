"""Check current state of all experiment outputs."""
import json
from pathlib import Path

HERE = Path(__file__).parent
for d in ['03_fasterrcnn', '04_yolov8s', '05_rtdetr']:
    out = HERE / d / 'output'
    print(f'\n=== {d} ===')
    if not out.exists():
        print('  (no output dir)')
        continue
    for f in sorted(out.iterdir()):
        sz = f.stat().st_size
        print(f'  {f.name}: {sz} bytes')
        if f.suffix == '.json' and f.name.startswith('results'):
            try:
                data = json.loads(f.read_text())
                print(f'    mAP50-95: {data.get("mAP50-95", "N/A")}')
                print(f'    mAP50: {data.get("mAP50", "N/A")}')
                proto = data.get('protocol', {})
                print(f'    epochs: {proto.get("epochs", "N/A")}')
                print(f'    seed: {proto.get("seed", "N/A")}')
                print(f'    input: {proto.get("input", "N/A")}')
            except Exception as e:
                print(f'    (parse error: {e})')

# Also check curves
print('\n=== Curve files ===')
for d in ['03_fasterrcnn', '04_yolov8s', '05_rtdetr']:
    out = HERE / d / 'output'
    csv = out / f'curve_{d.split("_")[1]}.csv'
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        print(f'  {csv.name}: {len(df)} epochs, last mAP50-95={df.iloc[-1]["metrics/mAP50-95(B)"]:.4f}')