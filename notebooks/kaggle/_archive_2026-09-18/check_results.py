import json, os

base = r"C:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle"
kernels = {
    "03_fasterrcnn": os.path.join(base, "03_fasterrcnn", "output"),
    "04_yolov8s": os.path.join(base, "04_yolov8s", "output"),
    "05_rtdetr": os.path.join(base, "05_rtdetr", "output"),
}

for name, d in kernels.items():
    print(f"=== {name} ===")
    files = os.listdir(d)
    print("files:", sorted(files))
    for f in sorted(files):
        if f.endswith(".json"):
            fp = os.path.join(d, f)
            try:
                r = json.load(open(fp))
                print(f"  {f}:")
                print(f"    mAP50-95: {r.get('mAP50-95', '?')}")
                print(f"    mAP50: {r.get('mAP50', '?')}")
                print(f"    params: {r.get('params_million', '?')}M")
                print(f"    latency: {r.get('latency_ms_per_image', '?')}ms")
                head = r.get("head_mid_tail_AP50-95", {})
                if head:
                    print(f"    head/mid/tail AP50-95: {head}")
            except Exception as e:
                print(f"  {f}: error={e}")
    print()
