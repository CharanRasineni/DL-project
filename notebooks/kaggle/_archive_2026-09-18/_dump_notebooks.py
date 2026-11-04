import json
from pathlib import Path

nb_paths = [
    Path("03_fasterrcnn/03_fasterrcnn.ipynb"),
    Path("04_yolov8s/04_yolov8s.ipynb"),
    Path("05_rtdetr/05_rtdetr.ipynb"),
    Path("05_rtdetr/05b/05b_eval.ipynb"),
]

for nb_path in nb_paths:
    print(f"\n{'='*70}")
    print(f"NOTEBOOK: {nb_path}")
    print(f"{'='*70}")
    if not nb_path.exists():
        print("  MISSING")
        continue
    
    with open(nb_path) as f:
        nb_data = json.load(f)
    
    md = nb_data.get("metadata", {})
    print(f"\n  kernelspec: {md.get('kernelspec', {})}\n  language_info: {md.get('language_info', {})}\n")
    
    print(f"  CELLS ({len(nb_data['cells'])}):")
    for i, cell in enumerate(nb_data["cells"]):
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        src_short = src.replace("\n", " | ").strip()[:250]
        cell_type = cell.get("cell_type", "?")
        exec_count = cell.get("execution_count", "?")
        
        if cell_type == "code":
            # Also show outputs briefly
            outputs = cell.get("outputs", [])
            out_summary = []
            for o in outputs:
                if o.get("output_type") == "stream":
                    text = "".join(o.get("text", []))[:100]
                    out_summary.append(f"  stdout: {text!r}")
                elif o.get("output_type") == "execute_result":
                    data = o.get("data", {})
                    text = data.get("text/plain", [""])[:1] if isinstance(data.get("text/plain"), list) else str(data.get("text/plain", ""))[:100]
                    out_summary.append(f"  result: {text}")
                elif o.get("output_type") == "error":
                    out_summary.append(f"  ERROR: {o.get('ename', '')}")
            out_str = " | ".join(out_summary) if out_summary else "(no outputs)"
            print(f"    [{i}] exec={exec_count} | {cell_type} | src: {src_short}")
            print(f"         outputs: {out_str}")
        else:
            print(f"    [{i}] {cell_type} | src: {src_short}")
