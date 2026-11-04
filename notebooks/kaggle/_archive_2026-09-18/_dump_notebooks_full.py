import json
from pathlib import Path

nb_paths = [
    Path("03_fasterrcnn/03_fasterrcnn.ipynb"),
    Path("04_yolov8s/04_yolov8s.ipynb"),
    Path("05_rtdetr/05_rtdetr.ipynb"),
]

for nb_path in nb_paths:
    print(f"\n{'='*70}")
    print(f"FULL NOTEBOOK: {nb_path}")
    print(f"{'='*70}")
    if not nb_path.exists():
        print("  MISSING")
        continue
    
    with open(nb_path) as f:
        nb_data = json.load(f)
    
    for i, cell in enumerate(nb_data["cells"]):
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        
        cell_type = cell.get("cell_type", "?")
        exec_count = cell.get("execution_count", "?")
        
        outputs = cell.get("outputs", [])
        out_strs = []
        for o in outputs:
            if o.get("output_type") == "stream":
                text = "".join(o.get("text", []))
                if text.strip():
                    out_strs.append(f"--- STDOUT ---\n{text}")
            elif o.get("output_type") == "execute_result":
                data = o.get("data", {})
                tp = data.get("text/plain", "")
                if isinstance(tp, list):
                    tp = "".join(tp)
                if tp.strip():
                    out_strs.append(f"--- RESULT ---\n{tp}")
            elif o.get("output_type") == "error":
                tb = "".join(o.get("traceback", []))
                out_strs.append(f"--- ERROR ---\n{tb[-500:] if len(tb) > 500 else tb}")
        
        out_block = "\n".join(out_strs) if out_strs else "(no outputs)"
        
        print(f"\n--- Cell [{i}] (exec={exec_count}, type={cell_type}) ---")
        print(f"SOURCE:\n{src}")
        if out_block != "(no outputs)":
            print(f"\nOUTPUTS:\n{out_block}")
