"""
train_watch.py - manages the 3-model training pipeline from Cursor.

Loop: check status of 03/04/05 -> push 04 as soon as a GPU slot frees ->
collect outputs (figures/tables/results jsons) whenever a kernel completes.

Run:  python train_watch.py            (poll forever until all done)
      python train_watch.py --collect-only   (just collect finished kernels)
Stop anytime with Ctrl+C - runs keep going on Kaggle; re-run later.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
K = ["eternalsorcerer/fathomnet-03-fasterrcnn",
     "eternalsorcerer/fathomnet-04-yolov8s",
     "eternalsorcerer/fathomnet-05-rtdetr"]
DIRS = {K[0]: HERE / "03_fasterrcnn", K[1]: HERE / "04_yolov8s", K[2]: HERE / "05_rtdetr"}

os.environ["PYTHONUTF8"] = "1"
os.environ.setdefault("KAGGLE_API_TOKEN", "")
if not os.environ["KAGGLE_API_TOKEN"]:
    key = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["key"]
    os.environ["KAGGLE_API_TOKEN"] = key


def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def status(slug):
    _, s = run(f"kaggle kernels status {slug}")
    if "COMPLETE" in s:
        return "COMPLETE"
    if "ERROR" in s or "CANCEL" in s:
        return "ERROR"
    if "RUNNING" in s:
        return "RUNNING"
    if "QUEUED" in s:
        return "QUEUED"
    return "UNKNOWN"


def collect(slug):
    d = DIRS[slug]
    (d / "output").mkdir(exist_ok=True)
    code, out = run(f"kaggle kernels output {slug} -p output", cwd=str(d))
    print(f"  collected {slug} -> {d/'output'}")
    for line in out.strip().splitlines()[-3:]:
        print("   ", line)


def main():
    collect_only = "--collect-only" in sys.argv
    done = {}
    # a kernel counts as already-launched if its latest version isn't in an ERROR/UNKNOWN state
    launched = {k: status(k) not in ("ERROR", "UNKNOWN") for k in K}
    pushed04 = launched[K[1]]
    FORCE_05 = "--force-05" in sys.argv  # pass this flag once: 05 v2 runs a broken eval; repush v3 at first free slot

    def launched04_run(states):
        return states[K[1]] == "RUNNING"
    while True:
        states = {k: status(k) for k in K}
        print(time.strftime("%H:%M:%S"), states, flush=True)
        for k, st in states.items():
            if st in ("COMPLETE", "ERROR") and k not in done:
                done[k] = st
                collect(k)
        if all(k in done for k in K):
            print("ALL DONE:", done)
            return
        if collect_only:
            return
        # push pending kernels when a GPU slot is free (max 2 concurrent)
        if list(states.values()).count("RUNNING") < 2:
            if not pushed04 and not launched04_run(states):
                code, out = run("kaggle kernels push -p .", cwd=str(DIRS[K[1]]))
                print("  push 04:", out.strip()[:120], flush=True)
                if "successfully" in out:
                    pushed04 = True
            # 05 v2 was pushed with a broken eval cell -> force repush v3 when possible
            if FORCE_05 and states[K[2]] != "RUNNING":
                code, out = run("kaggle kernels push -p .", cwd=str(DIRS[K[2]]))
                print("  push 05 v3:", out.strip()[:120], flush=True)
                if "successfully" in out:
                    FORCE_05 = False
        time.sleep(120)


if __name__ == "__main__":
    main()
