"""CPU-only bootstrap from a downloaded result directory; never loads model weights."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--resamples', type=int, default=200)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    assert args.resamples > 1
    os.environ['FATHOMNET_DATASET_JSON'] = str(root/'val_gt.json')
    import data as D
    from research_runner import ResearchEvaluator, atomic_json
    D.DATASET_JSON = root/'val_gt.json'
    D.WORK = root
    gt = json.loads(D.DATASET_JSON.read_text())
    split = json.loads((root/'split.json').read_text())
    counts = json.loads((root/'train_counts.json').read_text())
    result = json.loads((root/'results.json').read_text())
    predictions = json.loads((root/'preds.json').read_text())
    info = D.get_cat_info(gt)
    info['train_counts'] = counts
    ev = ResearchEvaluator(split, info, score_thr=result['protocol']['score_thr'], max_det=300)
    used = D.filter_preds(predictions, score_thr=ev.score_thr, max_det=300)
    intervals = ev.bootstrap_research(used, n=args.resamples, seed=42)
    atomic_json(root/'bootstrap_ci.json', intervals)
    print(json.dumps(intervals, indent=2))

if __name__ == '__main__':
    main()
