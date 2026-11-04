"""Apply the narrowly scoped Faster R-CNN evaluation upgrade."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / '03_fasterrcnn' / '03_fasterrcnn.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
cell = nb['cells'][4]
source = ''.join(cell['source'])
old = """# ── unified COCO eval via data.py Evaluator ──
evaluator = D.Evaluator(S, cat_info, score_thr=0.01, max_det=300)
results = evaluator.evaluate(preds, out_dir=OUT / 'eval_frcnn')"""
new = """# BEGIN VERIFIED EVALUATION
# Groups must come from the frozen TRAIN annotations, never validation counts.
assert set(S['train_ids']).isdisjoint(S['val_ids']), 'Train/validation overlap'
assert cat_info['train_counts'], 'Missing frozen training class counts'
evaluator = D.FullEvaluator(S, cat_info, score_thr=0.01, max_det=300)
assert evaluator.groups == D.define_groups(cat_info['train_counts'], head_k=3, mid_k=4)
results = evaluator.evaluate(preds, out_dir=OUT / 'eval_frcnn', bootstrap=0)
results['protocol']['coco_maxDets'] = [1, 10, 100]
results['protocol']['prediction_cap_per_image'] = 300
results['train_instance_counts'] = cat_info['train_counts']
results['evaluation_note'] = (
    'COCO bbox AP against available validation annotations; annotations are '
    'positive-unlabeled. Unmatched predictions are not necessarily biological '
    'false positives. Prediction cap=300/image; COCO maxDets=[1,10,100].'
)
# Save the complete metric record BEFORE optional timing or plotting can fail.
D.save_json(OUT / 'eval_frcnn' / 'results.json', results)
D.save_json(OUT / 'results_fasterrcnn.json', results)
pd.DataFrame([
    {'class': name, 'train_instances': cat_info['train_counts'].get(name, 0),
     'AP50-95': ap, 'AP50': results['per_class_AP50'][name],
     'group': next(g for g, names in results['groups'].items() if name in names)}
    for name, ap in results['per_class_AP50-95'].items()
]).to_csv(OUT / 'eval_frcnn' / 'per_class.csv', index=False)
# END VERIFIED EVALUATION"""
if old in source:
    source = source.replace(old, new, 1)
    source = source.replace("R = {\n    'model':", "R = {\n    **results,  # retain size AP, per-class AP50, groups and evaluator protocol\n    'model':", 1)
    source = source.replace("'protocol': {'seed': SEED", "'protocol': {**results['protocol'], 'seed': SEED", 1)
    cell['source'] = source.splitlines(keepends=True)
    # Changed code must not retain stale execution outputs.
    cell['outputs'] = []
    cell['execution_count'] = None
    header = ''.join(nb['cells'][0]['source'])
    header += ('\nEvaluation upgraded to FullEvaluator: full per-class/size metrics and '
               'frozen training-count groups are archived before optional analysis. '
               'Training remains the original 8-epoch pilot; this edit does not '
               'make the training protocol research-ready.\n')
    nb['cells'][0]['source'] = header.splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Updated:', path)
else:
    assert '# BEGIN VERIFIED EVALUATION' in source, 'Unexpected notebook contents'
    print('Already updated:', path)
