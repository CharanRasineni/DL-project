# Research rerun protocol and paper checklist

## Scope
These are instrumented standard-detector baselines, not a novel PU method. Grad-CAM
illustrates sensitivity of a declared class-score objective; it is neither a required
publication criterion nor evidence of novelty or biological correctness.

## Experiments
- Frozen image split: 5,817 training / 646 validation; seed 42. Exact membership hash checked.
- 32 categories mapped to contiguous detector labels; original category IDs restored for COCO evaluation.
- 30 epochs, nominal 640 pixels, pretrained models, horizontal-flip augmentation.
- FRCNN: SGD, batch 2, per-epoch COCO evaluation, best AP checkpoint.
- YOLOv8s / RT-DETR-L: AdamW, batch 4, native validation-fitness checkpoint selection.
- These are **not equal-compute experiments**. Resizing, gradient accumulation, checkpoint
  selection and optimization differ. A single training seed cannot establish family superiority.
- Final evaluation: score threshold 0.001, 300 prediction candidates/image,
  COCO maxDets [1,10,100]. AP/50/75, original-pixel size AP, per-class AP, head/mid/tail AP.
- Groups: train frequency top 3, next 4, remaining 25; not cumulative frequency thresholds.
- Validation is partially annotated (positive-unlabeled). Interpret unmatched predictions cautiously.

## Artifacts per run
config.json, split.json, train_counts.json, provenance.json (source/data/checkpoint hashes,
environment), recoverable checkpoints, preds.json, val_gt.json, results.json,
per_class.csv (training and validation counts), training_history.csv, training_curves.png/pdf,
qualitative images and available Grad-CAM PNG/PDF/NPY, figure_manifest.json,
stage_report.json, completed.json (or failed.json for failures inside R.run).

Optional figure failures do not erase core training/evaluation outputs. A completion status
of CORE_COMPLETE_ANALYSIS_INCOMPLETE is NOT an all-analysis PASS. Inspect per-image CAM status:
blank/zero heatmaps are rejected. Uniform seeded image selection is shared across models;
CAM target classes can differ because each model's highest-scoring prediction is explained.
CAM square resizing differs from AP preprocessing and is recorded in the figure manifest.

Latency currently measures one fixed validation image, decoded RGB to CPU detections,
including transfer/pre/postprocessing, excluding disk/decode. It is not dataset-wide timing.
Do not use it to claim general hardware-independent detector-family speedups.

## Avoid more GPU spending
No automatic training retry. Latest FRCNN optimizer/scaler state and Ultralytics resume.pt
are retained; recovery needs manual checkpoint inspection. Evaluation-only mode accepts an
explicit trusted checkpoint. Never load third-party untrusted pickled .pt files.
Image bootstrap runs separately on CPU via bootstrap_research.py using downloaded artifacts;
200 resamples is a starting point, not training-seed uncertainty. Rare-class absence varies
between resamples. Keep the machine online if using a local Kaggle watcher.

## Submission audit
The first two full submissions failed at import time after an erroneous numpy downgrade.
No epochs completed in those submissions. The numpy pin was removed. A real Kaggle smoke
then exposed an incompatible optional Ray Tune callback, now disabled. FRCNN's first smoke
completed core analysis and four CAMs; the surrounding test had an incorrect checkpoint
file glob. That assertion is corrected. The current smoke requires a .pt artifact,
completed.json PASS, and a per-family PASS.txt marker. Do not infer success from a job being
submitted or from an older smoke output. Check the remote version and source hashes.

## Watcher recovery (2026-09-17)
The local watcher stopped on Windows `PermissionError` while replacing watch_state.json;
this was not a remote training failure. State saving now uses unique temporary files,
strict JSON, and bounded filesystem-only retries. Permanent failure preserves the previous
state and blocks submission. Three offline regression tests reproduce/cover these cases:
`python _test_watch_research.py` (run from this directory).
Launch the watcher with Python UTF-8 mode (`python -X utf8 -u _watch_research.py`) to avoid
Windows log-download encoding errors. Keep only one watcher process active. The queued
RT-DETR submission still occurs at most once; failed training is never automatically retried.
The state file and watcher log, not this document, determine current run status.

## Before paper submission
1. Require all 30 training-history rows and 646 evaluated images for each final model.
2. Recompute manuscript tables only from each new results.json/per_class.csv.
3. Add CPU bootstrap intervals and validation support next to rare-class APs.
4. Use genuine exported qualitative/CAM figures; omit unavailable explanations honestly.
5. Correct task framing to positive-unlabeled; do not call these models PU-aware.
6. Remove unsupported plateau, equal-budget, definitive family-ranking and test-set claims.
7. Official test evaluation, multi-seed stability, PU ablations, broader latency measurement,
   and a current literature novelty assessment remain additional work.
8. The manuscript was NOT automatically updated with nonexistent new results.

Prior notebook snapshots are preserved in research_archive. Prior results remain in their
original output directories. New kernels use fathomnet-research-* identities.
