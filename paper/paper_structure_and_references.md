# Paper Structure and References

FathomNet CLEF 2026 Detector Benchmark
Skeleton only — every section below is a stub in `v1.tex`. Fill in once the
Kaggle runs (04 YOLOv8s, 05 RT-DETR) finish and the JSON result files are
copied into `paper/json/` and `paper/images/`.

> Scaffold mirrors `DarkMatter/CMS2015DarkMatterSearch(HTMHT)/paper`
> (`v1.tex` + `references.bib` + `.latexmkrc` + build/watch scripts) and
> reuses its template: `twocolumn` + `authblk`, `unsrt` bibliography with
> `\nocite{*}`, `images/` as the figure directory, a `Declarations` block,
> and the `\clearpage\onecolumn\raggedbottom\appendix` appendix convention.

---

## 1. Title and Abstract

**Working title**

> Object Detection in Under-Sampled Marine Biodiversity Imagery:
> Benchmarking Two-Stage, One-Stage, and Transformer-Based Detectors on
> FathomNet CLEF 2026

**Abstract contents (one paragraph, ~150 words)**

1. Problem: automated marine biodiversity monitoring from ROV/AUV imagery.
2. Data: FathomNet CLEF 2026 — images, instances, class count, long tail.
3. Models benchmarked: Faster R-CNN R50-FPN, YOLOv8s, RT-DETR-R18,
   RT-DETR-L.
4. Protocol: identical 90/10 stratified split, seed 42, COCO mAP.
5. Headline result: TBD after runs complete.
6. Scope statement: this is a **benchmark/comparison study**, not a new
   architecture.

---

## 2. Section Plan

| § | Section | Sub-sections | Source artefact |
|---|---------|--------------|-----------------|
| 1 | Introduction | motivation, FathomNet/CLEF, contributions | — |
| 2 | Data | dataset, imbalance/long-tail, geometry, split protocol | notebook 01/02 EDA |
| 3 | Methods | evaluation protocol, 4 detector subsections, training details | notebooks 03/04/05 |
| 4 | Results | main, per-class, head/mid/tail, size, latency, qualitative | `result.json` files |
| 5 | Discussion | findings, RT-DETR-L memory ceiling, transferability | — |
| 6 | Limitations | training budgets, single split/seed, val-only, hardware | — |
| 7 | Conclusion | summary, recommendation, future work | — |
| — | Declarations | funding, ethics, conflicts, data/code availability | — |
| A | Appendix | per-class AP, hyperparameters, EDA figures, reproducibility | `json/`, `images/` |

### 1. Introduction

- Motivation: scaling manual annotation/ID of deep-sea imagery.
- Why it is hard: turbidity, variable illumination, occlusion, camouflage,
  tiny/rarified taxa, extreme class imbalance.
- FathomNet + CLEF 2026 track context.
- Prior work: two-stage vs. one-stage vs. transformer detectors; underwater
  detection surveys.
- Contributions (bullet list):
  1. A controlled, equal-split four-detector benchmark on FathomNet CLEF 2026.
  2. Head/mid/tail and object-size stratified error analysis.
  3. An accuracy-vs-latency deployment trade-off for ROV/AUV use.
  4. A documented GPU-memory ceiling for RT-DETR-L on consumer/Kaggle GPUs.

### 2. Data

- **2.1 FathomNet CLEF 2026** — scope, collection platforms, annotation
  protocol, image/instance/class counts, blind test size.
- **2.2 Class imbalance and long tail** — counts per class; define
  head = top 3, mid = next 4, tail = remaining 25.
  Table `tab:class-counts`.
- **2.3 Image and object geometry** — resolution distribution, bbox area
  percentiles, spatial distribution of boxes.
- **2.4 Split protocol** — 90/10 stratified train/val, seed 42, indices
  persisted (`split.json`) so all four models see identical data.

### 3. Methods

- **3.1 Evaluation protocol** — COCO mAP@[.50:.95] and mAP50 via
  `pycocotools`; score threshold 0.01; max 300 detections/image; per-class AP
  and head/mid/tail aggregation; identical scoring for every model.
- **3.2 Faster R-CNN R50-FPN** — two-stage, anchor-based; resize min 800 /
  max 1333; 41.3 M params; 8-epoch pilot budget.
- **3.3 YOLOv8s** — one-stage, anchor-free, CSPDarknet + PAN, decoupled
  heads; COCO-pretrained; imgsz 960; batch 8; 30 epochs.
- **3.4 RT-DETR-R18** — transformer, anchor-free; ResNet-18 backbone +
  hybrid encoder + decoder with query selection; imgsz 640; 30 epochs.
- **3.5 RT-DETR-L** — large variant; **report the memory failure** as a
  feasibility/deployment finding rather than hiding it.
- **3.6 Common training details** — seed 42, identical val split, framework
  versions, single-GPU hardware.

### 4. Results

- **4.1 Main results** — Table `tab:main`: params (M), mAP50-95, mAP50,
  head/mid/tail, latency (ms/img).
- **4.2 Per-class analysis** — best/worst classes; morphology-driven
  failure modes.
- **4.3 Head/mid/tail degradation** — quantify the tail drop per family.
- **4.4 Object-size sensitivity** — AP by COCO small/medium/large buckets.
- **4.5 Inference cost** — accuracy vs. latency.
- **4.6 Qualitative examples** — success/failure panels.

### 5. Discussion

- Why one-stage / transformer models lead (or do not) in this regime.
- Implication of the RT-DETR-L memory ceiling for real deployments.
- Which findings should transfer to other underwater detection datasets and
  which are FathomNet-specific.
- Broader impact: human-in-the-loop annotation assistance.

### 6. Limitations

- Unequal training budgets (8 vs. 30 epochs) and no systematic per-model
  hyperparameter tuning.
- Validation-split numbers only; the blind test set is still pending.
- Single split and single seed — no variance estimate.
- Single hardware platform for the latency measurements.

### 7. Conclusion

- Summary of the ranking; practical recommendation for AUV/ROV pipelines;
  future work (augmentation, tiling for small objects, longer schedules,
  multi-seed variance).

### Declarations

- Funding, ethics/consent, clinical-trial registration, conflict of interest,
  data availability, code availability.

### Appendix

- A.1 Full per-class AP table for every model that completed training.
- A.2 Training hyperparameters actually used (the reference for the
  limitations above).
- A.3 EDA figures — `fig:eda-geometry` and `fig:qualitative`, currently
  placeholders in `v1.tex`.
- A.4 Reproducibility and code availability.
---

## 3. Known Numbers (as of scaffold creation)

From the completed Faster R-CNN pilot (`03_fasterrcnn/output`):

| Metric | Value |
|--------|-------|
| Params | 41.29 M |
| mAP@[.50:.95] | 0.308 |
| mAP@0.50 | 0.4584 |
| Head / Mid / Tail AP50-95 | 0.5032 / 0.4090 / 0.2684 |
| Latency | 141.39 ms/img |

Faster R-CNN pilot protocol: seed 42, val split 0.1, 8 epochs,
min800/max1333, score\_thr 0.01, max\_det 300. 32 classes.
Source: `json/results_fasterrcnn.json`.

**Pending (running on Kaggle):** YOLOv8s (kernel v7), RT-DETR-R18 (v4).

---

## 4. Reference Plan

Bib keys live in `references.bib`. Keys marked `% VERIFY` need their
metadata double-checked before submission.

| Key | Used in | Purpose |
|-----|---------|---------|
| `fathomnet2022` | §2.1, §1 | FathomNet database |
| `fathomnet-clef-2026` (verify) | §2.1 | the competition track |
| `fathomnet-2021-workshop` (verify) | §1, §5 | earlier FathomNet training set |
| `coco2014` | §3.1 | COCO metrics/dataset |
| `pycocotools` | §3.1 | evaluation implementation |
| `fasterrcnn2015` | §3.2 | two-stage baseline |
| `fpn2017` | §3.2 | FPN neck |
| `maskrcnn2017` | §3.2 | R50-FPN lineage (torchvision impl.) |
| `yolo2016` | §1, §3.3 | one-stage lineage |
| `ssd2016` | §1 | one-stage lineage |
| `fcos2019` | §3.3 | anchor-free lineage |
| `retinanet2017` | §5 | dense-detection loss context |
| `yolov8_2023` | §3.3 | YOLOv8 model |
| `ultralytics` | §3.3, §3.6 | training framework |
| `detr2020` | §3.4 | transformer detector lineage |
| `deformable-detr2021` | §3.4 | convergence improvements |
| `rtdetr2024` | §3.4, §3.5 | RT-DETR |
| `underwater-survey` (verify) | §1 | underwater detection prior work |
| `paszke2019pytorch` | §3.6 | framework |
| `kaggle` | §3.6 | compute platform |

---

## 5. Build

```powershell
cd paper
.\sync_results.cmd   # copy notebook outputs into json/ and images/
.\build.cmd          # one-shot latexmk build -> v1.pdf
.\watch.cmd          # continuous preview (latexmk -pvc)
```

The `.todo{}` macro renders remaining stubs in red so nothing is missed
before submission.

## 6. Checklist Before Submission

- [ ] Run `.\sync_results.cmd` to pull `results_*.json` into `paper/json/`
      and notebook figures into `paper/images/`.
- [ ] Copy EDA figures (class distribution, image geometry, object size)
      into `paper/images/`.
- [ ] Fill `tab:main` (YOLOv8s, RT-DETR rows), `tab:class-counts` and
      `tab:perclass`.
- [ ] Replace all `\todo{...}` markers.
- [ ] Resolve all `% VERIFY` bib entries.
- [ ] Add corresponding-author email (and ORCID, if available) for
      Charan Rasineni in `v1.tex` (single-author paper; no co-authors).
- [ ] Confirm venue page limit and switch to the venue template if needed.
