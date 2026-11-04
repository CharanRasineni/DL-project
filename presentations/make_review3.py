from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

IMG = r"C:\Users\rshit\OneDrive\Documents\Charan DL project\paper\images"
NAVY = RGBColor(0x0B, 0x2E, 0x59); BLUE = RGBColor(0x1F, 0x4E, 0x99)
LIGHT = RGBColor(0xEE, 0xF3, 0xFA); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22); ACCENT = RGBColor(0xBD, 0xD7, 0xEE)
GRAY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide(): return prs.slides.add_slide(BLANK)
def rect(s, l, t, w, h, fill):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh
def text(s, l, t, w, h, content, size=14, color=DARK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=4):
    tb = s.shapes.add_textbox(l, t, w, h); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    if isinstance(content, str): content = [content]
    for i, item in enumerate(content):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after)
        if isinstance(item, str): item = [(item, size, color, bold)]
        elif isinstance(item, tuple): item = [item]
        for txt, sz, col, bd in item:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = bd; r.font.name = "Calibri"
    return tb
def bullets(s, l, t, w, h, items, size=14, color=DARK, space_after=6):
    tb = s.shapes.add_textbox(l, t, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        if isinstance(item, str): item = [("•  " + item, size, color, False)]
        for txt, sz, col, bd in item:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = bd; r.font.name = "Calibri"
    return tb
def pic(s, path, l, t, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, l, t, **kw)

# ---------------- SLIDE 1: TITLE ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, prs.slide_height, NAVY)
rect(s, 0, Inches(5.15), prs.slide_width, Inches(0.12), BLUE)
text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.1), "Benchmarking Object Detectors on", size=32, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(0.8), "Under-Sampled Marine Biodiversity Imagery", size=32, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(3.4), Inches(11.5), Inches(0.5), "FathomNet CLEF 2026  |  Project Review 3 (Final)", size=20, color=ACCENT, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3), [
    [("Charan Rasineni", 18, WHITE, True)],
    [("B.Tech CSE  |  SCOPE, VIT-AP University, Amaravati", 14, ACCENT, False)],
    [("Mentor: Dr. Rajesh Duvvuru", 14, ACCENT, False)],
], align=PP_ALIGN.CENTER, space_after=6)

# ---------------- SLIDE 2: EXEC SUMMARY ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Executive Summary", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(5.5), [
    [("Benchmark of Faster R-CNN R50-FPN, YOLOv8s, RT-DETR-L on FathomNet CLEF 2026 underwater imagery.", 14, DARK, False)],
    [("Faster R-CNN: best accuracy (mAP 0.333 / 0.483); YOLOv8s: best accuracy–latency trade-off (0.313 / 0.428 at 24 ms).", 14, DARK, False)],
    [("RT-DETR-L: lowest accuracy (0.231 / 0.323) and slowest (62 ms) in this config.", 14, DARK, False)],
    [("All detectors show strong head-to-tail degradation; small objects undetected at 640 px.", 14, DARK, False)],
    [("Recommendation: YOLOv8s for AUV/ROV deployment; Faster R-CNN when latency is secondary.", 14, NAVY, True)],
], space_after=10)

# ---------------- SLIDE 3: DATASET & PROTOCOL ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Dataset & Protocol", size=28, color=WHITE, bold=True)
text(s, Inches(0.5), Inches(1.2), Inches(6.2), Inches(3.0), [
    [("5,817 train / 646 val images, 32 classes", 14, DARK, True)],
    [("Long-tailed: top-3 classes ~53% of annotations; rarest has 6 instances", 13, DARK, False)],
    [("Head: bony fish, brittle star, urchin | Mid: anemone, sea fan, sponge, stony coral | Tail: 25 classes", 13, DARK, False)],
], space_after=5)
bullets(s, Inches(0.5), Inches(4.0), Inches(6.2), Inches(2.9), [
    [("Stratified 90/10 split, seed 42", 13, DARK, False)],
    [("640 x 640 px input, 30 epochs", 13, DARK, False)],
    [("COCO mAP, thr 0.001, max det 300", 13, DARK, False)],
    [("NVIDIA T4 GPU, 15 GiB", 13, DARK, False)],
], space_after=6)
pic(s, IMG + "\\fig_class_imbalance.png", Inches(7.0), Inches(1.4), w=5.7)
# ---------------- SLIDE 4: FULL RESULTS TABLE ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Full Results — mAP & Latency", size=27, color=WHITE, bold=True)
rect(s, Inches(0.5), Inches(1.3), Inches(9.0), Inches(2.6), LIGHT)
text(s, Inches(0.8), Inches(1.45), Inches(8.5), Inches(2.3), [
    [("Detector        mAP[.50:.95]  mAP@0.50  mAP@0.75  Latency", 13, NAVY, True)],
    [("Faster R-CNN    0.333         0.483     0.354       46.4 ms", 13, DARK, False)],
    [("YOLOv8s         0.313         0.428     0.356       24.0 ms", 13, DARK, False)],
    [("RT-DETR-L       0.231         0.323     0.252       62.2 ms", 13, DARK, False)],
    [("YOLOv8s = 94% of FRCNN mAP at ~52% of latency & ~1/4 parameters.", 12.5, NAVY, True)],
], space_after=4)
bullets(s, Inches(0.5), Inches(4.2), Inches(12.4), Inches(2.8), [
    [("Head / Mid / Tail AP@[.50:.95] — FRCNN: 0.451 / 0.424 / 0.305; YOLOv8s: 0.436 / 0.408 / 0.283; RT-DETR-L: 0.429 / 0.345 / 0.189", 13, DARK, False)],
    [("Size AP@[.50:.95] (S / M / L) — FRCNN: 0.007 / 0.059 / 0.377; YOLOv8s: 0.000 / 0.063 / 0.350; RT-DETR-L: 0.000 / 0.075 / 0.254", 13, DARK, False)],
], space_after=6)

# ---------------- SLIDE 5: TRAINING CURVES ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Validation mAP@[.50:.95] Training Curves", size=26, color=WHITE, bold=True)
pic(s, IMG + "\\curve_fasterrcnn.png", Inches(0.4), Inches(1.3), w=4.0)
pic(s, IMG + "\\curve_yolov8s.png", Inches(4.7), Inches(1.3), w=4.0)
pic(s, IMG + "\\curve_rtdetr.png", Inches(9.0), Inches(1.3), w=4.0)
text(s, Inches(0.5), Inches(5.5), Inches(12.4), Inches(1.7), [
    [("FRCNN & YOLOv8s converge steadily; RT-DETR-L slower & noisier (transformer data-hunger / convergence challenge).", 13, DARK, False)],
], space_after=3)

# ---------------- SLIDE 6: PER-CLASS ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Per-Class AP@[.50:.95] (Faster R-CNN)", size=26, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.2), Inches(6.4), Inches(5.8), [
    [("Easy (>0.55): pyrosome 0.925, larvacean 0.718, hydroid 0.653, urchin 0.592, stony coral 0.586", 12, DARK, False)],
    [("Medium (0.2–0.4): anemone, sea fan, sponge, crab, octopus", 12, DARK, False)],
    [("Hard (~0.0): amphipod, bivalve, sea slug, sea squirt, barnacle", 12, DARK, False)],
    [("Ordering consistent across all three detectors — driven by annotation count & object size.", 12.5, NAVY, True)],
], space_after=7)
rect(s, Inches(7.2), Inches(1.3), Inches(5.6), Inches(2.2), LIGHT)
text(s, Inches(7.5), Inches(1.45), Inches(5.1), Inches(2.0), [
    [("Head / Mid / Tail AP@[.50:.95]:", 12.5, NAVY, True)],
    [("FRCNN:   0.451 / 0.424 / 0.305", 12, DARK, False)],
    [("YOLOv8s: 0.436 / 0.408 / 0.283", 12, DARK, False)],
    [("RT-DETR-L: 0.429 / 0.345 / 0.189", 12, DARK, False)],
], space_after=4)

# ---------------- SLIDE 7: INFERENCE COST ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Inference Cost — NVIDIA T4 (batch 1, FP32, 640 px)", size=26, color=WHITE, bold=True)
rect(s, Inches(0.5), Inches(1.4), Inches(7.0), Inches(2.5), LIGHT)
text(s, Inches(0.8), Inches(1.55), Inches(6.5), Inches(2.2), [
    [("YOLOv8s:  24.0 ms/img  (fastest)", 14, DARK, True)],
    [("FRCNN:    46.4 ms/img", 14, DARK, True)],
    [("RT-DETR-L: 62.2 ms/img  (slowest)", 14, DARK, True)],
    [("Memory: Ultralytics (YOLOv8s/RT-DETR-L) AMP batch 4; FRCNN FP32 batch 2.", 12, GRAY, False)],
], space_after=5)
bullets(s, Inches(8.0), Inches(1.4), Inches(4.9), Inches(5.4), [
    [("Deployment trade-off:", 14, NAVY, True)],
    [("• YOLOv8s: best accuracy–cost fit for embedded AUV/ROV", 13, DARK, False)],
    [("• FRCNN: accuracy ceiling when latency is secondary", 13, DARK, False)],
    [("• RT-DETR-L: high memory, not recommended from these runs", 13, DARK, False)],
], space_after=8)
# ---------------- SLIDE 8: QUALITATIVE ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Qualitative Detection Examples", size=28, color=WHITE, bold=True)
pic(s, IMG + "\\fig_qualitative_large.png", Inches(0.5), Inches(1.3), w=12.3)
text(s, Inches(0.5), Inches(6.55), Inches(12.4), Inches(0.7), [("Matched detections across Faster R-CNN, YOLOv8s, RT-DETR-L (GT overlaid, score thr 0.25). Easy classes detected consistently; rare/small/camouflaged taxa missed.", 11.5, GRAY, False)], align=PP_ALIGN.CENTER)

# ---------------- SLIDE 9: GRADCAM ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Grad-CAM Visual Attributions", size=28, color=WHITE, bold=True)
pic(s, IMG + "\\fig_gradcam_large.png", Inches(0.5), Inches(1.3), w=12.3)
text(s, Inches(0.5), Inches(6.55), Inches(12.4), Inches(0.7), [("Feature-activation comparison across deep layers — detectors attend to salient foreground on easy classes; diffuse/absent on hard tail classes.", 11.5, GRAY, False)], align=PP_ALIGN.CENTER)

# ---------------- SLIDE 10: DISCUSSION ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Discussion", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(5.5), [
    [("Ranking must be read with config caveats — results compare representative default setups, not strictly equal compute.", 14, DARK, False)],
    [("Head-to-tail degradation is inherited from the data distribution and transfers across long-tailed underwater datasets.", 14, DARK, False)],
    [("Small-object blind spot at 640 px transfers across architectures.", 14, DARK, False)],
    [("YOLOv8s sits at the accuracy–cost sweet spot; FRCNN is the accuracy ceiling; RT-DETR-L is the transformer baseline.", 14, DARK, False)],
], size=14, space_after=10)

# ---------------- SLIDE 11: LIMITATIONS ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Limitations", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(5.5), [
    [("Single split & seed; validation-only evaluation (no blind test set)", 14, DARK, False)],
    [("Nominally equal, not identical, compute across frameworks", 14, DARK, False)],
    [("Single hardware (NVIDIA T4); modest 30-epoch budget", 14, DARK, False)],
    [("No long-tailed algorithmic intervention (standard cross-entropy only)", 14, DARK, False)],
], size=14, space_after=12)

# ---------------- SLIDE 12: CONCLUSION ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Conclusion & Future Work", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(3.6), [
    [("FRCNN best accuracy; YOLOv8s best accuracy–latency trade-off; RT-DETR-L trails in this config.", 14, DARK, False)],
    [("Deploy YOLOv8s on AUV/ROV; use FRCNN when latency is secondary.", 14, DARK, False)],
], space_after=10)
rect(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(2.4), LIGHT)
text(s, Inches(0.8), Inches(4.55), Inches(11.8), Inches(2.1), [
    [("Future work:", 14, NAVY, True)],
    [("• Longer schedules, multi-seed evaluation, blind test-set submission", 13, DARK, False)],
    [("• Test-time tiling / higher resolution for small objects", 13, DARK, False)],
    [("• Long-tailed losses & resampling; model compression (quantisation/distillation)", 13, DARK, False)],
], space_after=5)

# ---------------- SLIDE 13: REFERENCES ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "References", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(5.5), [
    [("FathomNet (Scientific Reports, 2022); FathomNet CLEF 2026 (ImageCLEF LifeCLEF track)", 13, DARK, False)],
    [("Faster R-CNN (NeurIPS 2015); FPN (CVPR 2017); Mask R-CNN (ICCV 2017)", 13, DARK, False)],
    [("YOLOv8 / Ultralytics (2023)", 13, DARK, False)],
    [("DETR (ECCV 2020); Deformable DETR (ICLR 2021); RT-DETR (CVPR 2024)", 13, DARK, False)],
    [("PyTorch (NeurIPS 2019); pycocotools; Kaggle", 13, DARK, False)],
], size=13, space_after=8)

prs.save("Review3_Final_Report.pptx")
print("Saved Review3_Final_Report.pptx")