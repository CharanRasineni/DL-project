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
text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.1), "Benchmarking Object Detectors on", size=34, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(0.8), "Under-Sampled Marine Biodiversity Imagery", size=34, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(3.4), Inches(11.5), Inches(0.5), "FathomNet CLEF 2026  |  Project Review 2", size=20, color=ACCENT, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3), [
    [("Charan Rasineni", 18, WHITE, True)],
    [("B.Tech CSE  |  SCOPE, VIT-AP University, Amaravati", 14, ACCENT, False)],
    [("Mentor: Dr. Rajesh Duvvuru", 14, ACCENT, False)],
], align=PP_ALIGN.CENTER, space_after=6)

# ---------------- SLIDE 2: RECAP ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Recap: Models & Protocol", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.2), Inches(6.4), Inches(5.8), [
    [("Three detector families:", 15, NAVY, True)],
    [("• Faster R-CNN R50-FPN — two-stage", 14, DARK, False)],
    [("• YOLOv8s — one-stage", 14, DARK, False)],
    [("• RT-DETR-L — transformer-based", 14, DARK, False)],
    [("Identical evaluation protocol:", 15, NAVY, True)],
    [("• Stratified 90/10 split, seed 42", 13, DARK, False)],
    [("• 640 x 640 px input, 30 epochs", 13, DARK, False)],
    [("• COCO mAP via pycocotools", 13, DARK, False)],
    [("• Conf thr 0.001, max det 300", 13, DARK, False)],
], space_after=7)
pic(s, IMG + "\\fig_performance_tradeoff.png", Inches(7.2), Inches(1.4), w=5.5)
# ---------------- SLIDE 3: TRAINING CURVES ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Training Curves — Validation mAP@[.50:.95]", size=26, color=WHITE, bold=True)
pic(s, IMG + "\\curve_fasterrcnn.png", Inches(0.5), Inches(1.3), w=3.9)
pic(s, IMG + "\\curve_yolov8s.png", Inches(4.7), Inches(1.3), w=3.9)
pic(s, IMG + "\\curve_rtdetr.png", Inches(8.9), Inches(1.3), w=3.9)
text(s, Inches(0.5), Inches(5.3), Inches(12.4), Inches(1.9), [
    [("Observations:", 14, NAVY, True)],
    [("• Faster R-CNN and YOLOv8s converge steadily over 30 epochs.", 13, DARK, False)],
    [("• RT-DETR-L shows a slower, noisier learning curve (transformer data-hunger).", 13, DARK, False)],
    [("• All models share the identical data split, isolating architectural differences.", 13, DARK, False)],
], space_after=5)

# ---------------- SLIDE 4: OVERALL RESULTS ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Overall Detection Performance", size=28, color=WHITE, bold=True)
rect(s, Inches(0.5), Inches(1.3), Inches(7.0), Inches(2.6), LIGHT)
text(s, Inches(0.8), Inches(1.45), Inches(6.5), Inches(2.3), [
    [("mAP@[.50:.95]  |  mAP@0.50  |  Latency", 13, NAVY, True)],
    [("Faster R-CNN:  0.333   |   0.483   |   46.4 ms", 14, DARK, False)],
    [("YOLOv8s:       0.313   |   0.428   |   24.0 ms", 14, DARK, False)],
    [("RT-DETR-L:     0.231   |   0.323   |   62.2 ms", 14, DARK, False)],
    [("Faster R-CNN is the accuracy leader.", 13, NAVY, True)],
], space_after=5)
pic(s, IMG + "\\fig_performance_tradeoff.png", Inches(7.8), Inches(1.3), w=5.0)

# ---------------- SLIDE 5: PER-CLASS ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Per-Class Analysis (Faster R-CNN)", size=27, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.2), Inches(6.2), Inches(5.8), [
    [("Easy classes (AP > 0.55): pyrosome 0.925, larvacean 0.718, hydroid 0.653, urchin 0.592, stony coral 0.586", 12.5, DARK, False)],
    [("Medium classes (0.2–0.4): anemone 0.309, sea fan 0.405, sponge 0.397, crab 0.517", 12.5, DARK, False)],
    [("Hard tail classes (~0.0): amphipod, bivalve, sea slug, sea squirt, barnacle", 12.5, DARK, False)],
    [("Same ordering holds for YOLOv8s & RT-DETR-L — difficulty driven by data distribution, not architecture.", 13, NAVY, True)],
], space_after=8)
rect(s, Inches(7.0), Inches(1.3), Inches(5.8), Inches(2.2), LIGHT)
text(s, Inches(7.3), Inches(1.45), Inches(5.3), Inches(2.0), [
    [("Head / Mid / Tail AP@[.50:.95]:", 13, NAVY, True)],
    [("FRCNN:  0.451  |  0.424  |  0.305", 12.5, DARK, False)],
    [("YOLOv8s: 0.436  |  0.408  |  0.283", 12.5, DARK, False)],
    [("RT-DETR-L: 0.429 |  0.345  |  0.189", 12.5, DARK, False)],
    [("Consistent head-to-tail degradation across all three.", 12.5, NAVY, True)],
], space_after=4)

# ---------------- SLIDE 6: SIZE ANALYSIS ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Object-Size Stratified AP@[.50:.95]", size=27, color=WHITE, bold=True)
rect(s, Inches(0.5), Inches(1.4), Inches(7.0), Inches(2.4), LIGHT)
text(s, Inches(0.8), Inches(1.55), Inches(6.5), Inches(2.1), [
    [("Detector   |  Small  |  Medium  |  Large", 13, NAVY, True)],
    [("Faster R-CNN  | 0.007   | 0.059    | 0.377", 13, DARK, False)],
    [("YOLOv8s       | 0.000   | 0.063    | 0.350", 13, DARK, False)],
    [("RT-DETR-L     | 0.000   | 0.075    | 0.254", 13, DARK, False)],
    [("Small objects effectively undetected at 640 px.", 12.5, NAVY, True)],
], space_after=4)
bullets(s, Inches(0.5), Inches(4.1), Inches(12.4), Inches(2.9), [
    [("Large objects: moderate accuracy (0.25–0.38)", 14, DARK, False)],
    [("Medium objects: poor (0.06–0.08)", 14, DARK, False)],
    [("Small objects: ~0.00–0.01 — tiny marine invertebrates occupy only a few pixels after resizing", 14, DARK, False)],
    [("Future: higher-resolution inputs, test-time tiling, small-object augmentation", 14, NAVY, True)],
], space_after=7)

# ---------------- SLIDE 7: DISCUSSION ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Discussion So Far", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(5.0), [
    [("Ranking: two-stage > one-stage > transformer (this config); read with config caveats (resizing, optimiser, checkpoint rules differ).", 14, DARK, False)],
    [("YOLOv8s: best accuracy–cost balance — 94% of FRCNN mAP at ~52% latency & ~1/4 parameters.", 14, DARK, False)],
    [("Faster R-CNN: accuracy ceiling; RT-DETR-L: neither most accurate nor fastest here — transformer baseline.", 14, DARK, False)],
    [("Head-to-tail degradation generalises to long-tailed underwater datasets (annotation frequency + object scale).", 14, DARK, False)],
    [("Small-object blind spot at 640 px transfers across architectures.", 14, DARK, False)],
], space_after=10)
rect(s, Inches(0.5), Inches(6.05), Inches(12.3), Inches(0.85), LIGHT)
text(s, Inches(0.8), Inches(6.2), Inches(11.8), Inches(0.6), [("Review 3 target: qualitative results, Grad-CAM, full discussion, limitations, conclusion.", 13, NAVY, True)], align=PP_ALIGN.CENTER)

prs.save("Review2_Training_and_Results.pptx")
print("Saved Review2_Training_and_Results.pptx")