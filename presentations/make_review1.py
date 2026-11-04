from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

IMG = r"C:\Users\rshit\OneDrive\Documents\Charan DL project\paper\images"
NAVY = RGBColor(0x0B, 0x2E, 0x59); BLUE = RGBColor(0x1F, 0x4E, 0x99)
LIGHT = RGBColor(0xEE, 0xF3, 0xFA); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22); ACCENT = RGBColor(0xBD, 0xD7, 0xEE)

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
text(s, Inches(0.9), Inches(3.4), Inches(11.5), Inches(0.5), "FathomNet CLEF 2026  |  Project Review 1", size=20, color=ACCENT, align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3), [
    [("Charan Rasineni", 18, WHITE, True)],
    [("B.Tech CSE  |  SCOPE, VIT-AP University, Amaravati", 14, ACCENT, False)],
    [("Mentor: Dr. Rajesh Duvvuru", 14, ACCENT, False)],
], align=PP_ALIGN.CENTER, space_after=6)

# ---------------- SLIDE 2: PROBLEM ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Problem Statement", size=28, color=WHITE, bold=True)
text(s, Inches(0.5), Inches(1.2), Inches(6.4), Inches(5.9), [
    [("Why automate marine biodiversity monitoring?", 18, NAVY, True)],
    [("ROVs/AUVs collect terabytes of underwater imagery, far exceeding human expert capacity for manual annotation and species identification.", 14, DARK, False)],
    [("Underwater detection is hard:", 14, NAVY, True)],
    [("Light attenuation & colour cast (blue-green cast, red wavelengths absorbed first)", 13, DARK, False)],
    [("Turbidity, backscatter, variable illumination, occlusion", 13, DARK, False)],
    [("Biological camouflage; many taxa are small and rare", 13, DARK, False)],
    [("Extreme class imbalance — few head classes dominate, long tail of rare species", 13, DARK, False)],
    [("Goal: build a controlled detector benchmark to guide AUV/ROV deployment.", 14, NAVY, True)],
], space_after=8)
pic(s, IMG + "\\fig_class_imbalance.png", Inches(7.2), Inches(1.5), w=5.5)
# ---------------- SLIDE 3: DATASET ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Dataset: FathomNet CLEF 2026", size=28, color=WHITE, bold=True)
text(s, Inches(0.5), Inches(1.2), Inches(6.2), Inches(2.8), [
    [("5,817 labelled training images", 15, DARK, True)],
    [("646 validation images", 15, DARK, True)],
    [("32 taxonomic classes", 15, DARK, True)],
    [("Strongly long-tailed annotation distribution", 14, DARK, False)],
    [("Head: bony fish, brittle star, urchin (~53% of annotations)", 13, DARK, False)],
    [("Tail: 25 classes incl. amphipod, bivalve, sea slug (some with only 6 instances)", 13, DARK, False)],
], space_after=5)
pic(s, IMG + "\\fig_geometry.png", Inches(7.0), Inches(1.5), w=5.6)
rect(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(2.6), LIGHT)
text(s, Inches(0.8), Inches(4.35), Inches(11.8), Inches(2.3), [
    [("Split protocol (fair comparison across all models):", 15, NAVY, True)],
    [("• Identical stratified 90/10 train/val split, seed 42", 13, DARK, False)],
    [("• Same input size: 640 x 640 px", 13, DARK, False)],
    [("• Same COCO mAP evaluation via pycocotools", 13, DARK, False)],
    [("• Confidence threshold 0.001, max 300 detections/image", 13, DARK, False)],
    [("• Hardware: single NVIDIA T4 GPU (15 GiB)", 13, DARK, False)],
], space_after=5)

# ---------------- SLIDE 4: OBJECTIVES ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Objectives & Contributions", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(4.4), [
    [("Controlled benchmark of three detector families on underwater imagery", 15, DARK, True)],
    [("1. Two-stage: Faster R-CNN R50-FPN", 14, DARK, False)],
    [("2. One-stage: YOLOv8s", 14, DARK, False)],
    [("3. Transformer: RT-DETR-L", 14, DARK, False)],
    [("Head/mid/tail and object-size stratified error analysis", 15, DARK, True)],
    [("Accuracy–inference-cost–latency trade-off for ROV/AUV deployment", 15, DARK, True)],
    [("Documented GPU-memory ceiling for RT-DETR-L on commodity hardware (15 GiB T4)", 15, DARK, True)],
], space_after=10)
rect(s, Inches(0.5), Inches(6.05), Inches(12.3), Inches(0.85), LIGHT)
text(s, Inches(0.8), Inches(6.2), Inches(11.8), Inches(0.6), [("Benchmark/comparison study — not a new architecture.", 13, NAVY, True)], align=PP_ALIGN.CENTER)

# ---------------- SLIDE 5: METHODS OVERVIEW ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Methodology Overview", size=28, color=WHITE, bold=True)
cols = [("Faster R-CNN R50-FPN", "Two-stage detector\nRegion Proposal Network + RoIAlign\nResNet-50 + FPN backbone", Inches(0.5)),
        ("YOLOv8s", "One-stage detector\nAnchor-free head, direct box + class prediction\nCSPDarknet backbone", Inches(4.6)),
        ("RT-DETR-L", "End-to-end transformer detector\nHybrid encoder + IoU-aware queries, no NMS", Inches(8.7))]
for title, desc, lx in cols:
    rect(s, lx, Inches(1.4), Inches(4.0), Inches(2.5), LIGHT)
    text(s, lx + Inches(0.2), Inches(1.55), Inches(3.6), Inches(0.5), title, size=16, color=NAVY, bold=True)
    text(s, lx + Inches(0.2), Inches(2.1), Inches(3.6), Inches(1.7), desc, size=11.5, color=DARK, space_after=3)
rect(s, Inches(0.5), Inches(4.3), Inches(12.3), Inches(2.4), LIGHT)
text(s, Inches(0.8), Inches(4.45), Inches(11.8), Inches(2.1), [
    [("Training (30 epochs per model, identical data split):", 14, NAVY, True)],
    [("• Faster R-CNN: batch 2, single precision", 13, DARK, False)],
    [("• YOLOv8s / RT-DETR-L: batch 4, mixed precision (AMP)", 13, DARK, False)],
    [("Framework-specific resizing, optimiser & augmentation rules remain; results compare representative default setups.", 12.5, RGBColor(0x55,0x55,0x55), False)],
], space_after=5)

# ---------------- SLIDE 6: PLAN ----------------
s = slide()
rect(s, 0, 0, prs.slide_width, Inches(0.9), NAVY)
text(s, Inches(0.5), Inches(0.1), Inches(12), Inches(0.7), "Review 1 — Plan & Next Steps", size=28, color=WHITE, bold=True)
bullets(s, Inches(0.5), Inches(1.3), Inches(12.4), Inches(4.5), [
    [("Complete dataset EDA and publish split indices (split.json)", 14, DARK, False)],
    [("Finish Faster R-CNN baseline training & evaluation", 14, DARK, False)],
    [("Launch YOLOv8s and RT-DETR-L training runs (Kaggle)", 14, DARK, False)],
    [("Build evaluation harness: mAP, per-class AP, latency, size-stratified AP", 14, DARK, False)],
    [("Review 2 target: training curves + early quantitative results", 14, DARK, False)],
], space_after=12)
rect(s, Inches(0.5), Inches(6.05), Inches(12.3), Inches(0.85), LIGHT)
text(s, Inches(0.8), Inches(6.2), Inches(11.8), Inches(0.6), [("Deliverable for Review 1: problem definition, dataset, protocol, and methodology.", 13, NAVY, True)], align=PP_ALIGN.CENTER)

prs.save("Review1_Project_Kickoff.pptx")
print("Saved Review1_Project_Kickoff.pptx")