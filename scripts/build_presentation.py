"""Build a 10-minute editable slide deck from measured test summaries."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HeraldFolder"
TEST = OUT / "test"
names = ("restormer", "dncnn", "nafnet")
scores = [json.loads((TEST/f"{n}_summary.json").read_text(encoding="utf8")) for n in names]
if any(s["number_of_test_images"] != 500 for s in scores):
    raise ValueError("Full test results required")
best = max(scores, key=lambda s: s["by_source"]["PolyU"]["psnr_db"])
navy = RGBColor(20, 52, 75)
teal = RGBColor(11, 145, 145)
dark = RGBColor(37, 56, 68)
gray = RGBColor(92, 110, 120)
pale = RGBColor(231, 244, 245)
white = RGBColor(255, 255, 255)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def textbox(slide, text, x, y, w, h, size=20, color=dark, bold=False, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.02)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for j, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.space_after = Pt(12 if size >= 18 else 6)
        if align:
            p.alignment = align
    return box

def slide(title, number, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = white
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(.14), Inches(7.5))
    band.fill.solid(); band.fill.fore_color.rgb = teal
    band.line.fill.background()
    textbox(s, title, .65, .35, 12, .72, 31, navy, True)
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.65), Inches(1.14), Inches(12), Inches(.035))
    rule.fill.solid(); rule.fill.fore_color.rgb = pale; rule.line.fill.background()
    if subtitle:
        textbox(s, subtitle, .68, 1.3, 12, .55, 17, gray)
    textbox(s, f"iMAGE  |  {number:02d}", .68, 7.08, 3, .25, 10, gray)
    return s

s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid(); s.background.fill.fore_color.rgb = navy
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.7), Inches(1.15), Inches(.16), Inches(4.45))
bar.fill.solid(); bar.fill.fore_color.rgb = teal; bar.line.fill.background()
textbox(s, "iMAGE", 1.1, 1.4, 9, 1, 46, white, True)
textbox(s, "Three pretrained models for real image denoising", 1.1, 2.65, 10.4, 1.3, 30, white)
textbox(s, "Restormer  ·  DnCNN  ·  NAFNet", 1.1, 4.25, 10, .55, 21, RGBColor(174, 233, 231))
textbox(s, "Course project  |  22 November 2026  |  Group details pending", 1.1, 6.55, 11, .4, 14, white)

s = slide("What we tested", 2)
textbox(s, "Question", .75, 1.65, 2.5, .5, 20, teal, True)
textbox(s, "Which published denoiser works best on paired real camera noise?", 3.3, 1.57, 8.6, 1.0, 25, navy)
textbox(s, "Choice", .75, 3.1, 2.5, .5, 20, teal, True)
textbox(s, "Three pretrained models, as permitted by the instructor's reported clarification. NAFNet approval was confirmed by the project owner.", 3.3, 3.02, 8.8, 1.5, 22)
textbox(s, "Evidence", .75, 5.05, 2.5, .5, 20, teal, True)
textbox(s, "One fixed manifest, identical test crops, and results saved with checkpoint hashes.", 3.3, 5.0, 8.8, 1.3, 22)

s = slide("Data and splits", 3, "SIDD and PolyU aligned noisy/clean RGB pairs")
headers = ["Source", "Train", "Validation", "Test", "Total"]
rows = [["SIDD", "2,800", "350", "350", "3,500"], ["PolyU", "1,200", "150", "150", "1,500"], ["Total", "4,000", "500", "500", "5,000"]]
xpos = [.9, 4.1, 6.2, 8.6, 10.8]
for x,h in zip(xpos, headers): textbox(s,h,x,2.1,2.2,.5,19,navy,True)
for i,row in enumerate(rows):
    y = 3.0 + i*.8
    for x,val in zip(xpos,row): textbox(s,val,x,y,2.2,.5,20,dark,i==2)
textbox(s, "Scene groups stay in one project split. The manifest has 5,000 paired rows.", .9, 6.25, 11.5, .6, 18, gray)

s = slide("The three methods", 4)
methods = [
    ("Restormer", "Transformer architecture\nPublished real-denoising weights\n26.11 million parameters"),
    ("DnCNN", "20-layer residual CNN\nPublished Gaussian blind weights\n0.67 million parameters"),
    ("NAFNet", "Activation-free network\nPublished SIDD width-32 weights\n29.16 million parameters"),
]
for x,(name,desc) in zip((.8,4.65,8.5), methods):
    textbox(s,name,x,2.0,3.5,.6,25,teal,True)
    textbox(s,desc,x,2.9,3.6,2.65,20,dark)
textbox(s, "Pretraining conditions differ; the benchmark compares deployed checkpoints, not matched training recipes.", .8, 6.2, 11.8, .62, 17, gray)

s = slide("Experiment protocol", 5)
items = [
    "1  Select checkpoints on validation data only",
    "2  Evaluate the same 500 test pairs for every model",
    "3  Use 64 x 64 center crops and RGB PSNR/SSIM",
    "4  Time forward inference on CPU; exclude loading",
]
for i,item in enumerate(items): textbox(s,item,.9,1.7+i*.88,11.5,.7,23,navy)
textbox(s, "The report separates SIDD and PolyU because SIDD overlaps published pretraining data.", .9, 5.85, 11.5, .8, 18, gray)

s = slide("Measured test results", 6, "Mean PSNR on 64 px RGB crops; all 500 pairs")
chart = TEST/"comparison_by_source.png"
if chart.is_file():
    s.shapes.add_picture(str(chart), Inches(.8), Inches(1.8), width=Inches(7.5))
x = 8.7
textbox(s, "PolyU PSNR", x, 2.05, 3.5, .5, 21, teal, True)
for i,sc in enumerate(scores):
    textbox(s, f'{sc["model"].capitalize()}: {sc["by_source"]["PolyU"]["psnr_db"]:.2f} dB',
            x, 2.72+i*.57, 3.8, .45, 19, dark)
textbox(s, "Highest PolyU PSNR", x, 4.65, 3.7, .45, 18, gray)
textbox(s, best["model"].upper(), x, 5.18, 3.8, .65, 27, navy, True)

s = slide("Quality and speed", 7)
heads = ["Model", "500-pair PSNR", "SSIM", "CPU ms/crop", "Params M"]
xx = [.85, 3.3, 6.0, 8.4, 10.7]
for x,h in zip(xx,heads): textbox(s,h,x,2,2.5,.55,18,navy,True)
for i,sc in enumerate(scores):
    vals = [sc["model"].capitalize(), f'{sc["psnr_db"]:.2f}', f'{sc["ssim"]:.4f}',
            f'{sc["mean_runtime_ms"]:.0f}', f'{sc["parameters_m"]:.2f}']
    for x,val in zip(xx,vals): textbox(s,val,x,2.9+i*.78,2.4,.52,20,dark)
textbox(s, "Published pretraining times are unavailable. CPU timing is specific to this machine and crop size.", .85, 6.2, 11.6, .7, 17, gray)

s = slide("Fine-tuning decision", 8)
textbox(s, "DnCNN pilot", .85, 1.7, 4, .6, 24, teal, True)
textbox(s, "200 paired training steps\nOne completed epoch\n2.40 minutes including validation", .85, 2.5, 5, 2.0, 21)
textbox(s, "50-crop validation PSNR", 6.8, 1.7, 5.4, .6, 23, teal, True)
textbox(s, "Published baseline   35.54 dB\nFine-tuned trial         32.26 dB", 6.8, 2.55, 5.2, 1.35, 22, navy)
textbox(s, "The final test uses the published DnCNN checkpoint. The second CPU epoch was stopped after a severe slowdown.", .85, 5.65, 11.6, 1.0, 19, gray)

s = slide("How to read the results", 9)
textbox(s, "SIDD", .9, 1.75, 2, .5, 24, teal, True)
textbox(s, "Restormer and NAFNet were pretrained on SIDD training images. This project's SIDD subset comes from that collection.", 3.1, 1.68, 9.0, 1.35, 21)
textbox(s, "PolyU", .9, 3.7, 2, .5, 24, teal, True)
textbox(s, "Held-out PolyU scenes provide the clearer cross-dataset comparison for the published checkpoints.", 3.1, 3.62, 9.0, 1.35, 21)
textbox(s, "The 500-pair aggregate is descriptive; report both sources separately.", .9, 5.95, 11.3, .7, 20, navy, True)

s = slide("App and next steps", 10)
textbox(s, "Upload app", .9, 1.7, 3, .55, 23, teal, True)
textbox(s, "Real mode runs the three networks. With a clean reference, the app compares that upload. Otherwise it recommends the model with the highest PolyU PSNR when benchmark hashes match.", 3.8, 1.65, 8.4, 2.0, 20)
textbox(s, "Submission", .9, 4.25, 3, .55, 23, teal, True)
textbox(s, "Executed notebook, three Python method modules, report, slides, dataset links, and contribution statement. Add group names and verify the notebook in Colab.", 3.8, 4.2, 8.4, 1.55, 20)
textbox(s, "AI use: code audit, data preparation, experiments, and artifact drafting; team review required.", .9, 6.4, 11.6, .45, 14, gray)

dest = OUT/"iMAGE_presentation.pptx"
prs.save(dest)
print(dest)
