"""Build the project report from verified model summaries."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HeraldFolder"
TEST = OUT / "test"
NAMES = ("restormer", "dncnn", "nafnet")
summaries = [json.loads((TEST / f"{n}_summary.json").read_text(encoding="utf8")) for n in NAMES]
if any(s["number_of_test_images"] != 500 or s["limited_test"] for s in summaries):
    raise ValueError("All three full 500-image summaries are required")
if len({s["manifest_sha256"] for s in summaries}) != 1:
    raise ValueError("Manifest hashes do not match")
if len({(s["patch"], s["tile"], s["device"]) for s in summaries}) != 1:
    raise ValueError("Evaluation settings do not match")

# The same chart is also used in the presentation.
fig, ax = plt.subplots(figsize=(7.3, 3.4))
x = range(3)
ax.bar([v - .2 for v in x], [s["by_source"]["SIDD"]["psnr_db"] for s in summaries],
       .4, label="SIDD (pretraining overlap)", color="#355C8A")
ax.bar([v + .2 for v in x], [s["by_source"]["PolyU"]["psnr_db"] for s in summaries],
       .4, label="PolyU", color="#10A7A0")
ax.set_xticks(list(x), ["Restormer", "DnCNN", "NAFNet"])
ax.set_ylabel("Mean PSNR (dB), 64 px crops")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
chart = TEST / "comparison_by_source.png"
fig.savefig(chart, dpi=180)
plt.close(fig)

best = max(summaries, key=lambda s: s["by_source"]["PolyU"]["psnr_db"])
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=20,
                          leading=23, spaceAfter=12, textColor=colors.HexColor("#163A56")))
styles.add(ParagraphStyle(name="ReportH1", parent=styles["Heading1"], fontSize=12,
                          leading=15, spaceBefore=12, spaceAfter=6,
                          textColor=colors.HexColor("#126E80")))
styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=9.5,
                          leading=13, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontSize=8,
                          leading=10, spaceAfter=5))
P = lambda text, style="ReportBody": Paragraph(text, styles[style])
story = [
    P("iMAGE: Comparison of Three Pretrained Image Denoisers", "ReportTitle"),
    P("Course project report | 22 November 2026 | Group number and member names: pending"),
    P("<b>Abstract.</b> We evaluated Restormer, DnCNN, and NAFNet on one fixed set of "
      "5,000 paired real-noise images from SIDD and PolyU. All three methods were tested "
      "on the same 500 64 x 64 RGB center crops. " +
      best["model"].upper() + " had the highest PolyU test PSNR in this run. "
      "SIDD scores for Restormer and NAFNet need caution because their published "
      "checkpoints were trained on SIDD training images."),
    P("1. Introduction", "ReportH1"),
    P("The project asks which pretrained denoiser best restores images affected by real "
      "camera noise. The team selected three pretrained methods under a verbal "
      "clarification reported by the project owner; NAFNet was also approved. "
      "The written assignment describes a different method mix, so the team should "
      "retain the instructor's clarification with the submission."),
    P("2. Dataset and split", "ReportH1"),
    P("The fixed manifest contains 3,500 SIDD and 1,500 PolyU noisy/clean pairs. "
      "Training has 4,000 pairs (2,800 SIDD, 1,200 PolyU), validation has 500 "
      "(350, 150), and test has 500 (350, 150). Scene groups do not cross the "
      "project's splits. Manifest SHA-256: <font size=7>" +
      summaries[0]["manifest_sha256"] + "</font>. Images are stored locally under data/raw; "
      "dataset links appear in the references."),
    P("3. Methods", "ReportH1"),
    P("<b>Restormer:</b> the authors' real-image denoising checkpoint and transformer "
      "architecture. <b>DnCNN:</b> KAIR's color blind Gaussian checkpoint and a "
      "20-layer residual convolutional network. <b>NAFNet:</b> the width-32 SIDD "
      "checkpoint and the authors' activation-free architecture. All models use "
      "published weights; the team's unsuccessful DnCNN fine-tuning trial was not "
      "selected for the final benchmark."),
    P("4. Experimental protocol", "ReportH1"),
    P("Model selection used validation only. A 200-step DnCNN trial on paired real "
      "noise lowered PSNR on the same 50 validation crops from 35.54 to 32.26 dB, "
      "so the published DnCNN checkpoint was retained. Test inference uses 64 x "
      "64 center crops, RGB PSNR and SSIM with data range 1, and batch size 1. "
      "Runtime measures model forward passes on " + summaries[0]["device"].upper() +
      "; it excludes checkpoint loading. Test metrics are means of per-image scores."),
    P("5. Results", "ReportH1"),
]
data = [["Model", "500 PSNR", "SSIM", "SIDD PSNR", "PolyU PSNR", "ms/crop", "Params M"]]
for s in summaries:
    data.append([s["model"].capitalize(), f'{s["psnr_db"]:.2f}', f'{s["ssim"]:.4f}',
                 f'{s["by_source"]["SIDD"]["psnr_db"]:.2f}',
                 f'{s["by_source"]["PolyU"]["psnr_db"]:.2f}',
                 f'{s["mean_runtime_ms"]:.0f}', f'{s["parameters_m"]:.2f}'])
table = Table(data, colWidths=[2.2*cm, 2.0*cm, 1.6*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.0*cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#163A56")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EAF2F5")]),
    ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#C8D4D9")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("TOPPADDING", (0,0), (-1,-1), 7),
]))
story.extend([table, Spacer(1, 8), Image(str(chart), width=15.7*cm, height=7.3*cm),
              P("Training times for the published checkpoints were not measured and are "
                "not directly comparable. The completed first DnCNN trial epoch took "
                "2.40 minutes for 200 training steps plus validation; a second epoch "
                "was stopped after severe CPU slowdown.", "ReportSmall"),
              P("6. Discussion and limits", "ReportH1"),
              P("PolyU is the clearer cross-dataset comparison for these published "
                "checkpoints. Restormer and NAFNet were trained on SIDD training data; "
                "the project's SIDD subset comes from that same collection. Thus the "
                "350 SIDD test pairs are separated within this project but may have "
                "been seen during pretraining. The 500-pair aggregate is descriptive "
                "and should not be called independent held-out generalization. "
                "DnCNN started from Gaussian-noise training, which also makes the "
                "pretraining conditions unequal. The evaluation uses crops rather "
                "than full-resolution images, and all runtime numbers are CPU-specific."),
              P("7. Upload application", "ReportH1"),
              P("The FastAPI app accepts a noisy upload and an optional clean reference. "
                "Real mode runs the three networks. Without a clean reference, its "
                "recommendation uses PolyU PSNR only when all three active checkpoints "
                "match full benchmark summaries. Preview mode is explicitly labeled "
                "as demonstration filters."),
              P("8. Contributions and AI use", "ReportH1"),
              P("Member names and individual contributions are pending from the group. "
                "An AI assistant helped audit the repository, prepare and verify the "
                "data workflow, load published checkpoints, run validation and test "
                "measurements, create code and submission artifacts, and document "
                "limitations. The team must review the measurements and wording "
                "before submission."),
              P("References", "ReportH1"),
              P("SIDD dataset: https://abdokamel.github.io/sidd/ <br/>"
                "PolyU real noisy images: https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset <br/>"
                "Restormer: https://github.com/swz30/Restormer <br/>"
                "Restormer real-denoising setup: https://github.com/swz30/Restormer/blob/main/Denoising/README.md <br/>"
                "DnCNN/KAIR: https://github.com/cszn/KAIR <br/>"
                "NAFNet: https://github.com/megvii-research/NAFNet <br/>"
                "NAFNet SIDD setup: https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md",
                "ReportSmall")])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667B87"))
    canvas.drawString(1.8*cm, 1.3*cm, "iMAGE course project | Group details pending")
    canvas.drawRightString(A4[0]-1.8*cm, 1.3*cm, str(doc.page))
    canvas.restoreState()

pdf = OUT / "iMAGE_report.pdf"
doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=1.8*cm,
                        leftMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.8*cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(pdf)
