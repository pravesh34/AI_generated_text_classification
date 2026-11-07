"""
Generate MoE figures + appendix PDF, then merge into the main project report.

Outputs:
  results/figures/moe_accuracy_vs_length.png
  results/figures/moe_cross_domain.png
  results/figures/moe_experts.png
  results/figures/moe_type_gate.png
  Project_Report_MoE_Appendix.pdf  (standalone appendix)
  Project_Report_AI_Text_Classification.pdf  (original + appendix merged)
  report.pdf (copy of merged)

Run: PYTHONPATH=. python3 moe/make_moe_appendix.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
NAVY = "#1F2A44"; ACCENT = "#4C72B0"; ORANGE = "#DD8452"; GREEN = "#55A868"

base = pd.read_csv(os.path.join(ROOT, "results", "moe_baseline_comparison.csv"))
moe = pd.read_csv(os.path.join(ROOT, "results", "moe_routed_metrics.csv"))
exp = pd.read_csv(os.path.join(ROOT, "results", "moe_expert_metrics.csv"))

# ---------- Figure 1: Accuracy vs length ----------
order = ["first_25_words", "first_50_words", "first_100_words", "first_150_words", "first_200_words", "full"]
labels = {"first_25_words": "25w", "first_50_words": "50w", "first_100_words": "100w",
          "first_150_words": "150w", "first_200_words": "200w", "full": "full"}
b = base.set_index("test_condition").loc[order, "accuracy"].values
m = moe.set_index("test_condition").loc[order, "accuracy"].values
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(x, b, marker="o", lw=2.2, color=ORANGE, label="Baseline LR (single model)")
ax.plot(x, m, marker="s", lw=2.2, color=ACCENT, label="MoE routed (3 experts, th=0.50)")
for i, (bv, mv) in enumerate(zip(b, m)):
    if order[i] != "full":
        ax.annotate(f"{bv*100:.1f}%", (i, bv), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=ORANGE)
        ax.annotate(f"{mv*100:.1f}%", (i, mv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=ACCENT)
ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order])
ax.set_ylabel("Accuracy"); ax.set_ylim(0.55, 1.01)
ax.set_title("Accuracy vs Text Length — Baseline vs MoE (5k held-out, truncations)", fontweight="bold", color=NAVY)
ax.legend(loc="lower right", fontsize=9); ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "moe_accuracy_vs_length.png"), bbox_inches="tight"); plt.close()

# ---------- Figure 2: Cross-domain ----------
conds2 = ["full", "hc3_holdout", "modern_holdout"]
names2 = ["Original essays\n(full)", "HC3 chat\n(unseen)", "Modern LLM essays\n(unseen)"]
b2 = base.set_index("test_condition").loc[conds2, "accuracy"].values
m2 = moe.set_index("test_condition").loc[conds2, "accuracy"].values
fig, ax = plt.subplots(figsize=(9, 4.2))
xx = np.arange(len(conds2)); w = 0.38
ax.bar(xx - w / 2, b2 * 100, w, label="Baseline", color=ORANGE, edgecolor="black", alpha=0.9)
ax.bar(xx + w / 2, m2 * 100, w, label="MoE routed", color=ACCENT, edgecolor="black", alpha=0.9)
for xi, v in zip(xx - w / 2, b2):
    ax.text(xi, v * 100 + 0.7, f"{v*100:.1f}%", ha="center", fontsize=8)
for xi, v in zip(xx + w / 2, m2):
    ax.text(xi, v * 100 + 0.7, f"{v*100:.1f}%", ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(names2)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
ax.set_title("Cross-domain — baseline collapses on new text types; MoE holds up", fontweight="bold", color=NAVY)
ax.legend(loc="lower left", fontsize=9); ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "moe_cross_domain.png"), bbox_inches="tight"); plt.close()

# ---------- Figure 3: Per-expert metrics ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, col in zip(axes, ["accuracy", "f1", "roc_auc"]):
    ax.bar(exp["expert"], exp[col], color=[ACCENT, ORANGE, GREEN], edgecolor="black")
    ax.set_title(col.upper()); ax.set_ylim(0.85, 1.0)
    for i, v in enumerate(exp[col]):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
plt.suptitle("MoE per-expert validation (held-out 20% per bucket)", fontweight="bold", color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "moe_experts.png"), bbox_inches="tight"); plt.close()

# ---------- Figure 4: Type gate ----------
types = ["casual", "chat", "essay"]
prec = [0.77, 0.74, 0.90]; rec = [0.66, 0.84, 0.92]; f1s = [0.71, 0.79, 0.91]
fig, ax = plt.subplots(figsize=(9, 4.2))
xx = np.arange(len(types)); w = 0.26
ax.bar(xx - w, prec, w, label="Precision", color=ACCENT, edgecolor="black")
ax.bar(xx, rec, w, label="Recall", color=ORANGE, edgecolor="black")
ax.bar(xx + w, f1s, w, label="F1", color=GREEN, edgecolor="black")
ax.set_xticks(xx); ax.set_xticklabels(types); ax.set_ylim(0, 1.05)
ax.set_title("Type gate (chat / casual / essay) — val acc 80.5%, n=1,500", fontweight="bold", color=NAVY)
ax.legend(fontsize=9); ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "moe_type_gate.png"), bbox_inches="tight"); plt.close()

# ---------- Figure 5: Precision-recall flip at 25w ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
cats = ["Precision\n(human correctness)", "Recall\n(AI detection)"]
base_pr = [0.5333, 0.9865]; moe_pr = [0.9259, 0.6792]
xx = np.arange(2); w = 0.36
ax.bar(xx - w / 2, base_pr, w, label="Baseline @25w", color=ORANGE, edgecolor="black")
ax.bar(xx + w / 2, moe_pr, w, label="MoE @25w", color=ACCENT, edgecolor="black")
for xi, v in zip(xx - w / 2, base_pr):
    ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(xx + w / 2, moe_pr):
    ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(xx); ax.set_xticklabels(cats); ax.set_ylim(0, 1.08)
ax.set_title("25-word error flip — baseline over-calls AI; MoE under-calls AI (but wins 86% vs 67%)", fontweight="bold", color=NAVY, fontsize=10)
ax.legend(fontsize=9); ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "moe_pr_flip.png"), bbox_inches="tight"); plt.close()

print("Figures saved to results/figures/moe_*.png")

# =================== Appendix PDF ===================
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)
from reportlab.lib import colors

NAVY_HEX = HexColor("#1F2A44"); ACCENT_HEX = HexColor("#4C72B0"); LIGHT = HexColor("#EAF0F8")
styles = getSampleStyleSheet()
sTitle = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, leading=26, textColor=NAVY_HEX, alignment=TA_CENTER)
sH1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, leading=18, textColor=NAVY_HEX, spaceBefore=14, spaceAfter=6)
sH2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=ACCENT_HEX, spaceBefore=10, spaceAfter=4)
sBody = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6)
sBullet = ParagraphStyle("Bullet", parent=sBody, leftIndent=14, bulletIndent=6, spaceAfter=3)
sCell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_CENTER)
sCellL = ParagraphStyle("CellL", parent=sCell, alignment=0)
sCap = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.gray)

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.gray)
    canvas.drawCentredString(A4[0]/2, 12*mm, "NIT Sikkim — CSE Course Project (2026) — Human vs AI Text Classification — MoE Appendix")
    canvas.drawRightString(A4[0]-15*mm, 12*mm, f"Page {doc.page}")
    canvas.setStrokeColor(ACCENT_HEX); canvas.setLineWidth(0.5)
    canvas.line(15*mm, A4[1]-15*mm, A4[0]-15*mm, A4[1]-15*mm)
    canvas.restoreState()

OUT_APP = os.path.join(ROOT, "Project_Report_MoE_Appendix.pdf")
doc = SimpleDocTemplate(os.path.abspath(OUT_APP), pagesize=A4,
                        leftMargin=15*mm, rightMargin=15*mm, topMargin=20*mm, bottomMargin=16*mm,
                        title="MoE Appendix — Short / Long / VeryLong + Chat / Casual / Essay",
                        author="Pravesh Gedon Sharma & Ankit Rawat")
story = []
story.append(PageBreak())  # divider: appendix starts on fresh page after merge
story.append(Spacer(1, 20*mm))
story.append(Paragraph("Appendix — Mixture-of-Experts Upgrade<br/>Short / Long / VeryLong + Chat / Casual / Essay", sTitle))
story.append(Spacer(1, 4*mm))
story.append(Paragraph("This appendix is merged into the main report. It adds small-text datasets, length-specialized experts, a text-type gate, and threshold calibration.",
                       ParagraphStyle("sub", parent=sBody, alignment=TA_CENTER, fontSize=10)))
story.append(HRFlowable(width="60%", thickness=1.2, color=ACCENT_HEX, spaceAfter=8, spaceBefore=8, hAlign="CENTER"))
story.append(Paragraph("Code: <font face=\"Courier\" size=\"7\">moe/download_small_text.py, moe/train_moe.py, moe/predict_moe.py, moe/calibrate_thresholds.py</font><br/>Artifacts: <font face=\"Courier\" size=\"7\">models/moe/expert_{short,long,verylong}.pkl, type_gate.pkl, thresholds.json</font>",
                       ParagraphStyle("cov", parent=sBody, alignment=TA_CENTER, fontSize=8, textColor=colors.gray)))
story.append(PageBreak())

def H1(t): story.append(Paragraph(t, sH1))
def H2(t): story.append(Paragraph(t, sH2))
def P(t): story.append(Paragraph(t, sBody))
def B(items):
    for it in items: story.append(Paragraph(it, sBullet, bulletText="•"))
def FIG(path, w=160*mm):
    story.append(Image(os.path.abspath(os.path.join(ROOT, path)), width=w, height=w*0.45, kind="proportional"))
    story.append(Spacer(1, 2*mm))
def CAPT(t): story.append(Paragraph(t, sCap))
def TBL(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),LIGHT),("BOX",(0,0),(-1,-1),0.5,ACCENT_HEX),
                           ("INNERGRID",(0,0),(-1,-1),0.4,colors.gray),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                           ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]))
    story.append(t)

H1("A &nbsp; Motivation — Short-Text Failure")
P("The Phase-1 baseline scores 99%+ on full essays but collapses on short text: first 25 words → 67.4% (LR), first 50 → 83.3%, 1 sentence → 59.8%. Errors are one-sided: AI recall stays ~0.98 while human false-positives explode (~51% of humans called AI at 25w). This is expected — training never saw short text (train.csv median 363w, 5th percentile 180w) and TF-IDF + stylometrics need length.")
FIG("results/figures/moe_accuracy_vs_length.png", 165*mm)
CAPT("Figure A1: Accuracy vs truncated length (5k held-out). MoE lifts 25w 67%→86% and 1-sentence 60%→82%.")
FIG("results/figures/moe_pr_flip.png", 125*mm)
CAPT("Figure A2: Error flip at 25w. Baseline over-calls AI (P=0.53, R=0.99); MoE under-calls AI (P=0.93, R=0.68) — net acc still wins.")
P("The baseline also fails cross-domain: 48.6% on unseen HC3 chat answers (worse than chance) and 85.5% on modern GPT-4/Claude essays. It learned old-essay quirks, not authorship.")

H1("B &nbsp; New Datasets (Small Text + Modern Essays)")
P("Downloaded via <font face=\"Courier\" size=\"8\">moe/download_small_text.py</font> (JSONL/parquet directly from Hugging Face Hub):")
rows = [[Paragraph("Source",sCell), Paragraph("n",sCell), Paragraph("Median",sCell), Paragraph("Role",sCell)],
        [Paragraph("HC3 reddit_eli5+open_qa+finance<br/>(Hello-SimpleAI/HC3, Guo et al.)",sCellL), Paragraph("80,811<br/>(56k human / 25k AI)",sCell), Paragraph("117w,<br/>61% ≤150w",sCell), Paragraph("SHORT chat/casual",sCell)],
        [Paragraph("artnitolog/llm-generated-texts<br/>(GPT-4 Turbo/Omni, Claude 3 Opus, Llama3 70B, Command R+)",sCellL), Paragraph("17,997<br/>(3k human / 15k AI)",sCell), Paragraph("510w,<br/>68% &gt;400w",sCell), Paragraph("LONG/VERYLONG modern",sCell)],
        [Paragraph("Existing train.csv 20k balanced<br/>+ truncation augmentation",sCellL), Paragraph("20,000 essays<br/>+ 20k trunc copies",sCell), Paragraph("393w avg; trunc<br/>50/100/150/200/300w",sCell), Paragraph("LONG/VERYLONG +<br/>short-essay style",sCell)]]
TBL(rows, [62*mm, 32*mm, 28*mm, 38*mm]); CAPT("Table A1: New data. HC3 gives genuine short Q&A (human vs ChatGPT); modern essays give 2024-LLM text.")
P("Truncation augmentation: each train.csv essay contributes a prefix or random 50/100/150w copy (label preserved) so the short expert learns short-essay style, not just chat. Pool caps are balanced per class (seed 42): short 20k, long ~16.9k, verylong ~11.4k.")

H1("C &nbsp; MoE Architecture")
P("<b>Length gate (deterministic):</b> word count ≤150 → <b>short</b>, 151-400 → <b>long</b>, &gt;400 → <b>verylong</b> (150 = observed accuracy cliff, 400 ≈ dataset median). <b>Each expert</b> has its own Word TF-IDF (1,2) 15k + Char TF-IDF (char_wb 3,5) 25k + LogisticRegression(C=1.5, lbfgs) — same hyper-parameters as the baseline, but features specialized per length regime.")
P("<b>Type gate (learned):</b> LogisticRegression over 16 fast stylometric features (log words, sentence stats, TTR, hapax, contraction/informal/first/second-person rates, punctuation rates, FRE/FKGL, burstiness) → <b>chat / casual / essay</b>. Labels from source: HC3 <font face=\"Courier\" size=\"8\">open_qa</font> → chat, HC3 eli5/finance + truncated essays → casual, full/modern essays → essay. Trained on 7,500 balanced rows (2,500/type); validation acc 80.5%.")
FIG("results/figures/moe_type_gate.png", 130*mm)
CAPT("Figure A3: Type gate per-class scores. Essay easiest (0.91 F1); casual/chat confuse (casual recall 0.66), as expected for informal text.")
P("Inference (<font face=\"Courier\" size=\"8\">moe/predict_moe.py</font>) routes by length, reports label + P(AI) + expert + word count + type (+confidence) + reliability warning (&lt;25w / ≤50w).")

H1("D &nbsp; Per-Expert Results")
rows = [[Paragraph("Expert",sCell), Paragraph("Train / Val",sCell), Paragraph("Acc",sCell), Paragraph("F1",sCell), Paragraph("AUC",sCell)]]
for _, r in exp.iterrows():
    rows.append([Paragraph(str(r["expert"]),sCell), Paragraph(f"{int(r['n_train']):,} / {int(r['n_val']):,}",sCell),
                 Paragraph(f"{r['accuracy']:.4f}",sCell), Paragraph(f"{r['f1']:.4f}",sCell), Paragraph(f"{r['roc_auc']:.4f}",sCell)])
TBL(rows, [35*mm, 35*mm, 25*mm, 25*mm, 30*mm]); CAPT("Table A2: Per-bucket validation (results/moe_expert_metrics.csv).")
FIG("results/figures/moe_experts.png", 165*mm); CAPT("Figure A4: Per-expert accuracy/F1/AUC (all ≥0.97).")

H1("E &nbsp; Routed MoE vs Baseline (5k held-out + new domains)")
P("Routing = length → expert. Full-text MoE costs 0.3pp vs baseline (98.96% vs 99.26%) — the price of specialization. Everywhere else it wins, and on unseen domains the gap is decisive (HC3: 48.6%→98.5%, modern: 85.5%→95.7%).")
FIG("results/figures/moe_cross_domain.png", 150*mm)
CAPT("Figure A5: Cross-domain held-outs. Baseline is worse than chance on HC3 chat.")
rows = [[Paragraph("Condition",sCell), Paragraph("Baseline acc",sCell), Paragraph("MoE acc",sCell), Paragraph("Delta",sCell)]]
order_pairs = [("full","full"), ("first_25_words","first 25w"), ("first_50_words","first 50w"),
               ("first_1_sent","first 1 sent"), ("hc3_holdout","HC3 holdout"), ("modern_holdout","modern holdout")]
bidx = base.set_index("test_condition"); midx = moe.set_index("test_condition")
for key, lab in order_pairs:
    bv = float(bidx.loc[key, "accuracy"]); mv = float(midx.loc[key, "accuracy"])
    rows.append([Paragraph(lab,sCell), Paragraph(f"{bv*100:.2f}%",sCell), Paragraph(f"{mv*100:.2f}%",sCell),
                 Paragraph(f"+{(mv-bv)*100:.1f}pp",sCell)])
TBL(rows, [45*mm, 30*mm, 30*mm, 25*mm]); CAPT("Table A3: Headline comparison (results/moe_routed_metrics.csv + moe_baseline_comparison.csv).")
P("Per-bucket on full text: verylong 99.59% (n=1,958), long 98.81% (n=2,945), short 90.72% (n=97 naturally-short essays).")

H1("F &nbsp; Threshold Calibration")
P("The short expert under-calls AI on ultra-short essays (recall 0.68 at 25w). Calibrating on a calib half (2,500, seed 999, max F1) gave per-length thresholds 25w→0.32, 50w→0.40, 100w→0.56, 150w→0.64, implemented as <font face=\"Courier\" size=\"8\">th=0.28+0.0024·n</font> clamped [0.32,0.64] for the short bucket (<font face=\"Courier\" size=\"8\">models/moe/thresholds.json</font>). Long/verylong stayed at 0.50.")
rows = [[Paragraph("Eval-2500 condition",sCell), Paragraph("th=0.50",sCell), Paragraph("Adaptive",sCell), Paragraph("Note",sCell)],
        [Paragraph("first 25w",sCell), Paragraph("86.16% / F1 0.785",sCell), Paragraph("88.08% / F1 0.837",sCell), Paragraph("Recall 0.68→0.82",sCell)],
        [Paragraph("first 50w",sCell), Paragraph("93.12% / 0.905",sCell), Paragraph("93.08% / 0.909",sCell), Paragraph("Neutral (R↑ P↓)",sCell)],
        [Paragraph("first 100w",sCell), Paragraph("94.52% / 0.928",sCell), Paragraph("94.52% / 0.928",sCell), Paragraph("th≈0.52, identical",sCell)],
        [Paragraph("first 150w",sCell), Paragraph("94.96% / 0.935",sCell), Paragraph("96.20% / 0.949",sCell), Paragraph("+1.2pp",sCell)],
        [Paragraph("full",sCell), Paragraph("98.96%",sCell), Paragraph("98.92%",sCell), Paragraph("Flat",sCell)]]
TBL(rows, [40*mm, 40*mm, 40*mm, 40*mm]); CAPT("Table A4: Calibration on unseen eval half (results/moe_calibrated_metrics.csv). Caveat: HC3 natural short −0.6pp (human-heavy); essays gain. The &lt;50w warning is retained.")
P("<b>Demo flips:</b> a 31w borderline excerpt (P(AI)=0.374, th=0.354) is now AI-Generated (was Human under 0.50) — the intended fix, still low-confidence and flagged. A 14w casual human (P(AI)=0.035) stays clearly human.")

H1("G &nbsp; Discussion &amp; Reproducibility")
B(["<b>What MoE fixes:</b> length specialization + genuine short data. HC3 recall was the unlock — the baseline had never seen chat Q&A. The 20k→16k per-expert vocabulary (vs 75k joint) also avoids long-essay features drowning short text.",
   "<b>What remains hard:</b> &lt;25-50w is intrinsically weak (even MoE 86-93% with warnings). 150w essays still prefer the baseline 97.4% vs 95.6% — the short expert leans chat; blending experts at the boundary is future work.",
   "<b>Thresholds transfer:</b> calibration helps essay truncations but slightly hurts human-heavy HC3 natural short — thresholds are domain-sensitive; keep them length-adaptive, not global.",
   "<b>Type gate:</b> 80.5% is enough for routing insight, not for enforcement. Casual↔chat confusion is linguistic, not a bug."])
P("Reproduce: <font face=\"Courier\" size=\"8\">PYTHONPATH=. python3 moe/download_small_text.py</font> → <font face=\"Courier\" size=\"8\">moe/train_moe.py</font> → <font face=\"Courier\" size=\"8\">moe/calibrate_thresholds.py</font> → <font face=\"Courier\" size=\"8\">moe/make_moe_appendix.py</font>. All artifacts under <font face=\"Courier\" size=\"8\">models/moe/</font>, metrics under <font face=\"Courier\" size=\"8\">results/moe_*.csv</font>.")

H1("Appendix References")
B(["Guo, B. et al. How Close is ChatGPT to Human Experts? HC3 / Hello-SimpleAI/HC3. (2023).",
   "artnitolog/llm-generated-texts (GPT-4 Turbo/Omni, Claude 3 Opus, Llama3, Command R+ essay prompts).",
   "Roy, R. et al. A Comprehensive Dataset for Human vs AI Generated Text Detection (Defactify). arXiv:2510.22874 (2025) — context for cross-domain eval.",
   "Baseline report references apply (Hastie et al.; Robertson &amp; Zaragoza; Sentence-BERT/DistilBERT/RoBERTa)."])

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Appendix saved -> {OUT_APP} ({os.path.getsize(OUT_APP)/1024:.0f} KB)")

# ---------- Merge ----------
from pypdf import PdfWriter, PdfReader
orig = os.path.join(ROOT, "Project_Report_AI_Text_Classification.pdf")
appendix = OUT_APP
merged = os.path.join(ROOT, "Project_Report_AI_Text_Classification.merged.pdf")
writer = PdfWriter()
writer.append(orig); writer.append(appendix)
with open(merged, "wb") as f:
    writer.write(f)
# backup original (keep one backup), promote merged
backup = os.path.join(ROOT, "Project_Report_AI_Text_Classification.pre-moe.pdf")
if not os.path.exists(backup):
    os.rename(orig, backup)
else:
    os.remove(orig)
os.rename(merged, orig)
# report.pdf is the second published copy - sync it too
import shutil
shutil.copy(orig, os.path.join(ROOT, "report.pdf"))
print(f"Merged -> {orig} ({os.path.getsize(orig)/1024:.0f} KB), report.pdf synced, backup -> {backup}")
print(f"Pages: original backup + appendix merged total = {len(PdfReader(orig).pages)}")
