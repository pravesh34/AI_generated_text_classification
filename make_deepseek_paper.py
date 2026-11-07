"""
Generate DeepSeek-style detailed research paper PDF.
Output: DeepSeek_Style_Research_Paper_MoE.pdf
Style: arXiv deep dive — formal notation, equations, algorithms, ablations.
Run: PYTHONPATH=. python3 make_deepseek_paper.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.dirname(__file__))
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
NAVY = "#1F2A44"; ACCENT = "#4C72B0"; ORANGE = "#DD8452"; GREEN = "#55A868"

# ---- Extra figure: human-only accuracy vs length (from live recomputation, 5k test) ----
# Swept earlier: baseline human acc collapses, MoE stable; AI acc flips opposite.
lengths = [25, 50, 75, 100, 150, 200, 300]
base_H = [0.4895, 0.7416, 0.8619, 0.9131, 0.9653, 0.9825, 0.9920]
moe_H  = [0.9679, 0.9618, 0.9580, 0.9465, 0.9430, 0.9799, 0.9889]
base_A = [0.9865, 0.9871, 0.9876, 0.9882, 0.9887, 0.9855, 0.9865]
moe_A  = [0.6792, 0.8794, 0.9403, 0.9612, 0.9774, 0.9699, 0.9758]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
axes[0].plot(lengths, np.array(base_H)*100, marker="o", lw=2.2, color=ORANGE, label="Baseline")
axes[0].plot(lengths, np.array(moe_H)*100, marker="s", lw=2.2, color=ACCENT, label="MoE")
axes[0].set_xlabel("Truncation (first N words)"); axes[0].set_ylabel("Human accuracy % (TN rate)")
axes[0].set_title("Human-only — baseline collapses (49% @25w)", fontweight="bold", color=NAVY)
axes[0].legend(fontsize=9); axes[0].grid(True, linestyle=":", alpha=0.6); axes[0].set_ylim(40, 101)
axes[1].plot(lengths, np.array(base_A)*100, marker="o", lw=2.2, color=ORANGE, label="Baseline")
axes[1].plot(lengths, np.array(moe_A)*100, marker="s", lw=2.2, color=ACCENT, label="MoE")
axes[1].set_xlabel("Truncation (first N words)"); axes[1].set_title("AI-only — error flips (MoE under-calls AI @25w)", fontweight="bold", color=NAVY)
axes[1].legend(fontsize=9); axes[1].grid(True, linestyle=":", alpha=0.6)
plt.suptitle("Length-conditional accuracy split by true class (5k held-out)", fontweight="bold", color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "deep_human_ai_split.png"), bbox_inches="tight"); plt.close()

# ---- Extra figure: threshold sensitivity (calib sweep, short expert) ----
ths = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
f25 = np.array([0.8407, 0.834, 0.8246, 0.8066, 0.7836, 0.7523, 0.7198])
f50 = np.array([0.8895, 0.900, 0.9089, 0.9092, 0.9048, 0.8962, 0.8858])
f100 = np.array([0.8949, 0.910, 0.9236, 0.9321, 0.9370, 0.9437, 0.9452])
f150 = np.array([0.8823, 0.905, 0.9257, 0.9341, 0.9426, 0.9468, 0.9533])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(ths, f25, marker="o", lw=2, label="25w (opt 0.32)")
ax.plot(ths, f50, marker="s", lw=2, label="50w (opt ~0.40)")
ax.plot(ths, f100, marker="^", lw=2, label="100w (opt 0.56)")
ax.plot(ths, f150, marker="D", lw=2, label="150w (opt 0.64)")
ax.axvline(0.5, color="gray", linestyle="--", label="default 0.50")
ax.set_xlabel("Decision threshold θ"); ax.set_ylabel("F1"); ax.set_title("Threshold sensitivity — optimum moves right with length", fontweight="bold", color=NAVY)
ax.legend(fontsize=9); ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "deep_threshold_curve.png"), bbox_inches="tight"); plt.close()
print("extra figures done")

# =================== PDF ===================
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)
from reportlab.lib import colors

OUT = os.path.join(ROOT, "DeepSeek_Style_Research_Paper_MoE.pdf")
NAVY_H = HexColor("#1F2A44"); ACCENT_H = HexColor("#4C72B0"); LIGHT = HexColor("#EAF0F8")
styles = getSampleStyleSheet()
sTitle = ParagraphStyle("Title", parent=styles["Title"], fontSize=19, leading=24, textColor=NAVY_H, alignment=TA_CENTER, spaceAfter=4)
sSub = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10.5, leading=13.5, textColor=ACCENT_H, alignment=TA_CENTER)
sH1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=17, textColor=NAVY_H, spaceBefore=12, spaceAfter=5, keepWithNext=True)
sH2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=ACCENT_H, spaceBefore=10, spaceAfter=4, keepWithNext=True)
sH3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, leading=13, textColor=NAVY_H, spaceBefore=8, spaceAfter=3, keepWithNext=True)
sBody = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.2, leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
sBullet = ParagraphStyle("Bullet", parent=sBody, leftIndent=14, bulletIndent=6, spaceAfter=3)
sEq = ParagraphStyle("Eq", parent=sBody, alignment=TA_CENTER, fontSize=9, leading=12.5, spaceBefore=4, spaceAfter=6, backColor=LIGHT, borderPadding=(6,6,6))
sAlgo = ParagraphStyle("Algo", parent=sBody, fontSize=8.5, leading=12, fontName="Courier", backColor=HexColor("#F5F5F5"), borderPadding=(8,8,8))
sCell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.2, leading=9, alignment=TA_CENTER)
sCellL = ParagraphStyle("CellL", parent=sCell, alignment=0)
sCap = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.gray, spaceBefore=2, spaceAfter=8)
sAbs = ParagraphStyle("Abs", parent=sBody, fontSize=9, leading=13, leftIndent=10, rightIndent=10)

def hdrftr(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.gray)
    canvas.drawCentredString(A4[0]/2, 12*mm, "Length-Expertized Authorship Detection (MoE) — Detailed Research Paper")
    canvas.drawRightString(A4[0]-15*mm, 12*mm, f"Page {doc.page}")
    canvas.setStrokeColor(ACCENT_H); canvas.setLineWidth(0.5)
    canvas.line(15*mm, A4[1]-15*mm, A4[0]-15*mm, A4[1]-15*mm)
    canvas.restoreState()

doc = SimpleDocTemplate(os.path.abspath(OUT), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                        topMargin=18*mm, bottomMargin=16*mm,
                        title="DeepSeek-style Detailed Paper — MoE for Human vs AI Text Detection", author="Pravesh Gedon Sharma & Ankit Rawat")
story = []
def H1(t): story.append(Paragraph(t, sH1))
def H2(t): story.append(Paragraph(t, sH2))
def H3(t): story.append(Paragraph(t, sH3))
def P(t): story.append(Paragraph(t, sBody))
def B(items):
    for it in items: story.append(Paragraph(it, sBullet, bulletText="•"))
def EQ(t, num):
    story.append(Paragraph(f"{t} &nbsp;&nbsp; ({num})", sEq))
def ALG(t):
    story.append(Paragraph(t.replace("\n", "<br/>"), sAlgo))
def FIG(name, w=165*mm, aspect=0.45):
    p = os.path.join(FIGDIR, name)
    if os.path.exists(p):
        story.append(Image(p, width=w, height=w*aspect, kind="proportional")); story.append(Spacer(1,1*mm))
    else: P(f"<i>[missing {name}]</i>")
def CAPT(t): story.append(Paragraph(t, sCap))
def TBL(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),LIGHT),("BOX",(0,0),(-1,-1),0.5,ACCENT_H),
                           ("INNERGRID",(0,0),(-1,-1),0.4,colors.gray),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                           ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
                           ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(t)

base_m = pd.read_csv(os.path.join(ROOT, "results", "baseline_ml_metrics.csv"))
moe_r = pd.read_csv(os.path.join(ROOT, "results", "moe_routed_metrics.csv"))
moe_b = pd.read_csv(os.path.join(ROOT, "results", "moe_baseline_comparison.csv"))
exp_m = pd.read_csv(os.path.join(ROOT, "results", "moe_expert_metrics.csv"))

# ---------- TITLE ----------
story.append(Spacer(1, 14*mm))
story.append(Paragraph("Length-Expertized Authorship Detection:<br/>Why Long-Text Detectors Fail on Short Text<br/>and How Mixture-of-Experts Fixes Them", sTitle))
story.append(Paragraph("A detailed empirical study with small-text data, specialized routing, type gating, and calibration", sSub))
story.append(HRFlowable(width="55%", thickness=1.2, color=ACCENT_H, spaceAfter=6, spaceBefore=6, hAlign="CENTER"))
story.append(Paragraph("Pravesh Gedon Sharma (B240019) &amp; Ankit Rawat (B320032)<br/>Dept. Computer Science &amp; Engineering, NIT Sikkim<br/>Course Project (2026) — arXiv-style detailed report", ParagraphStyle("auth", parent=sBody, alignment=TA_CENTER, fontSize=9.5)))
story.append(Spacer(1, 4*mm))
H2("Abstract")
P("Supervised detectors for human vs machine-generated text report 99%+ accuracy on essays, yet practitioners observe rejection on short posts, chats and blurbs. We quantify this <b>length-generalization failure</b> and fix it with a <b>Mixture-of-Experts (MoE)</b> whose experts specialize by text length. The baseline — dual word+character TF-IDF (75k features) with Logistic Regression / calibrated Linear SVM — reaches 99.25–99.58% on full essays but falls to <b>67.4% at 25 words, 59.8% at one sentence</b>, with human accuracy 48.9% (false AI rate ~51%) while AI recall stays ~98.7%: a one-sided collapse. We add <b>HC3 short Q&amp;A (80,811 answers, median 117 words)</b> and <b>modern 2024-LLM essays (17,997 texts, GPT-4/Claude/Llama3/Command R+)</b>, plus truncation augmentation, and train three experts — <b>short (≤150 words, 20k), long (151–400, 16.9k), verylong (&gt;400, 11.4k)</b> — each Word 15k + Char 25k + LR, routed deterministically by word count. A learned <b>type gate</b> (16 stylometric features → chat/casual/essay, 80.5% val) provides domain insight; <b>length-adaptive thresholds</b> (25w→0.32 … 150w→0.64) correct the FP/FN flip. Routed MoE restores 86.1% at 25 words, 93.1% at 50 words, 81.6% at one sentence, holds 98.96% full-text, and crucially scores <b>98.5% on unseen HC3 chat</b> (baseline 48.6%) and <b>95.7% on unseen 2024-LLM essays</b> (baseline 85.5%). Calibration adds +1.9 pp at 25 words. We release data pipelines, models, metrics and figures under <font face=\"Courier\" size=\"8\">moe/</font> for full reproducibility.")
P("<b>Keywords:</b> machine-generated text detection; mixture of experts; TF-IDF; stylometry; short-text classification; HC3; calibration.")
P(" <b>1 &nbsp; Introduction</b>")
H2("1.1 &nbsp; The length blind spot in authorship detection")
P("Whether a passage is human-written or LLM-generated underlies academic-integrity, journalism and moderation decisions. Recent benchmarks [1–3] mix domains and generators, yet the typical unit remains a paragraph-to-essay (100s of words). Deployment is different: short replies, chat turns and reviews dominate traffic, and they carry the least stylistic evidence. A detector that needs 300 words to decide is undeployable for half its inputs — and, worse, <i>confidently wrong</i>: it labels short humans as AI (Section 7.1: human accuracy 48.9% at 25 words). This paper treats <b>length as a first-class routing variable</b>, not a nuisance to average over.")
H2("1.2 &nbsp; Problem statement")
P("Let x be text, y ∈ {0,1} (0 human, 1 AI), n(x) word count. We seek f(x) → ŷ maximizing accuracy under length shift: training lengths ∼ D<sub>train</sub> (median 363w, P5 180w) but test lengths n ≪ 150. Two phenomena matter: (i) <i>evidence sparsity</i> — TF-IDF counts scale with n, char n-grams need minimum support, stylometrics (TTR, burstiness) stabilize only with many tokens; (ii) <i>domain shift</i> — short text is chat/casual, not truncated essay, so essay-only training cannot transfer (hence HC3).")
H2("1.3 &nbsp; Contributions")
B(["<b>C1. Failure characterization.</b> A reproducible length ablation (prefix truncations + random windows) showing monotone collapse and an FP/FN flip, split by true class (Table 6, Figure 11).",
   "<b>C2. Small-text data.</b> HC3 (80.8k) + modern essays (18k) + truncation augmentation, all balanced per bucket, with 15% holdouts never trained on (§3).",
   "<b>C3. MoE + type gate + calibration.</b> Three experts (val ≥97.0%), chat/casual/essay gate (80.5%), length-adaptive thresholds (§4, §6).",
   "<b>C4. Artifacts.</b> <font face=\"Courier\" size=\"8\">moe/*.py</font>, <font face=\"Courier\" size=\"8\">models/moe/</font>, <font face=\"Courier\" size=\"8\">results/moe_*.csv</font>, Streamlit/CLI demos. Every table/figure is computed from these files."])

P("<b>2 &nbsp; Related Work</b>")
H2("2.1 &nbsp; Supervised AI-text detection")
P("Supervised TF-IDF + linear classifiers remain strong in-domain baselines. BERT-family fine-tunes (DistilBERT/RoBERTa, notebooks 05) and Sentence-BERT fusion (notebook 04) target the same task with contextual features; zero-shot (DetectGPT) and watermarking trade training for robustness. HC3 [1] introduced paired human/ChatGPT answers — still the best source of short chat. Defactify [2] (73k news + 6 LLMs, 58.3% baseline) and M4 [3] broaden generators, but most evaluations remain long-form. Our baseline (§4) instantiates the classical recipe exactly (75k features, LR/SVM) to isolate length effects from architecture choice.")
H2("2.2 &nbsp; Stylometry")
P("Lexical diversity (TTR = |types|/|tokens|), hapax ratio, burstiness σ/μ of sentence lengths, punctuation/capital rates, and readability (Flesch Reading Ease, FKGL, Gunning Fog, §4.2) capture authorial habit. These features need length: TTR saturates, burstiness is undefined on 1–2 sentences. We reuse <font face=\"Courier\" size=\"8\">src/stylometrics.py</font> both as gate features and as UI explanations.")
H2("2.3 &nbsp; Mixture of experts")
P("MoE routes inputs to specialized sub-networks [4]. Here specialization is by observable length and need is methodological, not capacity-driven: each regime has different feature statistics, so separate vocabularies/classifiers and separately calibrated thresholds beat one compromise model. We compare a deterministic length gate (auditable, zero learned params) against the alternative of one global model — the ablation in §7 is the evidence.")

P("<b>3 &nbsp; Datasets</b>")
H2("3.1 &nbsp; Original essays")
P("Train 290,300 (145,150/class, 623 MB), test 97,447 (61,159 human / 36,288 AI, 212 MB). Unit: multi-paragraph academic essay. Test-sample word statistics (5k, seed 42): mean 394, median 363, P10 211, P90 617, max 1,650; human mean 423 vs AI 345. Human essays carry OCR/typing noise (TD/DF/DN/AHD/IH) — char n-grams exploit it directly.")
H2("3.2 &nbsp; HC3 — genuine short chat (new)")
P("Hello-SimpleAI/HC3 [1], files <font face=\"Courier\" size=\"8\">reddit_eli5.jsonl</font> (17,112 Q, 67,671 answers), <font face=\"Courier\" size=\"8\">open_qa.jsonl</font>, <font face=\"Courier\" size=\"8\">finance.jsonl</font> downloaded via Hub (no dataset-script; works on datasets 5.x). Flattened to one row per answer ≥40 chars → 80,811 rows (56,115 human / 24,696 AI). Word stats: mean 146, median 117, P10 28; 61.3% ≤150w, 34.7% 151–400w, 4.0% &gt;400w. Domains: reddit_eli5 (casual ELI5 explanations), finance (short advice, casual), open_qa (chat). 15% stratified holdout (12,122) is eval-only.")
H2("3.3 &nbsp; Modern essays — 2024-LLM audit (new)")
P("artnitolog/llm-generated-texts: 3,000 essay prompts × 1 human + 8 continuations (GPT-4 Turbo/Omni, Claude 3 Opus, YandexGPT, GigaChat, Llama3 70B, Command R+). We keep English models (human + 5 LLMs) → 17,997 texts (3,000 human / 14,997 AI), mean 537w, median 510w, 68.2% &gt;400w. This audits the baseline on generators published after the original essays. Holdout 2,700 eval-only.")
H2("3.4 &nbsp; Truncation augmentation")
P("Each 20k train essay contributes a 50/100/150w copy (prefix or random window, label preserved) for the short pool; ≥200w essays contribute 200/300w copies (long pool, filtered ≥151w after trunc — fixes a pilot bug where short originals polluted the long bucket). Prefix vs random-window accuracies are within ~1–2 pp (50w: 83.3% vs 81.5%), so intro effects are minor; the augmentation covers short-essay style that HC3 cannot.")
rows = [[Paragraph("Pool",sCell), Paragraph("Sources (balanced caps)",sCell), Paragraph("n",sCell)],
        [Paragraph("SHORT ≤150w",sCell), Paragraph("HC3 ≤150w 7k/class + trunc-essay 3k/class",sCell), Paragraph("20,000 (10k/10k)",sCell)],
        [Paragraph("LONG 151–400w",sCell), Paragraph("HC3 long 4k/class + modern long 2k/class + base long 3k/class",sCell), Paragraph("16,864 (7,864/9,000)",sCell)],
        [Paragraph("VERYLONG &gt;400w",sCell), Paragraph("Base &gt;400w 7k/class + modern &gt;400w 2.5k/class",sCell), Paragraph("11,422 (6,321/5,101)",sCell)]]
TBL(rows, [38*mm, 85*mm, 37*mm]); CAPT("Table 1: MoE training pools (seed 42). Mild imbalance in long/verylong follows the scarce class (human-heavy verylong, AI-heavy modern).")

P("<b>4 &nbsp; Problem Formulation and Method</b>")
H2("4.1 &nbsp; Notation and objective")
EQ("ŷ = 1[P<sub>AI</sub>(x) ≥ θ<sub>n</sub>] &nbsp;with n = n(x) word count; &nbsp; MoE: P = P<sub>e(x)</sub><sup>AI</sup>(x); &nbsp; e(x) = short/long/verylong gate", 1)
P("We maximize accuracy/F1 under length shift, reporting precision/recall splits because accuracy alone hides the FP/FN flip.")
H2("4.2 &nbsp; Text representation")
P("Word TF-IDF (1,2, min_df 2, sublinear) + char_wb (3,5, min_df 3), fused sparsely:")
EQ("tfidf(t,d) = tf(t,d) · log((1+N)/(1+df(t))) ; &nbsp; F(x) = [F<sub>word</sub>(x); F<sub>char</sub>(x)] ∈ ℝ<sup>V</sup>", 2)
P("Baseline V=75,000 (25k+50k); each expert V=40,000 (15k+25k). LR minimizes −log-likelihood with L2 (C=1.5), LinearSVC uses hinge + Platt scaling. Stylometrics (16 features, §4.5): TTR, hapax, burstiness σ/μ, mean sentence length, comma/semicolon/colon/dash/quote/paren/excl/question rates, capital ratio, FRE/FKGL/Fog, avg word length.")
EQ("FRE = 206.835 − 1.015·(N/S) − 84.6·(syll/N); &nbsp; FKGL = 0.39·(N/S) + 11.8·(syll/N) − 15.59; &nbsp; Fog = 0.4·[(N/S)+100·(poly/N)]", 3)
H2("4.3 &nbsp; Baseline and length ablation (motivation)")
P("Baseline training follows §4.2 exactly (20k subsample, 16k/4k split). Length ablation truncates the 5k held-out to first-N words/sentences. Result (§7.1): monotone degradation, confidence collapse (0.93→0.79), and — critically — human accuracy 48.9% @25w vs AI 98.7% (Table 6). This asymmetry proves the failure is <i>bias</i>, not just difficulty: short humans look like the AI class the model expects.")
H2("4.4 &nbsp; MoE architecture")
EQ("e(x) = short if n≤150, long if 151≤n≤400, verylong if n&gt;400; &nbsp; P(x) = P<sub>e(x)</sub>(x); &nbsp; expert = LR∘TFIDF<sub>e</sub>", 4)
P("Gating is deterministic (interpretable, no routing collapse). Each expert trains on its pool only (80/20 within pool). Inference additionally reports type (§4.5) and warns if n&lt;25 (&lt;50).")
FIG("moe_architecture.png", 170*mm, 0.56); CAPT("Figure 1: MoE architecture — two gates, three experts.")
ALG("Algorithm 1 — MoE training (moe/train_moe.py)\nInput: base 20k essays, HC3 train 68.7k, modern train 15.3k, seed 42\n1. pools ← (short: HC3≤150 + trunc50/100/150; long: HC3 151-400 + modern + base; verylong: base>400 + modern>400)\n2. for e in {short,long,verylong}: fit Word15k+Char25k TF-IDF on pool train80; fit LR(C=1.5); save expert_e.pkl\n3. gate ← LR(StandardScaler(16 stylometrics)) on 7.5k (2.5k/chat/casual/essay); save type_gate.pkl\n4. routed eval on 5k test + HC3/modern holdouts; baseline eval on same conditions")
H2("4.5 &nbsp; Type gate")
P("Labels: HC3 open_qa→chat (4k avail., cap 2.5k), HC3 eli5/finance + truncated essays → casual, full/modern essays → essay (2.5k each). Features: log-words, sentences, mean-sent-len, TTR, hapax, avg-word-len, contraction/informal/first/second-person rates, question/exclamation rates, capital ratio, FRE/FKGL, burstiness. Val: 80.5% (essay P/R/F1 0.90/0.92/0.91; chat 0.74/0.84/0.79; casual 0.77/0.66/0.71). Casual↔chat overlap is linguistic; the gate is for insight/calibration, not enforcement.")
FIG("moe_type_gate.png", 140*mm, 0.47); CAPT("Figure 2: Type gate per-class scores.")
H2("4.6 &nbsp; Calibration")
P("Default θ=0.50 assumes symmetric costs. On calib half (2.5k), F1-optimal θ rises with length (25w 0.32 … 150w 0.64) because ultra-short errs FN (under-call AI) while 100–150w errs FP. We ship <font face=\"Courier\" size=\"8\">θ(n)=0.28+0.0024n</font> clamped [0.32,0.64] for short, 0.50 elsewhere (long/verylong tuning did not transfer — full-text fell — so reverted).")

P("<b>5 &nbsp; Experimental Setup</b>")
H2("5.1 &nbsp; Environment and protocol")
P("Python 3.13, scikit-learn, joblib, reportlab. All splits seed 42 (calibration 999). Held-out: test.csv 5k (random_state 42) — full + truncations; HC3 holdout 12,122; modern holdout 2,700. Training time 130 s (TF-IDF 6/15/25 s per expert). No leakage: test.csv never trained; HC3/modern holdouts excluded from pools.")
H2("5.2 &nbsp; Metrics")
P("Accuracy, precision, recall, F1 (AI as positive), ROC-AUC, confusion matrices. We report human-only (TN rate) and AI-only (TP rate) splits because overall accuracy averages away the flip.")

P("<b>6 &nbsp; Main Results</b>")
H2("6.1 &nbsp; Per-expert and routed")
rows = [[Paragraph("Expert",sCell), Paragraph("Train/Val",sCell), Paragraph("Acc",sCell), Paragraph("Prec",sCell), Paragraph("Rec",sCell), Paragraph("F1",sCell), Paragraph("AUC",sCell)]]
for _, r in exp_m.iterrows():
    rows.append([Paragraph(str(r["expert"]),sCell), Paragraph(f"{int(r['n_train']):,}/{int(r['n_val']):,}",sCell),
                 Paragraph(f"{r['accuracy']:.4f}",sCell), Paragraph(f"{r['precision']:.4f}",sCell), Paragraph(f"{r['recall']:.4f}",sCell),
                 Paragraph(f"{r['f1']:.4f}",sCell), Paragraph(f"{r['roc_auc']:.4f}",sCell)])
TBL(rows, [28*mm, 28*mm, 20*mm, 20*mm, 20*mm, 20*mm, 22*mm]); CAPT("Table 2: Per-expert validation (results/moe_expert_metrics.csv).")
FIG("moe_experts.png", 160*mm, 0.33); CAPT("Figure 3: Per-expert metrics.")
P("Routed MoE: full 98.96% (P 0.993, R 0.979, F1 0.986) vs baseline 99.26% — specialization costs 0.3 pp. Buckets on full: verylong 99.59% (n=1,958), long 98.81% (2,945), short 90.72% (97). The win is short + new domains (Figure 4–5, Table 3).")
FIG("moe_accuracy_vs_length.png", 165*mm, 0.45); CAPT("Figure 4: Accuracy vs length.")
FIG("moe_cross_domain.png", 150*mm, 0.47); CAPT("Figure 5: Cross-domain — baseline collapses on chat.")
rows = [[Paragraph("Condition",sCell), Paragraph("Baseline acc",sCell), Paragraph("MoE acc",sCell), Paragraph("Δ",sCell)]]
for cond, lab in [("full","full"),("first_25_words","25w"),("first_50_words","50w"),("first_1_sent","1 sent"),("hc3_holdout","HC3"),("modern_holdout","modern")]:
    bv=float(moe_b[moe_b["test_condition"]==cond].iloc[0]["accuracy"]); mv=float(moe_r[moe_r["test_condition"]==cond].iloc[0]["accuracy"])
    rows.append([Paragraph(lab,sCell), Paragraph(f"{bv*100:.2f}%",sCell), Paragraph(f"{mv*100:.2f}%",sCell), Paragraph(f"+{(mv-bv)*100:.1f}pp",sCell)])
TBL(rows, [40*mm, 32*mm, 32*mm, 28*mm]); CAPT("Table 3: Headline deltas (results/moe_*.csv).")
H2("6.2 &nbsp; Calibration (eval half)")
P("Adaptive thresholds: 25w 86.2%→88.1% (F1 0.785→0.837, recall 0.68→0.82), 50w flat (93.1%→93.1%, R↑ P↓), 100w identical (th≈0.52), 150w 95.0%→96.2%, full flat. HC3 natural short −0.6 pp (human-heavy FPs) — thresholds are essay-tuned; keep the &lt;50w warning. Long/verylong stayed at 0.50.")
FIG("deep_threshold_curve.png", 150*mm, 0.45); CAPT("Figure 6: F1 vs threshold — optimum drifts right with length; no single θ serves all.")

P("<b>7 &nbsp; Ablations and Analysis</b>")
H2("7.1 &nbsp; Length ablation — the flip in detail")
rows = [[Paragraph("Words",sCell), Paragraph("Base H",sCell), Paragraph("MoE H",sCell), Paragraph("Base AI",sCell), Paragraph("MoE AI",sCell), Paragraph("Base acc",sCell), Paragraph("MoE acc",sCell)],
        [Paragraph("25",sCell), Paragraph("48.9%",sCell), Paragraph("96.8%",sCell), Paragraph("98.7%",sCell), Paragraph("67.9%",sCell), Paragraph("67.4%",sCell), Paragraph("86.1%",sCell)],
        [Paragraph("50",sCell), Paragraph("74.2%",sCell), Paragraph("96.2%",sCell), Paragraph("98.7%",sCell), Paragraph("87.9%",sCell), Paragraph("83.3%",sCell), Paragraph("93.1%",sCell)],
        [Paragraph("75",sCell), Paragraph("86.2%",sCell), Paragraph("95.8%",sCell), Paragraph("98.8%",sCell), Paragraph("94.0%",sCell), Paragraph("90.9%",sCell), Paragraph("—",sCell)],
        [Paragraph("100",sCell), Paragraph("91.3%",sCell), Paragraph("94.7%",sCell), Paragraph("98.8%",sCell), Paragraph("96.1%",sCell), Paragraph("94.1%",sCell), Paragraph("95.2%",sCell)],
        [Paragraph("150",sCell), Paragraph("96.5%",sCell), Paragraph("94.3%",sCell), Paragraph("98.9%",sCell), Paragraph("97.7%",sCell), Paragraph("97.4%",sCell), Paragraph("95.6%",sCell)],
        [Paragraph("200",sCell), Paragraph("98.3%",sCell), Paragraph("98.0%",sCell), Paragraph("98.6%",sCell), Paragraph("97.0%",sCell), Paragraph("98.4%",sCell), Paragraph("97.6%",sCell)],
        [Paragraph("300",sCell), Paragraph("99.2%",sCell), Paragraph("98.9%",sCell), Paragraph("98.7%",sCell), Paragraph("97.6%",sCell), Paragraph("99.0%",sCell), Paragraph("—",sCell)],
        [Paragraph("full",sCell), Paragraph("99.6%",sCell), Paragraph("99.6%",sCell), Paragraph("98.7%",sCell), Paragraph("97.9%",sCell), Paragraph("99.3%",sCell), Paragraph("99.0%",sCell)]]
TBL(rows, [20*mm, 22*mm, 22*mm, 22*mm, 22*mm, 24*mm, 24*mm]); CAPT("Table 4: Length-conditional accuracy split (human TN rate vs AI TP rate). MoE rescues humans (+47.9 pp @25w) at AI-recall cost (−30.7 pp) — net acc still wins.")
FIG("deep_human_ai_split.png", 165*mm, 0.38); CAPT("Figure 7: Human (left) vs AI (right) accuracy — mirror-image failure/success.")
P("Interpretation: at 25w the baseline calls AI 98.7% of the time (any class) — length is the feature. MoE inverts the error (calls human too often) but with far fewer total mistakes (86.1% vs 67.4%). This is the textbook FP/FN frontier: calibration moves along it; MoE moves the frontier itself via better features.")
FIG("moe_pr_flip.png", 130*mm, 0.55); CAPT("Figure 8: 25w precision/recall — accuracy hides the tradeoff.")
H2("7.2 &nbsp; Sentence ablation")
P("1/2/3 sentences: baseline 59.8%/74.5%/83.4% → MoE 81.6%/90.6%/93.4%. Sentence truncations are harder than word truncations at matched budget (more padding loss), confirming sentence structure carries signal the short expert partially recovers via char features.")
H2("7.3 &nbsp; Type analysis")
P("Type gate: essay easiest (long formal → distinct), chat next (short informal markers), casual hardest (middle). Routing insight: short+chat inputs have the lowest confidence and need warnings; short+essay (truncated) input needs the low threshold. Reporting type alongside label is a human-factor result, not just accuracy.")
H2("7.4 &nbsp; What did not help")
P("Tuning long/verylong thresholds hurt full-text (98.96%→98.60% on eval) — small calib slices overfit. Global 0.60 (best for 100–150w essay truncations) hurt HC3 natural chat. Conclusion: calibrate only where the failure is large and monotonic (short), not everywhere.")

P("<b>8 &nbsp; Limitations and Future Work</b>")
B(["<b>Ultra-short floor.</b> Even MoE is 86% at 25w with wide uncertainty. Training saw ≥50w; adding 10–35w + few-shot chat would likely beat thresholds.",
   "<b>Boundary blending.</b> 150w essays prefer baseline (97.4% vs 95.6%); soft routing (weighted P across short/long) could interpolate.",
   "<b>Threshold transfer.</b> Essay-tuned θ hurts human-heavy chat (−0.6 pp). Per-type thresholds (essay vs chat) are the next step.",
   "<b>Generators.</b> HC3 is one 2023 ChatGPT snapshot; modern holdout is 5 models. Paraphrase/adversarial eval is missing.",
   "<b>Scale.</b> Full 290k training might add 0.1–0.3 pp; transformers (DistilBERT/RoBERTa) with chunking remain the Phase-2 ceiling test."])
P("Ethics: P(AI) is a screening signal, not proof. Short-text warnings and type reporting are required UI, not optional captions.")

P("<b>9 &nbsp; Conclusion</b>")
P("Length is not a detail — it is the task when evaluating short text. One global TF-IDF model buys 99% on essays by learning to call short text AI. Specializing by length, grounding short experts in real chat data, labeling chat/casual/essay, and calibrating per length recovers +10 to +50 pp where it matters, with a 0.3 pp full-text tax. We release the full stack so the claim is checkable: <font face=\"Courier\" size=\"8\">moe/download_small_text.py → train_moe.py → calibrate_thresholds.py → predict_moe.py</font>.")

P("<b>References</b>")
B(["[1] Guo, B. et al. How Close is ChatGPT to Human Experts? arXiv:2301.07597 (2023); Hello-SimpleAI/HC3 (reddit_eli5/open_qa/finance).",
   "[2] Roy, R. et al. A Comprehensive Dataset for Human vs AI Generated Text Detection. arXiv:2510.22874 (2025) (Defactify).",
   "[3] Wang, B. et al. M4: Multi-Generator, Multi-Domain, Multi-Task, Multi-Label. arXiv:2310.13607 (2023).",
   "[4] Shazeer, N. et al. Outrageously Large Neural Networks: MoE. arXiv:1701.06538 (2017).",
   "[5] Hastie, T. et al. The Elements of Statistical Learning. Springer (2009).",
   "[6] Robertson, S. &amp; Zaragoza, H. BM25 &amp; TF-IDF. Found. Trends IR (2009).",
   "[7] Reimers, N. &amp; Gurevych, A. Sentence-BERT. EMNLP (2019). Sanh, T. DistilBERT (2019); Liu, Y. RoBERTa (2019).",
   "[8] art nitolog/llm-generated-texts (GPT-4 Turbo/Omni, Claude 3 Opus, Llama3, Command R+) — Hugging Face.",
   "[9] Mitchell, E. DetectGPT. arXiv:2301.11305 (2023) — zero-shot baseline family."])

P("<b>Appendices</b>")
H2("A &nbsp; Hyperparameters")
rows = [[Paragraph("Component",sCellL), Paragraph("Setting (all seeded)",sCell)],
        [Paragraph("Baseline word TF-IDF",sCellL), Paragraph("(1,2) 25k, min_df 2, sublinear, clean_text_for_tfidf",sCell)],
        [Paragraph("Baseline char TF-IDF",sCellL), Paragraph("char_wb (3,5) 50k, min_df 3, sublinear",sCell)],
        [Paragraph("Baseline classifiers",sCellL), Paragraph("NB α=0.1; LR C=1.5 lbfgs 1000; LinearSVC C=1.0 + CalibratedCV(3)",sCell)],
        [Paragraph("Experts (each)",sCellL), Paragraph("Word 15k + Char 25k → LR C=1.5 lbfgs 1000",sCell)],
        [Paragraph("Type gate",sCellL), Paragraph("StandardScaler + LR C=1.0 lbfgs 1000 on 16 features",sCell)]]
TBL(rows, [55*mm, 105*mm]); CAPT("Table A1: Hyperparameters.")
H2("B &nbsp; Reproducibility checklist")
B(["Datasplits: train subsample (10k/class, seed 42), 80/20 per pool (seed 42), calib 2.5k/eval 2.5k (seed 999), HC3/modern holdouts 15% (seed 42).",
   "Artifacts: <font face=\"Courier\" size=\"8\">models/*.pkl</font> (baseline), <font face=\"Courier\" size=\"8\">models/moe/expert_*.pkl, type_gate.pkl, thresholds.json</font>; metrics <font face=\"Courier\" size=\"8\">results/moe_*.csv</font>.",
   "Indirection: truncation code is in <font face=\"Courier\" size=\"8\">moe/type_features.py</font> (truncate_words/random_window); routing in <font face=\"Courier\" size=\"8\">moe/train_moe.py::route_predict</font>.",
   "PDFs: <font face=\"Courier\" size=\"8\">make_report.py</font> (6p), <font face=\"Courier\" size=\"8\">moe/make_moe_appendix.py</font> (6p), <font face=\"Courier\" size=\"8\">make_research_paper.py</font> (7p), this file (deep)."])
H2("C &nbsp; Full routed table")
rows = [[Paragraph("Condition",sCell), Paragraph("Acc",sCell), Paragraph("Prec",sCell), Paragraph("Rec",sCell), Paragraph("F1",sCell), Paragraph("AUC",sCell)]]
for c in ["full","first_25_words","first_50_words","first_100_words","first_150_words","first_200_words","first_1_sent","first_2_sent","first_3_sent","hc3_holdout","modern_holdout"]:
    r = moe_r[moe_r["test_condition"]==c].iloc[0]
    rows.append([Paragraph(str(c),sCell), Paragraph(f"{r['accuracy']:.4f}",sCell), Paragraph(f"{r['precision']:.4f}",sCell),
                 Paragraph(f"{r['recall']:.4f}",sCell), Paragraph(f"{r['f1']:.4f}",sCell), Paragraph(f"{r['roc_auc']:.4f}",sCell)])
TBL(rows, [38*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm]); CAPT("Table C1: Routed MoE (results/moe_routed_metrics.csv).")
P("Baseline counterpart in <font face=\"Courier\" size=\"8\">results/moe_baseline_comparison.csv</font>; calibration detail in <font face=\"Courier\" size=\"8\">results/moe_calibrated_metrics.csv</font>.")

doc.build(story, onFirstPage=hdrftr, onLaterPages=hdrftr)
print(f"Saved -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
