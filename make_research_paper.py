"""
Generate Research Paper PDF: Mixture-of-Experts for Human vs AI Text Detection
Output: Research_Paper_MoE_AI_Text_Detection.pdf
Uses: results/*.csv (live metrics), results/figures/*.png (all figures)
Run: PYTHONPATH=. python3 make_research_paper.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.dirname(__file__))
FIGDIR = os.path.join(ROOT, "results", "figures")

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)
from reportlab.lib import colors

OUT = os.path.join(ROOT, "Research_Paper_MoE_AI_Text_Detection.pdf")
NAVY = HexColor("#1F2A44"); ACCENT = HexColor("#4C72B0"); LIGHT = HexColor("#EAF0F8")
styles = getSampleStyleSheet()
sTitle = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, leading=26, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
sSub = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=14, textColor=ACCENT, alignment=TA_CENTER)
sH1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=5, keepWithNext=True)
sH2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=True)
sBody = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5)
sBullet = ParagraphStyle("Bullet", parent=sBody, leftIndent=14, bulletIndent=6, spaceAfter=3)
sCell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.5, leading=9.5, alignment=TA_CENTER)
sCellL = ParagraphStyle("CellL", parent=sCell, alignment=0)
sCap = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.gray, spaceBefore=2, spaceAfter=8)
sAbs = ParagraphStyle("Abs", parent=sBody, fontSize=9.5, leading=13.5, leftIndent=10, rightIndent=10)

def hdrftr(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.gray)
    canvas.drawCentredString(A4[0]/2, 12*mm, "Human vs AI Text Detection via Mixture-of-Experts — NIT Sikkim Course Project (2026)")
    canvas.drawRightString(A4[0]-15*mm, 12*mm, f"Page {doc.page}")
    canvas.setStrokeColor(ACCENT); canvas.setLineWidth(0.5)
    canvas.line(15*mm, A4[1]-15*mm, A4[0]-15*mm, A4[1]-15*mm)
    canvas.restoreState()

doc = SimpleDocTemplate(os.path.abspath(OUT), pagesize=A4,
                        leftMargin=16*mm, rightMargin=16*mm, topMargin=18*mm, bottomMargin=16*mm,
                        title="Mixture-of-Experts for Human vs AI Text Detection", author="Pravesh Gedon Sharma & Ankit Rawat")
story = []
def H1(t): story.append(Paragraph(t, sH1))
def H2(t): story.append(Paragraph(t, sH2))
def P(t): story.append(Paragraph(t, sBody))
def B(items):
    for it in items: story.append(Paragraph(it, sBullet, bulletText="•"))
def FIG(name, w=165*mm, aspect=0.45):
    p = os.path.join(FIGDIR, name)
    if os.path.exists(p):
        story.append(Image(p, width=w, height=w*aspect, kind="proportional"))
        story.append(Spacer(1, 1*mm))
    else:
        P(f"<i>[missing figure {name}]</i>")
def CAPT(t): story.append(Paragraph(t, sCap))
def TBL(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),LIGHT),("BOX",(0,0),(-1,-1),0.5,ACCENT),
                           ("INNERGRID",(0,0),(-1,-1),0.4,colors.gray),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                           ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
                           ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(t)

# Load live metrics for tables
base_m = pd.read_csv(os.path.join(ROOT, "results", "baseline_ml_metrics.csv"))
moe_r = pd.read_csv(os.path.join(ROOT, "results", "moe_routed_metrics.csv"))
moe_b = pd.read_csv(os.path.join(ROOT, "results", "moe_baseline_comparison.csv"))
exp_m = pd.read_csv(os.path.join(ROOT, "results", "moe_expert_metrics.csv"))
cal_m = pd.read_csv(os.path.join(ROOT, "results", "moe_calibrated_metrics.csv"))

def X(df, cond): return df[df["test_condition"] == cond].iloc[0]

# ===== TITLE =====
story.append(Spacer(1, 18*mm))
story.append(Paragraph("Length-Expertized Authorship Detection:<br/>A Mixture-of-Experts for Human vs<br/>AI-Generated Text", sTitle))
story.append(Paragraph("Fixing the Short-Text Collapse with Dedicated Small-Text Data, Routing, and Calibration", sSub))
story.append(HRFlowable(width="55%", thickness=1.2, color=ACCENT, spaceAfter=6, spaceBefore=6, hAlign="CENTER"))
story.append(Paragraph("Pravesh Gedon Sharma (B240019) &amp; Ankit Rawat (B320032)<br/>Department of Computer Science &amp; Engineering<br/>National Institute of Technology Sikkim<br/>Course Project (2026)", ParagraphStyle("author", parent=sBody, alignment=TA_CENTER, fontSize=9.5)))
story.append(Spacer(1, 6*mm))
H2("Abstract")
P("Classical TF-IDF detectors for human vs AI text achieve 99%+ accuracy on long essays but collapse on short text: 67.4% at 25 words and 59.8% at one sentence, with humans systematically biased to AI. We diagnose the length failure, add genuine small-text data (HC3 Q&A, 80,811 answers, median 117 words; plus 17,997 modern-LLM essays from GPT-4, Claude 3, Llama3 and Command R+), and train a <b>Mixture-of-Experts (MoE)</b>: a deterministic length gate (short ≤150 / long 151–400 / verylong &gt;400 words) routing to three TF-IDF+LogisticRegression experts, plus a learned type gate (chat / casual / essay, 80.5% validation) and length-adaptive thresholds. On 5,000 held-out essays, MoE lifts short-text accuracy to 86.1% at 25 words, 93.1% at 50 words and 81.6% at one sentence, while holding 98.96% on full text. Cross-domain, MoE scores 98.5% on unseen HC3 chat answers (baseline 48.6%) and 95.7% on unseen 2024-LLM essays (baseline 85.5%). Calibration (25w→0.32 … 150w→0.64) further adds +1.9 pp at 25 words. The system is released as <font face=\"Courier\" size=\"8\">moe/</font> with routers, metrics and this paper's figures.")
P("<b>Keywords:</b> AI text detection; mixture of experts; TF-IDF; stylometrics; short-text classification; HC3; authorship attribution.")
P(" <b>1 &nbsp; Introduction</b>")
P("Large language models (ChatGPT, GPT-4, Claude) make fluent essays trivially available, creating an integrity problem in academia, journalism and moderation. Most published detectors are evaluated on documents hundreds of words long. Real deployment, however, includes tweets, chat replies, product reviews and single-sentence blurbs. We ask: <i>does a strong long-text detector survive truncation to short text, and what architecture fixes it?</i>")
P("Our Phase-1 baseline — dual word (1,2) + character (3,5) TF-IDF (75k features) with Logistic Regression (LR) and calibrated Linear SVM — scores 99.25–99.53% validation and 99.26–99.58% held-out accuracy on 20,000 balanced essays (Section 4). Truncating those same held-out essays destroys it: first 25 words → 67.4%, first 1 sentence → 59.8% (Section 5). AI recall stays ~0.98 while humans flip to AI at ~51% (25w). The model never saw short text (training median 363 words, 5th percentile 180).")
P("This paper makes three contributions. (1) A <b>length ablation</b> quantifying the failure per word/sentence budget. (2) <b>Small-text data + MoE</b>: HC3 short Q&A, modern 2024-LLM essays, and truncation augmentation, routed to length experts with a chat/casual/essay type gate. (3) <b>Threshold calibration</b> showing one global threshold is wrong per length. Code, datasets, models and metrics are under <font face=\"Courier\" size=\"8\">moe/</font> and <font face=\"Courier\" size=\"8\">results/moe_*.csv</font>.")

P("<b>2 &nbsp; Related Work</b>")
P("AI-text detection spans zero-shot (DetectGPT, likelihood curvature), watermarking, and supervised classifiers. HC3 (Guo et al.) [1] pairs human and ChatGPT answers to the same questions across domains; subsequent benchmarks (Defactify [2], M4 [3]) widen generators and domains but still skew long. OpenAI's classifier famously failed under 1,000 characters — the same cliff we replicate. Classical TF-IDF + linear SVM remains competitive on in-domain essays (Section 4) but, as we show, is brittle across lengths and generators. Mixture-of-experts routing by surface properties (here length) is a standard technique for specialized regimes [4]; our novelty is pairing length experts with per-length calibration and a learned text-type gate on stylometric features (<font face=\"Courier\" size=\"8\">src/stylometrics.py</font>: TTR, hapax, burstiness, Flesch ease/grade, punctuation rates).")

P("<b>3 &nbsp; Datasets</b>")
P("Original essays: <font face=\"Courier\" size=\"8\">data/train.csv</font> (290,300 rows, 145,150 per class) and <font face=\"Courier\" size=\"8\">data/test.csv</font> (97,447 rows, 61,159 human / 36,288 AI), columns <font face=\"Courier\" size=\"8\">text, generated</font>. Texts are multi-paragraph academic essays: test-sample mean 394 words, median 363, 10th percentile 211; human mean 423 vs AI 345 (Table 1, Figure 1–2). Training uses a 20,000 balanced subsample (10k/class, seed 42), 80/20 stratified → 16,000/4,000.")
rows = [[Paragraph("Dataset",sCell), Paragraph("Size",sCell), Paragraph("Human / AI",sCell), Paragraph("Median len",sCell), Paragraph("Role",sCell)],
        [Paragraph("train.csv",sCellL), Paragraph("290,300",sCell), Paragraph("145,150 / 145,150",sCell), Paragraph("~380w",sCell), Paragraph("baseline train",sCell)],
        [Paragraph("test.csv",sCellL), Paragraph("97,447",sCell), Paragraph("61,159 / 36,288",sCell), Paragraph("363w",sCell), Paragraph("held-out eval",sCell)],
        [Paragraph("HC3 (reddit_eli5<br/>+open_qa+finance)",sCellL), Paragraph("80,811",sCell), Paragraph("56,115 / 24,696",sCell), Paragraph("117w, 61% ≤150w",sCell), Paragraph("SHORT chat/casual",sCellL)],
        [Paragraph("Modern essays<br/>(GPT-4/Claude/Llama3/CmdR+)",sCellL), Paragraph("17,997",sCell), Paragraph("3,000 / 14,997",sCell), Paragraph("510w, 68% &gt;400w",sCell), Paragraph("LONG/VERYLONG 2024",sCellL)]]
TBL(rows, [48*mm, 20*mm, 32*mm, 30*mm, 40*mm]); CAPT("Table 1: Dataset scale. Held-out n=5,000 (seed 42) for all comparisons.")
FIG("class_distribution.png", 155*mm, 0.40); CAPT("Figure 1: Train is balanced; test is human-heavy (62.8%).")
FIG("length_hist.png", 155*mm, 0.40); CAPT("Figure 2: Essay length distribution (chars, 5k test). Almost no natively-short essays.")
P("New data (via <font face=\"Courier\" size=\"8\">moe/download_small_text.py</font>): <b>HC3</b> flattened to one row per answer (≥40 chars) — human vs ChatGPT to the same question, the only source of genuine short chat. <b>Modern essays</b> (artnitolog): 3,000 prompts × (human + 5 English LLMs), 2024 generators absent from the original essays. Truncation augmentation copies each train essay at 50/100/150w (short pool) and 200/300w (long pool) so short-essay style is covered. Pools are class-balanced (seed 42): short 20k, long 16.9k, verylong 11.4k; HC3/modern 15% holdouts are never trained on.")

P("<b>4 &nbsp; Methodology</b>")
P("<b>Baseline (Phase 1).</b> NFKC + whitespace + lowercase for TF-IDF (<font face=\"Courier\" size=\"8\">src/preprocessing.py</font>). Word TF-IDF (1,2, 25k, min_df 2) + char_wb (3,5, 50k, min_df 3), sublinear TF, fused via <font face=\"Courier\" size=\"8\">hstack</font> → 75k sparse features. Classifiers: MultinomialNB (α=0.1), LR (C=1.5, lbfgs), LinearSVC + 3-fold calibration (C=1.0). Validation (4k): NB 95.55%, LR 99.25% (F1 0.992, AUC 0.9987), SVM 99.53% (F1 0.995, AUC 0.9991); held-out (5k): LR 99.26%, SVM 99.58% (Table 2).")
rows = [[Paragraph("Model (val 4k)",sCell), Paragraph("Acc",sCell), Paragraph("Prec",sCell), Paragraph("Rec",sCell), Paragraph("F1",sCell), Paragraph("AUC",sCell)]]
for _, r in base_m.iterrows():
    rows.append([Paragraph(str(r["Model"]),sCellL), Paragraph(f"{r['accuracy']:.4f}",sCell), Paragraph(f"{r['precision']:.4f}",sCell),
                 Paragraph(f"{r['recall']:.4f}",sCell), Paragraph(f"{r['f1']:.4f}",sCell), Paragraph(f"{r['roc_auc']:.4f}",sCell)])
TBL(rows, [52*mm, 20*mm, 20*mm, 20*mm, 20*mm, 26*mm]); CAPT("Table 2: Baseline validation (results/baseline_ml_metrics.csv).")
FIG("model_comparison.png", 165*mm, 0.30); CAPT("Figure 3: Baseline accuracy/F1/AUC — all &gt;0.95, SVM best.")
P("<b>Short-text ablation.</b> Evaluated the frozen baseline on first-N-words and first-N-sentences truncations of the 5k held-out (and random windows — same result, so not an intro effect). At 25w accuracy 67.4% (P 0.53, R 0.99): 1,604 of 3,142 humans called AI. At 50w 83.3%, 100w 94.1%, 200w 98.4%; 1/2/3 sentences 59.8%/74.5%/83.4%. Mean LR confidence drops 0.93 (200w) → 0.79 (25w). Natively-short humans barely exist in test.csv (min human 48w), so truncated-human accuracy is the honest probe.")
P("<b>MoE.</b> Length gate ≤150 / 151–400 / &gt;400 words (cliff + median). Each expert: Word 15k + Char 25k + LR(C=1.5) on its pool only — features specialize (short vocab avoids long-essay dilution). Type gate: LR on 16 stylometric features → chat/casual/essay (7.5k balanced; HC3 open_qa→chat, eli5/finance+truncated→casual, full/modern→essay); val 80.5% (essay F1 0.91, chat 0.79, casual 0.71). Inference routes by length and reports type + warning for &lt;50w.")
FIG("moe_architecture.png", 170*mm, 0.56); CAPT("Figure 4: MoE architecture — two gates, three experts, adaptive threshold.")

P("<b>5 &nbsp; Experiments</b>")
P("Environment: Python 3.13, scikit-learn, TF-IDF as above (per-expert 15k+25k). Training (<font face=\"Courier\" size=\"8\">moe/train_moe.py</font>, 130 s): short TF-IDF 6.4 s, long 15 s, verylong 25 s. Evaluation: 5k held-out essays (full + truncations), 12,122 HC3 holdout, 2,700 modern holdout (both unseen). Calibration (<font face=\"Courier\" size=\"8\">moe/calibrate_thresholds.py</font>): 5k split 2,500 calib / 2,500 eval (seed 999), F1-max thresholds per length, eval on the other half + HC3 transfer. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrices.")

P("<b>6 &nbsp; Results</b>")
P("<b>Per-expert (val).</b> Short 97.00% (F1 0.970, AUC 0.996), long 98.19% (0.983/0.998), verylong 98.42% (0.982/0.996) (Table 3, Figure 6).")
rows = [[Paragraph("Expert",sCell), Paragraph("Train/Val",sCell), Paragraph("Acc",sCell), Paragraph("F1",sCell), Paragraph("AUC",sCell)]]
for _, r in exp_m.iterrows():
    rows.append([Paragraph(str(r["expert"]),sCell), Paragraph(f"{int(r['n_train']):,}/{int(r['n_val']):,}",sCell),
                 Paragraph(f"{r['accuracy']:.4f}",sCell), Paragraph(f"{r['f1']:.4f}",sCell), Paragraph(f"{r['roc_auc']:.4f}",sCell)])
TBL(rows, [32*mm, 38*mm, 26*mm, 26*mm, 26*mm]); CAPT("Table 3: MoE per-expert validation (results/moe_expert_metrics.csv).")
FIG("moe_experts.png", 160*mm, 0.33); CAPT("Figure 5: Per-expert metrics.")
P("<b>Routed vs baseline.</b> MoE holds full-text (98.96% vs 99.26%, −0.3 pp specialization cost; verylong bucket 99.59%) and dominates short + cross-domain (Figure 6–7, Table 4). HC3 is the headline: baseline 48.6% (it never learned chat), MoE 98.5%. Modern essays: 85.5%→95.7%.")
FIG("moe_accuracy_vs_length.png", 165*mm, 0.45); CAPT("Figure 6: Accuracy vs truncated length — gap is largest where the baseline fails.")
FIG("moe_cross_domain.png", 150*mm, 0.47); CAPT("Figure 7: Cross-domain held-outs.")
rows = [[Paragraph("Condition (5k)",sCell), Paragraph("Baseline",sCell), Paragraph("MoE",sCell), Paragraph("Δ",sCell)]]
for cond, lab in [("full","full"),("first_25_words","25w"),("first_50_words","50w"),("first_1_sent","1 sent"),("hc3_holdout","HC3"),("modern_holdout","modern")]:
    bv=float(X(moe_b,cond)["accuracy"]); mv=float(X(moe_r,cond)["accuracy"])
    rows.append([Paragraph(lab,sCell), Paragraph(f"{bv*100:.2f}%",sCell), Paragraph(f"{mv*100:.2f}%",sCell), Paragraph(f"+{(mv-bv)*100:.1f}pp",sCell)])
TBL(rows, [45*mm, 30*mm, 30*mm, 25*mm]); CAPT("Table 4: Headline deltas.")
FIG("confusion_matrices.png", 165*mm, 0.42); CAPT("Figure 8: Baseline confusion matrices (5k test).")
P("<b>Calibration.</b> Optima move with length (25w 0.32, 50w 0.40, 100w 0.56, 150w 0.64 → <font face=\"Courier\" size=\"8\">th=0.28+0.0024n</font>); long/verylong kept at 0.50. Eval-half: 25w 86.2%→88.1% (F1 0.785→0.837, recall 0.68→0.82), 150w 95.0%→96.2%, 50/100w flat, full flat. Caveat: HC3 natural short −0.6 pp (human-heavy) — thresholds are essay-tuned, so the &lt;50w warning stays. Demo flip: 31w borderline (P=0.374, th=0.354) goes Human→AI (correct-ish but low-confidence, flagged).")
rows = [[Paragraph("Eval-2500",sCell), Paragraph("th=0.50",sCell), Paragraph("Adaptive",sCell)]]
ev = cal_m[cal_m["subset"]=="eval"]
for cond, lab in [("first_25_words","25w"),("first_50_words","50w"),("first_100_words","100w"),("first_150_words","150w"),("full","full")]:
    a = ev[(ev["condition"]==cond)&(ev["threshold"]=="0.5")].iloc[0]; b = ev[(ev["condition"]==cond)&(ev["threshold"]=="calibrated")].iloc[0]
    rows.append([Paragraph(lab,sCell), Paragraph(f"{a['accuracy']:.4f}/{a['f1']:.3f}",sCell), Paragraph(f"{b['accuracy']:.4f}/{b['f1']:.3f}",sCell)])
TBL(rows, [40*mm, 50*mm, 50*mm]); CAPT("Table 5: Calibration eval (acc/F1). Detail in results/moe_calibrated_metrics.csv.")
FIG("moe_pr_flip.png", 130*mm, 0.55); CAPT("Figure 9: 25w precision-recall flip — accuracy win, tradeoff remains.")
FIG("moe_type_gate.png", 140*mm, 0.47); CAPT("Figure 10: Type gate — essay solid, casual↔chat confuse.")

P("<b>7 &nbsp; Discussion and Limitations</b>")
B(["<b>Why MoE works:</b> short text needs different evidence (word choices without discourse, chat markers) and different vocabularies; per-bucket TF-IDF prevents dilution. HC3 was the unlock — without it, there is no signal to learn.",
   "<b>Why calibration matters:</b> optimal thresholds encode the FP/FN tradeoff per regime. One threshold cannot serve both ultra-short (FN-prone) and 150w (FP-prone) text. Linear-adaptive th is a cheap, auditable compromise; per-domain thresholds would be better.",
   "<b>Limits:</b> (i) &lt;25–50w remains weak even with MoE (86–93%) — irreducible signal loss; (ii) 150w essays still prefer the baseline (97.4% vs 95.6% pre-calibration) — expert blending at boundaries is future work; (iii) short expert saw truncations ≥50w, so 25w is extrapolation; adding 10–35w training would likely help more than thresholds; (iv) type gate (80.5%) is for insight, not enforcement; casual recall 0.66 reflects real overlap.",
   "<b>Reproducibility:</b> all trims use seed 42/999; datasets via Hub JSONL/parquet need no script datasets; artifacts in <font face=\"Courier\" size=\"8\">models/moe/</font>."])

P("<b>8 &nbsp; Conclusion</b>")
P("Single-model TF-IDF detectors silently assume essay length. Once length is treated as a routing variable — with dedicated small-text data, specialized experts, a type gate, and per-length thresholds — short-text accuracy rises by 10–50 pp and cross-domain retrieval becomes viable, at a 0.3 pp full-text cost. The released <font face=\"Courier\" size=\"8\">moe/</font> stack (download → train → calibrate → predict) reproduces every number in this paper's tables and figures.")

P("<b>References</b>")
B(["[1] Guo, B. et al. How Close is ChatGPT to Human Experts? HC3. arXiv:2301.07597 (2023); Hello-SimpleAI/HC3 on Hugging Face.",
   "[2] Roy, R. et al. A Comprehensive Dataset for Human vs AI Generated Text Detection (Defactify). arXiv:2510.22874 (2025).",
   "[3] Wang, B. et al. M4: Multi-Generator, Multi-Domain, Multi-Task, Multi-Label. arXiv:2310.13607 (2023).",
   "[4] Shazeer, N. et al. Outrageously Large Neural Networks: Mixture-of-Experts. arXiv:1701.06538 (2017).",
   "[5] Hastie, T., Tibshirani, R., Friedman, J. The Elements of Statistical Learning. Springer (2009).",
   "[6] Robertson, S. &amp; Zaragoza, H. The Probabilistic Relevance Framework: BM25 &amp; TF-IDF. Foundations &amp; Trends IR (2009).",
   "[7] Reimers, N. &amp; Gurevych, A. Sentence-BERT. EMNLP (2019). Sanh, T. DistilBERT (2019); Liu, Y. RoBERTa (2019).",
   "[8] artnitolog/llm-generated-texts (GPT-4 Turbo/Omni, Claude 3 Opus, Llama3 70B, Command R+) on Hugging Face."])

P("<b>Appendices</b>")
B(["Streamlit demo: <font face=\"Courier\" size=\"8\">app/app.py</font> (baseline); MoE CLI: <font face=\"Courier\" size=\"8\">PYTHONPATH=. python3 moe/predict_moe.py \"...\"</font> (prints label, P(AI), threshold, expert, words, type, warning).",
   "Metrics: <font face=\"Courier\" size=\"8\">results/baseline_ml_metrics.csv, moe_expert_metrics.csv, moe_routed_metrics.csv, moe_baseline_comparison.csv, moe_calibrated_metrics.csv, moe_summary.json</font>.",
   "Run: <font face=\"Courier\" size=\"8\">moe/download_small_text.py → moe/train_moe.py → moe/calibrate_thresholds.py → moe/make_moe_appendix.py → make_research_paper.py</font>.",
   "Figure index: architecture (Fig.4), class (Fig.1), length (Fig.2), ablations (Fig.6–7), calibration (Fig.9), type (Fig.10). All PNGs in <font face=\"Courier\" size=\"8\">results/figures/</font>."])

doc.build(story, onFirstPage=hdrftr, onLaterPages=hdrftr)
print(f"Saved -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
