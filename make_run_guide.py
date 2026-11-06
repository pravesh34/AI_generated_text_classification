"""
make_run_guide.py — Build Code Walkthrough + Run Guide PDF
Output: Code_Workflow_and_Run_Guide.pdf
Covers: architecture, file-by-file walkthrough, end-to-end workflow, how to run (train/test/API/LangChain/Streamlit), troubleshooting.
"""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("results/figures", exist_ok=True)

# --- Simple workflow diagrams with matplotlib ---
def draw_workflow(path, blocks, arrows_title=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    n = len(blocks)
    y = 0.85
    for i, (title, desc, color) in enumerate(blocks):
        box = FancyBboxPatch((0.08, y-0.13), 0.84, 0.13, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#1F2A44", linewidth=1.2)
        ax.add_patch(box)
        ax.text(0.5, y-0.045, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2A44")
        ax.text(0.5, y-0.095, desc, ha="center", va="center", fontsize=7.5, color="#333333")
        if i < n-1:
            ax.annotate("", xy=(0.5, y-0.15), xytext=(0.5, y-0.155), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4C72B0"))
        y -= 0.16
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    if arrows_title: ax.set_title(arrows_title, fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

draw_workflow("results/figures/flow_train.png", [
    ("1. Load data/train.csv (290k) + data/test.csv (97k)", "text, generated  •  mean 2,241 chars", "#EAF0F8"),
    ("2. Balanced sample 10k+10k = 20k", "random_state=42", "#D6E4F5"),
    ("3. Stratified split 16k train / 4k val", "50% AI in both", "#EAF0F8"),
    ("4. Preprocess: NFKC + lowercase", "clean_text_for_tfidf()", "#D6E4F5"),
    ("5. Word TF-IDF (1,2) 25k + Char TF-IDF char_wb (3,5) 50k", "hstack → (16000, 75000) sparse", "#EAF0F8"),
    ("6. Train NB / Logistic Regression / Calibrated Linear SVM", "C=1.5 / C=1.0, cv=3", "#D6E4F5"),
    ("7. Evaluate val (4k) + held-out test (5k) → save models/*.pkl", "Acc 99%+, AUC 0.999", "#EAF0F8"),
], "TRAINING WORKFLOW — train_baseline.py (11 steps, 33s)")

draw_workflow("results/figures/flow_infer.png", [
    ("User input: --text / .txt file / folder / CSV / API POST / LangChain tool", "any length essay", "#FDF2E9"),
    ("test/inference.py: load_artifacts() → models/*.pkl", "word_vec + char_vec + lr/svm (75k features)", "#FDEBD0"),
    ("vectorize() + predict_proba() → P(AI)", "threshold 0.5 → Human / AI", "#FDF2E9"),
    ("Outputs: terminal table / predictions.csv / metrics.json / API JSON / Streamlit UI", "batch 1000 by 1000 (no OOM)", "#FDEBD0"),
], "INFERENCE WORKFLOW — test/inference.py + predict.py / api.py / Streamlit")

# --- Build PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable, Preformatted
from reportlab.lib import colors

OUT="Code_Workflow_and_Run_Guide.pdf"
NAVY=HexColor("#1F2A44"); ACCENT=HexColor("#4C72B0"); LIGHT=HexColor("#EAF0F8")
styles=getSampleStyleSheet()
sTitle=ParagraphStyle("T", parent=styles["Title"], fontSize=22, leading=28, textColor=NAVY, alignment=TA_CENTER)
sH1=ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=17, textColor=NAVY, spaceBefore=14, spaceAfter=6)
sH2=ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
sBody=ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=13, alignment=TA_JUSTIFY, spaceAfter=6)
sB=ParagraphStyle("Bullet", parent=sBody, leftIndent=14, bulletIndent=6, spaceAfter=3)
sCell=ParagraphStyle("C", parent=styles["Normal"], fontSize=7.5, leading=10, alignment=1)
sCellL=ParagraphStyle("CL", parent=sCell, alignment=0)
sCap=ParagraphStyle("Cap", parent=styles["Normal"], fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.gray)
sCode=ParagraphStyle("Code", parent=styles["Code"], fontSize=7, leading=9, leftIndent=6)
def hf(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica",7); canvas.setFillColor(colors.gray)
    canvas.drawCentredString(A4[0]/2, 12*mm, "Code Walkthrough & Run Guide — Human vs AI Text Classification (NIT Sikkim, 2026)")
    canvas.drawRightString(A4[0]-15*mm, 12*mm, f"Page {doc.page}"); canvas.restoreState()
def cover_hf(canvas, doc):
    if doc.page==1:
        canvas.saveState(); canvas.setFont("Helvetica",7); canvas.setFillColor(colors.gray)
        canvas.drawCentredString(A4[0]/2, 12*mm, "NIT Sikkim — CSE Course Project (2026)"); canvas.restoreState()
    else: hf(canvas, doc)
doc=SimpleDocTemplate(os.path.abspath(OUT), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=18*mm, bottomMargin=16*mm,
                      title="Code Walkthrough and Run Guide", author="Pravesh Gedon Sharma & Ankit Rawat")
story=[]
story.append(Spacer(1,28*mm))
story.append(Paragraph("Code Walkthrough<br/>&amp; Run Guide", sTitle))
story.append(Spacer(1,5*mm))
story.append(Paragraph("Classification of Human-Written and AI-Generated Text<br/>National Institute of Technology Sikkim — CSE", ParagraphStyle("s", parent=sBody, alignment=TA_CENTER, fontSize=10)))
story.append(HRFlowable(width="60%", thickness=1.2, color=ACCENT, spaceAfter=6, spaceBefore=6, hAlign="CENTER"))
story.append(Paragraph("Understand the workflow end-to-end • How to train, test, and serve the model in 6 ways<br/>Dual TF-IDF (75k features) • Logistic Regression &amp; Linear SVM — 99%+ test accuracy", ParagraphStyle("s2", parent=sBody, alignment=TA_CENTER, fontSize=8.5)))
story.append(Spacer(1,12*mm))
story.append(Paragraph("Authors: <b>Pravesh Gedon Sharma (B240019)</b> &amp; <b>Ankit Rawat (B320032)</b> • This PDF pairs with <font face=\"Courier\" size=\"7\">Code_Workflow_and_Run_Guide.pdf</font> (this file) and <font face=\"Courier\" size=\"7\">Project_Report_AI_Text_Classification.pdf</font> (results).", sBody))
story.append(PageBreak())

def H1(t): story.append(Paragraph(t, sH1))
def H2(t): story.append(Paragraph(t, sH2))
def P(t): story.append(Paragraph(t, sBody))
def BUL(items):
    for i in items: story.append(Paragraph(i, sB, bulletText="•"))
def CODE(t): story.append(Preformatted(textwrap.dedent(t).strip("\n"), sCode)); story.append(Spacer(1,2*mm))
def FIG(p, w=160*mm):
    story.append(Image(os.path.abspath(p), width=w, height=w*0.55, kind="proportional")); story.append(Spacer(1,2*mm))
def CAP(t): story.append(Paragraph(t, sCap))
def TBL(rows, widths):
    t=Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),LIGHT),("BOX",(0,0),(-1,-1),0.5,ACCENT),("INNERGRID",(0,0),(-1,-1),0.35,colors.gray),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(t); story.append(Spacer(1,2*mm))

H1("Contents")
for l in ["1 Overview & Architecture","2 Repository Map (file by file)","3 src/ — Shared Utilities","4 Training Workflow (train_baseline.py, 11 steps)","5 Inference Workflow (test/inference.py)","6 The 6 Ways to Run It (CLI, Folder, CSV, Terminal, API, LangChain, Streamlit)","7 How to Run — Copy-Paste Commands","8 Outputs & What Each File Means","9 Troubleshooting & FAQ","Appendix A: Key Snippets (load → vectorize → predict)"]:
    P(f"&nbsp;&nbsp;&nbsp;&nbsp;{l}")

H1("1 &nbsp; Overview & Architecture")
P("The project turns an essay into a 75,000-dimensional sparse vector (25k word TF-IDF + 50k char TF-IDF) and asks a linear classifier: <b>Human or AI?</b>. Nothing else. That simplicity is the point — it trains in 33 seconds, runs on a laptop, exposes a REST API, a LangChain tool, a CLI, and a web UI, all from the same 4 <font face=\"Courier\" size=\"7\">.pkl</font> files.")
FIG("results/figures/flow_train.png"); CAP("Figure 1: Training flow.")
FIG("results/figures/flow_infer.png"); CAP("Figure 2: Inference flow (all 6 front-ends share test/inference.py).")
BUL(["<b>Training (once):</b> <font face=\"Courier\" size=\"7\">train_baseline.py</font> → <font face=\"Courier\" size=\"7\">models/*.pkl</font> + <font face=\"Courier\" size=\"7\">results/baseline_ml_metrics.csv</font>",
     "<b>Inference (every time):</b> <font face=\"Courier\" size=\"7\">test/inference.py</font> loads the 4 pkls → <font face=\"Courier\" size=\"7\">predict.py</font> / <font face=\"Courier\" size=\"7\">api.py</font> / <font face=\"Courier\" size=\"7\">langchain_tool.py</font> / <font face=\"Courier\" size=\"7\">app/app.py</font> (Streamlit) → terminal answer + optional CSV/JSON.",
     "<b>Data:</b> <font face=\"Courier\" size=\"7\">data/train.csv</font> 290,300 rows (balanced) / <font face=\"Courier\" size=\"7\">data/test.csv</font> 97,447 rows. Columns: <font face=\"Courier\" size=\"7\">text, generated</font> (0=Human, 1=AI)."])

H1("2 &nbsp; Repository Map (file by file)")
TBL([
 [Paragraph("File",sCell), Paragraph("What it does",sCell), Paragraph("Lines",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">train_baseline.py</font>",sCellL), Paragraph("Full training pipeline (11 logged steps). Load → sample → split → TF-IDF → train NB/LR/SVM → evaluate → save.",sCell), Paragraph("142",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">make_report.py</font>",sCellL), Paragraph("Generates Project_Report_*.pdf (results + 5 figures) from live artifacts.",sCell), Paragraph("319",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">make_run_guide.py</font>",sCellL), Paragraph("Generates THIS guide PDF.",sCell), Paragraph("this",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">src/preprocessing.py</font>",sCellL), Paragraph("clean_text_for_tfidf() — NFKC + lowercase for TF-IDF.",sCell), Paragraph("45",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">src/stylometrics.py</font>",sCellL), Paragraph("TTR, burstiness, Flesch ease/grade, punctuation rates — used by Streamlit + tests.",sCell), Paragraph("162",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">src/evaluate.py</font>",sCellL), Paragraph("compute_classification_metrics(), plot_confusion_matrix(), plot_roc_curves().",sCell), Paragraph("95",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">src/semantic.py</font>",sCellL), Paragraph("Sentence-BERT embeddings + cosine — Phase 2 (not yet wired into inference).",sCell), Paragraph("80",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">app/app.py</font>",sCellL), Paragraph("Streamlit web UI (:8501). Loads models/ with @cache_resource, shows P(AI) + stylometrics.",sCell), Paragraph("110",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/inference.py</font>",sCellL), Paragraph("<b>Core inference engine.</b> load_artifacts(), vectorize(), predict_text/batch, evaluate_csv(), predict_folder().",sCell), Paragraph("158",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/predict.py</font>",sCellL), Paragraph("CLI: --text / --input / --folder / --csv / --interactive. Args → inference.py.",sCell), Paragraph("140",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/api.py</font>",sCellL), Paragraph("FastAPI (:8001): POST /predict, /predict/batch, /langchain/tool, GET /health, /model/info.",sCell), Paragraph("95",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/langchain_tool.py</font>",sCellL), Paragraph("LangChain BaseTool ai_human_classifier + batch tool. Agent-ready.",sCell), Paragraph("86",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"7\">test/terminal_demo.py</font>",sCellL), Paragraph("Terminal demo comparing direct vs langchain vs API in one run (+ --loop chat).",sCell), Paragraph("81",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/evaluate_examples.py</font>",sCellL), Paragraph("Reads all test/example_texts/*.txt, predicts both models, prints aggregate metrics.",sCell), Paragraph("62",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/test_model.py</font>",sCellL), Paragraph("Smoke tests: single predict + 2k eval (assert acc>0.90). --full = 97k.",sCell), Paragraph("33",sCell)],
], [40*mm, 95*mm, 15*mm])

H1("3 &nbsp; src/ — Shared Utilities")
H2("preprocessing.py")
P("<font face=\"Courier\" size=\"7\">clean_text_for_tfidf(text)</font> = NFKC → collapse whitespace → lowercase. Used as <font face=\"Courier\" size=\"7\">preprocessor=</font> in word TF-IDF so train and test are normalised identically. <font face=\"Courier\" size=\"7\">clean_text_for_stylometrics()</font> keeps casing/punctuation (capital ratio, commas) — never lowercases.")
H2("stylometrics.py")
P("<font face=\"Courier\" size=\"7\">extract_stylometric_features(text) → 24 floats</font> (num_words, ttr, hapax_ratio, mean_sentence_length, burstiness=σ/μ, comma_rate… flesch_reading_ease, flesch_kincaid_grade, gunning_fog). No sklearn — pure regex + syllable heuristics. Shown in Streamlit and returned by <font face=\"Courier\" size=\"7\">--stylometrics</font> / API <font face=\"Courier\" size=\"7\">stylometrics:true</font>.")
H2("evaluate.py / semantic.py")
P("<font face=\"Courier\" size=\"7\">evaluate.py</font> provides notebook plots (confusion + ROC) and metric dict. <font face=\"Courier\" size=\"7\">semantic.py</font> (Sentence-BERT) is Phase-2 scaffolding — researched but <b>not</b> part of the 75k TF-IDF inference path.")

H1("4 &nbsp; Training Workflow (train_baseline.py, 11 steps, 33 s)")
P("Run: <font face=\"Courier\" size=\"7\">venv/bin/python train_baseline.py</font>. Each step prints a timestamped log line — watch them in order:")
TBL([
 [Paragraph("Step",sCell), Paragraph("What happens",sCell), Paragraph("Output",sCell)],
 [Paragraph("1",sCell), Paragraph("Load data/train.csv (290,300) + data/test.csv (97k metadata). Prints shape, class counts, length stats.",sCell), Paragraph("train (290300,2)",sCell)],
 [Paragraph("2",sCell), Paragraph("Balanced sample: per-class <font face=\"Courier\" size=\"6\">sample(10000)</font> → concat → shuffle. Why 20k? Full 290k×75k TF-IDF is 14× slower; notebook 03 defines 20k.",sCell), Paragraph("(20000,2) 50/50",sCell)],
 [Paragraph("3",sCell), Paragraph("Stratified split (random_state=42, test_size=0.20) → 16k train / 4k val, both 50% AI.",sCell), Paragraph("train=16000 val=4000",sCell)],
 [Paragraph("4",sCell), Paragraph("Preprocessing demo: prints raw vs cleaned (lowercase) for 2 rows.",sCell), Paragraph("clean_text_for_tfidf",sCell)],
 [Paragraph("5",sCell), Paragraph("Word TF-IDF (1,2) 25k, min_df=2, sublinear_tf. fit on train, transform val.",sCell), Paragraph("(16000,25000) in 6.2s",sCell)],
 [Paragraph("6",sCell), Paragraph("Char TF-IDF char_wb (3,5) 50k, min_df=3. fit/transform.",sCell), Paragraph("(16000,50000) in 15.8s",sCell)],
 [Paragraph("7",sCell), Paragraph("Fusion hstack(word,char).tocsr(). Prints nnz (non-zero cells).",sCell), Paragraph("(16000,75000) nnz 34.8M",sCell)],
 [Paragraph("8",sCell), Paragraph("Train NB / LR (C=1.5) / LinearSVC+CalibratedCV(3). Each evals on 4k val: accuracy, precision, recall, F1, ROC-AUC, confusion, full report.",sCell), Paragraph("SVM 99.53% acc",sCell)],
 [Paragraph("9",sCell), Paragraph("Save results/baseline_ml_metrics.csv (Model,accuracy,…).",sCell), Paragraph("results/*.csv",sCell)],
 [Paragraph("10",sCell), Paragraph("Held-out test: load test.csv, sample 5000 (seed 42), transform with frozen vectorizers, LR+SVM eval.",sCell), Paragraph("LR 99.26% SVM 99.58%",sCell)],
 [Paragraph("11",sCell), Paragraph("Save 4 pkls to models/. Prints TOTAL TIME 33s.",sCell), Paragraph("models/*.pkl",sCell)],
], [12*mm, 85*mm, 40*mm])
P("<b>Two pandas-3 fixes</b> are baked in (your run hit them): per-class sampling instead of <font face=\"Courier\" size=\"7\">groupby.apply</font> (which drops <font face=\"Courier\" size=\"7\">generated</font>), and <font face=\"Courier\" size=\"7\">np.array(..., dtype=object)</font> for Arrow-backed strings before sklearn split.")

H1("5 &nbsp; Inference Workflow (test/inference.py)")
P("<font face=\"Courier\" size=\"7\">test/inference.py</font> is the <b>one place</b> that touches models. Every front-end (CLI, API, LangChain, Streamlit) calls it — change inference once, all 6 ways update.")
TBL([
 [Paragraph("Function",sCell), Paragraph("One-liner",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">load_artifacts()</font>",sCellL), Paragraph("joblib.load ×4 from ../models/. Raises FileNotFoundError with 'Train first' hint if missing.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">vectorize(texts, art)</font>",sCellL), Paragraph("word.transform + char.transform → hstack().tocsr(). Single str auto-wrapped to list.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">predict_proba(texts, art, model)</font>",sCellL), Paragraph("clf.predict_proba(X)[:,1] → P(AI) array. model='lr'|'svm'.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">predict_text(text, art, model, stylometrics)</font>",sCellL), Paragraph("Single-text dict: predicted_label (≥0.5→AI), probability_ai/human, confidence, optional 24 stylometrics.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">predict_batch(texts, batch_size=1000)</font>",sCellL), Paragraph("Chunked to avoid OOM on 97k/290k CSVs. Returns np.array of P(AI).",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">evaluate_csv(csv, sample_n)</font>",sCellL), Paragraph("Loads CSV, batches predict, computes accuracy/precision/recall/F1/AUC + confusion + report, returns (metrics, df with prob_ai/pred/correct).",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">predict_folder(folder)</font>",sCellL), Paragraph("glib *.txt/*.md, predict_text each + stylometrics + file path.",sCell)],
], [45*mm, 95*mm])

H1("6 &nbsp; The 6 Ways to Run It")
TBL([
 [Paragraph("#",sCell), Paragraph("Way",sCell), Paragraph("Command / Endpoint",sCell), Paragraph("Best for",sCell)],
 [Paragraph("1",sCell), Paragraph("CLI single / file",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">test/predict.py --text \"...\" / --input file.txt</font>",sCell), Paragraph("One-off classify",sCell)],
 [Paragraph("2",sCell), Paragraph("CLI folder / CSV",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">--folder test/example_texts</font> / <font face=\"Courier\" size=\"6\">--csv data/test.csv --sample 5000</font>",sCell), Paragraph("Batch eval",sCell)],
 [Paragraph("3",sCell), Paragraph("Terminal chat",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">test/terminal_demo.py --loop</font> or <font face=\"Courier\" size=\"6\">predict.py --interactive</font>",sCell), Paragraph("Try text live",sCell)],
 [Paragraph("4",sCell), Paragraph("REST API",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">test/api.py</font> → POST <font face=\"Courier\" size=\"6\">/predict</font> (8001)",sCell), Paragraph("Other programs / curl",sCell)],
 [Paragraph("5",sCell), Paragraph("LangChain tool",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">langchain_tool.ai_human_classifier.run(...)</font>",sCell), Paragraph("LLM agents",sCell)],
 [Paragraph("6",sCell), Paragraph("Web UI",sCellL), Paragraph("<font face=\"Courier\" size=\"6\">app/app.py</font> (Streamlit :8501)",sCell), Paragraph("Demo to humans",sCell)],
], [10*mm, 30*mm, 58*mm, 30*mm])

H1("7 &nbsp; How to Run — Copy-Paste Commands (from project root)")
H2("A. First-time setup (once)")
CODE("""
cd ~/Desktop/AI_generated_text_classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "langchain-core>=0.3" "fastapi>=0.110" reportlab
# data: ensure data/train.csv (623MB) + data/test.csv (212MB) exist
ls -lh data/
""")
H2("B. Train (once — 33 seconds with data present)")
CODE("""
venv/bin/python train_baseline.py
# → models/word_tfidf_vectorizer.pkl (969K)
#   models/char_tfidf_vectorizer.pkl (1.6M)
#   models/logistic_regression_baseline.pkl (587K)
#   models/linear_svm_baseline.pkl (1.7M)
#   results/baseline_ml_metrics.csv
""")
H2("C. Test / Inference — all 6 ways")
CODE("""
# 1 — single text (terminal)
venv/bin/python test/predict.py --text "Your essay here..." --model lr --stylometrics

# 2 — file / folder (13 examples in test/example_texts/)
venv/bin/python test/predict.py --input test/example_texts/human_essay_school.txt --model svm
venv/bin/python test/predict.py --folder test/example_texts --model lr
venv/bin/python test/evaluate_examples.py          # custom evaluator (both models, aggregate)

# 3 — terminal chat (no browser)
venv/bin/python test/terminal_demo.py
venv/bin/python test/terminal_demo.py --api        # needs API running (see D)
venv/bin/python test/predict.py --interactive --model lr

# 4 — REST API (server, second terminal)
venv/bin/python test/api.py                        # → http://localhost:8001/docs
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \\
  -d '{"text":"In todays rapidly evolving digital landscape...","model":"lr","stylometrics":true}'

# 5 — LangChain tool (inside Python / agent)
venv/bin/python test/langchain_tool.py             # no-LLM demo
#   in agent: from test.langchain_tool import ai_human_classifier

# 6 — Web UI (second terminal)
venv/bin/streamlit run app/app.py --server.port 8501   # → http://localhost:8501
""")
H2("D. Full evaluation on whole files (batch inference)")
CODE("""
venv/bin/python test/predict.py --csv data/test.csv --model lr --out test/results           # 97k rows
venv/bin/python test/predict.py --csv data/train.csv --model svm --out test/results_train_full # 290k rows
venv/bin/python test/test_model.py              # 2k-row smoke test
venv/bin/python test/test_model.py --full       # full 97k
""")
H2("E. Report PDF")
CODE("""
venv/bin/python make_report.py        # → Project_Report_AI_Text_Classification.pdf (361KB)
venv/bin/python make_run_guide.py     # → Code_Workflow_and_Run_Guide.pdf (this file)
""")

H1("8 &nbsp; Outputs & What Each File Means")
TBL([
 [Paragraph("File",sCell), Paragraph("What it is",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">models/*.pkl (4)</font>",sCellL), Paragraph("Frozen vectorizers + classifiers. The ONLY things inference loads. Retrain to replace.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">results/baseline_ml_metrics.csv</font>",sCellL), Paragraph("Val metrics per model (train_baseline.py step 9).",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/results/predictions.csv</font>",sCellL), Paragraph("Per-row: text, generated, prob_ai, pred, correct. Review errors here.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/results/metrics.json</font>",sCellL), Paragraph("Same metrics as terminal + confusion matrix + full classification report.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/example_texts/</font>",sCellL), Paragraph("13 .txt examples (human_/ai_ prefix = ground truth). <font face=\"Courier\" size=\"6\">evaluate_examples.py</font> scores them.",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">test/example_test_set.csv</font>",sCellL), Paragraph("20-row balanced CSV for quick batch eval (--csv test/example_test_set.csv).",sCell)],
 [Paragraph("<font face=\"Courier\" size=\"6\">results/figures/*.png</font>",sCellL), Paragraph("5 figures for the PDF (class dist, length, comparison, confusion, ROC).",sCell)],
], [50*mm, 90*mm])

H1("9 &nbsp; Troubleshooting & FAQ")
BUL(["<b>FileNotFoundError: models/*.pkl missing</b> → you haven't trained yet. Run <font face=\"Courier\" size=\"7\">venv/bin/python train_baseline.py</font> (needs <font face=\"Courier\" size=\"7\">data/train.csv</font>).",
     "<b>KeyError 'generated' / Arrow TypeError</b> on pandas 3 → already fixed in train_baseline.py (per-class sampling + np.array dtype=object). Do not revert to groupby.apply.",
     "<b>API 'Connection refused' on 8001</b> → API not running. Start <font face=\"Courier\" size=\"7\">venv/bin/python test/api.py</font> in a <b>separate terminal</b> (keep it open).",
     "<b>uvicorn test.api:app fails</b> → <font face=\"Courier\" size=\"7\">test/</font> needs <font face=\"Courier\" size=\"7\">__init__.py</font> (added) OR run <font face=\"Courier\" size=\"7\">python test/api.py</font> directly (recommended).",
     "<b>Short text predicted wrong</b> (e.g. human_sample.txt 26 words → AI) → expected. Model is trained on ~400-word essays (mean 2,241 chars). Very short/clean text is out-of-distribution. Test with essay-length paths.",
     "<b>Streamlit not showing new model</b> → kill old process (<font face=\"Courier\" size=\"7\">pkill -f \"streamlit run\"</font>) then restart — @cache_resource caches artifacts.",
     "<b>OOM on full train.csv</b> → don't load whole CSV in memory; use <font face=\"Courier\" size=\"7\">--csv ... --sample 5000</font> or batch_size=1000 in <font face=\"Courier\" size=\"7\">evaluate_csv()</font>."])

H1("Appendix A &nbsp; Key Snippets (load → vectorize → predict)")
P("This is the entire inference in 10 lines — everything else is wrapper around it:")
CODE("""
import joblib
from scipy.sparse import hstack

word_vec = joblib.load("models/word_tfidf_vectorizer.pkl")
char_vec = joblib.load("models/char_tfidf_vectorizer.pkl")
lr       = joblib.load("models/logistic_regression_baseline.pkl")

text = open("test/example_texts/ai_essay_distance_learning.txt").read()
X = hstack([word_vec.transform([text]), char_vec.transform([text])]).tocsr()
p_ai = lr.predict_proba(X)[0, 1]
print("AI-Generated" if p_ai >= 0.5 else "Human-Written", f"P(AI)={p_ai:.4f}")
# → AI-Generated P(AI)=0.9856
""")
P("Higher-level: <font face=\"Courier\" size=\"7\">from test.inference import load_artifacts, predict_text; r = predict_text(text, load_artifacts(), model='lr')</font>; REST: <font face=\"Courier\" size=\"7\">POST /predict</font>; LangChain: <font face=\"Courier\" size=\"7\">ai_human_classifier.run({'text': text})</font>.")
P("<i>Generated automatically by <font face=\"Courier\" size=\"7\">make_run_guide.py</font> from live artifacts.</i>")

doc.build(story, onFirstPage=cover_hf, onLaterPages=hf)
print(f"Saved -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
