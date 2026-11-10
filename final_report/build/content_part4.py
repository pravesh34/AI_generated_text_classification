"""Part IV — Code Walkthrough & Reproducibility Guide.

Reconstructed from make_run_guide.py with current repository line counts,
the six front-ends, run commands, outputs, troubleshooting, a
reproducibility checklist and two appendices (snippets + deployment).
"""
from blocks import (part, h1, h2, h3, p, bullets, numbered, fig, table, code,
                    pagebreak, spacer, refs)


def _pc(x, d=2):
    return f"{x * 100:.{d}f}%"


def part4(M):
    base = M["base"]
    routed = M["routed"]

    blocks = [
        part("Part IV — Code Walkthrough & Reproducibility Guide",
             "Architecture, file-by-file repository map, the eleven-step training "
             "workflow, the six inference front-ends, copy-paste commands, "
             "troubleshooting and a full reproducibility checklist"),

        h1("1  Overview and Architecture"),
        p("The project turns an essay into a 75,000-dimensional sparse vector "
          "(25,000 word TF-IDF + 50,000 character TF-IDF) and asks a linear "
          "classifier: <b>Human or AI?</b>. That simplicity is the point — the "
          "baseline trains in 33 seconds on a laptop CPU, and the same four "
          "<code>.pkl</code> artifacts feed every interface the system exposes: "
          "a REST API, a LangChain tool, a CLI, a terminal demo, a Streamlit UI "
          "and two cloud deployments. The Mixture-of-Experts layer "
          "(<code>moe/</code>) adds three length-specialised models behind a "
          "deterministic router without changing that contract: one inference "
          "core, many front-ends."),
        fig("flow_train", 0.92, repeat=True),
        fig("flow_infer", 0.88, repeat=True),
        bullets([
            "<b>Training (once):</b> <code>train_baseline.py</code> → "
            "<code>models/*.pkl</code> + <code>results/baseline_ml_metrics.csv</code>; "
            "<code>moe/train_moe.py</code> → <code>models/moe/*.pkl</code> + "
            "<code>results/moe_*.csv</code>.",
            "<b>Inference (every time):</b> <code>test/inference.py</code> loads "
            "the pickles once; <code>predict.py</code>, <code>api.py</code>, "
            "<code>langchain_tool.py</code>, <code>app/app.py</code>, "
            "<code>moe/predict_moe.py</code> and the Vercel function all call the "
            "same core.",
            "<b>Data:</b> <code>data/train.csv</code> 290,300 rows (balanced) and "
            "<code>data/test.csv</code> 97,447 rows; columns <code>text, "
            "generated</code> (0 = Human, 1 = AI), plus "
            "<code>data/moe/hc3_small.csv</code> (80,811) and "
            "<code>data/moe/modern_essays.csv</code> (17,997).",
        ]),

        # ==================================================================
        h1("2  Repository Map (file by file)"),
        p("Line counts below are from the released repository at the time of "
          "writing; the two document generators that produced the source PDFs for "
          "this report are listed for completeness."),
        table(
            "p4_map",
            "Repository map — what each file does and how large it is.",
            ["File", "What it does", "Lines"],
            [
                ["<code>train_baseline.py</code>",
                 "Full baseline training pipeline (11 logged steps): load → sample "
                 "→ split → TF-IDF → train NB / LR / SVM → evaluate → save.",
                 "142"],
                ["<code>src/preprocessing.py</code>",
                 "<code>clean_text_for_tfidf()</code> — NFKC + lowercase, used as "
                 "<code>preprocessor=</code> so train and test normalise "
                 "identically.", "45"],
                ["<code>src/stylometrics.py</code>",
                 "24 stylometric features (TTR, hapax, burstiness, Flesch ease / "
                 "grade, punctuation rates) — Streamlit evidence cards, type-gate "
                 "features, tests.", "162"],
                ["<code>src/evaluate.py</code>",
                 "<code>compute_classification_metrics()</code>, "
                 "<code>plot_confusion_matrix()</code>, "
                 "<code>plot_roc_curves()</code>.", "95"],
                ["<code>src/semantic.py</code>",
                 "Sentence-BERT embeddings + cosine similarity — Phase-2 "
                 "scaffolding, <b>not</b> wired into the inference path.", "80"],
                ["<code>test/inference.py</code>",
                 "<b>Core inference engine.</b> <code>load_artifacts()</code>, "
                 "<code>vectorize()</code>, <code>predict_text/batch()</code>, "
                 "<code>evaluate_csv()</code>, <code>predict_folder()</code>.",
                 "158"],
                ["<code>test/predict.py</code>",
                 "CLI: <code>--text / --input / --folder / --csv / --interactive</code>.",
                 "140"],
                ["<code>test/api.py</code>",
                 "FastAPI on :8001 — <code>POST /predict</code>, "
                 "<code>/predict/batch</code>, <code>/langchain/tool</code>; "
                 "<code>GET /health</code>, <code>/model/info</code>.", "95"],
                ["<code>test/langchain_tool.py</code>",
                 "LangChain <code>BaseTool</code> <code>ai_human_classifier</code> "
                 "+ batch tool — agent-ready.", "86"],
                ["<code>test/terminal_demo.py</code>",
                 "Terminal demo comparing direct vs LangChain vs API paths "
                 "(<code>--loop</code> chat mode).", "81"],
                ["<code>test/evaluate_examples.py</code>",
                 "Predicts all 13 <code>test/example_texts/*.txt</code> with both "
                 "models and prints aggregate metrics.", "62"],
                ["<code>test/test_model.py</code>",
                 "Smoke tests: single predict + 2k evaluation (asserts accuracy "
                 "&gt; 0.90); <code>--full</code> runs all 97k.", "33"],
                ["<code>app/app.py</code>",
                 "Streamlit demonstration UI (:8501) — five pages, four inference "
                 "modes, live model status, cached artifact loading.", "546"],
                ["<code>moe/train_moe.py</code>",
                 "Builds the three pools, fits per-bucket TF-IDF + LR experts, "
                 "trains the type gate, runs routed evaluation.", "386"],
                ["<code>moe/calibrate_thresholds.py</code>",
                 "Length-swept F1-optimal thresholds on the 2,500-essay "
                 "calibration half (seed 999) → "
                 "<code>models/moe/thresholds.json</code>.", "173"],
                ["<code>moe/predict_moe.py</code>",
                 "Routed MoE CLI — prints label, P(AI), expert, threshold, word "
                 "count, type and warning.", "85"],
                ["<code>moe/type_features.py</code>",
                 "16 type-gate features + truncation helpers "
                 "(<code>truncate_words</code>, <code>random_window</code>).",
                 "127"],
                ["<code>moe/download_small_text.py</code>",
                 "HC3 + modern-LLM download and flattening (Hub JSONL/parquet, no "
                 "dataset scripts).", "118"],
                ["<code>web/api/predict.py</code>",
                 "Vercel serverless predictor serving the static site and "
                 "<code>POST /api/predict</code>.", "269"],
                ["<code>web/index.html</code>",
                 "Project site deployed to Vercel and GitHub Pages.", "203"],
                ["<code>run_all.sh</code>",
                 "One-command setup → train → test → API (:8001) → UI (:8501) with "
                 "health checks.", "90"],
                ["<code>make_report.py</code>",
                 "Generates the original Part-I results report (5 figures) from "
                 "live artifacts.", "319"],
                ["<code>moe/make_moe_appendix.py</code>",
                 "Generates the original MoE appendix PDF (5 figures) from "
                 "<code>results/moe_*.csv</code>.", "276"],
                ["<code>make_research_paper.py</code>",
                 "Generates the original conference paper (Part II source).",
                 "179"],
                ["<code>make_deepseek_paper.py</code>",
                 "Generates the original deep-dive paper (Part III source).",
                 "291"],
            ],
            widths=[44, 106, 16],
            left_cols=(0,),
        ),
        p("Supporting directories: <code>models/</code> (4 baseline pickles) and "
          "<code>models/moe/</code> (<code>expert_short.pkl</code>, "
          "<code>expert_long.pkl</code>, <code>expert_verylong.pkl</code>, "
          "<code>type_gate.pkl</code>, <code>thresholds.json</code>, "
          "<code>gate_config.json</code>); <code>results/</code> (metrics CSVs, "
          "figures); <code>notebooks/</code> (six EDA / feature / transformer / "
          "ablation notebooks); <code>test/example_texts/</code> (13 labelled "
          "examples); <code>docs/</code> (GitHub Pages site and report PDF)."),

        # ==================================================================
        h1("3  src/ — Shared Utilities"),
        h2("3.1  preprocessing.py"),
        p("<code>clean_text_for_tfidf(text)</code> = NFKC normalisation → "
          "whitespace collapse → lowercase. It is passed as "
          "<code>preprocessor=</code> to the word TF-IDF vectorizer, so training "
          "and inference normalise identically — a single source of truth. "
          "<code>clean_text_for_stylometrics()</code> is the deliberate "
          "counterpart: it preserves casing and punctuation because capital "
          "ratio and comma rate are features, and never lowercases."),
        h2("3.2  stylometrics.py"),
        p("<code>extract_stylometric_features(text)</code> returns 24 floats "
          "(word count, TTR, hapax ratio, mean sentence length, burstiness "
          "σ/μ, comma and semicolon rates, Flesch Reading Ease, Flesch–Kincaid "
          "grade, Gunning Fog, …). It is pure regex + syllable heuristics with "
          "no scikit-learn dependency, which is why the type gate, the "
          "Streamlit evidence cards and the CLI <code>--stylometrics</code> flag "
          "can all afford it on every call [6]."),
        h2("3.3  evaluate.py and semantic.py"),
        p("<code>evaluate.py</code> provides the metric dictionary and the "
          "confusion / ROC plots used by the notebooks. <code>semantic.py</code> "
          "(Sentence-BERT + cosine similarity) is Phase-2 scaffolding: researched "
          "and shipped, but <b>not</b> part of the 75k TF-IDF inference path — "
          "no score in this report comes from it."),

        # ==================================================================
        h1("4  Training Workflow (train_baseline.py — 11 steps, 33 s)"),
        p("Run <code>venv/bin/python train_baseline.py</code> with the data "
          "present. Each step prints a timestamped log line; "
          "{{f:flow_train}} is the same flow as a diagram. {{t:p4_steps}} gives "
          "the step-by-step contract and the outputs to expect."),
        table(
            "p4_steps",
            "The eleven logged steps of train_baseline.py.",
            ["Step", "What happens", "Output"],
            [
                ["1", "Load <code>data/train.csv</code> (290,300) + "
                      "<code>data/test.csv</code> (97k). Prints shape, class "
                      "counts and length statistics.",
                 "train (290300, 2)"],
                ["2", "Balanced sample: per-class "
                      "<code>sample(10000)</code> → concat → shuffle "
                      "(random_state 42). 20k because 290k × 75k TF-IDF is ~14× "
                      "slower; notebook 03 defines the budget.",
                 "(20000, 2) 50/50"],
                ["3", "Stratified split (<code>test_size=0.20</code>, seed 42) → "
                      "16k train / 4k validation, both 50% AI.",
                 "train = 16000, val = 4000"],
                ["4", "Preprocessing demo: raw vs cleaned text printed for two "
                      "rows.", "<code>clean_text_for_tfidf</code>"],
                ["5", "Word TF-IDF (1,2 grams) 25k features, "
                      "<code>min_df=2</code>, <code>sublinear_tf</code>; fit on "
                      "train, transform val.", "(16000, 25000) ~6 s"],
                ["6", "Character TF-IDF <code>char_wb</code> (3,5 grams) 50k "
                      "features, <code>min_df=3</code>; fit / transform.",
                 "(16000, 50000) ~16 s"],
                ["7", "Fusion: "
                      "<code>hstack(word, char).tocsr()</code>; nnz printed.",
                 "(16000, 75000)"],
                ["8", "Train NB / LR (<code>C=1.5</code>) / LinearSVC + "
                      "<code>CalibratedClassifierCV(cv=3)</code>; each evaluated "
                      "on the 4k validation split (accuracy, precision, recall, "
                      "F1, ROC-AUC, confusion, full report).",
                 "SVM 99.53% val acc"],
                ["9", "Save <code>results/baseline_ml_metrics.csv</code>.",
                 "results/*.csv"],
                ["10", "Held-out test: load test.csv, sample 5,000 (seed 42), "
                       "transform with the <i>frozen</i> vectorizers, evaluate "
                       "LR and SVM.", "LR 99.26%, SVM 99.58%"],
                ["11", "Save the four pickles to <code>models/</code>; total "
                       "training time printed.", "models/*.pkl"],
            ],
            widths=[12, 118, 36],
        ),
        p("Two pandas-3 fixes are baked into the script (both were hit during "
          "project development): per-class sampling replaces "
          "<code>groupby.apply</code>, which dropped the "
          "<code>generated</code> column, and <code>np.array(..., "
          "dtype=object)</code> wraps Arrow-backed strings before the sklearn "
          "split. Do not revert either."),
        p("The Mixture-of-Experts is trained separately and takes about 130 s: "
          "<code>PYTHONPATH=. python3 moe/train_moe.py</code> builds the three "
          "pools ({{t:pools}} in Chapter 8), fits each expert's 40,000-feature "
          "TF-IDF, trains the type gate and writes "
          "<code>results/moe_expert_metrics.csv</code> and "
          "<code>results/moe_routed_metrics.csv</code>; "
          "<code>moe/calibrate_thresholds.py</code> then produces "
          "<code>models/moe/thresholds.json</code>."),

        # ==================================================================
        h1("5  Inference Workflow (test/inference.py)"),
        p("<code>test/inference.py</code> is the <b>one place</b> that touches "
          "the baseline artifacts. Every front-end — CLI, batch, API, LangChain, "
          "Streamlit and the Vercel function — calls it, so a change here updates "
          "all six at once. {{f:flow_infer}} shows the shared path; {{t:p4_infer}} "
          "is the function contract."),
        table(
            "p4_infer",
            "The inference engine's public functions (test/inference.py, 158 lines).",
            ["Function", "One-liner"],
            [
                ["<code>load_artifacts()</code>",
                 "<code>joblib.load</code> ×4 from <code>models/</code>; raises "
                 "<code>FileNotFoundError</code> with a 'train first' hint if "
                 "missing."],
                ["<code>vectorize(texts, art)</code>",
                 "word transform + char transform → "
                 "<code>hstack().tocsr()</code>; a single string is auto-wrapped "
                 "into a list."],
                ["<code>predict_proba(texts, art, model)</code>",
                 "<code>clf.predict_proba(X)[:, 1]</code> → P(AI) array; "
                 "<code>model='lr'</code> or <code>'svm'</code>."],
                ["<code>predict_text(text, art, model, stylometrics)</code>",
                 "Single-text dict: label (≥ 0.5 → AI), "
                 "<code>probability_ai/human</code>, confidence, optional 24 "
                 "stylometrics."],
                ["<code>predict_batch(texts, batch_size=1000)</code>",
                 "Chunked prediction to avoid out-of-memory on the 97k/290k "
                 "CSVs."],
                ["<code>evaluate_csv(csv, sample_n)</code>",
                 "Batch prediction + accuracy / precision / recall / F1 / AUC, "
                 "confusion matrix and full classification report."],
                ["<code>predict_folder(folder)</code>",
                 "Globs <code>*.txt</code>/<code>*.md</code>, predicts each with "
                 "stylometrics and file paths."],
            ],
            widths=[58, 108],
        ),
        p("The routed path wraps this engine: <code>moe/predict_moe.py</code> "
          "counts words, applies the length gate, loads the matching expert, "
          "applies the length-adaptive threshold, and additionally reports the "
          "type-gate prediction and the low-confidence warning for inputs under "
          "25 / 50 words. Both paths share vectorization and artifact loading, "
          "so baseline and MoE predictions are directly comparable."),
        # ==================================================================
        h1("6  Installation and Environment"),
        p("Everything runs from the project root on a plain laptop CPU — no GPU "
          "is required for the classical pipeline, and none was used for any "
          "number in this report. First-time setup:"),
        code("""
            cd AI_generated_text_classification
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt      # scikit-learn==1.9.1, pandas,
                                                 # scipy, joblib, streamlit, ...
            pip install "langchain-core>=0.3" "fastapi>=0.110" reportlab
            ls -lh data/                          # train.csv ~653 MB, test.csv ~222 MB
        """),
        p("Environment facts verified against the repository: Python 3.13; "
          "scikit-learn pinned to <code>==1.9.1</code> (pickle compatibility — "
          "the bundled models must unpickle unchanged on Vercel); LangChain "
          "superset split out because it is optional for training; and the four "
          "baseline pickles total about 4.8 MB while "
          "<code>models/moe/</code> adds the three experts, the type gate and "
          "the threshold configuration."),
        p("<b>One-command run.</b> <code>./run_all.sh</code> (90 lines) installs, "
          "trains, evaluates, starts the API (:8001) and Streamlit (:8501), with "
          "health checks and log files. Useful flags:"),
        bullets([
            "<code>./run_all.sh</code> — full pipeline: install → train → test → "
            "API → UI.",
            "<code>./run_all.sh --skip-train</code> — reuse the existing "
            "<code>models/</code>.",
            "<code>./run_all.sh --skip-install</code> — reuse the existing "
            "<code>venv/</code>.",
            "<code>./run_all.sh --no-api --no-ui</code> — training and "
            "evaluation only.",
            "<code>./run_all.sh --help</code> — usage; ports: API 8001, "
            "Streamlit 8501 (kill strays with <code>pkill -f \"streamlit run|"
            "test/api.py\"</code>).",
        ]),

        # ==================================================================
        h1("7  How to Run — The Six Front-ends and Copy-Paste Commands"),
        p("{{t:p4_ways}} inventories the six ways to call the same model; the "
          "command blocks below are executable from the project root exactly as "
          "printed."),
        table(
            "p4_ways",
            "The six inference front-ends (Part IV view; cf. "
            "{{t:frontends}} for the full system inventory).",
            ["#", "Way", "Command / endpoint", "Best for"],
            [
                ["1", "CLI single / file",
                 "<code>test/predict.py --text \"...\" / --input file.txt</code>",
                 "One-off classification"],
                ["2", "CLI folder / CSV",
                 "<code>--folder test/example_texts</code> / "
                 "<code>--csv data/test.csv --sample 5000</code>",
                 "Batch evaluation"],
                ["3", "Terminal chat",
                 "<code>test/terminal_demo.py --loop</code> or "
                 "<code>predict.py --interactive</code>", "Trying text live"],
                ["4", "REST API",
                 "<code>test/api.py</code> → POST <code>/predict</code> (:8001)",
                 "Other programs / curl"],
                ["5", "LangChain tool",
                 "<code>ai_human_classifier.run(...)</code>", "LLM agents"],
                ["6", "Web UI",
                 "<code>app/app.py</code> (Streamlit :8501)",
                 "Demonstration to humans"],
            ],
            widths=[8, 32, 84, 42],
        ),
        h2("7.1  Train and evaluate"),
        code("""
            venv/bin/python train_baseline.py
            #   models/word_tfidf_vectorizer.pkl      (969 KB)
            #   models/char_tfidf_vectorizer.pkl      (1.6 MB)
            #   models/logistic_regression_baseline.pkl (587 KB)
            #   models/linear_svm_baseline.pkl        (1.7 MB)
            #   results/baseline_ml_metrics.csv       TOTAL TIME ~33 s

            PYTHONPATH=. python3 moe/download_small_text.py   # HC3 + modern essays
            PYTHONPATH=. python3 moe/train_moe.py             # 3 experts + gate, ~130 s
            PYTHONPATH=. python3 moe/calibrate_thresholds.py  # thresholds.json
        """),
        h2("7.2  Inference — all six ways"),
        code("""
            # 1 — single text (terminal)
            venv/bin/python test/predict.py --text "Your essay here..." --model lr --stylometrics

            # 2 — file / folder (13 labelled examples)
            venv/bin/python test/predict.py --input test/example_texts/human_essay_school.txt --model svm
            venv/bin/python test/predict.py --folder test/example_texts --model lr
            venv/bin/python test/evaluate_examples.py

            # 3 — terminal chat (no browser)
            venv/bin/python test/terminal_demo.py
            venv/bin/python test/terminal_demo.py --api        # needs the API running
            venv/bin/python test/predict.py --interactive --model lr

            # 4 — REST API (second terminal)
            venv/bin/python test/api.py                         # → http://localhost:8001/docs
            curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \\
              -d '{"text":"In todays rapidly evolving digital landscape...","model":"lr","stylometrics":true}'

            # 5 — LangChain tool
            venv/bin/python test/langchain_tool.py              # no-LLM demo
            #   in an agent: from test.langchain_tool import ai_human_classifier

            # 6 — Streamlit web UI (second terminal)
            venv/bin/streamlit run app/app.py --server.port 8501   # → http://localhost:8501

            # routed Mixture-of-Experts CLI (any of the above, for MoE)
            PYTHONPATH=. python3 moe/predict_moe.py "your text"
        """),
        h2("7.3  Batch evaluation and reports"),
        code("""
            venv/bin/python test/predict.py --csv data/test.csv  --model lr  --out test/results           # 97k rows
            venv/bin/python test/predict.py --csv data/train.csv --model svm --out test/results_train_full # 290k rows
            venv/bin/python test/test_model.py            # 2k-row smoke test (asserts acc > 0.90)
            venv/bin/python test/test_model.py --full     # full 97k
        """),

        # ==================================================================
        h1("8  Outputs — What Each File Means"),
        table(
            "p4_outputs",
            "Artifacts written by training and evaluation.",
            ["File", "What it is"],
            [
                ["<code>models/*.pkl</code> (4)",
                 "Frozen vectorizers + classifiers. The <i>only</i> things "
                 "inference loads; retrain to replace."],
                ["<code>models/moe/*.pkl</code> + "
                 "<code>thresholds.json</code>",
                 "Three length experts, the stylometric type gate and the "
                 "length-adaptive threshold configuration."],
                ["<code>results/baseline_ml_metrics.csv</code>",
                 "Validation metrics per model (train_baseline.py, step 9)."],
                ["<code>results/moe_*.csv</code> + "
                 "<code>moe_summary.json</code>",
                 "Baseline-comparison, routed, per-expert, calibrated metrics and "
                 "type-gate accuracy — every MoE table in this report reads "
                 "these."],
                ["<code>test/results/predictions.csv</code>",
                 "Per-row: text, generated, prob_ai, pred, correct — review "
                 "errors here."],
                ["<code>test/results/metrics.json</code>",
                 "Same metrics as the terminal + confusion matrix + full "
                 "classification report."],
                ["<code>test/example_texts/</code>",
                 "13 <code>.txt</code> examples (<code>human_</code>/"
                 "<code>ai_</code> prefix = ground truth), scored by "
                 "<code>evaluate_examples.py</code>."],
                ["<code>test/example_test_set.csv</code>",
                 "20-row balanced CSV for quick batch tests "
                 "(<code>--csv test/example_test_set.csv</code>)."],
                ["<code>results/figures/*.png</code>",
                 "The report's figure sources (15 original + 2 rebuilt); see "
                 "Figure_Inventory.md for provenance."],
                ["<code>api.log</code> / <code>streamlit.log</code>",
                 "Server logs written by <code>run_all.sh</code>."],
            ],
            widths=[56, 110],
        ),

        # ==================================================================
        h1("9  Troubleshooting and FAQ"),
        bullets([
            "<b><code>FileNotFoundError: models/*.pkl</code></b> — you have not "
            "trained yet. Run <code>venv/bin/python train_baseline.py</code> "
            "(needs <code>data/train.csv</code>).",
            "<b><code>KeyError 'generated'</code> / Arrow TypeError on pandas 3 "
            "— already fixed in <code>train_baseline.py</code> (per-class "
            "sampling + <code>np.array(..., dtype=object)</code>). Do not revert "
            "to <code>groupby.apply</code>.",
            "<b>API connection refused on :8001</b> — the API is not running. "
            "Start <code>venv/bin/python test/api.py</code> in a separate "
            "terminal and keep it open.",
            "<b><code>uvicorn test.api:app</code> fails</b> — "
            "<code>test/</code> needs <code>__init__.py</code> (it ships), or "
            "simply run <code>python test/api.py</code> directly.",
            "<b>Short text predicted wrong</b> (a 26-word human file called AI) "
            "— expected for the baseline: it was trained on essays with a "
            f"{M['ds']['train.csv']['median']:.0f}-word median. This is exactly "
            "the failure Chapter 7 measures and the MoE fixes; test short inputs "
            "through <code>moe/predict_moe.py</code>.",
            "<b>Streamlit not showing the new model</b> — kill the old process "
            "(<code>pkill -f \"streamlit run\"</code>) and restart: "
            "<code>@st.cache_resource</code> caches artifacts.",
            "<b>Out-of-memory on the full train.csv</b> — never load it whole: "
            "use <code>--csv ... --sample 5000</code> or the built-in "
            "<code>batch_size=1000</code> in <code>evaluate_csv()</code>.",
            "<b>Pickle version errors on deploy</b> — scikit-learn is pinned "
            "<code>==1.9.1</code> everywhere for a reason: pickles only unpickle "
            "reliably within the same minor version.",
        ]),

        # ==================================================================
        h1("10  Reproducibility Checklist"),
        p("Every table, figure and headline number in this report can be "
          "regenerated from the released artifacts by following this checklist."),
        numbered([
            "<b>Data.</b> <code>data/train.csv</code> (290,300), "
            "<code>data/test.csv</code> (97,447), "
            "<code>data/moe/hc3_small.csv</code> (80,811) and "
            "<code>data/moe/modern_essays.csv</code> (17,997) present as-is; "
            "columns <code>text, generated</code>.",
            "<b>Splits (all seeded).</b> Baseline: 10k/class sample (seed 42) → "
            "80/20 stratified → held-out 5,000 (seed 42). MoE pools: balanced, "
            "seed 42. Calibration: 2,500 / 2,500 (seed 999). HC3 and modern "
            "holdouts: 15% (seed 42), never trained on.",
            "<b>Train.</b> <code>venv/bin/python train_baseline.py</code> (~33 s) "
            "then <code>moe/train_moe.py</code> (~130 s) and "
            "<code>moe/calibrate_thresholds.py</code>.",
            "<b>Evaluate.</b> The training script itself writes "
            "<code>results/baseline_ml_metrics.csv</code>; "
            "<code>moe/make_moe_appendix.py</code> and the MoE training script "
            "write <code>results/moe_*.csv</code>.",
            "<b>Figures.</b> All 17 PNGs live in <code>results/figures/</code> "
            "(plus <code>docs/assets/live-inference.png</code>); the two "
            "recreated diagrams are rebuilt by "
            "<code>final_report/build/make_missing_figures.py</code>.",
            "<b>Documents.</b> The four original generators "
            "(<code>make_report.py</code>, <code>moe/make_moe_appendix.py</code>, "
            "<code>make_research_paper.py</code>, "
            "<code>make_deepseek_paper.py</code>, <code>make_run_guide.py</code>) "
            "produced the source PDFs for Parts I–IV; this consolidated report "
            "rebuilds all parts from the same verified CSVs via "
            "<code>final_report/build/build.py</code>.",
            "<b>Verification.</b> "
            "<code>final_report/build/recompute_test_metrics.py</code> and "
            "<code>verify_dataset_stats.py</code> independently recompute the "
            "headline test metrics and corpus statistics from the raw CSVs; "
            "Quality_Audit.md records what they found.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Appendix A — Key Snippet: load → vectorize → predict"),
        p("The entire inference path in ten lines; everything else is a wrapper "
          "around it:"),
        code("""
            import joblib
            from scipy.sparse import hstack

            word_vec = joblib.load("models/word_tfidf_vectorizer.pkl")
            char_vec = joblib.load("models/char_tfidf_vectorizer.pkl")
            lr       = joblib.load("models/logistic_regression_baseline.pkl")

            text = open("test/example_texts/ai_essay_distance_learning.txt").read()
            X = hstack([word_vec.transform([text]), char_vec.transform([text])]).tocsr()
            p_ai = lr.predict_proba(X)[0, 1]
            print("AI-Generated" if p_ai >= 0.5 else "Human-Written", f"P(AI)={p_ai:.4f}")
            # -> AI-Generated P(AI)=0.9856
        """),
        p("Higher-level equivalents: "
          "<code>from test.inference import load_artifacts, predict_text; "
          "r = predict_text(text, load_artifacts(), model='lr')</code> for "
          "Python; <code>POST /predict</code> for the REST API; "
          "<code>ai_human_classifier.run({'text': text})</code> for LangChain; "
          "and <code>PYTHONPATH=. python3 moe/predict_moe.py \"...\"</code> for "
          "the routed system with threshold and type reporting."),

        h1("Appendix B — Deployment"),
        p("Two deployments serve the released system; both consume the same "
          "artifacts as the local CLI."),
        table(
            "p4_deploy",
            "Deployment targets.",
            ["Target", "URL", "What is served"],
            [
                ["Vercel (production)",
                 "<code>moe-text-classifier-pravesh12.vercel.app</code>",
                 "Static project site (<code>web/index.html</code>), results, "
                 "figures and report PDFs, plus <code>POST /api/predict</code> "
                 "from one Python serverless function (<code>web/api/predict.py</code>, "
                 "269 lines). Models bundle in <code>web/api/models/</code> "
                 "(~10 MB) with a GitHub fallback "
                 "(<code>MODEL_BASE_URL</code>); <code>src/</code> and "
                 "<code>moe/</code> ship so the pickles unpickle; "
                 "scikit-learn pinned <code>==1.9.1</code>."],
                ["GitHub Pages", "<code>docs/</code> directory of the repository",
                 "Mirror of the site: figures, results tables and the published "
                 "report PDF."],
                ["Local",
                 "<code>localhost:8001</code> (API), <code>localhost:8501</code> "
                 "(Streamlit)",
                 "<code>./run_all.sh</code> — full stack with health checks and "
                 "logs."],
            ],
            widths=[30, 46, 90],
        ),
        p("The serverless predictor implements the identical contract as "
          "<code>test/api.py</code> (single-text JSON in, label / P(AI) / "
          "confidence out) with a length-aware MoE branch, so numbers returned "
          "by the deployed site match the local CLI to the fourth decimal."),
    ]
    return blocks
