"""Part I — Complete Project Report, Chapters 1–6."""
from blocks import (part, h1, h2, p, abstract, bullets, numbered, fig,
                    table, code, eq, pagebreak, spacer)

def _f4(x):
    return f"{x:.4f}"

def _pc(x, d=1):
    return f"{x * 100:.{d}f}%"


def part1a(M):
    ds = M["ds"]            # dataset_stats_verified.json
    ver = M["test"]         # metrics_verified.json
    val = M["val_rows"]     # from results/baseline_ml_metrics.csv
    tr, te = ds["train.csv"], ds["test.csv"]
    hc, mo = ds["hc3_small.csv"], ds["modern_essays.csv"]
    ws = ver["word_stats"]
    lr_t, svm_t = ver["models"]["lr"], ver["models"]["svm"]

    blocks = [
        part("Part I — Complete Project Report",
             "Chapters 1–12: problem, data, baseline method, short-text failure, "
             "Mixture-of-Experts, results, application, limitations, conclusion"),

        # ==================================================================
        h1("Chapter 1 — Introduction and Problem Statement"),
        h2("1.1  Background and motivation"),
        p("Large language models (LLMs) such as ChatGPT, GPT-4 and Claude produce fluent, "
          "well-structured text on demand. What used to require an educated writer — a "
          "coherent essay, a news summary, a product review — is now available to anyone in "
          "seconds. The upside is obvious; the cost is that text itself stops being evidence "
          "of authorship. Universities need to protect academic integrity, newsrooms need to "
          "attribute bylines, and platforms need to moderate generated content at scale. "
          "Deciding whether a passage was <i>human-written</i> or <i>AI-generated</i> has "
          "therefore become a standard problem in natural language processing and information "
          "forensics."),
        p("The literature offers a familiar recipe: collect paired human/machine text, extract "
          "features, train a classifier, report accuracy. Benchmarks such as HC3 [1], "
          "Defactify [2] and M4 [3] follow it and produce headline numbers in the high 90s. "
          "This project started from exactly that recipe — a dual TF-IDF pipeline with classical "
          "linear classifiers — and reproduced the headline: <b>99.58% accuracy on held-out "
          "essays</b>. The interesting part of this project is what happened next: we attacked "
          "our own model with short inputs and watched it fail in a systematic, biased way, then "
          "rebuilt it until it did not."),
        h2("1.2  Problem statement"),
        p("Let <i>x</i> be a text, <i>y</i> ∈ {0, 1} its true label (0 = human, 1 = AI), and "
          "<i>n</i>(<i>x</i>) its length in words. We seek a classifier "
          "<i>f</i>(<i>x</i>) → <i>ŷ</i> that maximises accuracy when the length distribution "
          "at test time differs sharply from training: the training corpus has a median of "
          f"{tr['median']:.0f} words (5th percentile {tr['p5']:.0f} words), while real "
          "deployment inputs — chat replies, comments, reviews, one-sentence blurbs — may carry "
          "fewer than 50 words. Two failure mechanisms make this hard: (i) <i>evidence "
          "sparsity</i>, because TF-IDF counts, character n-grams and stylometric statistics all "
          "need tokens to stabilise; and (ii) <i>domain shift</i>, because short text is mostly "
          "chat and casual writing, a register an essay-trained model has never seen."),
        h2("1.3  The length blind spot"),
        p("Quantitatively, the problem is severe. The frozen Phase-1 baseline reaches "
          f"{_pc(ver['models']['lr']['accuracy'], 2)} accuracy on full-length held-out essays, "
          "but only 67.4% when the same essays are truncated to their first 25 words, and 59.8% "
          "at a single sentence. More disturbing than the drop is its <i>direction</i>: at 25 "
          "words the model labels 51% of genuine human texts as AI (human-only accuracy 48.9%, "
          "worse than a coin flip) while still recognising 98.7% of AI texts. A detector that "
          "confidently accuses short humans is worse than an uninformative one — {{f:acc_vs_len}} "
          "and {{f:human_ai_split}} quantify this failure and are analysed in Chapter 7."),
        h2("1.4  Contributions"),
        bullets([
            "<b>A reproducible baseline.</b> Dual word+character TF-IDF (75,000 features) with "
            "Multinomial Naive Bayes, Logistic Regression and a calibrated Linear SVM, trained "
            "in 33 seconds and evaluated on a 5,000-essay held-out test "
            "(best model: " + _pc(svm_t['accuracy'], 2) + " accuracy, ROC-AUC "
            + f"{svm_t['roc_auc']:.4f}).",
            "<b>A length ablation study.</b> First-N-words and first-N-sentences truncations of "
            "the same held-out essays, with accuracy split by true class, revealing a one-sided "
            "false-positive collapse (Chapter 7).",
            "<b>Cross-domain evaluation.</b> Two never-trained-on holdouts — HC3 chat answers and "
            "2024-LLM essays — showing the baseline falls to 48.6% and 85.5% (Chapter 7).",
            "<b>A Mixture-of-Experts detector.</b> A deterministic length gate, three "
            "length-specialised TF-IDF + LR experts trained on newly downloaded small-text data, "
            "a learned chat/casual/essay type gate, and length-adaptive thresholds (Chapters "
            "8–9): 86.1% at 25 words, 98.96% full text, 98.5% on unseen chat.",
            "<b>A complete, reproducible system.</b> Six inference front-ends (CLI, batch, REST "
            "API, LangChain tool, Streamlit application, Vercel serverless API), trained models "
            "in version control, and all metrics released as CSV (Chapter 10, Part IV).",
        ]),
        h2("1.5  Organisation of the report"),
        bullets([
            "<b>Part I (Chapters 1–12)</b> — the complete project report: problem, data, "
            "baseline methodology and results, short-text failure analysis, MoE architecture and "
            "results, the deployed application, limitations and conclusion.",
            "<b>Part II</b> — the conference-format research paper: <i>Length-Expertized "
            "Authorship Detection: A Mixture-of-Experts for Human vs AI-Generated Text</i>.",
            "<b>Part III</b> — the detailed research paper with formal notation, equations and "
            "ablation analysis: <i>Why Long-Text Detectors Fail on Short Text and How "
            "Mixture-of-Experts Fixes Them</i>.",
            "<b>Part IV</b> — technical documentation and reproducibility guide: repository map, "
            "training and inference workflows, commands, API and troubleshooting.",
        ]),
        p("Figures and tables are numbered continuously across all four parts; each part keeps "
          "its own section numbering so the two papers remain readable as standalone documents."),

        # ==================================================================
        pagebreak(),
        h1("Chapter 2 — Objectives and Scope"),
        h2("2.1  Objectives"),
        numbered([
            "Build an end-to-end classification pipeline that separates human-written from "
            "AI-generated text with classical, fully interpretable features (stylometrics + "
            "dual TF-IDF) and linear classifiers.",
            "Quantify the pipeline's behaviour across the evaluation axes that matter — "
            "accuracy, precision, recall, F1, ROC-AUC, confusion matrices and feature-family "
            "comparisons.",
            "Stress-test the trained model under length shift (truncation to 25–200 words and "
            "to single sentences) and under domain shift (unseen chat and unseen generators), "
            "and characterise the resulting error structure.",
            "Acquire genuine small-text training data and modern-LLM essay data, and combine "
            "them with truncation augmentation of the original corpus.",
            "Design and train a length-routed Mixture-of-Experts with a learned text-type gate "
            "and length-adaptive decision thresholds that fixes the short-text failure without "
            "sacrificing full-text accuracy.",
            "Package everything as a usable system — training script, CLI, REST API, LangChain "
            "tool, web application and documentation — so that every reported number can be "
            "reproduced from the released artifacts.",
        ]),
        h2("2.2  Scope and implementation status"),
        p("Accuracy of claims matters more than breadth in an engineering report. Every row of "
          "{{t:status}} states what actually exists in the repository and what was actually "
          "measured; components that were designed but not trained are marked as such and are "
          "not credited with any reported score."),
        table(
            "status",
            "Implementation status of every project component (verified against the repository).",
            ["Component", "Where it lives", "Status"],
            [
                ["Dual TF-IDF baseline (NB, LR, calibrated SVM)",
                 "train_baseline.py, models/*.pkl",
                 "Implemented and evaluated — all Part I results"],
                ["Stylometric features (TTR, burstiness, readability)",
                 "src/stylometrics.py",
                 "Implemented — used by the demo UI and by the MoE type gate"],
                ["MoE experts, length gate, type gate, thresholds",
                 "moe/*.py, models/moe/*",
                 "Implemented and evaluated — Chapters 8–9"],
                ["Inference engine + six front-ends",
                 "test/inference.py, test/, app/app.py",
                 "Implemented and exercised — Chapter 10, Part IV"],
                ["EDA and feature notebooks 01–04",
                 "notebooks/",
                 "Implemented as notebooks — supporting analysis"],
                ["Transformer fine-tuning (DistilBERT, RoBERTa)",
                 "notebooks/05_*.ipynb",
                 "Prepared but not trained in this report — no scores reported"],
                ["Sentence-BERT semantic fusion",
                 "src/semantic.py",
                 "Scaffolding only — not wired into the inference path"],
                ["Feature-family ablation study",
                 "notebooks/06_*.ipynb",
                 "Planned Phase-3 work — not part of reported results"],
            ],
            widths=[58, 50, 56],
        ),
        h2("2.3  Explicit non-claims"),
        bullets([
            "No transformer (DistilBERT/RoBERTa) score is reported anywhere in this report, "
            "because no transformer was trained or evaluated as part of the recorded results.",
            "Stylometric ablation and Sentence-BERT fusion are <i>available code paths</i>, not "
            "completed experiments; they are listed as future work in Chapter 11.",
            "All reported metrics come from frozen artifacts in <code>models/</code> and "
            "<code>models/moe/</code> evaluated by the released scripts — none are hand-entered "
            "or extrapolated.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 3 — Dataset Description and Analysis"),
        h2("3.1  Corpora and scale"),
        p("Four corpora are used. The original essay pair (<code>data/train.csv</code>, "
          "<code>data/test.csv</code>) carries columns <code>text, generated</code> where "
          "<code>generated=0</code> is human and <code>generated=1</code> is AI. The two "
          "additional corpora — HC3 question-answer pairs and modern-2024-LLM essays — were "
          "downloaded for the MoE upgrade. {{t:ds_scale}} summarises all four; every figure in "
          "it was recomputed directly from the CSV files during preparation of this report."),
        table(
            "ds_scale",
            "The four corpora: scale, class balance and length profile.",
            ["Dataset", "Rows", "Human / AI", "Median length", "Role in the project"],
            [
                ["<code>data/train.csv</code>", f"{tr['rows']:,}",
                 f"{tr['human']:,} / {tr['ai']:,}",
                 f"{tr['median']:.0f} words",
                 "Baseline training (20k balanced subsample)"],
                ["<code>data/test.csv</code>", f"{te['rows']:,}",
                 f"{te['human']:,} / {te['ai']:,}",
                 f"{te['median']:.0f} words",
                 "Held-out evaluation (5,000-sample)"],
                ["HC3 <code>reddit_eli5</code> + <code>open_qa</code> + <code>finance</code>",
                 f"{hc['rows']:,}", f"{hc['human']:,} / {hc['ai']:,}",
                 f"{hc['median']:.0f} words ({hc['pct_le_150']:.0f}% ≤ 150w)",
                 "SHORT expert — genuine chat Q&amp;A"],
                ["Modern essays (GPT-4, Claude 3, Llama 3, Command R+)",
                 f"{mo['rows']:,}", f"{mo['human']:,} / {mo['ai']:,}",
                 f"{mo['median']:.0f} words ({mo['pct_gt_400']:.0f}% &gt; 400w)",
                 "LONG/VERYLONG experts — 2024 generators"],
            ],
            widths=[48, 20, 32, 34, 30],
        ),
        p(f"The training corpus is exactly balanced ({tr['human']:,} essays per class); the test "
          f"corpus is human-heavy ({te['human'] / te['rows'] * 100:.1f}% human), which is the "
          "harder, more realistic direction for a deployed detector. {{f:class_dist}} shows the "
          "class balance of both files."),
        fig("class_dist", 0.86),

        h2("3.2  Length profile"),
        p(f"The defining property of the original corpora is that they contain essays, not short "
          f"text. Across all {tr['rows']:,} training essays the median length is "
          f"{tr['median']:.0f} words with a 5th percentile of {tr['p5']:.0f} words; the test "
          f"corpus mirrors this (median {te['median']:.0f} words). In the 5,000-essay evaluation "
          f"sample the mean is {ws['mean']:.0f} words and the median {ws['median']:.0f}, with "
          f"the shortest human essay at {ws['human_min']} words and human essays averaging "
          f"{ws['human_mean']:.0f} words against {ws['ai_mean']:.0f} for AI essays — human "
          "writing in this corpus is simply longer. {{f:length_hist}} plots the character-length "
          "distribution: there is essentially no mass below 200 characters, which means the "
          "model never saw a short input during training. This single fact drives the failure "
          "analysed in Chapter 7."),
        fig("length_hist", 0.86),

        h2("3.3  Text characteristics"),
        p("<b>Human example</b> (verbatim from <code>test.csv</code>): <i>\"There are many "
          "advantages TD limiting the number DF cars DN the roads…\"</i> — OCR/typing artefacts "
          "such as <code>TD</code>, <code>DF</code>, <code>DN</code> are pervasive in human "
          "essays and are strong character n-gram cues. <b>AI example</b>: <i>\"In today's "
          "world, having a positive attitude is a crucial factor in achieving success…\"</i> — "
          "polished formal structure with headings such as \"Pros of Distance Learning: "
          "1. Flexibility\". AI text is orthographically clean; human text is noisy. The "
          "baseline exploits exactly this asymmetry (Chapter 4, §4.5)."),
        p("Because the unit of data is a multi-paragraph essay (median "
          f"{te['median']:.0f} words, roughly {te.get('median_chars', 2092):,.0f} characters "
          "at the sample level), chunking is required for any transformer-based alternative — "
          "one reason the classical pipeline remains the practical choice for this project."),

        h2("3.4  Small-text and modern-generator corpora (MoE upgrade)"),
        p(f"<b>HC3</b> (Hello-SimpleAI/HC3, Guo et al. [1]) was flattened to one row per answer "
          f"(≥ 40 characters) from three files: <code>reddit_eli5.jsonl</code>, "
          f"<code>open_qa.jsonl</code> and <code>finance.jsonl</code>. It yields "
          f"{hc['rows']:,} paired answers ({hc['human']:,} human, {hc['ai']:,} ChatGPT) with a "
          f"median of {hc['median']:.0f} words; {hc['pct_le_150']:.1f}% are ≤ 150 words, "
          f"{hc['pct_151_400']:.1f}% fall in 151–400 words and only {hc['pct_gt_400']:.1f}% "
          "exceed 400. It is the only source in the project of <i>genuine short chat</i>: the "
          "same question answered by a human and by ChatGPT."),
        p(f"<b>Modern essays</b> (<code>artnitolog/llm-generated-texts</code> [8]) supply "
          f"{mo['rows']:,} texts — {mo['human']:,} human essays against {mo['ai']:,} "
          "machine essays from GPT-4 Turbo/Omni, Claude 3 Opus, Llama 3 70B and Command R+, "
          f"which are generators the original corpus never saw. Median length "
          f"{mo['median']:.0f} words, {mo['pct_gt_400']:.0f}% longer than 400 words — this "
          "corpus audits the baseline on 2024-era models."),
        p("<b>Truncation augmentation</b> complements both: every essay in the 20,000-essay "
          "training subsample additionally contributes 50/100/150-word copies (prefix or random "
          "window, label preserved) to the short pool, and essays over 200 words contribute "
          "200/300-word copies to the long pool. This teaches the short expert <i>short-essay</i> "
          "style, which HC3 cannot provide. The resulting training pools are specified in "
          "{{t:pools}} (Chapter 8)."),

        h2("3.5  Data availability and governance"),
        bullets([
            "<code>data/train.csv</code> (653 MB) and <code>data/test.csv</code> (222 MB) are "
            "<b>gitignored</b> — they are catalogued with Google-Drive placeholders in "
            "<code>data/README.md</code>; small tracked samples are provided as "
            "<code>data/sample_*.csv</code>.",
            "<code>data/moe/*.csv</code> (132 MB) is also gitignored but fully reproducible: "
            "<code>PYTHONPATH=. python3 moe/download_small_text.py</code> re-downloads both "
            "corpora from the Hugging Face Hub.",
            "A zipped bundle <code>data/ai_text_moe_data.zip</code> (40 MB) exists locally for "
            "offline sharing.",
            "All random operations are seeded (seed 42 for sampling and splits, 999 for "
            "calibration) so every table in this report can be regenerated bit-for-bit.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 4 — Baseline Methodology"),
        p("The baseline answers one question with a deliberately simple model: can a linear "
          "classifier over surface statistics separate human from AI essays? Everything is "
          "implemented in <code>train_baseline.py</code> (11 logged steps) with shared helpers "
          "in <code>src/</code>."),

        h2("4.1  Preprocessing"),
        bullets([
            "<code>clean_text_for_tfidf(text)</code> — NFKC Unicode normalisation, whitespace "
            "collapse, lowercasing. Applied as the <code>preprocessor=</code> of the word "
            "TF-IDF so training and inference normalise text identically.",
            "<code>clean_text_for_stylometrics(text)</code> — preserves casing and punctuation "
            "(capital ratio, comma/semicolon rates) because stylometric features read them; it "
            "never lowercases.",
            "Vectorizers are <b>fit on the training split only</b>; validation, test and the "
            "MoE holdouts are merely transformed — no vocabulary or IDF leakage.",
        ]),

        h2("4.2  Stylometric and readability features"),
        p("Although the baseline classifier consumes TF-IDF features only, the project extracts "
          "a 24-value stylometric vector (<code>src/stylometrics.py</code>) [6] for two purposes: "
          "human-readable explanations in the demo interface, and the 16 inputs of the MoE type "
          "gate. These include type-token ratio (TTR), hapax ratio, mean sentence length, "
          "burstiness (σ/μ of sentence lengths), punctuation and capitalisation rates, and three "
          "readability scores defined as:"),
        eq("FRE = 206.835 − 1.015 · (N/S) − 84.6 · (syll/N)&nbsp;&nbsp;&nbsp;&nbsp;"
           "FKGL = 0.39 · (N/S) + 11.8 · (syll/N) − 15.59&nbsp;&nbsp;&nbsp;&nbsp;"
           "Fog = 0.4 · [(N/S) + 100 · (poly/N)]"),
        p("where N is the number of words, S the number of sentences, syll/N the average "
          "syllables per word and poly/N the fraction of polysyllabic words. All three need "
          "several sentences before they stabilise — a fact that becomes decisive in Chapter 7."),

        h2("4.3  Dual TF-IDF representation"),
        p("Two views of every document are extracted and fused ({{f:tfidf_fusion}}):"),
        bullets([
            "<b>Word TF-IDF</b> — unigrams and bigrams, <code>ngram_range=(1,2)</code>, "
            "<code>max_features=25000</code>, <code>min_df=2</code>, "
            "<code>sublinear_tf=True</code>. Captures lexical choice and common phrases.",
            "<b>Character TF-IDF</b> — <code>analyzer='char_wb'</code>, "
            "<code>ngram_range=(3,5)</code>, <code>max_features=50000</code>, "
            "<code>min_df=3</code>. Captures typos, OCR noise, punctuation habits and sub-word "
            "morphology.",
            "<b>Fusion</b> — <code>hstack([word, char]).tocsr()</code> producing a "
            "<b>75,000-dimensional</b> sparse vector; the 16,000-essay training matrix holds "
            "34.8 million non-zero cells.",
        ]),
        fig("tfidf_fusion", 0.9),
        p("Formally, term weighting follows the standard TF-IDF scheme [5] and concatenation gives "
          "the feature map used by every classifier:"),
        eq("tfidf(t, d) = tf(t, d) · log( (1 + N) / (1 + df(t)) )&nbsp;&nbsp;&nbsp;&nbsp;"
           "F(x) = [ F<sub>word</sub>(x) ; F<sub>char</sub>(x) ] ∈ ℝ<sup>75,000</sup>"),

        h2("4.4  Classifiers"),
        bullets([
            "<b>Multinomial Naive Bayes</b> (<code>alpha=0.1</code>) — a fast generative count "
            "baseline; it works on the (non-negative) TF-IDF matrix directly.",
            "<b>Logistic Regression</b> (<code>C=1.5</code>, <code>lbfgs</code>, "
            "<code>max_iter=1000</code>, L2 penalty) — the discriminative workhorse; it yields "
            "well-behaved probabilities P(AI) used by the interface.",
            "<b>Linear SVM</b> (<code>C=1.0</code>) wrapped in "
            "<code>CalibratedClassifierCV(cv=3)</code> — a max-margin classifier with "
            "3-fold Platt scaling so it too outputs P(AI).",
        ]),
        p("All three are linear models over the same 75,000 features [4]; comparing them isolates "
          "the effect of the decision rule from the effect of the representation."),

        h2("4.5  Why character n-grams matter"),
        p("Human essays in this corpus carry OCR and typing noise (<code>TD</code>, "
          "<code>AHD</code>, <code>IH</code>, dropped apostrophes), while machine text is "
          "orthographically clean. Character n-grams observe this directly at the sub-word "
          "level, which is why the character branch contributes much of the baseline's margin "
          "over a word-only model — and why the same branch becomes brittle when the input is "
          "too short to contain a representative sample of those patterns."),

        # ==================================================================
        pagebreak(),
        h1("Chapter 5 — Experimental Setup"),
        h2("5.1  Environment and protocol"),
        p("Everything runs on a laptop CPU: Python 3.10+ (developed and executed on Python 3.13), "
          "scikit-learn, pandas, NumPy, SciPy, joblib and reportlab. The environment is created "
          "from <code>requirements.txt</code>; the two scripts that matter are "
          "<code>train_baseline.py</code> (baseline, 33 s) and "
          "<code>PYTHONPATH=. python3 moe/train_moe.py</code> (MoE, ~130 s)."),
        code("""
            # environment
            python3 -m venv venv && source venv/bin/activate
            pip install -r requirements.txt

            # baseline: 11 logged steps, ~33 s
            venv/bin/python train_baseline.py

            # MoE upgrade: data -> experts -> thresholds
            PYTHONPATH=. python3 moe/download_small_text.py
            PYTHONPATH=. python3 moe/train_moe.py
            PYTHONPATH=. python3 moe/calibrate_thresholds.py
        """),
        h2("5.2  Sampling, splits and seeds"),
        p("The baseline trains on a balanced 20,000-essay subsample (10,000 per class) rather "
          "than all 290,300 rows: the full 290k × 75k TF-IDF would be about 14× slower with "
          "diminishing returns, and notebook 03 defines the 20k protocol. {{t:protocol}} lists "
          "every split used anywhere in the project, including the MoE splits described in "
          "Chapter 8, so that a single table governs reproducibility."),
        table(
            "protocol",
            "Complete list of splits and random seeds.",
            ["Split", "Size", "Seed", "Used for"],
            [
                ["Balanced subsample of train.csv", "20,000 (10k/class)", "42",
                 "Baseline training pool"],
                ["Stratified train / validation", "16,000 / 4,000", "42",
                 "Baseline fit and validation metrics"],
                ["Held-out test sample", "5,000", "42",
                 "All headline baseline &amp; ablation numbers"],
                ["MoE pools (per bucket)", "80 / 20 within pool", "42",
                 "Expert training and per-expert validation"],
                ["Calibration / evaluation halves", "2,500 / 2,500", "999",
                 "Threshold tuning and unbiased eval"],
                ["HC3 &amp; modern holdouts", "15% of each corpus", "42",
                 "Cross-domain evaluation (never trained on)"],
            ],
            widths=[54, 36, 16, 60],
        ),
        fig("flow_train", 0.78),
        h2("5.3  Metrics"),
        p("AI is the positive class throughout. We report accuracy, precision, recall, F1 "
          "(harmonic mean), ROC-AUC and the full confusion matrix. Because accuracy averages "
          "away directional bias, all length-conditional analyses in Chapter 7 additionally "
          "report <i>human-only accuracy</i> (true-negative rate) and <i>AI-only accuracy</i> "
          "(true-positive rate) — this is where the failure actually hides."),
        h2("5.4  Leakage controls"),
        bullets([
            "TF-IDF vocabularies and IDF weights are fit on the training split only.",
            "<code>test.csv</code> never enters any training or tuning step.",
            "HC3 and modern-essay holdouts (15% stratified) are excluded from every training "
            "pool by construction.",
            "Calibration uses a disjoint half (seed 999) and is evaluated on the other half.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 6 — Baseline Results and Evaluation"),
        h2("6.1  Validation results (4,000 essays)"),
        p("All three classifiers exceed 95% validation accuracy; the linear models exceed 99%. "
          "{{t:val}} is reproduced from <code>results/baseline_ml_metrics.csv</code>, the file "
          "written by step 9 of the training script."),
        table(
            "val",
            "Baseline validation metrics on the stratified 4,000-essay split "
            "(AI is the positive class).",
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[r["model"], _f4(r["accuracy"]), _f4(r["precision"]), _f4(r["recall"]),
              _f4(r["f1"]), _f4(r["roc_auc"])] for r in val],
            widths=[46, 22, 24, 22, 22, 26],
        ),
        p("The calibrated Linear SVM leads on every metric (99.53% accuracy, F1 0.9952, AUC "
          "0.9991), with Logistic Regression within 0.3 percentage points and the advantage of "
          "natively calibrated probabilities. Naive Bayes trails mainly in recall on AI "
          "(0.935): it misroutes 130 AI essays to the human class where LR misroutes 46. "
          "{{f:model_cmp}} visualises the same comparison."),
        fig("model_cmp", 0.92),

        h2("6.2  Held-out test results (5,000 essays)"),
        p("The frozen vectorizers and classifiers were then applied to 5,000 essays sampled "
          "from <code>test.csv</code> (seed 42). {{t:test}} reports these metrics; every value "
          "was recomputed from the released <code>models/*.pkl</code> artifacts while preparing "
          "this report and matches <code>results/moe_baseline_comparison.csv</code> exactly."),
        table(
            "test",
            "Held-out test metrics (5,000 essays, seed 42), recomputed from the frozen "
            "model artifacts.",
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [
                ["Logistic Regression", _f4(lr_t["accuracy"]), _f4(lr_t["precision"]),
                 _f4(lr_t["recall"]), _f4(lr_t["f1"]), _f4(lr_t["roc_auc"])],
                ["Linear SVM (calibrated)", _f4(svm_t["accuracy"]), _f4(svm_t["precision"]),
                 _f4(svm_t["recall"]), _f4(svm_t["f1"]), _f4(svm_t["roc_auc"])],
            ],
            widths=[46, 22, 24, 22, 22, 26],
        ),
        p(f"Despite the test set being human-heavy ({ver['n_human']:,} human vs "
          f"{ver['n_ai']:,} AI essays), precision stays above 0.99 for both models. The "
          "confusion matrices in {{f:confusion}} give the raw counts: Logistic Regression makes "
          f"{lr_t['fp']} false positives and {lr_t['fn']} false negatives "
          f"(TN {lr_t['tn']:,} / TP {lr_t['tp']:,}); the calibrated SVM makes "
          f"{svm_t['fp']} false positives and {svm_t['fn']} false negatives "
          f"(TN {svm_t['tn']:,} / TP {svm_t['tp']:,}) — {svm_t['fp'] + svm_t['fn']} mistakes in "
          f"5,000 essays. {{f:roc}} shows both ROC curves hugging the top-left corner "
          f"(AUC {lr_t['roc_auc']:.4f} and {svm_t['roc_auc']:.4f})."),
        fig("confusion", 0.95),
        fig("roc", 0.62),

        h2("6.3  Discussion"),
        bullets([
            "<b>Why 99% is believable here.</b> Each essay provides hundreds of weak cues — "
            "diction, character noise, punctuation habits — and a linear model simply sums them. "
            "Essay length guarantees a dense feature vector, so the result is not an artefact.",
            "<b>The test set is harder than validation</b> (37% AI, unseen writers) yet scores "
            "match, confirming that the 20,000-essay subsample generalises to unseen rows of "
            "<code>test.csv</code>.",
            "<b>SVM vs LR.</b> The SVM's margin buys +0.32 pp; LR's calibrated probabilities "
            "buy interface ergonomics. Both are carried forward — the demo lets the user switch, "
            "and the MoE experts use the LR objective.",
            "<b>What these numbers do not say.</b> They describe full-length essays from the "
            "same corpus family. Chapter 7 shows that almost every one of them collapses when "
            "the input is short or from a new domain — the honest reading of 99.5% is "
            "\"99.5% <i>on essays like the ones it trained on</i>\".",
        ]),
    ]
    return blocks
