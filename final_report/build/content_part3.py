"""Part III — Deep-dive research paper (formal treatment), preserved from
make_deepseek_paper.py: notation, equations, algorithm, ablations, appendices."""
from blocks import (part, h1, h2, h3, p, abstract, bullets, numbered, fig,
                    table, code, eq, pagebreak, spacer, subtitle, authors, refs)


def _pc(x, d=2):
    return f"{x * 100:.{d}f}%"


def _f4(x):
    return f"{x:.4f}"


def _f3(x):
    return f"{x:.3f}"


def part3(M):
    base = M["base"]
    routed = M["routed"]
    experts = M["experts"]
    ds = M["ds"]
    tr, te = ds["train.csv"], ds["test.csv"]
    hc, mo = ds["hc3_small.csv"], ds["modern_essays.csv"]
    summary = M["summary"]

    blocks = [
        part("Part III — Deep-Dive Research Paper",
             "“…Why Long-Text Detectors Fail on Short Text and How "
             "Mixture-of-Experts Fixes Them” — formal notation, equations, "
             "algorithm, ablation analysis, calibration experiments and technical "
             "appendices"),

        h1("Length-Expertized Authorship Detection: Why Long-Text Detectors Fail on "
           "Short Text and How Mixture-of-Experts Fixes Them"),
        subtitle("A detailed empirical study with small-text data, specialized "
                 "routing, type gating, and calibration"),
        authors("Pravesh Gedon Sharma (B240019) &amp; Ankit Rawat (B320032)  ·  "
                "Dept. Computer Science &amp; Engineering, NIT Sikkim  ·  "
                "Course Project (2026) — arXiv-style detailed report"),
        spacer(4),

        h2("Abstract"),
        abstract(
            "Supervised detectors for human vs machine-generated text report 99%+ "
            "accuracy on essays, yet practitioners observe rejection on short posts, "
            "chats and blurbs. We quantify this <b>length-generalization failure</b> "
            "and fix it with a <b>Mixture-of-Experts (MoE)</b> whose experts specialize "
            "by text length. The baseline — dual word+character TF-IDF (75k features) "
            "with Logistic Regression / calibrated Linear SVM — reaches 99.25–99.58% on "
            "full essays but falls to <b>67.4% at 25 words, 59.8% at one sentence</b>, "
            "with human accuracy 48.9% (false AI rate ≈51%) while AI recall stays "
            "≈98.7%: a one-sided collapse. We add <b>HC3 short Q&amp;A (80,811 answers, "
            "median 117 words)</b> and <b>modern 2024-LLM essays (17,997 texts, "
            "GPT-4/Claude/Llama3/Command R+)</b>, plus truncation augmentation, and "
            "train three experts — <b>short (≤150 words, 20k), long (151–400, 16.9k), "
            "verylong (&gt;400, 11.4k)</b> — each Word 15k + Char 25k + LR, routed "
            "deterministically by word count. A learned <b>type gate</b> (16 stylometric "
            "features → chat/casual/essay, 80.5% val) provides domain insight; "
            "<b>length-adaptive thresholds</b> (25w→0.32 … 150w→0.64) correct the FP/FN "
            "flip. Routed MoE restores 86.1% at 25 words, 93.1% at 50 words, 81.6% at "
            "one sentence, holds 98.96% full-text, and crucially scores <b>98.5% on "
            "unseen HC3 chat</b> (baseline 48.6%) and <b>95.7% on unseen 2024-LLM "
            "essays</b> (baseline 85.5%). Calibration adds +1.7 pp at 25 words. We "
            "release data pipelines, models, metrics and figures under "
            "<code>moe/</code> for full reproducibility."),
        p("<b>Keywords:</b> machine-generated text detection; mixture of experts; "
          "TF-IDF; stylometry; short-text classification; HC3; calibration."),

        h2("1  Introduction"),
        h3("1.1  The length blind spot in authorship detection"),
        p("Whether a passage is human-written or LLM-generated underlies "
          "academic-integrity, journalism and moderation decisions. Recent benchmarks "
          "[1–3] mix domains and generators, yet the typical unit remains a "
          "paragraph-to-essay (100s of words). Deployment is different: short replies, "
          "chat turns and reviews dominate traffic, and they carry the least stylistic "
          "evidence. A detector that needs 300 words to decide is undeployable for half "
          "its inputs — and, worse, <i>confidently wrong</i>: it labels short humans as "
          "AI (Section 7.1: human accuracy 48.9% at 25 words). This paper treats "
          "<b>length as a first-class routing variable</b>, not a nuisance to average "
          "over."),
        h3("1.2  Problem statement"),
        p("Let <i>x</i> be text, <i>y</i> ∈ {0,1} (0 human, 1 AI), <i>n</i>(<i>x</i>) "
          "word count. We seek <i>f</i>(<i>x</i>) → <i>ŷ</i> maximizing accuracy under "
          "length shift: training lengths ∼ D<sub>train</sub> (median "
          f"{tr['median']:.0f}w, P5 {tr['p5']:.0f}w) but test lengths <i>n</i> ≪ 150. "
          "Two phenomena matter: (i) <i>evidence sparsity</i> — TF-IDF counts scale "
          "with <i>n</i>, char n-grams need minimum support, stylometrics (TTR, "
          "burstiness) stabilize only with many tokens; (ii) <i>domain shift</i> — "
          "short text is chat/casual, not truncated essay, so essay-only training "
          "cannot transfer (hence HC3)."),
        h3("1.3  Contributions"),
        bullets([
            "<b>C1. Failure characterization.</b> A reproducible length ablation "
            "(prefix truncations + random windows) showing monotone collapse and an "
            "FP/FN flip, split by true class (Table {{t:p3_split}}, "
            "{{f:human_ai_split}}).",
            "<b>C2. Small-text data.</b> HC3 (80.8k) + modern essays (18k) + truncation "
            "augmentation, all balanced per bucket, with 15% holdouts never trained on "
            "(§3).",
            "<b>C3. MoE + type gate + calibration.</b> Three experts (val ≥97.0%), "
            "chat/casual/essay gate (80.5%), length-adaptive thresholds (§4, §6).",
            "<b>C4. Artifacts.</b> <code>moe/*.py</code>, <code>models/moe/</code>, "
            "<code>results/moe_*.csv</code>, Streamlit/CLI demos. Every table/figure is "
            "computed from these files.",
        ]),

        h2("2  Related Work"),
        h3("2.1  Supervised AI-text detection"),
        p("Supervised TF-IDF + linear classifiers remain strong in-domain baselines. "
          "BERT-family fine-tunes (DistilBERT/RoBERTa, notebooks 05) and Sentence-BERT "
          "fusion (notebook 04) target the same task with contextual features; "
          "zero-shot (DetectGPT) and watermarking trade training for robustness. HC3 "
          "[1] introduced paired human/ChatGPT answers — still the best source of short "
          "chat. Defactify [2] (73k news + 6 LLMs, 58.3% baseline) and M4 [3] broaden "
          "generators, but most evaluations remain long-form. Our baseline (§4) "
          "instantiates the classical recipe exactly (75k features, LR/SVM) to isolate "
          "length effects from architecture choice."),
        h3("2.2  Stylometry"),
        p("Lexical diversity (TTR = |types|/|tokens|), hapax ratio, burstiness σ/μ of "
          "sentence lengths, punctuation/capital rates, and readability (Flesch Reading "
          "Ease, FKGL, Gunning Fog, §4.2) capture authorial habit. These features need "
          "length: TTR saturates, burstiness is undefined on 1–2 sentences. We reuse "
          "<code>src/stylometrics.py</code> both as gate features and as UI "
          "explanations."),
        h3("2.3  Mixture of experts"),
        p("MoE routes inputs to specialized sub-networks [4]. Here specialization is by "
          "observable length and need is methodological, not capacity-driven: each "
          "regime has different feature statistics, so separate vocabularies/classifiers "
          "and separately calibrated thresholds beat one compromise model. We compare a "
          "deterministic length gate (auditable, zero learned params) against the "
          "alternative of one global model — the ablation in §7 is the evidence."),

        h2("3  Datasets"),
        h3("3.1  Original essays"),
        p(f"Train {tr['rows']:,} ({tr['human']:,}/class, {tr['size_mb']} MB), test "
          f"{te['rows']:,} ({te['human']:,} human / {te['ai']:,} AI, {te['size_mb']} "
          "MB). Unit: multi-paragraph academic essay. Test-set word statistics (full "
          "file): mean 394, median 363, P10 211, P90 617, max 1,650; human mean 423 vs "
          "AI 345. Human essays carry OCR/typing noise (TD/DF/DN/AHD/IH) — char n-grams "
          "exploit it directly."),
        h3("3.2  HC3 — genuine short chat (new)"),
        p("Hello-SimpleAI/HC3 [1], files <code>reddit_eli5.jsonl</code> (17,112 Q, "
          "67,671 answers), <code>open_qa.jsonl</code>, <code>finance.jsonl</code> "
          "downloaded via Hub (no dataset-script; works on datasets 5.x). Flattened to "
          f"one row per answer ≥40 chars → {hc['rows']:,} rows ({hc['human']:,} human / "
          f"{hc['ai']:,} AI). Word stats: mean {hc['mean']:.0f}, median "
          f"{hc['median']:.0f}, P10 28; {hc['pct_le_150']:.1f}% ≤150w, "
          f"{hc['pct_151_400']:.1f}% 151–400w, {hc['pct_gt_400']:.1f}% &gt;400w. "
          "Domains: reddit_eli5 (casual ELI5 explanations), finance (short advice, "
          "casual), open_qa (chat). 15% stratified holdout (12,122) is eval-only."),
        h3("3.3  Modern essays — 2024-LLM audit (new)"),
        p("artnitolog/llm-generated-texts: 3,000 essay prompts × 1 human + 8 "
          "continuations (GPT-4 Turbo/Omni, Claude 3 Opus, YandexGPT, GigaChat, Llama3 "
          "70B, Command R+). We keep English models (human + 5 LLMs) → "
          f"{mo['rows']:,} texts ({mo['human']:,} human / {mo['ai']:,} AI), mean "
          f"{mo['mean']:.0f}w, median {mo['median']:.0f}w, "
          f"{mo['pct_gt_400']:.1f}% &gt;400w. This audits the baseline on generators "
          "published after the original essays. Holdout 2,700 eval-only."),
        h3("3.4  Truncation augmentation"),
        p("Each 20k train essay contributes a 50/100/150w copy (prefix or random "
          "window, label preserved) for the short pool; ≥200w essays contribute 200/300w "
          "copies (long pool, filtered ≥151w after trunc — fixes a pilot bug where short "
          "originals polluted the long bucket). Prefix vs random-window accuracies are "
          "within ~1–2 pp (50w: 83.3% vs 81.5%), so intro effects are minor; the "
          "augmentation covers short-essay style that HC3 cannot."),
        table(
            "p3_pools",
            "MoE training pools (seed 42). Mild imbalance in long/verylong follows the "
            "scarce class (human-heavy verylong, AI-heavy modern).",
            ["Pool", "Sources (balanced caps)", "n"],
            [
                ["SHORT ≤150 w",
                 "HC3 ≤150w 7k/class + trunc-essay 3k/class",
                 "20,000 (10k/10k)"],
                ["LONG 151–400 w",
                 "HC3 long 4k/class + modern long 2k/class + base long 3k/class",
                 "16,864 (7,864/9,000)"],
                ["VERYLONG &gt;400 w",
                 "Base &gt;400w 7k/class + modern &gt;400w 2.5k/class",
                 "11,422 (6,321/5,101)"],
            ],
            widths=[38, 85, 41],
        ),
        h2("4  Problem Formulation and Method"),
        h3("4.1  Notation and objective"),
        eq("ŷ = 1[ P<sub>AI</sub>(x) ≥ θ<sub>n</sub> ]&nbsp;&nbsp;with n = n(x) word "
           "count;&nbsp;&nbsp; MoE: P(x) = P<sub>e(x)</sub><sup>AI</sup>(x);&nbsp;&nbsp;"
           "e(x) = short / long / verylong gate"),
        p("We maximize accuracy/F1 under length shift, reporting precision/recall "
          "splits because accuracy alone hides the FP/FN flip."),
        h3("4.2  Text representation"),
        p("Word TF-IDF (1,2, min_df 2, sublinear) + char_wb (3,5, min_df 3), fused "
          "sparsely:"),
        eq("tfidf(t, d) = tf(t, d) · log( (1+N) / (1+df(t)) )&nbsp;;&nbsp;&nbsp;"
           "F(x) = [ F<sub>word</sub>(x) ; F<sub>char</sub>(x) ] ∈ ℝ<sup>V</sup>"),
        p("Baseline V = 75,000 (25k+50k); each expert V = 40,000 (15k+25k). LR "
          "minimizes −log-likelihood with L2 (C=1.5), LinearSVC uses hinge + Platt "
          "scaling. Stylometrics (16 features, §4.5): TTR, hapax, burstiness σ/μ, mean "
          "sentence length, comma/semicolon/colon/dash/quote/paren/excl/question rates, "
          "capital ratio, FRE/FKGL/Fog, avg word length."),
        eq("FRE = 206.835 − 1.015 · (N/S) − 84.6 · (syll/N)&nbsp;;&nbsp;&nbsp;FKGL = "
           "0.39 · (N/S) + 11.8 · (syll/N) − 15.59&nbsp;;&nbsp;&nbsp;Fog = 0.4 · "
           "[ (N/S) + 100 · (poly/N) ]"),
        h3("4.3  Baseline and length ablation (motivation)"),
        p("Baseline training follows §4.2 exactly (20k subsample, 16k/4k split). Length "
          "ablation truncates the 5k held-out to first-N words/sentences. Result "
          "(§7.1): monotone degradation, confidence collapse (0.93→0.79), and — "
          "critically — human accuracy 48.9% @25w vs AI 98.7% (Table {{t:p3_split}}). "
          "This asymmetry proves the failure is <i>bias</i>, not just difficulty: short "
          "humans look like the AI class the model expects."),
        h3("4.4  MoE architecture"),
        eq("e(x) = short if n ≤ 150, long if 151 ≤ n ≤ 400, verylong if n &gt; 400;"
           "&nbsp;&nbsp; P(x) = P<sub>e(x)</sub>(x);&nbsp;&nbsp; expert = LR ∘ "
           "TFIDF<sub>e</sub>"),
        p("Gating is deterministic (interpretable, no routing collapse). Each expert "
          "trains on its pool only (80/20 within pool). Inference additionally reports "
          "type (§4.5) and warns if n &lt; 25 (&lt; 50)."),
        fig("moe_arch", 0.9, repeat=True),
        code("""Algorithm 1 — MoE training (moe/train_moe.py)
Input: base 20k essays, HC3 train 68.7k, modern train 15.3k, seed 42
1. pools ← (short: HC3≤150 + trunc50/100/150; long: HC3 151-400 + modern + base;
            verylong: base>400 + modern>400)
2. for e in {short, long, verylong}:
       fit Word15k + Char25k TF-IDF on pool train 80%;
       fit LR(C=1.5, lbfgs, max_iter=1000);  save expert_e.pkl
3. gate ← LR(StandardScaler(16 stylometrics)) on 7.5k (2.5k/chat/casual/essay);
       save type_gate.pkl
4. routed eval on 5k test + HC3/modern holdouts;
       baseline eval on the same conditions"""),
        h3("4.5  Type gate"),
        p("Labels: HC3 <code>open_qa</code>→chat (4k avail., cap 2.5k), HC3 "
          "eli5/finance + truncated essays → casual, full/modern essays → essay (2.5k "
          "each). Features: log-words, sentences, mean-sent-len, TTR, hapax, "
          "avg-word-len, contraction/informal/first/second-person rates, "
          "question/exclamation rates, capital ratio, FRE/FKGL, burstiness. Val: "
          f"{summary['type_gate_accuracy'] * 100:.1f}% (essay P/R/F1 0.90/0.92/0.91; "
          "chat 0.74/0.84/0.79; casual 0.77/0.66/0.71). Casual↔chat overlap is "
          "linguistic; the gate is for insight/calibration, not enforcement."),
        fig("type_gate", 0.74, repeat=True),
        h3("4.6  Calibration"),
        p("Default θ=0.50 assumes symmetric costs. On calib half (2.5k), F1-optimal θ "
          "rises with length (25w 0.32 … 150w 0.64) because ultra-short errs FN "
          "(under-call AI) while 100–150w errs FP. We ship "
          "<code>θ(n)=0.28+0.0024n</code> clamped [0.32,0.64] for short, 0.50 "
          "elsewhere (long/verylong tuning did not transfer — full-text fell — so "
          "reverted)."),
        h2("5  Experimental Setup"),
        h3("5.1  Environment and protocol"),
        p("Python 3.13, scikit-learn, joblib, reportlab. All splits seed 42 "
          "(calibration 999). Held-out: test.csv 5k (random_state 42) — full + "
          "truncations; HC3 holdout 12,122; modern holdout 2,700. Training time 130 s "
          "(TF-IDF 6/15/25 s per expert). No leakage: test.csv never trained; "
          "HC3/modern holdouts excluded from pools."),
        h3("5.2  Metrics"),
        p("Accuracy, precision, recall, F1 (AI as positive), ROC-AUC, confusion "
          "matrices. We report human-only (TN rate) and AI-only (TP rate) splits "
          "because overall accuracy averages away the flip."),

        h2("6  Main Results"),
        h3("6.1  Per-expert and routed"),
        table(
            "p3_experts",
            "Per-expert validation (results/moe_expert_metrics.csv).",
            ["Expert", "Train/Val", "Acc", "Prec", "Rec", "F1", "AUC"],
            [[r["expert"], f"{int(r['n_train']):,}/{int(r['n_val']):,}",
              _f4(r["accuracy"]), _f4(r["precision"]), _f4(r["recall"]),
              _f4(r["f1"]), _f4(r["roc_auc"])] for r in experts],
            widths=[26, 32, 22, 22, 22, 22, 24],
        ),
        fig("moe_experts", 0.88, repeat=True),
        p("Routed MoE: full " + _pc(routed["full"]["accuracy"]) + " (P 0.993, R 0.979, "
          "F1 0.986) vs baseline " + _pc(base["full"]["accuracy"]) + " — specialization "
          "costs 0.3 pp. Buckets on full: verylong 99.59% (n=1,958), long 98.81% "
          "(2,945), short 90.72% (97). The win is short + new domains "
          "({{f:acc_vs_len}}–{{f:cross_domain}}, Table {{t:p3_headline}})."),
        fig("acc_vs_len", 0.9, repeat=True),
        fig("cross_domain", 0.8, repeat=True),
        table(
            "p3_headline",
            "Headline deltas (results/moe_*.csv).",
            ["Condition", "Baseline acc", "MoE acc", "Δ"],
            [[lab, _pc(base[k]["accuracy"]), _pc(routed[k]["accuracy"]),
              f"{(routed[k]['accuracy'] - base[k]['accuracy']) * 100:+.1f} pp"]
             for k, lab in [("full", "full"), ("first_25_words", "25w"),
                            ("first_50_words", "50w"), ("first_1_sent", "1 sent"),
                            ("hc3_holdout", "HC3"), ("modern_holdout", "modern")]],
            widths=[44, 34, 34, 30],
        ),
        h3("6.2  Calibration (eval half)"),
        p("Adaptive thresholds: 25w "
          f"{_pc(M['calib_map'][('first_25_words', '0.5')]['accuracy'])}→"
          f"{_pc(M['calib_map'][('first_25_words', 'calibrated')]['accuracy'])} (F1 "
          f"{_f3(M['calib_map'][('first_25_words', '0.5')]['f1'])}→"
          f"{_f3(M['calib_map'][('first_25_words', 'calibrated')]['f1'])}, recall "
          "0.68→0.84), 150w "
          f"{_pc(M['calib_map'][('first_150_words', '0.5')]['accuracy'])}→"
          f"{_pc(M['calib_map'][('first_150_words', 'calibrated')]['accuracy'])}, "
          "50w and 100w move slightly in opposite directions, full softens "
          f"({_pc(M['calib_map'][('full', '0.5')]['accuracy'])}→"
          f"{_pc(M['calib_map'][('full', 'calibrated')]['accuracy'])}). HC3 natural "
          "short −0.7 pp (human-heavy FPs) — thresholds are essay-tuned; keep the "
          "&lt;50w warning. Long/verylong stayed at 0.50."),
        fig("threshold_curve", 0.84, repeat=True),
        h2("7  Ablations and Analysis"),
        h3("7.1  Length ablation — the flip in detail"),
        table(
            "p3_split",
            "Length-conditional accuracy split (human TN rate vs AI TP rate). MoE "
            "rescues humans (+47.9 pp @25w) at AI-recall cost (−30.7 pp) — net acc "
            "still wins.",
            ["Words", "Base H", "MoE H", "Base AI", "MoE AI", "Base acc", "MoE acc"],
            [
                ["25", "48.9%", "96.8%", "98.7%", "67.9%", "67.4%", "86.1%"],
                ["50", "74.2%", "96.2%", "98.7%", "87.9%", "83.3%", "93.1%"],
                ["75", "86.2%", "95.8%", "98.8%", "94.0%", "90.9%", "—"],
                ["100", "91.3%", "94.7%", "98.8%", "96.1%", "94.1%", "95.2%"],
                ["150", "96.5%", "94.3%", "98.9%", "97.7%", "97.4%", "95.6%"],
                ["200", "98.3%", "98.0%", "98.6%", "97.0%", "98.4%", "97.6%"],
                ["300", "99.2%", "98.9%", "98.7%", "97.6%", "99.0%", "—"],
                ["full", "99.6%", "99.6%", "98.7%", "97.9%", "99.3%", "99.0%"],
            ],
            widths=[20, 24, 22, 22, 20, 24, 24],
        ),
        fig("human_ai_split", 0.95, repeat=True),
        p("Interpretation: at 25w the baseline calls AI 98.7% of the time (any class) "
          "— length is the feature. MoE inverts the error (calls human too often) but "
          "with far fewer total mistakes (86.1% vs 67.4%). This is the textbook FP/FN "
          "frontier: calibration moves along it; MoE moves the frontier itself via "
          "better features."),
        fig("pr_flip", 0.7, repeat=True),
        h3("7.2  Sentence ablation"),
        p("1/2/3 sentences: baseline 59.8%/74.5%/83.4% → MoE "
          f"{_pc(routed['first_1_sent']['accuracy'])}/"
          f"{_pc(routed['first_2_sent']['accuracy'])}/"
          f"{_pc(routed['first_3_sent']['accuracy'])}. Sentence truncations are harder "
          "than word truncations at matched budget (more padding loss), confirming "
          "sentence structure carries signal the short expert partially recovers via "
          "char features."),
        h3("7.3  Type analysis"),
        p("Type gate: essay easiest (long formal → distinct), chat next (short "
          "informal markers), casual hardest (middle). Routing insight: short+chat "
          "inputs have the lowest confidence and need warnings; short+essay (truncated) "
          "input needs the low threshold. Reporting type alongside label is a "
          "human-factor result, not just accuracy."),
        h3("7.4  What did not help"),
        p("Tuning long/verylong thresholds hurt full-text (98.96%→98.60% on eval) — "
          "small calib slices overfit. Global 0.60 (best for 100–150w essay "
          "truncations) hurt HC3 natural chat. Conclusion: calibrate only where the "
          "failure is large and monotonic (short), not everywhere."),

        h2("8  Limitations and Future Work"),
        bullets([
            "<b>Ultra-short floor.</b> Even MoE is 86% at 25w with wide uncertainty. "
            "Training saw ≥50w; adding 10–35w + few-shot chat would likely beat "
            "thresholds.",
            "<b>Boundary blending.</b> 150w essays prefer baseline (97.4% vs 95.6%); "
            "soft routing (weighted P across short/long) could interpolate.",
            "<b>Threshold transfer.</b> Essay-tuned θ hurts human-heavy chat "
            "(−0.7 pp). Per-type thresholds (essay vs chat) are the next step.",
            "<b>Generators.</b> HC3 is one 2023 ChatGPT snapshot; modern holdout is 5 "
            "models. Paraphrase/adversarial eval is missing.",
            "<b>Scale.</b> Full 290k training might add 0.1–0.3 pp; transformers "
            "(DistilBERT/RoBERTa) with chunking remain the Phase-2 ceiling test.",
        ]),
        p("Ethics: P(AI) is a screening signal, not proof. Short-text warnings and "
          "type reporting are required UI, not optional captions."),

        h2("9  Conclusion"),
        p("Length is not a detail — it is the task when evaluating short text. One "
          "global TF-IDF model buys 99% on essays by learning to call short text AI. "
          "Specializing by length, grounding short experts in real chat data, labeling "
          "chat/casual/essay, and calibrating per length recovers +10 to +50 pp where "
          "it matters, with a 0.3 pp full-text tax. We release the full stack so the "
          "claim is checkable: <code>moe/download_small_text.py → train_moe.py → "
          "calibrate_thresholds.py → predict_moe.py</code>."),

        h2("References"),
        refs([
            "[1] B. Guo et al., “How Close is ChatGPT to Human Experts? "
            "arXiv:2301.07597 (2023); Hello-SimpleAI/HC3 (reddit_eli5/open_qa/finance).",
            "[2] R. Roy et al., “A Comprehensive Dataset for Human vs AI Generated Text "
            "Detection.” arXiv:2510.22874 (2025) (Defactify).",
            "[3] B. Wang et al., “M4: Multi-Generator, Multi-Domain, Multi-Task, "
            "Multi-Label.” arXiv:2310.13607 (2023).",
            "[4] N. Shazeer et al., “Outrageously Large Neural Networks: MoE.” "
            "arXiv:1701.06538 (2017).",
            "[5] T. Hastie et al., <i>The Elements of Statistical Learning</i>. "
            "Springer (2009).",
            "[6] S. Robertson and H. Zaragoza, “BM25 &amp; TF-IDF.” <i>Found. Trends "
            "IR</i> (2009).",
            "[7] N. Reimers and I. Gurevych, “Sentence-BERT.” EMNLP (2019); T. Sanh et "
            "al., “DistilBERT” (2019); Y. Liu et al., “RoBERTa” (2019).",
            "[8] <i>artnitolog/llm-generated-texts</i> (GPT-4 Turbo/Omni, Claude 3 Opus, "
            "Llama3, Command R+) — Hugging Face.",
            "[9] E. Mitchell et al., “DetectGPT.” arXiv:2301.11305 (2023) — zero-shot "
            "baseline family.",
        ]),

        h2("Appendices"),
        h3("A  Hyperparameters"),
        table(
            "p3_hyper",
            "Hyperparameters (all seeded).",
            ["Component", "Setting"],
            [
                ["Baseline word TF-IDF",
                 "(1,2) 25k, min_df 2, sublinear, clean_text_for_tfidf"],
                ["Baseline char TF-IDF",
                 "char_wb (3,5) 50k, min_df 3, sublinear"],
                ["Baseline classifiers",
                 "NB α=0.1; LR C=1.5 lbfgs 1000; LinearSVC C=1.0 + CalibratedCV(3)"],
                ["Experts (each)",
                 "Word 15k + Char 25k → LR C=1.5 lbfgs 1000"],
                ["Type gate",
                 "StandardScaler + LR C=1.0 lbfgs 1000 on 16 features"],
                ["Thresholds",
                 "short θ(n)=0.28+0.0024n clamped [0.32, 0.64]; long/verylong 0.50"],
            ],
            widths=[55, 109],
        ),
        h3("B  Reproducibility checklist"),
        bullets([
            "Data splits: train subsample (10k/class, seed 42), 80/20 per pool (seed "
            "42), calib 2.5k/eval 2.5k (seed 999), HC3/modern holdouts 15% (seed 42).",
            "Artifacts: <code>models/*.pkl</code> (baseline), "
            "<code>models/moe/expert_*.pkl, type_gate.pkl, thresholds.json</code>; "
            "metrics <code>results/moe_*.csv</code>.",
            "Indirection: truncation code is in <code>moe/type_features.py</code> "
            "(truncate_words/random_window); routing in "
            "<code>moe/train_moe.py::route_predict</code>.",
            "Documents: the four original generators "
            "(<code>make_report.py</code>, <code>moe/make_moe_appendix.py</code>, "
            "<code>make_research_paper.py</code>, <code>make_deepseek_paper.py</code>) "
            "produced the source PDFs for Parts I–IV; this consolidated report rebuilds "
            "all four from the same verified CSVs.",
        ]),
        h3("C  Full routed table"),
        table(
            "p3_routed",
            "Routed MoE, all evaluation conditions (results/moe_routed_metrics.csv).",
            ["Condition", "Acc", "Prec", "Rec", "F1", "AUC"],
            [[k, _f4(routed[k]["accuracy"]), _f4(routed[k]["precision"]),
              _f4(routed[k]["recall"]), _f4(routed[k]["f1"]),
              _f4(routed[k]["roc_auc"])]
             for k in ["full", "first_25_words", "first_50_words", "first_100_words",
                       "first_150_words", "first_200_words", "first_1_sent",
                       "first_2_sent", "first_3_sent", "hc3_holdout",
                       "modern_holdout"]],
            widths=[40, 25, 25, 25, 25, 25],
        ),
        p("Baseline counterpart in <code>results/moe_baseline_comparison.csv</code>; "
          "calibration detail in <code>results/moe_calibrated_metrics.csv</code>."),
    ]
    return blocks
