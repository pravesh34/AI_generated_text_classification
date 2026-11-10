"""Part II — Research paper (conference format), preserved from
make_research_paper.py and rebuilt on verified metrics."""
from blocks import (part, h1, h2, h3, p, abstract, bullets, fig, table, code,
                    eq, pagebreak, spacer, subtitle, authors, refs)


def _pc(x, d=2):
    return f"{x * 100:.{d}f}%"


def _f4(x):
    return f"{x:.4f}"


def _f3(x):
    return f"{x:.3f}"


def part2(M):
    base = M["base"]
    routed = M["routed"]
    experts = M["experts"]
    calib = M["calib"]
    val = M["val_rows"]
    ds = M["ds"]
    tr, te = ds["train.csv"], ds["test.csv"]
    hc, mo = ds["hc3_small.csv"], ds["modern_essays.csv"]

    def ev(cond, thr):
        for r in calib:
            if r["subset"] == "eval" and r["condition"] == cond and r["threshold"] == thr:
                return r
        raise KeyError((cond, thr))

    blocks = [
        part("Part II — Research Paper",
             "“Length-Expertized Authorship Detection: A Mixture-of-Experts for "
             "Human vs AI-Generated Text” — conference-format paper, preserved as "
             "originally written and re-verified against the result CSVs"),

        h1("Length-Expertized Authorship Detection: A Mixture-of-Experts for "
           "Human vs AI-Generated Text"),
        subtitle("Fixing the Short-Text Collapse with Dedicated Small-Text Data, "
                 "Routing, and Calibration"),
        authors("Pravesh Gedon Sharma (B240019) &amp; Ankit Rawat (B320032)  ·  "
                "Department of Computer Science &amp; Engineering  ·  "
                "National Institute of Technology Sikkim  ·  Course Project (2026)"),
        spacer(4),

        h2("Abstract"),
        abstract(
            "Classical TF-IDF detectors for human vs AI text achieve 99%+ accuracy on "
            "long essays but collapse on short text: 67.4% at 25 words and 59.8% at one "
            "sentence, with humans systematically biased to AI. We diagnose the length "
            "failure, add genuine small-text data (HC3 Q&amp;A, 80,811 answers, median "
            "117 words; plus 17,997 modern-LLM essays from GPT-4, Claude 3, Llama 3 and "
            "Command R+), and train a <b>Mixture-of-Experts (MoE)</b>: a deterministic "
            "length gate (short ≤150 / long 151–400 / verylong &gt;400 words) routing to "
            "three TF-IDF + Logistic Regression experts, plus a learned type gate "
            "(chat / casual / essay, 80.5% validation) and length-adaptive thresholds. "
            "On 5,000 held-out essays, MoE lifts short-text accuracy to 86.1% at 25 "
            "words, 93.1% at 50 words and 81.6% at one sentence, while holding 98.96% on "
            "full text. Cross-domain, MoE scores 98.5% on unseen HC3 chat answers "
            "(baseline 48.6%) and 95.7% on unseen 2024-LLM essays (baseline 85.5%). "
            "Calibration (25w→0.32 … 150w→0.64) further adds +1.7 pp at 25 words. The "
            "system is released as <code>moe/</code> with routers, metrics and this "
            "paper's figures."),
        p("<b>Keywords:</b> AI text detection; mixture of experts; TF-IDF; stylometrics; "
          "short-text classification; HC3; authorship attribution."),

        h2("1  Introduction"),
        p("Large language models (ChatGPT, GPT-4, Claude) make fluent essays trivially "
          "available, creating an integrity problem in academia, journalism and "
          "moderation. Most published detectors are evaluated on documents hundreds of "
          "words long. Real deployment, however, includes tweets, chat replies, product "
          "reviews and single-sentence blurbs. We ask: <i>does a strong long-text "
          "detector survive truncation to short text, and what architecture fixes "
          "it?</i>"),
        p("Our Phase-1 baseline — dual word (1,2) + character (3,5) TF-IDF (75k "
          "features) with Logistic Regression (LR) and calibrated Linear SVM — scores "
          "99.25–99.53% validation and 99.26–99.58% held-out accuracy on 20,000 "
          "balanced essays (Section 4). Truncating those same held-out essays destroys "
          "it: first 25 words → 67.4%, first 1 sentence → 59.8% (Section 5). AI recall "
          "stays ≈0.98 while humans flip to AI at ≈51% (25w). The model never saw short "
          "text (training median 356 words, 5th percentile 177)."),
        p("This paper makes three contributions. (1) A <b>length ablation</b> "
          "quantifying the failure per word/sentence budget. (2) <b>Small-text data + "
          "MoE</b>: HC3 short Q&amp;A, modern 2024-LLM essays, and truncation "
          "augmentation, routed to length experts with a chat/casual/essay type gate. "
          "(3) <b>Threshold calibration</b> showing one global threshold is wrong per "
          "length. Code, datasets, models and metrics are under <code>moe/</code> and "
          "<code>results/moe_*.csv</code>."),

        h2("2  Related Work"),
        p("AI-text detection spans zero-shot (DetectGPT, likelihood curvature), "
          "watermarking, and supervised classifiers. HC3 (Guo et al.) [1] pairs human "
          "and ChatGPT answers to the same questions across domains; subsequent "
          "benchmarks (Defactify [2], M4 [3]) widen generators and domains but still "
          "skew long. OpenAI's classifier famously failed under 1,000 characters — the "
          "same cliff we replicate. Classical TF-IDF + linear SVM remains competitive "
          "on in-domain essays (Section 4) but, as we show, is brittle across lengths "
          "and generators. Mixture-of-experts routing by surface properties (here "
          "length) is a standard technique for specialized regimes [4]; our novelty is "
          "pairing length experts with per-length calibration and a learned text-type "
          "gate on stylometric features (<code>src/stylometrics.py</code>: TTR, hapax, "
          "burstiness, Flesch ease/grade, punctuation rates)."),

        h2("3  Datasets"),
        p("Original essays: <code>data/train.csv</code> (290,300 rows, 145,150 per "
          "class) and <code>data/test.csv</code> (97,447 rows, 61,159 human / 36,288 "
          "AI), columns <code>text, generated</code>. Texts are multi-paragraph "
          "academic essays: full test-set word statistics mean 394, median 363, 10th "
          "percentile 211, 90th percentile 617, max 1,650; human mean 423 vs AI 345 "
          "(Table {{t:p2_ds}}, {{f:class_dist}}–{{f:length_hist}}). Training uses a "
          "20,000 balanced subsample (10k/class, seed 42), 80/20 stratified → "
          "16,000/4,000."),
        table(
            "p2_ds",
            "Dataset scale (Part II). Held-out n = 5,000 (seed 42) for all comparisons.",
            ["Dataset", "Size", "Human / AI", "Median length", "Role"],
            [
                ["train.csv", f"{tr['rows']:,}",
                 f"{tr['human']:,} / {tr['ai']:,}",
                 f"{tr['median']:.0f} w (P5 {tr['p5']:.0f} w)", "baseline train"],
                ["test.csv", f"{te['rows']:,}",
                 f"{te['human']:,} / {te['ai']:,}",
                 f"{te['median']:.0f} w", "held-out eval"],
                ["HC3 (reddit_eli5<br/>+ open_qa + finance)", f"{hc['rows']:,}",
                 f"{hc['human']:,} / {hc['ai']:,}",
                 f"{hc['median']:.0f} w, {hc['pct_le_150']:.0f}% ≤150 w",
                 "SHORT chat/casual"],
                ["Modern essays<br/>(GPT-4/Claude/Llama3/Cmd R+)", f"{mo['rows']:,}",
                 f"{mo['human']:,} / {mo['ai']:,}",
                 f"{mo['median']:.0f} w, {mo['pct_gt_400']:.0f}% &gt;400 w",
                 "LONG/VERYLONG 2024"],
            ],
            widths=[48, 20, 32, 36, 32],
        ),
        fig("class_dist", 0.86, repeat=True),
        fig("length_hist", 0.86, repeat=True),
        p("New data (via <code>moe/download_small_text.py</code>): <b>HC3</b> flattened "
          "to one row per answer (≥40 chars) — human vs ChatGPT to the same question, "
          "the only source of genuine short chat. <b>Modern essays</b> (artnitolog): "
          "3,000 prompts × (human + 5 English LLMs), 2024 generators absent from the "
          "original essays. Truncation augmentation copies each train essay at "
          "50/100/150 w (short pool) and 200/300 w (long pool) so short-essay style is "
          "covered. Pools are class-balanced (seed 42): short 20k, long 16.9k, verylong "
          "11.4k; HC3/modern 15% holdouts are never trained on."),

        h2("4  Methodology"),
        p("<b>Baseline (Phase 1).</b> NFKC + whitespace + lowercase for TF-IDF "
          "(<code>src/preprocessing.py</code>). Word TF-IDF (1,2, 25k, min_df 2) + "
          "char_wb (3,5, 50k, min_df 3), sublinear TF, fused via <code>hstack</code> → "
          "75k sparse features. Classifiers: MultinomialNB (α=0.1), LR (C=1.5, lbfgs), "
          "LinearSVC + 3-fold calibration (C=1.0). Validation (4k): NB 95.55%, LR "
          "99.25% (F1 0.992, AUC 0.9987), SVM 99.53% (F1 0.995, AUC 0.9991); held-out "
          "(5k): LR 99.26%, SVM 99.58% (Table {{t:p2_val}})."),
        table(
            "p2_val",
            "Baseline validation, 4,000 essays (results/baseline_ml_metrics.csv).",
            ["Model (val 4k)", "Acc", "Prec", "Rec", "F1", "AUC"],
            [[r["model"], _f4(r["accuracy"]), _f4(r["precision"]), _f4(r["recall"]),
              _f4(r["f1"]), _f4(r["roc_auc"])] for r in val],
            widths=[52, 22, 22, 22, 22, 26],
        ),
        fig("model_cmp", 0.9, repeat=True),
        p("<b>Short-text ablation.</b> Evaluated the frozen baseline on first-N-words "
          "and first-N-sentences truncations of the 5k held-out (and random windows — "
          "same result, so not an intro effect). At 25w accuracy 67.4% (P 0.53, "
          "R 0.99): 1,604 of 3,142 humans called AI. At 50w 83.3%, 100w 94.1%, 200w "
          "98.4%; 1/2/3 sentences 59.8%/74.5%/83.4%. Mean LR confidence drops 0.93 "
          "(200w) → 0.79 (25w). Natively-short humans barely exist in test.csv (min "
          "human 48 w), so truncated-human accuracy is the honest probe."),
        p("<b>MoE.</b> Length gate ≤150 / 151–400 / &gt;400 words (cliff + median). "
          "Each expert: Word 15k + Char 25k + LR(C=1.5) on its pool only — features "
          "specialize (short vocab avoids long-essay dilution). Type gate: LR on 16 "
          "stylometric features → chat/casual/essay (7.5k balanced; HC3 "
          "<code>open_qa</code>→chat, eli5/finance+truncated→casual, full/modern→essay); "
          "val 80.5% (essay F1 0.91, chat 0.79, casual 0.71). Inference routes by "
          "length and reports type + warning for &lt;50w."),
        fig("moe_arch", 0.9, repeat=True),

        h2("5  Experiments"),
        p("Environment: Python 3.13, scikit-learn, TF-IDF as above (per-expert "
          "15k+25k). Training (<code>moe/train_moe.py</code>, 130 s): short TF-IDF "
          "6.4 s, long 15 s, verylong 25 s. Evaluation: 5k held-out essays (full + "
          "truncations), 12,122 HC3 holdout, 2,700 modern holdout (both unseen). "
          "Calibration (<code>moe/calibrate_thresholds.py</code>): 5k split 2,500 "
          "calib / 2,500 eval (seed 999), F1-max thresholds per length, eval on the "
          "other half + HC3 transfer. Metrics: accuracy, precision, recall, F1, "
          "ROC-AUC, confusion matrices."),

        h2("6  Results"),
        p("<b>Per-expert (val).</b> Short 97.00% (F1 0.970, AUC 0.996), long 98.19% "
          "(0.983/0.998), verylong 98.42% (0.982/0.996) (Table {{t:p2_experts}}, "
          "{{f:moe_experts}})."),
        table(
            "p2_experts",
            "MoE per-expert validation (results/moe_expert_metrics.csv).",
            ["Expert", "Train/Val", "Acc", "F1", "AUC"],
            [[r["expert"], f"{int(r['n_train']):,}/{int(r['n_val']):,}",
              _f4(r["accuracy"]), _f4(r["f1"]), _f4(r["roc_auc"])] for r in experts],
            widths=[32, 38, 30, 30, 30],
        ),
        fig("moe_experts", 0.88, repeat=True),
        p("<b>Routed vs baseline.</b> MoE holds full-text (98.96% vs 99.26%, −0.3 pp "
          "specialization cost; verylong bucket 99.59%) and dominates short + "
          "cross-domain ({{f:acc_vs_len}}–{{f:cross_domain}}, Table {{t:p2_headline}}). "
          "HC3 is the headline: baseline 48.6% (it never learned chat), MoE 98.5%. "
          "Modern essays: 85.5%→95.7%."),
        fig("acc_vs_len", 0.9, repeat=True),
        fig("cross_domain", 0.8, repeat=True),
        table(
            "p2_headline",
            "Headline deltas (results/moe_routed_metrics.csv vs "
            "moe_baseline_comparison.csv).",
            ["Condition (5k)", "Baseline", "MoE", "Δ"],
            [[lab,
              _pc(base[k]["accuracy"]), _pc(routed[k]["accuracy"]),
              f"{(routed[k]['accuracy'] - base[k]['accuracy']) * 100:+.1f} pp"]
             for k, lab in [("full", "full"), ("first_25_words", "25w"),
                            ("first_50_words", "50w"), ("first_1_sent", "1 sent"),
                            ("hc3_holdout", "HC3"), ("modern_holdout", "modern")]],
            widths=[45, 32, 32, 30],
        ),
        fig("confusion", 0.9, repeat=True),
        p("<b>Calibration.</b> Optima move with length (25w 0.32, 50w 0.40, 100w 0.56, "
          "150w 0.64 → <code>th=0.28+0.0024n</code>); long/verylong kept at 0.50. "
          "Eval-half results are in Table {{t:p2_calib}}: 25w "
          f"{_pc(ev('first_25_words', '0.5')['accuracy'])}→"
          f"{_pc(ev('first_25_words', 'calibrated')['accuracy'])} (F1 "
          f"{_f3(ev('first_25_words', '0.5')['f1'])}→"
          f"{_f3(ev('first_25_words', 'calibrated')['f1'])}, recall 0.68→0.84), 150w "
          f"{_pc(ev('first_150_words', '0.5')['accuracy'])}→"
          f"{_pc(ev('first_150_words', 'calibrated')['accuracy'])}, 100w "
          f"{_pc(ev('first_100_words', '0.5')['accuracy'])}→"
          f"{_pc(ev('first_100_words', 'calibrated')['accuracy'])}, while 50w "
          f"({_pc(ev('first_50_words', '0.5')['accuracy'])}→"
          f"{_pc(ev('first_50_words', 'calibrated')['accuracy'])}) and full text "
          f"({_pc(ev('full', '0.5')['accuracy'])}→"
          f"{_pc(ev('full', 'calibrated')['accuracy'])}) soften slightly — the reason "
          "the shipped configuration tunes only the short bucket. Caveat: HC3 natural "
          "short −0.7 pp (human-heavy) — thresholds are essay-tuned, so the &lt;50w "
          "warning stays. Demo flip: 31w borderline (P=0.374, th=0.354) goes "
          "Human→AI (correct-ish but low-confidence, flagged)."),
        table(
            "p2_calib",
            "Calibration on the eval half (acc/F1). Detail in "
            "results/moe_calibrated_metrics.csv.",
            ["Eval-2500", "th=0.50", "Adaptive"],
            [[lab,
              f"{_pc(ev(k, '0.5')['accuracy'])} / {_f3(ev(k, '0.5')['f1'])}",
              f"{_pc(ev(k, 'calibrated')['accuracy'])} / {_f3(ev(k, 'calibrated')['f1'])}"]
             for k, lab in [("first_25_words", "25w"), ("first_50_words", "50w"),
                            ("first_100_words", "100w"), ("first_150_words", "150w"),
                            ("full", "full")]],
            widths=[40, 60, 60],
        ),
        fig("pr_flip", 0.7, repeat=True),
        fig("type_gate", 0.78, repeat=True),

        h2("7  Discussion and Limitations"),
        bullets([
            "<b>Why MoE works:</b> short text needs different evidence (word choices "
            "without discourse, chat markers) and different vocabularies; per-bucket "
            "TF-IDF prevents dilution. HC3 was the unlock — without it, there is no "
            "signal to learn.",
            "<b>Why calibration matters:</b> optimal thresholds encode the FP/FN "
            "tradeoff per regime. One threshold cannot serve both ultra-short "
            "(FN-prone) and 150w (FP-prone) text. Linear-adaptive th is a cheap, "
            "auditable compromise; per-domain thresholds would be better.",
            "<b>Limits:</b> (i) &lt;25–50w remains weak even with MoE (86–93%) — "
            "irreducible signal loss; (ii) 150w essays still prefer the baseline "
            "(97.4% vs 95.6% pre-calibration) — expert blending at boundaries is "
            "future work; (iii) short expert saw truncations ≥50w, so 25w is "
            "extrapolation; adding 10–35w training would likely help more than "
            "thresholds; (iv) type gate (80.5%) is for insight, not enforcement; "
            "casual recall 0.66 reflects real overlap.",
            "<b>Reproducibility:</b> all trims use seed 42/999; datasets via Hub "
            "JSONL/parquet need no script datasets; artifacts in "
            "<code>models/moe/</code>.",
        ]),

        h2("8  Conclusion"),
        p("Single-model TF-IDF detectors silently assume essay length. Once length is "
          "treated as a routing variable — with dedicated small-text data, specialized "
          "experts, a type gate, and per-length thresholds — short-text accuracy rises "
          "by 10–50 pp and cross-domain retrieval becomes viable, at a 0.3 pp full-text "
          "cost. The released <code>moe/</code> stack (download → train → calibrate → "
          "predict) reproduces every number in this paper's tables and figures."),

        h2("References"),
        refs([
            "[1] B. Guo et al., “How Close is ChatGPT to Human Experts? HC3,” "
            "arXiv:2301.07597, 2023; Hello-SimpleAI/HC3 on Hugging Face.",
            "[2] R. Roy et al., “A Comprehensive Dataset for Human vs AI Generated Text "
            "Detection” (Defactify), arXiv:2510.22874, 2025.",
            "[3] B. Wang et al., “M4: Multi-Generator, Multi-Domain, Multi-Task, "
            "Multi-Label,” arXiv:2310.13607, 2023.",
            "[4] N. Shazeer et al., “Outrageously Large Neural Networks: "
            "Mixture-of-Experts,” arXiv:1701.06538, 2017.",
            "[5] T. Hastie, R. Tibshirani and J. Friedman, <i>The Elements of "
            "Statistical Learning</i>, Springer, 2009.",
            "[6] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: "
            "BM25 &amp; TF-IDF,” <i>Foundations &amp; Trends in Information "
            "Retrieval</i>, 2009.",
            "[7] N. Reimers and I. Gurevych, “Sentence-BERT,” EMNLP, 2019; T. Sanh et "
            "al., “DistilBERT,” 2019; Y. Liu et al., “RoBERTa,” 2019.",
            "[8] <i>artnitolog/llm-generated-texts</i> (GPT-4 Turbo/Omni, Claude 3 Opus, "
            "Llama3 70B, Command R+) on Hugging Face.",
        ]),

        h2("Appendices"),
        bullets([
            "Streamlit demo: <code>app/app.py</code> (baseline); MoE CLI: "
            "<code>PYTHONPATH=. python3 moe/predict_moe.py \"...\"</code> (prints "
            "label, P(AI), threshold, expert, words, type, warning).",
            "Metrics: <code>results/baseline_ml_metrics.csv</code>, "
            "<code>moe_expert_metrics.csv</code>, <code>moe_routed_metrics.csv</code>, "
            "<code>moe_baseline_comparison.csv</code>, "
            "<code>moe_calibrated_metrics.csv</code>, <code>moe_summary.json</code>.",
            "Run: <code>moe/download_small_text.py → moe/train_moe.py → "
            "moe/calibrate_thresholds.py → moe/make_moe_appendix.py → "
            "make_research_paper.py</code>.",
            "Figure index (original paper numbering): architecture, class balance, "
            "length, ablations, calibration, precision/recall flip, type gate — all "
            "PNGs in <code>results/figures/</code>; global numbers are preserved in "
            "the List of Figures.",
        ]),
    ]
    return blocks
