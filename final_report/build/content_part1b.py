"""Part I — Complete Project Report, Chapters 7–12.

Short-text failure analysis, MoE architecture, MoE results, application,
limitations, conclusion and the Part I reference list.
"""
from blocks import (h1, h2, h3, p, bullets, numbered, fig, table, code, eq,
                    pagebreak, spacer, refs)


def _pc(x, d=2):
    return f"{x * 100:.{d}f}%"


def _f4(x):
    return f"{x:.4f}"


def _f3(x):
    return f"{x:.3f}"


def part1b(M):
    base = M["base"]        # condition -> row (moe_baseline_comparison.csv)
    routed = M["routed"]    # condition -> row (moe_routed_metrics.csv)
    experts = M["experts"]  # list of rows (moe_expert_metrics.csv)
    calib = M["calib"]      # list of rows (moe_calibrated_metrics.csv)
    summary = M["summary"]  # moe_summary.json
    th = M["thresholds"]    # models/moe/thresholds.json

    def ev(cond, thr):
        """calibrated-metrics row for (condition, threshold)."""
        for r in calib:
            if r["subset"] == "eval" and r["condition"] == cond and r["threshold"] == thr:
                return r
        raise KeyError((cond, thr))

    def hc3cal(thr):
        for r in calib:
            if r["subset"] == "hc3_holdout" and r["condition"] == "full" and r["threshold"] == thr:
                return r
        raise KeyError(thr)

    blocks = [
        # ==================================================================
        pagebreak(),
        h1("Chapter 7 — Short-Text Failure Analysis"),
        p("Chapters 4–6 established that the baseline is an excellent essay classifier. "
          "This chapter attacks that result. The frozen model — the same vectorizers and "
          "classifiers whose artifacts sit in <code>models/</code> — is re-evaluated on "
          "truncated versions of the 5,000 held-out essays, on unseen chat data and on "
          "unseen generators. Nothing is retrained: every number below comes from "
          "<code>results/moe_baseline_comparison.csv</code>, written by the project's own "
          "evaluation scripts."),

        h2("7.1  Accuracy collapses monotonically with length"),
        p("Each held-out essay is cut to its first <i>N</i> words (and, separately, its "
          "first <i>N</i> sentences) and re-classified. {{t:ablation}} gives the full "
          "sweep. Accuracy falls from " + _pc(base["full"]["accuracy"]) + " on full essays "
          "to " + _pc(base["first_100_words"]["accuracy"]) + " at 100 words, "
          + _pc(base["first_50_words"]["accuracy"]) + " at 50 words, "
          + _pc(base["first_25_words"]["accuracy"]) + " at 25 words and "
          + _pc(base["first_1_sent"]["accuracy"]) + " at a single sentence — a "
          f"{(base['full']['accuracy'] - base['first_1_sent']['accuracy']) * 100:.0f} "
          "percentage-point loss from a change that adds no new information, only removes it. "
          "Random-window truncation gives the same curve as prefix truncation "
          "(within 1–2 pp at 50 words: 83.3% prefix vs 81.5% window), so the effect is "
          "not an artefact of cutting at sentence boundaries."),
        table(
            "ablation",
            "Length ablation of the frozen baseline on the 5,000 held-out essays "
            "(AI is the positive class).",
            ["Condition", "Accuracy", "Precision", "Recall", "F1"],
            [
                ["First 1 sentence", _f4(base["first_1_sent"]["accuracy"]),
                 _f4(base["first_1_sent"]["precision"]), _f4(base["first_1_sent"]["recall"]),
                 _f4(base["first_1_sent"]["f1"])],
                ["First 25 words", _f4(base["first_25_words"]["accuracy"]),
                 _f4(base["first_25_words"]["precision"]), _f4(base["first_25_words"]["recall"]),
                 _f4(base["first_25_words"]["f1"])],
                ["First 50 words", _f4(base["first_50_words"]["accuracy"]),
                 _f4(base["first_50_words"]["precision"]), _f4(base["first_50_words"]["recall"]),
                 _f4(base["first_50_words"]["f1"])],
                ["First 100 words", _f4(base["first_100_words"]["accuracy"]),
                 _f4(base["first_100_words"]["precision"]), _f4(base["first_100_words"]["recall"]),
                 _f4(base["first_100_words"]["f1"])],
                ["First 150 words", _f4(base["first_150_words"]["accuracy"]),
                 _f4(base["first_150_words"]["precision"]), _f4(base["first_150_words"]["recall"]),
                 _f4(base["first_150_words"]["f1"])],
                ["First 200 words", _f4(base["first_200_words"]["accuracy"]),
                 _f4(base["first_200_words"]["precision"]), _f4(base["first_200_words"]["recall"]),
                 _f4(base["first_200_words"]["f1"])],
                ["Full text", _f4(base["full"]["accuracy"]), _f4(base["full"]["precision"]),
                 _f4(base["full"]["recall"]), _f4(base["full"]["f1"])],
            ],
            widths=[44, 28, 28, 28, 30],
        ),
        p("Two details in {{t:ablation}} matter more than the accuracy column. First, "
          "<b>recall on AI never drops</b> — it stays between 0.98 and 0.99 at every "
          "length, because the model keeps calling short text AI. Second, "
          "<b>precision collapses</b>: at 25 words it is " + _f4(base["first_25_words"]["precision"]) +
          ", meaning that most of the texts flagged as AI are in fact human. Mean "
          "classifier confidence also erodes (0.93 at 200 words → 0.79 at 25 words), "
          "but not enough to stop the decision rule from firing. "
          "{{f:acc_vs_len}} plots the whole curve next to the Mixture-of-Experts system "
          "built in Chapters 8–9."),
        fig("acc_vs_len", 0.92),

        h2("7.2  The failure is one-sided: short humans become AI"),
        p("Aggregate accuracy hides the structure of the error. Splitting accuracy by "
          "true class ({{t:split}}) shows that at 25 words the model still recognises "
          + _pc(base["first_25_words"]["recall"]) + " of AI texts but only "
          "<b>48.9% of human texts</b> — worse than a coin flip. On the 3,142 human "
          "essays in the test sample that means 1,604 genuine human texts are labelled "
          "AI-generated. At one sentence human accuracy falls the same way while AI "
          "recall holds: the model has learned that <i>short ⇒ AI</i>."),
        table(
            "split",
            "Length-conditional accuracy split by true class (human = true-negative "
            "rate, AI = true-positive rate; 5,000 held-out essays).",
            ["Length", "Base human", "MoE human", "Base AI", "MoE AI",
             "Base acc", "MoE acc"],
            [
                ["25 w", "48.9%", "96.8%", "98.7%", "67.9%", "67.4%", "86.1%"],
                ["50 w", "74.2%", "96.2%", "98.7%", "87.9%", "83.3%", "93.1%"],
                ["75 w", "86.2%", "95.8%", "98.8%", "94.0%", "90.9%", "—"],
                ["100 w", "91.3%", "94.7%", "98.8%", "96.1%", "94.1%", "95.2%"],
                ["150 w", "96.5%", "94.3%", "98.9%", "97.7%", "97.4%", "95.6%"],
                ["200 w", "98.3%", "98.0%", "98.6%", "97.0%", "98.4%", "97.6%"],
                ["300 w", "99.2%", "98.9%", "98.7%", "97.6%", "99.0%", "—"],
                ["Full", "99.6%", "99.6%", "98.7%", "97.9%", "99.3%", "99.0%"],
            ],
            widths=[20, 24, 22, 22, 20, 24, 24],
        ),
        p("The MoE columns of {{t:split}} are the fix developed in Chapters 8–9 and are "
          "shown here for comparison; the baseline columns are the subject of this "
          "chapter. {{f:human_ai_split}} renders the same table as two curves: the "
          "human curve (left) collapses and is rescued, the AI curve (right) shows the "
          "compensating trade-off the MoE makes at very short lengths."),
        fig("human_ai_split", 0.95),
        p("Why does this direction of error matter more than the total? Because the two "
          "errors are not socially equal. Calling a machine text human is a moderation "
          "miss; calling a student's text machine-generated is a false accusation. A "
          "detector whose human-only accuracy is 48.9% cannot be deployed on short "
          "inputs at all, regardless of its overall accuracy — and the overall number "
          "(67.4%) was already misleading, since it is propped up by the AI class."),

        h2("7.3  Cross-domain generalisation"),
        p("Length is not the only axis on which the baseline breaks. Two corpora that "
          "were never used for training or tuning were held out at 15% and evaluated "
          "directly:"),
        bullets([
            "<b>HC3 chat answers</b> (human vs ChatGPT replies to the same Reddit and "
            "OpenQA questions): <b>" + _pc(base["hc3_holdout"]["accuracy"]) +
            " accuracy</b> — worse than chance, with precision "
            + _f4(base["hc3_holdout"]["precision"]) + ". The model labels most chat "
            "answers AI regardless of their true author, exactly the failure mode of "
            "§7.2, now triggered by register rather than length alone.",
            "<b>Modern 2024-LLM essays</b> (GPT-4 Turbo/Omni, Claude 3 Opus, Llama 3 "
            "70B, Command R+): <b>" + _pc(base["modern_holdout"]["accuracy"]) +
            " accuracy</b> — respectable but far below the "
            + _pc(base["full"]["accuracy"]) + " in-domain figure. The baseline learned "
            "the surface quirks of its own corpus (OCR noise, essay prompts of a "
            "particular era), not authorship per se.",
        ]),
        p("Together these results reframe the headline of Chapter 6: 99.3% is the "
          "accuracy of a model on <i>essays like its training data</i>, and on "
          "everything else it ranges from 85% down to chance. {{f:cross_domain}} "
          "compares both systems on all three domains and is discussed fully in "
          "Chapter 9."),

        h2("7.4  Why the model fails"),
        bullets([
            "<b>Evidence sparsity.</b> TF-IDF counts scale with token count; character "
            "n-grams need minimum document frequency to survive <code>min_df</code>; "
            "stylometric statistics (type-token ratio, burstiness, readability) are "
            "unstable on a handful of sentences. At 25 words the 75,000-dimensional "
            "vector is almost all zeros in the discriminative regions.",
            "<b>Domain shift.</b> Short text in deployment is chat, replies and "
            "comments — a register absent from a corpus whose training median is "
            f"{M['ds']['train.csv']['median']:.0f} words (5th percentile "
            f"{M['ds']['train.csv']['p5']:.0f}). What looks like a length problem is "
            "also a topic and style problem.",
            "<b>Learned prior.</b> Because almost every short training example the "
            "model ever saw was AI-generated, the posterior for short inputs is "
            "dominated by the class prior, not the evidence. The 0.5 decision "
            "threshold then converts that prior directly into false accusations.",
            "<b>No calibration on this regime.</b> The threshold was never tuned for "
            "short text; Chapter 8 shows the optimal threshold moves from 0.32 at 25 "
            "words to 0.64 at 150 words.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 8 — Mixture-of-Experts Architecture"),
        p("The failure analysis suggests a design principle: instead of one model "
          "compromising across regimes, give each length regime its own specialist and "
          "route explicitly. This is the mixture-of-experts idea [9] applied to surface "
          "form rather than to neural capacity — the gate is a word count, the experts "
          "are classical TF-IDF classifiers, and the whole system trains in about two "
          "minutes on a laptop CPU."),
        p("{{f:moe_arch}} puts the whole design in one picture: the deterministic "
          "length gate selects one of three TF-IDF + Logistic Regression experts, "
          "a separate stylometric type gate annotates every input with chat / "
          "casual / essay context, and the length-adaptive threshold converts the "
          "routed probability into the final label."),
        fig("moe_arch", 0.9),

        h2("8.1  Motivation and design constraints"),
        bullets([
            "<b>Specialise by an observable variable.</b> Length is free, deterministic "
            "and interpretable — no learned router can collapse or drift.",
            "<b>Fix the data, not just the model.</b> The baseline never saw short "
            "text; every expert must be trained on text that actually lives in its "
            "bucket (Chapter 3, §3.4).",
            "<b>Keep the baseline honest.</b> The full-text regime already works "
            "(99.3%); the MoE must not pay more than a fraction of a point for its "
            "short-text gains.",
            "<b>Report uncertainty.</b> Inputs below 50 words carry little signal for "
            "any model; the interface must say so rather than pretend otherwise.",
        ]),

        h2("8.2  Datasets for the upgrade"),
        p("Three sources feed the experts ({{t:pools}}): the HC3 chat corpus and the "
          "modern-LLM essays described in Chapter 3, plus truncation augmentation of the "
          "original 20,000-essay training subsample. Pools are class-balanced with seed "
          "42; the 15% HC3 and modern holdouts are excluded from every pool by "
          "construction, so the cross-domain numbers of Chapter 9 are honest."),
        table(
            "pools",
            "Mixture-of-Experts training pools (balanced per class, seed 42).",
            ["Pool", "Sources (balanced caps)", "Size"],
            [
                ["SHORT ≤ 150 w",
                 "HC3 ≤150 w (7k/class) + truncated-essay copies 50/100/150 w "
                 "(3k/class)",
                 "20,000 (10k / 10k)"],
                ["LONG 151–400 w",
                 "HC3 long (4k/class) + modern long (2k/class) + base essays "
                 "(3k/class)",
                 "16,864 (7,864 / 9,000)"],
                ["VERYLONG > 400 w",
                 "Base essays >400 w (7k/class) + modern >400 w (2.5k/class)",
                 "11,422 (6,321 / 5,101)"],
            ],
            widths=[34, 84, 46],
        ),
        p("The mild imbalance in the LONG and VERYLONG pools follows the scarce class "
          "in each source (human-heavy verylong, AI-heavy modern essays) rather than a "
          "sampling error; per-expert validation is stratified accordingly."),

        h2("8.3  Length router"),
        p("The gate is a word count with two cut points chosen from the data: 150 words "
          "is where the baseline's accuracy cliff levels off, and 400 words is "
          "approximately the corpus median. Formally, with <i>n</i> = word count of the "
          "input:"),
        eq("e(x) = short if n ≤ 150,  long if 151 ≤ n ≤ 400,  verylong if n &gt; 400;"
           "&nbsp;&nbsp;&nbsp;ŷ = 1[ P<sub>e(x)</sub>(x) ≥ θ(n) ]"),
        p("Routing is applied identically at training and inference time; there are no "
          "learned gate parameters and therefore no routing errors — a text can only be "
          "misclassified by its expert, never by the gate."),

        h2("8.4  The three experts"),
        p("Each expert is the baseline recipe shrunk to its regime: word TF-IDF "
          "(1,2 grams) with 15,000 features + character TF-IDF (char_wb 3–5 grams) with "
          "25,000 features, fused to 40,000 dimensions, classified by Logistic "
          "Regression (<code>C=1.5</code>, <code>lbfgs</code>, <code>max_iter=1000</code>, "
          "L2). Two deliberate differences from the baseline:"),
        bullets([
            "<b>Per-bucket vocabularies.</b> Each expert's TF-IDF is fit on its own "
            "pool only, so long-essay vocabulary cannot crowd out the short-essay and "
            "chat vocabulary that the SHORT expert needs (and vice versa).",
            "<b>Smaller feature budgets.</b> 15k+25k instead of 25k+50k: a short "
            "document cannot support 75,000 informative dimensions, and the smaller "
            "matrix trains in seconds per expert.",
        ]),
        p("All three experts are trained by <code>moe/train_moe.py</code> in "
          "approximately 130 s end-to-end (TF-IDF fit: 6 s short, 15 s long, 25 s "
          "verylong) and saved as <code>models/moe/expert_{short,long,verylong}.pkl</code>."),

        h2("8.5  Type gate: chat, casual or essay"),
        p("Length alone cannot distinguish a 120-word chat reply from a 120-word "
          "truncated essay, and the two look different to a classifier. A second, "
          "<i>learned</i> gate therefore predicts the text type from 16 fast stylometric "
          "features — log word count, sentence count and mean sentence length, type-token "
          "ratio, hapax ratio, average word length, contraction / informal-marker / "
          "first-person / second-person rates, question and exclamation rates, "
          "capitalisation ratio, Flesch Reading Ease and FKGL, and burstiness "
          "(σ/μ of sentence lengths) [6]. Labels come from the data source: HC3 "
          "<code>open_qa</code> → chat, HC3 <code>eli5</code>/<code>finance</code> plus "
          "truncated essays → casual, full and modern essays → essay; 2,500 balanced "
          "examples per type. A Logistic Regression over standardised features reaches "
          f"<b>{summary['type_gate_accuracy'] * 100:.1f}% validation accuracy</b>."),
        fig("type_gate", 0.72),
        p("Per-class behaviour matches linguistic intuition: essay is easiest "
          "(P/R/F1 = 0.90/0.92/0.91), chat next (0.74/0.84/0.79), casual hardest "
          "(0.77/0.66/0.71) because casual and chat genuinely overlap. The gate's job "
          "is routing insight and UI context — it is reported alongside every "
          "prediction, never used as evidence of authorship."),

        h2("8.6  Length-adaptive threshold calibration"),
        p("The default decision threshold θ = 0.50 assumes symmetric costs, but "
          "Chapter 7 showed the errors are not symmetric at any length. On a dedicated "
          "calibration half (2,500 essays, seed 999) the F1-optimal threshold was swept "
          "per truncation length; the optimum drifts right as text grows:"),
        eq(f"θ(n) = clip(0.28 + 0.0024 · n, 0.32, 0.64)&nbsp;&nbsp;for n ≤ 150 "
           f"(short expert);&nbsp;&nbsp;θ = 0.50 for the long and verylong experts"),
        p("with fitted points 25 w → 0.32, 50 w → 0.40, 100 w → 0.56, 150 w → 0.64 "
          "(shipped as <code>models/moe/thresholds.json</code>; long/verylong were "
          "tried at tuned values and reverted to 0.50 because full-text accuracy "
          "fell). {{f:threshold_curve}} shows why a single global threshold cannot "
          "serve: the F1 curves for different lengths cross, and the default 0.50 is "
          "optimal for none of them."),
        fig("threshold_curve", 0.88),
        p("Calibration is deliberately conservative: it is applied only to the short "
          "bucket, where the failure is large and monotonic, and every prediction under "
          "50 words still carries an explicit low-confidence warning in the interface."),

        h2("8.7  Inference path"),
        p("<code>moe/predict_moe.py</code> (and the shared engine it calls) performs, "
          "for any input text: word count → length gate → expert TF-IDF + Logistic "
          "Regression → P(AI) → length-adaptive threshold → label, plus the type-gate "
          "prediction, the expert used, the threshold applied, a confidence value and a "
          "reliability warning for inputs under 25 / 50 words. The same path serves the "
          "CLI, the REST API and the Streamlit interface described in Chapter 10."),

        # ==================================================================
        pagebreak(),
        h1("Chapter 9 — Mixture-of-Experts Results and Comparative Analysis"),
        p("All results in this chapter are produced by <code>moe/train_moe.py</code>, "
          "<code>moe/calibrate_thresholds.py</code> and the evaluation half of the "
          "5,000-essay sample (seed 999 split for calibration, seed 42 elsewhere); the "
          "CSVs under <code>results/moe_*.csv</code> are the source of record."),

        h2("9.1  Per-expert validation"),
        p("Each expert is validated on a held-out 20% of its own pool ({{t:experts}}, "
          "{{f:moe_experts}}). Every expert reaches ≥ 0.97 accuracy and ≥ 0.995 ROC-AUC "
          "within its regime — including the SHORT expert, which is exactly the regime "
          "where the single baseline model scored 48.9% on human text."),
        table(
            "experts",
            "Per-expert validation on the held-out 20% of each pool "
            "(results/moe_expert_metrics.csv).",
            ["Expert", "Train / Val", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[r["expert"].capitalize(),
              f"{int(r['n_train']):,} / {int(r['n_val']):,}",
              _f4(r["accuracy"]), _f4(r["precision"]), _f4(r["recall"]),
              _f4(r["f1"]), _f4(r["roc_auc"])] for r in experts],
            widths=[22, 32, 22, 22, 20, 22, 24],
        ),
        fig("moe_experts", 0.92),

        h2("9.2  Routed system versus baseline"),
        p("Routing the 5,000 held-out essays (and their truncations, and both unseen "
          "corpora) through the MoE gives the headline comparison of {{t:headline}}. "
          "The pattern of Chapter 7 is inverted: every short condition improves "
          "dramatically, both unseen domains improve dramatically, and full-text "
          "accuracy is paid for with "
          + f"{(base['full']['accuracy'] - routed['full']['accuracy']) * 100:.2f} pp "
          "— the price of specialisation, and the only regression in the table."),
        table(
            "headline",
            "Baseline versus routed Mixture-of-Experts on identical inputs "
            "(results/moe_baseline_comparison.csv vs moe_routed_metrics.csv).",
            ["Condition", "Baseline", "MoE routed", "Δ"],
            [
                ["Full text", _pc(base["full"]["accuracy"]), _pc(routed["full"]["accuracy"]),
                 f"{(routed['full']['accuracy'] - base['full']['accuracy']) * 100:+.1f} pp"],
                ["First 25 words", _pc(base["first_25_words"]["accuracy"]),
                 _pc(routed["first_25_words"]["accuracy"]),
                 f"{(routed['first_25_words']['accuracy'] - base['first_25_words']['accuracy']) * 100:+.1f} pp"],
                ["First 50 words", _pc(base["first_50_words"]["accuracy"]),
                 _pc(routed["first_50_words"]["accuracy"]),
                 f"{(routed['first_50_words']['accuracy'] - base['first_50_words']['accuracy']) * 100:+.1f} pp"],
                ["First 100 words", _pc(base["first_100_words"]["accuracy"]),
                 _pc(routed["first_100_words"]["accuracy"]),
                 f"{(routed['first_100_words']['accuracy'] - base['first_100_words']['accuracy']) * 100:+.1f} pp"],
                ["First 150 words", _pc(base["first_150_words"]["accuracy"]),
                 _pc(routed["first_150_words"]["accuracy"]),
                 f"{(routed['first_150_words']['accuracy'] - base['first_150_words']['accuracy']) * 100:+.1f} pp"],
                ["First 1 sentence", _pc(base["first_1_sent"]["accuracy"]),
                 _pc(routed["first_1_sent"]["accuracy"]),
                 f"{(routed['first_1_sent']['accuracy'] - base['first_1_sent']['accuracy']) * 100:+.1f} pp"],
                ["HC3 chat (unseen)", _pc(base["hc3_holdout"]["accuracy"]),
                 _pc(routed["hc3_holdout"]["accuracy"]),
                 f"{(routed['hc3_holdout']['accuracy'] - base['hc3_holdout']['accuracy']) * 100:+.1f} pp"],
                ["Modern LLM essays (unseen)", _pc(base["modern_holdout"]["accuracy"]),
                 _pc(routed["modern_holdout"]["accuracy"]),
                 f"{(routed['modern_holdout']['accuracy'] - base['modern_holdout']['accuracy']) * 100:+.1f} pp"],
            ],
            widths=[50, 30, 32, 30],
        ),
        p("Three readings of {{t:headline}}: (i) the short-text problem is largely "
          "solved — 25-word accuracy rises from "
          + _pc(base["first_25_words"]["accuracy"]) + " to "
          + _pc(routed["first_25_words"]["accuracy"]) + ", one sentence from "
          + _pc(base["first_1_sent"]["accuracy"]) + " to "
          + _pc(routed["first_1_sent"]["accuracy"]) + "; (ii) the cross-domain problem "
          "is solved outright — 48.6% → 98.5% on chat is the difference between a "
          "useless detector and a deployable one; (iii) the cost is bounded and "
          "concentrated where the baseline was already strong. {{f:cross_domain}} "
          "shows the domain view; the length view is {{f:acc_vs_len}}."),
        fig("cross_domain", 0.82),
        p("On full text the buckets behave differently: VERYLONG 99.59% (n = 1,958), "
          "LONG 98.81% (n = 2,945) and SHORT 90.72% (n = 97 essays that are natively "
          "short) — the residual full-text loss comes almost entirely from the short "
          "bucket, where the short expert (trained heavily on chat) is a worse essay "
          "classifier than the baseline."),

        h2("9.3  The error structure flips"),
        p("Accuracy gains usually come with a precision/recall trade, and at 25 words "
          "the trade is visible ({{f:pr_flip}}). The baseline over-calls AI "
          "(precision 0.53, recall 0.99): it is wrong often, and wrong in the "
          "accusatory direction. The MoE at θ = 0.50 under-calls AI (precision 0.93, "
          "recall 0.68): far fewer total mistakes (86.1% vs 67.4% accuracy), but the "
          "residual errors now lean toward letting machine text through rather than "
          "condemning human text. Threshold calibration (§9.4) moves along this "
          "frontier deliberately."),
        fig("pr_flip", 0.72),

        h2("9.4  Threshold calibration on the evaluation half"),
        p("Applying the calibrated thresholds of §8.6 to the untouched evaluation half "
          "gives {{t:calib}}. The intended effects appear: 25-word accuracy "
          f"{_pc(ev('first_25_words', '0.5')['accuracy'])} → "
          f"{_pc(ev('first_25_words', 'calibrated')['accuracy'])} with F1 "
          f"{_f3(ev('first_25_words', '0.5')['f1'])} → "
          f"{_f3(ev('first_25_words', 'calibrated')['f1'])} (AI recall 0.68 → 0.84), "
          "and 150-word accuracy improves by about one point. Two honest caveats "
          "appear in the same table: 50-word accuracy dips "
          f"({ev('first_50_words', '0.5')['accuracy'] * 100:.1f}% → "
          f"{ev('first_50_words', 'calibrated')['accuracy'] * 100:.1f}%) because the "
          "0.40 threshold buys AI recall at the price of human precision on a "
          "human-heavy sample, and full-text accuracy softens slightly "
          f"({ev('full', '0.5')['accuracy'] * 100:.2f}% → "
          f"{ev('full', 'calibrated')['accuracy'] * 100:.2f}%) — which is why the "
          "shipped configuration keeps θ = 0.50 outside the short bucket."),
        table(
            "calib",
            "Calibration on the held-out evaluation half (2,500 essays, seed 999): "
            "default θ = 0.50 versus length-adaptive thresholds.",
            ["Condition", "θ = 0.50 (acc / F1)", "Adaptive θ (acc / F1)", "Δ acc"],
            [
                ["First 25 words",
                 f"{_pc(ev('first_25_words', '0.5')['accuracy'])} / {_f3(ev('first_25_words', '0.5')['f1'])}",
                 f"{_pc(ev('first_25_words', 'calibrated')['accuracy'])} / {_f3(ev('first_25_words', 'calibrated')['f1'])}",
                 f"{(ev('first_25_words', 'calibrated')['accuracy'] - ev('first_25_words', '0.5')['accuracy']) * 100:+.1f} pp"],
                ["First 50 words",
                 f"{_pc(ev('first_50_words', '0.5')['accuracy'])} / {_f3(ev('first_50_words', '0.5')['f1'])}",
                 f"{_pc(ev('first_50_words', 'calibrated')['accuracy'])} / {_f3(ev('first_50_words', 'calibrated')['f1'])}",
                 f"{(ev('first_50_words', 'calibrated')['accuracy'] - ev('first_50_words', '0.5')['accuracy']) * 100:+.1f} pp"],
                ["First 100 words",
                 f"{_pc(ev('first_100_words', '0.5')['accuracy'])} / {_f3(ev('first_100_words', '0.5')['f1'])}",
                 f"{_pc(ev('first_100_words', 'calibrated')['accuracy'])} / {_f3(ev('first_100_words', 'calibrated')['f1'])}",
                 f"{(ev('first_100_words', 'calibrated')['accuracy'] - ev('first_100_words', '0.5')['accuracy']) * 100:+.1f} pp"],
                ["First 150 words",
                 f"{_pc(ev('first_150_words', '0.5')['accuracy'])} / {_f3(ev('first_150_words', '0.5')['f1'])}",
                 f"{_pc(ev('first_150_words', 'calibrated')['accuracy'])} / {_f3(ev('first_150_words', 'calibrated')['f1'])}",
                 f"{(ev('first_150_words', 'calibrated')['accuracy'] - ev('first_150_words', '0.5')['accuracy']) * 100:+.1f} pp"],
                ["Full text",
                 f"{_pc(ev('full', '0.5')['accuracy'])} / {_f3(ev('full', '0.5')['f1'])}",
                 f"{_pc(ev('full', 'calibrated')['accuracy'])} / {_f3(ev('full', 'calibrated')['f1'])}",
                 f"{(ev('full', 'calibrated')['accuracy'] - ev('full', '0.5')['accuracy']) * 100:+.1f} pp"],
            ],
            widths=[40, 40, 44, 32],
        ),
        p("The transfer caveat matters: on the natural (untruncated) HC3 holdout the "
          "same thresholds cost "
          f"{(hc3cal('calibrated')['accuracy'] - hc3cal('0.5')['accuracy']) * 100:+.1f} pp "
          f"({hc3cal('0.5')['accuracy'] * 100:.2f}% → "
          f"{hc3cal('calibrated')['accuracy'] * 100:.2f}%), because that sample is "
          "human-heavy and essay-tuned thresholds over-predict AI. Thresholds are "
          "therefore shipped per length, never globally, and the interface keeps its "
          "sub-50-word warning. A concrete demo effect: a 31-word borderline excerpt "
          "(P(AI) = 0.374, θ = 0.354) flips from Human to AI-Generated — the intended "
          "fix, still low-confidence and flagged as such."),

        h2("9.5  Summary of findings"),
        bullets([
            "<b>Length routing + real small-text data</b> recovers 18.6 pp at 25 "
            "words, 21.8 pp at one sentence and 50.0 pp on unseen chat, at a 0.30 pp "
            "full-text cost.",
            "<b>Per-expert validation ≥ 97%</b> in every regime confirms the failure "
            "was never that the task is hard — it was that one model was asked to "
            "serve three different tasks.",
            "<b>Calibration helps where the failure is systematic</b> (short) and is "
            "neutral-to-harmful elsewhere; it is applied surgically, not globally.",
            "<b>What remains hard:</b> inputs under 25–50 words (86–93% even with the "
            "MoE), boundary texts near 150 words (the baseline still wins 97.4% vs "
            "95.6% pre-calibration), and any generator outside the two audited sets.",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 10 — Application and Implementation"),
        p("The research result is only useful if it can be run. This chapter describes "
          "the shipped system: one inference engine, six front-ends, two cloud "
          "deployments — all consuming the same frozen artifacts from "
          "<code>models/</code> and <code>models/moe/</code>."),

        h2("10.1  Streamlit demonstration interface"),
        p("<code>app/app.py</code> (546 lines) is the conference demo interface. Its "
          "sidebar navigation offers five pages — <i>Inference</i>, <i>Training "
          "Report</i>, <i>Testing Report</i>, <i>Figures</i> and <i>About</i> — plus a "
          "live model-status panel (3/3 experts, type gate, baseline, results). The "
          "inference page offers four modes: <i>MoE Router (recommended)</i>, "
          "<i>Baseline Logistic Regression</i>, <i>Baseline Linear SVM</i> and "
          "<i>Compare: MoE vs Baseline</i>; artifacts are loaded once through "
          "<code>@st.cache_resource</code>. A prediction shows P(AI), the routed "
          "expert, the text type, the threshold in force, a reliability warning for "
          "short inputs and the stylometric evidence cards (word count, TTR, "
          "burstiness, Flesch ease/grade). {{f:screenshot}} shows the interface "
          "performing live MoE inference."),
        fig("screenshot", 0.9),
        p("Run it with <code>streamlit run app/app.py --server.port 8501</code>. The "
          "Training Report and Testing Report pages render the project's own CSVs "
          "(<code>results/baseline_ml_metrics.csv</code>, <code>results/moe_*.csv</code>, "
          "<code>results/moe_summary.json</code>) as tables and plots, so the demo "
          "always displays the same numbers as this report."),

        h2("10.2  Model inference engine"),
        p("<code>test/inference.py</code> (158 lines) is the single place that touches "
          "the baseline artifacts; <code>moe/predict_moe.py</code> adds the routed path. "
          "Every front-end calls these functions, so a change here updates all six "
          "interfaces at once:"),
        bullets([
            "<code>load_artifacts()</code> — joblib-loads the four baseline pickles "
            "(word vectorizer, char vectorizer, LR, calibrated SVM) with a "
            "'train first' hint if they are missing.",
            "<code>vectorize(texts, artifacts)</code> — word + character transform "
            "followed by <code>hstack(...).tocsr()</code>; a single string is "
            "auto-wrapped into a list.",
            "<code>predict_proba(texts, artifacts, model)</code> — returns the P(AI) "
            "array for <code>model='lr'</code> or <code>'svm'</code>.",
            "<code>predict_text(...)</code> / <code>predict_batch(...)</code> — "
            "single-text dict (label, probabilities, confidence, optional 24 "
            "stylometrics) and chunked batch prediction (batch size 1,000 to avoid "
            "out-of-memory on 97k/290k CSVs).",
            "<code>evaluate_csv(...)</code> / <code>predict_folder(...)</code> — "
            "full CSV evaluation with metrics + confusion matrix, and folder-wide "
            "<code>.txt</code>/<code>.md</code> classification.",
        ]),
        fig("flow_infer", 0.86),

        h2("10.3  Command-line interface"),
        p("<code>test/predict.py</code> (140 lines) wraps the engine for terminal use:"),
        code("""
            # single text with stylometric evidence
            venv/bin/python test/predict.py --text "Your essay here..." --model lr --stylometrics

            # one file, a whole folder, or a CSV sample
            venv/bin/python test/predict.py --input test/example_texts/human_essay_school.txt --model svm
            venv/bin/python test/predict.py --folder test/example_texts --model lr
            venv/bin/python test/predict.py --csv data/test.csv --model lr --sample 5000 --out test/results

            # interactive loop / routed MoE CLI
            venv/bin/python test/predict.py --interactive --model lr
            PYTHONPATH=. python3 moe/predict_moe.py "your text"
        """),

        h2("10.4  REST API and LangChain integration"),
        p("<code>test/api.py</code> (95 lines) exposes the engine through FastAPI on "
          "port 8001; <code>test/langchain_tool.py</code> (86 lines) wraps it as a "
          "LangChain <code>BaseTool</code> named <code>ai_human_classifier</code> that "
          "an LLM agent can call. Both are implemented and exercised by "
          "<code>test/terminal_demo.py</code>, which compares direct, LangChain and "
          "API paths in one run. {{t:api}} lists the endpoints."),
        table(
            "api",
            "REST API endpoints implemented in test/api.py (FastAPI, port 8001).",
            ["Method", "Endpoint", "Purpose"],
            [
                ["GET", "<code>/health</code>", "Liveness check (used by run_all.sh)"],
                ["GET", "<code>/model/info</code>", "Loaded model names, feature size, classes"],
                ["POST", "<code>/predict</code>",
                 "Classify one text: <code>{text, model, stylometrics}</code> → "
                 "label, P(AI), confidence, optional stylometrics"],
                ["POST", "<code>/predict/batch</code>",
                 "Classify a list of texts in one request"],
                ["POST", "<code>/langchain/tool</code>",
                 "Invoke the LangChain tool over HTTP"],
            ],
            widths=[22, 46, 96],
        ),
        code("""
            curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \\
              -d '{"text":"In todays rapidly evolving digital landscape...","model":"lr","stylometrics":true}'
        """),

        h2("10.5  System architecture"),
        p("{{f:sys_arch}} puts the whole system in one picture: training produces the "
          "shared artifacts, one inference core consumes them, and six front-ends plus "
          "two deployments sit on top. {{t:frontends}} is the same inventory in "
          "tabular form."),
        fig("sys_arch", 0.95),
        table(
            "frontends",
            "The six inference front-ends and two deployments.",
            ["#", "Interface", "Entry point", "Best for"],
            [
                ["1", "CLI — single/file", "<code>test/predict.py --text / --input</code>",
                 "One-off classification"],
                ["2", "CLI — folder/CSV batch", "<code>--folder / --csv --sample</code>",
                 "Batch evaluation"],
                ["3", "Terminal demo", "<code>test/terminal_demo.py [--loop]</code>",
                 "Trying text live, path comparison"],
                ["4", "REST API", "<code>test/api.py</code> → :8001",
                 "Other programs, curl"],
                ["5", "LangChain tool", "<code>ai_human_classifier</code>",
                 "LLM agents"],
                ["6", "Streamlit UI", "<code>app/app.py</code> → :8501",
                 "Demonstration to humans"],
            ],
            widths=[8, 40, 66, 50],
        ),

        h2("10.6  Deployments"),
        bullets([
            "<b>Vercel (production):</b> <code>https://moe-text-classifier-pravesh12.vercel.app</code> "
            "— a single Python serverless function serves the static project site "
            "(<code>web/index.html</code>, results, figures, reports) and "
            "<code>POST /api/predict</code>; models are bundled in "
            "<code>web/api/models/</code> (~10 MB) with a GitHub fallback "
            "(<code>MODEL_BASE_URL</code>), and <code>src/</code> + <code>moe/</code> "
            "are shipped so the pickles unpickle. scikit-learn is pinned "
            "<code>==1.9.1</code> for pickle compatibility.",
            "<b>GitHub Pages:</b> <code>docs/</code> mirrors the site, figures and "
            "the published report PDF.",
            "<b>Local one-command run:</b> <code>./run_all.sh</code> installs, trains, "
            "evaluates, starts the API (:8001) and Streamlit (:8501) with health "
            "checks (see Part IV).",
        ]),

        # ==================================================================
        pagebreak(),
        h1("Chapter 11 — Limitations and Future Work"),
        h2("11.1  Limitations"),
        bullets([
            "<b>Ultra-short floor.</b> Even the routed system reaches only "
            + _pc(routed["first_25_words"]["accuracy"]) + " at 25 words and "
            + _pc(routed["first_1_sent"]["accuracy"]) + " at one sentence, with wide "
            "uncertainty. The short expert was trained on truncations ≥ 50 words, so "
            "25 words is extrapolation; some of the residual error is irreducible "
            "signal loss.",
            "<b>Boundary behaviour.</b> Around 150 words the baseline still beats the "
            "MoE (97.4% vs 95.6% pre-calibration): the short expert leans toward chat "
            "style. Hard routing cannot blend experts at the boundary.",
            "<b>Threshold transfer.</b> Calibrated thresholds are essay-tuned; they "
            "cost about 0.7 pp on the human-heavy natural HC3 holdout. Domain-sensitive "
            "thresholds would need per-type calibration.",
            "<b>Generator coverage.</b> HC3 is a 2023 ChatGPT snapshot and the modern "
            "audit covers five 2024 models. Paraphrasing, translation and adversarial "
            "editing were not evaluated — a detector that survives truncation may not "
            "survive a paraphraser.",
            "<b>Corpus and scale.</b> The baseline trains on a 20,000-essay subsample "
            "of 290,300 (full training might add 0.1–0.3 pp); the essays are student "
            "writing from one source, so news, research and review text remain "
            "untested in-domain.",
            "<b>Components designed but not trained.</b> Transformer fine-tuning "
            "(DistilBERT/RoBERTa, notebook 05), Sentence-BERT fusion "
            "(<code>src/semantic.py</code>, notebook 04) and the feature-family "
            "ablation (notebook 06) exist as code but produced no reported scores in "
            "this project; they are future work, not results.",
        ]),
        h2("11.2  Future work"),
        numbered([
            "Train 10–35-word examples (and few-shot chat) so the short expert stops "
            "extrapolating — likely a larger gain than any threshold tuning.",
            "Soft routing: weight the short/long experts near the 150-word boundary "
            "instead of hard switching.",
            "Per-type thresholds (essay vs chat) fitted on the type-gate output, with "
            "held-out transfer tests like §9.4.",
            "Phase-2 transformers: DistilBERT/RoBERTa with 512-token chunking [7] as "
            "the capacity ceiling test against the classical pipeline.",
            "Feature-family ablation (word-only vs char-only vs stylometric-only) and "
            "Sentence-BERT + stylometric fusion.",
            "Full 290k training, paraphrase/adversarial robustness evaluation, and "
            "cross-domain calibration on news and review text.",
        ]),
        h2("11.3  Ethics of deployment"),
        p("P(AI) is a screening signal, not proof of authorship. The two failure "
          "directions are not equivalent: false accusations of machine authorship harm "
          "people, while missed machine text harms institutions. Any deployment of "
          "this system should therefore keep the short-input warning, the type report "
          "and the confidence value visible, and should never auto-accuse on the "
          "strength of a single probability — particularly below 50 words, where even "
          "the calibrated system is explicitly uncertain."),

        # ==================================================================
        pagebreak(),
        h1("Chapter 12 — Conclusion"),
        p("This project set out to separate human-written from AI-generated text and "
          "ended by characterising when that separation is possible at all. The "
          "engineering result is a dual TF-IDF pipeline (25k word + 50k character "
          "features, 75,000 dimensions) with three linear classifiers that reaches "
          + _pc(M["test"]["models"]["svm"]["accuracy"]) + " accuracy and ROC-AUC "
          + f"{M['test']['models']['svm']['roc_auc']:.4f}" + " on a 5,000-essay "
          "held-out test, trained in 33 seconds and reproducible from released "
          "artifacts. The scientific result is that this number is conditional: "
          "truncation to 25 words drops it to "
          + _pc(base["first_25_words"]["accuracy"]) + ", one sentence to "
          + _pc(base["first_1_sent"]["accuracy"]) + ", unseen chat to "
          + _pc(base["hc3_holdout"]["accuracy"]) + " — and that the collapse is "
          "one-sided, penalising human authors rather than machine text."),
        p("The fix required data, architecture and calibration together: 80,811 genuine "
          "human/ChatGPT chat answers, 17,997 modern-LLM essays and truncation "
          "augmentation feed three length-specialised experts behind a deterministic "
          "length gate; a learned stylometric type gate reports chat/casual/essay "
          "context; and thresholds adapt linearly with length (0.32 → 0.64 over "
          "25–150 words). The routed system reaches "
          + _pc(routed["first_25_words"]["accuracy"]) + " at 25 words, "
          + _pc(routed["first_1_sent"]["accuracy"]) + " at one sentence, "
          + _pc(routed["hc3_holdout"]["accuracy"]) + " on unseen chat and "
          + _pc(routed["modern_holdout"]["accuracy"]) + " on unseen 2024-LLM essays, "
          "while holding " + _pc(routed["full"]["accuracy"]) + " on full essays — a "
          "0.30 pp full-text cost for a system that is deployable where the baseline "
          "was not."),
        p("Everything reported here is checkable: the models, the data pipelines, the "
          "six inference front-ends, the metrics CSVs and the figures are released in "
          "the repository, and Part IV gives the exact commands to regenerate every "
          "table and figure in this report from those artifacts. The two research "
          "papers in Parts II and III restate the same evidence in publication form — "
          "the first as a compact conference paper, the second with the full formal "
          "treatment, ablations and appendices."),
        p("<b>Final statement.</b> Length is not a detail of authorship detection — "
          "for short text it is the task. Treating length as a routing variable, "
          "grounding each expert in data from its own regime, and calibrating per "
          "length converts a detector that confidently mis-accuses short human writing "
          "into one that is honest about what it can and cannot know."),

        h2("References (Part I)"),
        refs([
            "[1] B. Guo et al., “How Close is ChatGPT to Human Experts? Question-Answer "
            "Benchmark HC3,” arXiv:2301.07597, 2023; <i>Hello-SimpleAI/HC3</i> "
            "[Hugging Face].",
            "[2] R. Roy et al., “A Comprehensive Dataset for Human vs AI Generated Text "
            "Detection” (Defactify), arXiv:2510.22874, 2025.",
            "[3] B. Wang et al., “M4: Multi-Generator, Multi-Domain, Multi-Task, "
            "Multi-Label Benchmark for AI-Generated Text Detection,” "
            "arXiv:2310.13607, 2023.",
            "[4] T. Hastie, R. Tibshirani and J. Friedman, <i>The Elements of "
            "Statistical Learning</i>, 2nd ed. Springer, 2009.",
            "[5] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: "
            "BM25 and TF-IDF,” <i>Foundations and Trends in Information Retrieval</i>, "
            "vol. 3, no. 4, 2009.",
            "[6] E. Mitchell et al., “The State and Fate of Linguistic Uncertainty,” "
            "2020 (stylometric feature practice).",
            "[7] N. Reimers and I. Gurevych, “Sentence-BERT,” in <i>Proc. EMNLP</i>, "
            "2019; T. Sanh et al., “DistilBERT,” 2019; Y. Liu et al., “RoBERTa,” 2019.",
            "[8] <i>artnitolog/llm-generated-texts</i> (GPT-4 Turbo/Omni, Claude 3 Opus, "
            "Llama 3 70B, Command R+), Hugging Face.",
            "[9] N. Shazeer et al., “Outrageously Large Neural Networks: The "
            "Sparsely-Gated Mixture-of-Experts Layer,” arXiv:1701.06538, 2017.",
            "[10] Streamlit, FastAPI, scikit-learn and Hugging Face Transformers "
            "documentation, 2024–2026.",
        ]),
    ]
    return blocks
