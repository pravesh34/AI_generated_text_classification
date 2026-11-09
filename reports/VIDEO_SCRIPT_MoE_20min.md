# 20-Minute Video Script — "Why 99% AI Detectors Fail on Short Text"

**Target:** YouTube / class seminar / conference talk
**Duration:** ~20 min spoken (130–150 wpm ≈ 2,800 words)
**Materials:** `results/figures/*.png`, `moe/predict_moe.py` live demo, slides/screenshare, whiteboard for equations
**One-line pitch:** Your detector is 99% — until you paste 25 words, then it's worse than a coin flip on humans. Here's why, and how a Mixture-of-Experts fixes it.

---

## [0:00–1:00] HOOK — The 25-Word Test

**[VISUAL: Live terminal, paste 20-word human sentence, classifier screams "AI 99%"]**

**SAY:**
Welcome. Quick test. [types short human sentence from the paper demo] This is human-written — my last YouTube comment. My detector just told me it's AI with 99% confidence. Wrong. That is not a bug in one model. That is the story of this entire paper.

We built a detector that scores 99.5% on essays. Ninety-nine point five. And it fails like this on short text — not a little, catastrophically. In the next twenty minutes I'll show you the exact numbers, the one-sided error that causes them, the new data we downloaded to fix it, a Mixture-of-Experts router that lifts 25-word accuracy from 67% to 86% and unseen chat from coin-flip to 98.5% — and the one trick with thresholds you must know if you deploy anything like this.

Hold that failure in mind. Everything is coming back to it.

**[CUT: title card — "Length-Expertized Authorship Detection"]**

---

## [1:00–3:00] THE PROBLEM — Why Authorship Matters

**[VISUAL: Title card fades, slide: academic integrity / journalism / moderation logos]**

**SAY:**
Why care? Large language models made fluent essays free. Professors, editors, moderators need to know: human or machine? The literature says: train a classifier, get 99%. HC3, Defactify, M4 — all long-form benchmarks.

But open a group chat, a Reddit thread, a product review. Those are ten, twenty, fifty words. Nobody has a thousand-word essay there. So the question this paper asks is brutally simple:

*Does a strong long-text detector survive being cut down to short text?*

Spoiler: no. And the way it fails is not random. It's biased.

**[VISUAL: Figure — length_hist.png]**

Our training essays: median 363 words, 5th percentile 180 words. In other words, the model has never seen a short input in its life. And the smallest human essay in test is 48 words. So when you show it 20 words — it's out of distribution. Taking the punchline forward: we will truncate the same held-out essays to 25, 50, 100, 150 words, 1–3 sentences, and watch accuracy fall off a cliff. That's Section 7.1 in the paper, Figure 6.

---

## [3:00–5:00] BASELINE — The 99% Machine

**[VISUAL: Diagram — Text → Word TF-IDF + Char TF-IDF → LR / SVM]**

**SAY:**
First, the baseline, so you see what we started from. It's Phase 1 of the project, fully reproducible in `train_baseline.py`.

Preprocessing is boring: Unicode-normalize, collapse whitespace, lowercase for TF-IDF. Then dual TF-IDF fusion: word n-grams 1–2, 25,000 features — captures word choice, phrases. Character n-grams, char_wb 3–5, 50,000 features — captures typos, punctuation, sub-word morphology. Concatenated, that's 75,000 sparse features per essay.

Three classifiers: Multinomial Naive Bayes alpha 0.1, Logistic Regression C=1.5, and Linear SVM with 3-fold calibration — so you get P(AI) out. Train on 20,000 balanced essays (10k per class), validate on 4,000 stratified.

Numbers — Table 2 in the paper: LR validation 99.25% accuracy, F1 0.992, AUC 0.9987. SVM 99.53%, F1 0.995. On 5,000 completely held-out essays from test.csv: LR 99.26%, SVM 99.58%, ROC-AUC 0.9995. Confusions: SVM 8 false positives, 11 false negatives in 5,000.

Why so high? Hundreds of weak cues summed. A human essay leaves 400 fingerprints — diction, errors like TD for "to", bursty sentences. Even a linear model adds them up. Char n-grams help enormously because humans leave OCR noise; AI is orthographically clean. That's Figure 3, model comparison — all three models above 95% AUC.

So: excellent machine. On essays.

**[TRANSITION: overlay graph dropping from 99% to 60%]**

---

## [5:00–7:30] THE ABLATION — Watching 99% Become 67%

**[VISUAL: Figure — moe_accuracy_vs_length.png, annotate 25w 67.4%, 1-sent 59.8%]**

**SAY:**
Now the ablation. Same frozen model, same 5,000 held-out essays, just truncated to first-N words. Random windows give the same answer — it's not intro paragraphs, it's length.

First 25 words: 67.4% accuracy. That's the headline. First 50: 83.3%. First 100: 94.1%. First 150: 97.4%. First 200: 98.4%. Full text, 99.3%.

Sentences are even harsher: first 1 sentence, 59.8%. Two sentences, 74.5%. Three, 83.4%. One hundred characters, 60.3%.

LR confidence collapses with length: mean confidence 0.93 at 200 words, 0.79 at 25 words. Only 36% of 25-word predictions are above 0.9 confident, vs 79% at 200 words. The model knows it's guessing and still guesses.

Baseline error shape at 25 words: precision 0.53, recall 0.99. Read that: when it says AI, it's wrong half the time; when it's actually AI, it almost always says AI. Translation: it calls <i>everything</i> short AI. There is a length shortcut.

**[VISUAL: Figure — deep_human_ai_split.png, point at human line plunging]**

Split by true class — this is Table 4 in the deep paper, and it should shock you. Human accuracy at 25 words: 48.95%. Less than a coin flip. The model is <i>worse than chance</i> on short humans. AI accuracy at 25 words: 98.65%. So the overall 67% is hiding a disaster: half the humans flipped to AI.

At 50 words humans recover to 74.2%. At 100 words, 91.3%. At 150, 96.5%. You need about 150 words before the baseline stops being biased.

Why? Think evidence. Word bigrams need length to be seen. Char n-grams need support. Stylometrics — TTR, burstiness, Flesch ease — need 5+ sentences to be stable. With 25 words you have maybe 1.5 sentences. Burstiness σ/μ is noise with n=2. So the TF-IDF vector is sparse, the stylometric cards in the Streamlit app wobble, and the linear boundary defaults to the AI prior it learned from clean long essays. Short human informality looks "too easy, too clean" next to a formal AI essay? No — the geometry just collapses.

That's the failure. Diagnosis, not garnish: a length domain shift plus one-sided bias.

---

## [7:30–10:00] WHY IT'S HARD — Cross-Domain Collapse

**[VISUAL: Figure — moe_cross_domain.png]**

**SAY:**
It gets worse when we leave essays. We held out two completely unseen domains.

HC3 chat Q&A — human vs ChatGPT answers to the same question, Reddit ELI5, open QA, finance. Baseline: 48.56%. Worse than chance. It never learned chat style, so it fails at the task it will actually meet.

Modern 2024 LLMs — GPT-4 Turbo, GPT-4 Omni, Claude 3 Opus, Llama 3, Command R+ essays the original data never saw. Baseline: 85.5%. A 14-point drop. New generators, new phrasing, old detector.

These two numbers tell the fix: we don't need a bigger model. We need <i>short data</i>, <i>new-generator data</i>, and <i>routing</i>. One model cannot average over three regimes that hate each other.

---

## [10:00–12:30] NEW DATA — Where Short Text Comes From

**[VISUAL: Table 1 — datasets, highlight HC3 median 117w]**

**SAY:**
So we downloaded. Two new sources, one script: `moe/download_small_text.py`.

One: HC3, Hello-SimpleAI, Guo et al. Reddit ELI5 alone is 17,000 questions, 67,000 answers. Plus open_qa and finance. Flattened: 80,811 answers — 56,000 human, 24,700 AI. Median 117 words. 61% are ≤150 words. This is genuine small text: humans explaining salt on roads, advising on co-signing loans, answering open questions; ChatGPT doing the same. Perfect for the short expert. Fifteen percent held out, never trained.

Two: modern essays, artnitolog/llm-generated-texts. 3,000 prompts, each with a human essay plus GPT-4, Claude, Llama 3, Command R+. Keep English only: 17,997 texts, median 510 words, 68% over 400 words. That's your verylong modern audit.

Plus augmentation from the original 20k essays: every essay contributes a 50/100/150-word copy for short — prefix or random window — and 200/300-word copies for long. Different label, same author's style. This teaches short-<i>essay</i> style; HC3 teaches chat style. You need both.

Final pools, balanced: short 20,000 (14k HC3 + 6k truncated), long 16,864, verylong 11,422. Code: `moe/type_features.py` defines buckets.

---

## [12:30–15:30] THE MIXTURE OF EXPERTS — Walk the Diagram

**[VISUAL: Figure — moe_architecture.png, trace with cursor]**

**SAY:**
Architecture — Figure 1 in the paper, follow the arrows.

Input text, any length. First branch: <b>length gate</b>, deterministic. If word count ≤150, short. 151 to 400, long. Over 400, verylong. Why 150 and 400? 150 is the cliff we measured — below it humans flip. 400 is roughly the overall median. Zero learned parameters, fully auditable. You always know why it routed.

Second branch: <b>type gate</b>, learned. Sixteen stylometric features — log words, sentence stats, TTR, hapax, contraction rate, first/second-person rate, question rate, Flesch ease, burstiness — through StandardScaler + LogisticRegression to three labels: chat, casual, essay. Labels come from source: HC3 open_qa is chat, ELI5/finance plus truncated essays is casual, full/modern essays is essay. Balanced 2,500 each. Validation 80.5%: essay F1 0.91, chat 0.79, casual 0.71. Casual and chat confuse — expected — but it's good enough for routing insight and the warning system.

Now three expert boxes. Each is independent: word TF-IDF 15k + char TF-IDF 25k = 40k features (smaller than baseline 75k, so it focuses), plus Logistic Regression C=1.5, same as baseline. Short expert trains only on short pool; long only on long; verylong only on verylong. This is classic MoE — specialized vocabularies, not one diluted giant.

At inference, `moe/predict_moe.py` routes, returns label, P(AI), confidence, threshold used, expert, word count, type, and a reliability warning if under 50 words. Under 25 words: big red warning. That warning is a feature, not an apology.

Validation per expert: short 97.0%, long 98.2%, verylong 98.4%. All above 97% in-domain. Good. But in-domain is easy. Held-out is the proof.

---

## [15:30–17:30] RESULTS — Does It Fix Short Text?

**[VISUAL: Figure — moe_accuracy_vs_length.png again, now MoE line above baseline]**

**SAY:**
Same 5k held-out, same truncations, routed MoE with threshold 0.50.

Full text: 98.96% vs baseline 99.26%. We pay 0.3 points for specialization. Verylong bucket alone is 99.59% — better than baseline. Long 98.81%. Short natively-short essays only 97 samples, 90.7% — small n, noisy.

Short truncations: 25 words, 86.06% vs 67.42% — up 18.6 points. 50 words, 93.12% vs 83.28 — up 9.8. 100 words, 95.2% vs 94.1 — up 1.1. 150 words, 95.6% vs 97.4% — down 1.8. 200 words, 97.6% vs 98.4 — down 0.7. One sentence: 81.64% vs 59.82% — up 21.8 points. Two sentences 90.6, three 93.4.

Read the error flip, Figure 8: at 25 words baseline precision 0.53 recall 0.99; MoE precision 0.93 recall 0.68. We traded over-calling AI for under-calling AI. Net accuracy wins by a mile, but we still miss one in three short AIs. That's the frontier calibration will push.

Cross-domain is where MoE earns its keep: HC3 holdout 98.52% vs 48.56% baseline — a fifty-point jump. Modern holdout 95.70% vs 85.52% — ten points. That is not tuning. That is learning the domain for the first time.

---

## [17:30–19:00] CALIBRATION — One Threshold to Rule Them All? No.

**[VISUAL: Figure — deep_threshold_curve.png, F1 curves crossing]**

**SAY:**
Thresholds. Default 0.50 assumes symmetric cost. Sweep F1 vs threshold per length, calib half 2,500.

25 words wants 0.32 — lower it, catch more AI. 50 wants 0.40. 100 wants 0.56. 150 wants 0.64. The optimum <i>moves right with length</i>. At 25 words false negatives dominate, so drop the bar. At 150 words false positives dominate — short humans getting flagged — so raise it.

We shipped `th = 0.28 + 0.0024 times n`, clamped 0.32 to 0.64 for short. Long and verylong stayed 0.50 — tuning them hurt full-text, small calib slices overfit.

Eval-half results: 25 words 86.2% → 88.1%, F1 0.785 → 0.837, recall 0.68 → 0.82 — the flip fixed. 150 words 95.0% → 96.2%. 50 and 100 flat (threshold ≈0.45 and 0.52). Honest caveat: HC3 natural short drops 0.6 points — essay-tuned thresholds add false positives on human-heavy chat. So we keep the warning. Calibration is domain-sensitive; treat 0.32 as essay-calibrated, not universal.

**[LIVE DEMO: predict_moe.py on 31-word borderline, P=0.374, th=0.354 → AI-Generated; 14-word casual human P=0.035 → Human]**

That 31-word case flips from Human to AI under adaptive threshold. Low confidence, flagged. That's calibration in action.

---

## [19:00–20:00] LIMITS, FUTURE WORK, CLOSE

**[VISUAL: Recap table — 67%→86%, 48%→98%, +10pp modern]**

**SAY:**
Limits, quickly. Under 25 words is irreducible — signal is gone; MoE is 86%, not 99%, and wants training data at 10–35 words, not just thresholds. At 150-word essays baseline still wins pre-calibration; soft blending of short/long experts at the boundary is future work. Type gate at 80.5% is insight, not enforcement. And P(AI) is screening, never proof — ship the warning with every deployment.

Future: DistilBERT/RoBERTa with chunking for verylong, per-type thresholds, adversarial paraphrase eval.

Reproduce everything: `moe/download_small_text.py` → `moe/train_moe.py` → `moe/calibrate_thresholds.py` → `moe/predict_moe.py`. Metrics in `results/moe_*.csv`, figures in `results/figures/`, deep paper in `DeepSeek_Style_Research_Paper_MoE.pdf`.

Takeaway: if your detector assumes 300 words, it will lie to you on 25. Length is the task. Route it, train for it, calibrate it — and you keep the 99% without the 67% cliff. Thanks for watching.

**[END CARD: code links + "Try it: PYTHONPATH=. python3 moe/predict_moe.py 'your text'"]**

---

### Teleprompter Notes
- Pause after the hook failure (2 sec) — let the audience feel it.
- Numbers to memorize: 67% @25w, 48.9% human @25w, 86% fixed, 48.6%→98.5% HC3, thresholds 0.32→0.64.
- If live demo fails, pre-record the three `predict_moe.py` outputs (short AI, casual human, 31w flip).

### Runtime Check
~2,800 words at 140 wpm = 20 minutes with pauses/demos. Cut [2.3] and [7.2] to hit 15 minutes if needed.
