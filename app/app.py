"""
Conference Demo Interface: Human vs. AI Text Classification (Baseline + MoE)
Department of Computer Science & Engineering, NIT Sikkim
Authors: Pravesh Gedon Sharma (B240019) & Ankit Rawat (B320032)

Run from project root:
  streamlit run app/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stylometrics import extract_stylometric_features  # noqa: E402
from moe.type_features import (  # noqa: E402
    TYPE_FEATURES,
    extract_type_features,
    length_bucket,
    word_count,
)

MODELS_DIR = ROOT / "models"
MOE_DIR = MODELS_DIR / "moe"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

st.set_page_config(
    page_title="Human vs AI Text — MoE Classifier | NIT Sikkim",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1280px; }
.hero { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0369a1 100%); border-radius: 1.25rem; padding: 1.6rem 1.8rem 1.4rem; color: #f8fafc; margin-bottom: 1.2rem; box-shadow: 0 8px 32px rgba(15,23,42,.22); }
.hero h1 { font-size: 1.85rem; line-height: 1.15; margin: 0 0 .35rem; color: #fff; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 0; color: #cbd5e1; font-size: .98rem; line-height: 1.55; }
.hero .badge { display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #e0f2fe; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; padding: .25rem .65rem; border-radius: 999px; margin-bottom: .7rem; }
.stat-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1rem; }
.stat { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18); border-radius: .8rem; padding: .55rem .85rem; color: #e2e8f0; font-size: .82rem; min-width: 0; }
.stat b { color: #fff; font-size: .95rem; display: block; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 1.1rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15,23,42,.05); }
.result-label { font-weight: 800; font-size: 1.25rem; margin: 0 0 .25rem; }
.result-sub { color: #64748b; font-size: .88rem; margin: 0; }
.pill { display: inline-block; font-size: .75rem; font-weight: 700; padding: .2rem .6rem; border-radius: 999px; margin-right: .3rem; }
.pill-human { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.pill-ai { background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }
.pill-neutral { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.small-note { color: #64748b; font-size: .88rem; }
.fig-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 1rem 1.1rem 1.1rem; margin-bottom: 1rem; }
.fig-card img { border-radius: .75rem; border: 1px solid #e2e8f0; }
.detail-label { color: #0f172a; font-weight: 700; font-size: .85rem; letter-spacing: .04em; text-transform: uppercase; margin: .9rem 0 .25rem; }
.detail-text { color: #334155; font-size: .93rem; line-height: 1.65; }
div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5e7eb; padding: .6rem .8rem; border-radius: .8rem; }
div[data-testid="stMetric"] label[data-testid="stMetricLabel"] { color: #64748b; font-weight: 600; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #e2e8f0 !important; }
.stButton button { border-radius: .8rem; font-weight: 700; }
.stTextArea textarea { border-radius: .8rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="hero">
<div class="badge">NIT Sikkim &middot; CSE &middot; Conference Demo</div>
<h1>Human-Written vs AI-Generated Text</h1>
<p>Mixture-of-Experts classifier that fixes short-text collapse — 86.1% @ 25 words (baseline 67.4%), 98.5% on unseen chat (baseline 48.6%).<br>
<b>Pravesh Gedon Sharma (B240019) &amp; Ankit Rawat (B320032)</b> &nbsp;·&nbsp; Department of Computer Science &amp; Engineering</p>
<div class="stat-row">
<div class="stat"><b>86.1%</b>@ 25 words</div>
<div class="stat"><b>98.96%</b>full text</div>
<div class="stat"><b>98.5%</b>HC3 holdout</div>
<div class="stat"><b>3 experts</b>short / long / verylong</div>
</div>
</div>""",
    unsafe_allow_html=True,
)


# --- Loaders ---
@st.cache_resource
def load_baseline():
    artifacts = {}
    for key, name in [
        ("word_vec", "word_tfidf_vectorizer.pkl"),
        ("char_vec", "char_tfidf_vectorizer.pkl"),
        ("lr", "logistic_regression_baseline.pkl"),
        ("svm", "linear_svm_baseline.pkl"),
    ]:
        p = MODELS_DIR / name
        if p.exists():
            artifacts[key] = joblib.load(p)
    return artifacts


@st.cache_resource
def load_moe():
    experts = {}
    for bucket in ["short", "long", "verylong"]:
        p = MOE_DIR / f"expert_{bucket}.pkl"
        if p.exists():
            experts[bucket] = joblib.load(p)
    gate = None
    gate_p = MOE_DIR / "type_gate.pkl"
    if gate_p.exists():
        gate = joblib.load(gate_p)
    thresholds = {}
    th_p = MOE_DIR / "thresholds.json"
    if th_p.exists():
        with open(th_p) as f:
            thresholds = json.load(f)
    return {"experts": experts, "gate": gate, "thresholds": thresholds}


@st.cache_data
def load_json(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_csv(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def fmt_df(df: pd.DataFrame | None, pct_cols: list[str] | None = None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in pct_cols or ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else x)
    return out


def moe_threshold(bucket: str, n: int, thresholds: dict) -> float:
    if bucket == "short" and "short_formula" in thresholds:
        fm = thresholds["short_formula"]
        return float(min(fm["high"], max(fm["low"], fm["intercept"] + fm["slope"] * n)))
    if bucket in thresholds:
        return float(thresholds[bucket])
    return 0.5


def predict_moe_cached(text: str, moe: dict) -> dict:
    experts = moe["experts"]
    n = word_count(text)
    bucket = length_bucket(n)
    if bucket not in experts:
        return {"error": f"Missing MoE expert: {bucket}. Train models first (moe/train_moe.py).", "words": n, "expert": bucket}
    ex = experts[bucket]
    X = hstack([ex["word_vec"].transform([text]), ex["char_vec"].transform([text])]).tocsr()
    prob_ai = float(ex["clf"].predict_proba(X)[0, 1])
    th = moe_threshold(bucket, n, moe["thresholds"])
    label = "AI-Generated" if prob_ai >= th else "Human-Written"

    type_label, type_conf = "unknown", 0.0
    gate = moe.get("gate")
    if gate is not None:
        feats = extract_type_features(text)
        Xg = np.array([[feats[k] for k in gate["features"]]])
        type_label = str(gate["clf"].predict(Xg)[0])
        type_conf = float(gate["clf"].predict_proba(Xg).max())

    warn = ""
    if n < 25:
        warn = "WARNING: <25 words — very low signal, treat as uncertain."
    elif n <= 50:
        warn = "CAUTION: <=50 words — short-text range, check confidence."

    return {
        "label": label,
        "prob_ai": prob_ai,
        "prob_human": 1 - prob_ai,
        "confidence": max(prob_ai, 1 - prob_ai),
        "threshold_used": th,
        "expert": bucket,
        "words": n,
        "text_type": type_label,
        "type_confidence": type_conf,
        "warning": warn,
    }


def predict_baseline(text: str, baseline: dict, which: str) -> dict | None:
    if "word_vec" not in baseline or "char_vec" not in baseline:
        return None
    model = baseline.get("lr") if which == "lr" else baseline.get("svm", baseline.get("lr"))
    if model is None:
        return None
    X = hstack([baseline["word_vec"].transform([text]), baseline["char_vec"].transform([text])]).tocsr()
    prob_ai = float(model.predict_proba(X)[0, 1])
    return {"label": "AI-Generated" if prob_ai >= 0.5 else "Human-Written", "prob_ai": prob_ai, "prob_human": 1 - prob_ai}


baseline = load_baseline()
moe = load_moe()
summary = load_json("moe_summary.json")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Inference", "Training Report", "Testing Report", "Figures", "About"],
        label_visibility="collapsed",
        horizontal=False,
    )
    st.divider()
    st.caption("MODEL STATUS")
    m1, m2 = st.columns(2)
    m1.metric("MoE experts", f"{len(moe['experts'])}/3")
    m2.metric("Type gate", "yes" if moe.get("gate") else "no")
    m1.metric("Baseline", "yes" if "word_vec" in baseline else "no")
    m2.metric("Results", "yes" if summary else "partial")
    st.divider()
    st.caption("QUICK RUN")
    st.code('PYTHONPATH=. python3 moe/predict_moe.py "your text"', language="bash")
    st.caption("Streamlit: `streamlit run app/app.py`")


# --- Inference ---
if page == "Inference":
    st.subheader("Live Inference")
    st.caption("Paste any text — chat message, short answer, or full essay. MoE routes by length, labels text type, and warns when signal is too short.")
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        mode = st.segmented_control(
            "Model",
            ["MoE Router (recommended)", "Baseline Logistic Regression", "Baseline Linear SVM", "Compare: MoE vs Baseline"],
            default="MoE Router (recommended)",
            label_visibility="collapsed",
        )
        if mode is None:
            mode = "MoE Router (recommended)"
        user_text = st.text_area("Input text", height=300, placeholder="Paste a passage, short answer, or chat message...", label_visibility="collapsed")
        if user_text.strip():
            st.caption(f"{len(user_text.split())} words · {len(user_text)} chars")
        run = st.button("Classify  →", type="primary", use_container_width=True)

    with c2:
        st.markdown("#### Result")
        if run:
            if not user_text.strip():
                st.warning("Please enter text.")
            else:
                sty = extract_stylometric_features(user_text)

                def render_moe(res: dict):
                    if "error" in res:
                        st.error(res["error"])
                        return
                    pill = "pill-ai" if res["label"] == "AI-Generated" else "pill-human"
                    st.markdown(f'<div style="border:1px solid #e2e8f0;border-radius:1rem;padding:1rem;background:#fff">'
                                f'<span class="pill {pill}">{res["label"]}</span>'
                                f'<p class="result-sub">confidence {res["confidence"]*100:.2f}% &middot; P(AI)={res["prob_ai"]:.4f} &middot; P(Human)={res["prob_human"]:.4f}</p></div>',
                                unsafe_allow_html=True)
                    st.progress(min(max(res["prob_ai"], 0.0), 1.0), text=f"P(AI) = {res['prob_ai']:.2%}")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Expert", res["expert"])
                    d2.metric("Words", f"{res['words']}")
                    d3.metric("Threshold", f"{res['threshold_used']:.3f}")
                    st.caption(f"text type = {res['text_type']} ({res['type_confidence']*100:.1f}% confidence)")
                    if res["warning"]:
                        st.warning(res["warning"])

                if mode == "MoE Router (recommended)":
                    render_moe(predict_moe_cached(user_text, moe))
                elif mode == "Baseline Logistic Regression":
                    r = predict_baseline(user_text, baseline, "lr")
                    if not r:
                        st.info("Baseline models not found in `models/`. Run `train_baseline.py` or notebook 03.")
                    else:
                        (st.error if r["label"] == "AI-Generated" else st.success)(f"**{r['label']}** — P(AI)={r['prob_ai']*100:.2f}%")
                elif mode == "Baseline Linear SVM":
                    r = predict_baseline(user_text, baseline, "svm")
                    if not r:
                        st.info("Baseline SVM not found in `models/`.")
                    else:
                        (st.error if r["label"] == "AI-Generated" else st.success)(f"**{r['label']}** — P(AI)={r['prob_ai']*100:.2f}%")
                else:
                    a, b = st.columns(2)
                    with a:
                        st.markdown("**MoE**")
                        render_moe(predict_moe_cached(user_text, moe))
                    with b:
                        st.markdown("**Baseline (LR)**")
                        r = predict_baseline(user_text, baseline, "lr")
                        if not r:
                            st.info("Baseline not available.")
                        else:
                            (st.error if r["label"] == "AI-Generated" else st.success)(f"**{r['label']}** — P(AI)={r['prob_ai']*100:.2f}%")

                st.divider()
                st.markdown("#### Stylometric summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Words", f"{int(sty['num_words'])}")
                m1.metric("TTR", f"{sty['ttr']:.3f}")
                m2.metric("Avg sentence length", f"{sty['mean_sentence_length']:.2f}")
                m2.metric("Burstiness", f"{sty['burstiness']:.3f}")
                m3.metric("Flesch Reading Ease", f"{sty['flesch_reading_ease']:.2f}")
                m3.metric("FK Grade", f"{sty['flesch_kincaid_grade']:.2f}")
        else:
            st.info("Enter text and click **Classify**. MoE routes by length (short \u2192 long \u2192 verylong) and reports expert, text type, and reliability warnings.")


# --- Training Report ---
elif page == "Training Report":
    st.subheader("Training Report")
    st.caption("How the models were trained — experts, validation, and reproduction commands.")
    st.caption("MoE expert training/validation metrics and type gate (from `results/moe_*`).")

    exp = load_csv("moe_expert_metrics.csv")
    if exp is not None:
        st.markdown("#### MoE experts (validation)")
        show = fmt_df(exp, ["accuracy", "precision", "recall", "f1", "roc_auc"])
        st.dataframe(show, use_container_width=True, hide_index=True)
        if summary and "experts" in summary:
            st.json({"gate": summary.get("gate"), "type_gate_accuracy": summary.get("type_gate_accuracy")})
    else:
        st.info("`results/moe_expert_metrics.csv` not found.")

    st.divider()
    st.markdown("#### Baseline training/validation reference")
    base = load_csv("baseline_ml_metrics.csv")
    if base is not None:
        st.dataframe(fmt_df(base), use_container_width=True, hide_index=True)
    else:
        st.info("`results/baseline_ml_metrics.csv` not found.")

    st.divider()
    st.markdown("#### Reproduce training")
    st.code(
        "PYTHONPATH=. python3 moe/download_small_text.py\n"
        "PYTHONPATH=. python3 moe/train_moe.py\n"
        "PYTHONPATH=. python3 moe/calibrate_thresholds.py",
        language="bash",
    )


# --- Testing Report ---
elif page == "Testing Report":
    st.subheader("Testing / Evaluation Report")
    st.caption("Held-out evaluation: routed vs baseline across lengths and unseen domains.")
    tabs = st.tabs(["Routed (by length)", "Calibrated", "Baseline comparison", "Raw JSON"])

    with tabs[0]:
        routed = load_csv("moe_routed_metrics.csv")
        if routed is not None:
            st.dataframe(fmt_df(routed), use_container_width=True, hide_index=True)
        else:
            st.info("`results/moe_routed_metrics.csv` not found.")

    with tabs[1]:
        cal = load_csv("moe_calibrated_metrics.csv")
        if cal is not None:
            st.dataframe(fmt_df(cal), use_container_width=True, hide_index=True)
        else:
            st.info("`results/moe_calibrated_metrics.csv` not found.")

    with tabs[2]:
        comp = load_csv("moe_baseline_comparison.csv")
        if comp is not None:
            left, right = st.columns(2)
            with left:
                st.markdown("**Baseline (same eval slices)**")
                st.dataframe(fmt_df(comp), use_container_width=True, hide_index=True)
            with right:
                st.markdown("**MoE routed**")
                routed = load_csv("moe_routed_metrics.csv")
                if routed is not None:
                    st.dataframe(fmt_df(routed), use_container_width=True, hide_index=True)
            st.caption("Headline: MoE improves short-text and cross-domain robustness while keeping full-text accuracy high (see README/results).")
        else:
            st.info("`results/moe_baseline_comparison.csv` not found.")

    with tabs[3]:
        if summary:
            st.json(summary)
        else:
            st.info("`results/moe_summary.json` not found.")


# --- Figures ---
elif page == "Figures":
    st.subheader("Figures")
    st.caption("MoE architecture, per-expert validation, length & cross-domain results, baseline evaluation, and pipeline diagrams. Each figure includes Description and Key observations.")
    st.caption("Results, evaluation metrics and pipeline diagrams for the human vs AI text classification system. All figures are generated from held-out test data (results/moe_*.csv, results/baseline_ml_metrics.csv).")

    FIG_DETAILS = {
        "moe_architecture.png": (
            "Figure 1 — Mixture-of-Experts Architecture (Short / Long / VeryLong + Chat / Casual / Essay)",
            "Input text of any length is processed by two parallel gates. The **length gate** is deterministic (word count): ≤150 words → short, 151–400 → long, >400 → verylong. The **type gate** is a Logistic Regression over 16 stylometric features that labels text as chat, casual, or essay (validation accuracy 80.5%). Each expert consists of Word TF-IDF (1,2) 15k + Char TF-IDF (3,5) 25k followed by Logistic Regression (C=1.5). Output reports label (Human/AI), P(AI), confidence, routed expert, word count, text type, and reliability warning for short inputs.",
            "Training data: HC3 short Q&A (80.8k, median 117w) + modern LLM essays (18k, GPT-4/Claude/Llama3, median 510w) + train.csv 20k with truncation augmentation (50/100/150w, 200/300w). Short expert uses length-adaptive threshold 0.32–0.64; long and verylong use 0.50. This routing allows each expert to specialize in its length regime instead of a single model handling all lengths.",
        ),
        "moe_experts.png": (
            "Figure 2 — Per-Expert Validation (Held-out 20% per bucket)",
            "Three panels show Accuracy, F1, and ROC-AUC per expert. Short (n=4,000 val): 97.00% / 97.00% / 0.9960. Long (n=3,373): 98.19% / 98.32% / 0.9982. Verylong (n=2,285): 98.42% / 98.21% / 0.9955. Short-expert confusion: FP 63 / FN 57, approximately balanced across classes.",
            "Short text is the most difficult (least signal within 150 words) yet maintains 97.0% accuracy, confirming that specialization is effective. Long and verylong experts achieve 98%+ with AUC above 0.995, indicating strong separation in each regime.",
        ),
        "moe_accuracy_vs_length.png": (
            "Figure 3 — Accuracy vs Text Length: Baseline vs MoE Routed (5k held-out, truncations)",
            "Evaluation on the same 5,000 essays truncated to 25w, 50w, 100w, 150w, 200w and full length. Baseline LR: 67.4% (25w), 83.3% (50w), 94.1% (100w), 97.4% (150w), 97.6% (200w), 99.3% (full). MoE routed: **86.1% (25w, +18.7)**, **93.1% (50w, +9.8)**, 95.2% (100w), 95.9% (150w), 97.4% (200w), 99.0% (full).",
            "The gap is confined to short text: MoE is consistently above baseline below 150 words and converges with baseline above 150 words. Full-text performance is preserved (98.96% MoE vs 99.26% baseline), confirming no degradation on long text.",
        ),
        "moe_cross_domain.png": (
            "Figure 4 — Cross-Domain Generalization (Unseen Datasets)",
            "Original essays (full): Baseline 99.3%, MoE 99.0%. HC3 chat (unseen short Q&A): Baseline **48.6%**, MoE **98.5%**. Modern LLM essays (GPT-4/Claude, unseen): Baseline 85.5%, MoE **95.7%**. Baseline was not trained on HC3 or modern essays; MoE training included both sources.",
            "On familiar essays both systems are near-identical. On unseen text types, baseline degrades to chance on chat (48.6%) while MoE retains 98.5%, and MoE improves modern holdout by 10.2 points. This indicates generalization beyond truncation augmentation to new domains.",
        ),
        "moe_type_gate.png": (
            "Figure 5 — Type Gate Performance (Chat / Casual / Essay, n=1,500)",
            "Validation accuracy **80.5%**. Essay: Precision 0.90, Recall 0.92, F1 0.91. Chat: Precision 0.74, Recall 0.84, F1 0.79. Casual: Precision 0.77, Recall 0.66, F1 0.71. Features: log word count, sentence count/length, TTR, hapax ratio, average word length, contraction/informal/first-person/second-person rates, question/exclamation rates, capitalization, Flesch Reading Ease, Flesch-Kincaid Grade, burstiness.",
            "Essay is most separable (longer, more formal, higher grade level). Casual overlaps with both chat and essay and is therefore hardest. The type gate is used for interpretability and reliability indication (e.g. chat + short), while expert routing is determined by length.",
        ),
        "moe_pr_flip.png": (
            "Figure 6 — Precision and Recall at 25 Words (Error Balance)",
            "At 25 words: Precision (human correctness) — Baseline 0.53 vs MoE **0.93**. Recall (AI detection) — Baseline **0.99** vs MoE 0.68. Accuracy 67.4% vs **86.1%**; F1 0.69 vs **0.78**.",
            "Baseline over-predicts AI at short lengths (high recall, low precision: humans frequently called AI). MoE under-predicts AI (low recall, high precision: some AI called human) but yields higher accuracy and F1. Given very limited signal below 25 words, MoE's threshold calibration (0.32 at 25w) favors avoiding false human accusations, with confidence and warnings displayed.",
        ),
        "model_comparison.png": (
            "Figure 7 — Baseline Model Comparison (Mutlinomial NB vs Logistic Regression vs Linear SVM)",
            "Accuracy: NB 95.55%, LR 99.25%, Linear SVM (calibrated) **99.52%**. F1: NB 95.46%, LR 99.25%, SVM 99.52%. ROC-AUC: NB 0.9928, LR 0.9987, SVM 0.9991. All evaluated on the same held-out test set.",
            "Linear classifiers outperform Multinomial NB by approximately 4 points across metrics. LR was selected for baseline and for MoE experts because it provides direct P(AI) without additional calibration, while SVM requires calibration for probability estimates.",
        ),
        "confusion_matrices.png": (
            "Figure 8 — Confusion Matrices (Logistic Regression vs Linear SVM, 5,000 held-out)",
            "Logistic Regression: TN 3,129 (99.6%), FP 13 (0.4%), FN 24 (1.3%), TP 1,834 (98.7%) — 37 total errors. Linear SVM: TN 3,133 (99.7%), FP 9 (0.3%), FN 12 (0.6%), TP 1,846 (99.4%) — 21 total errors. Test set is ~63% human, so TN counts are larger than TP counts.",
            "Errors are low and balanced between false positives and false negatives; no systematic bias toward either class on full-length text. SVM has slightly fewer false negatives than LR.",
        ),
        "roc_curves.png": (
            "Figure 9 — ROC Curves (5,000 held-out test sample)",
            "AUC: Logistic Regression **0.9994**, Linear SVM **0.9995**, Chance line 0.5. Both curves remain in the top-left corner, achieving near 100% true positive rate at approximately 0% false positive rate.",
            "AUC near 1.0 indicates near-perfect ranking of Human vs AI across all thresholds, not only at 0.5. Full-length text is highly separable; short-text AUC is lower (short expert 0.9960), consistent with reduced signal at short lengths.",
        ),
        "class_distribution.png": (
            "Figure 10 — Class Distribution (Train.csv vs Test.csv)",
            "Train.csv (290,300): Human 50% / AI 50% (approximately 145k each). Test.csv (97,447): Human **63%** / AI **37%** (approximately 61k / 36k).",
            "Training is exactly balanced, avoiding prior bias toward either class. Test is imbalanced, so accuracy alone is insufficient (an all-Human baseline would score 63%). Evaluation therefore includes Precision, Recall, F1, ROC-AUC and confusion matrices in addition to accuracy.",
        ),
        "length_hist.png": (
            "Figure 11 — Text Length Distribution (5,000 test sample, characters)",
            "Distribution is unimodal with peak near 2,000 characters and **median 2,092 characters (~363 words)**. The range extends to ~6,000 characters with a right tail. Texts below 1,000 characters are relatively few.",
            "Median length (363w) falls in the long bucket (151–400w), so the long expert receives the majority of original training data. Short texts (≤150w) are in the left tail and required additional HC3 short Q&A data and truncation augmentation. Cutoffs 150 and 400 were derived from this distribution and length-ablation results.",
        ),
        "flow_train.png": (
            "Figure 12 — Training Workflow (train_baseline.py, 11 steps, 33s)",
            "Steps: (1) Load train.csv (290k) + test.csv (97k). (2) Balanced sample 10k + 10k = 20k (random_state=42). (3) Stratified split 16k train / 4k validation (50% AI in both). (4) Preprocess: NFKC normalization + lowercase. (5) TF-IDF: Word (1,2) 25k + Char char_wb (3,5) 50k, hstack to 16k × 75k sparse matrix. (6) Train NB / Logistic Regression (C=1.5) / Linear SVM (C=1.0, cv=3). (7) Evaluate on validation (4k) + held-out test (5k) and save models/*.pkl. Result: accuracy 99%+, AUC 0.999.",
            "The pipeline is stratified and reproducible in 33 seconds on CPU (no GPU required). Sparse TF-IDF keeps memory efficient. Saved .pkl artifacts (models/*.pkl, models/moe/*.pkl) allow inference without retraining.",
        ),
        "flow_infer.png": (
            "Figure 13 — Inference Workflow (test/inference.py, predict.py / api.py / Streamlit)",
            "Input modes: --text, .txt file, folder, CSV, API POST, LangChain tool — any length. Processing: load_artifacts() loads word_vec, char_vec, lr/svm (75k features); vectorize() + predict_proba() → P(AI); threshold 0.5 → Human/AI. Outputs: terminal table, predictions.csv, metrics.json, API JSON, Streamlit UI. Batch size 1,000 (no OOM). MoE inference adds: length routing, type labeling, length-adaptive threshold, and short-text reliability warning (<25w / ≤50w).",
            "The same preprocessing and vectorizers as training are reused to prevent train–test skew. P(AI) provides confidence; the UI displays P(AI), threshold, expert, word count, and text type. Batching allows the full 97k test file to be processed without out-of-memory errors.",
        ),
    }

    figure_sets = {
        "MoE overview": [
            "moe_architecture.png",
            "moe_experts.png",
            "moe_accuracy_vs_length.png",
            "moe_cross_domain.png",
            "moe_type_gate.png",
            "moe_pr_flip.png",
        ],
        "Baseline & evaluation": [
            "model_comparison.png",
            "confusion_matrices.png",
            "roc_curves.png",
            "class_distribution.png",
            "length_hist.png",
        ],
        "Pipeline": [
            "flow_train.png",
            "flow_infer.png",
        ],
    }
    for title, names in figure_sets.items():
        st.markdown(f"#### {title}")
        st.divider()
        for name in names:
            p = FIGURES_DIR / name
            if p.exists():
                cap, desc, interp = FIG_DETAILS[name]
                st.image(str(p), caption=cap, use_container_width=True)
                st.markdown(f"**Description**")
                st.markdown(desc)
                st.markdown(f"**Key observations**")
                st.markdown(interp)
                st.divider()


# --- About ---
else:
    st.subheader("About")
    st.markdown(
        """
**Purpose.** Conference-ready interface for the NIT Sikkim project on classifying human-written vs AI-generated text.

**What it shows**
- **Inference:** MoE router (length-gated experts + type gate + calibrated thresholds) and baseline TF-IDF classifiers.
- **Training report:** expert validation metrics, gate info, baseline reference.
- **Testing report:** routed/calibrated metrics and baseline-vs-MoE comparison from `results/`.
- **Figures:** architecture, length/cross-domain performance, confusion matrices, ROC curves, and pipeline diagrams.

**Notes for presentation**
- Very short inputs (<25–50 words) are inherently uncertain; the MoE reports warnings/confidence.
- Full-text accuracy remains high while short-text/cross-domain robustness improves over the baseline (see Testing Report).

**Key commands**
"""
    )
    st.code(
        "streamlit run app/app.py\n"
        "PYTHONPATH=. python3 moe/predict_moe.py \"example text\"",
        language="bash",
    )
