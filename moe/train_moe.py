"""
Train Mixture-of-Experts (MoE) for Human vs AI text classification.

Fixes the short-text failure of the baseline (see results/baseline_ml_metrics.csv
+ prior ablation: 25w -> 67% acc, 50w -> 83%, 1 sent -> 60%, humans biased to AI).

Architecture
------------
- Length gate (deterministic, auditable): word count -> short/long/verylong
      short    <= 150 words
      long     151-400 words
      verylong > 400 words          (cliff at ~100-150w, median ~363w in train.csv)
- Each expert has its OWN TF-IDF + LogisticRegression (classic MoE:
  each expert learns features for its length regime):
      word TF-IDF (1,2) 15k, min_df=2, sublinear
      char_wb TF-IDF (3,5) 25k, min_df=3, sublinear
      LR(C=1.5, lbfgs, max_iter=1000)  [same hyper as baseline]
- Type gate (learned): LR over 16 fast stylometric features ->
  chat / casual / essay, trained on source labels:
      HC3 open_qa  -> chat,  HC3 eli5/finance -> casual,
      truncated essays -> casual, full/modern essays -> essay.
  Used for routing insight + per-type eval + calibration warning.

Data pools (balanced, seed=42)
------------------------------
- SHORT: HC3 answers <=150w (chat/casual Q&A, new small-text data) +
         truncated copies of train.csv essays (prefix/random 50/100/150w)
         so short-ESSAY style is covered.
- LONG:  HC3 151-400w + modern essays 151-400w + train.csv essays
         truncated to 200/300w + natural train.csv 151-400w slice.
- VERYLONG: train.csv natural >400w + modern essays >400w (GPT-4/Claude/Llama3).

Held-out: data/test.csv 5k sample (same as train_baseline.py) + truncated
variants + 15% holdouts of HC3/modern never seen in training.

Run: PYTHONPATH=. python3 moe/train_moe.py
"""
import json
import os
import sys
import time
import gc
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text_for_tfidf
from moe.type_features import (
    length_bucket, word_count, truncate_words, random_window,
    type_feature_matrix, TYPE_FEATURES,
)

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models", "moe")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RNG = np.random.default_rng(42)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
    }


def balanced_sample(df, n_per_class, seed=42):
    """Sample min(n_per_class) rows per class (0/1), shuffled."""
    rng = np.random.default_rng(seed)
    out = []
    for lab in [0, 1]:
        sub = df[df["generated"] == lab]
        k = min(n_per_class, len(sub))
        out.append(sub.sample(k, random_state=seed + int(lab)).reset_index(drop=True))
    return pd.concat(out).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    t0 = time.time()
    log("MOE Step 1/8: Load sources")
    # --- existing long essays (same 10k/class subsample as baseline) ---
    df_raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["text", "generated"])
    log(f"  train.csv {df_raw.shape} counts={df_raw['generated'].value_counts().to_dict()}")
    df0 = df_raw[df_raw["generated"] == 0].sample(10000, random_state=42)
    df1 = df_raw[df_raw["generated"] == 1].sample(10000, random_state=42)
    base = pd.concat([df0, df1]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    base["source"] = "essay_base"
    del df_raw
    gc.collect()
    log(f"  base 20k essays: {base.shape}")

    hc3 = pd.read_csv(os.path.join(ROOT, "data", "moe", "hc3_small.csv"),
                      usecols=["text", "generated", "domain"])
    modern = pd.read_csv(os.path.join(ROOT, "data", "moe", "modern_essays.csv"),
                         usecols=["text", "generated", "model"])
    log(f"  HC3 {hc3.shape} {hc3['generated'].value_counts().to_dict()} domains={hc3['domain'].value_counts().to_dict()}")
    log(f"  modern {modern.shape} {modern['generated'].value_counts().to_dict()}")

    # hold out 15% of hc3/modern for eval (never trained on)
    hc3_tr, hc3_ho = train_test_split(hc3, test_size=0.15, random_state=42, stratify=hc3["generated"])
    mo_tr, mo_ho = train_test_split(modern, test_size=0.15, random_state=42, stratify=modern["generated"])
    log(f"  hc3 train {hc3_tr.shape} / holdout {hc3_ho.shape}; modern train {mo_tr.shape} / holdout {mo_ho.shape}")

    hc3_tr = hc3_tr.copy(); hc3_tr["source"] = "hc3"
    mo_tr = mo_tr.copy(); mo_tr["source"] = "modern"
    base["w"] = base["text"].str.split().str.len()
    hc3_tr["w"] = hc3_tr["text"].str.split().str.len()
    mo_tr["w"] = mo_tr["text"].str.split().str.len()

    log("MOE Step 2/8: Build length pools (truncation augmentation for short)")
    # --- truncated copies of base essays ---
    trunc_rows = []
    for _, r in base.iterrows():
        n = int(RNG.choice([50, 100, 150]))
        txt = truncate_words(r["text"], n) if RNG.random() < 0.5 else random_window(r["text"], n, RNG)
        trunc_rows.append({"text": txt, "generated": r["generated"], "source": "trunc_essay", "w": len(txt.split())})
    trunc = pd.DataFrame(trunc_rows)
    log(f"  truncated essays from base: {trunc.shape} {trunc['generated'].value_counts().to_dict()}")

    # LONG pool from base: natural 151-400 + truncated 200/300
    base_long = base[(base["w"] > 150) & (base["w"] <= 400)].copy()
    base_long = base_long.sample(min(6000, len(base_long)), random_state=42)
    trunc300_rows = []
    for _, r in base[base["w"] >= 200].sample(min(6000, (base["w"] >= 200).sum()), random_state=43).iterrows():
        n = int(RNG.choice([200, 300]))
        txt = truncate_words(r["text"], n)
        if len(txt.split()) >= 151:
            trunc300_rows.append({"text": txt, "generated": r["generated"], "source": "trunc_essay", "w": len(txt.split())})
    base_long = pd.concat([base_long[["text", "generated", "source"]], pd.DataFrame(trunc300_rows)], ignore_index=True)

    hc3_short = hc3_tr[hc3_tr["w"] <= 150][["text", "generated", "source"]]
    hc3_long = hc3_tr[(hc3_tr["w"] > 150) & (hc3_tr["w"] <= 400)][["text", "generated", "source"]]
    mo_long = mo_tr[(mo_tr["w"] > 150) & (mo_tr["w"] <= 400)][["text", "generated", "source"]]
    mo_vl = mo_tr[mo_tr["w"] > 400][["text", "generated", "source"]]
    base_vl = base[base["w"] > 400][["text", "generated", "source"]]
    log(f"  hc3_short={len(hc3_short)} hc3_long={len(hc3_long)} mo_long={len(mo_long)} mo_vl={len(mo_vl)} base_vl={len(base_vl)} base_long={len(base_long)}")

    # balanced caps per pool (seed 42)
    pool_short = pd.concat([
        balanced_sample(hc3_short, 7000),
        balanced_sample(trunc[["text", "generated", "source"]], 3000),
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    pool_long = pd.concat([
        balanced_sample(hc3_long, 4000),
        balanced_sample(mo_long, 2000),
        balanced_sample(base_long, 3000),
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    pool_vl = pd.concat([
        balanced_sample(base_vl, 7000),
        balanced_sample(mo_vl, 2500),
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    pools = {"short": pool_short, "long": pool_long, "verylong": pool_vl}
    for k, v in pools.items():
        log(f"  POOL {k}: n={len(v)} counts={v['generated'].value_counts().to_dict()} sources={v['source'].value_counts().to_dict()}")
        v["w"] = v["text"].str.split().str.len()
        log(f"    words mean={v['w'].mean():.0f} median={v['w'].median():.0f} max={v['w'].max()} min={v['w'].min()}")

    log("MOE Step 3/8: Train 3 experts (per-bucket TF-IDF + LR)")
    experts = {}
    expert_metrics = {"expert": "bucket", "accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
    rows = []
    for bucket, pool in pools.items():
        texts = pool["text"].astype(str).tolist()
        y = pool["generated"].astype(int).values
        X_tr_txt, X_va_txt, y_tr, y_va = train_test_split(
            np.array(texts, dtype=object), y, test_size=0.20, random_state=42, stratify=y)
        log(f"  [{bucket}] train={len(X_tr_txt)} val={len(X_va_txt)}")
        t = time.time()
        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=2,
                                   sublinear_tf=True, preprocessor=clean_text_for_tfidf)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000,
                                   min_df=3, sublinear_tf=True)
        Xtr_w = word_vec.fit_transform(X_tr_txt)
        Xva_w = word_vec.transform(X_va_txt)
        Xtr_c = char_vec.fit_transform(X_tr_txt)
        Xva_c = char_vec.transform(X_va_txt)
        Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
        Xva = hstack([Xva_w, Xva_c]).tocsr()
        log(f"    tfidf done {time.time()-t:.1f}s vocab word={len(word_vec.vocabulary_)} char={len(char_vec.vocabulary_)} X={Xtr.shape}")
        del Xtr_w, Xtr_c, Xva_w, Xva_c, X_tr_txt, X_va_txt
        clf = LogisticRegression(C=1.5, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(Xtr, y_tr)
        pred = clf.predict(Xva)
        prob = clf.predict_proba(Xva)[:, 1]
        m = metrics(y_va, pred, prob)
        cm = confusion_matrix(y_va, pred)
        log(f"    {bucket} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} cm={cm.tolist()}")
        log(classification_report(y_va, pred, target_names=["Human", "AI"]))
        rows.append({"expert": bucket, "n_train": int(len(y_tr)), "n_val": int(len(y_va)), **m,
                     "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]), "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1])})
        joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf, "bucket": bucket,
                     "short_max": 150, "long_max": 400}, os.path.join(MODELS_DIR, f"expert_{bucket}.pkl"))
        del word_vec, char_vec, clf, Xtr, Xva
        gc.collect()

    log("MOE Step 4/8: Train type gate (chat/casual/essay)")
    # pseudo-labels by source; then learn LR on stylometrics
    def type_label(src, domain=None):
        if src == "hc3":
            return "chat" if domain == "open_qa" else "casual"
        return "essay" if src in ("essay_base", "modern") else "casual"

    hc3_dom = hc3_tr.copy()
    # keep domain for gate labels, rebuild label series
    gate_src = []
    for df_ in [pool_short, pool_long, pool_vl]:
        gate_src.append(df_[["text", "source"]].copy())
    # build gate data per-type balanced (chat scarce: open_qa only ~4k rows).
    # chat <= open_qa train rows, casual <= eli5/finance + trunc, essay <= base + modern.
    chat_pool = hc3_tr[hc3_tr["domain"] == "open_qa"]
    casual_pool = pd.concat([
        hc3_tr[hc3_tr["domain"] != "open_qa"][["text"]],
        trunc[["text"]],
    ], ignore_index=True)
    essay_pool = pd.concat([base[["text"]], mo_tr[["text"]]], ignore_index=True)
    n_chat = min(2500, len(chat_pool))
    n_casual = min(2500, len(casual_pool))
    n_essay = min(2500, len(essay_pool))
    gate_frames = []
    for _, r in chat_pool.sample(n_chat, random_state=44).iterrows():
        gate_frames.append({"text": r["text"], "type": "chat"})
    for _, r in casual_pool.sample(n_casual, random_state=45).iterrows():
        gate_frames.append({"text": r["text"], "type": "casual"})
    for _, r in essay_pool.sample(n_essay, random_state=46).iterrows():
        gate_frames.append({"text": r["text"], "type": "essay"})
    gate_df = pd.DataFrame(gate_frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"  gate n={len(gate_df)} {gate_df['type'].value_counts().to_dict()}")
    Xg = type_feature_matrix(gate_df["text"].tolist())
    yg = gate_df["type"].to_numpy(dtype=object)
    Xg_tr, Xg_va, yg_tr, yg_va = train_test_split(Xg, yg, test_size=0.20, random_state=42, stratify=yg)
    type_clf = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42))])
    type_clf.fit(Xg_tr, yg_tr)
    type_pred = type_clf.predict(Xg_va)
    type_acc = float(accuracy_score(yg_va, type_pred))
    log(f"  type gate acc={type_acc:.4f}")
    log(classification_report(yg_va, type_pred))
    joblib.dump({"clf": type_clf, "features": TYPE_FEATURES},
                os.path.join(MODELS_DIR, "type_gate.pkl"))
    with open(os.path.join(MODELS_DIR, "gate_config.json"), "w") as f:
        json.dump({"short_max_words": 150, "long_max_words": 400,
                   "gate": "word_count", "type_features": TYPE_FEATURES,
                   "type_accuracy_val": type_acc}, f, indent=2)

    log("MOE Step 5/8: Route evaluation vs baseline (held-out test.csv 5k)")
    df_test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["text", "generated"])
    df_te = df_test.sample(5000, random_state=42).reset_index(drop=True)
    del df_test
    gc.collect()
    y_te = df_te["generated"].astype(int).values

    def route_predict(texts):
        experts_l = {k: joblib.load(os.path.join(MODELS_DIR, f"expert_{k}.pkl")) for k in ["short", "long", "verylong"]}
        buck_preds = {"short": [], "long": [], "verylong": []}
        buck_probs = {"short": [], "long": [], "verylong": []}
        all_pred, all_prob, all_bucket = [], [], []
        # batch by bucket for speed
        import collections
        groups = collections.defaultdict(list)
        for i, t in enumerate(texts):
            groups[length_bucket(word_count(t))].append(i)
        pred_arr = np.zeros(len(texts), dtype=int)
        prob_arr = np.zeros(len(texts))
        bucket_arr = [""] * len(texts)
        for b, idxs in groups.items():
            ex = experts_l[b]
            batch = [texts[i] for i in idxs]
            Xb = hstack([ex["word_vec"].transform(batch), ex["char_vec"].transform(batch)]).tocsr()
            p = ex["clf"].predict(Xb)
            pr = ex["clf"].predict_proba(Xb)[:, 1]
            for j, i in enumerate(idxs):
                pred_arr[i] = p[j]; prob_arr[i] = pr[j]; bucket_arr[i] = b
            del Xb
        del experts_l
        gc.collect()
        return pred_arr, prob_arr, bucket_arr

    def eval_routed(texts, y, label):
        pred, prob, buckets = route_predict(texts)
        m = metrics(y, pred, prob)
        log(f"  {label:32s} n={len(y)} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
            f"short_n={sum(1 for b in buckets if b=='short')} long_n={sum(1 for b in buckets if b=='long')} vl_n={sum(1 for b in buckets if b=='verylong')}")
        return m, pred, prob

    moe_rows = []
    m_full, _, _ = eval_routed(df_te["text"].astype(str).tolist(), y_te, "MoE full text")
    moe_rows.append({"test_condition": "full", **m_full})
    for n in [25, 50, 100, 150, 200]:
        texts = [truncate_words(t, n) for t in df_te["text"]]
        m, _, _ = eval_routed(texts, y_te, f"MoE first {n} words")
        moe_rows.append({"test_condition": f"first_{n}_words", **m})
    import re as _re
    def first_sents(t, k):
        import re
        s = re.split(r"(?<=[.!?])\s+", str(t))
        return " ".join(s[:k])
    for k in [1, 2, 3]:
        texts = [first_sents(t, k) for t in df_te["text"]]
        m, _, _ = eval_routed(texts, y_te, f"MoE first {k} sent")
        moe_rows.append({"test_condition": f"first_{k}_sent", **m})
    # held-out HC3 + modern (new domains)
    hc3_ho_texts = hc3_ho["text"].astype(str).tolist()
    y_hc3 = hc3_ho["generated"].astype(int).values
    m_h, _, _ = eval_routed(hc3_ho_texts, y_hc3, "MoE HC3 holdout")
    moe_rows.append({"test_condition": "hc3_holdout", **m_h})
    mo_ho_texts = mo_ho["text"].astype(str).tolist()
    y_mo = mo_ho["generated"].astype(int).values
    m_m, _, _ = eval_routed(mo_ho_texts, y_mo, "MoE modern holdout")
    moe_rows.append({"test_condition": "modern_holdout", **m_m})
    # per-bucket on full
    pred_f, prob_f, buck_f = route_predict(df_te["text"].astype(str).tolist())
    for b in ["short", "long", "verylong"]:
        mask = np.array([x == b for x in buck_f])
        if mask.sum() > 0 and len(set(y_te[mask])) > 1:
            mb = metrics(y_te[mask], pred_f[mask], prob_f[mask])
            log(f"    bucket {b}: n={mask.sum()} acc={mb['accuracy']:.4f} f1={mb['f1']:.4f}")

    log("MOE Step 6/8: Compare vs baseline single model on same conditions")
    base_dir = os.path.join(ROOT, "models")
    if os.path.exists(os.path.join(base_dir, "word_tfidf_vectorizer.pkl")):
        wv = joblib.load(os.path.join(base_dir, "word_tfidf_vectorizer.pkl"))
        cv = joblib.load(os.path.join(base_dir, "char_tfidf_vectorizer.pkl"))
        lr = joblib.load(os.path.join(base_dir, "logistic_regression_baseline.pkl"))

        def base_eval(texts, y, label):
            Xb = hstack([wv.transform(texts), cv.transform(texts)]).tocsr()
            p = lr.predict(Xb); pr = lr.predict_proba(Xb)[:, 1]
            m = metrics(y, p, pr)
            log(f"  [BASE] {label:28s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
            return m

        base_rows = [{"test_condition": "full", **base_eval(df_te["text"].astype(str).tolist(), y_te, "full")}]
        for n in [25, 50, 100, 150, 200]:
            base_rows.append({"test_condition": f"first_{n}_words",
                              **base_eval([truncate_words(t, n) for t in df_te["text"]], y_te, f"first_{n}")})
        base_rows.append({"test_condition": "hc3_holdout", **base_eval(hc3_ho_texts, y_hc3, "hc3")})
        base_rows.append({"test_condition": "modern_holdout", **base_eval(mo_ho_texts, y_mo, "modern")})
        base_rows.append({"test_condition": "first_1_sent",
                          **base_eval([first_sents(t, 1) for t in df_te["text"]], y_te, "first_1_sent")})
        pd.DataFrame(base_rows).to_csv(os.path.join(RESULTS_DIR, "moe_baseline_comparison.csv"), index=False)
        log(f"  baseline comparison -> results/moe_baseline_comparison.csv")
        del wv, cv, lr
        gc.collect()

    log("MOE Step 7/8: Save results")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "moe_expert_metrics.csv"), index=False)
    pd.DataFrame(moe_rows).to_csv(os.path.join(RESULTS_DIR, "moe_routed_metrics.csv"), index=False)
    summary = {"experts": rows, "routed": moe_rows, "type_gate_accuracy": type_acc,
               "gate": {"short_max_words": 150, "long_max_words": 400}}
    with open(os.path.join(RESULTS_DIR, "moe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(pd.DataFrame(rows).to_string(index=False))
    log(pd.DataFrame(moe_rows).to_string(index=False))
    log(f"TOTAL TIME {time.time()-t0:.1f}s. Artifacts -> models/moe/")


if __name__ == "__main__":
    main()
