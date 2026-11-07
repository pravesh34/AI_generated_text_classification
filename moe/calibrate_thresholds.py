"""
Calibrate per-bucket decision thresholds for the MoE experts.

Why: default 0.50 is not optimal per length. Sweep on truncated test.csv
showed: 25w wants ~0.30 (MoE under-calls AI: P=0.93/R=0.68), while
100-150w (and HC3 holdout) want ~0.60 (false AI calls dominate).
A single global threshold can't fix both -> length-adaptive thresholds.

Protocol (no leakage):
  - test.csv 5k (never trained on) split 50/50 stratified (seed 999):
    calib 2500 for threshold selection, eval 2500 for reporting.
  - HC3 holdout (12k, also never trained) used as extra transfer check.
  - Objective: maximize F1 on calib; tie-break by accuracy.

Outputs:
  - models/moe/thresholds.json
  - results/moe_calibrated_metrics.csv

Run: PYTHONPATH=. python3 moe/calibrate_thresholds.py
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moe.type_features import length_bucket, word_count

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(ROOT, "models", "moe")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

def trunc(t, n):
    return " ".join(str(t).split()[:n])

def probs_for(texts, experts):
    """Route by length, return (prob_ai, bucket) per text."""
    import collections
    groups = collections.defaultdict(list)
    for i, t in enumerate(texts):
        groups[length_bucket(word_count(t))].append(i)
    prob = np.zeros(len(texts))
    buck = [""] * len(texts)
    for b, idxs in groups.items():
        ex = experts[b]
        batch = [texts[i] for i in idxs]
        X = hstack([ex["word_vec"].transform(batch), ex["char_vec"].transform(batch)]).tocsr()
        pr = ex["clf"].predict_proba(X)[:, 1]
        for j, i in enumerate(idxs):
            prob[i] = pr[j]
            buck[i] = b
    return prob, buck

def best_threshold(y, p, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.30, 0.76, 0.02), 2)
    best = (0.5, -1, -1)  # th, f1, acc
    rows = []
    for th in grid:
        pred = (p >= th).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        acc = accuracy_score(y, pred)
        rows.append((float(th), float(acc), float(f1)))
        if (f1, acc) > (best[1], best[2]):
            best = (float(th), float(f1), float(acc))
    return best[0], rows

def main():
    experts = {b: joblib.load(os.path.join(MODELS, f"expert_{b}.pkl")) for b in ["short", "long", "verylong"]}
    df_test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["text", "generated"])
    df_te = df_test.sample(5000, random_state=42).reset_index(drop=True)
    del df_test
    calib, eval_df = train_test_split(df_te, test_size=0.5, random_state=999, stratify=df_te["generated"])
    calib = calib.reset_index(drop=True); eval_df = eval_df.reset_index(drop=True)
    y_c = calib["generated"].astype(int).values
    y_e = eval_df["generated"].astype(int).values
    print(f"calib n={len(calib)} eval n={len(eval_df)}")

    # probs on calib for each condition
    conds = {
        "ultra_25": [trunc(t, 25) for t in calib["text"]],
        "ultra_50": [trunc(t, 50) for t in calib["text"]],
        "short_100": [trunc(t, 100) for t in calib["text"]],
        "short_150": [trunc(t, 150) for t in calib["text"]],
        "full": calib["text"].astype(str).tolist(),
    }
    P = {}
    for k, texts in conds.items():
        P[k], _ = probs_for(texts, experts)

    th_ultra, _ = best_threshold(y_c, np.concatenate([P["ultra_25"], P["ultra_50"]]) if False else P["ultra_25"])
    # combine 25+50 for ultra (stack y duplicated)
    y_ultra = np.concatenate([y_c, y_c]); p_ultra = np.concatenate([P["ultra_25"], P["ultra_50"]])
    th_ultra, rows_ultra = best_threshold(y_ultra, p_ultra)
    y_short = np.concatenate([y_c, y_c]); p_short = np.concatenate([P["short_100"], P["short_150"]])
    th_short, rows_short = best_threshold(y_short, p_short)
    # long / verylong from full-text slices routed to those buckets
    _, buck_full = probs_for(conds["full"], experts)
    buck_full = np.array(buck_full)
    th_long, rows_long = best_threshold(y_c[buck_full == "long"], P["full"][buck_full == "long"])
    th_vl, rows_vl = best_threshold(y_c[buck_full == "verylong"], P["full"][buck_full == "verylong"])
    print(f"calibrated: ultra(<=50w)={th_ultra} short(51-150w)={th_short} long={th_long} verylong={th_vl}")
    for label, rows_ in [("ultra", rows_ultra), ("short", rows_short), ("long", rows_long), ("verylong", rows_vl)]:
        top = sorted(rows_, key=lambda r: (r[2], r[1]), reverse=True)[:3]
        print(f" {label} top3 (th,acc,f1): {[(round(t,2),round(a,4),round(f,4)) for t,a,f in top]}")

    thresholds = {"short_ultra_words_max": 50, "short_ultra": float(th_ultra),
                  "short": float(th_short), "long": float(th_long), "verylong": float(th_vl),
                  "default": 0.5}
    with open(os.path.join(MODELS, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    def apply_thresh(prob, buck, n_words):
        out = np.zeros(len(prob), dtype=int)
        for i in range(len(prob)):
            b, n = buck[i], n_words[i]
            if b == "short" and n <= 50:
                th = th_ultra
            elif b == "short":
                th = th_short
            elif b == "long":
                th = th_long
            else:
                th = th_vl
            out[i] = int(prob[i] >= th)
        return out

    def eval_all(df, y, label):
        rows = []
        eval_conds = {
            "full": df["text"].astype(str).tolist(),
            "first_25_words": [trunc(t, 25) for t in df["text"]],
            "first_50_words": [trunc(t, 50) for t in df["text"]],
            "first_100_words": [trunc(t, 100) for t in df["text"]],
            "first_150_words": [trunc(t, 150) for t in df["text"]],
            "first_200_words": [trunc(t, 200) for t in df["text"]],
        }
        for cond, texts in eval_conds.items():
            prob, buck = probs_for(texts, experts)
            preds = {0.50: (prob >= 0.50).astype(int), "calibrated": apply_thresh(prob, buck, np.array([word_count(t) for t in texts]))}
            for name, pred in preds.items():
                rows.append({"subset": label, "condition": cond, "threshold": name,
                             "accuracy": float(accuracy_score(y, pred)),
                             "precision": float(precision_score(y, pred, zero_division=0)),
                             "recall": float(recall_score(y, pred, zero_division=0)),
                             "f1": float(f1_score(y, pred, zero_division=0)),
                             "roc_auc": float(roc_auc_score(y, prob))})
        return rows

    print("\n--- EVAL half (2500, unseen for calibration) ---")
    rows_eval = eval_all(eval_df, y_e, "eval")
    # HC3 holdout as transfer check
    hc3 = pd.read_csv(os.path.join(ROOT, "data", "moe", "hc3_small.csv"), usecols=["text", "generated"])
    _, hc3_ho = train_test_split(hc3, test_size=0.15, random_state=42, stratify=hc3["generated"])
    hc3_ho = hc3_ho.reset_index(drop=True)
    y_h = hc3_ho["generated"].astype(int).values
    rows_hc3 = eval_all(hc3_ho, y_h, "hc3_holdout")
    all_rows = rows_eval + rows_hc3
    for r in all_rows:
        if r["condition"] in ("full", "first_25_words", "first_50_words", "first_100_words", "first_150_words") or r["subset"] == "hc3_holdout":
            print(f"{r['subset']:11s} {r['condition']:16s} {str(r['threshold']):10s} acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")
    pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS, "moe_calibrated_metrics.csv"), index=False)
    print(f"\nSaved {os.path.join(MODELS, 'thresholds.json')} and {os.path.join(RESULTS, 'moe_calibrated_metrics.csv')}")


if __name__ == "__main__":
    main()
