"""Recompute held-out test metrics from the FROZEN model artifacts.

Purpose: technical audit — every baseline test number reported in the report
must be reproducible from models/*.pkl + data/test.csv (same seed 42 sample
that produced results/figures/confusion_matrices.png).

Writes: final_report/build/metrics_verified.json
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

print("loading models ...")
word_vec = joblib.load(os.path.join(ROOT, "models/word_tfidf_vectorizer.pkl"))
char_vec = joblib.load(os.path.join(ROOT, "models/char_tfidf_vectorizer.pkl"))
lr = joblib.load(os.path.join(ROOT, "models/logistic_regression_baseline.pkl"))
svm = joblib.load(os.path.join(ROOT, "models/linear_svm_baseline.pkl"))

print("loading test.csv (212 MB) ...")
df = pd.read_csv(os.path.join(ROOT, "data/test.csv"), usecols=["text", "generated"])
s = df.sample(5000, random_state=42).reset_index(drop=True)
texts = np.array(s["text"].tolist(), dtype=object)
y = np.array(s["generated"].tolist(), dtype=int)

print("vectorizing ...")
X = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()

out = {
    "n_test": int(len(y)),
    "n_human": int((y == 0).sum()),
    "n_ai": int((y == 1).sum()),
    "word_stats": {},
    "models": {},
}

# word-length statistics of the same sample (corpus analysis in Chapter 3)
words = np.array([len(str(t).split()) for t in texts])
hw = words[y == 0]
aw = words[y == 1]
out["word_stats"] = {
    "mean": float(words.mean()), "median": float(np.median(words)),
    "p10": float(np.percentile(words, 10)), "p90": float(np.percentile(words, 90)),
    "min": int(words.min()), "max": int(words.max()),
    "human_mean": float(hw.mean()), "ai_mean": float(aw.mean()),
    "human_min": int(hw.min()),
    "median_chars": float(np.median([len(str(t)) for t in texts])),
    "mean_chars": float(np.mean([len(str(t)) for t in texts])),
}

for name, model in (("lr", lr), ("svm", svm)):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    out["models"][name] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(name, out["models"][name])

dest = os.path.join(os.path.dirname(__file__), "metrics_verified.json")
with open(dest, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", dest)
