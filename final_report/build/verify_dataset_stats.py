"""Verify corpus-level statistics from the actual CSV files.

Writes: final_report/build/dataset_stats_verified.json
Covered claims: row counts, class counts, word-length median/mean/percentiles
for train.csv, test.csv, HC3 and modern-LLM essays.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out = {}


def wordstats(series: pd.Series) -> dict:
    w = series.astype(str).str.split().str.len().to_numpy()
    return {
        "n": int(len(w)),
        "mean": float(np.mean(w)),
        "median": float(np.median(w)),
        "p5": float(np.percentile(w, 5)),
        "p10": float(np.percentile(w, 10)),
        "p90": float(np.percentile(w, 90)),
        "max": int(np.max(w)),
    }


print("train.csv ...", flush=True)
tr = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["text", "generated"])
out["train.csv"] = {
    "rows": int(len(tr)),
    "human": int((tr["generated"] == 0).sum()),
    "ai": int((tr["generated"] == 1).sum()),
    "size_mb": round(os.path.getsize(os.path.join(ROOT, "data/train.csv")) / 1e6),
}
out["train.csv"].update(wordstats(tr["text"]))
print(out["train.csv"], flush=True)
hw = tr[tr["generated"] == 0]["text"].astype(str).str.split().str.len()
aw = tr[tr["generated"] == 1]["text"].astype(str).str.split().str.len()
out["train.csv"]["human_mean"] = float(hw.mean())
out["train.csv"]["ai_mean"] = float(aw.mean())
del tr

print("test.csv ...", flush=True)
te = pd.read_csv(os.path.join(ROOT, "data/test.csv"), usecols=["text", "generated"])
out["test.csv"] = {
    "rows": int(len(te)),
    "human": int((te["generated"] == 0).sum()),
    "ai": int((te["generated"] == 1).sum()),
    "size_mb": round(os.path.getsize(os.path.join(ROOT, "data/test.csv")) / 1e6),
}
out["test.csv"].update(wordstats(te["text"]))
print(out["test.csv"], flush=True)
del te

print("hc3 ...", flush=True)
hc = pd.read_csv(os.path.join(ROOT, "data/moe/hc3_small.csv"))
label_col = "generated" if "generated" in hc.columns else (
    "label" if "label" in hc.columns else hc.columns[-1])
txt_col = "text" if "text" in hc.columns else hc.columns[0]
w = hc[txt_col].astype(str).str.split().str.len()
out["hc3_small.csv"] = {
    "rows": int(len(hc)),
    "label_col": str(label_col),
    "human": int((hc[label_col] == 0).sum()),
    "ai": int((hc[label_col] == 1).sum()),
    "median": float(np.median(w)), "mean": float(np.mean(w)),
    "pct_le_150": float((w <= 150).mean() * 100),
    "pct_151_400": float(((w > 150) & (w <= 400)).mean() * 100),
    "pct_gt_400": float((w > 400).mean() * 100),
    "size_mb": round(os.path.getsize(os.path.join(ROOT, "data/moe/hc3_small.csv")) / 1e6),
}
print(out["hc3_small.csv"], flush=True)
del hc

print("modern ...", flush=True)
mo = pd.read_csv(os.path.join(ROOT, "data/moe/modern_essays.csv"))
label_col = "generated" if "generated" in mo.columns else (
    "label" if "label" in mo.columns else mo.columns[-1])
txt_col = "text" if "text" in mo.columns else mo.columns[0]
w = mo[txt_col].astype(str).str.split().str.len()
out["modern_essays.csv"] = {
    "rows": int(len(mo)),
    "label_col": str(label_col),
    "human": int((mo[label_col] == 0).sum()),
    "ai": int((mo[label_col] == 1).sum()),
    "median": float(np.median(w)), "mean": float(np.mean(w)),
    "pct_gt_400": float((w > 400).mean() * 100),
    "size_mb": round(os.path.getsize(os.path.join(ROOT, "data/moe/modern_essays.csv")) / 1e6),
}
print(out["modern_essays.csv"], flush=True)

dest = os.path.join(os.path.dirname(__file__), "dataset_stats_verified.json")
with open(dest, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", dest)
