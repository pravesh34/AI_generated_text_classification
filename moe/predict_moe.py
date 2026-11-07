"""
Inference for the Mixture-of-Experts (MoE) upgrade.

Routes by length (<=150 short / 151-400 long / >400 verylong), then reports:
  - predicted label (AI-generated vs Human-written)
  - probability + confidence
  - routed expert, word count, detected text type (chat/casual/essay)
  - reliability warning for very short text.

Usage:
  PYTHONPATH=. python3 moe/predict_moe.py "your text here"
  echo "text" | PYTHONPATH=. python3 moe/predict_moe.py -
"""
import json
import os
import sys
import joblib
from scipy.sparse import hstack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moe.type_features import length_bucket, word_count, extract_type_features, TYPE_FEATURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(ROOT, "models", "moe")


def _threshold(bucket: str, n: int) -> float:
    """Length-adaptive threshold. Short expert fitted per length on calib half:
    25w->0.32, 50w->0.40, 100w->0.56, 150w->0.64; linear approx below."""
    th_path = os.path.join(MODELS, "thresholds.json")
    if os.path.exists(th_path):
        with open(th_path) as f:
            cfg = json.load(f)
        if bucket == "short" and "short_formula" in cfg:
            fm = cfg["short_formula"]
            return float(min(fm["high"], max(fm["low"], fm["intercept"] + fm["slope"] * n)))
        if bucket in cfg:
            return float(cfg[bucket])
    return 0.5


def predict(text: str):
    n = word_count(text)
    bucket = length_bucket(n)
    ex = joblib.load(os.path.join(MODELS, f"expert_{bucket}.pkl"))
    X = hstack([ex["word_vec"].transform([text]), ex["char_vec"].transform([text])]).tocsr()
    prob_ai = float(ex["clf"].predict_proba(X)[0, 1])
    th = _threshold(bucket, n)
    label = "AI-Generated" if prob_ai >= th else "Human-Written"
    # type gate
    type_label, type_conf = "unknown", 0.0
    gate_path = os.path.join(MODELS, "type_gate.pkl")
    if os.path.exists(gate_path):
        gate = joblib.load(gate_path)
        feats = extract_type_features(text)
        import numpy as np
        Xg = np.array([[feats[k] for k in gate["features"]]])
        type_label = gate["clf"].predict(Xg)[0]
        type_conf = float(gate["clf"].predict_proba(Xg).max())
    conf = max(prob_ai, 1 - prob_ai)
    warn = ""
    if n < 25:
        warn = "WARNING: <25 words - very low signal, treat as uncertain."
    elif n <= 50:
        warn = "CAUTION: <=50 words - short-text range, check confidence."
    return {
        "label": label,
        "prob_ai": round(prob_ai, 4),
        "prob_human": round(1 - prob_ai, 4),
        "confidence": round(conf, 4),
        "threshold_used": round(th, 3),
        "expert": bucket,
        "words": n,
        "text_type": type_label,
        "type_confidence": round(type_conf, 3),
        "warning": warn,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-h":
        print(__doc__)
        sys.exit(1)
    text = sys.stdin.read() if sys.argv[1] == "-" else " ".join(sys.argv[1:])
    print(json.dumps(predict(text), indent=2))
