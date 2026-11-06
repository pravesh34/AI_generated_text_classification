"""
test/inference.py — Core inference engine for Human vs AI Text Classification.
Loads artifacts from ../models/ and provides single / batch / CSV evaluation APIs.
Works with: Dual TF-IDF (word 1-2 + char_wb 3-5) + Logistic Regression / Linear SVM (Calibrated).
"""
import os
import sys
import json
from pathlib import Path

# project root is parent of test/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

from src.stylometrics import extract_stylometric_features

MODELS_DIR = ROOT / "models"


def load_artifacts(models_dir=None):
    """Load vectorizers + classifiers. Returns dict."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    artifacts = {}
    paths = {
        "word_vec": models_dir / "word_tfidf_vectorizer.pkl",
        "char_vec": models_dir / "char_tfidf_vectorizer.pkl",
        "lr": models_dir / "logistic_regression_baseline.pkl",
        "svm": models_dir / "linear_svm_baseline.pkl",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {models_dir}: {missing}\n"
            f"Train first: venv/bin/python train_baseline.py"
        )
    for k, p in paths.items():
        artifacts[k] = joblib.load(p)
    return artifacts


def vectorize(texts, artifacts):
    """texts: list[str] -> sparse matrix (n, 75000)."""
    if isinstance(texts, str):
        texts = [texts]
    Xw = artifacts["word_vec"].transform(texts)
    Xc = artifacts["char_vec"].transform(texts)
    return hstack([Xw, Xc]).tocsr()


def predict_proba(texts, artifacts, model="lr"):
    """Returns array of P(AI) for each text."""
    X = vectorize(texts, artifacts)
    clf = artifacts[model]
    return clf.predict_proba(X)[:, 1]


def predict_text(text, artifacts=None, model="lr", show_stylometrics=False):
    """
    Single-text inference.
    Returns dict: predicted_label, probability_ai, probability_human, confidence, stylometrics (optional)
    """
    if artifacts is None:
        artifacts = load_artifacts()
    if not text or not text.strip():
        raise ValueError("Empty text — paste some text to classify.")
    p_ai = float(predict_proba([text], artifacts, model)[0])
    label = "AI-Generated" if p_ai >= 0.5 else "Human-Written"
    confidence = p_ai if p_ai >= 0.5 else 1.0 - p_ai
    result = {
        "predicted_label": label,
        "probability_ai": p_ai,
        "probability_human": 1.0 - p_ai,
        "confidence": confidence,
        "model": "Logistic Regression" if model == "lr" else "Linear SVM",
        "text_preview": text[:120].replace("\n", " "),
    }
    if show_stylometrics:
        stylo = extract_stylometric_features(text)
        result["stylometrics"] = {
            k: float(v) for k, v in stylo.items()
        }
    return result


def predict_batch(texts, artifacts=None, model="lr", batch_size=1000):
    """Efficient batch inference (avoids OOM on full test.csv 97k rows)."""
    if artifacts is None:
        artifacts = load_artifacts()
    probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        probs.extend(predict_proba(chunk, artifacts, model).tolist())
    return np.array(probs)


def evaluate_csv(csv_path, artifacts=None, model="lr", sample_n=None, seed=42, batch_size=1000):
    """
    Full evaluation on a CSV with columns text,generated.
    Returns (metrics dict, df with predictions).
    """
    if artifacts is None:
        artifacts = load_artifacts()
    df = pd.read_csv(csv_path, usecols=["text", "generated"])
    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)
    texts = np.array(df["text"].astype(str).tolist(), dtype=object)
    y_true = np.array(df["generated"].tolist(), dtype=int)
    y_prob = predict_batch(texts, artifacts, model=model, batch_size=batch_size)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Human", "AI"], output_dict=True
        ),
    }
    df = df.copy()
    df["prob_ai"] = y_prob
    df["pred"] = y_pred
    df["correct"] = (y_pred == y_true)
    return metrics, df


def predict_folder(folder, artifacts=None, model="lr"):
    """Predict on all .txt files in a folder. Returns list of dicts."""
    if artifacts is None:
        artifacts = load_artifacts()
    folder = Path(folder)
    results = []
    for p in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        r = predict_text(text, artifacts, model=model, show_stylometrics=True)
        r["file"] = p.name
        r["path"] = str(p)
        results.append(r)
    return results


def save_results(results, metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # predictions.csv if results looks like a DataFrame-like list
    if results is not None and len(results) > 0 and isinstance(results[0], dict) and "prob_ai" not in results[0]:
        pd.DataFrame(results).to_csv(out_dir / "predictions.csv", index=False)
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
