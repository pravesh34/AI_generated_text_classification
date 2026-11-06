#!/usr/bin/env python3
"""
test/evaluate_examples.py — Load model, read ALL example .txt files, evaluate, and report.
Ground truth is encoded in filename: prefix `human_` -> 0, `ai_` -> 1.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_artifacts, predict_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def infer_label(p: Path):
    n = p.name.lower()
    if n.startswith("human_") or "_human" in n:
        return 0
    if n.startswith("ai_") or "_ai" in n:
        return 1
    return None

def main():
    folder = ROOT / "test" / "example_texts"
    files = sorted(list(folder.glob("*.txt")))
    print(f"Found {len(files)} example files in {folder}\n")
    art = load_artifacts()
    rows = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        true = infer_label(p)
        for model in ["lr", "svm"]:
            r = predict_text(text, art, model=model, show_stylometrics=False)
            pred = 1 if r["predicted_label"] == "AI-Generated" else 0
            print(f"{p.name:40} true={true} model={model} pred={pred} P(AI)={r['probability_ai']:.4f} conf={r['confidence']*100:.1f}% {'✓' if pred==true else '✗'}")
        # store LR for aggregate metrics
        r_lr = predict_text(text, art, model="lr")
        pred_lr = 1 if r_lr["predicted_label"]=="AI-Generated" else 0
        rows.append({"file": p.name, "true": true, "pred_lr": pred_lr, "p_ai": r_lr["probability_ai"], "chars": len(text), "words": len(text.split())})
    # aggregate
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred_lr"] for r in rows]
    y_prob = [r["p_ai"] for r in rows]
    if len(set(y_true))>1:
        print(f"\n=== AGGREGATE (Logistic Regression, n={len(rows)}) ===")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
        try:
            print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
        except: pass
        print(f"Confusion matrix [[TN FP][FN TP]]: {confusion_matrix(y_true, y_pred).tolist()}")
        print(classification_report(y_true, y_pred, target_names=["Human","AI"]))
    # save CSV
    import pandas as pd
    out = ROOT / "test" / "results"
    out.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(out / "example_folder_evaluation.csv", index=False)
    print(f"\nSaved -> test/results/example_folder_evaluation.csv")

if __name__ == "__main__":
    main()
