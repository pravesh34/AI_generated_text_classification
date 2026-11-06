#!/usr/bin/env python3
"""
test/predict.py — CLI entry-point for Human vs AI Text Classification inference.

Usage examples (from project root):
  # single text
  venv/bin/python test/predict.py --text "Your essay here..." --model lr --stylometrics

  # from a file
  venv/bin/python test/predict.py --input test/example_texts/human_sample.txt --model svm

  # batch CSV -> evaluation + predictions.csv
  venv/bin/python test/predict.py --csv data/test.csv --model lr --sample 5000 --out test/results

  # whole folder of .txt files
  venv/bin/python test/predict.py --folder test/example_texts --model lr --stylometrics

  # full test.csv (97k rows, ~30-60s)
  venv/bin/python test/predict.py --csv data/test.csv --model svm --out test/results

Outputs: predictions.csv + metrics.json in --out dir; prints full metrics to terminal.
"""
import argparse
import sys
import json
from pathlib import Path

# allow `python test/predict.py` and `python -m test.predict`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import load_artifacts, predict_text, predict_folder, evaluate_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="AI vs Human text classification inference")
    p.add_argument("--text", type=str, help="Raw text string to classify")
    p.add_argument("--input", type=str, help="Path to a .txt / .md file")
    p.add_argument("--folder", type=str, help="Folder of .txt files to classify")
    p.add_argument("--csv", type=str, help="CSV with columns text,generated for full evaluation")
    p.add_argument("--model", choices=["lr", "svm"], default="lr", help="lr=Logistic Regression, svm=Linear SVM")
    p.add_argument("--sample", type=int, default=None, help="Random sample N rows from --csv (default: all)")
    p.add_argument("--out", type=str, default="test/results", help="Output dir for predictions.csv / metrics.json")
    p.add_argument("--stylometrics", action="store_true", help="Show stylometric features")
    p.add_argument("--interactive", action="store_true", help="Interactive loop: type text, Ctrl+C to quit")
    args = p.parse_args()

    if not any([args.text, args.input, args.folder, args.csv, args.interactive]):
        p.print_help()
        sys.exit(1)

    print("Loading models from models/ ...")
    artifacts = load_artifacts()
    print(f"Loaded: word_vec ({len(artifacts['word_vec'].vocabulary_):,} features), "
          f"char_vec ({len(artifacts['char_vec'].vocabulary_):,} features), "
          f"model={'Logistic Regression' if args.model=='lr' else 'Linear SVM'}")
    print("=" * 70)

    # --- single text ---
    if args.text:
        r = predict_text(args.text, artifacts, model=args.model, show_stylometrics=args.stylometrics)
        print(f"Predicted : {r['predicted_label']}")
        print(f"P(AI)     : {r['probability_ai']:.4f}")
        print(f"P(Human)  : {r['probability_human']:.4f}")
        print(f"Confidence: {r['confidence']*100:.1f}%")
        if args.stylometrics and "stylometrics" in r:
            print("\nStylometrics:")
            for k, v in r["stylometrics"].items():
                print(f"  {k:25} {v:.3f}")
        return

    # --- single file ---
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
        r = predict_text(text, artifacts, model=args.model, show_stylometrics=args.stylometrics)
        print(f"File     : {args.input}")
        print(f"Predicted: {r['predicted_label']}")
        print(f"P(AI)    : {r['probability_ai']:.4f}  P(Human): {r['probability_human']:.4f}  Confidence: {r['confidence']*100:.1f}%")
        if args.stylometrics and "stylometrics" in r:
            print("\nStylometrics:")
            for k, v in r["stylometrics"].items():
                print(f"  {k:25} {v:.3f}")
        return

    # --- folder batch ---
    if args.folder:
        results = predict_folder(args.folder, artifacts, model=args.model)
        for r in results:
            print(f"{r['file']:30} -> {r['predicted_label']:15} P(AI)={r['probability_ai']:.3f} conf={r['confidence']*100:.1f}%")
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame(results).to_csv(Path(args.out) / "folder_predictions.csv", index=False)
            print(f"\nSaved -> {args.out}/folder_predictions.csv ({len(results)} files)")
        return

    # --- CSV evaluation (the "whole inference work") ---
    if args.csv:
        print(f"Evaluating: {args.csv} (model={args.model}, sample={args.sample or 'ALL'})")
        metrics, df = evaluate_csv(args.csv, artifacts, model=args.model, sample_n=args.sample)
        print(f"\n{'Accuracy':10} : {metrics['accuracy']:.4f}")
        print(f"{'Precision':10} : {metrics['precision']:.4f}")
        print(f"{'Recall':10} : {metrics['recall']:.4f}")
        print(f"{'F1':10} : {metrics['f1']:.4f}")
        print(f"{'ROC-AUC':10} : {metrics['roc_auc']:.4f}")
        print(f"{'N':10} : {metrics['n_samples']}")
        print(f"Confusion matrix [[TN FP][FN TP]]: {metrics['confusion_matrix']}")
        print("\nClassification report:")
        print(json.dumps(metrics["classification_report"], indent=2))
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "predictions.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved -> {out}/predictions.csv ({len(df)} rows) and {out}/metrics.json")
        return

    # --- interactive ---
    if args.interactive:
        print("Interactive mode — paste text, press Enter twice to classify, Ctrl+C to quit.")
        while True:
            print("\n--- paste text (end with a blank line) ---")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            text = "\n".join(lines)
            if not text.strip():
                continue
            r = predict_text(text, artifacts, model=args.model, show_stylometrics=args.stylometrics)
            print(f"=> {r['predicted_label']}  P(AI)={r['probability_ai']:.4f}  conf={r['confidence']*100:.1f}%")


if __name__ == "__main__":
    main()
