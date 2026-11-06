# Inference / Test — Human vs AI Text Classification

All inference code for the trained Dual TF-IDF models. Uses artifacts in `../models/`:
- `word_tfidf_vectorizer.pkl` (25k features)
- `char_tfidf_vectorizer.pkl` (50k features)
- `logistic_regression_baseline.pkl` (C=1.5)
- `linear_svm_baseline.pkl` (Calibrated LinearSVC)

## Files
- `inference.py` — core engine: `load_artifacts()`, `predict_text()`, `predict_batch()`, `evaluate_csv()`, `predict_folder()`. Reusable from any script (`from test.inference import ...`).
- `predict.py` — CLI for everything (text / file / folder / CSV / interactive).
- `test_model.py` — smoke tests + evaluation driver.
- `example_texts/` — two sample `.txt` files (Human / AI).
- `results/` — outputs (`predictions.csv`, `metrics.json`).

## Quick start (from project root)
```bash
source venv/bin/activate

# 1) single text
venv/bin/python test/predict.py --text "Your essay here..." --model lr --stylometrics

# 2) from a file
venv/bin/python test/predict.py --input test/example_texts/human_sample.txt --model svm

# 3) whole folder of .txt files
venv/bin/python test/predict.py --folder test/example_texts --model lr --stylometrics

# 4) evaluate on CSV (5k sample — ~10s)
venv/bin/python test/predict.py --csv data/test.csv --model lr --sample 5000 --out test/results

# 5) full test.csv (97,447 rows — batch inference)
venv/bin/python test/predict.py --csv data/test.csv --model lr --out test/results
venv/bin/python test/predict.py --csv data/train.csv --model svm --out test/results_train_full

# 6) interactive loop
venv/bin/python test/predict.py --interactive --model lr --stylometrics

# smoke tests
venv/bin/python test/test_model.py          # 2k-row eval
venv/bin/python test/test_model.py --full   # full 97k eval
```

## Programmatic use
```python
from test.inference import load_artifacts, predict_text, evaluate_csv
art = load_artifacts()  # from ../models/
r = predict_text("Some text...", art, model="lr", show_stylometrics=True)
print(r["predicted_label"], r["probability_ai"])  # AI-Generated / Human-Written, P(AI)

metrics, df = evaluate_csv("data/test.csv", art, model="lr", sample_n=5000)
print(metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
```

## Outputs
- `predictions.csv`: `text, generated, prob_ai, pred, correct`
- `metrics.json`: accuracy, precision, recall, f1, roc_auc, confusion matrix, classification report

## Notes
- Batch inference processes 1,000 rows at a time to avoid OOM on the full 97k / 290k files.
- Predictions are thresholded at P(AI) ≥ 0.5.
- Stylometrics come from `src/stylometrics.py` (TTR, burstiness, Flesch ease/grade, etc.).
