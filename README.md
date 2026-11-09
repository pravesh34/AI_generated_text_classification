<!--
  Classification of Human-Written and AI-Generated Text — NIT Sikkim
  Live demo: streamlit run app/app.py  ·  Reports: reports/
-->
<div align="center">

# 🔎 Classification of Human-Written and AI-Generated Text

### NIT Sikkim · Department of Computer Science & Engineering

**Pravesh Gedon Sharma (B240019) · Ankit Rawat (B320032)** · Course Project 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://python.org) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org) [![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B.svg)](app/app.py) [![MoE](https://img.shields.io/badge/MoE-3%20experts-2563eb.svg)](#-moe-architecture) [![Accuracy 86.1% @25w](https://img.shields.io/badge/25w-86.1%25-vs-67.4%25-brightgreen.svg)](#-headline-results)

**Live web (permanent):** https://moe-text-classifier-pravesh12.vercel.app · **Live API:** `POST /api/predict` · **Streamlit demo:** `streamlit run app/app.py` · **Reports:** [`reports/`](reports/README.md)

> A Mixture-of-Experts classifier that fixes the single model's short-text collapse —
> **67.4% → 86.1% @ 25 words** and **48.6% → 98.5% on unseen chat**, while keeping full-text accuracy at ~99%.

</div>

---

## 📑 Contents

- [Overview](#-overview)
- [Headline results](#-headline-results)
- [MoE architecture](#-moe-architecture)
- [Try it](#-try-it-5-minutes)
- [Reports & papers](#-reports--papers)
- [Reproduce (MoE)](#-reproduce-the-moe-upgrade)
- [Project structure](#-project-structure)
- [Data & models](#-data--models)

---

## ✨ Overview

With LLMs everywhere, telling human-written from AI-generated text matters for academic integrity and NLP.

| Phase | What we did |
|---|---|
| **1 — Traditional ML + stylometrics** | Lexical diversity (TTR, hapax), burstiness, punctuation, readability (Flesch, Fog), word/char n-gram TF-IDF + Sentence-BERT. Classifiers: Logistic Regression, Linear SVM, Random Forest |
| **2 — Fine-tuned transformers** | DistilBERT / RoBERTa with chunking for long documents (notebooks 05) |
| **3 — Evaluation & ablation** | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, feature-family ablation |

**MoE upgrade (this repo):** a length-gated Mixture-of-Experts that fixes the baseline's short-text failure. A deterministic length gate routes to **short / long / verylong** TF-IDF+LR experts trained on HC3 short Q&A + modern-LLM essays + truncation augmentation; a type gate labels **chat / casual / essay**; thresholds adapt per length. Routed accuracy **86.1% @ 25w**, **98.96% full text**, **98.5% HC3 holdout** (vs 48.6% baseline). Code in `moe/`, artifacts in `models/moe/`, metrics in `results/moe_*.csv`.

---

## 📊 Headline results

5k held-out essays (`results/moe_routed_metrics.csv` vs `results/moe_baseline_comparison.csv`):

| Condition | Baseline | MoE routed | Δ |
|---|---:|---:|---:|
| Full text | 99.26% | 98.96% | ≈ tie |
| First 25 words | 67.42% | **86.06%** | **+18.6** |
| First 50 words | 83.28% | **93.12%** | **+9.8** |
| First 1 sentence | 59.82% | **81.64%** | **+21.8** |
| HC3 holdout (unseen chat) | 48.56% | **98.52%** | **+49.9** |
| Modern holdout (GPT-4 / Claude) | 85.52% | **95.70%** | **+10.2** |

Per-expert validation: short **97.0%** (F1 97.0%, AUC 0.996) · long **98.2%** · verylong **98.4%**. Type gate **80.5%** (chat/casual/essay). Figures with full explanations: **Figures tab in the demo app** → `results/figures/`.

---

## 🧩 MoE architecture

```text
                text
                 |
      length gate (word count, deterministic)
        ├─ ≤150w  → expert_short   (HC3 + truncated essays)
        ├─ 151-400 → expert_long   (HC3 long + modern long + base long)
        └─ >400w  → expert_verylong (base >400w + modern >400w)
                 |
      each expert = Word TF-IDF (1,2) 15k + Char TF-IDF (3,5) 25k + LR(C=1.5)
                 |
      + type gate (LR over 16 stylometric feats → chat / casual / essay)
      + length-adaptive threshold (short: 0.32→0.64) + short-text warning
```

| Source | n | Use |
|---|---|---|
| HC3 `reddit_eli5` + `open_qa` + `finance` | 80,811 answers (median 117w, 61% ≤150w) | SHORT (chat/casual) |
| `artnitolog/llm-generated-texts` (GPT-4 Turbo/Omni, Claude 3 Opus, Llama3, Command R+) | 17,997 (median 510w, 68% >400w) | LONG/VERYLONG modern |
| `data/train.csv` (20k balanced subsample) | 20,000 essays (~393w avg) | LONG/VERYLONG + truncation augmentation |

Details: [`moe/README.md`](moe/README.md)

---

## 🚀 Try it — 5 minutes

```bash
# 1. environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt "datasets>=2.16" huggingface_hub pyarrow reportlab pypdf seaborn

# 2. interactive demo (MoE + baseline inference, reports, figures)
streamlit run app/app.py

# 3. CLI inference (routes by length, labels type, warns on short text)
PYTHONPATH=. python3 moe/predict_moe.py "I think sports are great because they keep kids busy after school"
# → {"label": "Human-Written", "prob_ai": 0.0054, "expert": "short", ...}

# 4. batch / API / LangChain (optional)
python3 test/predict.py --csv data/test.csv --model lr --sample 5000 --out test/results
python3 test/api.py    # FastAPI :8001  →  /docs
```

Models are tracked in Git (~9 MB) — **no retraining needed to run inference**. Notebooks: `jupyter notebook` → `notebooks/01`…`06` in order.

---

## 📄 Reports & papers

All documents in [`reports/`](reports/README.md) (originals also in repo root):

| File | About |
|---|---|
| `report.pdf` (= `Project_Report_AI_Text_Classification.pdf`) | NIT Sikkim project report, 12 pages: baseline + MoE appendix |
| `Research_Paper_MoE_AI_Text_Detection.pdf` | 7-page paper: baseline vs MoE, ablation, cross-domain |
| `DeepSeek_Style_Research_Paper_MoE.pdf` | 9-page detailed paper: formal method (§4), per-length ablation, calibration |
| `Project_Report_MoE_Appendix.pdf` | Standalone 6-page MoE appendix |
| `Code_Workflow_and_Run_Guide.pdf` | Setup → train → test → API → UI |
| `VIDEO_SCRIPT_MoE_20min.md` | 20-minute narrated talk script |

---

## 🔁 Reproduce the MoE upgrade

Data lives in **Google Drive** (links in `data/README.md`), not in Git. Models **are** in Git.

```bash
# 1. small-text datasets → data/moe/ (126 MB, gitignored)
PYTHONPATH=. python3 moe/download_small_text.py

# 2. train 3 experts + type gate → models/moe/ (~130s)
PYTHONPATH=. python3 moe/train_moe.py

# 3. calibrate thresholds → models/moe/thresholds.json
PYTHONPATH=. python3 moe/calibrate_thresholds.py

# 4. rebuild figures + papers
PYTHONPATH=. python3 moe/make_moe_appendix.py
PYTHONPATH=. python3 make_research_paper.py
PYTHONPATH=. python3 make_deepseek_paper.py
cp *.pdf reports/
```

---

## 🗂 Project structure

```text
├── app/app.py                 # Streamlit conference demo (MoE + baseline, reports, figures)
├── moe/                       # MoE upgrade: download / type_features / train / calibrate / predict
├── src/                       # preprocessing, stylometrics, semantic, evaluate
├── notebooks/                 # 01 EDA → 06 evaluation (Jupyter)
├── models/*.pkl + models/moe/ # baseline TF-IDF+LR/SVM + 3 MoE experts + type gate (~9 MB, tracked)
├── results/                   # moe_*.csv + summary.json + figures/
├── reports/                   # all PDFs + VIDEO_SCRIPT + reports/README.md index
├── data/                      # large CSVs (gitignored → Drive) + tracked samples
├── test/                      # inference.py, predict.py, api.py, terminal_demo, LangChain tool
├── report.pdf  Research_Paper*.pdf  DeepSeek*.pdf  Project_Report_MoE_Appendix.pdf
├── requirements.txt  train_baseline.py  run_all.sh
└── README.md
```

> **Data ≠ Git. Models = Git.** Large CSVs stay out of Git (see `.gitignore`) and live on Google Drive — catalogued in [`data/README.md`](data/README.md). Models (baseline + MoE `.pkl`) **are** tracked so the repo runs without retraining.

---

## 🔗 Data & models

- Training / test CSVs: `data/train.csv` (290k) / `data/test.csv` (97k) — via Drive (`data/README.md`). Small tracked samples: `data/sample_train.csv` / `sample_test.csv`.
- MoE data: `moe/download_small_text.py` → `data/moe/hc3_small.csv`, `modern_essays.csv` or Drive zip in `data/README.md`.
- Baseline artifacts: `models/word_tfidf_vectorizer.pkl`, `char_tfidf_vectorizer.pkl`, `logistic_regression_baseline.pkl`, `linear_svm_baseline.pkl`.
- MoE artifacts: `models/moe/expert_{short,long,verylong}.pkl`, `type_gate.pkl`, `thresholds.json`, `gate_config.json`.
- Metrics: `results/baseline_ml_metrics.csv`, `results/moe_expert/routed/baseline_comparison/calibrated_metrics.csv`, `results/moe_summary.json`.

---

<div align="center">

*Built for academic discussion. Very short inputs (<25–50 words) are inherently uncertain — the MoE reports confidence and warnings.*

**NIT Sikkim · CSE · 2026** — Sharma (B240019) & Rawat (B320032)

</div>
