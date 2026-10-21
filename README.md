# Classification of Human-Written and AI-Generated Text

> **National Institute of Technology Sikkim**  
> Department of Computer Science & Engineering  
> AI & ML Project (2026)  
> Authors: Pravesh Gedon Sharma (B240019) & Ankit Rawat (B320032)

---

## 📌 Project Overview
With the rapid proliferation of Large Language Models (LLMs) such as ChatGPT, Claude, and LLaMA, distinguishing human-written text from machine-generated text has become essential for academic integrity, misinformation prevention, and digital forensics.

This project implements a comprehensive **three-phase approach**:
1. **Phase 1: Traditional Machine Learning & Feature Engineering**
   - Linguistic & Stylometric features (Lexical diversity, Sentence burstiness, Punctuation rates, Readability metrics FRE/FKGL/Fog, POS distributions)
   - N-gram representations (Word TF-IDF & Character n-grams with `char_wb`)
   - Semantic representations (Sentence-BERT embeddings & consecutive sentence cosine similarity statistics)
   - Baseline models: Logistic Regression, Linear SVM, Random Forest, and Early/Late Feature-Fusion models
2. **Phase 2: Fine-Tuned Transformer Models**
   - Transfer learning using encoder-based models (DistilBERT, RoBERTa)
   - Sequence classification with chunking / sliding window
3. **Phase 3: Evaluation, Ablation Study & Generalization Analysis**
   - Common benchmark metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
   - Ablation across feature families (Lexical vs Stylistic vs Semantic vs Hybrid)
   - Generalization test on unseen generators
   - Interactive Streamlit web application

---

## 🗂️ Repository Structure

```
├── data/
│   ├── raw/                  # Raw dataset files (train.csv, test.csv)
│   └── processed/            # Cleaned, tokenized, or downsampled sample splits
├── notebooks/                # Sequential, pedagogical Jupyter Notebooks
│   ├── 01_eda_and_data_exploration.ipynb
│   ├── 02_stylometric_and_linguistic_features.ipynb
│   ├── 03_tfidf_and_baseline_models.ipynb
│   ├── 04_semantic_features_and_fusion.ipynb
│   ├── 05_transformer_distilbert_roberta.ipynb
│   └── 06_evaluation_ablation_and_comparison.ipynb
├── src/                      # Modular Python scripts for reusable logic
│   ├── __init__.py
│   ├── preprocessing.py      # Text cleaning and normalization
│   ├── stylometrics.py       # Stylometric and readability metrics
│   ├── semantic.py           # SBERT embedding & similarity extractors
│   └── evaluate.py           # Metrics calculation and plotting utilities
├── models/                   # Serialized model checkpoints and vectorizers
├── results/                  # Confusion matrices, ROC curves, CSV benchmark tables
├── app/                      # Streamlit interactive application
│   └── app.py
├── report.pdf                # NIT Sikkim Project Report
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 Setup Instructions

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset
Ensure `train.csv` and `test.csv` are placed in `data/raw/`.
The dataset structure contains:
- `text`: String passage to classify.
- `generated`: Binary target (`0` for Human-written, `1` for AI-generated).

### 3. Running Notebooks
Launch Jupyter Lab / Notebook:
```bash
jupyter lab
```
Navigate to `notebooks/` and run the numbered notebooks sequentially from `01` to `06`.
