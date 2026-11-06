"""
Detailed Phase 1 Training: Dual TF-IDF + LR / Linear SVM / NB
Project: Classification of Human-Written and AI-Generated Text
Follows: notebooks/03_tfidf_and_baseline_models.ipynb
Data: data/train.csv (290k), data/test.csv (97k)
"""
import sys, os, time, gc
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.preprocessing import clean_text_for_tfidf

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else 0.0,
    }

t0=time.time()
log("STEP 1/11: Load full data/train.csv and data/test.csv")
train_path="data/train.csv"; test_path="data/test.csv"
# only read needed columns, fast
df_raw=pd.read_csv(train_path, usecols=["text","generated"])
log(f"  train.csv shape={df_raw.shape} cols={list(df_raw.columns)}")
log(f"  class counts: {df_raw['generated'].value_counts().to_dict()}")
log(f"  text len chars mean={df_raw['text'].astype(str).str.len().mean():.0f} median={df_raw['text'].astype(str).str.len().median():.0f} max={df_raw['text'].astype(str).str.len().max()}")
# don't load full test yet for EDA, just count
log(f"  (test.csv will be loaded later for held-out evaluation)")

log("STEP 2/11: Balanced subsample 10k per class (20k total) as in notebook 03")
SAMPLE_SIZE=10000
# pandas 3.x drops the group key with groupby(...).apply(), so sample per-class explicitly
df0 = df_raw[df_raw["generated"] == 0].sample(SAMPLE_SIZE, random_state=42)
df1 = df_raw[df_raw["generated"] == 1].sample(SAMPLE_SIZE, random_state=42)
df_sub = pd.concat([df0, df1]).sample(frac=1.0, random_state=42).reset_index(drop=True)
log(f"  subsample shape={df_sub.shape} counts={df_sub['generated'].value_counts().to_dict()}")
del df_raw; gc.collect()
log("  -> Why subsample? Notebook 03 uses 20k for speed; full 290k TF-IDF (75k features) would be ~14x slower/memory-heavy. Keeps balanced classes.")

log("STEP 3/11: Stratified 80/20 train/val split (keeps 50/50 class ratio)")
# pandas 3.x uses Arrow-backed strings; convert to plain numpy object arrays for sklearn
X_all = np.array(df_sub["text"].tolist(), dtype=object)
y_all = np.array(df_sub["generated"].tolist(), dtype=int)
train_texts, val_texts, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.20, random_state=42, stratify=y_all
)
log(f"  train={len(train_texts):,} val={len(val_texts):,} y_train AI%={y_train.mean():.2f} y_val AI%={y_val.mean():.2f}")
del df_sub; gc.collect()

log("STEP 4/11: Preprocessing demo (clean_text_for_tfidf)")
for i in range(2):
    raw=str(train_texts[i])[:120].replace("\n"," ")
    cln=clean_text_for_tfidf(str(train_texts[i]))[:120]
    log(f"  raw[{i}]: {raw!r}")
    log(f"  cln[{i}]: {cln!r}  <- NFKC + whitespace normalize + lowercase")

log("STEP 5/11: Word TF-IDF (1,2) max_features=25000 min_df=2 sublinear_tf")
t=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2), max_features=25000, min_df=2, sublinear_tf=True, preprocessor=clean_text_for_tfidf)
X_train_word=word_vec.fit_transform(train_texts)
X_val_word=word_vec.transform(val_texts)
log(f"  word vocab={len(word_vec.vocabulary_):,} X_train_word={X_train_word.shape} ({time.time()-t:.1f}s)")
log(f"  example features: {word_vec.get_feature_names_out()[:10].tolist()}")

log("STEP 6/11: Char TF-IDF char_wb (3,5) max_features=50000 min_df=3 sublinear_tf")
t=time.time()
char_vec=TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=50000, min_df=3, sublinear_tf=True)
X_train_char=char_vec.fit_transform(train_texts)
X_val_char=char_vec.transform(val_texts)
log(f"  char vocab={len(char_vec.vocabulary_):,} X_train_char={X_train_char.shape} ({time.time()-t:.1f}s)")
del train_texts, val_texts; gc.collect()

log("STEP 7/11: Fuse: hstack(word, char) -> combined sparse matrix")
X_train_tfidf=hstack([X_train_word, X_train_char]).tocsr()
X_val_tfidf=hstack([X_val_word, X_val_char]).tocsr()
log(f"  X_train_tfidf={X_train_tfidf.shape} nnz={X_train_tfidf.nnz:,}")
log(f"  X_val_tfidf={X_val_tfidf.shape} nnz={X_val_tfidf.nnz:,}")
del X_train_word, X_train_char, X_val_word, X_val_char; gc.collect()

results=[]
def train_and_eval(name, model):
    t=time.time()
    log(f"STEP 8/11: Train {name} ...")
    model.fit(X_train_tfidf, y_train)
    log(f"  fit done ({time.time()-t:.1f}s) -> evaluating on val (4k)...")
    y_pred=model.predict(X_val_tfidf); y_prob=model.predict_proba(X_val_tfidf)[:,1]
    m=metrics(y_val, y_pred, y_prob)
    log(f"  {name}: {m}")
    log(f"  confusion matrix [[TN FP][FN TP]]=\n{confusion_matrix(y_val, y_pred)}")
    log(classification_report(y_val, y_pred, target_names=["Human","AI"]))
    return {"Model": name, **m}, model, y_prob

r1, nb_model, p1 = train_and_eval("Multinomial NB", MultinomialNB(alpha=0.1))
r2, lr_model, p2 = train_and_eval("Logistic Regression", LogisticRegression(C=1.5, max_iter=1000, solver="lbfgs", random_state=42))
r3, svm_model, p3 = train_and_eval("Linear SVM (Calibrated)", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3))
results=[r1,r2,r3]

os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
import csv
pd.DataFrame(results).to_csv("results/baseline_ml_metrics.csv", index=False)
log("STEP 9/11: Save metrics -> results/baseline_ml_metrics.csv")
log(pd.DataFrame(results).to_string(index=False))

log("STEP 10/11: Held-out evaluation on data/test.csv (random 5000 sample)")
t=time.time()
df_test=pd.read_csv(test_path, usecols=["text","generated"])
log(f"  test.csv shape={df_test.shape} counts={df_test['generated'].value_counts().to_dict()}")
df_test_s=df_test.sample(5000, random_state=42).reset_index(drop=True)
X_test_txt = np.array(df_test_s["text"].tolist(), dtype=object)
y_test = np.array(df_test_s["generated"].tolist(), dtype=int)
X_test_word=word_vec.transform(X_test_txt)
X_test_char=char_vec.transform(X_test_txt)
X_test_tfidf=hstack([X_test_word, X_test_char]).tocsr()
for name, model in [("Logistic Regression", lr_model), ("Linear SVM", svm_model)]:
    pred=model.predict(X_test_tfidf); prob=model.predict_proba(X_test_tfidf)[:,1]
    m=metrics(y_test, pred, prob)
    log(f"  TEST {name}: {m} ({time.time()-t:.1f}s)")
del df_test, df_test_s, X_test_word, X_test_char, X_test_tfidf; gc.collect()

log("STEP 11/11: Save artifacts (for Streamlit app/app.py)")
joblib.dump(word_vec, "models/word_tfidf_vectorizer.pkl")
joblib.dump(char_vec, "models/char_tfidf_vectorizer.pkl")
joblib.dump(lr_model, "models/logistic_regression_baseline.pkl")
joblib.dump(svm_model, "models/linear_svm_baseline.pkl")
log("  saved: word_tfidf_vectorizer.pkl, char_tfidf_vectorizer.pkl, logistic_regression_baseline.pkl, linear_svm_baseline.pkl")
log(f"TOTAL TIME: {time.time()-t0:.1f}s")
