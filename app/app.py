"""
Streamlit Web Application: Human vs. AI Text Classification
Department of Computer Science & Engineering, NIT Sikkim
Authors: Pravesh Gedon Sharma (B240019) & Ankit Rawat (B320032)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import joblib
from scipy.sparse import hstack

from src.preprocessing import clean_text_for_tfidf
from src.stylometrics import extract_stylometric_features

st.set_page_config(
    page_title="AI vs Human Text Classification",
    layout="wide"
)

st.title("Classification of Human-Written and AI-Generated Text")
st.markdown("""
**National Institute of Technology Sikkim** — Department of Computer Science & Engineering  
*Authors: Pravesh Gedon Sharma (B240019) & Ankit Rawat (B320032)*
""")

st.write("---")

# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression (Dual TF-IDF)", "Linear SVM (Calibrated)"]
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    word_vec_path = os.path.join(base_dir, "word_tfidf_vectorizer.pkl")
    char_vec_path = os.path.join(base_dir, "char_tfidf_vectorizer.pkl")
    lr_path = os.path.join(base_dir, "logistic_regression_baseline.pkl")
    svm_path = os.path.join(base_dir, "linear_svm_baseline.pkl")

    artifacts = {}
    if os.path.exists(word_vec_path) and os.path.exists(char_vec_path):
        artifacts["word_vec"] = joblib.load(word_vec_path)
        artifacts["char_vec"] = joblib.load(char_vec_path)
    if os.path.exists(lr_path):
        artifacts["lr"] = joblib.load(lr_path)
    if os.path.exists(svm_path):
        artifacts["svm"] = joblib.load(svm_path)
    return artifacts

artifacts = load_artifacts()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Input Passage")
    user_text = st.text_area(
        "Enter text to classify:",
        height=280,
        placeholder="Paste text passage here..."
    )
    classify_btn = st.button("Classify Text", type="primary", use_container_width=True)

with col2:
    st.subheader("Classification Results")
    if classify_btn:
        if not user_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            # Stylometric inspection
            stylo = extract_stylometric_features(user_text)

            # Check if models are trained and saved
            if "word_vec" in artifacts and "char_vec" in artifacts:
                X_word = artifacts["word_vec"].transform([user_text])
                X_char = artifacts["char_vec"].transform([user_text])
                X_feat = hstack([X_word, X_char]).tocsr()

                model = artifacts["lr"] if "Logistic" in model_choice else artifacts.get("svm", artifacts["lr"])
                prob_ai = float(model.predict_proba(X_feat)[0, 1])
                prob_human = 1.0 - prob_ai

                if prob_ai >= 0.5:
                    st.error(f"### Predicted Class: **AI-Generated**\n**Confidence: {prob_ai * 100:.2f}%**")
                else:
                    st.success(f"### Predicted Class: **Human-Written**\n**Confidence: {prob_human * 100:.2f}%**")

                st.progress(prob_ai)
                st.caption(f"AI Probability: {prob_ai:.4f} | Human Probability: {prob_human:.4f}")
            else:
                st.info("Trained model files not found in models/ directory. Run notebook 03 to generate baseline weights.")

            # Stylometric Breakdown Cards
            st.write("---")
            st.markdown("#### Stylometric Summary")
            m1, m2 = st.columns(2)
            m1.metric("Word Count", f"{int(stylo['num_words'])}")
            m1.metric("Sentence Burstiness (sigma / mu)", f"{stylo['burstiness']:.2f}")
            m1.metric("Flesch Reading Ease", f"{stylo['flesch_reading_ease']:.1f}")

            m2.metric("Lexical Diversity (TTR)", f"{stylo['ttr']:.2f}")
            m2.metric("Avg Sentence Length", f"{stylo['mean_sentence_length']:.1f}")
            m2.metric("Flesch-Kincaid Grade", f"{stylo['flesch_kincaid_grade']:.1f}")
