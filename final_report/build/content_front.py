"""Front matter: cover, abstract, keywords, abbreviations, TOC/LOF/LOT."""
from blocks import (h1, h2, p, abstract, bullets, pagebreak, toc, spacer,
                    list_table)

TITLE = "Classification of Human-Written and\nAI-Generated Text"
SUBTITLE = ("Consolidated Project Report — Baseline Pipeline, "
            "Mixture-of-Experts Upgrade, Research Papers and Reproducibility Guide")
AUTHORS = [("Pravesh Gedon Sharma", "B240019"), ("Ankit Rawat", "B320032")]
INSTITUTION = ("National Institute of Technology Sikkim\n"
               "Department of Computer Science & Engineering")
COURSE = "Course Project 2026  ·  Report compiled November 2026"


def front_matter():
    return [
        ("cover",),

        h1("Abstract"),
        abstract(
            "Large language models make fluent essay writing freely available, and the ability to "
            "separate human-authored text from machine-generated text now underpins academic "
            "integrity, journalism and content moderation. This report consolidates the complete "
            "course project: a three-phase comparative study of human vs AI text classification, "
            "the short-text failure it exposed, and the Mixture-of-Experts (MoE) system built to "
            "fix it. The Phase-1 baseline fuses word TF-IDF (1–2 grams, 25,000 features) with "
            "character TF-IDF (char_wb 3–5 grams, 50,000 features) into a 75,000-dimensional "
            "sparse representation and classifies with Multinomial Naive Bayes, Logistic "
            "Regression and a calibrated Linear SVM. On a balanced 20,000-essay experiment the "
            "best model reaches 99.53% validation accuracy, and 99.58% accuracy with ROC-AUC "
            "0.9995 on a 5,000-essay held-out test. That headline number conceals a structural "
            "failure: truncating the same held-out essays to 25 words drops accuracy to 67.4%, "
            "and one sentence to 59.8%, because human-written short text is misclassified as AI "
            "in 51% of cases (human-only accuracy 48.9%) while AI recall stays at 98.7% — a "
            "one-sided collapse caused by training data with a median length of 356 words and by "
            "cross-domain shift (48.6% accuracy on unseen HC3 chat answers, worse than chance). "
            "The project therefore downloads genuine small-text data (80,811 HC3 human/ChatGPT "
            "question-answer pairs, median 117 words), 17,997 modern-2024-LLM essays (GPT-4, "
            "Claude 3, Llama 3, Command R+), augments the original corpus with truncated copies, "
            "and trains a Mixture-of-Experts: a deterministic length gate (≤150 / 151–400 / >400 "
            "words) routes to three dedicated TF-IDF + Logistic Regression experts, a learned "
            "type gate classifies chat / casual / essay from 16 stylometric features (80.5% "
            "validation), and the decision threshold adapts linearly with length (0.32 at 25 "
            "words to 0.64 at 150 words). The routed system lifts 25-word accuracy from 67.4% to "
            "86.1% and one-sentence accuracy from 59.8% to 81.6% while holding 98.96% on full "
            "essays, and generalises to unseen domains at 98.5% (HC3 chat) and 95.7% (modern "
            "essays). The complete stack — training script, six inference front-ends "
            "(CLI, batch, REST API, LangChain tool, Streamlit application, Vercel serverless "
            "API), trained models, metrics and figures — is released for full reproducibility."),
        p("<b>Keywords:</b> AI-generated text detection; authorship attribution; mixture of "
          "experts; TF-IDF; stylometry; short-text classification; HC3; threshold calibration; "
          "academic integrity."),

        pagebreak(),
        h1("List of Abbreviations"),
        _abbrev_table(),

        pagebreak(),
        h1("Contents"),
        toc("toc"),
        pagebreak(),
        h1("List of Figures"),
        toc("lof"),
        pagebreak(),
        h1("List of Tables"),
        toc("lot"),
    ]


ABBR = [
    ("AUC", "Area Under the ROC Curve"),
    ("CSV", "Comma-Separated Values"),
    ("F1", "Harmonic mean of precision and recall"),
    ("FN / FP", "False Negative / False Positive"),
    ("Fog", "Gunning Fog readability index"),
    ("FRE", "Flesch Reading Ease"),
    ("FKGL", "Flesch–Kincaid Grade Level"),
    ("HC3", "Human-ChatGPT Comparison Corpus (Guo et al., 2023)"),
    ("LLM", "Large Language Model"),
    ("LR", "Logistic Regression"),
    ("MoE", "Mixture of Experts"),
    ("NB", "Multinomial Naive Bayes"),
    ("NFKC", "Unicode Normalization Form Compatibility Composition"),
    ("OCR", "Optical Character Recognition"),
    ("P(AI)", "Predicted probability that a text is AI-generated"),
    ("PR", "Precision / Recall"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SBERT", "Sentence-BERT (sentence embeddings)"),
    ("SVM", "Support Vector Machine (linear kernel here)"),
    ("TF–IDF", "Term Frequency–Inverse Document Frequency"),
    ("TN / TP", "True Negative / True Positive"),
    ("TTR", "Type-Token Ratio (lexical diversity)"),
    ("w", "words (unit of text length)"),
]


def _abbrev_table():
    rows = [[a, b] for a, b in ABBR]
    return list_table("Abbreviations used in this report.",
                      ["Abbreviation", "Expansion"], rows, [34, 130], (0,))
