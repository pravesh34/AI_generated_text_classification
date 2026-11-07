"""
Shared helpers for the MoE upgrade: length buckets + text-type gate features.

Buckets (from the short-text ablation: accuracy cliff at ~100-150 words,
dataset median ~363 words):
  short     : <= 150 words
  long      : 151 - 400 words
  verylong  : > 400 words

Text types (router labels):
  chat   : short informal Q&A (HC3 open_qa) - first/second person, questions
  casual : informal-medium (HC3 reddit_eli5/finance, truncated essays)
  essay  : long formal academic (full essays)

The type gate is a Logistic Regression over fast stylometric features, so
inference can label any new text as chat/casual/essay without a transformer.
"""
import re
import numpy as np

SHORT_MAX = 150
LONG_MAX = 400

_CONTRACTION_RE = re.compile(
    r"\b(i'm|i've|i'll|you're|you've|you'll|he's|she's|it's|that's|what's|don't|doesn't|didn't|won't|can't|cannot|n't|'ll|'re|'ve)\b",
    re.I,
)
_INFORMAL_RE = re.compile(
    r"\b(lol|haha|yeah|yep|gonna|wanna|kinda|sorta|btw|tbh|imo|btw|ain't|yea|nah|yo|hey)\b",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|we|our|us)\b", re.I)
_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours)\b", re.I)


def word_count(text: str) -> int:
    return len(str(text).split()) if text else 0


def length_bucket(n_words: int) -> str:
    if n_words <= SHORT_MAX:
        return "short"
    if n_words <= LONG_MAX:
        return "long"
    return "verylong"


def truncate_words(text: str, n: int) -> str:
    w = str(text).split()
    return " ".join(w[:n])


def random_window(text: str, n: int, rng=None) -> str:
    w = str(text).split()
    if len(w) <= n:
        return str(text)
    if rng is None:
        rng = np.random.default_rng()
    s = int(rng.integers(0, len(w) - n + 1))
    return " ".join(w[s : s + n])


TYPE_FEATURES = [
    "log_words",
    "sentences",
    "mean_sentence_length",
    "ttr",
    "hapax_ratio",
    "avg_word_length",
    "contraction_rate",
    "informal_rate",
    "first_person_rate",
    "second_person_rate",
    "question_rate",
    "exclamation_rate",
    "capital_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "burstiness",
]


def extract_type_features(text: str) -> dict:
    """Fast stylometric gate features for chat/casual/essay routing."""
    from src.stylometrics import extract_stylometric_features

    s = extract_stylometric_features(text)
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", str(text))
    n = max(1, len(words))
    lower = str(text).lower()
    feats = {
        "log_words": float(np.log1p(len(words))),
        "sentences": s["num_sentences"],
        "mean_sentence_length": s["mean_sentence_length"],
        "ttr": s["ttr"],
        "hapax_ratio": s["hapax_ratio"],
        "avg_word_length": s["avg_word_length"],
        "contraction_rate": len(_CONTRACTION_RE.findall(lower)) / n,
        "informal_rate": len(_INFORMAL_RE.findall(lower)) / n,
        "first_person_rate": len(_FIRST_PERSON_RE.findall(lower)) / n,
        "second_person_rate": len(_SECOND_PERSON_RE.findall(lower)) / n,
        "question_rate": s["question_rate"],
        "exclamation_rate": s["exclamation_rate"],
        "capital_ratio": s["capital_ratio"],
        "flesch_reading_ease": s["flesch_reading_ease"],
        "flesch_kincaid_grade": s["flesch_kincaid_grade"],
        "burstiness": s["burstiness"],
    }
    return feats


def type_feature_matrix(texts) -> np.ndarray:
    rows = [extract_type_features(t) for t in texts]
    return np.array([[r[k] for k in TYPE_FEATURES] for r in rows], dtype=float)


def heuristic_type(text: str) -> str:
    """Fallback label when no source label exists (used to pseudo-label data)."""
    n = word_count(text)
    f = extract_type_features(text)
    if n <= 80 and (f["mean_sentence_length"] < 14 or f["question_rate"] > 0.5):
        return "chat"
    if n <= 150:
        return "casual" if f["flesch_kincaid_grade"] < 9 else "chat"
    if n > 250 and f["flesch_kincaid_grade"] > 8:
        return "essay"
    return "casual" if n <= 250 else "essay"
