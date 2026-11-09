"""
Text preprocessing and cleaning utilities for Human vs. AI text classification.
Matches Section 4.1 of the NIT Sikkim Project Report.
"""

import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated spaces, tabs, and newlines into single spaces.
    Preserves text structure without clutter.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_text_for_tfidf(text: str) -> str:
    """
    Normalization strictly intended for TF-IDF word representations:
    - Unicode normalization (NFKC)
    - Lowercasing
    - Preserves internal word boundaries and punctuation where appropriate
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_whitespace(text)
    return text.lower()


def clean_text_for_stylometrics(text: str) -> str:
    """
    Cleans raw text for stylometric extraction while preserving:
    - Original casing (essential for uppercase/capital ratio)
    - Punctuation marks (semicolons, colons, em-dashes, commas, exclamation, question marks)
    - Sentence delimiters (. ! ?)
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return normalize_whitespace(text)
