"""
Stylometric and Linguistic Feature Extraction module.
Implements Section 4.2 of the NIT Sikkim Project Report:
1. Lexical diversity (TTR, hapax ratio, avg word length)
2. Sentence structure (mean, std, min, max, burstiness sigma/mu)
3. Punctuation rates & capital letter ratio
4. Readability indices (Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog)
5. Syntactic/POS distributions
"""

import math
import re
import string
from typing import Dict, List, Any
import numpy as np


def count_syllables_word(word: str) -> int:
    """
    Estimates syllable count using standard heuristic rules.
    """
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    matches = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(matches))


def flesch_reading_ease(words_count: int, sentences_count: int, syllables_count: int) -> float:
    """
    FRE = 206.835 - 1.015 * (N / S) - 84.6 * (syllables / N)
    Higher score = easier text to read.
    """
    if words_count == 0 or sentences_count == 0:
        return 0.0
    asl = words_count / sentences_count
    asw = syllables_count / words_count
    return float(206.835 - (1.015 * asl) - (84.6 * asw))


def flesch_kincaid_grade_level(words_count: int, sentences_count: int, syllables_count: int) -> float:
    """
    FKGL = 0.39 * (N / S) + 11.8 * (syllables / N) - 15.59
    Corresponds to US school grade reading level.
    """
    if words_count == 0 or sentences_count == 0:
        return 0.0
    asl = words_count / sentences_count
    asw = syllables_count / words_count
    return float((0.39 * asl) + (11.8 * asw) - 15.59)


def gunning_fog_index(words_count: int, sentences_count: int, complex_words_count: int) -> float:
    """
    Gunning Fog = 0.4 * [ (N / S) + 100 * (complex_words / N) ]
    Complex words: words with >= 3 syllables.
    """
    if words_count == 0 or sentences_count == 0:
        return 0.0
    asl = words_count / sentences_count
    pcw = (complex_words_count / words_count) * 100.0
    return float(0.4 * (asl + pcw))


def extract_stylometric_features(text: str, ttr_window: int = 200) -> Dict[str, float]:
    """
    Extracts comprehensive stylometric feature dictionary from raw document.
    """
    if not isinstance(text, str) or not text.strip():
        text = ""

    total_chars = len(text)
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text)
    num_words = len(words)

    # Sentences split by sentence terminating punctuation (. ! ?)
    raw_sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    num_sentences = max(1, len(raw_sents))

    # Syllable statistics
    syllable_counts = [count_syllables_word(w) for w in words]
    total_syllables = sum(syllable_counts)
    complex_words = sum(1 for c in syllable_counts if c >= 3)

    # 1. Lexical Diversity
    sample_words = [w.lower() for w in words[:ttr_window]] if num_words >= ttr_window else [w.lower() for w in words]
    ttr = (len(set(sample_words)) / len(sample_words)) if sample_words else 0.0
    
    word_freq = {}
    for w in [w.lower() for w in words]:
        word_freq[w] = word_freq.get(w, 0) + 1
    hapax_legomena = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = (hapax_legomena / num_words) if num_words > 0 else 0.0

    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = float(np.mean(word_lengths))

    # 2. Sentence Structure & Burstiness
    sent_lengths = [len(re.findall(r"\b[A-Za-z0-9'-]+\b", s)) for s in raw_sents] if raw_sents else [0]
    mean_sent_len = float(np.mean(sent_lengths))
    std_sent_len = float(np.std(sent_lengths))
    min_sent_len = float(np.min(sent_lengths))
    max_sent_len = float(np.max(sent_lengths))
    burstiness = (std_sent_len / mean_sent_len) if mean_sent_len > 0 else 0.0

    # Paragraph structure
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    num_paras = max(1, len(paragraphs))
    words_per_para = num_words / num_paras

    # 3. Punctuation rates per 100 characters & capitalization
    base_chars = max(1, total_chars)
    comma_rate = (text.count(",") / base_chars) * 100.0
    semicolon_rate = (text.count(";") / base_chars) * 100.0
    colon_rate = (text.count(":") / base_chars) * 100.0
    dash_rate = ((text.count("-") + text.count("—") + text.count("–")) / base_chars) * 100.0
    quote_rate = ((text.count('"') + text.count("'")) / base_chars) * 100.0
    paren_rate = ((text.count("(") + text.count(")")) / base_chars) * 100.0
    exclamation_rate = (text.count("!") / base_chars) * 100.0
    question_rate = (text.count("?") / base_chars) * 100.0
    
    alpha_chars = sum(1 for c in text if c.isalpha())
    capital_chars = sum(1 for c in text if c.isupper())
    capital_ratio = (capital_chars / alpha_chars) if alpha_chars > 0 else 0.0

    # 4. Readability Indices
    fre = flesch_reading_ease(num_words, num_sentences, total_syllables)
    fkgl = flesch_kincaid_grade_level(num_words, num_sentences, total_syllables)
    fog = gunning_fog_index(num_words, num_sentences, complex_words)
    syllables_per_word = (total_syllables / num_words) if num_words > 0 else 0.0

    return {
        "num_words": float(num_words),
        "num_sentences": float(num_sentences),
        "avg_word_length": avg_word_length,
        "ttr": float(ttr),
        "hapax_ratio": float(hapax_ratio),
        "mean_sentence_length": mean_sent_len,
        "std_sentence_length": std_sent_len,
        "min_sentence_length": min_sent_len,
        "max_sentence_length": max_sent_len,
        "burstiness": float(burstiness),
        "words_per_paragraph": float(words_per_para),
        "comma_rate": float(comma_rate),
        "semicolon_rate": float(semicolon_rate),
        "colon_rate": float(colon_rate),
        "dash_rate": float(dash_rate),
        "quote_rate": float(quote_rate),
        "paren_rate": float(paren_rate),
        "exclamation_rate": float(exclamation_rate),
        "question_rate": float(question_rate),
        "capital_ratio": float(capital_ratio),
        "flesch_reading_ease": float(fre),
        "flesch_kincaid_grade": float(fkgl),
        "gunning_fog": float(fog),
        "syllables_per_word": float(syllables_per_word)
    }
