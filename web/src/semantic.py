"""
Semantic Feature Extraction using Sentence-BERT embeddings.
Implements Section 4.4 of the NIT Sikkim Project Report:
1. Document dense representation via mean sentence embeddings
2. Consecutive sentence cosine similarity progression (mean, std, min, max)
3. Sentence-to-centroid cosine similarity (mean, std)
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np


def split_sentences_semantic(text: str) -> List[str]:
    """
    Splits text into meaningful sentences for embedding extraction.
    """
    if not isinstance(text, str):
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]
    return sents if sents else [text.strip()]


def extract_semantic_features(
    sentences: List[str],
    sbert_model: Any,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Computes SBERT embeddings and similarity transition statistics.
    As described in Section 4.4 of the report:
    - High and uniform consecutive similarity indicates smooth/repetitive AI flow.
    - Lower or variable similarity captures human conceptual drift.
    """
    if not sentences:
        return {
            "doc_embedding": np.zeros(384, dtype=np.float32),
            "consec_sim_mean": 0.0,
            "consec_sim_std": 0.0,
            "consec_sim_min": 0.0,
            "centroid_sim_mean": 0.0,
            "centroid_sim_std": 0.0
        }

    # Encode sentences
    embeddings = sbert_model.encode(sentences, normalize_embeddings=normalize, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Document centroid embedding (mean pooled)
    doc_embedding = np.mean(embeddings, axis=0)
    norm_doc = np.linalg.norm(doc_embedding)
    if norm_doc > 1e-8:
        centroid_norm = doc_embedding / norm_doc
    else:
        centroid_norm = doc_embedding

    # Consecutive sentence cosine similarity
    if len(embeddings) > 1:
        consec_sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        consec_mean = float(np.mean(consec_sims))
        consec_std = float(np.std(consec_sims))
        consec_min = float(np.min(consec_sims))
    else:
        consec_mean = 1.0
        consec_std = 0.0
        consec_min = 1.0

    # Centroid similarity
    centroid_sims = np.dot(embeddings, centroid_norm)
    cent_mean = float(np.mean(centroid_sims))
    cent_std = float(np.std(centroid_sims))

    return {
        "doc_embedding": doc_embedding,
        "consec_sim_mean": consec_mean,
        "consec_sim_std": consec_std,
        "consec_sim_min": consec_min,
        "centroid_sim_mean": cent_mean,
        "centroid_sim_std": cent_std
    }
