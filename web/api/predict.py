"""
Vercel serverless MoE inference endpoint.
GET  /api/predict  -> {"status": "ok", ...}
POST /api/predict  {"text": "..."} -> MoE router result JSON

Models are bundled at api/models/ (copied from models/) and are also
available in the cloud via GitHub raw. If a local file is missing, it
is fetched from MODEL_BASE_URL (default: GitHub raw) into /tmp.

Runs on Vercel Python runtime. Deps: web/requirements.txt
(sklearn pinned to the version the .pkl files were trained with).
"""
import json
import os
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Pickled experts reference src.preprocessing (TF-IDF tokenizer). Bundled at
# web/src + web/moe — ensure project root is importable on Vercel.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from scipy.sparse import hstack

HERE = Path(__file__).resolve().parent
LOCAL_MODELS = HERE / "models"  # api/models/ — bundled with deployment
CLOUD_BASE = os.environ.get(
    "MODEL_BASE_URL",
    "https://raw.githubusercontent.com/pravesh34/AI_generated_text_classification/main/models",
)

# ---- stylometrics (copied from src/stylometrics.py, self-contained) ----
def _syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


def extract_stylometrics(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        text = ""
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text)
    n_words = len(words)
    raw_sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sents = max(1, len(raw_sents))
    syl = [_syllables(w) for w in words]
    total_syl = sum(syl)
    sample = [w.lower() for w in words[:200]] if n_words >= 200 else [w.lower() for w in words]
    ttr = (len(set(sample)) / len(sample)) if sample else 0.0
    freq = {}
    for w in [w.lower() for w in words]:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1) / n_words if n_words else 0.0
    avg_wl = float(np.mean([len(w) for w in words])) if words else 0.0
    sent_lens = [len(re.findall(r"\b[A-Za-z0-9'-]+\b", s)) for s in raw_sents] if raw_sents else [0]
    mean_sl = float(np.mean(sent_lens))
    std_sl = float(np.std(sent_lens))
    burst = (std_sl / mean_sl) if mean_sl > 0 else 0.0
    base = max(1, len(text))
    alpha = sum(1 for c in text if c.isalpha())
    cap = sum(1 for c in text if c.isupper())
    cap_ratio = (cap / alpha) if alpha > 0 else 0.0
    fre = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (total_syl / n_words) if n_words and n_sents else 0.0
    fk = 0.39 * (n_words / n_sents) + 11.8 * (total_syl / n_words) - 15.59 if n_words and n_sents else 0.0
    return {
        "num_words": float(n_words), "num_sentences": float(n_sents),
        "avg_word_length": avg_wl, "ttr": float(ttr), "hapax_ratio": float(hapax),
        "mean_sentence_length": mean_sl, "burstiness": float(burst),
        "question_rate": (text.count("?") / base) * 100.0,
        "exclamation_rate": (text.count("!") / base) * 100.0,
        "capital_ratio": cap_ratio,
        "flesch_reading_ease": float(fre), "flesch_kincaid_grade": float(fk),
    }


# ---- type features (copied from moe/type_features.py) ----
_CONTRACTION_RE = re.compile(r"\b(i'm|i've|i'll|you're|you've|you'll|he's|she's|it's|that's|what's|don't|doesn't|didn't|won't|can't|cannot|n't|'ll|'re|'ve)\b", re.I)
_INFORMAL_RE = re.compile(r"\b(lol|haha|yeah|yep|gonna|wanna|kinda|sorta|btw|tbh|imo|ain't|yea|nah|yo|hey)\b", re.I)
_FIRST_RE = re.compile(r"\b(i|me|my|we|our|us)\b", re.I)
_SECOND_RE = re.compile(r"\b(you|your|yours)\b", re.I)


def word_count(text: str) -> int:
    return len(str(text).split()) if text else 0


def length_bucket(n: int) -> str:
    if n <= 150:
        return "short"
    if n <= 400:
        return "long"
    return "verylong"


def extract_type_features(text: str) -> dict:
    s = extract_stylometrics(text)
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", str(text))
    n = max(1, len(words))
    lower = str(text).lower()
    return {
        "log_words": float(np.log1p(len(words))),
        "sentences": s["num_sentences"],
        "mean_sentence_length": s["mean_sentence_length"],
        "ttr": s["ttr"],
        "hapax_ratio": s["hapax_ratio"],
        "avg_word_length": s["avg_word_length"],
        "contraction_rate": len(_CONTRACTION_RE.findall(lower)) / n,
        "informal_rate": len(_INFORMAL_RE.findall(lower)) / n,
        "first_person_rate": len(_FIRST_RE.findall(lower)) / n,
        "second_person_rate": len(_SECOND_RE.findall(lower)) / n,
        "question_rate": s["question_rate"],
        "exclamation_rate": s["exclamation_rate"],
        "capital_ratio": s["capital_ratio"],
        "flesch_reading_ease": s["flesch_reading_ease"],
        "flesch_kincaid_grade": s["flesch_kincaid_grade"],
        "burstiness": s["burstiness"],
    }


# ---- model loading (bundled local first, GitHub cloud fallback) ----
_CACHE: dict = {}


def _resolve(rel: str) -> Path:
    """Local bundled path, else download from GitHub cloud into /tmp."""
    local = LOCAL_MODELS / rel
    if local.exists():
        return local
    cloud = Path("/tmp") / "moe-models" / rel
    cloud.parent.mkdir(parents=True, exist_ok=True)
    if not cloud.exists():
        url = f"{CLOUD_BASE.rstrip('/')}/{rel}"
        urllib.request.urlretrieve(url, cloud)
    return cloud


def _load(rel: str):
    if rel not in _CACHE:
        _CACHE[rel] = joblib.load(_resolve(rel))
    return _CACHE[rel]


def _threshold(bucket: str, n: int) -> float:
    try:
        with open(_resolve("moe/thresholds.json")) as f:
            cfg = json.load(f)
        if bucket == "short" and "short_formula" in cfg:
            fm = cfg["short_formula"]
            return float(min(fm["high"], max(fm["low"], fm["intercept"] + fm["slope"] * n)))
        if bucket in cfg:
            return float(cfg[bucket])
    except Exception:
        pass
    return 0.5


def predict_moe(text: str) -> dict:
    n = word_count(text)
    bucket = length_bucket(n)
    ex = _load(f"moe/expert_{bucket}.pkl")
    X = hstack([ex["word_vec"].transform([text]), ex["char_vec"].transform([text])]).tocsr()
    prob_ai = float(ex["clf"].predict_proba(X)[0, 1])
    th = _threshold(bucket, n)
    label = "AI-Generated" if prob_ai >= th else "Human-Written"
    type_label, type_conf = "unknown", 0.0
    try:
        gate = _load("moe/type_gate.pkl")
        feats = extract_type_features(text)
        Xg = np.array([[feats[k] for k in gate["features"]]])
        type_label = str(gate["clf"].predict(Xg)[0])
        type_conf = float(gate["clf"].predict_proba(Xg).max())
    except Exception:
        pass
    warn = ""
    if n < 25:
        warn = "WARNING: <25 words - very low signal, treat as uncertain."
    elif n <= 50:
        warn = "CAUTION: <=50 words - short-text range, check confidence."
    return {
        "label": label, "prob_ai": round(prob_ai, 4),
        "prob_human": round(1 - prob_ai, 4), "confidence": round(max(prob_ai, 1 - prob_ai), 4),
        "threshold_used": round(th, 3), "expert": bucket, "words": n,
        "text_type": type_label, "type_confidence": round(type_conf, 3), "warning": warn,
    }


# Static site files are bundled inside api/ (index.html, figures/, reports/)
# for Vercel; fall back to web/ when running locally.
WEB_ROOT = HERE if (HERE / "index.html").exists() else HERE.parent
_MIME = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
         ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".pdf": "application/pdf", ".svg": "image/svg+xml", ".txt": "text/plain",
         ".ico": "image/x-icon", ".json": "application/json", ".md": "text/markdown"}


class handler(BaseHTTPRequestHandler):
    def _serve_file(self, rel: str):
        """Serve a static file from web/ (index.html, figures/, reports/)."""
        try:
            rel_clean = rel.split("?")[0].split("#")[0].strip("/")
            if not rel_clean:
                rel_clean = "index.html"
            candidate = (WEB_ROOT / rel_clean).resolve()
            if not str(candidate).startswith(str(WEB_ROOT.resolve())) or not candidate.is_file():
                return False
            if ".vercel" in candidate.parts or candidate.suffix == ".py":
                return False
            mime = _MIME.get(candidate.suffix.lower(), "application/octet-stream")
            data = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return True
        except Exception:
            return False

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(204, {})

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/api/predict", "/api/predict/", "/api/health"):
            self._json(200, {"status": "ok", "service": "MoE AI-vs-Human text classifier",
                             "models": "bundled api/models/ + GitHub cloud fallback",
                             "usage": "POST JSON {\"text\": \"...\"}", "site": "/"})
            return
        if self._serve_file(path):
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/predict", "/api/predict/"):
            self._json(404, {"error": "not found — POST to /api/predict"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            text = (json.loads(raw.decode() or "{}") or {}).get("text", "")
            if not str(text).strip():
                self._json(400, {"error": "Empty text — paste some text to classify."})
                return
            self._json(200, predict_moe(str(text)))
        except Exception as e:
            self._json(500, {"error": f"inference failed: {e}"})

    def log_message(self, *args):
        pass
