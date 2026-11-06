#!/usr/bin/env python3
"""
test/api.py — FastAPI service exposing the model as a REST API + LangChain tool endpoint.

Run:        venv/bin/uvicorn test.api:app --host 0.0.0.0 --port 8001 --reload
            (or: venv/bin/python test/api.py)
Health:     GET  http://localhost:8001/health
Predict:    POST http://localhost:8001/predict  {"text": "...", "model": "lr", "stylometrics": true}
Batch:      POST http://localhost:8001/predict/batch  {"texts": [...], "model": "svm"}
Info:       GET  http://localhost:8001/model/info
Docs:       http://localhost:8001/docs  (Swagger UI)

cURL example:
  curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
    -d '{"text": "In todays rapidly evolving digital landscape...", "model": "lr"}'
"""
import sys
from pathlib import Path
from typing import List, Optional
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from inference import load_artifacts, predict_text, predict_batch

app = FastAPI(title="Human vs AI Text Classifier — API", description="Dual TF-IDF + LR/SVM (75k features, 99%+ test acc)", version="1.0.0")
_ART = None
def art():
    global _ART
    if _ART is None:
        _ART = load_artifacts()
    return _ART

class PredictRequest(BaseModel):
    text: str = Field(..., description="Passage to classify")
    model: str = Field("lr", description="'lr' or 'svm'")
    stylometrics: bool = Field(False, description="Include stylometrics")

class PredictResponse(BaseModel):
    predicted_label: str
    probability_ai: float
    probability_human: float
    confidence: float
    model_used: str
    stylometrics: Optional[dict] = None

class BatchRequest(BaseModel):
    texts: List[str]
    model: str = "lr"

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

@app.get("/model/info")
def model_info():
    a = art()
    return {
        "word_features": len(a["word_vec"].vocabulary_),
        "char_features": len(a["char_vec"].vocabulary_),
        "total_features": len(a["word_vec"].vocabulary_) + len(a["char_vec"].vocabulary_),
        "models": {"lr": "Logistic Regression (C=1.5)", "svm": "Linear SVM Calibrated (C=1.0)"},
        "test_metrics_5k": {"lr": {"acc": 0.9926, "f1": 0.9900, "auc": 0.9994}, "svm": {"acc": 0.9958, "f1": 0.9943, "auc": 0.9995}},
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    r = predict_text(req.text, art(), model=req.model, show_stylometrics=req.stylometrics)
    return PredictResponse(
        predicted_label=r["predicted_label"],
        probability_ai=r["probability_ai"],
        probability_human=r["probability_human"],
        confidence=r["confidence"],
        model_used=r["model"],
        stylometrics=r.get("stylometrics"),
    )

@app.post("/predict/batch")
def predict_batch_endpoint(req: BatchRequest):
    probs = predict_batch(req.texts, art(), model=req.model)
    return [{"predicted_label": "AI-Generated" if float(p)>=0.5 else "Human-Written", "probability_ai": float(p)} for p in probs]

@app.post("/langchain/tool")
def langchain_tool_endpoint(req: PredictRequest):
    """Same as /predict but shaped for LangChain agents (name/description for tool calling)."""
    from langchain_tool import ai_human_classifier
    out = ai_human_classifier.run({"text": req.text, "model": req.model, "stylometrics": req.stylometrics})
    return {"tool": ai_human_classifier.name, "result": out}

if __name__ == "__main__":
    import uvicorn
    print("Starting API on http://localhost:8001  (Swagger: http://localhost:8001/docs)")
    uvicorn.run(app, host="0.0.0.0", port=8001)
