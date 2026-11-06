"""
test/langchain_tool.py — LangChain tool wrapper around the TF-IDF model.
Lets any LangChain agent call `ai_human_classifier` as a tool.
No LLM key needed for the tool itself; plug it into a real LLM agent later.

Install: pip install langchain-core
Run demo: venv/bin/python test/langchain_tool.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from inference import load_artifacts, predict_text, predict_batch

_ARTIFACTS = None

def get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()
    return _ARTIFACTS

class ClassifyInput(BaseModel):
    text: str = Field(description="Essay / passage text to classify as Human-Written or AI-Generated")
    model: str = Field(default="lr", description="Classifier: 'lr' = Logistic Regression, 'svm' = Linear SVM (Calibrated)")
    stylometrics: bool = Field(default=False, description="Include stylometric features (TTR, burstiness, Flesch) in result")

class AIHumanClassifierTool(BaseTool):
    name: str = "ai_human_classifier"
    description: str = (
        "Classify whether a text passage is Human-Written or AI-Generated. "
        "Uses a Dual TF-IDF model (word 1-2 grams + char_wb 3-5 grams, 75k features) "
        "trained on 20,000 essays (16k train / 4k val, 99%+ test accuracy). "
        "Input: text (required), model ('lr' or 'svm'), stylometrics (bool). "
        "Returns predicted_label, probability_ai (0-1), confidence, and optional stylometrics."
    )
    args_schema: Type[ClassifyInput] = ClassifyInput

    def _run(self, text: str, model: str = "lr", stylometrics: bool = False) -> str:
        art = get_artifacts()
        r = predict_text(text, art, model=model, show_stylometrics=stylometrics)
        import json
        return json.dumps(r, indent=2)

# singleton for agents
ai_human_classifier = AIHumanClassifierTool()

class BatchClassifyTool(BaseTool):
    name: str = "ai_human_classifier_batch"
    description: str = "Batch-classify multiple texts at once. Input: texts (list of strings), model ('lr'|'svm'). Returns list of {predicted_label, probability_ai}."
    def _run(self, texts: list, model: str = "lr") -> str:
        art = get_artifacts()
        probs = predict_batch(texts, art, model=model)
        import json
        out = [{"predicted_label": "AI-Generated" if float(p) >= 0.5 else "Human-Written",
                "probability_ai": float(p)} for p in probs]
        return json.dumps(out, indent=2)

batch_classifier = BatchClassifyTool()

# --- No-LLM runnable demo (pure tool call, no API key needed) ---
def run_demo():
    print("=== LangChain Tool Demo (no LLM needed) ===")
    print(f"Tool: {ai_human_classifier.name}")
    print(f"Description: {ai_human_classifier.description[:120]}...\n")
    examples = [
        ("Human essay (typos)", Path(ROOT/"test/example_texts/human_essay_school.txt").read_text()[:600]),
        ("AI essay (structured)", Path(ROOT/"test/example_texts/ai_essay_distance_learning.txt").read_text()[:600]),
        ("Real Human (test.csv)", Path(ROOT/"test/example_texts/human_real_01.txt").read_text()[:600]),
        ("Real AI (test.csv)", Path(ROOT/"test/example_texts/ai_real_01.txt").read_text()[:600]),
    ]
    for name, txt in examples:
        print(f"--- {name} ---")
        print(ai_human_classifier.run({"text": txt, "model": "lr", "stylometrics": False}))
        print()
    # batch demo
    print("=== Batch Tool ===")
    print(batch_classifier.run({"texts": [e[1] for e in examples], "model": "svm"}))

if __name__ == "__main__":
    run_demo()
