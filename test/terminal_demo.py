#!/usr/bin/env python3
"""
test/terminal_demo.py — Terminal chat that calls the model WITHOUT Streamlit.
Demonstrates all 3 call styles in one terminal session:
  1) Direct inference (inference.predict_text) — fastest
  2) LangChain tool (langchain_tool.ai_human_classifier.run) — agent-ready
  3) REST API (httpx -> localhost:8001) — if api.py is running

Run: venv/bin/python test/terminal_demo.py
     venv/bin/python test/terminal_demo.py --api   (also hit http://localhost:8001/predict)
     venv/bin/python test/terminal_demo.py --loop  (interactive terminal chat)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_artifacts, predict_text
from langchain_tool import ai_human_classifier

DEMO = [
    ("Human essay (school)", (ROOT/"test/example_texts/human_essay_school.txt").read_text()[:500]),
    ("AI essay (distance learning)", (ROOT/"test/example_texts/ai_essay_distance_learning.txt").read_text()[:500]),
    ("Real Human (test.csv)", (ROOT/"test/example_texts/human_real_02.txt").read_text()[:500]),
    ("Real AI (test.csv)", (ROOT/"test/example_texts/ai_real_02.txt").read_text()[:500]),
]

def via_direct(art, text, model="lr"):
    return predict_text(text, art, model=model)

def via_langchain(text, model="lr"):
    import json
    return json.loads(ai_human_classifier.run({"text": text, "model": model, "stylometrics": False}))

def via_api(text, model="lr"):
    import httpx
    r = httpx.post("http://localhost:8001/predict", json={"text": text, "model": model, "stylometrics": False}, timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    use_api = "--api" in sys.argv
    loop = "--loop" in sys.argv
    print("Loading models (75k features)...")
    art = load_artifacts()
    print("Ready. Demo cases:\n")
    for name, txt in DEMO:
        print(f"--- {name} ---")
        r1 = via_direct(art, txt, "lr")
        print(f"[direct]    {r1['predicted_label']:15} P(AI)={r1['probability_ai']:.4f}")
        r2 = via_langchain(txt, "lr")
        print(f"[langchain] {r2['predicted_label']:15} P(AI)={r2['probability_ai']:.4f}  (tool={ai_human_classifier.name})")
        if use_api:
            try:
                r3 = via_api(txt, "lr")
                print(f"[api]       {r3['predicted_label']:15} P(AI)={r3['probability_ai']:.4f}")
            except Exception as e:
                print(f"[api]       FAILED (is api.py running on 8001?): {e}")
        print()
    if loop:
        print("=== Interactive terminal (type essay, blank line to classify, Ctrl+C quit) ===")
        while True:
            print("\n--- paste text ---")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting terminal demo.")
                break
            text = "\n".join(lines)
            if not text.strip():
                continue
            r = via_direct(art, text, "lr")
            print(f"=> {r['predicted_label']}  P(AI)={r['probability_ai']:.4f}  conf={r['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()
