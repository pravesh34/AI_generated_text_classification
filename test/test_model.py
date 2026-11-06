"""
test/test_model.py — Quick smoke tests + full evaluation driver.
Run: venv/bin/python test/test_model.py
     venv/bin/python test/test_model.py --full   (whole test.csv 97k)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_artifacts, predict_text, evaluate_csv

def test_single():
    art = load_artifacts()
    human = "I went to the market yesterday and bought some fresh vegetables. The weather was nice and sunny, and I enjoyed walking there with my friend. We talked about school and our plans for the weekend."
    ai = "In today's rapidly evolving digital landscape, it is important to note that understanding various methodologies can lead to improved outcomes. Furthermore, research suggests that leveraging data-driven approaches facilitates more efficient decision-making processes across multiple dimensions."
    for name, txt in [("Human", human), ("AI", ai)]:
        r = predict_text(txt, art, model="lr", show_stylometrics=True)
        assert "predicted_label" in r and "probability_ai" in r
        print(f"{name}: -> {r['predicted_label']} P(AI)={r['probability_ai']:.3f} TTR={r['stylometrics']['ttr']:.2f}")

def test_evaluate(sample=2000):
    art = load_artifacts()
    metrics, df = evaluate_csv(str(ROOT / "data" / "test.csv"), art, model="lr", sample_n=sample)
    assert metrics["accuracy"] > 0.90, f"Accuracy too low: {metrics['accuracy']}"
    print(f"EVAL {sample} rows: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    test_single()
    if "--full" in sys.argv:
        test_evaluate(sample=None)
    else:
        test_evaluate(sample=2000)
    print("All tests passed ✓")
