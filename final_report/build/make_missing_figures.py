"""Recreate the two architecture diagrams that do NOT exist as original assets.

Both are honest schematics drawn from the actual repository source code
(train_baseline.py parameters; real modules/endpoints/deploy targets).
No experimental numbers are invented here.

Outputs:
  results/figures/gen_tfidf_fusion.png
  results/figures/gen_system_architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(ROOT, "results", "figures")

NAVY = "#1F2A44"
ACCENT = "#4C72B0"
LIGHT = "#EAF0F8"
LIGHT2 = "#D6E4F5"
GREY = "#6b7280"
ORANGE = "#DD8452"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9


def box(ax, x, y, w, h, title, desc="", fc=LIGHT, ts=9.5, ds=7.6, ec=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=1.1))
    if desc:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + h * 0.28, desc, ha="center", va="center",
                fontsize=ds, color="#333333")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=NAVY)


def arrow(ax, x1, y1, x2, y2, color=ACCENT, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- Figure 1
def tfidf_fusion():
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.axis("off")
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.03, 1.03)

    box(ax, 0.005, 0.42, 0.135, 0.17, "Input text", "any essay,\nany length", fc="#FFFFFF")
    box(ax, 0.175, 0.42, 0.16, 0.17, "Preprocessing",
        "NFKC unicode\ncollapse whitespace\nlowercase", fc=LIGHT2, ds=7.2)

    box(ax, 0.375, 0.63, 0.235, 0.17, "Word TF-IDF",
        "ngram (1,2)  ·  25,000 features\nmin_df=2  ·  sublinear_tf", fc=LIGHT, ds=7.2)
    box(ax, 0.375, 0.14, 0.235, 0.17, "Character TF-IDF",
        "char_wb (3,5)  ·  50,000 features\nmin_df=3  ·  sublinear_tf", fc=LIGHT, ds=7.2)

    box(ax, 0.655, 0.385, 0.14, 0.2, "Fusion\nhstack", "75,000-dim\nsparse vector", fc=LIGHT2, ds=7.2)

    box(ax, 0.83, 0.66, 0.165, 0.14, "Logistic Regression", "C = 1.5, lbfgs", fc="#FFFFFF", ts=8.6, ds=7.2)
    box(ax, 0.83, 0.44, 0.165, 0.14, "Linear SVM (cal.)", "C = 1.0, cv=3 Platt", fc="#FFFFFF", ts=8.6, ds=7.2)
    box(ax, 0.83, 0.22, 0.165, 0.14, "Multinomial NB", "alpha = 0.1", fc="#FFFFFF", ts=8.6, ds=7.2)

    arrow(ax, 0.143, 0.505, 0.173, 0.505)
    arrow(ax, 0.337, 0.505, 0.373, 0.68)
    arrow(ax, 0.337, 0.49, 0.373, 0.24)
    arrow(ax, 0.613, 0.70, 0.655, 0.55)
    arrow(ax, 0.613, 0.22, 0.655, 0.43)
    arrow(ax, 0.797, 0.52, 0.828, 0.70)
    arrow(ax, 0.797, 0.505, 0.828, 0.51)
    arrow(ax, 0.797, 0.49, 0.828, 0.30)

    ax.text(0.5, 0.035, "Dual TF-IDF feature fusion — train_baseline.py (fit on train split only, "
                        "transform on val/test: no leakage)",
            ha="center", fontsize=8.4, style="italic", color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "gen_tfidf_fusion.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def system_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.axis("off")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.045, 1.02)

    ax.text(0.5, 0.99, "End-to-end system architecture (local training → six inference front-ends → cloud deploy)",
            ha="center", fontsize=10.5, fontweight="bold", color=NAVY)

    # Data / training column
    box(ax, 0.01, 0.70, 0.20, 0.13, "data/", "train.csv 290,300\ntest.csv 97,447", fc="#FFFFFF", ds=7.2)
    box(ax, 0.01, 0.50, 0.20, 0.13, "train_baseline.py", "11 steps · 33 s\n20k subsample, seed 42", fc=LIGHT2, ds=7.2)
    box(ax, 0.01, 0.30, 0.20, 0.13, "models/*.pkl (4)", "word vec · char vec\nLR · calibrated SVM", fc=LIGHT, ds=7.2)
    box(ax, 0.01, 0.09, 0.20, 0.13, "models/moe/*.pkl", "3 experts · type gate\nthresholds.json", fc=LIGHT, ds=7.2)
    arrow(ax, 0.11, 0.68, 0.11, 0.635)
    arrow(ax, 0.11, 0.48, 0.11, 0.435)
    arrow(ax, 0.11, 0.28, 0.11, 0.225)

    # Inference core
    box(ax, 0.275, 0.36, 0.20, 0.24, "Inference core",
        "test/inference.py\nload_artifacts()\nvectorize() → hstack\npredict_proba() → P(AI)\n\n"
        "moe/predict_moe.py\nlength gate + type gate", fc=LIGHT2, ds=7.4, ts=10)
    arrow(ax, 0.215, 0.365, 0.273, 0.46)
    arrow(ax, 0.215, 0.155, 0.273, 0.42)

    # Six front-ends
    front = [
        ("CLI — test/predict.py", "single / file / folder / CSV"),
        ("Batch — terminal_demo.py", "interactive + --loop chat"),
        ("REST — test/api.py :8001", "POST /predict, /predict/batch"),
        ("LangChain tool", "ai_human_classifier (BaseTool)"),
        ("Streamlit UI :8501", "app/app.py — live + reports"),
        ("Vercel serverless", "web/api/predict.py /api/predict"),
    ]
    y = 0.88
    for i, (t, d) in enumerate(front):
        box(ax, 0.545, y - 0.09, 0.235, 0.088, t, d, fc=LIGHT if i % 2 == 0 else "#FFFFFF", ts=8.2, ds=7.0)
        arrow(ax, 0.478, 0.48 + (2.5 - i) * 0.012, 0.543, y - 0.045, lw=1.1)
        y -= 0.14

    # Outputs / deploy
    box(ax, 0.825, 0.64, 0.165, 0.13, "GitHub Pages", "docs/  —  site +\nFull_Report.pdf + figures", fc=LIGHT, ds=7.2)
    box(ax, 0.825, 0.44, 0.165, 0.13, "Vercel (production)", "moe-text-classifier-\npravesh12.vercel.app", fc=LIGHT, ds=7.2)
    box(ax, 0.825, 0.24, 0.165, 0.13, "results/", "metrics CSV ·\nfigures/*.png", fc="#FFFFFF", ds=7.2)
    arrow(ax, 0.783, 0.70, 0.823, 0.70)
    arrow(ax, 0.783, 0.50, 0.823, 0.50)
    arrow(ax, 0.783, 0.30, 0.823, 0.30)

    ax.text(0.5, -0.005, "One inference engine, six front-ends (local) + two cloud deployments; "
                         "all four .pkl artifacts are shared by every path.",
            ha="center", fontsize=8.4, style="italic", color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "gen_system_architecture.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    tfidf_fusion()
    system_architecture()
    from PIL import Image
    for n in ("gen_tfidf_fusion.png", "gen_system_architecture.png"):
        im = Image.open(os.path.join(FIGDIR, n))
        print(f"{n}: {im.size[0]}x{im.size[1]} aspect={im.size[1]/im.size[0]:.3f}")
