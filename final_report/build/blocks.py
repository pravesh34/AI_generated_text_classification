"""Block model for the consolidated report.

One content stream (list of blocks) renders to BOTH PDF (reportlab) and
DOCX (python-docx).  Content files only use the constructors below.

Inline markup (valid for both renderers):
    <b>bold</b> <i>italic</i> <code>monospace</code>
    <sub>subscript</sub> <sup>superscript</sup>
    XML entities: &amp; &lt; &gt; &quot;
Cross references (resolved after assembly):
    {{f:fig_id}}  -> "Figure N"
    {{t:tab_id}}  -> "Table N"
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(ROOT, "results", "figures")
ASSETDIR = os.path.join(ROOT, "docs", "assets")

# --------------------------------------------------------------------------
# Figure registry — canonical captions + provenance (for Figure_Inventory.md)
# status: recovered  = original PNG generated during the project
#         recreated  = drawn now from real source code / real screenshot
# --------------------------------------------------------------------------
FIGS = {
    "class_dist": dict(
        file=os.path.join(FIGDIR, "class_distribution.png"),
        caption="Class balance of the corpora: train.csv is perfectly balanced "
                "(145,150 per class); test.csv is human-heavy (62.8% human).",
        status="recovered", origin="make_report.py (original Figure 1)"),
    "length_hist": dict(
        file=os.path.join(FIGDIR, "length_hist.png"),
        caption="Essay length distribution (characters, 5,000-sample test set). "
                "The corpus contains essentially no natively short texts.",
        status="recovered", origin="make_report.py (original Figure 2)"),
    "tfidf_fusion": dict(
        file=os.path.join(FIGDIR, "gen_tfidf_fusion.png"),
        caption="Dual TF-IDF feature fusion and the three linear classifiers, "
                "as implemented in train_baseline.py.",
        status="recreated", origin="drawn from train_baseline.py / src/preprocessing.py (no original figure existed)"),
    "flow_train": dict(
        file=os.path.join(FIGDIR, "flow_train.png"),
        caption="Training workflow of train_baseline.py — eleven logged steps, "
                "33 s end-to-end with the data present.",
        status="recovered", origin="make_run_guide.py (original Figure 1)"),
    "model_cmp": dict(
        file=os.path.join(FIGDIR, "model_comparison.png"),
        caption="Validation accuracy, F1 and ROC-AUC of the three baseline "
                "classifiers (4,000-sample validation split).",
        status="recovered", origin="make_report.py (original Figure 3)"),
    "confusion": dict(
        file=os.path.join(FIGDIR, "confusion_matrices.png"),
        caption="Confusion matrices on the 5,000-sample held-out test: Logistic "
                "Regression (left) and calibrated Linear SVM (right).",
        status="recovered", origin="make_report.py (original Figure 4)"),
    "roc": dict(
        file=os.path.join(FIGDIR, "roc_curves.png"),
        caption="ROC curves of Logistic Regression and Linear SVM on the held-out "
                "test sample; both reach AUC ≈ 0.999.",
        status="recovered", origin="make_report.py (original Figure 5)"),
    "acc_vs_len": dict(
        file=os.path.join(FIGDIR, "moe_accuracy_vs_length.png"),
        caption="Accuracy versus text length on the 5,000 held-out essays: the "
                "frozen baseline (orange) collapses as the input shrinks; the routed "
                "MoE (blue, developed in Chapters 8–9) lifts the short regime.",
        status="recovered", origin="moe/make_moe_appendix.py (original Figure A1)"),
    "human_ai_split": dict(
        file=os.path.join(FIGDIR, "deep_human_ai_split.png"),
        caption="Length-conditional accuracy split by true class. Left: human-only "
                "accuracy — the baseline falls to 48.9% at 25 words. Right: AI-only "
                "accuracy — the error direction inverts under the MoE.",
        status="recovered", origin="make_deepseek_paper.py (original Figure 7)"),
    "moe_arch": dict(
        file=os.path.join(FIGDIR, "moe_architecture.png"),
        caption="Mixture-of-Experts architecture: deterministic length gate, three "
                "per-bucket TF-IDF + Logistic Regression experts, learned type gate "
                "(chat / casual / essay) and length-adaptive decision threshold.",
        status="recovered", origin="make_research_paper.py (original Figure 4)"),
    "type_gate": dict(
        file=os.path.join(FIGDIR, "moe_type_gate.png"),
        caption="Type-gate per-class precision, recall and F1 (validation accuracy "
                "80.5%, n = 1,500 per class).",
        status="recovered", origin="moe/make_moe_appendix.py (original Figure A3)"),
    "moe_experts": dict(
        file=os.path.join(FIGDIR, "moe_experts.png"),
        caption="Per-expert validation accuracy, F1 and ROC-AUC — every expert "
                "reaches ≥ 0.97 within its own length bucket.",
        status="recovered", origin="moe/make_moe_appendix.py (original Figure A4)"),
    "pr_flip": dict(
        file=os.path.join(FIGDIR, "moe_pr_flip.png"),
        caption="The 25-word precision/recall flip: the baseline over-calls AI "
                "(P = 0.53, R = 0.99); the MoE under-calls AI (P = 0.93, R = 0.68) "
                "while winning on total accuracy.",
        status="recovered", origin="moe/make_moe_appendix.py (original Figure A2)"),
    "cross_domain": dict(
        file=os.path.join(FIGDIR, "moe_cross_domain.png"),
        caption="Cross-domain evaluation. The baseline collapses to 48.6% on unseen "
                "HC3 chat answers (worse than chance) and to 85.5% on unseen 2024-LLM "
                "essays; the routed MoE holds 98.5% and 95.7%.",
        status="recovered", origin="moe/make_moe_appendix.py (original Figure A5)"),
    "threshold_curve": dict(
        file=os.path.join(FIGDIR, "deep_threshold_curve.png"),
        caption="F1 versus decision threshold for four truncation lengths. The optimum "
                "drifts right with length (25w → 0.32 … 150w → 0.64); no single global "
                "threshold serves all regimes.",
        status="recovered", origin="make_deepseek_paper.py (original Figure 6)"),
    "flow_infer": dict(
        file=os.path.join(FIGDIR, "flow_infer.png"),
        caption="Inference workflow: every front-end (CLI, batch, API, LangChain, "
                "Streamlit, Vercel) goes through test/inference.py.",
        status="recovered", origin="make_run_guide.py (original Figure 2)"),
    "screenshot": dict(
        file=os.path.join(ASSETDIR, "live-inference.png"),
        caption="Screenshot of the Streamlit demonstration interface "
                "(app/app.py) performing live MoE inference with reliability warning, "
                "routed metrics and stylometric evidence.",
        status="recovered", origin="docs/assets/live-inference.png (captured during project development)"),
    "sys_arch": dict(
        file=os.path.join(FIGDIR, "gen_system_architecture.png"),
        caption="End-to-end system architecture: local training produces the shared "
                ".pkl artifacts consumed by one inference core and six front-ends, "
                "with GitHub Pages and Vercel deployments on top.",
        status="recreated", origin="drawn from test/, app/, web/ source code (no original figure existed)"),
}

# --------------------------------------------------------------------------
# Block constructors
# --------------------------------------------------------------------------


def part(title, subtitle=""):
    return ("part", title, subtitle)


def h1(title):
    return ("h1", title)


def h2(title):
    return ("h2", title)


def h3(title):
    return ("h3", title)


def p(text):
    return ("p", text)


def abstract(text):
    return ("abstract", text)


def bullets(items):
    return ("bullets", list(items))


def numbered(items):
    return ("numbered", list(items))


def fig(fig_id, width=0.92, repeat=False, caption=None):
    """Insert a registered figure.  repeat=True shows it again without a new
    list-of-figures entry (caption notes the original location)."""
    assert fig_id in FIGS, f"unknown figure id {fig_id}"
    return ("fig", fig_id, width, repeat, caption)


def table(tab_id, caption, header, rows, widths, left_cols=(0,), repeat_header=True):
    """widths: column widths in mm (sum should be <= ~166)."""
    assert tab_id not in TABLE_CAPTIONS, f"duplicate table id {tab_id}"
    TABLE_CAPTIONS[tab_id] = caption
    return ("table", tab_id, caption, header, rows, list(widths),
            tuple(left_cols), repeat_header)


def code(text):
    return ("code", text)


def subtitle(text):
    """Centred accent subtitle (paper title pages)."""
    return ("subtitle", text)


def authors(text):
    """Centred author block (paper title pages)."""
    return ("authors", text)


def refs(items):
    """Reference list: each item already carries its [n] label; no bullets."""
    return ("refs", list(items))


def list_table(caption, header, rows, widths, left_cols=(0,)):
    """Unnumbered reference-style table (no figure/table numbering, no list entry)."""
    return ("listtable", caption, header, rows, list(widths), tuple(left_cols))


def eq(text, label=None):
    return ("eq", text, label)


def pagebreak():
    return ("pagebreak",)


def spacer(mm=4):
    return ("spacer", mm)


def toc(kind):  # kind in {"toc", "lof", "lot"}
    return ("toc", kind)


TABLE_CAPTIONS = {}


# --------------------------------------------------------------------------
# Assembly: assign figure/table numbers, resolve {{f:..}} / {{t:..}} refs
# --------------------------------------------------------------------------
class Document:
    def __init__(self, parts):
        """parts: list of (part_title, subtitle, blocks)."""
        self.parts = parts
        self.fig_no = {}        # id -> int (order of first insertion)
        self.tab_no = {}        # id -> int
        self.fig_section = {}   # id -> section heading of first insertion
        self.tab_section = {}
        self.section = "Front matter"
        blocks = []
        for _t, _s, bl in parts:
            blocks.extend(bl)
        self._assign(blocks)
        self.blocks = blocks

    def _assign(self, blocks):
        nf = nt = 0
        for b in blocks:
            kind = b[0]
            if kind in ("h1", "h2", "h3"):
                if kind == "h1":
                    self.section = b[1]
            elif kind == "fig":
                fid, repeat = b[1], b[3]
                if fid not in self.fig_no:
                    nf += 1
                    self.fig_no[fid] = nf
                    self.fig_section[fid] = self.section
            elif kind == "table":
                tid = b[1]
                if tid not in self.tab_no:
                    nt += 1
                    self.tab_no[tid] = nt
                    self.tab_section[tid] = self.section

    # ---- reference resolution ------------------------------------------
    def resolve(self, text):
        import re
        def rf(m):
            fid = m.group(1)
            if fid not in self.fig_no:
                raise KeyError(f"reference to unknown figure '{fid}'")
            return f"Figure {self.fig_no[fid]}"
        def rt(m):
            tid = m.group(1)
            if tid not in self.tab_no:
                raise KeyError(f"reference to unknown table '{tid}'")
            return f"Table {self.tab_no[tid]}"
        text = re.sub(r"\{\{f:([a-z0-9_]+)\}\}", rf, text)
        text = re.sub(r"\{\{t:([a-z0-9_]+)\}\}", rt, text)
        return text


def fig_caption_text(doc, fid, repeat_of=None):
    cap = FIGS[fid]["caption"]
    n = doc.fig_no[fid]
    if repeat_of:
        return f"Figure {n}: {cap} (repeated from {repeat_of})."
    return f"Figure {n}: {cap}"
