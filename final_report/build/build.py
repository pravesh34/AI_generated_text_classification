"""Build the consolidated report.

Loads verified metrics (JSON) + result CSVs into the M mapping, assembles
the five content parts into one Document and renders the PDF
(reportlab) — the DOCX renderer plugs in the same way.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.abspath(os.path.join(HERE, ".."))

METRICS_JSON = os.path.join(HERE, "metrics_verified.json")
DATASET_JSON = os.path.join(HERE, "dataset_stats_verified.json")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def _rows(path, key=None, converters=None):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    conv = converters or {}
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            d[k] = conv[k](v) if k in conv else v
        out.append(d)
    if key is None:
        return out
    return {r[key]: r for r in out}


def load_metrics():
    with open(METRICS_JSON, encoding="utf-8") as fh:
        test = json.load(fh)
    with open(DATASET_JSON, encoding="utf-8") as fh:
        ds = json.load(fh)

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    conv = {c: _f for c in metric_cols}

    base = _rows(os.path.join(RESULTS, "moe_baseline_comparison.csv"),
                 key="test_condition", converters=conv)
    routed = _rows(os.path.join(RESULTS, "moe_routed_metrics.csv"),
                   key="test_condition", converters=conv)
    experts = _rows(os.path.join(RESULTS, "moe_expert_metrics.csv"),
                    converters=dict(conv, n_train=int, n_val=int,
                                    cm_tn=int, cm_fp=int, cm_fn=int,
                                    cm_tp=int))
    calib = _rows(os.path.join(RESULTS, "moe_calibrated_metrics.csv"),
                  converters=conv)          # threshold stays a string
    val_rows = _rows(os.path.join(RESULTS, "baseline_ml_metrics.csv"),
                     converters=conv)
    for r in val_rows:
        if "Model" in r:
            r["model"] = r.pop("Model")
    with open(os.path.join(RESULTS, "moe_summary.json"),
              encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(os.path.join(ROOT, "models", "moe", "thresholds.json"),
              encoding="utf-8") as fh:
        thresholds = json.load(fh)

    calib_map = {(r["condition"], r["threshold"]): r
                 for r in calib if r["subset"] == "eval"}

    return {
        "test": test,
        "ds": ds,
        "base": base,
        "routed": routed,
        "experts": experts,
        "calib": calib,
        "calib_map": calib_map,
        "val_rows": val_rows,
        "summary": summary,
        "thresholds": thresholds,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def assemble(M):
    import content_front
    import content_part1a
    import content_part1b
    import content_part2
    import content_part3
    import content_part4
    from blocks import Document

    return Document([
        ("Front Matter", "", content_front.front_matter()),
        ("Part I — Complete Project Report", "",
         content_part1a.part1a(M) + content_part1b.part1b(M)),
        ("Part II — Research Paper", "", content_part2.part2(M)),
        ("Part III — Deep-Dive Research Paper", "", content_part3.part3(M)),
        ("Part IV — Code Walkthrough & Reproducibility Guide", "",
         content_part4.part4(M)),
    ])


def main():
    M = load_metrics()
    doc = assemble(M)

    from blocks import FIGS
    print("assembled: %d blocks, %d figures, %d tables"
          % (len(doc.blocks), len(doc.fig_no), len(doc.tab_no)))

    import render_pdf
    out = os.path.join(OUTDIR, "Final_Consolidated_Project_Report.pdf")
    render_pdf.render(doc, out)
    print("PDF ->", out, "(%.0f KB)" % (os.path.getsize(out) / 1024))
    return out


if __name__ == "__main__":
    main()
