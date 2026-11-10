# 📄 Reports

**One consolidated document:** [`Full_Report.pdf`](Full_Report.pdf) — 39 pages, everything the project produced.

| Part | Content | Pages |
|---|---|---|
| **Cover + contents** | Title page, authors, table of contents with page numbers | 1 |
| **Part I — Project report** | Three phases (stylometrics + dual TF-IDF, transformers, evaluation), short-text failure analysis, MoE upgrade, calibration, results | 12 |
| **Part II — Conference paper** | *Length-Expertized Authorship Detection* — baseline vs MoE, short-text ablation, cross-domain evaluation, architecture | 7 |
| **Part III — Deep-dive paper** | Formal method, per-length human/AI split (48.9% human @25w flip), threshold calibration analysis, appendices | 9 |
| **Part IV — Run guide** | Code workflow: setup → train → infer → API → UI | 7 |

Also here: `VIDEO_SCRIPT_MoE_20min.md` — 20-minute narrated talk script with visual cues.

## What happened to the old PDFs

All earlier single-purpose PDFs (`report.pdf`, `Project_Report_AI_Text_Classification.pdf`, `Research_Paper_MoE_...pdf`, `DeepSeek_Style_...pdf`, `Project_Report_MoE_Appendix.pdf`, `Code_Workflow_and_Run_Guide.pdf`, `*.pre-moe.pdf`) were **merged into `Full_Report.pdf`** and **removed from GitHub** (`.gitignore` now ignores `*.pdf` except the consolidated file). Their local copies are untouched in this folder and in the repo root.

Copies of `Full_Report.pdf` also live in `docs/reports/` (GitHub Pages) and `web/reports/` (Vercel) so both sites serve the same document.

## Rebuilding `Full_Report.pdf`

The generator script is intentionally **not** in Git (only PDF files are shared). Rebuild with a short merge: cover page via `reportlab`, then `pypdf` appends `report.pdf`, `Research_Paper_MoE_AI_Text_Detection.pdf`, `DeepSeek_Style_Research_Paper_MoE.pdf`, `Code_Workflow_and_Run_Guide.pdf` with divider pages between parts — or simply re-download the published `Full_Report.pdf`.

## Reading order

1. **5 minutes:** root [`README.md`](../README.md) — results table
2. **The full story:** `Full_Report.pdf`, Part I
3. **Formal method & ablation:** Part III
4. **Reproduce:** Part IV + [`../moe/README.md`](../moe/README.md)
