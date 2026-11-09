# 📄 Reports & Papers

All project documents in one place. Originals also remain in the repo root — this folder is the canonical index for sharing and deployment.

| File | What it is | Pages | Pages / size |
|---|---|---|---|
| `report.pdf` (= `Project_Report_AI_Text_Classification.pdf`) | **Main project report** — baseline (6p) + MoE appendix (6p): short-text failure → MoE → calibration | 12 | 690 KB |
| `Research_Paper_MoE_AI_Text_Detection.pdf` | **Conference paper** — baseline vs MoE, short-text ablation, cross-domain, architecture | 7 | 881 KB |
| `DeepSeek_Style_Research_Paper_MoE.pdf` | **Detailed paper** — formal method (§4), per-length ablation, calibration analysis, appendices | 9 | 835 KB |
| `Project_Report_MoE_Appendix.pdf` | Standalone MoE appendix (same as merged appendix in `report.pdf`) | 6 | 330 KB |
| `Code_Workflow_and_Run_Guide.pdf` | Workflow & run guide — setup, training, inference, API, UI | — | 112 KB |
| `Project_Report_AI_Text_Classification.pre-moe.pdf` | Archived pre-MoE report (baseline only, before upgrade) | — | 361 KB |
| `VIDEO_SCRIPT_MoE_20min.md` | 20-minute narrated talk script with visuals cue sheet | — | 14 KB |

## Reading order

1. **Quick overview (5 min):** this table + root [`README.md`](../README.md) § Results
2. **Full story:** `report.pdf` — the submitted report
3. **Paper version:** `Research_Paper_MoE_AI_Text_Detection.pdf` for conference / viva
4. **Deep dive:** `DeepSeek_Style_Research_Paper_MoE.pdf` — method, ablation, calibration
5. **Reproduce:** `Code_Workflow_and_Run_Guide.pdf` + `../moe/README.md`

## Regenerate

```bash
PYTHONPATH=. python3 make_report.py
PYTHONPATH=. python3 make_research_paper.py
PYTHONPATH=. python3 make_deepseek_paper.py
PYTHONPATH=. python3 moe/make_moe_appendix.py
cp *.pdf reports/
```
