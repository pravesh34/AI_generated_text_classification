# Data Catalogue — large files live in Drive, not Git

Large CSVs are **gitignored** (see `.gitignore`) and hosted on Google Drive.
GitHub tracks only `sample_*.csv` + scripts/metrics/models. Download from Drive instead of `git clone` for data.

## Drive links

| File | Drive link |
| :--- | :--- |
| `data/moe/hc3_small.csv` (62 MB) | **TODO — paste Drive share URL here** |
| `data/moe/modern_essays.csv` (64 MB) | **TODO — paste Drive share URL here** |
| `data/moe/ai_text_moe_data.zip` (optional bundle, ~30 MB zipped) | **TODO — paste Drive share URL here** |
| `data/train.csv` (623 MB) / `data/test.csv` (212 MB) | **TODO — paste Drive share URL here** (original essays) |

## Upload (how to replace these links)

1. From this folder, make a bundle (optional but recommended):
   `zip -r ai_text_moe_data.zip data/moe/*.csv` → ~30 MB upload.
   For essays: `zip data_essays.zip data/train.csv data/test.csv` (~400 MB zipped).
2. Upload to Google Drive → New → File upload → set sharing to **Anyone with the link** → copy link.
3. Paste the links above, then `git add data/README.md && git commit && git push`.

## Reproduce without Drive (script path)

```bash
# HC3 + modern essays (Hub JSONL/parquet, no manual upload needed)
PYTHONPATH=. python3 moe/download_small_text.py
```

Hub sources: `Hello-SimpleAI/HC3` (`reddit_eli5`, `open_qa`, `finance` JSONLs) + `artnitolog/llm-generated-texts` parquet.
Original `train.csv` / `test.csv` are not script-downloadable — use the Drive links above.
