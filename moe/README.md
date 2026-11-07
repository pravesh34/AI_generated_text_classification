# MoE Upgrade: Short / Long / VeryLong + Chat / Casual / Essay

Fixes the short-text failure of the baseline single model (67% @ 25w, 83% @ 50w,
60% @ 1 sentence; humans biased to AI) by adding dedicated small-text data and
routing to length-specialized experts.

## Datasets (downloaded via `download_small_text.py`)

| Source | n | Use | Why |
| :--- | :--- | :--- | :--- |
| HC3 `reddit_eli5` + `open_qa` + `finance` (Hello-SimpleAI/HC3) | 80,811 answers (56k human / 25k AI), median 117w, 61% ≤150w | SHORT (chat/casual) | Genuine short Q&A chat text: human vs ChatGPT answers |
| `artnitolog/llm-generated-texts` (GPT-4 Turbo/Omni, Claude 3 Opus, Llama3, Command R+) | 17,997 (3k human + 15k AI), median 510w, 68% >400w | LONG/VERYLONG modern | Modern-LLM essays (2024) complement old ChatGPT essays |
| Existing `data/train.csv` (20k balanced subsample) | 20,000 essays (~393w avg) | LONG/VERYLONG + truncation aug | Full essays + 50/100/150w and 200/300w truncated copies |

## Architecture

```
              text
                |
        length gate (word count, deterministic)
        ├─ ≤150w  → expert_short   (HC3 + truncated essays)
        ├─ 151-400 → expert_long   (HC3 long + modern long + base long)
        └─ >400w  → expert_verylong (base >400w + modern >400w)
                |
        each expert = Word TF-IDF (1,2) 15k + Char TF-IDF (3,5) 25k + LR(C=1.5)
                |
        + type gate (LR over 16 stylometric feats → chat / casual / essay)
```

Type labels: HC3 `open_qa` → chat, HC3 `reddit_eli5`/`finance` → casual,
truncated essays → casual, full + modern essays → essay.

## Files

- `moe/download_small_text.py` — downloads → `data/moe/hc3_small.csv`, `data/moe/modern_essays.csv`
- `moe/type_features.py` — length buckets (150/400) + 16-feature type extractor
- `moe/train_moe.py` — trains 3 experts + type gate → `models/moe/`, evals vs baseline
- `moe/predict_moe.py` — router inference with type + reliability warning

Results: `results/moe_expert_metrics.csv`, `results/moe_routed_metrics.csv`,
`results/moe_baseline_comparison.csv`, `results/moe_summary.json`.

## Run

```bash
PYTHONPATH=. python3 moe/download_small_text.py
PYTHONPATH=. python3 moe/train_moe.py
PYTHONPATH=. python3 moe/predict_moe.py "I think the answer is maybe... let me explain..."
```
