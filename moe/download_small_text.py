"""
Download SHORT-text datasets for the Mixture-of-Experts (MoE) upgrade.

Sources (both public on Hugging Face Hub):
  1. Hello-SimpleAI/HC3 (Guo et al. 2023) — Human ChatGPT Comparison Corpus.
     JSONL files downloaded directly (no datasets-script needed; works on
     datasets 5.x where .py scripts are removed). Flattened to one row per
     answer: text, generated (0=human, 1=chatgpt), domain, source.
     Used for: SHORT expert (chat / casual Q&A). reddit_eli5 is the big,
     most casual chunk (17k questions); open_qa + finance add diversity.
  2. artnitolog/llm-generated-texts — 3000 essay prompts x (1 human + 8 LLM
     continuations: GPT-4 Turbo, GPT-4 Omni, Claude 3 Opus, YandexGPT,
     GigaChat, Llama3 70B, Command R+). Used for: LONG/VERYLONG expert
     with modern LLMs. English-only models kept (GPT-4 Turbo/Omni, Claude,
     Llama3, Command R+); Yandex/GigaChat dropped (mostly Russian refusals).

Output (data/moe/):
  - hc3_small.csv        : text, generated, domain, source
  - modern_essays.csv    : text, generated, prompt, model

Run:  PYTHONPATH=. python3 moe/download_small_text.py
"""
import json
import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "moe")
os.makedirs(OUT, exist_ok=True)

HC3_FILES = {
    "reddit_eli5": "reddit_eli5.jsonl",
    "open_qa": "open_qa.jsonl",
    "finance": "finance.jsonl",
}
ENGLISH_MODELS = [
    "GPT4 Turbo 2024-04-09",
    "GPT4 Omni",
    "Claude 3 Opus",
    "Llama3 70B",
    "Command R+",
]


def download_hc3() -> pd.DataFrame:
    rows = []
    for domain, fname in HC3_FILES.items():
        path = hf_hub_download("Hello-SimpleAI/HC3", fname, repo_type="dataset")
        print(f"[HC3] {domain}: {path}")
        n_q = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                n_q += 1
                for ans in j.get("human_answers", []) or []:
                    if ans and len(ans.strip()) >= 40:
                        rows.append(
                            {"text": ans.strip(), "generated": 0, "domain": domain, "source": "hc3"}
                        )
                for ans in j.get("chatgpt_answers", []) or []:
                    if ans and len(ans.strip()) >= 40:
                        rows.append(
                            {"text": ans.strip(), "generated": 1, "domain": domain, "source": "hc3"}
                        )
        print(f"  -> {n_q} questions, cumulative rows={len(rows)}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "hc3_small.csv"), index=False)
    print(f"[HC3] saved {len(df)} rows -> data/moe/hc3_small.csv")
    print(df["generated"].value_counts().to_dict(), df["domain"].value_counts().to_dict())
    return df


def download_modern_essays() -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "artnitolog/llm-generated-texts",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    t = pq.read_table(path).to_pylist()
    rows = []
    for r in t:
        prompt = str(r.get("prompt", ""))
        human = r.get("human")
        if human and len(str(human).strip()) >= 100:
            rows.append({"text": str(human).strip(), "generated": 0, "prompt": prompt, "model": "human"})
        for m in ENGLISH_MODELS:
            v = r.get(m)
            if v and len(str(v).strip()) >= 100 and len(str(v).strip().split()) >= 20:
                rows.append({"text": str(v).strip(), "generated": 1, "prompt": prompt, "model": m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "modern_essays.csv"), index=False)
    print(f"[ESSAYS] saved {len(df)} rows -> data/moe/modern_essays.csv")
    print(df["generated"].value_counts().to_dict(), df["model"].value_counts().to_dict())
    return df


def length_report():
    for f in ["hc3_small.csv", "modern_essays.csv"]:
        df = pd.read_csv(os.path.join(OUT, f))
        w = df["text"].astype(str).str.split().str.len()
        print(f"\n== {f}: n={len(df)} words mean={w.mean():.0f} median={w.median():.0f} "
              f"p10={w.quantile(0.1):.0f} p90={w.quantile(0.9):.0f} min={w.min()} max={w.max()}")
        print(f"   <=150w (SHORT): {(w <= 150).mean():.1%}  | 151-400w (LONG): {((w > 150) & (w <= 400)).mean():.1%}  | >400w (VERYLONG): {(w > 400).mean():.1%}")


if __name__ == "__main__":
    download_hc3()
    download_modern_essays()
    length_report()
    print("\nDone. Next: PYTHONPATH=. python3 moe/train_moe.py")
