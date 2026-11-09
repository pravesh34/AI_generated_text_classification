# Web deployment (Vercel)

Permanent production: **https://moe-text-classifier-pravesh12.vercel.app**
Legacy alias: https://web-pravesh12.vercel.app

## What is deployed

| Part | Location | Notes |
|---|---|---|
| Static site | `web/index.html` (mirrored to `web/api/index.html`) | Results, figures, reports, run guide. Served by the function at `/` — a single Python function handles all routes, so `api/predict.py` serves both site (`GET /`, `/figures/*`, `/reports/*`) and API (`/api/predict`) |
| Live inference API | `web/api/predict.py` → `POST /api/predict` | Vercel Python serverless (3.12), 1024 MB, 60s |
| Models | `web/api/models/` (bundled, ~10 MB) | Copied from `../models/`; GitHub cloud fallback via `MODEL_BASE_URL` |
| Python deps | `web/pyproject.toml` | sklearn pinned `==1.9.1` (pickle compat); `requires-python >=3.11` |

The `.pkl` files in GitHub (`models/`, `models/moe/`) are the cloud source of truth.
`web/api/models/` is a bundled copy so Vercel cold-starts without fetching.
`src.preprocessing` is referenced inside the pickles — `web/src/` + `web/moe/` are bundled for unpickling.

## Test

```bash
curl -X POST "https://moe-text-classifier-pravesh12.vercel.app/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I think sports are great because they keep kids busy after school"}'
# → {"label": "Human-Written", "prob_ai": 0.0054, "expert": "short", ...}

curl "https://moe-text-classifier-pravesh12.vercel.app/api/predict"   # health
```

## Redeploy

```bash
cd web
cp ../models/*.pkl api/models/ && cp ../models/moe/*.pkl ../models/moe/*.json api/models/moe/
cp -r ../src web/ 2>/dev/null || true   # refresh src/moe packages if changed
npx vercel deploy --prod --scope pravesh12
```

Deployment Protection (Vercel Authentication) is disabled for public access.
Anonymous temporary deployments (`*.tmp.now`) expire in ~60 min unless claimed — use the production URL above.
