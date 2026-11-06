#!/usr/bin/env bash
set -euo pipefail
# run_all.sh — One-command setup → train → test → API → UI
#
# Usage:
#   ./run_all.sh                 # full: install + train + test + API + UI
#   ./run_all.sh --skip-train    # skip training (use existing models/)
#   ./run_all.sh --skip-install  # skip pip install (use existing venv/)
#   ./run_all.sh --no-api --no-ui  # only train + test
#   ./run_all.sh --help
#
# Ports: API 8001, Streamlit 8501 (kill existing with: pkill -f "streamlit run|test/api.py")
# Logs:  test/results/, api.log, streamlit.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="venv/bin/python"
SKIP_TRAIN=0; SKIP_INSTALL=0; NO_API=0; NO_UI=0
for a in "$@"; do case "$a" in
  --skip-train) SKIP_TRAIN=1;;
  --skip-install) SKIP_INSTALL=1;;
  --no-api) NO_API=1;;
  --no-ui) NO_UI=1;;
  --help|-h) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
  *) echo "Unknown arg: $a (try --help)"; exit 1;;
esac; done

echo "=== [1/6] Project root: $ROOT ==="
ls -d data src test app models 2>/dev/null || echo "  (missing dirs — clone repo first?)"
echo "Data files:"
ls -lh data/train.csv data/test.csv 2>/dev/null || echo "  WARNING: data/train.csv or data/test.csv missing"

if [ "$SKIP_INSTALL" -eq 0 ]; then
  echo "=== [2/6] venv + dependencies ==="
  if [ ! -x "$PY" ]; then python3 -m venv venv; fi
  venv/bin/pip install -q --upgrade pip
  venv/bin/pip install -q -r requirements.txt "langchain-core>=0.3" "fastapi>=0.110" reportlab
  echo "  venv: $($PY --version)  ✓"
else
  echo "=== [2/6] venv (skipped install) ==="
  test -x "$PY" || { echo "  ERROR: venv missing. Run without --skip-install."; exit 1; }
  echo "  using $($PY --version)"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
  echo "=== [3/6] Train (train_baseline.py, 11 steps, ~30-60s) ==="
  $PY train_baseline.py
else
  echo "=== [3/6] Train (skipped — using existing models/) ==="
  ls -lh models/*.pkl
fi

echo "=== [4/6] Test (examples + 5k held-out) ==="
mkdir -p test/results
$PY test/evaluate_examples.py
$PY test/predict.py --csv data/test.csv --model lr --sample 5000 --out test/results
$PY test/terminal_demo.py
echo "  test outputs → test/results/predictions.csv + metrics.json"

if [ "$NO_API" -eq 0 ]; then
  echo "=== [5/6] API (FastAPI :8001) ==="
  pkill -f "test/api.py" 2>/dev/null || true
  nohup $PY test/api.py > api.log 2>&1 &
  echo "  API pid $! → http://localhost:8001/docs (log: api.log)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if $PY -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)" 2>/dev/null; then
      echo "  API health: OK ✓"; break
    fi
    sleep 1
  done
  $PY -c "import httpx,json; print(httpx.post('http://127.0.0.1:8001/predict', json={'text':'In todays rapidly evolving digital landscape, research suggests leveraging data-driven approaches facilitates decision making. Furthermore, it is crucial to note that understanding various methodologies can lead to improved outcomes.','model':'lr'}).json())"
else
  echo "=== [5/6] API (skipped) ==="
fi

if [ "$NO_UI" -eq 0 ]; then
  echo "=== [6/6] Web UI (Streamlit :8501) ==="
  pkill -f "streamlit run" 2>/dev/null || true
  nohup venv/bin/streamlit run app/app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
  echo "  Streamlit pid $! → http://localhost:8501 (log: streamlit.log)"
else
  echo "=== [6/6] Web UI (skipped) ==="
fi

echo ""
echo "Done ✓"
echo "  API:      http://localhost:8001/docs"
echo "  Streamlit:http://localhost:8501"
echo "  Test CSV: test/results/predictions.csv (5000 rows)"
echo "  Stop:     pkill -f 'test/api.py|streamlit run'"
