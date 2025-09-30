#!/usr/bin/env bash
set -euo pipefail

# Ensure certifi is present and point SSL to it (helps on macOS python.org builds)
python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("certifi") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "certifi"])
PY
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"

python -m src.train \
  --config configs/baseline.yaml \
  --exp-name resnet20_cosine

