#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("certifi") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "certifi"])
PY
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"

python -m src.eval \
  --checkpoint checkpoints/resnet20_cosine/best.pt
i
