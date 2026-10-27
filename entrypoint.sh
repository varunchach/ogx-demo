#!/bin/bash
set -euo pipefail

exec uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT:-8000}" \
  --workers 1 \
  --log-level info
