#!/usr/bin/env sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -n "${FEETFIT_PYTHON_EXE:-}" ]; then
  exec "$FEETFIT_PYTHON_EXE" -u -B scripts/run_local_server.py
elif [ -x .venv/bin/python ]; then
  exec .venv/bin/python -u -B scripts/run_local_server.py
else
  exec python3 -u -B scripts/run_local_server.py
fi
