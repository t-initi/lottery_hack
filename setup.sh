#!/usr/bin/env bash
# Create the venv and install all Python dependencies.
# Run once:  bash setup.sh
# Activate:  source .venv/bin/activate

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo "==> Creating virtual environment in ${VENV_DIR}/"
"$PYTHON" -m venv "$VENV_DIR"

echo "==> Upgrading pip"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

echo "==> Installing requirements"
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt

echo ""
echo "Done. To activate:"
echo "  source .venv/bin/activate"
echo ""
echo "Then start the backend:"
echo "  uvicorn engine.server:app --reload --port 8001"
echo ""
echo "Fetch latest draw data:"
echo "  python scripts/fetch_draws.py"
