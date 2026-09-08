#!/bin/zsh
set -eu
studio_app_dir="${0:A:h}"
cd "$studio_app_dir"
if ! command -v python3.12 >/dev/null 2>&1; then
  print 'Native Python 3.12 is required. See START_HERE.md, then run setup again.'
  exit 1
fi
python3.12 -B scripts/setup.py --initialize-ssd --download-models
../runtime/venv/bin/python -B scripts/verify_models.py --model baseline
print 'Setup and real local model checks passed. Open Start Studio.command.'
