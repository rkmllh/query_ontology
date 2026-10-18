#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
  source "$ROOT_DIR/venv/bin/activate"
fi

python -m query_ontology create-dict
python -m query_ontology create-embedding
