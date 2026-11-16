#!/bin/bash
# Construit "Numbered - Batch Cooking.app" dans dist/ (application Mac autonome, sans Python requis).
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
.venv/bin/pip install pyinstaller >/dev/null
.venv/bin/pyinstaller --noconfirm --windowed --name "Numbered - Batch Cooking" \
  --osx-bundle-identifier com.numbered.batchcooking main.py
echo "Terminé : dist/Numbered - Batch Cooking.app (glissez-la dans Applications)"
