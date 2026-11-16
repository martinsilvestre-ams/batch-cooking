#!/bin/bash
# Construit "Image Batch.app" dans dist/ (application Mac autonome, sans Python requis).
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
.venv/bin/pip install pyinstaller >/dev/null
.venv/bin/pyinstaller --noconfirm --windowed --name "Image Batch" \
  --osx-bundle-identifier com.imagebatch.app main.py
echo "Terminé : dist/Image Batch.app (glissez-la dans Applications)"
