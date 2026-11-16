#!/bin/bash
# Double-cliquez sur ce fichier dans le Finder pour lancer Numbered - Batch Cooking.
# Au premier lancement, il installe ce qu'il faut dans un dossier .venv (1 à 2 min).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Premier lancement : installation des dépendances…"
  python3 -m venv .venv || { echo "Python 3 introuvable. Installez-le avec : brew install python"; read -r; exit 1; }
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt || { echo "Échec de l'installation."; read -r; exit 1; }
fi
exec .venv/bin/python main.py "$@"
