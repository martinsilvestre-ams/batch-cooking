"""Détourage local (sans internet, hors téléchargement initial du modèle).

Utilise des modèles ONNX de segmentation (ceux publiés par le projet rembg)
exécutés avec onnxruntime. Le modèle est téléchargé une seule fois dans
~/Library/Application Support/ImageBatch/models (ou ~/.imagebatch/models).
"""

from __future__ import annotations

import os
import ssl
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

_BASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"

# clé -> (libellé, fichier, taille d'entrée, moyenne, écart-type, normalisation)
MODELS = {
    "isnet": ("Rapide (ISNet, 170 Mo)", "isnet-general-use.onnx", 1024,
              (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), "minmax"),
    "birefnet": ("Précis (BiRefNet, 220 Mo)", "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", 1024,
                 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "sigmoid"),
}

_sessions: dict = {}
_lock = threading.Lock()


def available() -> Optional[str]:
    """Renvoie None si le détourage est possible, sinon un message d'erreur."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return f"Module manquant ({exc}). Lancez : .venv/bin/pip install -r requirements.txt"
    return None


def models_dir() -> Path:
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "ImageBatch" / "models"
    else:
        d = Path(os.environ.get("IMAGEBATCH_MODELS", Path.home() / ".imagebatch" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(key: str) -> Path:
    return models_dir() / MODELS[key][1]


def is_downloaded(key: str) -> bool:
    return model_path(key).exists()


def fetch(url: str, dst: Path, progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Télécharge url vers dst (une seule fois)."""
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".part")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "BatchCooking"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    tmp.replace(dst)
    return dst


def download(key: str, progress: Optional[Callable[[int, int], None]] = None) -> Path:
    return fetch(_BASE + MODELS[key][1], model_path(key), progress)


def _session(key: str):
    with _lock:
        if key not in _sessions:
            import onnxruntime as ort

            try:
                ort.disable_telemetry_events()
            except Exception:
                pass
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 2)  # garde l'interface fluide
            _sessions[key] = ort.InferenceSession(
                str(download(key)), opts, providers=["CPUExecutionProvider"])
        return _sessions[key]


def compute_mask(img: Image.Image, key: str = "isnet") -> Image.Image:
    """Masque du sujet principal (L, 0 = fond, 255 = sujet), à la taille de img."""
    import numpy as np

    _label, _file, size, mean, std, norm = MODELS[key]
    sess = _session(key)
    rgb = img.convert("RGB")
    x = np.asarray(rgb.resize((size, size), Image.BILINEAR), dtype=np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    if norm == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-out))
    else:
        mi, ma = float(out.min()), float(out.max())
        out = (out - mi) / max(ma - mi, 1e-6)
    m = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "L")
    return m.resize(img.size, Image.BILINEAR)
