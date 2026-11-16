"""Agrandissement de la toile par IA (« outpainting »), sans étirer l'image.

Deux moteurs :
- "local"  : LaMa (ONNX, 92 Mo, téléchargé une fois), gratuit et hors ligne.
             Très propre sur les fonds simples (studio, mur, sol).
- "openai" : API d'images d'OpenAI (clé API requise, payant à l'usage).
             Meilleur sur les décors complexes. L'image est envoyée à OpenAI.

Dans les deux cas, les pixels d'origine sont recollés à l'identique : seule la
zone ajoutée vient de l'IA.
"""

from __future__ import annotations

import base64
import io
import json
import ssl
import threading
import urllib.error
import urllib.request
import uuid

from PIL import Image, ImageFilter

from . import background as BG

LAMA_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/inpainting_lama/inpainting_lama_2025jan.onnx")
LAMA_FILE = "inpainting_lama_2025jan.onnx"
LAMA_SIZE = 512
WORK_MAX = 1024  # l'IA locale travaille sur une copie réduite (fond = peu de détails)

_lama_sess = None
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Outils communs
# --------------------------------------------------------------------------- #
def _composite_original(filled: Image.Image, img: Image.Image, ox: int, oy: int) -> Image.Image:
    """Recolle l'image d'origine sur le résultat de l'IA, avec un raccord adouci."""
    out = filled.convert("RGB").copy()
    w, h = img.size
    f = max(2, round(min(w, h) * 0.01))
    # on n'adoucit que les bords qui touchent une zone ajoutée
    edges = Image.new("L", (w, h), 255)
    nw, nh = out.size
    from PIL import ImageDraw
    d = ImageDraw.Draw(edges)
    if ox > 0:
        d.rectangle([0, 0, f, h], fill=0)
    if ox + w < nw:
        d.rectangle([w - f - 1, 0, w, h], fill=0)
    if oy > 0:
        d.rectangle([0, 0, w, f], fill=0)
    if oy + h < nh:
        d.rectangle([0, h - f - 1, w, h], fill=0)
    mask = edges.filter(ImageFilter.GaussianBlur(f / 2))
    out.paste(img.convert("RGB"), (ox, oy), mask)
    return out


def _prefill(img: Image.Image, size: tuple[int, int], ox: int, oy: int) -> Image.Image:
    """Toile de départ : l'image + ses bords prolongés (sert de base aux IA)."""
    w, h = img.size
    nw, nh = size
    src = img.convert("RGB")
    canvas = Image.new("RGB", size)
    canvas.paste(src.resize((nw, nh), Image.BILINEAR))
    if oy > 0:
        canvas.paste(src.crop((0, 0, w, 1)).resize((w, oy)), (ox, 0))
    if oy + h < nh:
        canvas.paste(src.crop((0, h - 1, w, h)).resize((w, nh - h - oy)), (ox, oy + h))
    if ox > 0:
        canvas.paste(canvas.crop((ox, 0, ox + 1, nh)).resize((ox, nh)), (0, 0))
    if ox + w < nw:
        canvas.paste(canvas.crop((ox + w - 1, 0, ox + w, nh)).resize((nw - w - ox, nh)), (ox + w, 0))
    canvas.paste(src, (ox, oy))
    return canvas


# --------------------------------------------------------------------------- #
# IA locale : LaMa
# --------------------------------------------------------------------------- #
def lama_ready() -> bool:
    return (BG.models_dir() / LAMA_FILE).exists()


def lama_download(progress=None):
    return BG.fetch(LAMA_URL, BG.models_dir() / LAMA_FILE, progress)


def _lama_session():
    global _lama_sess
    with _lock:
        if _lama_sess is None:
            import onnxruntime as ort

            try:
                ort.disable_telemetry_events()
            except Exception:
                pass
            import os

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 2)  # garde l'interface fluide
            _lama_sess = ort.InferenceSession(str(lama_download()), opts, providers=["CPUExecutionProvider"])
        return _lama_sess


def _lama(tile: Image.Image, mask: Image.Image) -> Image.Image:
    import numpy as np

    sess = _lama_session()
    w, h = tile.size
    x = np.asarray(tile.convert("RGB").resize((LAMA_SIZE, LAMA_SIZE), Image.BICUBIC), dtype=np.float32)
    x = (x.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    m = (np.asarray(mask.resize((LAMA_SIZE, LAMA_SIZE), Image.NEAREST)) > 127).astype(np.float32)[None, None]
    out = sess.run(None, {"image": x, "mask": m})[0][0].transpose(1, 2, 0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)


def _extend_top(cur: Image.Image, amount: int) -> Image.Image:
    """Ajoute `amount` pixels en haut, par petites bandes successives.

    LaMa remplit mal les grandes zones collées au bord : chaque bande (≤ 15 %
    de la tuile) est donc placée au milieu de la tuile, entre un miroir du
    contexte au-dessus et l'image en dessous, et pré-remplie avec le bord."""
    import numpy as np

    cur = cur.convert("RGB")
    total = amount
    while amount > 0:
        w, h = cur.size
        win = max(64, min(w, int(h * 1.4)))
        step = max(1, min(amount, round(win * 0.15)))
        ov = max(2, min(h // 4, round(step * 0.4)))  # lignes connues régénérées puis fondues
        gen = step + ov
        mh = max(4, (win - gen) // 3)
        c = win - step - mh  # contexte connu (inclut les ov lignes)
        xs = [0] if w <= win else list(range(0, w - win, int(win * 0.75))) + [w - win]
        acc = np.zeros((gen, w, 3), np.float32)
        wsum = np.zeros((gen, w, 1), np.float32)
        for x0 in xs:
            ctx = cur.crop((x0, 0, x0 + win, c))
            tile = Image.new("RGB", (win, win))
            tile.paste(ctx.crop((0, 0, win, mh)).transpose(Image.FLIP_TOP_BOTTOM), (0, 0))
            tile.paste(cur.crop((x0, 0, x0 + win, 1)).resize((win, step)), (0, mh))
            tile.paste(ctx, (0, mh + step))
            mask = Image.new("L", (win, win), 0)
            mask.paste(255, (0, mh, win, mh + gen))
            out = np.asarray(_lama(tile, mask).crop((0, mh, win, mh + gen)), np.float32)
            ramp = np.ones((1, win, 1), np.float32)
            if len(xs) > 1:
                f = max(2, int(win * 0.12))
                ramp[0, :f, 0] = np.linspace(0.05, 1, f)
                ramp[0, -f:, 0] = np.linspace(1, 0.05, f)
            acc[:, x0:x0 + win] += out * ramp
            wsum[:, x0:x0 + win] += ramp
        band = acc / wsum
        known = np.asarray(cur.crop((0, 0, w, ov)), np.float32)
        # corrige la dérive de couleur : la partie régénérée doit coller à l'original
        band += (known.mean(axis=(0, 1)) - band[step:].mean(axis=(0, 1)))[None, None, :]
        # fondu entre la partie régénérée et l'original sur les ov lignes
        a = np.linspace(0, 1, ov, dtype=np.float32)[:, None, None]
        blended = band[step:] * (1 - a) + known * a
        top = np.concatenate([band[:step], blended], axis=0).clip(0, 255).astype(np.uint8)
        new = Image.new("RGB", (w, h + step))
        new.paste(cur, (0, step))
        new.paste(Image.fromarray(top), (0, 0))
        cur = new
        amount -= step
    return _flatten_row_drift(cur, total)


def _flatten_row_drift(cur: Image.Image, added: int) -> Image.Image:
    """Supprime les bandes de teinte entre les passes de l'IA : la luminosité/couleur
    moyenne de chaque ligne ajoutée est ramenée vers celle du bord d'origine.
    Le décalage est lissé, donc les motifs fins (joints, rayures) sont conservés."""
    import numpy as np

    if added <= 0:
        return cur
    a = np.asarray(cur, np.float32).copy()
    k = max(1, min(12, a.shape[0] - added))
    target = a[added:added + k].mean(axis=(0, 1))
    rows = a[:added].mean(axis=1)  # (added, 3)
    off = rows - target
    sigma = max(3.0, added * 0.03)
    x = np.arange(-int(3 * sigma), int(3 * sigma) + 1, dtype=np.float32)
    ker = np.exp(-x ** 2 / (2 * sigma ** 2))
    ker /= ker.sum()
    pad = len(x) // 2
    sm = np.stack([np.convolve(np.pad(off[:, c], pad, mode="edge"), ker, mode="valid") for c in range(3)], 1)
    a[:added] -= sm[:, None, :]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


_ROT = {"top": (None, None), "bottom": (Image.ROTATE_180, Image.ROTATE_180),
        "left": (Image.ROTATE_270, Image.ROTATE_90), "right": (Image.ROTATE_90, Image.ROTATE_270)}


def outpaint_local(img: Image.Image, size: tuple[int, int], ox: int, oy: int) -> Image.Image:
    w, h = img.size
    nw, nh = size
    f = min(1.0, WORK_MAX / max(nw, nh))
    small = img.convert("RGB").resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    pads = {"top": round(oy * f), "bottom": round((nh - h - oy) * f),
            "left": round(ox * f), "right": round((nw - w - ox) * f)}
    cur = small
    for side in ("top", "bottom", "left", "right"):
        if pads[side] <= 0:
            continue
        before, after = _ROT[side]
        if before is not None:
            cur = cur.transpose(before)
        cur = _extend_top(cur, pads[side])
        if after is not None:
            cur = cur.transpose(after)
    return _composite_original(cur.resize((nw, nh), Image.LANCZOS), img, ox, oy)


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
OPENAI_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]
OPENAI_PROMPT = ("Outpainting: extend the existing photo into the empty (transparent) area so the result "
                 "looks like one seamless, natural photograph. Continue the background, surfaces, lighting, "
                 "shadows and perspective of the original. Do not add any new object, product, text, logo "
                 "or person. Do not change the existing part of the image.")


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (name, data) in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                     f'Content-Type: image/png\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def outpaint_openai(img: Image.Image, size: tuple[int, int], ox: int, oy: int, api_key: str,
                    model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    if not api_key:
        raise RuntimeError("Clé API OpenAI manquante (volet « Changer le ratio »).")
    nw, nh = size
    # format accepté par l'API le plus proche, image centrée dedans
    tw, th = min(OPENAI_SIZES, key=lambda s: abs(s[0] / s[1] - nw / nh))
    scale = min(tw / nw, th / nh)
    sw, sh = round(nw * scale), round(nh * scale)
    lx, ly = (tw - sw) // 2, (th - sh) // 2

    base = Image.new("RGB", (tw, th))
    base.paste(_prefill(img, size, ox, oy).resize((sw, sh), Image.LANCZOS), (lx, ly))
    mask = Image.new("RGBA", (tw, th), (0, 0, 0, 0))  # transparent = zone à générer
    keep = (lx + round(ox * scale), ly + round(oy * scale),
            lx + round((ox + img.width) * scale), ly + round((oy + img.height) * scale))
    mask.paste((0, 0, 0, 255), keep)

    fields = {"model": model, "prompt": OPENAI_PROMPT, "size": f"{tw}x{th}", "quality": quality, "n": "1",
              "input_fidelity": "high"}
    body, ctype = _multipart(fields, {"image": ("image.png", _png(base)), "mask": ("mask.png", _png(mask))})
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    def call(b, ct):
        req = urllib.request.Request("https://api.openai.com/v1/images/edits", data=b, method="POST",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": ct})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            return json.loads(r.read())

    try:
        data = call(body, ctype)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        if e.code == 400 and "input_fidelity" in msg:  # modèle qui ne prend pas ce réglage
            fields.pop("input_fidelity")
            body, ctype = _multipart(fields, {"image": ("image.png", _png(base)),
                                              "mask": ("mask.png", _png(mask))})
            data = call(body, ctype)
        else:
            try:
                msg = json.loads(msg)["error"]["message"]
            except Exception:
                pass
            raise RuntimeError(f"OpenAI ({e.code}) : {msg}") from None
    result = Image.open(io.BytesIO(base64.b64decode(data["data"][0]["b64_json"]))).convert("RGB")
    if result.size != (tw, th):
        result = result.resize((tw, th), Image.LANCZOS)
    filled = result.crop((lx, ly, lx + sw, ly + sh)).resize((nw, nh), Image.LANCZOS)
    return _composite_original(filled, img, ox, oy)


def outpaint(img: Image.Image, size: tuple[int, int], ox: int, oy: int, engine: str,
             api_key: str = "", model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    if engine == "openai":
        return outpaint_openai(img, size, ox, oy, api_key, model, quality)
    return outpaint_local(img, size, ox, oy)
