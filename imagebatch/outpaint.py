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
def _composite_original(filled: Image.Image, img: Image.Image, ox: int, oy: int,
                        feather: float = 0.03) -> Image.Image:
    """Recolle l'image d'origine sur le résultat de l'IA, avec un raccord adouci."""
    out = filled.convert("RGB").copy()
    w, h = img.size
    f = max(16, round(min(w, h) * feather))  # fondu de raccord (≈ 20-40 px)
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
        top = np.concatenate([band[:step], blended], axis=0).clip(0, 255).round().astype(np.uint8)
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
    # contraste : l'IA « lisse » les motifs fins ; on remet l'écart-type du bord d'origine
    t_std = a[added:added + k].std(axis=1).mean(axis=0) + 1e-3  # (3,)
    r_mean = a[:added].mean(axis=1, keepdims=True)
    r_std = a[:added].std(axis=1)  # (added, 3)
    r_std = np.stack([np.convolve(np.pad(r_std[:, c], pad, mode="edge"), ker, mode="valid")
                      for c in range(3)], 1)
    gain = np.clip(t_std[None, :] / (r_std + 1e-3), 1.0, 1.6)[:, None, :]
    a[:added] = r_mean + (a[:added] - r_mean) * gain
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _noise_sigma(a) -> float:
    """Grain de l'image (écart-type du bruit fin), estimé de façon robuste (MAD)
    sur l'écart à une version floutée. Plafonné pour ne pas exagérer."""
    import numpy as np

    if a.shape[0] < 5 or a.shape[1] < 5:
        return 0.0
    lum = Image.fromarray(a.mean(axis=2).clip(0, 255).round().astype(np.uint8))
    hp = np.asarray(lum, np.float32) - np.asarray(lum.filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    return float(min(10.0, 1.4826 * np.median(np.abs(hp - np.median(hp))) * 1.15))


def _grain(shape, sigma: float, seed: int = 0):
    """Grain photo propre (bruit gaussien légèrement adouci), d'écart-type ≈ sigma."""
    import numpy as np

    n = np.random.default_rng(seed).normal(0, 1, (shape[0] + 2, shape[1] + 2)).astype(np.float32)
    n = (4 * n[1:-1, 1:-1] + n[:-2, 1:-1] + n[2:, 1:-1] + n[1:-1, :-2] + n[1:-1, 2:]) / 8
    return n * (sigma / max(float(n.std()), 1e-6))


def _smooth_extend_top(cur: Image.Image, amount: int, seed: int = 0) -> Image.Image:
    """Ajoute `amount` lignes en haut sans recopier de pixels : la couleur du bord est
    lissée (les défauts de compression disparaissent), prolongée en fondu vers la
    teinte moyenne du bord, puis un grain léger et propre est ajouté."""
    import numpy as np

    cur = cur.convert("RGB")
    w, h = cur.size
    a = np.asarray(cur, np.float32)
    hb = max(4, min(h, round(h * 0.08)))
    band = a[:hb]
    # profil de couleur par colonne, pondéré vers le bord, puis très lissé horizontalement
    wts = np.exp(-np.arange(hb, dtype=np.float32) / max(1.0, hb / 3))[:, None, None]
    prof = (band * wts).sum(0) / wts.sum()
    sig = max(2.0, w * 0.04)
    x = np.arange(-int(3 * sig), int(3 * sig) + 1, dtype=np.float32)
    ker = np.exp(-x ** 2 / (2 * sig ** 2))
    ker /= ker.sum()
    pad = len(x) // 2
    prof = np.stack([np.convolve(np.pad(prof[:, c], pad, mode="reflect" if w > pad else "edge"), ker, "valid")
                     for c in range(3)], 1)
    flat = band.reshape(-1, 3)
    lo, hi = np.percentile(flat, [10, 90], axis=0)
    far = np.where((flat >= lo) & (flat <= hi), flat, np.nan)
    far = np.nan_to_num(np.nanmean(far, axis=0), nan=0.0) if flat.size else flat.mean(axis=0)  # moyenne robuste
    d = np.arange(amount, 0, -1, dtype=np.float32)[:, None, None]  # distance au bord
    t = 1 - np.exp(-d / max(8.0, h * 0.25))
    top = prof[None] * (1 - t) + far[None, None] * t
    g = _noise_sigma(band)
    if g > 0.3:
        top = top + _grain((amount, w), g, seed)[:, :, None]
    out = Image.new("RGB", (w, h + amount))
    out.paste(Image.fromarray(top.clip(0, 255).round().astype(np.uint8)), (0, 0))
    out.paste(cur, (0, amount))
    return out


def smooth_fill(img: Image.Image, size: tuple[int, int], ox: int, oy: int) -> Image.Image:
    """Fond généré sans étirement (rapide, hors ligne), voir _smooth_extend_top."""
    w, h = img.size
    nw, nh = size
    pads = {"top": oy, "bottom": nh - h - oy, "left": ox, "right": nw - w - ox}
    cur = img.convert("RGB")
    for i, side in enumerate(("top", "bottom", "left", "right")):
        if pads[side] <= 0:
            continue
        before, after = _ROT[side]
        if before is not None:
            cur = cur.transpose(before)
        cur = _smooth_extend_top(cur, pads[side], seed=i)
        if after is not None:
            cur = cur.transpose(after)
    return _composite_original(cur, img, ox, oy, feather=0.06)


def _is_plain(img: Image.Image, ox: int, oy: int, size: tuple[int, int]) -> bool:
    """Vrai si les bords à prolonger sont un fond quasi uni (studio, mur…) :
    le fond généré lisse y est plus propre que l'IA."""
    import numpy as np

    w, h = img.size
    nw, nh = size
    a = np.asarray(img.convert("RGB").resize((min(w, 512), min(h, 512)), Image.BOX), np.float32)
    a = np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32)
    hh, ww = a.shape[:2]
    k = max(2, round(min(hh, ww) * 0.08))
    bands = []
    if oy > 0:
        bands.append(a[:k])
    if oy + h < nh:
        bands.append(a[-k:])
    if ox > 0:
        bands.append(a[:, :k])
    if ox + w < nw:
        bands.append(a[:, -k:])
    if not bands:
        return False
    # variation lente le long du bord autorisée (dégradé), mais pas de motif
    for b in bands:
        b = b if b.shape[0] <= b.shape[1] else b.transpose(1, 0, 2)
        col = b.mean(axis=0)
        if float(np.abs(np.diff(col, axis=0)).mean()) > 1.5 or float((b - col[None]).std()) > 6:
            return False
    return True


_ROT = {"top": (None, None), "bottom": (Image.ROTATE_180, Image.ROTATE_180),
        "left": (Image.ROTATE_270, Image.ROTATE_90), "right": (Image.ROTATE_90, Image.ROTATE_270)}


def outpaint_local(img: Image.Image, size: tuple[int, int], ox: int, oy: int) -> Image.Image:
    import numpy as np

    w, h = img.size
    nw, nh = size
    if _is_plain(img, ox, oy, size):
        return smooth_fill(img, size, ox, oy)
    f = min(1.0, WORK_MAX / max(nw, nh))
    small = img.convert("RGB").resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    grain = _noise_sigma(np.asarray(small, np.float32))
    # débruitage : les artefacts de compression ne doivent pas être recopiés par l'IA
    small = small.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
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
    big = cur.resize((nw, nh), Image.LANCZOS)
    if grain > 0.3:
        n = _grain((nh, nw), grain, 0)
        big = Image.fromarray((np.asarray(big, np.float32) + n[:, :, None]).clip(0, 255).round().astype(np.uint8))
    return _composite_original(big, img, ox, oy)


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


def _openai_edit(base: Image.Image, mask: Image.Image, prompt: str, api_key: str,
                 model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    """Appelle images/edits : base (RGB) + masque (RGBA, transparent = à générer)."""
    if not api_key:
        raise RuntimeError("Clé API OpenAI manquante (volet « IA OpenAI »).")
    tw, th = base.size
    fields = {"model": model, "prompt": prompt, "size": f"{tw}x{th}", "quality": quality, "n": "1",
              "input_fidelity": "high"}
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    def call():
        body, ctype = _multipart(fields, {"image": ("image.png", _png(base)), "mask": ("mask.png", _png(mask))})
        req = urllib.request.Request("https://api.openai.com/v1/images/edits", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": ctype})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            return json.loads(r.read())

    try:
        data = call()
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        if e.code == 400 and "input_fidelity" in msg:  # modèle qui ne prend pas ce réglage
            fields.pop("input_fidelity")
            data = call()
        else:
            try:
                msg = json.loads(msg)["error"]["message"]
            except Exception:
                pass
            raise RuntimeError(f"OpenAI ({e.code}) : {msg}") from None
    result = Image.open(io.BytesIO(base64.b64decode(data["data"][0]["b64_json"]))).convert("RGB")
    return result if result.size == (tw, th) else result.resize((tw, th), Image.LANCZOS)


def _letterbox(size: tuple[int, int]):
    """Format accepté par l'API le plus proche, et placement de la toile dedans."""
    nw, nh = size
    tw, th = min(OPENAI_SIZES, key=lambda s: abs(s[0] / s[1] - nw / nh))
    scale = min(tw / nw, th / nh)
    sw, sh = round(nw * scale), round(nh * scale)
    return tw, th, scale, sw, sh, (tw - sw) // 2, (th - sh) // 2


def outpaint_openai(img: Image.Image, size: tuple[int, int], ox: int, oy: int, api_key: str,
                    model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    nw, nh = size
    tw, th, scale, sw, sh, lx, ly = _letterbox(size)
    base = Image.new("RGB", (tw, th))
    base.paste(_prefill(img, size, ox, oy).resize((sw, sh), Image.LANCZOS), (lx, ly))
    mask = Image.new("RGBA", (tw, th), (0, 0, 0, 0))  # transparent = zone à générer
    keep = (lx + round(ox * scale), ly + round(oy * scale),
            lx + round((ox + img.width) * scale), ly + round((oy + img.height) * scale))
    mask.paste((0, 0, 0, 255), keep)
    result = _openai_edit(base, mask, OPENAI_PROMPT, api_key, model, quality)
    filled = result.crop((lx, ly, lx + sw, ly + sh)).resize((nw, nh), Image.LANCZOS)
    return _composite_original(filled, img, ox, oy)


BG_PROMPT = ("Replace only the background (the transparent area) with: {desc}. Keep the product exactly "
             "as it is: same shape, colors, labels and text. Make it look like a real professional product "
             "photo with consistent lighting, perspective and a natural contact shadow under the product. "
             "Do not add any other product, text or logo.")


def generate_background(img: Image.Image, subject_mask: Image.Image, description: str, api_key: str,
                        model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    """Nouveau fond généré par OpenAI d'après une description ; le produit
    (masque de détourage) est recollé à l'identique par-dessus."""
    w, h = img.size
    tw, th, scale, sw, sh, lx, ly = _letterbox((w, h))
    rgb = img.convert("RGB")
    base = Image.new("RGB", (tw, th))
    base.paste(rgb.resize((sw, sh), Image.LANCZOS), (lx, ly))
    # zone à générer = tout sauf le produit (légèrement élargi pour garder son contour)
    keep = subject_mask.resize((sw, sh), Image.BILINEAR).point(lambda v: 255 if v > 100 else 0)
    keep = keep.filter(ImageFilter.MaxFilter(5))
    mask = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    mask.paste((0, 0, 0, 255), (lx, ly), keep)
    desc = description.strip() or "a clean, softly lit professional studio background"
    result = _openai_edit(base, mask, BG_PROMPT.format(desc=desc), api_key, model, quality)
    out = result.crop((lx, ly, lx + sw, ly + sh)).resize((w, h), Image.LANCZOS)
    out.paste(rgb, (0, 0), subject_mask.filter(ImageFilter.GaussianBlur(max(1, max(w, h) / 1000))))
    return out


def outpaint(img: Image.Image, size: tuple[int, int], ox: int, oy: int, engine: str,
             api_key: str = "", model: str = "gpt-image-1", quality: str = "high") -> Image.Image:
    if engine == "openai":
        return outpaint_openai(img, size, ox, oy, api_key, model, quality)
    return outpaint_local(img, size, ox, oy)


# --------------------------------------------------------------------------- #
# Retouche d'une zone (gommer une trace, un défaut…)
# --------------------------------------------------------------------------- #
RETOUCH_PROMPT = ("Remove any mark, stain, seam, artifact or object in the transparent area and fill it so it "
                  "blends seamlessly with the surrounding photo (same surface, texture, lighting). Do not add "
                  "anything new.")


def _retouch_window(img_size, box, min_side=64):
    """Zone de travail carrée autour de la sélection (avec du contexte)."""
    w, h = img_size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    side = int(min(max(w, h), max(min_side, bw * 2.2, bh * 2.2)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0 = int(min(max(0, cx - side / 2), max(0, w - side)))
    wy0 = int(min(max(0, cy - side / 2), max(0, h - side)))
    return wx0, wy0, min(w, wx0 + side), min(h, wy0 + side)


def retouch(img: Image.Image, rects, engine: str = "local", api_key: str = "",
            model: str = "gpt-image-1", quality: str = "high", prompt: str = "") -> Image.Image:
    """Regénère par IA les zones sélectionnées. rects : (x, y, l, h) relatifs (0..1)."""
    import numpy as np

    has_alpha = img.mode in ("RGBA", "LA")
    alpha = img.getchannel("A") if has_alpha else None
    out = img.convert("RGB")
    w, h = out.size
    for (rx, ry, rw, rh) in rects:
        box = (max(0, int(rx * w)), max(0, int(ry * h)),
               min(w, int(np.ceil((rx + rw) * w))), min(h, int(np.ceil((ry + rh) * h))))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        win = _retouch_window((w, h), box)
        tile = out.crop(win)
        tw, th = tile.size
        lb = (box[0] - win[0], box[1] - win[1], box[2] - win[0], box[3] - win[1])
        if engine == "openai":
            size = OPENAI_SIZES[0]
            scale = min(size[0] / tw, size[1] / th)
            sw, sh = max(1, round(tw * scale)), max(1, round(th * scale))
            base = Image.new("RGB", size)
            base.paste(tile.resize((sw, sh), Image.LANCZOS))
            m = Image.new("RGBA", size, (0, 0, 0, 255))
            m.paste((0, 0, 0, 0), tuple(round(v * scale) for v in lb))  # transparent = à refaire
            res = _openai_edit(base, m, (prompt.strip() + ". " if prompt.strip() else "") + RETOUCH_PROMPT,
                               api_key, model, quality)
            filled = res.crop((0, 0, sw, sh)).resize((tw, th), Image.LANCZOS)
        else:
            mask = Image.new("L", (tw, th), 0)
            mask.paste(255, lb)
            # sélection légèrement élargie : la trace ne doit pas déborder
            grow = max(2, round(max(tw, th) / 128))
            mask = mask.filter(ImageFilter.MaxFilter(2 * (grow // 2) + 1))
            src = tile.filter(ImageFilter.MedianFilter(3))
            filled = _lama(src, mask)
            g = _noise_sigma(np.asarray(tile, np.float32))
            if g > 0.3:
                n = _grain((th, tw), g, 0)
                filled = Image.fromarray((np.asarray(filled, np.float32) + n[:, :, None])
                                         .clip(0, 255).round().astype(np.uint8))
        # on ne remplace que la sélection, avec un bord adouci
        f = max(2, round(min(lb[2] - lb[0], lb[3] - lb[1]) * 0.08))
        soft = Image.new("L", (tw, th), 0)
        soft.paste(255, (max(0, lb[0] - f), max(0, lb[1] - f), min(tw, lb[2] + f), min(th, lb[3] + f)))
        soft = soft.filter(ImageFilter.GaussianBlur(f / 2))
        tile.paste(filled, (0, 0), soft)
        out.paste(tile, win[:2])
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out
