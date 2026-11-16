"""Traitement d'images (sans interface) : recadrage, redimensionnement,
retouches et conversion. Utilisé par l'interface et testable seul."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

try:  # Prise en charge optionnelle des photos iPhone (HEIC)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except Exception:  # pragma: no cover
    HEIF_OK = False

INPUT_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif",
}
if HEIF_OK:
    INPUT_EXTENSIONS |= {".heic", ".heif"}

# Format de sortie -> (format Pillow, extension)
OUTPUT_FORMATS = {
    "Identique": (None, None),
    "JPEG": ("JPEG", ".jpg"),
    "PNG": ("PNG", ".png"),
    "WEBP": ("WEBP", ".webp"),
    "TIFF": ("TIFF", ".tif"),
    "BMP": ("BMP", ".bmp"),
}

ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
    "4:5": (4, 5),
    "2:3": (2, 3),
    "9:16": (9, 16),
}


@dataclass
class Settings:
    # Recadrage : "none" | "ratio" | "manual"
    crop_mode: str = "none"
    crop_ratio: tuple[int, int] = (1, 1)
    # Rectangle manuel en coordonnées relatives (0..1) : x, y, largeur, hauteur
    crop_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    # Remplacement du fond (détourage)
    bg_enabled: bool = False
    bg_model: str = "isnet"  # "isnet" | "birefnet"
    bg_fill: str = "transparent"  # "transparent" | "color" | "studio" | "image" | "blur"
    bg_color: tuple[int, int, int] = (255, 255, 255)
    bg_image: str = ""
    bg_feather: int = 1  # adoucissement du contour (pour 1000 px)
    bg_trim: bool = False  # recadrer sur le sujet
    bg_margin: int = 10  # marge autour du sujet, en % du plus grand côté
    bg_ai: bool = False  # nouveau fond généré par l'IA OpenAI
    bg_prompt: str = ""  # description du nouveau fond
    bg_shadow: int = 0  # ombre sous le produit, 0 (aucune) à 100
    bg_shadow_mode: str = "contact"  # "contact" (sous le produit) | "sun" (ombre portée par un soleil)
    sun_x: float = -0.5  # position du soleil (-1..1, centre = au-dessus du produit)
    sun_y: float = -0.5
    sun_blur: int = 50  # douceur de l'ombre portée (0..100)

    # Changement de ratio en ajoutant du fond (sans recadrer)
    pad_enabled: bool = False
    pad_ratio: tuple[int, int] = (1, 1)
    # "auto_color" | "color" | "blur" | "edge" | "transparent" | "ai_local" | "ai_openai"
    pad_fill: str = "blur"
    # "auto" (côté opposé au produit) | "center" | "start" (haut/gauche) | "end" (bas/droite)
    pad_position: str = "auto"
    pad_color: tuple[int, int, int] = (255, 255, 255)
    pad_blur: int = 40
    openai_key: str = ""
    openai_model: str = "gpt-image-1"
    openai_quality: str = "high"  # "low" | "medium" | "high"
    # retouches IA : zones (x, y, l, h) relatives à l'image après ajout du fond, avant redimensionnement
    retouch: tuple = ()
    retouch_engine: str = "local"  # "local" | "openai"

    # Redimensionnement : "none" | "width" | "height" | "fit" | "exact" | "percent"
    resize_mode: str = "none"
    width: int = 1920
    height: int = 1080
    percent: int = 50
    no_upscale: bool = True

    # Retouches
    rotate: int = 0  # 0, 90, 180, 270 (sens horaire)
    flip_h: bool = False
    flip_v: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    grayscale: bool = False

    # Sortie
    out_format: str = "Identique"
    quality: int = 90
    suffix: str = ""
    keep_metadata: bool = False


def load(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respecte l'orientation de l'appareil
    return img


def _crop(img: Image.Image, s: Settings) -> Image.Image:
    w, h = img.size
    if s.crop_mode == "ratio":
        rw, rh = s.crop_ratio
        target = rw / rh
        if w / h > target:
            nw, nh = round(h * target), h
        else:
            nw, nh = w, round(w / target)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh))
    if s.crop_mode == "manual":
        x, y, cw, ch = s.crop_rect
        box = (
            max(0, round(x * w)),
            max(0, round(y * h)),
            min(w, round((x + cw) * w)),
            min(h, round((y + ch) * h)),
        )
        if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
            return img.crop(box)
    return img


def _resize(img: Image.Image, s: Settings) -> Image.Image:
    w, h = img.size
    mode = s.resize_mode
    if mode == "none":
        return img
    if mode == "width":
        nw, nh = s.width, round(h * s.width / w)
    elif mode == "height":
        nw, nh = round(w * s.height / h), s.height
    elif mode == "fit":
        scale = min(s.width / w, s.height / h)
        nw, nh = round(w * scale), round(h * scale)
    elif mode == "exact":
        nw, nh = s.width, s.height
    elif mode == "percent":
        nw, nh = round(w * s.percent / 100), round(h * s.percent / 100)
    else:
        return img
    if s.no_upscale and mode != "exact" and (nw > w or nh > h):
        return img
    nw, nh = max(1, nw), max(1, nh)
    return img.resize((nw, nh), Image.LANCZOS)


def _geometry(img: Image.Image, s: Settings) -> Image.Image:
    if s.rotate:
        img = img.rotate(-s.rotate, expand=True)
    if s.flip_h:
        img = ImageOps.mirror(img)
    if s.flip_v:
        img = ImageOps.flip(img)
    return img


def padded_size(size: tuple[int, int], ratio: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    t = ratio[0] / ratio[1]
    if w / h > t:
        return w, max(h, round(w / t))
    return max(w, round(h * t)), h


def estimate_background(img: Image.Image) -> tuple[int, int, int]:
    """Couleur de fond estimée à partir du bord le plus uniforme de l'image.

    Prendre la médiane de tout le pourtour se trompe quand le produit occupe une
    grande partie d'un bord ; le bord le plus homogène est presque toujours du fond."""
    import numpy as np

    small = img.convert("RGB")
    small.thumbnail((256, 256))
    a = np.asarray(small, dtype=np.int16)
    h, w = a.shape[:2]
    b = max(1, round(min(w, h) * 0.03))
    strips = [a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)]
    best = None
    for st in strips:
        med = np.median(st, axis=0)
        spread = float(np.median(np.abs(st - med).max(axis=1)))
        if best is None or spread < best[0]:
            best = (spread, med)
    return tuple(int(round(c)) for c in best[1])  # type: ignore[return-value]


def edge_contact(img: Image.Image, tol: int = 40) -> dict:
    """Part (0..1) de chaque bord occupée par le sujet.

    Image transparente : on regarde l'opacité. Sinon, on compare les pixels du bord
    à la couleur de fond estimée (détection par couleur)."""
    small = img.copy()
    small.thumbnail((256, 256))
    w, h = small.size
    b = max(1, round(min(w, h) * 0.02))
    boxes = {"top": (0, 0, w, b), "bottom": (0, h - b, w, h),
             "left": (0, 0, b, h), "right": (w - b, 0, w, h)}
    out = {}
    if small.mode in ("RGBA", "LA"):
        alpha = small.getchannel("A")
        for k, box in boxes.items():
            data = list(alpha.crop(box).getdata())
            out[k] = sum(1 for a in data if a > 128) / len(data)
        return out
    rgb = small.convert("RGB")
    bg = estimate_background(rgb)
    for k, box in boxes.items():
        data = list(rgb.crop(box).getdata())
        diff = sum(1 for c in data if max(abs(c[0] - bg[0]), abs(c[1] - bg[1]), abs(c[2] - bg[2])) > tol)
        out[k] = diff / len(data)
    return out


def pad_offsets(img: Image.Image, new_size: tuple[int, int], position: str) -> tuple[int, int, str]:
    """Position de l'image dans la nouvelle toile.

    position : "auto" (détecte le côté où le produit touche le bord et ajoute
    le fond du côté opposé), "center", "start" (fond ajouté en haut / à gauche),
    "end" (fond ajouté en bas / à droite). Renvoie (ox, oy, explication)."""
    w, h = img.size
    nw, nh = new_size
    dx, dy = nw - w, nh - h
    vertical = dy > 0
    a, b = ("top", "bottom") if vertical else ("left", "right")
    where = {"top": "en haut", "bottom": "en bas", "left": "à gauche", "right": "à droite"}

    if position == "auto":
        c = edge_contact(img)
        thr = 0.08
        if c[b] > thr and c[b] > c[a] * 2 + 0.02:
            position = "start"  # produit collé en bas / à droite -> fond en haut / à gauche
        elif c[a] > thr and c[a] > c[b] * 2 + 0.02:
            position = "end"
        else:
            position = "center"
    if position == "start":
        off, why = (dx, dy), f"fond ajouté {where[a]}"
    elif position == "end":
        off, why = (0, 0), f"fond ajouté {where[b]}"
    else:
        off, why = (dx // 2, dy // 2), "fond ajouté des deux côtés"
    return off[0], off[1], why


def _edge_band(strip: Image.Image, size: tuple[int, int]) -> Optional[Image.Image]:
    """Étire une bande de bord (moyennée pour éviter les stries) à la taille voulue."""
    if size[0] <= 0 or size[1] <= 0:
        return None
    return strip.resize(size, Image.BILINEAR)


def _pad(img: Image.Image, s: Settings, info: Optional[dict] = None, ai_fn=None) -> Image.Image:
    """Agrandit la toile au ratio voulu et remplit le fond manquant, sans déformer
    le produit : avec la position automatique, le fond est ajouté du côté opposé
    au bord touché par le produit."""
    if not s.pad_enabled:
        return img
    w, h = img.size
    nw, nh = padded_size((w, h), s.pad_ratio)
    if (nw, nh) == (w, h):
        return img
    has_alpha = img.mode in ("RGBA", "LA")
    src = img.convert("RGBA")
    ox, oy, why = pad_offsets(img, (nw, nh), s.pad_position)
    if info is not None:
        info["pad"] = why

    if s.pad_fill in ("ai_local", "ai_openai") and has_alpha:
        # fond déjà transparent (détourage) : rien à inventer, on garde la transparence
        s = Settings(**{**s.__dict__, "pad_fill": "transparent"})
    if s.pad_fill in ("ai_local", "ai_openai"):
        if ai_fn is None:
            from .outpaint import outpaint
            out = outpaint(img, (nw, nh), ox, oy, "openai" if s.pad_fill == "ai_openai" else "local",
                           s.openai_key, s.openai_model, s.openai_quality)
        else:
            out = ai_fn(img, (nw, nh), ox, oy)
        if out is not None:
            return out
        # aperçu pas encore prêt : fond généré lisse en attendant
        s = Settings(**{**s.__dict__, "pad_fill": "generated"})

    if s.pad_fill == "transparent":
        canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    elif s.pad_fill == "color":
        canvas = Image.new("RGBA", (nw, nh), (*s.pad_color, 255))
    elif s.pad_fill == "auto_color":
        canvas = Image.new("RGBA", (nw, nh), (*estimate_background(img), 255))
    elif s.pad_fill == "generated":
        from .outpaint import smooth_fill
        out = smooth_fill(img, (nw, nh), ox, oy)
        if has_alpha:
            canvas = out.convert("RGBA")
        else:
            return out
    elif s.pad_fill == "edge":
        canvas = Image.new("RGBA", (nw, nh))
        k = max(1, min(w, h) // 200)  # quelques pixels moyennés
        if nw > w:  # bandes gauche / droite
            left = src.crop((0, 0, k, h)).resize((1, h), Image.BOX)
            right = src.crop((w - k, 0, w, h)).resize((1, h), Image.BOX)
            for band, x in ((_edge_band(left, (ox, h)), 0), (_edge_band(right, (nw - w - ox, h)), ox + w)):
                if band:
                    canvas.paste(band, (x, oy))
        if nh > h:  # bandes haut / bas
            top = src.crop((0, 0, w, k)).resize((w, 1), Image.BOX)
            bottom = src.crop((0, h - k, w, h)).resize((w, 1), Image.BOX)
            for band, y in ((_edge_band(top, (w, oy)), 0), (_edge_band(bottom, (w, nh - h - oy)), oy + h)):
                if band:
                    canvas.paste(band, (ox, y))
    else:  # "blur" : l'image agrandie pour couvrir la toile, floutée
        scale = max(nw / w, nh / h)
        # on floute sur une version réduite (beaucoup plus rapide), puis on agrandit
        small_f = min(1.0, 400 / max(nw, nh))
        cw, ch = max(1, round(w * scale * small_f)), max(1, round(h * scale * small_f))
        bg = src.convert("RGB").resize((cw, ch), Image.BILINEAR)
        tw, th = max(1, round(nw * small_f)), max(1, round(nh * small_f))
        bg = bg.crop(((cw - tw) // 2, (ch - th) // 2, (cw - tw) // 2 + tw, (ch - th) // 2 + th))
        bg = bg.filter(ImageFilter.GaussianBlur(max(1, s.pad_blur * small_f * max(nw, nh) / 1000)))
        bg = ImageEnhance.Brightness(bg).enhance(0.85)
        canvas = bg.resize((nw, nh), Image.BICUBIC).convert("RGBA")

    canvas.alpha_composite(src, (ox, oy))
    if s.pad_fill == "transparent" or has_alpha:
        return canvas
    return canvas.convert("RGB")


def _cover(bg: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Redimensionne bg pour couvrir size (comme background-size: cover)."""
    w, h = size
    scale = max(w / bg.width, h / bg.height)
    nw, nh = max(w, round(bg.width * scale)), max(h, round(bg.height * scale))
    bg = bg.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return bg.crop((left, top, left + w, top + h))


_bg_image_cache: dict = {}


def _load_bg_image(path: str) -> Image.Image:
    if path not in _bg_image_cache:
        _bg_image_cache.clear()
        _bg_image_cache[path] = load(path).convert("RGB")
    return _bg_image_cache[path]


def _replace_background(img: Image.Image, mask: Image.Image, s: Settings) -> Image.Image:
    w, h = img.size
    if s.bg_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(s.bg_feather * max(w, h) / 1000))
    src = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):  # combine avec la transparence existante
        mask = ImageChops.multiply(mask, src.getchannel("A"))

    if s.bg_trim:
        box = mask.point(lambda v: 255 if v > 24 else 0).getbbox()
        if box:
            m = round(max(box[2] - box[0], box[3] - box[1]) * s.bg_margin / 100)
            box = (box[0] - m, box[1] - m, box[2] + m, box[3] + m)
            src, mask = src.crop(box), mask.crop(box)  # hors image = transparent
            img = img.convert("RGB").crop(box)
            w, h = src.size

    if s.bg_ai:
        from .outpaint import generate_background
        return generate_background(img.convert("RGB"), mask, s.bg_prompt, s.openai_key,
                                   s.openai_model, s.openai_quality)

    if s.bg_fill == "color":
        canvas = Image.new("RGBA", (w, h), (*s.bg_color, 255))
    elif s.bg_fill == "studio":
        canvas = studio_background((w, h), s.bg_color).convert("RGBA")
    elif s.bg_fill == "image" and s.bg_image:
        canvas = _cover(_load_bg_image(s.bg_image), (w, h)).convert("RGBA")
    elif s.bg_fill == "blur":
        bg = img.convert("RGB")
        f = min(1.0, 400 / max(w, h))
        small = bg.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
        small = small.filter(ImageFilter.GaussianBlur(max(2, 25 * f * max(w, h) / 1000)))
        canvas = small.resize((w, h), Image.BICUBIC).convert("RGBA")
    else:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if s.bg_shadow > 0 and s.bg_shadow_mode == "sun":
        canvas = _add_sun_shadow(canvas, mask, s.bg_shadow / 100, s.sun_x, s.sun_y, s.sun_blur / 100)
    elif s.bg_shadow > 0:
        canvas = _add_shadow(canvas, mask, s.bg_shadow / 100)
    subject = src.copy()
    subject.putalpha(mask)
    canvas.alpha_composite(subject)
    return canvas


def studio_background(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    """Fond « studio » : la couleur choisie, plus claire au centre et en haut, qui
    s'assombrit doucement vers les bords et le bas (comme un fond papier éclairé)."""
    import numpy as np

    w, h = size
    sw, sh = max(2, min(w, 400)), max(2, min(h, 400))
    y = np.linspace(0, 1, sh, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, sw, dtype=np.float32)[None, :]
    light = 1.06 - 0.10 * y ** 1.5                                  # plus sombre vers le bas
    light = light - 0.07 * np.clip(x ** 2 + ((y - 0.4) * 1.3) ** 2 - 0.15, 0, None)  # vignettage doux
    a = np.asarray(color, np.float32)[None, None, :] * light[:, :, None]
    small = Image.fromarray(a.clip(0, 255).round().astype(np.uint8))
    out = np.asarray(small.resize((w, h), Image.BICUBIC), np.float32)
    out += np.random.default_rng(0).normal(0, 1.2, (h, w, 1)).astype(np.float32)  # évite les bandes
    return Image.fromarray(out.clip(0, 255).round().astype(np.uint8))


def _add_sun_shadow(canvas: Image.Image, mask: Image.Image, strength: float,
                    sun_x: float, sun_y: float, softness: float) -> Image.Image:
    """Ombre portée « naturelle » : la silhouette projetée à l'opposé du soleil.

    sun_x / sun_y : position du soleil (-1..1) ; plus il est loin du centre (soleil bas),
    plus l'ombre est longue. L'ombre est nette près du produit et devient plus floue
    et plus claire en s'éloignant, comme une vraie ombre."""
    import math

    w, h = canvas.size
    dist = min(1.0, math.hypot(sun_x, sun_y))
    if dist < 0.02:
        return _add_shadow(canvas, mask, strength)
    # travail à basse résolution (ombre = basses fréquences), puis agrandissement
    f = min(1.0, 700 / max(w, h))
    sw, sh = max(1, round(w * f)), max(1, round(h * f))
    m = mask.resize((sw, sh), Image.BILINEAR).point(lambda v: 255 if v > 100 else 0)
    box = m.getbbox()
    if not box:
        return canvas
    size = max(box[2] - box[0], box[3] - box[1])
    dx, dy = -sun_x / dist, -sun_y / dist  # l'ombre part à l'opposé du soleil
    length = size * (0.04 + 0.9 * dist ** 1.6)
    steps = max(4, min(400, int(length / 1.5)))  # pas fins : pas d'escalier
    acc = Image.new("L", (sw, sh), 0)
    for i in range(steps + 1):
        t = i / steps
        k = (1 - t) ** 1.2 * 0.8 + 0.2 * (1 - t)  # de plus en plus claire en s'éloignant
        layer = Image.new("L", (sw, sh), 0)
        layer.paste(m.point(lambda v, k=k: int(v * k)), (round(dx * length * t), round(dy * length * t)))
        acc = ImageChops.lighter(acc, layer)
    # flou progressif : net près du produit (valeurs fortes), flou au loin (valeurs faibles)
    near = acc.filter(ImageFilter.GaussianBlur(1 + softness * size * 0.01))
    far = acc.filter(ImageFilter.GaussianBlur(2 + softness * size * 0.08))
    acc = Image.composite(near, far, near)
    alpha = acc.resize((w, h), Image.BICUBIC).point(lambda v: int(v * min(1.0, strength) * 0.75))
    # ombre légèrement teintée (plus naturelle qu'un noir pur)
    tint = canvas.convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))
    dark = Image.new("RGBA", (w, h), tuple(int(c * 0.25) for c in tint) + (0,))
    dark.putalpha(alpha)
    out = canvas.copy()
    out.alpha_composite(dark)
    return out


def _add_shadow(canvas: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
    """Ombre réaliste sous le produit, calculée à partir du détourage :
    une ombre de contact sombre et nette sous la base, et une ombre portée douce."""
    w, h = canvas.size
    m = mask.point(lambda v: 255 if v > 100 else 0)
    box = m.getbbox()
    if not box:
        return canvas
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    size = max(bw, bh)
    shadow = Image.new("L", (w, h), 0)
    # ombre portée : la silhouette écrasée au sol, décalée vers le bas, très floue
    flat_h = max(2, round(bh * 0.12))
    flat = m.crop(box).resize((bw, flat_h), Image.BILINEAR)
    shadow.paste(flat, (x0, y1 - flat_h // 2), flat)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, size * 0.035)))
    soft = shadow.point(lambda v: int(v * 0.55))
    # ombre de contact : la base de l'objet, peu floue
    base = Image.new("L", (w, h), 0)
    band = max(2, round(bh * 0.03))
    base.paste(m.crop((x0, y1 - band, x1, y1)), (x0, y1 - band // 2))
    base = base.filter(ImageFilter.GaussianBlur(max(1, size * 0.008)))
    sh = ImageChops.lighter(soft, base.point(lambda v: int(v * 0.8)))
    sh = sh.point(lambda v: int(v * min(1.0, strength)))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dark.putalpha(sh)
    out = canvas.copy()
    out.alpha_composite(dark)
    return out


def _enhance(img: Image.Image, s: Settings) -> Image.Image:
    # Les filtres ImageEnhance travaillent en RGB(A)/L
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        img = img.convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")

    def enhance(im, cls, factor):
        if abs(factor - 1.0) < 1e-3:
            return im
        if im.mode in ("RGBA", "LA"):
            alpha = im.getchannel("A")
            base = im.convert("RGB" if im.mode == "RGBA" else "L")
            base = cls(base).enhance(factor)
            base.putalpha(alpha)
            return base
        return cls(im).enhance(factor)

    img = enhance(img, ImageEnhance.Brightness, s.brightness)
    img = enhance(img, ImageEnhance.Contrast, s.contrast)
    if img.mode in ("RGB", "RGBA"):
        img = enhance(img, ImageEnhance.Color, s.saturation)
    img = enhance(img, ImageEnhance.Sharpness, s.sharpness)

    if s.grayscale:
        img = img.convert("LA" if img.mode in ("RGBA", "LA") else "L")
    return img


def process(img: Image.Image, s: Settings, mask_fn=None, info: Optional[dict] = None, ai_fn=None) -> Image.Image:
    """Ordre : recadrage -> rotation/miroir -> retouches -> remplacement du fond
    -> ajout de fond (ratio) -> redimensionnement.

    mask_fn(img) -> masque L ou None ; par défaut le détourage est calculé ici."""
    img = _crop(img, s)
    img = _geometry(img, s)
    mask = None
    if s.bg_enabled:
        if mask_fn is None:
            from .background import compute_mask
            mask = compute_mask(img, s.bg_model)
        else:
            mask = mask_fn(img)
    img = _enhance(img, s)
    if mask is not None:
        img = _replace_background(img, mask, s)
        if s.bg_ai or s.bg_fill in ("color", "studio", "image", "blur"):
            img = img.convert("RGB")
    img = _pad(img, s, info, ai_fn)
    if s.retouch:
        from .outpaint import retouch
        img = retouch(img, s.retouch, s.retouch_engine, s.openai_key, s.openai_model, s.openai_quality)
    img = _resize(img, s)
    return img


def output_path(src: Path, out_dir: Path, s: Settings) -> Path:
    fmt, ext = OUTPUT_FORMATS.get(s.out_format, (None, None))
    if ext is None:
        ext = src.suffix.lower()
        if ext in (".heic", ".heif"):
            ext = ".jpg"  # Pillow n'écrit pas le HEIC par défaut
    base = out_dir / f"{src.stem}{s.suffix}{ext}"
    candidate, i = base, 1
    while candidate.exists():
        candidate = out_dir / f"{src.stem}{s.suffix}_{i}{ext}"
        i += 1
    return candidate


_EXT_TO_FORMAT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
    ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP", ".gif": "GIF",
}


def save(img: Image.Image, dst: Path, s: Settings, src_img: Image.Image | None = None) -> None:
    fmt = _EXT_TO_FORMAT.get(dst.suffix.lower(), "PNG")
    kwargs: dict = {}

    if fmt in ("JPEG", "BMP"):
        # Pas de transparence : on aplatit sur fond blanc
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg if not s.grayscale else bg.convert("L")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    if fmt == "JPEG":
        kwargs.update(quality=s.quality, optimize=True, progressive=True)
    elif fmt == "WEBP":
        kwargs.update(quality=s.quality, method=4)
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    elif fmt == "TIFF":
        kwargs.update(compression="tiff_lzw")

    if s.keep_metadata and src_img is not None:
        icc = src_img.info.get("icc_profile")
        if icc and fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
            kwargs["icc_profile"] = icc
        exif = src_img.getexif()
        if exif and fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
            exif[0x0112] = 1  # orientation déjà appliquée
            kwargs["exif"] = exif.tobytes()

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, fmt, **kwargs)


def process_file(src: str | Path, out_dir: str | Path, s: Settings, mask_fn=None) -> Path:
    src, out_dir = Path(src), Path(out_dir)
    original = Image.open(src)
    img = ImageOps.exif_transpose(original)
    img.load()
    result = process(img, s, mask_fn)
    dst = output_path(src, out_dir, s)
    save(result, dst, s, original)
    return dst


def collect_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS and not p.name.startswith(".")
    )


def output_size(size: tuple[int, int], s: Settings) -> tuple[int, int]:
    """Calcule la taille finale sans traiter l'image (pour l'aperçu)."""
    w, h = size
    if s.crop_mode == "ratio":
        rw, rh = s.crop_ratio
        t = rw / rh
        w, h = (round(h * t), h) if w / h > t else (w, round(w / t))
    elif s.crop_mode == "manual":
        x, y, cw, ch = s.crop_rect
        nw = min(w, round((x + cw) * w)) - max(0, round(x * w))
        nh = min(h, round((y + ch) * h)) - max(0, round(y * h))
        if nw >= 1 and nh >= 1:
            w, h = nw, nh
    if s.rotate in (90, 270):
        w, h = h, w
    if s.bg_enabled and s.bg_trim:
        return (0, 0)  # dépend du contenu : inconnu sans calculer le masque
    if s.pad_enabled:
        w, h = padded_size((w, h), s.pad_ratio)
    # Réutilise la logique de _resize sur une taille abstraite
    class _Fake:
        def __init__(self, sz):
            self.size = sz

        def resize(self, sz, *_a, **_k):
            return _Fake(sz)

    return _resize(_Fake((w, h)), s).size  # type: ignore[arg-type]
