"""Traitement d'images (sans interface) : recadrage, redimensionnement,
retouches et conversion. Utilisé par l'interface et testable seul."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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

    # Changement de ratio en ajoutant du fond (sans recadrer)
    pad_enabled: bool = False
    pad_ratio: tuple[int, int] = (1, 1)
    # "color" | "blur" | "edge" | "transparent"
    pad_fill: str = "blur"
    pad_color: tuple[int, int, int] = (255, 255, 255)
    pad_blur: int = 40

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


def _pad(img: Image.Image, s: Settings) -> Image.Image:
    """Agrandit la toile au ratio voulu et remplit le fond manquant."""
    if not s.pad_enabled:
        return img
    w, h = img.size
    nw, nh = padded_size((w, h), s.pad_ratio)
    if (nw, nh) == (w, h):
        return img
    has_alpha = img.mode in ("RGBA", "LA")
    src = img.convert("RGBA")
    ox, oy = (nw - w) // 2, (nh - h) // 2

    if s.pad_fill == "transparent":
        canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    elif s.pad_fill == "color":
        canvas = Image.new("RGBA", (nw, nh), (*s.pad_color, 255))
    elif s.pad_fill == "edge":
        canvas = Image.new("RGBA", (nw, nh))
        if nw > w:  # bandes gauche / droite
            canvas.paste(src.crop((0, 0, 1, h)).resize((ox, h)), (0, oy))
            canvas.paste(src.crop((w - 1, 0, w, h)).resize((nw - w - ox, h)), (ox + w, oy))
        if nh > h:  # bandes haut / bas
            canvas.paste(src.crop((0, 0, w, 1)).resize((w, oy)), (ox, 0))
            canvas.paste(src.crop((0, h - 1, w, h)).resize((w, nh - h - oy)), (ox, oy + h))
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


def process(img: Image.Image, s: Settings) -> Image.Image:
    """Ordre : recadrage -> rotation/miroir -> retouches -> ajout de fond -> redimensionnement."""
    img = _crop(img, s)
    img = _geometry(img, s)
    img = _enhance(img, s)
    img = _pad(img, s)
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


def process_file(src: str | Path, out_dir: str | Path, s: Settings) -> Path:
    src, out_dir = Path(src), Path(out_dir)
    original = Image.open(src)
    img = ImageOps.exif_transpose(original)
    img.load()
    result = process(img, s)
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
    if s.pad_enabled:
        w, h = padded_size((w, h), s.pad_ratio)
    # Réutilise la logique de _resize sur une taille abstraite
    class _Fake:
        def __init__(self, sz):
            self.size = sz

        def resize(self, sz, *_a, **_k):
            return _Fake(sz)

    return _resize(_Fake((w, h)), s).size  # type: ignore[arg-type]
