from pathlib import Path

from PIL import Image

from imagebatch import processing as P


def make(tmp: Path, name="a.png", size=(400, 200), mode="RGBA"):
    p = tmp / name
    Image.new(mode, size, (200, 50, 50, 255) if mode == "RGBA" else (200, 50, 50)).save(p)
    return p


def test_ratio_crop_and_resize(tmp_path):
    src = make(tmp_path)
    s = P.Settings(crop_mode="ratio", crop_ratio=(1, 1), resize_mode="width", width=100, out_format="JPEG")
    dst = P.process_file(src, tmp_path / "out", s)
    assert dst.suffix == ".jpg"
    with Image.open(dst) as im:
        assert im.size == (100, 100) and im.mode == "RGB"
    assert P.output_size((400, 200), s) == (100, 100)


def test_manual_crop_rotate(tmp_path):
    src = make(tmp_path, size=(400, 200), mode="RGB")
    s = P.Settings(crop_mode="manual", crop_rect=(0.0, 0.0, 0.5, 1.0), rotate=90)
    dst = P.process_file(src, tmp_path / "out", s)
    with Image.open(dst) as im:
        assert im.size == (200, 200)


def test_fit_no_upscale_and_edits(tmp_path):
    src = make(tmp_path, size=(300, 100))
    s = P.Settings(resize_mode="fit", width=1000, height=1000, brightness=1.3, contrast=0.8,
                   saturation=0, sharpness=2, grayscale=True, flip_h=True, out_format="WEBP")
    dst = P.process_file(src, tmp_path / "out", s)
    with Image.open(dst) as im:
        assert im.size == (300, 100)
    assert P.output_size((300, 100), s) == (300, 100)


def test_no_overwrite(tmp_path):
    src = make(tmp_path)
    out = tmp_path / "out"
    s = P.Settings(suffix="")
    a = P.process_file(src, out, s)
    b = P.process_file(src, out, s)
    assert a != b and a.exists() and b.exists()


def test_collect(tmp_path):
    make(tmp_path, "x.png")
    (tmp_path / "notes.txt").write_text("x")
    sub = tmp_path / "sub"; sub.mkdir(); make(sub, "y.png")
    assert len(P.collect_images(tmp_path)) == 1
    assert len(P.collect_images(tmp_path, recursive=True)) == 2


def test_pad_ratio_fills(tmp_path):
    src = make(tmp_path, size=(400, 200), mode="RGB")
    for fill in ("blur", "color", "edge", "transparent"):
        s = P.Settings(pad_enabled=True, pad_ratio=(1, 1), pad_fill=fill, pad_color=(0, 0, 255),
                       out_format="PNG", suffix="_" + fill)
        dst = P.process_file(src, tmp_path / "out", s)
        with Image.open(dst) as im:
            assert im.size == (400, 400)
            if fill == "color":
                assert im.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
            if fill == "transparent":
                assert im.getpixel((0, 0))[3] == 0
            assert im.convert("RGB").getpixel((200, 200)) == (200, 50, 50)
    # portrait 9:16 : on ajoute des bandes en haut et en bas
    s = P.Settings(pad_enabled=True, pad_ratio=(9, 16), pad_fill="edge")
    assert P.output_size((400, 200), s) == (400, 711)
    dst = P.process_file(src, tmp_path / "out", s)
    with Image.open(dst) as im:
        assert im.size == (400, 711)


def test_replace_background_with_given_mask(tmp_path):
    from PIL import ImageDraw
    img = Image.new("RGB", (200, 100), (10, 200, 10))
    mask = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(mask).rectangle([80, 30, 119, 69], fill=255)
    s = P.Settings(bg_enabled=True, bg_fill="color", bg_color=(0, 0, 255), bg_feather=0,
                   bg_trim=True, bg_margin=0)
    out = P.process(img, s, mask_fn=lambda _im: mask)
    assert out.size == (40, 40)
    assert out.getpixel((20, 20)) == (10, 200, 10)
    s.bg_trim = False
    out = P.process(img, s, mask_fn=lambda _im: mask)
    assert out.size == (200, 100) and out.getpixel((2, 2)) == (0, 0, 255)
    s.bg_fill = "transparent"
    out = P.process(img, s, mask_fn=lambda _im: mask)
    assert out.mode == "RGBA" and out.getpixel((2, 2))[3] == 0


def test_pad_auto_extends_opposite_side_of_product():
    from PIL import ImageDraw
    bg = (230, 230, 228)
    im = Image.new("RGB", (800, 500), bg)
    ImageDraw.Draw(im).rectangle([300, 150, 500, 499], fill=(30, 90, 160))  # sort par le bas
    s = P.Settings(pad_enabled=True, pad_ratio=(1, 1), pad_fill="edge", pad_position="auto")
    info = {}
    out = P.process(im, s, info=info)
    assert out.size == (800, 800) and info["pad"] == "fond ajouté en haut"
    assert out.getpixel((400, 799)) == (30, 90, 160)  # produit toujours collé en bas
    assert out.getpixel((400, 10)) == bg  # fond prolongé en haut, produit non étiré
    assert P.pad_offsets(im.transpose(Image.FLIP_TOP_BOTTOM), (800, 800), "auto")[1] == 0
    centered = Image.new("RGB", (800, 500), bg)
    ImageDraw.Draw(centered).ellipse([300, 150, 500, 350], fill=(200, 0, 0))
    assert P.pad_offsets(centered, (800, 800), "auto")[1] == 150
    s.pad_fill = "auto_color"
    assert P.process(im, s).getpixel((5, 5)) == bg


def test_openai_outpaint_keeps_original(monkeypatch):
    import base64
    import io
    import json
    import urllib.request

    from imagebatch import outpaint as OP  # noqa: F401

    sent = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, context=None, timeout=None):
        sent["body"] = req.data
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1024), (10, 200, 10)).save(buf, "PNG")
        return Resp(json.dumps({"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    img = Image.new("RGB", (400, 200), (30, 90, 160))
    out = P.process(img, P.Settings(pad_enabled=True, pad_ratio=(1, 1), pad_fill="ai_openai",
                                    pad_position="end", openai_key="sk-test"))
    assert out.size == (400, 400)
    assert out.getpixel((200, 100)) == (30, 90, 160)  # original intact
    assert out.getpixel((200, 380)) == (10, 200, 10)  # zone ajoutée = IA
    assert b'name="mask"' in sent["body"] and b"1024x1024" in sent["body"]


def test_combined_bg_ai_and_pad_ai(monkeypatch):
    """Fond IA + agrandissement IA + recadrage sur le sujet, cumulés."""
    import base64
    import io
    import json
    import urllib.request

    from PIL import ImageDraw

    calls = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, context=None, timeout=None):
        size = req.data.split(b'name="size"\r\n\r\n')[1].split(b"\r\n")[0].decode()
        w, h = map(int, size.split("x"))
        calls.append((size, b"Replace only the background" in req.data))
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (10, 200, 10)).save(buf, "PNG")
        return Resp(json.dumps({"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    img = Image.new("RGB", (400, 200), (240, 240, 240))
    mask = Image.new("L", (400, 200), 0)
    ImageDraw.Draw(mask).rectangle([150, 50, 249, 199], fill=255)
    ImageDraw.Draw(img).rectangle([150, 50, 249, 199], fill=(30, 90, 160))
    s = P.Settings(bg_enabled=True, bg_ai=True, bg_prompt="marbre", bg_feather=0, openai_key="sk-test",
                   pad_enabled=True, pad_ratio=(1, 1), pad_fill="ai_openai", pad_position="center")
    out = P.process(img, s, mask_fn=lambda _im: mask)
    assert out.size == (400, 400) and out.mode == "RGB"
    assert out.getpixel((200, 200)) == (30, 90, 160)  # produit intact
    assert out.getpixel((20, 150)) == (10, 200, 10)  # fond généré
    assert out.getpixel((200, 20)) == (10, 200, 10)  # zone ajoutée générée
    assert [c[1] for c in calls] == [True, False]  # 1 appel fond, 1 appel agrandissement


def test_transparent_bg_with_pad_ai_stays_transparent():
    img = Image.new("RGB", (200, 100), (240, 240, 240))
    mask = Image.new("L", (200, 100), 0)
    mask.paste(255, (80, 20, 120, 100))
    s = P.Settings(bg_enabled=True, bg_fill="transparent", bg_feather=0,
                   pad_enabled=True, pad_ratio=(1, 1), pad_fill="ai_local")
    out = P.process(img, s, mask_fn=lambda _im: mask)  # aucun appel IA nécessaire
    assert out.size == (200, 200) and out.mode == "RGBA" and out.getpixel((5, 5))[3] == 0


def test_auto_position_with_large_product_at_bottom():
    # Produit qui occupe toute la largeur et la majeure partie des côtés :
    # le fond doit être estimé sur le bord du haut, et ajouté en haut.
    img = Image.new("RGB", (400, 300), (245, 245, 245))
    prod = Image.linear_gradient("L").rotate(90).resize((400, 230)).point(lambda v: 20 + v // 2).convert("RGB")
    img.paste(prod, (0, 70))
    assert P.estimate_background(img) == (245, 245, 245)
    ox, oy, why = P.pad_offsets(img, (400, 400), "auto")
    assert (ox, oy) == (0, 100) and "haut" in why


def test_generated_fill_does_not_stretch_pixels():
    # Bord très bruité : l'étirement ferait des stries verticales, le fond généré non.
    import numpy as np

    rng = np.random.default_rng(0)
    a = (120 + rng.normal(0, 15, (300, 400, 3))).clip(0, 255).astype("uint8")
    img = Image.fromarray(a)
    out = P.process(img, P.Settings(pad_enabled=True, pad_ratio=(1, 1), pad_fill="generated", pad_position="start"))
    assert out.size == (400, 400)
    added = np.asarray(out, np.float32)[:60].mean(axis=2)
    assert added.mean(axis=0).std() < 3  # pas de colonnes recopiées
    assert np.allclose(np.asarray(out)[200:], a[100:])  # l'original est intact


def test_retouch_removes_mark_in_selected_zone(monkeypatch):
    # La zone sélectionnée est regénérée (IA simulée), le reste de l'image est intact.
    import numpy as np
    from imagebatch import outpaint as OP

    monkeypatch.setattr(OP, "_lama", lambda tile, mask: Image.new("RGB", tile.size, (128, 128, 128)))
    img = Image.new("RGB", (400, 300), (128, 128, 128))
    img.paste((0, 0, 0), (100, 100, 120, 110))  # trace
    s = P.Settings(retouch=((0.24, 0.32, 0.08, 0.06),))
    out = np.asarray(P.process(img, s), np.int16)
    assert np.abs(out[100:110, 100:120] - 128).max() < 10
    assert (out[:50] == 128).all()


def test_studio_background_with_shadow():
    import numpy as np
    from PIL import ImageDraw

    img = Image.new("RGB", (400, 400), (0, 200, 0))
    mask = Image.new("L", (400, 400), 0)
    ImageDraw.Draw(mask).rectangle((150, 100, 250, 300), fill=255)
    s = P.Settings(bg_enabled=True, bg_fill="studio", bg_color=(230, 230, 230), bg_shadow=80)
    out = np.asarray(P.process(img, s, mask_fn=lambda im: mask), np.int16)
    assert out.shape == (400, 400, 3)
    assert out[:, :, 1].max() <= 255 and abs(int(out[50, 200, 1]) - int(out[50, 200, 0])) < 6  # plus de vert
    assert out[305, 200].mean() < out[305, 20].mean() - 15  # ombre sous le produit


def test_sun_shadow_falls_opposite_the_sun():
    import numpy as np
    from PIL import ImageDraw

    img = Image.new("RGB", (400, 400), (240, 240, 240))
    mask = Image.new("L", (400, 400), 0)
    ImageDraw.Draw(mask).rectangle((150, 150, 250, 250), fill=255)
    s = P.Settings(bg_enabled=True, bg_fill="color", bg_color=(240, 240, 240), bg_shadow=80,
                   bg_shadow_mode="sun", sun_x=-0.8, sun_y=-0.6)  # soleil en haut à gauche
    out = np.asarray(P.process(img, s, mask_fn=lambda im: mask), np.int16).mean(axis=2)
    assert out[258, 258] < out[142, 142] - 10  # ombre en bas à droite, rien en haut à gauche
