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
