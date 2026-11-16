"""Interface de bureau (PySide6) pour le traitement d'images par lot."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSizePolicy, QSpinBox, QSplitter, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from . import background as BG
from . import outpaint as OP
from . import processing as P

APP_NAME = "Numbered - Batch Cooking"


def asset_path(name: str) -> Path:
    """Fichier du dossier assets/, aussi dans l'app construite (PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / name
PREVIEW_MAX = 1400
PREVIEW_BG = QColor(222, 222, 222)  # fond gris clair derrière les images  # taille max de l'image utilisée pour l'aperçu
THUMB = 64


def pil_to_qimage(img: Image.Image) -> QImage:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    if img.mode == "RGB":
        data = img.tobytes("raw", "RGB")
        q = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    else:
        data = img.tobytes("raw", "RGBA")
        q = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
    return q.copy()  # détache des octets Python


# --------------------------------------------------------------------------- #
# Section repliable (fermée par défaut)
# --------------------------------------------------------------------------- #
class Section(QWidget):
    def __init__(self, title: str, expanded: bool = False, collapsible: bool = True):
        super().__init__()
        self.collapsible = collapsible
        self._title = title
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setStyleSheet(
            "QToolButton { border: none; border-radius: 4px; padding: 7px 6px; font-weight: 600;"
            " text-align: left; background: #38383c; }"
            "QToolButton:hover { background: #424247; }")
        self.header.toggled.connect(self._toggle)
        if not collapsible:  # toujours ouvert
            expanded = True
            self.header.setChecked(True)
            self.header.setEnabled(False)
            self.header.setStyleSheet(self.header.styleSheet() + "QToolButton:disabled { color: palette(text); }")
        self.body = QWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self._active = False
        self._toggle(expanded)

    def _toggle(self, on: bool):
        if not hasattr(self, "body"):
            return
        if not self.collapsible:
            self.header.setArrowType(Qt.NoArrow)
            self.body.setVisible(True)
            return
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)

    def set_active(self, active: bool):
        """Pastille dans le titre quand l'option modifie les images."""
        if active != self._active:
            self._active = active
            self.header.setText(f"{self._title}   ●" if active else self._title)


# --------------------------------------------------------------------------- #
# Aperçu avec zone de recadrage interactive
# --------------------------------------------------------------------------- #
def parse_color(text: str) -> QColor | None:
    """« #F2F2F2 », « F2F2F2 », « #FFF » ou « 242, 242, 242 » -> QColor (None si invalide)."""
    import re

    t = text.strip()
    nums = re.findall(r"\d+", t)
    if re.fullmatch(r"(rgb\s*\()?\s*\d+\s*[,; ]\s*\d+\s*[,; ]\s*\d+\s*\)?", t, re.I) and \
            all(int(n) <= 255 for n in nums):
        return QColor(*(int(n) for n in nums[:3]))
    h = t.lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}", h):
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return QColor("#" + h)
    return None


class PreviewWidget(QWidget):
    cropChanged = Signal(tuple)
    retouchDrawn = Signal(tuple)  # zone (x, y, l, h) relative à l'image affichée

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self._pix: QPixmap | None = None
        self.edit_crop = False
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.ratio: float | None = None  # largeur/hauteur en pixels de l'image
        self._img_aspect = 1.0
        self._drag = None  # ("new"|"move", point de départ, rect de départ)
        self.retouch_mode = False
        self.retouch_rects: list = []
        self._rdrag = None  # (point de départ, rect en cours)
        self.message = "Ajoutez des images (bouton ou glisser-déposer)"

    def set_image(self, pix: QPixmap | None, img_aspect: float = 1.0):
        self._pix = pix
        self._img_aspect = img_aspect
        self.update()

    def _target_rect(self) -> QRectF:
        if not self._pix:
            return QRectF()
        pw, ph = self._pix.width(), self._pix.height()
        scale = min((self.width() - 20) / pw, (self.height() - 20) / ph, 1.0 if pw > 600 else 4.0)
        w, h = pw * scale, ph * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), PREVIEW_BG)  # gris clair fixe, même en thème sombre
        if not self._pix:
            p.setPen(QColor(70, 70, 70))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        # damier pour la transparence
        r = self._target_rect()
        p.save()
        p.setClipRect(r)
        s = 10
        for yi in range(int(r.height() / s) + 1):
            for xi in range(int(r.width() / s) + 1):
                c = QColor(220, 220, 220) if (xi + yi) % 2 else QColor(250, 250, 250)
                p.fillRect(QRectF(r.x() + xi * s, r.y() + yi * s, s, s), c)
        p.restore()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(r, self._pix, QRectF(self._pix.rect()))

        if self.retouch_rects or self._rdrag:
            rects = list(self.retouch_rects) + ([self._rdrag[1]] if self._rdrag else [])
            for i, (x, y, w, h) in enumerate(rects):
                rr = QRectF(r.x() + x * r.width(), r.y() + y * r.height(), w * r.width(), h * r.height())
                p.fillRect(rr, QColor(255, 140, 0, 50))
                p.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
                p.drawRect(rr)

        if self.edit_crop:
            x, y, w, h = self.crop
            cr = QRectF(r.x() + x * r.width(), r.y() + y * r.height(), w * r.width(), h * r.height())
            shade = QColor(0, 0, 0, 130)
            p.fillRect(QRectF(r.left(), r.top(), r.width(), cr.top() - r.top()), shade)
            p.fillRect(QRectF(r.left(), cr.bottom(), r.width(), r.bottom() - cr.bottom()), shade)
            p.fillRect(QRectF(r.left(), cr.top(), cr.left() - r.left(), cr.height()), shade)
            p.fillRect(QRectF(cr.right(), cr.top(), r.right() - cr.right(), cr.height()), shade)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawRect(cr)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.DashLine))
            for i in (1, 2):
                p.drawLine(QPointF(cr.left() + cr.width() * i / 3, cr.top()),
                           QPointF(cr.left() + cr.width() * i / 3, cr.bottom()))
                p.drawLine(QPointF(cr.left(), cr.top() + cr.height() * i / 3),
                           QPointF(cr.right(), cr.top() + cr.height() * i / 3))

    def _rel(self, pos) -> tuple[float, float]:
        r = self._target_rect()
        x = (pos.x() - r.x()) / r.width()
        y = (pos.y() - r.y()) / r.height()
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _inside_crop(self, rx, ry) -> bool:
        x, y, w, h = self.crop
        return x <= rx <= x + w and y <= ry <= y + h

    def mousePressEvent(self, e):
        if self.retouch_mode and self._pix and e.button() == Qt.LeftButton:
            rx, ry = self._rel(e.position())
            self._rdrag = ((rx, ry), (rx, ry, 0.0, 0.0))
            return
        if not (self.edit_crop and self._pix) or e.button() != Qt.LeftButton:
            return
        rx, ry = self._rel(e.position())
        full = self.crop == (0.0, 0.0, 1.0, 1.0)
        mode = "move" if self._inside_crop(rx, ry) and not full else "new"
        self._drag = (mode, (rx, ry), self.crop)

    def mouseMoveEvent(self, e):
        if self.retouch_mode and self._pix:
            self.setCursor(Qt.CrossCursor)
            if self._rdrag:
                (sx, sy), _ = self._rdrag
                rx, ry = self._rel(e.position())
                self._rdrag = ((sx, sy), (min(sx, rx), min(sy, ry), abs(rx - sx), abs(ry - sy)))
                self.update()
            return
        if self.edit_crop and self._pix and not self._drag:
            rx, ry = self._rel(e.position())
            full = self.crop == (0.0, 0.0, 1.0, 1.0)
            self.setCursor(Qt.SizeAllCursor if self._inside_crop(rx, ry) and not full else Qt.CrossCursor)
        if not self._drag:
            return
        mode, (sx, sy), (cx, cy, cw, ch) = self._drag
        rx, ry = self._rel(e.position())
        if mode == "move":
            nx = min(max(cx + rx - sx, 0.0), 1.0 - cw)
            ny = min(max(cy + ry - sy, 0.0), 1.0 - ch)
            self.crop = (nx, ny, cw, ch)
        else:
            w, h = abs(rx - sx), abs(ry - sy)
            if self.ratio:
                # ratio en pixels -> ratio en coordonnées relatives
                rel_ratio = self.ratio / self._img_aspect
                if w / max(h, 1e-6) > rel_ratio:
                    h = w / rel_ratio
                else:
                    w = h * rel_ratio
                w = min(w, sx if rx < sx else 1 - sx)
                h = min(h, sy if ry < sy else 1 - sy)
                if w / max(h, 1e-6) > rel_ratio:
                    w = h * rel_ratio
                else:
                    h = w / rel_ratio
            x = sx - w if rx < sx else sx
            y = sy - h if ry < sy else sy
            self.crop = (x, y, w, h)
        self.update()

    def mouseReleaseEvent(self, _e):
        if self._rdrag:
            rect = self._rdrag[1]
            self._rdrag = None
            if rect[2] > 0.005 and rect[3] > 0.005:
                self.retouchDrawn.emit(rect)
            self.update()
            return
        if self._drag:
            self._drag = None
            x, y, w, h = self.crop
            if w < 0.01 or h < 0.01:
                self.crop = (0.0, 0.0, 1.0, 1.0)
            self.cropChanged.emit(self.crop)
            self.update()


# --------------------------------------------------------------------------- #
# Détourage de l'aperçu en tâche de fond
# --------------------------------------------------------------------------- #
class MaskThread(QThread):
    status = Signal(str)
    done = Signal(object, object)  # clé, masque (ou None)
    failed = Signal(object, str)

    def __init__(self, img: Image.Image, s: P.Settings, key):
        super().__init__()
        self.img, self.s, self.key = img, s, key

    def run(self):
        try:
            if not BG.is_downloaded(self.s.bg_model):
                def prog(done, total):
                    pct = f" {done * 100 // total} %" if total else ""
                    self.status.emit(f"Téléchargement du modèle de détourage (une seule fois){pct}…")
                BG.download(self.s.bg_model, prog)
            self.status.emit("Détourage en cours…")
            img = P._geometry(P._crop(self.img, self.s), self.s)
            self.done.emit(self.key, BG.compute_mask(img, self.s.bg_model))
        except Exception as exc:
            self.failed.emit(self.key, str(exc))


# --------------------------------------------------------------------------- #
# Export en tâche de fond
# --------------------------------------------------------------------------- #
class AiPreviewThread(QThread):
    """Calcule l'aperçu avec l'IA (fond et/ou agrandissement), sans bloquer l'interface."""
    status = Signal(str)
    done = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, img: Image.Image, s: P.Settings, key, mask_fn):
        super().__init__()
        self.img, self.s, self.key, self.mask_fn = img, s, key, mask_fn

    def run(self):
        try:
            if ((self.s.pad_enabled and self.s.pad_fill == "ai_local")
                    or (self.s.retouch and self.s.retouch_engine == "local")) and not OP.lama_ready():
                OP.lama_download(lambda d, t: self.status.emit(
                    f"Téléchargement de l'IA locale (une seule fois) {d * 100 // max(t, 1)} %…"))
            if self.s.bg_enabled and not BG.is_downloaded(self.s.bg_model):
                BG.download(self.s.bg_model)
            self.status.emit("IA en cours sur l'aperçu…")
            self.done.emit(self.key, P.process(self.img, self.s, self.mask_fn))
        except Exception as exc:
            self.failed.emit(self.key, str(exc))


PAD_TAGS = {"fond ajouté en haut": "fond ↑", "fond ajouté en bas": "fond ↓", "fond ajouté à gauche": "fond ←",
            "fond ajouté à droite": "fond →", "fond ajouté des deux côtés": "fond ↕"}


class DetectThread(QThread):
    """Calcule, pour chaque image du lot, de quel côté le fond sera ajouté."""
    found = Signal(int, str, str)  # génération, chemin, étiquette

    def __init__(self, gen: int, jobs: list):
        super().__init__()
        self.gen, self.jobs, self.cancelled = gen, jobs, False

    def run(self):
        for path, s in self.jobs:
            if self.cancelled:
                return
            try:
                with Image.open(path) as im:
                    im.draft("RGB", (512, 512))
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((400, 400))
                    info: dict = {}
                    P.process(im.convert("RGB"), s, None, info)
                tag = info.get("pad", "")
            except Exception:
                tag = ""
            self.found.emit(self.gen, path, tag)


class ExportThread(QThread):
    progress = Signal(int, str)
    done = Signal(int, list)

    def __init__(self, files: list[Path], out_dir: Path, settings: P.Settings,
                 overrides: dict | None = None):
        super().__init__()
        self.files, self.out_dir, self.settings = files, out_dir, settings
        self.overrides = overrides or {}
        self.cancelled = False

    def run(self):
        errors, ok = [], 0
        if self.settings.bg_enabled and not BG.is_downloaded(self.settings.bg_model):
            try:
                BG.download(self.settings.bg_model,
                            lambda d, t: self.progress.emit(0, f"modèle de détourage ({d * 100 // max(t, 1)} %)"))
            except Exception as exc:
                self.done.emit(0, [f"Téléchargement du modèle impossible : {exc}"])
                return
        local_retouch = any(o.get("retouch") for o in self.overrides.values()) and self.settings.retouch_engine == "local"
        if ((self.settings.pad_enabled and self.settings.pad_fill == "ai_local") or local_retouch) \
                and not OP.lama_ready():
            try:
                OP.lama_download(lambda d, t: self.progress.emit(0, f"IA locale ({d * 100 // max(t, 1)} %)"))
            except Exception as exc:
                self.done.emit(0, [f"Téléchargement de l'IA locale impossible : {exc}"])
                return
        for i, f in enumerate(self.files):
            if self.cancelled:
                break
            self.progress.emit(i, f.name)
            try:
                o = self.overrides.get(str(f))
                s = P.Settings(**{**self.settings.__dict__, **o}) if o else self.settings
                P.process_file(f, self.out_dir, s)
                ok += 1
            except Exception as exc:  # on continue avec les suivantes
                errors.append(f"{f.name} : {exc}")
        self.progress.emit(len(self.files), "")
        self.done.emit(ok, errors)


# --------------------------------------------------------------------------- #
# Fenêtre principale
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.qsettings = QSettings("ImageBatch", "ImageBatch")
        self.preview_src: Image.Image | None = None
        self.preview_path: Path | None = None
        self.overrides: dict[str, dict] = {}  # réglages propres à une image (chemin -> champs Settings)
        self.thread: ExportThread | None = None
        self.mask_cache: dict = {}
        self.mask_threads: list = []
        self.mask_pending = None
        self.ai_cache: dict = {}
        self.ai_threads: list = []
        self.ai_pending = None
        self.detect_tags: dict = {}  # chemin -> côté où le fond est ajouté
        self.detect_gen = 0
        self.detect_thread = None
        self.detect_threads: list = []
        self.detect_timer = QTimer(self)
        self.detect_timer.setSingleShot(True)
        self.detect_timer.setInterval(400)
        self.detect_timer.timeout.connect(self._run_detection)

        splitter = QSplitter()
        splitter.addWidget(self._build_file_panel())
        self.preview = PreviewWidget()
        self.preview.cropChanged.connect(self._on_crop_drawn)
        self.preview.retouchDrawn.connect(self._on_retouch_drawn)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.preview, 1)
        self.compare = QCheckBox("Afficher l'original (avant)")
        self.compare.toggled.connect(self.refresh_preview)
        self.info = QLabel("")
        bar = QHBoxLayout()
        bar.addWidget(self.compare)
        bar.addStretch()
        bar.addWidget(self.info)
        cl.addLayout(bar)
        ai_bar = QHBoxLayout()
        self.ai_preview_btn = QPushButton("Générer l'aperçu IA de cette image")
        self.ai_preview_btn.clicked.connect(self._start_ai_preview)
        self.ai_status = QLabel("")
        ai_bar.addWidget(self.ai_preview_btn)
        ai_bar.addWidget(self.ai_status, 1)
        cl.addLayout(ai_bar)
        splitter.addWidget(center)
        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([280, 760, 360])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._update_crop_ui()
        self._update_bg_ui()
        self._update_pad_ui()
        self._update_resize_ui()

    # ---------- panneaux ----------
    def _build_file_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        b_files = QPushButton("Ajouter des images…")
        b_files.clicked.connect(self.add_files_dialog)
        b_folder = QPushButton("Ajouter un dossier…")
        b_folder.clicked.connect(self.add_folder_dialog)
        row.addWidget(b_files)
        row.addWidget(b_folder)
        v.addLayout(row)
        self.recursive = QCheckBox("Inclure les sous-dossiers")
        v.addWidget(self.recursive)
        self.list = QListWidget()
        self.list.setIconSize(QSize(THUMB, THUMB))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentItemChanged.connect(self._on_select)
        v.addWidget(self.list, 1)
        row2 = QHBoxLayout()
        b_rm = QPushButton("Retirer la sélection")
        b_rm.clicked.connect(self.remove_selected)
        b_clear = QPushButton("Tout vider")
        b_clear.clicked.connect(self.clear_files)
        row2.addWidget(b_rm)
        row2.addWidget(b_clear)
        v.addLayout(row2)
        self.count = QLabel("0 image")
        v.addWidget(self.count)
        return w

    def _slider(self, label: str, form: QFormLayout) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(0, 200)
        s.setValue(100)
        val = QLabel("0")
        val.setMinimumWidth(32)
        s.valueChanged.connect(lambda x: val.setText(f"{x - 100:+d}" if x != 100 else "0"))
        s.valueChanged.connect(self.refresh_preview)
        s.mouseDoubleClickEvent = lambda _e: s.setValue(100)
        h = QHBoxLayout()
        h.addWidget(s, 1)
        h.addWidget(val)
        form.addRow(label, h)
        return s

    def _build_settings_panel(self) -> QWidget:
        inner = QWidget()
        v = QVBoxLayout(inner)

        # Recadrage
        g = self.sec_crop = Section("Recadrage")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.crop_mode = QComboBox()
        self.crop_mode.addItem("Aucun", "none")
        self.crop_mode.addItem("Centré selon un ratio", "ratio")
        self.crop_mode.addItem("Zone dessinée (même zone pour toutes)", "manual")
        self.crop_mode.currentIndexChanged.connect(self._update_crop_ui)
        f.addRow("Mode", self.crop_mode)
        self.crop_ratio = QComboBox()
        self.crop_ratio.addItem("Libre", None)
        for k, r in P.ASPECT_RATIOS.items():
            self.crop_ratio.addItem(k, r)
        self.crop_ratio.setCurrentIndex(1)
        self.crop_ratio.currentIndexChanged.connect(self._on_ratio_changed)
        f.addRow("Ratio", self.crop_ratio)
        self.crop_hint = QLabel("Dessinez la zone sur l'aperçu, puis déplacez-la.")
        self.crop_hint.setWordWrap(True)
        self.crop_reset = QPushButton("Réinitialiser la zone")
        self.crop_reset.clicked.connect(self._reset_crop)
        f.addRow(self.crop_hint)
        f.addRow(self.crop_reset)
        v.addWidget(g)

        # Remplacement du fond
        g = self.sec_bg = Section("Remplacer le fond (détourage)")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.bg_enabled = QCheckBox("Activer")
        self.bg_enabled.toggled.connect(self._update_bg_ui)
        f.addRow(self.bg_enabled)
        self.bg_model = QComboBox()
        for key, m in BG.MODELS.items():
            self.bg_model.addItem(m[0], key)
        self.bg_model.currentIndexChanged.connect(self.refresh_preview)
        f.addRow("Modèle", self.bg_model)
        self.bg_fill = QComboBox()
        for label, key in [("Transparent", "transparent"), ("Couleur unie", "color"),
                           ("Image…", "image"), ("Original flouté", "blur")]:
            self.bg_fill.addItem(label, key)
        self.bg_fill.currentIndexChanged.connect(self._update_bg_ui)
        f.addRow("Nouveau fond", self.bg_fill)
        self.bg_color = QColor(255, 255, 255)
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.clicked.connect(self._choose_bg_color)
        self.bg_color_label = QLabel("Couleur")
        self.bg_color_hex = QLineEdit()
        self.bg_color_hex.setPlaceholderText("#FFFFFF ou 255,255,255")
        self.bg_color_hex.setToolTip("Tapez un code couleur : #RRGGBB, RRGGBB, #RGB ou R,V,B, puis Entrée")
        self.bg_color_hex.editingFinished.connect(lambda: self._typed_color("bg"))
        self.bg_color_box = QWidget()
        hb = QHBoxLayout(self.bg_color_box)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(self.bg_color_btn, 1)
        hb.addWidget(self.bg_color_hex, 1)
        f.addRow(self.bg_color_label, self.bg_color_box)
        self.bg_image_btn = QPushButton("Choisir une image de fond…")
        self.bg_image_btn.clicked.connect(self._choose_bg_image)
        self.bg_image_path = ""
        self.bg_image_label = QLabel("Image")
        f.addRow(self.bg_image_label, self.bg_image_btn)
        self.bg_feather = QSlider(Qt.Horizontal)
        self.bg_feather.setRange(0, 10)
        self.bg_feather.setValue(1)
        self.bg_feather.valueChanged.connect(self.refresh_preview)
        f.addRow("Adoucir le contour", self.bg_feather)
        trim_row = QHBoxLayout()
        self.bg_trim = QCheckBox("Recadrer sur le sujet, marge")
        self.bg_trim.toggled.connect(self.refresh_preview)
        self.bg_margin = QSpinBox(); self.bg_margin.setRange(0, 100); self.bg_margin.setValue(10)
        self.bg_margin.setSuffix(" %")
        self.bg_margin.valueChanged.connect(self.refresh_preview)
        trim_row.addWidget(self.bg_trim)
        trim_row.addWidget(self.bg_margin)
        f.addRow(trim_row)
        self.bg_ai_cb = QCheckBox("Créer le nouveau fond avec l'IA (OpenAI)")
        self.bg_ai_cb.toggled.connect(self._update_bg_ui)
        f.addRow(self.bg_ai_cb)
        self.bg_prompt = QLineEdit(self.qsettings.value("bg_prompt", ""))
        self.bg_prompt.setPlaceholderText("ex. table en marbre blanc, lumière douce de studio")
        self.bg_prompt.editingFinished.connect(
            lambda: self.qsettings.setValue("bg_prompt", self.bg_prompt.text()))
        self.bg_prompt_label = QLabel("Décrire le fond")
        f.addRow(self.bg_prompt_label, self.bg_prompt)
        self.bg_status = QLabel("")
        self.bg_status.setWordWrap(True)
        f.addRow(self.bg_status)
        self.bg_widgets = [self.bg_model, self.bg_fill, self.bg_color_btn, self.bg_color_hex, self.bg_image_btn,
                           self.bg_feather, self.bg_trim, self.bg_margin, self.bg_ai_cb, self.bg_prompt]
        v.addWidget(g)

        # Changement de ratio avec ajout de fond
        g = self.sec_pad = Section("Changer le ratio (ajouter du fond)")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.pad_enabled = QCheckBox("Activer")
        self.pad_enabled.toggled.connect(self._update_pad_ui)
        f.addRow(self.pad_enabled)
        self.pad_ratio = QComboBox()
        for k, r in P.ASPECT_RATIOS.items():
            self.pad_ratio.addItem(k, r)
        self.pad_ratio.currentIndexChanged.connect(self.refresh_preview)
        f.addRow("Ratio", self.pad_ratio)
        self.pad_fill = QComboBox()
        for label, key in [("Fond généré propre (sans étirement)", "generated"),
                           ("Prolonger le bord (étire les pixels)", "edge"), ("Couleur du fond détectée", "auto_color"),
                           ("Couleur unie", "color"), ("Image floutée", "blur"),
                           ("Transparent (PNG/WEBP)", "transparent")]:
            self.pad_fill.addItem(label, key)
        self.pad_fill.currentIndexChanged.connect(self._update_pad_ui)
        f.addRow("Fond", self.pad_fill)
        self.pad_position = QComboBox()
        for label, key in [("Auto : côté opposé au produit", "auto"), ("Des deux côtés (centré)", "center"),
                           ("En haut / à gauche", "start"), ("En bas / à droite", "end")]:
            self.pad_position.addItem(label, key)
        self.pad_position.setToolTip("Auto détecte si le produit touche le haut ou le bas (ou un côté)\n"
                                     "par la couleur des bords, et ajoute le fond uniquement à l'opposé.")
        self.pad_position.currentIndexChanged.connect(self.refresh_preview)
        f.addRow("Ajouter le fond (lot)", self.pad_position)
        self.pad_position_img = QComboBox()
        self.pad_position_img.addItem("Comme le lot", None)
        for label, key in [("Auto : côté opposé au produit", "auto"), ("Des deux côtés (centré)", "center"),
                           ("En haut / à gauche", "start"), ("En bas / à droite", "end")]:
            self.pad_position_img.addItem(label, key)
        self.pad_position_img.setToolTip("Réglage propre à l'image sélectionnée (ou aux images sélectionnées),\n"
                                         "prioritaire sur le réglage du lot. Les images concernées sont marquées ✎.")
        self.pad_position_img.currentIndexChanged.connect(self._on_image_position)
        f.addRow("Pour cette image", self.pad_position_img)
        self.clear_overrides_btn = QPushButton("Remettre toutes les images comme le lot")
        self.clear_overrides_btn.clicked.connect(self.clear_overrides)
        f.addRow(self.clear_overrides_btn)
        self.pad_color = QColor(255, 255, 255)
        self.pad_color_btn = QPushButton()
        self.pad_color_btn.clicked.connect(self._choose_pad_color)
        self.pad_color_label = QLabel("Couleur")
        self.pad_color_hex = QLineEdit()
        self.pad_color_hex.setPlaceholderText("#FFFFFF ou 255,255,255")
        self.pad_color_hex.setToolTip("Tapez un code couleur : #RRGGBB, RRGGBB, #RGB ou R,V,B, puis Entrée")
        self.pad_color_hex.editingFinished.connect(lambda: self._typed_color("pad"))
        self.pad_color_box = QWidget()
        hb = QHBoxLayout(self.pad_color_box)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(self.pad_color_btn, 1)
        hb.addWidget(self.pad_color_hex, 1)
        f.addRow(self.pad_color_label, self.pad_color_box)
        self.pad_blur = QSlider(Qt.Horizontal)
        self.pad_blur.setRange(5, 120)
        self.pad_blur.setValue(40)
        self.pad_blur.valueChanged.connect(self.refresh_preview)
        self.pad_blur_label = QLabel("Intensité du flou")
        f.addRow(self.pad_blur_label, self.pad_blur)
        self.ai_local_cb = QCheckBox("Agrandir avec l'IA locale (gratuite, hors ligne)")
        self.ai_openai_cb = QCheckBox("Agrandir avec l'IA OpenAI (clé API, payant)")
        self.ai_local_cb.toggled.connect(lambda on: self._on_ai_toggled(self.ai_local_cb, on))
        self.ai_openai_cb.toggled.connect(lambda on: self._on_ai_toggled(self.ai_openai_cb, on))
        f.addRow(self.ai_local_cb)
        f.addRow(self.ai_openai_cb)
        self.pad_widgets = [self.pad_ratio, self.pad_fill, self.pad_position, self.pad_position_img,
                            self.clear_overrides_btn, self.pad_color_btn, self.pad_color_hex, self.pad_blur,
                            self.ai_local_cb, self.ai_openai_cb]
        v.addWidget(g)

        # Retouche IA d'une zone, image par image
        g = self.sec_retouch = Section("Retouche IA d'une zone")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.retouch_btn = QPushButton("Sélectionner une zone sur l'aperçu")
        self.retouch_btn.setCheckable(True)
        self.retouch_btn.toggled.connect(self._on_retouch_mode)
        f.addRow(self.retouch_btn)
        self.retouch_engine = QComboBox()
        self.retouch_engine.addItem("IA locale (gratuite, hors ligne)", "local")
        self.retouch_engine.addItem("IA OpenAI (clé API, payant)", "openai")
        self.retouch_engine.currentIndexChanged.connect(self.refresh_preview)
        f.addRow("Moteur", self.retouch_engine)
        row = QHBoxLayout()
        self.retouch_undo = QPushButton("Annuler la dernière zone")
        self.retouch_undo.clicked.connect(self._retouch_undo)
        self.retouch_clear = QPushButton("Tout effacer")
        self.retouch_clear.clicked.connect(self._retouch_clear)
        row.addWidget(self.retouch_undo)
        row.addWidget(self.retouch_clear)
        f.addRow(row)
        note = QLabel("Tracez un rectangle sur la trace à gommer : l'IA la regénère, pour cette image "
                      "seulement (marquée ✎), à l'aperçu et à l'export. Faites les retouches en dernier : "
                      "si vous changez le recadrage ou le ratio ensuite, les zones ne tombent plus au bon endroit.")
        note.setWordWrap(True)
        f.addRow(note)
        v.addWidget(g)

        # Réglages OpenAI, communs au fond IA et à l'agrandissement IA
        g = self.sec_openai = Section("IA OpenAI (clé API)")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.openai_key = QLineEdit(self.qsettings.value("openai_key", ""))
        self.openai_key.setEchoMode(QLineEdit.Password)
        self.openai_key.setPlaceholderText("sk-… (platform.openai.com)")
        self.openai_key.editingFinished.connect(
            lambda: self.qsettings.setValue("openai_key", self.openai_key.text().strip()))
        self.openai_key.editingFinished.connect(self._update_ai_bar)
        f.addRow("Clé API", self.openai_key)
        self.openai_quality = QComboBox()
        for label, key in [("Haute", "high"), ("Moyenne (moins chère)", "medium"), ("Basse", "low")]:
            self.openai_quality.addItem(label, key)
        f.addRow("Qualité", self.openai_quality)
        note = QLabel("Utilisée par « Créer le nouveau fond avec l'IA » et « Agrandir avec l'IA OpenAI ». "
                      "Les images concernées sont envoyées à OpenAI (≈ 0,05 à 0,20 € par appel).")
        note.setWordWrap(True)
        f.addRow(note)
        v.addWidget(g)

        # Redimensionnement
        g = self.sec_resize = Section("Redimensionnement")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.resize_mode = QComboBox()
        for label, key in [("Aucun", "none"), ("Largeur fixe", "width"), ("Hauteur fixe", "height"),
                           ("Tenir dans un cadre", "fit"), ("Dimensions exactes (déforme)", "exact"),
                           ("Pourcentage", "percent")]:
            self.resize_mode.addItem(label, key)
        self.resize_mode.currentIndexChanged.connect(self._update_resize_ui)
        f.addRow("Mode", self.resize_mode)
        self.w_spin = QSpinBox(); self.w_spin.setRange(1, 20000); self.w_spin.setValue(1920); self.w_spin.setSuffix(" px")
        self.h_spin = QSpinBox(); self.h_spin.setRange(1, 20000); self.h_spin.setValue(1080); self.h_spin.setSuffix(" px")
        self.pct_spin = QSpinBox(); self.pct_spin.setRange(1, 1000); self.pct_spin.setValue(50); self.pct_spin.setSuffix(" %")
        for sp in (self.w_spin, self.h_spin, self.pct_spin):
            sp.valueChanged.connect(self.refresh_preview)
        self.w_label, self.h_label, self.pct_label = QLabel("Largeur"), QLabel("Hauteur"), QLabel("Échelle")
        f.addRow(self.w_label, self.w_spin)
        f.addRow(self.h_label, self.h_spin)
        f.addRow(self.pct_label, self.pct_spin)
        self.no_upscale = QCheckBox("Ne jamais agrandir")
        self.no_upscale.setChecked(True)
        self.no_upscale.toggled.connect(self.refresh_preview)
        f.addRow(self.no_upscale)
        v.addWidget(g)

        # Retouches
        g = self.sec_edit = Section("Retouches")
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.rotate = QComboBox()
        for label, deg in [("0°", 0), ("90° horaire", 90), ("180°", 180), ("90° anti-horaire", 270)]:
            self.rotate.addItem(label, deg)
        self.rotate.currentIndexChanged.connect(self.refresh_preview)
        f.addRow("Rotation", self.rotate)
        flips = QHBoxLayout()
        self.flip_h = QCheckBox("Miroir horizontal")
        self.flip_v = QCheckBox("Miroir vertical")
        for c in (self.flip_h, self.flip_v):
            c.toggled.connect(self.refresh_preview)
            flips.addWidget(c)
        f.addRow(flips)
        self.brightness = self._slider("Luminosité", f)
        self.contrast = self._slider("Contraste", f)
        self.saturation = self._slider("Saturation", f)
        self.sharpness = self._slider("Netteté", f)
        self.grayscale = QCheckBox("Noir et blanc")
        self.grayscale.toggled.connect(self.refresh_preview)
        f.addRow(self.grayscale)
        b_reset = QPushButton("Réinitialiser les retouches")
        b_reset.clicked.connect(self._reset_edits)
        f.addRow(b_reset)
        v.addWidget(g)

        # Export
        g = self.sec_export = Section("Export", collapsible=False)
        f = QFormLayout(g.body)
        f.setContentsMargins(8, 4, 4, 10)
        self.out_format = QComboBox()
        self.out_format.addItems(list(P.OUTPUT_FORMATS))
        self.out_format.currentIndexChanged.connect(self._update_quality_ui)
        f.addRow("Format", self.out_format)
        self.quality = QSpinBox(); self.quality.setRange(1, 100); self.quality.setValue(90)
        self.quality_label = QLabel("Qualité")
        f.addRow(self.quality_label, self.quality)
        self.suffix = QLineEdit("_edit")
        self.suffix.setPlaceholderText("ex. _web (vide = même nom)")
        f.addRow("Suffixe du nom", self.suffix)
        self.keep_meta = QCheckBox("Conserver EXIF / profil couleur")
        f.addRow(self.keep_meta)
        out_row = QHBoxLayout()
        self.out_dir = QLineEdit(self.qsettings.value("out_dir", str(Path.home() / "Pictures" / "Batch Cooking")))
        b_out = QPushButton("…")
        b_out.setFixedWidth(32)
        b_out.clicked.connect(self.choose_out_dir)
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(b_out)
        f.addRow("Dossier", out_row)
        v.addWidget(g)
        v.addStretch()

        self._update_quality_ui()
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        # Barre d'export toujours visible sous les réglages
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        btns = QHBoxLayout()
        self.export_btn = QPushButton("Exporter toutes les images")
        self.export_btn.setDefault(True)
        self.export_btn.setMinimumHeight(36)
        self.export_btn.clicked.connect(self.export)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_export)
        btns.addWidget(self.export_btn, 1)
        btns.addWidget(self.cancel_btn)
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(scroll, 1)
        pl.addWidget(self.progress)
        pl.addLayout(btns)
        panel.setMinimumWidth(340)
        return panel

    def _build_menu(self):
        m = self.menuBar().addMenu("Fichier")
        for text, slot, key in [("Ajouter des images…", self.add_files_dialog, "Ctrl+O"),
                                ("Ajouter un dossier…", self.add_folder_dialog, "Ctrl+Shift+O"),
                                ("Exporter", self.export, "Ctrl+E")]:
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)

    # ---------- réglages ----------
    def settings_for(self, path) -> P.Settings:
        """Réglages du lot, complétés par ceux propres à l'image."""
        s = self.current_settings()
        o = self.overrides.get(str(path)) if path is not None else None
        return P.Settings(**{**s.__dict__, **o}) if o else s

    def _on_image_position(self, *_):
        if self.preview_path is None:
            return
        value = self.pad_position_img.currentData()
        items = self.list.selectedItems() or [self.list.currentItem()]
        for item in items:
            if item is None:
                continue
            key = item.data(Qt.UserRole)
            self._set_override(key, "pad_position", value)
            self._mark_item(item)
        self.refresh_preview()

    def _set_override(self, key: str, field: str, value):
        o = self.overrides.setdefault(key, {})
        if value is None or value == ():
            o.pop(field, None)
        else:
            o[field] = value
        if not o:
            self.overrides.pop(key, None)

    # ---------- retouche IA ----------
    def _on_retouch_mode(self, on):
        self.preview.retouch_mode = on
        if on and self.compare.isChecked():
            self.compare.setChecked(False)
        self.retouch_btn.setText("Terminer la sélection" if on else "Sélectionner une zone sur l'aperçu")
        if not on:
            self.preview.setCursor(Qt.ArrowCursor)

    def _retouches(self) -> tuple:
        return tuple(self.overrides.get(str(self.preview_path), {}).get("retouch", ()))

    def _set_retouches(self, rects: tuple):
        if self.preview_path is None:
            return
        self._set_override(str(self.preview_path), "retouch", tuple(rects))
        item = self.list.currentItem()
        if item is not None:
            self._mark_item(item)
        self.refresh_preview()
        if rects:
            self._start_ai_preview()

    def _on_retouch_drawn(self, rect):
        self._set_retouches(self._retouches() + (tuple(round(v, 5) for v in rect),))

    def _retouch_undo(self):
        self._set_retouches(self._retouches()[:-1])

    def _retouch_clear(self):
        self._set_retouches(())

    def _mark_item(self, item):
        key = item.data(Qt.UserRole)
        tag = PAD_TAGS.get(self.detect_tags.get(key, ""), "")
        item.setText(Path(key).name + (f"   {tag}" if tag else "") + ("  ✎" if key in self.overrides else ""))

    def _run_detection(self):
        """Affiche dans la liste le côté où le fond sera ajouté, calculé image par image."""
        if self.detect_thread is not None:
            self.detect_thread.cancelled = True
        self.detect_gen += 1
        if not self.current_settings().pad_enabled:
            self.detect_tags.clear()
            for i in range(self.list.count()):
                self._mark_item(self.list.item(i))
            return
        jobs = []
        for i in range(self.list.count()):
            key = self.list.item(i).data(Qt.UserRole)
            s = self.settings_for(key)
            jobs.append((key, P.Settings(**{**s.__dict__, "bg_enabled": False, "retouch": (),
                                            "pad_fill": "color", "resize_mode": "none"})))
        th = DetectThread(self.detect_gen, jobs)
        th.found.connect(self._on_detected)
        th.finished.connect(lambda th=th: self.detect_threads.remove(th) if th in self.detect_threads else None)
        self.detect_threads.append(th)
        self.detect_thread = th
        th.start()

    def _on_detected(self, gen, path, tag):
        if gen != self.detect_gen:
            return
        self.detect_tags[path] = tag
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == path:
                self._mark_item(item)
                break

    def _sync_image_position(self):
        o = self.overrides.get(str(self.preview_path), {}) if self.preview_path else {}
        self.pad_position_img.blockSignals(True)
        self.pad_position_img.setCurrentIndex(max(0, self.pad_position_img.findData(o.get("pad_position"))))
        self.pad_position_img.blockSignals(False)

    def clear_overrides(self):
        self.overrides.clear()
        for i in range(self.list.count()):
            self._mark_item(self.list.item(i))
        self._sync_image_position()
        self.refresh_preview()

    def current_settings(self) -> P.Settings:
        ratio = self.crop_ratio.currentData()
        return P.Settings(
            crop_mode=self.crop_mode.currentData(),
            crop_ratio=tuple(ratio) if ratio else (1, 1),
            crop_rect=self.preview.crop,
            bg_enabled=self.bg_enabled.isChecked(),
            bg_model=self.bg_model.currentData(),
            bg_fill=self.bg_fill.currentData(),
            bg_color=(self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()),
            bg_image=self.bg_image_path,
            bg_feather=self.bg_feather.value(),
            bg_trim=self.bg_trim.isChecked(),
            bg_margin=self.bg_margin.value(),
            bg_ai=self.bg_ai_cb.isChecked(),
            bg_prompt=self.bg_prompt.text().strip(),
            pad_enabled=self.pad_enabled.isChecked(),
            pad_ratio=tuple(self.pad_ratio.currentData()),
            pad_fill=("ai_local" if self.ai_local_cb.isChecked() else
                      "ai_openai" if self.ai_openai_cb.isChecked() else self.pad_fill.currentData()),
            pad_position=self.pad_position.currentData(),
            openai_key=self.openai_key.text().strip(),
            openai_quality=self.openai_quality.currentData(),
            retouch_engine=self.retouch_engine.currentData() if hasattr(self, "retouch_engine") else "local",
            pad_color=(self.pad_color.red(), self.pad_color.green(), self.pad_color.blue()),
            pad_blur=self.pad_blur.value(),
            resize_mode=self.resize_mode.currentData(),
            width=self.w_spin.value(), height=self.h_spin.value(), percent=self.pct_spin.value(),
            no_upscale=self.no_upscale.isChecked(),
            rotate=self.rotate.currentData(),
            flip_h=self.flip_h.isChecked(), flip_v=self.flip_v.isChecked(),
            brightness=self.brightness.value() / 100, contrast=self.contrast.value() / 100,
            saturation=self.saturation.value() / 100, sharpness=self.sharpness.value() / 100,
            grayscale=self.grayscale.isChecked(),
            out_format=self.out_format.currentText(), quality=self.quality.value(),
            suffix=self.suffix.text(), keep_metadata=self.keep_meta.isChecked(),
        )

    def _update_crop_ui(self):
        mode = self.crop_mode.currentData()
        self.crop_ratio.setEnabled(mode != "none")
        # "Libre" n'a de sens qu'en mode zone dessinée
        item = self.crop_ratio.model().item(0)
        item.setEnabled(mode == "manual")
        if mode == "ratio" and self.crop_ratio.currentIndex() == 0:
            self.crop_ratio.setCurrentIndex(1)
        self.crop_hint.setVisible(mode == "manual")
        self.crop_reset.setVisible(mode == "manual")
        self._on_ratio_changed()

    def _on_ratio_changed(self):
        r = self.crop_ratio.currentData()
        self.preview.ratio = (r[0] / r[1]) if r else None
        self.refresh_preview()

    def _on_crop_drawn(self, _rect):
        self.refresh_info()

    def _reset_crop(self):
        self.preview.crop = (0.0, 0.0, 1.0, 1.0)
        self.refresh_preview()

    def _update_bg_ui(self):
        on = self.bg_enabled.isChecked()
        err = BG.available() if on else None
        for w in self.bg_widgets:
            w.setEnabled(on and not err)
        fill = self.bg_fill.currentData()
        bg_ai = self.bg_ai_cb.isChecked()
        self.bg_fill.setEnabled(on and not err and not bg_ai)  # l'IA remplace le choix de fond
        for wdg, vis in [(self.bg_color_box, fill == "color" and not bg_ai),
                         (self.bg_color_label, fill == "color" and not bg_ai),
                         (self.bg_image_btn, fill == "image" and not bg_ai),
                         (self.bg_image_label, fill == "image" and not bg_ai),
                         (self.bg_prompt, bg_ai), (self.bg_prompt_label, bg_ai)]:
            wdg.setVisible(vis)
        self.bg_color_btn.setText(self.bg_color.name().upper())
        if not self.bg_color_hex.hasFocus():
            self.bg_color_hex.setText(self.bg_color.name().upper())
        self.bg_color_btn.setStyleSheet(
            f"background:{self.bg_color.name()}; color:{'#000' if self.bg_color.lightness() > 128 else '#fff'};")
        if err:
            self.bg_status.setText(err)
        elif on and not BG.is_downloaded(self.bg_model.currentData()):
            self.bg_status.setText("Le modèle sera téléchargé au premier usage (une seule fois).")
        else:
            self.bg_status.setText("")
        self._update_ai_bar()
        self.refresh_preview()

    def _choose_bg_color(self):
        c = QColorDialog.getColor(self.bg_color, self, "Couleur du nouveau fond")
        if c.isValid():
            self.bg_color = c
            self._update_bg_ui()

    def _choose_bg_image(self):
        exts = " ".join(f"*{e}" for e in sorted(P.INPUT_EXTENSIONS))
        f, _ = QFileDialog.getOpenFileName(self, "Image de fond", self.qsettings.value("in_dir", str(Path.home())),
                                           f"Images ({exts})")
        if f:
            self.bg_image_path = f
            self.bg_image_btn.setText(Path(f).name)
            self.refresh_preview()

    def _mask_key(self, s: P.Settings):
        return (str(self.preview_path), s.bg_model, s.crop_mode, s.crop_ratio, s.crop_rect,
                s.rotate, s.flip_h, s.flip_v)

    def _preview_mask_fn(self, s: P.Settings):
        """Renvoie une fonction masque si le masque est prêt, sinon lance le calcul."""
        key = self._mask_key(s)
        if key in self.mask_cache:
            mask = self.mask_cache[key]
            return lambda _img: mask
        if self.mask_pending != key and BG.available() is None:
            self.mask_pending = key
            th = MaskThread(self.preview_src, s, key)
            th.status.connect(self.bg_status.setText)
            th.done.connect(self._on_mask_done)
            th.failed.connect(self._on_mask_failed)
            th.finished.connect(lambda th=th: self.mask_threads.remove(th) if th in self.mask_threads else None)
            self.mask_threads.append(th)
            th.start()
        return None

    def _on_mask_done(self, key, mask):
        if len(self.mask_cache) > 30:
            self.mask_cache.clear()
        self.mask_cache[key] = mask
        if self.mask_pending == key:
            self.mask_pending = None
        self.bg_status.setText("")
        self.refresh_preview()

    def _on_mask_failed(self, key, msg):
        if self.mask_pending == key:
            self.mask_pending = None
        self.bg_status.setText(f"Échec du détourage : {msg}")

    def _ai_key(self, s: P.Settings):
        d = {k: v for k, v in s.__dict__.items()
             if k not in ("resize_mode", "width", "height", "percent", "no_upscale", "out_format",
                          "quality", "suffix", "keep_metadata", "openai_key")}
        return (str(self.preview_path), repr(sorted(d.items())))

    def _ai_active(self, s: P.Settings) -> bool:
        return ((s.bg_enabled and s.bg_ai) or (s.pad_enabled and s.pad_fill in ("ai_local", "ai_openai"))
                or bool(s.retouch))

    def _uses_openai(self, s: P.Settings) -> bool:
        return ((s.bg_enabled and s.bg_ai) or (s.pad_enabled and s.pad_fill == "ai_openai")
                or (bool(s.retouch) and s.retouch_engine == "openai"))

    def _update_ai_bar(self):
        if not hasattr(self, "ai_preview_btn") or not hasattr(self, "openai_key"):
            return
        s = self.current_settings()
        active = self._ai_active(s)
        self.ai_preview_btn.setVisible(active)
        self.ai_status.setVisible(active)
        if active and self.ai_pending is None:
            msg = "L'IA est appliquée à l'export ; aperçu IA sur demande."
            if self._uses_openai(s) and not s.openai_key:
                msg = "Collez votre clé dans le volet « IA OpenAI (clé API) »."
            self.ai_status.setText(msg)

    def _start_ai_preview(self, *_):
        if self.preview_src is None:
            return
        s = P.Settings(**{**self.settings_for(self.preview_path).__dict__, "resize_mode": "none"})
        mask_fn = None
        if s.bg_enabled:
            mask_fn = self._preview_mask_fn(s)
            if mask_fn is None:
                self.ai_status.setText("Détourage en cours, réessayez dans un instant.")
                return
        if self._uses_openai(s) and not s.openai_key:
            self.ai_status.setText("Collez d'abord votre clé dans le volet « IA OpenAI (clé API) ».")
            self.sec_openai.header.setChecked(True)
            return
        if self.ai_pending is not None or BG.available():
            return
        key = self._ai_key(s)
        self.ai_pending = key
        self.ai_preview_btn.setEnabled(False)
        th = AiPreviewThread(self.preview_src, s, key, mask_fn)
        th.status.connect(self.ai_status.setText)
        th.done.connect(self._on_ai_done)
        th.failed.connect(self._on_ai_failed)
        th.finished.connect(lambda th=th: self.ai_threads.remove(th) if th in self.ai_threads else None)
        self.ai_threads.append(th)
        th.start()

    def _on_ai_done(self, key, out):
        if len(self.ai_cache) > 20:
            self.ai_cache.clear()
        if out is not None:
            self.ai_cache[key] = out
        if self.ai_pending == key:
            self.ai_pending = None
        self.ai_preview_btn.setEnabled(True)
        self.ai_status.setText("Aperçu IA prêt.")
        # une zone a été ajoutée pendant le calcul : on relance pour l'inclure
        if self.preview_path is not None and self._retouches():
            cur = self._ai_key(P.Settings(**{**self.settings_for(self.preview_path).__dict__, "resize_mode": "none"}))
            if cur != key and cur not in self.ai_cache:
                QTimer.singleShot(0, self._start_ai_preview)
        self.refresh_preview()

    def _on_ai_failed(self, key, msg):
        if self.ai_pending == key:
            self.ai_pending = None
        self.ai_preview_btn.setEnabled(True)
        self.ai_status.setText(f"Échec de l'IA : {msg}")

    def _on_ai_toggled(self, box, on):
        other = self.ai_openai_cb if box is self.ai_local_cb else self.ai_local_cb
        if on and other.isChecked():
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        self._update_pad_ui()

    def _update_pad_ui(self):
        on = self.pad_enabled.isChecked()
        for w in self.pad_widgets:
            w.setEnabled(on)
        fill = self.pad_fill.currentData()
        self.pad_color_box.setVisible(fill == "color")
        self.pad_color_label.setVisible(fill == "color")
        self.pad_blur.setVisible(fill == "blur")
        self.pad_blur_label.setVisible(fill == "blur")
        is_oa = self.ai_openai_cb.isChecked()
        is_ai = is_oa or self.ai_local_cb.isChecked()
        self.pad_fill.setEnabled(on and not is_ai)  # l'IA remplace le choix de fond
        self._update_ai_bar()
        self.pad_color_btn.setText(self.pad_color.name().upper())
        if not self.pad_color_hex.hasFocus():
            self.pad_color_hex.setText(self.pad_color.name().upper())
        self.pad_color_btn.setStyleSheet(
            f"background:{self.pad_color.name()}; color:{'#000' if self.pad_color.lightness() > 128 else '#fff'};")
        self.refresh_preview()

    def _typed_color(self, which: str):
        field = getattr(self, f"{which}_color_hex")
        c = parse_color(field.text())
        if c is None:
            field.setText(getattr(self, f"{which}_color").name().upper())  # code invalide : on revient
            self.statusBar().showMessage("Code couleur non reconnu (ex. #F2F2F2 ou 242,242,242)", 4000)
            return
        setattr(self, f"{which}_color", c)
        field.setText(c.name().upper())
        (self._update_bg_ui if which == "bg" else self._update_pad_ui)()

    def _choose_pad_color(self):
        c = QColorDialog.getColor(self.pad_color, self, "Couleur du fond")
        if c.isValid():
            self.pad_color = c
            self._update_pad_ui()

    def _update_resize_ui(self):
        m = self.resize_mode.currentData()
        show_w = m in ("width", "fit", "exact")
        show_h = m in ("height", "fit", "exact")
        for wdg, vis in [(self.w_label, show_w), (self.w_spin, show_w), (self.h_label, show_h),
                         (self.h_spin, show_h), (self.pct_label, m == "percent"),
                         (self.pct_spin, m == "percent"), (self.no_upscale, m not in ("none", "exact"))]:
            wdg.setVisible(vis)
        self.refresh_preview()

    def _update_quality_ui(self):
        fmt = self.out_format.currentText()
        vis = fmt in ("JPEG", "WEBP", "Identique")
        self.quality.setVisible(vis)
        self.quality_label.setVisible(vis)

    def _reset_edits(self):
        for s in (self.brightness, self.contrast, self.saturation, self.sharpness):
            s.setValue(100)
        self.rotate.setCurrentIndex(0)
        self.flip_h.setChecked(False)
        self.flip_v.setChecked(False)
        self.grayscale.setChecked(False)

    # ---------- fichiers ----------
    def add_paths(self, paths):
        existing = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        added = 0
        for p in paths:
            p = Path(p)
            files = P.collect_images(p, self.recursive.isChecked()) if p.is_dir() else [p]
            for f in files:
                if f.suffix.lower() not in P.INPUT_EXTENSIONS or str(f) in existing:
                    continue
                item = QListWidgetItem(f.name)
                item.setData(Qt.UserRole, str(f))
                item.setToolTip(str(f))
                item.setIcon(self._thumb(f))
                self.list.addItem(item)
                existing.add(str(f))
                added += 1
                if added % 20 == 0:
                    QApplication.processEvents()
        self._update_count()
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(0)
        self.statusBar().showMessage(f"{added} image(s) ajoutée(s)", 4000)

    def _thumb(self, path: Path) -> QIcon:
        try:
            with Image.open(path) as im:
                im.draft("RGB", (THUMB * 2, THUMB * 2))
                im = ImageOps.exif_transpose(im)
                im.thumbnail((THUMB, THUMB))
                return QIcon(QPixmap.fromImage(pil_to_qimage(im)))
        except Exception:
            return QIcon()

    def add_files_dialog(self):
        exts = " ".join(f"*{e}" for e in sorted(P.INPUT_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des images",
                                                self.qsettings.value("in_dir", str(Path.home())),
                                                f"Images ({exts})")
        if files:
            self.qsettings.setValue("in_dir", str(Path(files[0]).parent))
            self.add_paths(files)

    def add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier",
                                             self.qsettings.value("in_dir", str(Path.home())))
        if d:
            self.qsettings.setValue("in_dir", d)
            self.add_paths([d])

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.overrides.pop(item.data(Qt.UserRole), None)
            self.list.takeItem(self.list.row(item))
        self._update_count()
        if not self.list.count():
            self._clear_preview()

    def clear_files(self):
        self.list.clear()
        self.overrides.clear()
        self._update_count()
        self._clear_preview()

    def _update_count(self):
        n = self.list.count()
        self.count.setText(f"{n} image{'s' if n > 1 else ''}")

    def _clear_preview(self):
        self.preview_src = None
        self.preview_path = None
        self.preview.set_image(None)
        self.info.setText("")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier d'export", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    # ---------- aperçu ----------
    def _on_select(self, item, _prev=None):
        if not item:
            return
        path = Path(item.data(Qt.UserRole))
        try:
            with Image.open(path) as im:
                self.orig_size = ImageOps.exif_transpose(im).size if im.getexif().get(0x0112, 1) > 4 else im.size
                im.draft("RGB", (PREVIEW_MAX, PREVIEW_MAX))
                im = ImageOps.exif_transpose(im)
                im.load()
                im.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
                self.preview_src = im.copy()
                self.preview_path = path
                self._sync_image_position()
        except Exception as exc:
            self.preview_src = None
            self.preview.message = f"Impossible d'ouvrir {path.name} : {exc}"
            self.preview.set_image(None)
            return
        self.refresh_preview()

    def _update_section_markers(self, s: P.Settings):
        if not hasattr(self, "sec_export"):
            return
        self.sec_crop.set_active(s.crop_mode != "none")
        self.sec_bg.set_active(s.bg_enabled)
        self.sec_pad.set_active(s.pad_enabled)
        self.sec_retouch.set_active(bool(s.retouch))
        self.sec_resize.set_active(s.resize_mode != "none")
        self.sec_edit.set_active(bool(s.rotate or s.flip_h or s.flip_v or s.grayscale) or any(
            abs(x - 1.0) > 1e-3 for x in (s.brightness, s.contrast, s.saturation, s.sharpness)))
        self.sec_export.set_active(False)

    def refresh_preview(self, *_):
        if not hasattr(self, "preview") or not hasattr(self, "export_btn"):
            return
        self.detect_timer.start()  # recalcule en tâche de fond le côté détecté de chaque image
        s = self.settings_for(self.preview_path)
        self._update_section_markers(s)
        manual = s.crop_mode == "manual"
        self.preview.edit_crop = manual and not self.compare.isChecked()
        if self.preview_src is None:
            self.preview.update()
            return
        cached = None
        if self.compare.isChecked():
            img = self.preview_src
        elif manual:
            # On dessine la zone sur l'image non recadrée et sans rotation
            img = self.preview_src
        else:
            preview_s = P.Settings(**{**s.__dict__, "resize_mode": "none"})
            mask_fn = None
            if s.bg_enabled:
                mask_fn = self._preview_mask_fn(s)
                if mask_fn is None:  # masque pas encore prêt : aperçu sans détourage
                    preview_s.bg_enabled = False
            self.preview_info = {}
            fallback = {"bg_ai": False, "retouch": ()}  # sans IA en attendant l'aperçu IA
            if preview_s.pad_fill in ("ai_local", "ai_openai"):
                fallback["pad_fill"] = "generated"
            img = P.process(self.preview_src, P.Settings(**{**preview_s.__dict__, **fallback}),
                            mask_fn, self.preview_info)
            cached = self.ai_cache.get(self._ai_key(preview_s)) if self._ai_active(preview_s) else None
            if cached is not None:
                img = cached  # aperçu IA déjà calculé pour ces réglages
        retouched = not (self.compare.isChecked() or manual) and s.retouch
        self.preview.retouch_rects = list(s.retouch) if retouched and cached is None else []
        self.preview.set_image(QPixmap.fromImage(pil_to_qimage(img)),
                               self.preview_src.width / self.preview_src.height)
        self.refresh_info()

    def refresh_info(self):
        if self.preview_src is None:
            return
        s = self.settings_for(self.preview_path)
        ow, oh = self.orig_size
        nw, nh = P.output_size((ow, oh), s)
        out = f"{nw}×{nh} px" if nw else "taille selon le sujet"
        note = getattr(self, "preview_info", {}).get("pad") if s.pad_enabled else None
        self.info.setText(f"{self.preview_path.name} — {ow}×{oh} → {out}" + (f" ({note})" if note else ""))

    # ---------- export ----------
    def export(self):
        if self.thread and self.thread.isRunning():
            return
        files = [Path(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())]
        if not files:
            QMessageBox.information(self, APP_NAME, "Ajoutez d'abord des images.")
            return
        out = Path(self.out_dir.text()).expanduser()
        s = self.current_settings()
        if not s.suffix and any(f.parent.resolve() == out.resolve() for f in files if out.exists()):
            # Jamais d'écrasement : output_path ajoute _1, _2… mais on prévient
            if QMessageBox.question(self, APP_NAME,
                                    "Le dossier d'export contient des originaux et aucun suffixe n'est "
                                    "défini. Les copies seront numérotées (_1, _2…). Continuer ?") != QMessageBox.Yes:
                return
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Dossier d'export invalide : {exc}")
            return
        self.qsettings.setValue("out_dir", str(out))
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.export_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.thread = ExportThread(files, out, s, dict(self.overrides))
        self.thread.progress.connect(self._on_progress)
        self.thread.done.connect(self._on_done)
        self.thread.start()

    def cancel_export(self):
        if self.thread:
            self.thread.cancelled = True

    def _on_progress(self, i, name):
        self.progress.setValue(i)
        if name:
            self.statusBar().showMessage(f"Traitement de {name}…")

    def _on_done(self, ok, errors):
        self.progress.setVisible(False)
        self.export_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        out = self.out_dir.text()
        msg = f"{ok} image(s) exportée(s) dans :\n{out}"
        if errors:
            msg += f"\n\n{len(errors)} erreur(s) :\n" + "\n".join(errors[:15])
        self.statusBar().showMessage(f"{ok} image(s) exportée(s)", 8000)
        box = QMessageBox(QMessageBox.Warning if errors else QMessageBox.Information, "Export terminé", msg,
                          parent=self)
        done_pic = QPixmap(str(asset_path("fin.jpg")))
        if not done_pic.isNull():  # image de fin de tâche
            box.setIconPixmap(done_pic.scaledToHeight(180, Qt.SmoothTransformation))
        open_btn = box.addButton("Ouvrir le dossier", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() == open_btn:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))


def apply_dark_theme(app: QApplication) -> None:
    """Thème sombre, identique sur tous les systèmes."""
    from PySide6.QtGui import QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    base, window, text = QColor(30, 30, 32), QColor(42, 42, 45), QColor(230, 230, 230)
    accent = QColor(64, 128, 230)
    for role, color in [
        (QPalette.Window, window), (QPalette.WindowText, text), (QPalette.Base, base),
        (QPalette.AlternateBase, QColor(48, 48, 52)), (QPalette.ToolTipBase, QColor(55, 55, 60)),
        (QPalette.ToolTipText, text), (QPalette.Text, text), (QPalette.Button, QColor(55, 55, 60)),
        (QPalette.ButtonText, text), (QPalette.BrightText, QColor(255, 90, 90)),
        (QPalette.Highlight, accent), (QPalette.HighlightedText, QColor(255, 255, 255)),
        (QPalette.Link, accent), (QPalette.PlaceholderText, QColor(140, 140, 140)),
    ]:
        pal.setColor(role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(115, 115, 115))
    pal.setColor(QPalette.Disabled, QPalette.Button, QColor(45, 45, 48))
    app.setPalette(pal)
    app.setStyleSheet("QToolTip { border: 1px solid #555; }")


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    icon = asset_path("icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.add_paths(sys.argv[1:])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
