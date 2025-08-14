#!/usr/bin/env python3

"""
EZEnhance — Photo Colorizer & Enhancer (drag-and-drop old photo restoration)
Super-modern flat UI, rounded corners, green/gray theme, emojis included 😎

Dependencies:
  pip install PySide6 opencv-python numpy pillow

Optional AI colorization (auto-detected if files exist in script folder):
  - colorization_deploy_v2.prototxt
  - colorization_release_v2.caffemodel
  - pts_in_hull.npy

Author: Crayton Litton
License: MIT
"""

import os
import sys
import io
import math
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List
from PySide6.QtCore import QSize, QSizeF

import numpy as np
from PIL import Image

APP_NAME = "EZEnhance"

# OpenCV is used for processing. We carefully guard imports/availability.
try:
    import cv2
except Exception as e:
    cv2 = None

# -------------------------- UI (PySide6) -------------------------------------
from PySide6.QtCore import (Qt, QSize, QRect, QMimeData, QByteArray, QBuffer,
                            Signal, QObject)
from PySide6.QtGui import (QGuiApplication, QIcon, QPixmap, QDragEnterEvent,
                           QDropEvent, QAction, QCursor)
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QStyle, QFrame, QMessageBox,
    QGraphicsDropShadowEffect, QGridLayout, QSplitter, QSizePolicy, QCheckBox
)

# -------------------------- Helpers ------------------------------------------

def pil_to_qpixmap(pil: Image.Image) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    data = pil.tobytes("raw", "RGBA")
    qimg = QPixmap()
    qimg.loadFromData(data, "RGBA")
    # QPixmap.loadFromData with raw bytes isn't reliable for raw RGBA; use QImage buffer route
    from PySide6.QtGui import QImage
    qimage = QImage(data, pil.width, pil.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimage)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert cv2 BGR/GRAY to PIL RGB."""
    if img is None:
        raise ValueError("Empty image")
    if len(img.shape) == 2:
        return Image.fromarray(img, mode="L").convert("RGB")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil: Image.Image) -> np.ndarray:
    """Convert PIL RGB/RGBA/L to cv2 BGR."""
    if pil.mode in ("RGBA", "LA"):
        pil = pil.convert("RGB")
    elif pil.mode == "L":
        pil = pil.convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def is_grayscale(img_bgr: np.ndarray) -> bool:
    if img_bgr is None or img_bgr.ndim != 3:
        return True
    b, g, r = cv2.split(img_bgr)
    d1 = cv2.absdiff(b, g)
    d2 = cv2.absdiff(g, r)
    # If channels are almost equal everywhere, assume grayscale
    return (np.mean(d1) + np.mean(d2)) < 1.0

def safe_imread(path: str) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        # Fallback via PIL to handle exotic formats
        try:
            pil = Image.open(path)
            return pil_to_cv2(pil)
        except Exception:
            return None
    # Guarantee 3-channel BGR for processing
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    elif data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data

def cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    return pil_to_qpixmap(cv2_to_pil(img_bgr))

def fit_image_to_label(img: QPixmap, label: QLabel) -> QPixmap:
    """Scale pixmap to fit the QLabel, DPI-aware, PySide6-safe."""
    if img.isNull():
        return img

    # Make sure we end up with a QSize, regardless of how PySide handles the math
    dpr = getattr(label, "devicePixelRatioF", lambda: 1.0)()
    size_f = QSizeF(label.size()) * float(dpr)   # always QSizeF here
    target = size_f.toSize()                     # -> QSize

    if target.width() <= 0 or target.height() <= 0:
        return img

    return img.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def add_shadow(widget: QWidget, radius=24, alpha=0.2):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(radius)
    shadow.setXOffset(0)
    shadow.setYOffset(8)
    from PySide6.QtGui import QColor
    shadow.setColor(QColor(0, 0, 0, int(alpha * 255)))
    widget.setGraphicsEffect(shadow)

# --------------------- Image Processing Pipeline -----------------------------

@dataclass
class RestoreSettings:
    denoise_strength: int = 10        # 0-30
    scratch_remove: int = 15          # 0-30
    contrast: int = 20                # 0-100 (CLAHE strength)
    sharpen: int = 15                 # 0-50
    upscale: int = 0                  # 0=none, 1=1.5x, 2=2x

@dataclass
class ColorizeSettings:
    strength: int = 70                # 0-100 mix
    vibrance: int = 20                # 0-100
    warmth: int = 10                  # -50..+50 mapped to 0..100 slider in UI via offset
    use_ai_if_available: bool = True

class Processor:
    def __init__(self):
        self.ai_ready = False
        self.net = None
        self.pts = None
        self._try_init_ai_colorizer()

    def _try_init_ai_colorizer(self):
        """Load OpenCV colorization (Caffe) if files exist. Quietly degrade if absent."""
        if cv2 is None:
            return
        base = os.path.dirname(os.path.abspath(__file__))
        prototxt = os.path.join(base, "colorization_deploy_v2.prototxt")
        caffemodel = os.path.join(base, "colorization_release_v2.caffemodel")
        pts_in_hull = os.path.join(base, "pts_in_hull.npy")
        if not (os.path.exists(prototxt) and os.path.exists(caffemodel) and os.path.exists(pts_in_hull)):
            return
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            pts = np.load(pts_in_hull)
            # Populate cluster centers as 1x1 convolution kernel
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype(np.float32)]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
            self.net = net
            self.pts = pts
            self.ai_ready = True
        except Exception:
            self.ai_ready = False
            self.net = None
            self.pts = None

    # -------- Restoration steps --------
    def auto_restore(self, img_bgr: np.ndarray, rs: RestoreSettings) -> np.ndarray:
        img = img_bgr.copy()
        # 1) Gentle denoise
        if rs.denoise_strength > 0:
            h = int(np.interp(rs.denoise_strength, [0, 30], [0, 12]))
            if h > 0:
                img = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

        # 2) Dust/Scratch removal via median + morphological opening
        if rs.scratch_remove > 0:
            k = max(1, int(np.interp(rs.scratch_remove, [0, 30], [1, 3])))
            img = cv2.medianBlur(img, 2 * k + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # blend to preserve edges
            img = cv2.addWeighted(img, 0.7, opened, 0.3, 0)

        # 3) Contrast via CLAHE in LAB
        if rs.contrast > 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab)
            clip = np.interp(rs.contrast, [0, 100], [1.0, 4.0])
            clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
            L2 = clahe.apply(L)
            lab2 = cv2.merge([L2, a, b])
            img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # 4) Unsharp mask (edge-preserving)
        if rs.sharpen > 0:
            amt = np.interp(rs.sharpen, [0, 50], [0.0, 1.2])
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
            img = cv2.addWeighted(img, 1 + amt, blurred, -amt, 0)

        # 5) Optional upscale
        if rs.upscale == 1:
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)
        elif rs.upscale == 2:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)

        return img

    # --------- Colorization ----------
    def _ai_colorize_lab(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.ai_ready or self.net is None:
            return None
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L = img_lab[:, :, 0]
        L_rs = cv2.resize(L, (224, 224))
        L_rs = L_rs.astype(np.float32) - 50  # mean-center

        blob = cv2.dnn.blobFromImage(L_rs)
        self.net.setInput(blob)
        ab_dec = self.net.forward([self.net.getLayer(self.net.getLayerId("class8_ab")).name,
                                   self.net.getLayer(self.net.getLayerId("conv8_313_rh")).name])[0]
        ab_dec = ab_dec.squeeze().transpose((1, 2, 0))  # 224x224x2
        ab_up = cv2.resize(ab_dec, (w, h))
        lab_out = np.zeros((h, w, 3), dtype=np.float32)
        lab_out[:, :, 0] = L
        lab_out[:, :, 1:] = ab_up * 2.0  # gentle scaling
        img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    def _smart_tint_colorize(self, img_bgr: np.ndarray, vibrance=0.2, warmth=0.1) -> np.ndarray:
        """
        Fallback "colorization" that infers subtle chroma from luminance structure.
        It's not true AI colorization, but it breathes life into grayscale scans:
          - Boosts mid-tone chroma
          - Adds warm skin-friendly bias (controllable)
          - Preserves edges via guided filtering
        """
        img = img_bgr.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        Lf = L.astype(np.float32) / 255.0

        # Create pseudo a/b from gradients (edges hint at boundaries of objects)
        gx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = cv2.GaussianBlur(mag, (0, 0), 1.0)

        # Base chroma fields
        a_new = (np.tanh((Lf - 0.5) * 2.0) * 20.0)  # subtle green-magenta shift
        b_new = ((Lf - 0.5) * 40.0)                 # subtle blue-yellow shift

        # Warmth bias (skin-friendly)
        b_new += warmth * 25.0

        # Edge-aware vibrance modulation
        vib = vibrance * (1.0 - mag)
        a2 = a.astype(np.float32) + a_new * vib
        b2 = b.astype(np.float32) + b_new * vib

        lab2 = cv2.merge([L.astype(np.uint8), np.clip(a2, 0, 255).astype(np.uint8),
                          np.clip(b2, 0, 255).astype(np.uint8)])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out

    def colorize(self, img_bgr: np.ndarray, cs: ColorizeSettings) -> np.ndarray:
        if img_bgr is None:
            return img_bgr

        # If already colored, we’ll do a tasteful vibrance/temperature pass instead
        if not is_grayscale(img_bgr):
            return self.boost_color(img_bgr, cs.vibrance / 100.0, cs.warmth / 50.0)

        out_ai = None
        if cs.use_ai_if_available and self.ai_ready:
            try:
                out_ai = self._ai_colorize_lab(img_bgr)
            except Exception:
                out_ai = None

        if out_ai is None:
            # Fallback smart tint
            vib = cs.vibrance / 100.0
            warm = cs.warmth / 50.0
            out = self._smart_tint_colorize(img_bgr, vibrance=vib, warmth=warm)
        else:
            out = out_ai
            # gentle post vibrance
            out = self.boost_color(out, cs.vibrance / 100.0, cs.warmth / 50.0)

        # Mix with original to control strength
        alpha = np.clip(cs.strength / 100.0, 0, 1)
        return cv2.addWeighted(out, alpha, img_bgr, 1 - alpha, 0)

    def boost_color(self, img_bgr: np.ndarray, vibrance=0.2, warmth=0.0) -> np.ndarray:
        img = img_bgr.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        # Vibrance: increase saturation proportionally more for lower-sat pixels
        factor = 1.0 + 0.8 * vibrance
        S = S + (255 - S) * (factor - 1.0) * 0.6
        S = np.clip(S, 0, 255)
        hsv = cv2.merge([H, S, V])
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Warmth via LAB b-channel bias
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        b = np.clip(b.astype(np.float32) + warmth * 8.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
        return img

# ---------------------------- UI Widgets -------------------------------------

class DropArea(QFrame):
    fileDropped = Signal(str)

    def __init__(self, theme_css: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setObjectName("DropArea")
        self.setStyleSheet(theme_css)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)

        self.icon = QLabel("🖼️")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 64px;")

        self.text = QLabel(f"Drag & drop a photo here\nor click Open 📂\nWelcome to {APP_NAME}!")
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setStyleSheet("color: #9AA3A8; font-size: 16px;")

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setObjectName("Preview")
        self.preview.setStyleSheet("""
            QLabel#Preview {
                background: #1E1F22;
                border-radius: 16px;
                border: 1px solid #2B2D31;
                min-height: 320px;
            }
        """)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        layout.addWidget(self.preview)

        add_shadow(self.preview, radius=32, alpha=0.25)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.fileDropped.emit(path)
                break

class RoundButton(QPushButton):
    def __init__(self, text, accent=False, parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setObjectName("RoundButtonAccent" if accent else "RoundButton")
        self.setMinimumHeight(44)

class LabeledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, emoji: str, text: str, minv: int, maxv: int, init: int):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.lab = QLabel(f"{emoji} {text}")
        self.lab.setStyleSheet("color:#C9D1D5;")
        self.sld = QSlider(Qt.Horizontal)
        self.sld.setMinimum(minv)
        self.sld.setMaximum(maxv)
        self.sld.setValue(init)
        self.val = QLabel(str(init))
        self.val.setFixedWidth(36)
        self.val.setAlignment(Qt.AlignRight)
        for w in (self.lab, self.sld, self.val):
            lay.addWidget(w)
        self.sld.valueChanged.connect(self._on_change)

    def _on_change(self, v: int):
        self.val.setText(str(v))
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self.sld.value()

# ---------------------------- Main Window ------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — 🛠️🎨✨")
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.setMinimumSize(1000, 680)

        self.processor = Processor()
        self.original_bgr: Optional[np.ndarray] = None
        self.current_bgr: Optional[np.ndarray] = None
        self.history: List[np.ndarray] = []
        self.compare_mode = False

        self.restore_settings = RestoreSettings()
        self.color_settings = ColorizeSettings()

        self._apply_theme()
        self._build_ui()

    # ---------------- Theme ----------------
    def _apply_theme(self):
        # Deep gray canvas + neon-ish green accents
        self.setStyleSheet("""
            QMainWindow {
                background: #0F1113;
            }
            QFrame#RootCard {
                background: #131517;
                border: 1px solid #282C31;
                border-radius: 20px;
            }
            QFrame#ToolCard {
                background: #15181B;
                border: 1px solid #2B3036;
                border-radius: 16px;
            }
            QLabel {
                color: #C9D1D5;
                font-size: 14px;
            }
            QLabel#Title {
                color: #E6F4EA;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton#RoundButton {
                background: #1A1D21;
                color: #C9D1D5;
                border: 1px solid #2E343B;
                border-radius: 12px;
                padding: 10px 14px;
            }
            QPushButton#RoundButton:hover {
                background: #20252A;
                border-color: #3A424A;
            }
            QPushButton#RoundButton:pressed {
                background: #0F1215;
            }
            QPushButton#RoundButtonAccent {
                background: #16A34A; /* green accent */
                color: #08130B;
                border: 1px solid #18B352;
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton#RoundButtonAccent:hover {
                background: #19B955;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1E2328;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #16A34A;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
                border: 1px solid #19B955;
            }
            QSlider::sub-page:horizontal {
                background: #2ED16B;
                border-radius: 3px;
            }
            QLabel#Preview {
                background: #1E1F22;
                border-radius: 16px;
                border: 1px solid #2B2D31;
            }
            QCheckBox {
                color: #C9D1D5;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #3A424A;
                border-radius: 5px;
                background: #15181B;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #18B352;
                background: #16A34A;
                border-radius: 5px;
            }
        """)

    # ---------------- UI Build -------------
    def _build_ui(self):
        root = QFrame()
        root.setObjectName("RootCard")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        # Header row
        header = QHBoxLayout()
        title = QLabel(f"🧪 {APP_NAME} — Old Photo Restoration")
        title.setObjectName("Title")

        self.btn_open = RoundButton("📂 Open")
        self.btn_save = RoundButton("💾 Save")
        self.btn_undo = RoundButton("↩ Undo")
        self.btn_compare = RoundButton("↔ Compare")
        self.btn_restore = RoundButton("🧼 Restore", accent=True)
        self.btn_color = RoundButton("🎨 Colorize", accent=True)
        self.btn_auto = RoundButton("✨ Restore + Colorize", accent=True)

        for b in [self.btn_open, self.btn_save, self.btn_undo, self.btn_compare,
                  self.btn_restore, self.btn_color, self.btn_auto]:
            header.addWidget(b)
        header.addStretch(1)

        # Content split: left drop/preview, right controls
        split = QSplitter()
        split.setOrientation(Qt.Horizontal)

        # Left: Drop area + preview
        left_card = DropArea(theme_css="")
        left_card.setObjectName("ToolCard")
        left_wrap = QVBoxLayout(left_card)
        left_wrap.setContentsMargins(16, 16, 16, 16)
        left_wrap.setSpacing(10)

        # Right: Controls
        right = QFrame()
        right.setObjectName("ToolCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # Restore controls
        lbl_restore = QLabel("🧼 Restoration")
        lbl_restore.setObjectName("Title")
        s_denoise = LabeledSlider("🔉", "Denoise", 0, 30, self.restore_settings.denoise_strength)
        s_scratch = LabeledSlider("🩹", "Dust/Scratch", 0, 30, self.restore_settings.scratch_remove)
        s_contrast = LabeledSlider("🌗", "Contrast", 0, 100, self.restore_settings.contrast)
        s_sharpen = LabeledSlider("🔪", "Sharpen", 0, 50, self.restore_settings.sharpen)
        s_upscale = LabeledSlider("⤴️", "Upscale (0/1/2x)", 0, 2, self.restore_settings.upscale)

        # Color controls
        lbl_color = QLabel("🎨 Colorization")
        lbl_color.setObjectName("Title")
        s_strength = LabeledSlider("💪", "Strength", 0, 100, self.color_settings.strength)
        s_vibrance = LabeledSlider("🌈", "Vibrance", 0, 100, self.color_settings.vibrance)
        s_warmth = LabeledSlider("🔥", "Warmth", 0, 100, 50 + self.color_settings.warmth)  # center=50
        chk_ai = QCheckBox("Use AI model if available 🤖")
        chk_ai.setChecked(self.color_settings.use_ai_if_available)

        # Assemble right panel
        right_layout.addWidget(lbl_restore)
        right_layout.addWidget(s_denoise)
        right_layout.addWidget(s_scratch)
        right_layout.addWidget(s_contrast)
        right_layout.addWidget(s_sharpen)
        right_layout.addWidget(s_upscale)
        right_layout.addSpacing(8)
        right_layout.addWidget(lbl_color)
        right_layout.addWidget(s_strength)
        right_layout.addWidget(s_vibrance)
        right_layout.addWidget(s_warmth)
        right_layout.addWidget(chk_ai)
        right_layout.addStretch(1)

        split.addWidget(left_card)
        split.addWidget(right)
        split.setSizes([800, 300])

        root_layout.addWidget(title)
        root_layout.addLayout(header)
        root_layout.addWidget(split)

        add_shadow(left_card.preview, radius=28, alpha=0.25)
        add_shadow(left_card, radius=24, alpha=0.18)
        add_shadow(right, radius=24, alpha=0.18)

        self.setCentralWidget(root)

        # Connect signals
        left_card.fileDropped.connect(self._open_path)
        self.btn_open.clicked.connect(self._open_dialog)
        self.btn_save.clicked.connect(self._save_dialog)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_compare.clicked.connect(self._toggle_compare)
        self.btn_restore.clicked.connect(self._do_restore)
        self.btn_color.clicked.connect(self._do_colorize)
        self.btn_auto.clicked.connect(self._do_auto)

        s_denoise.valueChanged.connect(lambda v: setattr(self.restore_settings, "denoise_strength", v))
        s_scratch.valueChanged.connect(lambda v: setattr(self.restore_settings, "scratch_remove", v))
        s_contrast.valueChanged.connect(lambda v: setattr(self.restore_settings, "contrast", v))
        s_sharpen.valueChanged.connect(lambda v: setattr(self.restore_settings, "sharpen", v))
        s_upscale.valueChanged.connect(lambda v: setattr(self.restore_settings, "upscale", v))

        s_strength.valueChanged.connect(lambda v: setattr(self.color_settings, "strength", v))
        s_vibrance.valueChanged.connect(lambda v: setattr(self.color_settings, "vibrance", v))
        s_warmth.valueChanged.connect(lambda v: setattr(self.color_settings, "warmth", v - 50))
        chk_ai.stateChanged.connect(lambda s: setattr(self.color_settings, "use_ai_if_available", s == Qt.Checked))

        # Keep handles for preview label + drop text/icon
        self.drop = left_card
        self.preview = left_card.preview
        self.drop.icon.setVisible(True)
        self.drop.text.setVisible(True)

    # ----------------- Image Ops -----------------

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, f"Open Image — {APP_NAME}", "", "Images (*.png *.jpg *.jpeg *.tif *.bmp)")
        if path:
            self._open_path(path)

    def _open_path(self, path: str):
        try:
            img = safe_imread(path)
            if img is None:
                raise ValueError("Could not read image.")
            self.original_bgr = img
            self.current_bgr = img.copy()
            self.history = [img.copy()]
            self._show_image(self.current_bgr)
            self._toast("✅ Loaded image into EZEnhance. Drop more or start enhancing!")
        except Exception as e:
            self._error(f"Failed to open image:\n{e}")

    def _save_dialog(self):
        if self.current_bgr is None:
            self._toast("⚠️ No image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, f"Save Image — {APP_NAME}", "EZEnhance_output.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            try:
                ext = os.path.splitext(path)[1].lower()
                encode = ".png" if ext in (".png", "") else ".jpg"
                params = [cv2.IMWRITE_PNG_COMPRESSION, 4] if encode == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
                cv2.imencode(encode, self.current_bgr, params)[1].tofile(path)
                self._toast("💾 Saved with EZEnhance!")
            except Exception as e:
                self._error(f"Save failed:\n{e}")

    def _push_history(self):
        if self.current_bgr is not None:
            # limit history depth
            if len(self.history) > 20:
                self.history.pop(0)
            self.history.append(self.current_bgr.copy())

    def _undo(self):
        if len(self.history) >= 2:
            self.history.pop()
            self.current_bgr = self.history[-1].copy()
            self._show_image(self.current_bgr)
            self._toast("↩ Undid last step.")
        else:
            self._toast("🙃 Nothing to undo.")

    def _toggle_compare(self):
        if self.original_bgr is None or self.current_bgr is None:
            self._toast("Open an image first.")
            return
        self.compare_mode = not self.compare_mode
        if self.compare_mode:
            self._show_compare()
            self._toast("↔ Comparing (original ⟷ result) in EZEnhance.")
        else:
            self._show_image(self.current_bgr)

    def _do_restore(self):
        if self.current_bgr is None:
            self._toast("Open an image first.")
            return
        try:
            self._push_history()
            out = self.processor.auto_restore(self.current_bgr, self.restore_settings)
            self.current_bgr = out
            self._show_image(out)
            self._toast("🧼 Restored!")
        except Exception as e:
            self._error(f"Restore failed:\n{e}")

    def _do_colorize(self):
        if self.current_bgr is None:
            self._toast("Open an image first.")
            return
        try:
            self._push_history()
            out = self.processor.colorize(self.current_bgr, self.color_settings)
            self.current_bgr = out
            self._show_image(out)
            if self.processor.ai_ready and self.color_settings.use_ai_if_available and is_grayscale(self.history[-1]):
                self._toast("🎨 AI colorized (with a dash of vibrance) — EZEnhance!")
            else:
                self._toast("🎨 Smart tint applied (fallback color boost).")
        except Exception as e:
            self._error(f"Colorize failed:\n{e}")

    def _do_auto(self):
        if self.current_bgr is None:
            self._toast("Open an image first.")
            return
        try:
            self._push_history()
            step1 = self.processor.auto_restore(self.current_bgr, self.restore_settings)
            step2 = self.processor.colorize(step1, self.color_settings)
            self.current_bgr = step2
            self._show_image(step2)
            self._toast("✨ Auto: restore + colorize complete! (EZEnhance)")
        except Exception as e:
            self._error(f"Auto process failed:\n{e}")

    # ----------------- Preview Rendering -----------------

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # Refit preview image on resize
        if self.current_bgr is not None and not self.compare_mode:
            self._show_image(self.current_bgr, refit_only=True)
        elif self.compare_mode and self.original_bgr is not None:
            self._show_compare(refit_only=True)

    def _show_image(self, img_bgr: np.ndarray, refit_only=False):
        if img_bgr is None:
            return
        pix = cv2_to_qpixmap(img_bgr)
        pix = fit_image_to_label(pix, self.preview)
        self.preview.setPixmap(pix)
        if not refit_only:
            self.drop.icon.setVisible(False)
            self.drop.text.setVisible(False)

    def _show_compare(self, refit_only=False):
        """Side-by-side compare inside the same preview, with a vertical divider."""
        if self.original_bgr is None or self.current_bgr is None:
            return
        # Build combined image
        h1, w1 = self.original_bgr.shape[:2]
        h2, w2 = self.current_bgr.shape[:2]
        h = max(h1, h2)
        scale1 = self._fit_scale(w1, h1, self.preview.width() // 2, self.preview.height())
        scale2 = self._fit_scale(w2, h2, self.preview.width() // 2, self.preview.height())
        o_small = cv2.resize(self.original_bgr, (int(w1*scale1), int(h1*scale1)), interpolation=cv2.INTER_AREA)
        n_small = cv2.resize(self.current_bgr,  (int(w2*scale2), int(h2*scale2)), interpolation=cv2.INTER_AREA)

        h = max(o_small.shape[0], n_small.shape[0])
        w = o_small.shape[1] + n_small.shape[1] + 6  # divider
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 32, 36)
        canvas[0:o_small.shape[0], 0:o_small.shape[1]] = o_small
        # divider
        canvas[:, o_small.shape[1]:o_small.shape[1]+6] = (42, 184, 92)
        canvas[0:n_small.shape[0], o_small.shape[1]+6:o_small.shape[1]+6+n_small.shape[1]] = n_small

        pix = cv2_to_qpixmap(canvas)
        pix = fit_image_to_label(pix, self.preview)
        self.preview.setPixmap(pix)
        if not refit_only:
            self.drop.icon.setVisible(False)
            self.drop.text.setVisible(False)

    def _fit_scale(self, w, h, maxw, maxh):
        if w == 0 or h == 0:
            return 1.0
        return min(maxw / w, maxh / h)

    # ----------------- UX Helpers -----------------

    def _toast(self, msg: str):
        # Simple message box substitute (non-blocking in status)
        self.statusBar().showMessage(msg, 4000)

    def _error(self, msg: str):
        QMessageBox.critical(self, APP_NAME, msg)

# ---------------------------- App Entry --------------------------------------

def main():
    # High-DPI awareness
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("EZEnhance")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
