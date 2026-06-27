from __future__ import annotations

import argparse
import base64
import cv2
import json
import os
import re
import requests
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imkit as imk
import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QColor, QFontDatabase
from PIL import Image, ImageOps

from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.detection.processor import TextBlockDetector
from modules.inpainting.schema import Config
from modules.inpainting.lama import LaMa
from modules.ocr.processor import OCRProcessor
from modules.rendering.render import get_best_render_area, is_vertical_block, pyside_word_wrap
from modules.translation.processor import Translator
from modules.utils.image_utils import generate_mask, get_smart_text_color
from modules.utils.language_utils import get_language_code, get_layout_direction, is_no_space_lang
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import MODEL_MAP, format_translations
from scripts.storage import delete_prefix, download_prefix, normalize_key, upload_directory


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_STAMP_PATH = ROOT / "assets" / "brand-stamp.png"
_STAMP_WARNING_PRINTED = False


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def page_number_from_path(path: Path, fallback: int = 1) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return fallback
    try:
        return int(matches[-1])
    except ValueError:
        return fallback


def should_stamp_page(page_number: int, args: argparse.Namespace) -> bool:
    if not parse_bool(getattr(args, "stamp_enabled", True)):
        return False
    every = int(getattr(args, "stamp_every_pages", 5) or 0)
    return every > 0 and page_number > 0 and page_number % every == 0


def resolve_stamp_path(args: argparse.Namespace) -> Path:
    raw_path = Path(str(getattr(args, "stamp_path", DEFAULT_STAMP_PATH)))
    return raw_path if raw_path.is_absolute() else (ROOT / raw_path).resolve()


def save_image_for_format(image: Image.Image, out_path: Path) -> None:
    suffix = out_path.suffix.lower()
    if suffix == ".webp":
        image.save(out_path, quality=95, method=6)
    elif suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(out_path, quality=95, optimize=True)
    else:
        image.save(out_path)


def find_stamp_position(base_rgb: np.ndarray, stamp_w: int, stamp_h: int, full_width: bool = False) -> tuple[int, int] | None:
    page_h, page_w = base_rgb.shape[:2]
    pad = max(8, int(page_w * 0.018))
    box_w = stamp_w if full_width else stamp_w + pad * 2
    box_h = stamp_h + pad * 2
    if box_w > page_w or box_h >= page_h:
        return None

    y_min = max(pad, int(page_h * 0.035))
    y_max = page_h - box_h - max(pad, int(page_h * 0.035))
    x_min = 0 if full_width else pad
    x_max = 0 if full_width else page_w - box_w - pad
    if y_max <= y_min or x_max < x_min:
        return None

    y_step = max(20, min(96, box_h // 2))
    best: tuple[float, int, int] | None = None

    for y in range(y_min, y_max + 1, y_step):
        x_values = [0] if full_width else [x_min, x_max]
        for x in x_values:
            region = base_rgb[y:y + box_h, x:x + box_w]
            if region.size == 0:
                continue
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            mean = float(gray.mean())
            std = float(gray.std())
            saturation = float(hsv[:, :, 1].mean())
            dark_ratio = float((gray < 185).mean())
            edge_ratio = float((cv2.Canny(gray, 60, 160) > 0).mean())
            white_ratio = float(((gray > 238) & (hsv[:, :, 1] < 34)).mean())

            safe_white = white_ratio >= 0.92 and dark_ratio <= 0.018 and edge_ratio <= 0.018
            safe_plain = mean >= 242 and std <= 11 and saturation <= 24 and dark_ratio <= 0.012
            if not (safe_white or safe_plain):
                continue

            score = (std * 1.8) + (edge_ratio * 700.0) + (dark_ratio * 900.0) - (white_ratio * 25.0)
            if best is None or score < best[0]:
                best = (score, x if full_width else x + pad, y + pad)

    if best is None:
        return None
    return best[1], best[2]


def apply_page_stamp(out_path: Path, source_path: Path, args: argparse.Namespace, fallback_page: int = 1) -> bool:
    global _STAMP_WARNING_PRINTED
    page_number = page_number_from_path(source_path, fallback_page)
    if not should_stamp_page(page_number, args):
        return False

    stamp_path = resolve_stamp_path(args)
    if not stamp_path.exists():
        if not _STAMP_WARNING_PRINTED:
            print(f"  stamp skipped: asset not found ({stamp_path})", flush=True)
            _STAMP_WARNING_PRINTED = True
        return False

    try:
        with Image.open(out_path) as base_img, Image.open(stamp_path) as stamp_img:
            base = ImageOps.exif_transpose(base_img).convert("RGBA")
            stamp = ImageOps.exif_transpose(stamp_img).convert("RGBA")
            width_ratio = float(getattr(args, "stamp_width_ratio", 1.0))
            full_width = width_ratio >= 0.95
            target_w = base.width if full_width else int(base.width * width_ratio)
            target_w = max(90, min(target_w, base.width if full_width else int(base.width * 0.42)))
            target_h = max(1, int(target_w * stamp.height / stamp.width))
            stamp = stamp.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if parse_bool(getattr(args, "stamp_black_background", False)):
                black_backing = Image.new("RGBA", stamp.size, (0, 0, 0, 255))
                black_backing.alpha_composite(stamp)
                stamp = black_backing

            opacity = max(0.0, min(1.0, float(getattr(args, "stamp_opacity", 1.0))))
            if opacity < 1.0:
                alpha = stamp.getchannel("A").point(lambda value: int(value * opacity))
                stamp.putalpha(alpha)

            pos = find_stamp_position(np.array(base.convert("RGB")), stamp.width, stamp.height, full_width)
            if pos is None:
                print(f"  stamp skipped [{page_number}]: safe blank area not found", flush=True)
                return False

            base.alpha_composite(stamp, dest=pos)
            save_image_for_format(base, out_path)
            print(f"  stamp [{page_number}]: {out_path.name} @ {pos[0]},{pos[1]}", flush=True)
            return True
    except Exception as exc:
        print(f"  stamp skipped [{page_number}]: {exc}", flush=True)
        return False


def apply_page_stamp_to_outputs(out_paths: list[Path], source_path: Path, args: argparse.Namespace, fallback_page: int = 1) -> bool:
    page_number = page_number_from_path(source_path, fallback_page)
    if not should_stamp_page(page_number, args):
        return False
    for out_path in out_paths:
        if apply_page_stamp(out_path, source_path, args, fallback_page):
            return True
    return False


class LocalUI:
    @staticmethod
    def tr(value: str) -> str:
        return value


@dataclass
class LocalRenderSettings:
    alignment_id: int = 1
    font_family: str = "Arial"
    min_font_size: int = 10
    max_font_size: int = 40
    color: str = "#000000"
    upper_case: bool = True
    outline: bool = True
    outline_color: str = "#FFFFFF"
    outline_width: str = "2"
    bold: bool = False
    italic: bool = False
    underline: bool = False
    line_spacing: str = "1.0"
    direction: QtCore.Qt.LayoutDirection = QtCore.Qt.LayoutDirection.LeftToRight


class LocalSettings:
    def __init__(self, args: argparse.Namespace):
        self.ui = LocalUI()
        self.args = args

    def get_tool_selection(self, name: str) -> str:
        return {
            "translator": self.args.translator,
            "detector": self.args.detector,
            "ocr": self.args.ocr,
            "inpainter": self.args.inpainter,
        }[name]

    def is_gpu_enabled(self) -> bool:
        return bool(self.args.gpu)

    def get_credentials(self, service: str = "") -> dict:
        if service == "Open AI GPT":
            return {"api_key": os.environ.get("OPENAI_API_KEY", "")}
        return {"save_key": False}

    def get_llm_settings(self) -> dict:
        return {"extra_context": self.args.extra_context, "image_input_enabled": False}

    def get_hd_strategy_settings(self) -> dict:
        return {
            "strategy": self.args.hd_strategy,
            "resize_limit": self.args.resize_limit,
            "crop_margin": 512,
            "crop_trigger_size": 512,
        }


class LocalMain:
    def __init__(self, settings: LocalSettings, source_lang: str, target_lang: str):
        self.settings_page = settings
        self.lang_mapping = {}
        self.button_to_alignment = {
            0: QtCore.Qt.AlignmentFlag.AlignLeft,
            1: QtCore.Qt.AlignmentFlag.AlignCenter,
            2: QtCore.Qt.AlignmentFlag.AlignRight,
        }
        self.source_lang = source_lang
        self.target_lang = target_lang

    def render_settings(self) -> LocalRenderSettings:
        return LocalRenderSettings(
            font_family=self.settings_page.args.resolved_font,
            min_font_size=self.settings_page.args.min_font_size,
            max_font_size=self.settings_page.args.max_font_size,
            upper_case=self.settings_page.args.uppercase,
            outline=not self.settings_page.args.no_outline,
            direction=get_layout_direction(self.target_lang),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Temizleme reçetesi: YOLOv8 metin segmentasyonu + balon-içi Otsu + block-aware LaMa
# ─────────────────────────────────────────────────────────────────────────────
_SEGMENTER = None
_LAMA = None
_MODEL_LOCK = threading.Lock()


def _get_segmenter():
    """ogkalu comic-text-segmenter (YOLOv8) — pixel-level metin maskesi. Thread-safe lazy init.

    Önemli: YOLO modeli ilk inference'da kendini fuse eder (Conv+BN birleştirip
    'bn' attribute'unu siler). Birden fazla worker thread aynı anda ilk kez
    çağırırsa fuse race oluşur ("'Conv' object has no attribute 'bn'"). Bunu
    önlemek için modeli lock İÇİNDE bir kez ısıtıp fuse'u tetikliyoruz ve
    _SEGMENTER'ı ancak ondan SONRA set ediyoruz."""
    global _SEGMENTER
    if _SEGMENTER is None:
        with _MODEL_LOCK:
            if _SEGMENTER is None:
                from ultralytics import YOLO
                from modules.utils.download import ModelDownloader, ModelID
                ModelDownloader.get(ModelID.COMIC_TEXT_SEGMENTER)
                path = ModelDownloader.primary_path(ModelID.COMIC_TEXT_SEGMENTER)
                model = YOLO(str(path))
                # Warm-up: fuse'u tek thread'de tetikle (paralel kullanım öncesi)
                model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False, imgsz=640)
                _SEGMENTER = model
    return _SEGMENTER


def _get_lama():
    global _LAMA
    if _LAMA is None:
        with _MODEL_LOCK:
            if _LAMA is None:
                _LAMA = LaMa(device="cpu", backend="onnx")
    return _LAMA


def build_segmenter_mask(image: np.ndarray, blocks=None) -> np.ndarray:
    """Pixel-perfect metin maskesi: YOLOv8 segmentasyonu + balon-içi Otsu (izole
    işaretleri toplar, balon kenarına değmez) + closing/dilation.

    blocks verilirse, maske SADECE OCR metni OKUNAN blokların bölgesine
    sınırlanır. Böylece OCR'ın okuyamadığı SFX / ses efektleri / yalın
    işaretler (!, ? gibi) temizlenmez, orijinal haliyle bırakılır."""
    h, w = image.shape[:2]
    conf = float(os.environ.get("SEG_CONF", "0.25"))
    close_k = int(os.environ.get("MASK_CLOSE", "13"))
    dilate_k = int(os.environ.get("MASK_DILATE", "13"))

    segmenter = _get_segmenter()
    res = segmenter.predict(image[..., ::-1], conf=conf, verbose=False, imgsz=1024)
    seg = np.zeros((h, w), np.uint8)
    for r in res:
        if r.masks is None:
            continue
        for s in r.masks.data:
            sm = s.cpu().numpy().astype(np.uint8) * 255
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
            seg[sm > 127] = 255
    if seg.sum() == 0:
        return seg

    # Sadece OCR metni okunan blokların bölgesini temizle; okunamayanları (SFX,
    # işaretler) olduğu gibi bırak.
    if blocks is not None:
        pad = int(os.environ.get("CLEAN_TEXT_PAD", "20"))
        keep = np.zeros((h, w), np.uint8)
        has_text = False
        for blk in blocks:
            if not (getattr(blk, "text", "") or "").strip():
                continue
            has_text = True
            x1, y1, x2, y2 = [int(v) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
            keep[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 255
        if not has_text:
            return np.zeros((h, w), np.uint8)
        seg = cv2.bitwise_and(seg, keep)
        if seg.sum() == 0:
            return seg

    # Her segmenter bileşeninin etrafında, balon kenarına değmeden Otsu ile
    # izole işaretleri (ünlem kuyruğu vb.) topla.
    otsu_full = np.zeros((h, w), np.uint8)
    n, labels = cv2.connectedComponents(seg)
    for lbl in range(1, n):
        ys, xs = np.where(labels == lbl)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        ey, ex = int((y2 - y1) * 0.12), int((x2 - x1) * 0.06)
        by1, bx1 = max(0, y1 - ey), max(0, x1 - ex)
        by2, bx2 = min(h, y2 + ey), min(w, x2 + ex)
        roi = image[by1:by2, bx1:bx2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        nn, lab2, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        roi_area = roi.shape[0] * roi.shape[1]
        for i in range(1, nn):
            a = st[i, cv2.CC_STAT_AREA]
            if 8 <= a <= roi_area * 0.5:  # çok büyük (kenar sızması) hariç
                otsu_full[by1:by2, bx1:bx2][lab2 == i] = 255

    mask = cv2.bitwise_or(seg, otsu_full)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
    return mask


def block_inpaint(image: np.ndarray, mask: np.ndarray, config) -> np.ndarray:
    """Block-aware full-res LaMa: her metin bölgesini 1.7x bağlamla kırpıp ayrı
    inpaint eder (tüm sayfayı küçültmeden → keskin sonuç)."""
    h, w = image.shape[:2]
    lama = _get_lama()
    result = image.copy()
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for lbl in range(1, n):
        ys, xs = np.where(labels == lbl)
        if len(ys) < 10:
            continue
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * 0.35) + 12, int(bh * 0.35) + 12
        cy1, cx1 = max(0, y1 - my), max(0, x1 - mx)
        cy2, cx2 = min(h, y2 + my), min(w, x2 + mx)
        crop = image[cy1:cy2, cx1:cx2].copy()
        cmask = mask[cy1:cy2, cx1:cx2].copy()
        if cmask.sum() == 0:
            continue
        out = imk.convert_scale_abs(lama(crop, cmask, config))
        cm = cmask > 0
        result[cy1:cy2, cx1:cx2][cm] = out[cm]
    return result


def resolve_font(args: argparse.Namespace) -> str:
    font_file = args.font_file.strip() if args.font_file else ""
    candidates = []
    if font_file:
        candidates.append(Path(font_file))
        candidates.append(ROOT / font_file)
    candidates.extend(ROOT.glob("*.ttf"))
    candidates.extend(ROOT.glob("*.otf"))

    for candidate in candidates:
        if not candidate.exists() or candidate.suffix.lower() not in {".ttf", ".otf"}:
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate.resolve()))
        if font_id == -1:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            print(f"Font loaded: {candidate.name} -> {families[0]}", flush=True)
            return families[0]

    return args.font


def contrast_outline_color(text_color: QColor) -> QColor:
    red = text_color.red()
    green = text_color.green()
    blue = text_color.blue()
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    return QColor("#000000") if luminance > 150 else QColor("#FFFFFF")


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def clean_white_bubble_ghosts(image: np.ndarray, blocks, args: argparse.Namespace) -> np.ndarray:
    if os.environ.get("CLEAN_WHITE_BUBBLES", "true").lower() != "true":
        return image

    result = image.copy()
    height, width = result.shape[:2]
    pad = int(os.environ.get("BUBBLE_CLEAN_PAD", "36"))
    edge_guard = int(os.environ.get("BUBBLE_EDGE_GUARD", "10"))
    min_luma = int(os.environ.get("BUBBLE_CLEAN_MIN_LUMA", "185"))
    max_chroma = int(os.environ.get("BUBBLE_CLEAN_MAX_CHROMA", "42"))
    text_max_luma = int(os.environ.get("BUBBLE_TEXT_MAX_LUMA", "238"))
    text_max_chroma = int(os.environ.get("BUBBLE_TEXT_MAX_CHROMA", "110"))
    fill_region = os.environ.get("BUBBLE_CLEAN_FILL_REGION", "mask").lower()
    fill_delta = int(os.environ.get("BUBBLE_CLEAN_FILL_DELTA", "8"))
    ghost_delta = int(os.environ.get("BUBBLE_CLEAN_GHOST_DELTA", "4"))

    for block in blocks:
        if getattr(block, "text_class", None) != "text_bubble" or getattr(block, "bubble_xyxy", None) is None:
            continue

        bx1, by1, bx2, by2 = [int(v) for v in block.bubble_xyxy]
        bx1, by1, bx2, by2 = clamp_box(bx1, by1, bx2, by2, width, height)
        if bx2 <= bx1 or by2 <= by1:
            continue
        bubble_w = bx2 - bx1
        bubble_h = by2 - by1
        guard = max(edge_guard, int(min(bubble_w, bubble_h) * 0.055))

        bubble = result[by1:by2, bx1:bx2]
        if bubble.size == 0:
            continue
        luma = bubble.mean(axis=2)
        chroma = bubble.max(axis=2) - bubble.min(axis=2)
        whiteish = (luma >= min_luma) & (chroma <= max_chroma)
        if whiteish.mean() < 0.45:
            continue

        sample = bubble[whiteish]
        fill = np.median(sample, axis=0).astype(np.uint8) if sample.size else np.array([255, 255, 255], dtype=np.uint8)
        tx1, ty1, tx2, ty2 = [int(v) for v in block.xyxy]
        tx1, ty1, tx2, ty2 = clamp_box(tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad, width, height)
        tx1, ty1, tx2, ty2 = max(tx1, bx1 + guard), max(ty1, by1 + guard), min(tx2, bx2 - guard), min(ty2, by2 - guard)
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        target = result[ty1:ty2, tx1:tx2]
        if fill_region == "bbox":
            target[:, :] = fill
            result[ty1:ty2, tx1:tx2] = target
            continue
        target_luma = target.mean(axis=2)
        target_chroma = target.max(axis=2) - target.min(axis=2)
        delta = np.abs(target.astype(np.int16) - fill.astype(np.int16)).max(axis=2)
        interior_mask = ((target_luma >= min_luma - 35) & (target_chroma <= max_chroma + 45)).astype(np.uint8) * 255
        interior_kernel = np.ones((11, 11), np.uint8)
        interior_mask = imk.morphology_ex(interior_mask, imk.MORPH_CLOSE, interior_kernel)
        interior_mask = imk.dilate(interior_mask, interior_kernel, iterations=1) > 0
        ink_mask = (
            ((target_luma <= text_max_luma) & ((target_chroma <= text_max_chroma) | (target_luma <= 165)))
            | ((delta >= fill_delta) & (target_luma <= 245) & (target_chroma <= max(text_max_chroma, 70)))
            | ((delta >= ghost_delta) & (target_luma >= 205) & (target_luma <= 252) & (target_chroma <= 70))
        ).astype(np.uint8) * 255
        if ink_mask.mean() <= 0:
            continue
        kernel = np.ones((9, 9), np.uint8)
        ink_mask = imk.morphology_ex(ink_mask, imk.MORPH_CLOSE, kernel)
        ink_mask = (imk.dilate(ink_mask, kernel, iterations=1) > 0) & interior_mask
        target[ink_mask] = fill
        result[ty1:ty2, tx1:tx2] = target

    return result


def recover_empty_bubble_text_with_gpt(image: np.ndarray, blocks, args: argparse.Namespace) -> None:
    if os.environ.get("GPT_OCR_EMPTY_BUBBLES", "false").lower() != "true":
        return
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return
    candidates = [
        block
        for block in blocks
        if getattr(block, "text_class", None) == "text_bubble" and not (getattr(block, "text", "") or "").strip()
    ]
    if not candidates:
        return
    max_items = int(os.environ.get("GPT_OCR_EMPTY_MAX", "10"))
    candidates = candidates[:max_items]
    pil_image = Image.fromarray(image[..., ::-1]).convert("RGB")
    width, height = pil_image.size
    pad = int(os.environ.get("GPT_OCR_EMPTY_PAD", "28"))
    content = [
        {
            "type": "text",
            "text": (
                "Read the text inside each cropped manga speech bubble. "
                "Return only compact JSON where keys are crop_0, crop_1, etc. "
                "If a crop is genuinely blank, use an empty string. Preserve English source text."
            ),
        }
    ]
    for index, block in enumerate(candidates):
        x1, y1, x2, y2 = [int(v) for v in block.xyxy]
        x1, y1, x2, y2 = clamp_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = pil_image.crop((x1, y1, x2, y2))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "text", "text": f"crop_{index}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    model = MODEL_MAP.get(args.translator, os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": "You are a precise OCR engine."}]},
                        {"role": "user", "content": content},
                    ],
                    "max_completion_tokens": 400,
                }
            ),
            timeout=90,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return
        recovered = json.loads(match.group(0))
    except Exception as exc:
        print(f"    empty-bubble GPT OCR skipped: {exc}", flush=True)
        return

    for index, block in enumerate(candidates):
        value = str(recovered.get(f"crop_{index}", "") or "").strip()
        if value:
            block.text = value


def restore_untranslated_regions(original: np.ndarray, cleaned: np.ndarray, blocks) -> np.ndarray:
    if os.environ.get("RESTORE_UNTRANSLATED_REGIONS", "true").lower() != "true":
        return cleaned

    failed_blocks = []
    for block in blocks:
        translation = (getattr(block, "translation", "") or "").strip()
        if not translation or len(translation) <= 1:
            text = (getattr(block, "text", "") or "").strip()
            compact = re.sub(r"\s+", "", text)
            # Preserve untranslated non-Latin/SFX-like regions, but do not bring back
            # isolated OCR garbage such as stray T/L/Y marks on plain white panels.
            if _has_cjk_or_hangul(text):
                failed_blocks.append(block)
                continue
            if compact and _latin_signal_ratio(compact) < 0.25 and len(compact) >= 2:
                failed_blocks.append(block)
    if not failed_blocks:
        return cleaned
    failed_mask = generate_mask(original, failed_blocks)
    if failed_mask.max() == 0:
        return cleaned
    result = cleaned.copy()
    result[failed_mask > 0] = original[failed_mask > 0]
    return result


def _box_area(box) -> int:
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_intersection(a, b) -> int:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def _overlap_on_smaller(a, b) -> float:
    denom = min(_box_area(a), _box_area(b))
    if denom <= 0:
        return 0.0
    return _box_intersection(a, b) / denom


def _norm_block_text(value: str) -> str:
    return re.sub(r"\W+", "", (value or "").lower())


def _has_cjk_or_hangul(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]", text or ""))


def _latin_signal_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    latin = re.findall(r"[A-Za-z]", compact)
    return len(latin) / max(1, len(compact))


def drop_non_english_ocr_blocks(blocks):
    """Ignore raw CJK/Hangul SFX so it is neither cleaned nor translated."""
    if os.environ.get("DROP_NON_ENGLISH_OCR_BLOCKS", "true").lower() != "true":
        return blocks
    kept = []
    dropped = 0
    min_latin_ratio = float(os.environ.get("NON_ENGLISH_MIN_LATIN_RATIO", "0.35"))
    for block in blocks:
        text = (getattr(block, "text", "") or "").strip()
        if text and _has_cjk_or_hangul(text) and _latin_signal_ratio(text) < min_latin_ratio:
            dropped += 1
            continue
        kept.append(block)
    if dropped:
        print(f"  drop non-English OCR blocks: -{dropped}", flush=True)
    return kept


def _is_background_dense_text(block) -> bool:
    if os.environ.get("DROP_DENSE_BACKGROUND_TEXT", "true").lower() != "true":
        return False
    if getattr(block, "text_class", None) != "text_free":
        return False
    text = (getattr(block, "text", "") or "").strip()
    compact = re.sub(r"\s+", "", text)
    if len(compact) < int(os.environ.get("DENSE_TEXT_MIN_CHARS", "120")):
        return False
    x1, y1, x2, y2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
    area = max(1, (x2 - x1) * (y2 - y1))
    density = len(compact) / area
    return density >= float(os.environ.get("DENSE_TEXT_MIN_DENSITY", "0.0022"))


def prune_ocr_blocks(blocks, image: np.ndarray | None = None):
    """Drop OCR-empty detector noise and collapse duplicate boxes before clean/render."""
    if os.environ.get("PRUNE_OCR_BLOCKS", "true").lower() != "true":
        return blocks

    drop_empty = os.environ.get("DROP_EMPTY_OCR_BLOCKS", "true").lower() == "true"
    duplicate_overlap = float(os.environ.get("OCR_DUPLICATE_OVERLAP", "0.92"))
    kept = []
    dropped_empty = 0
    dropped_duplicate = 0

    for block in blocks:
        text = (getattr(block, "text", "") or "").strip()
        if drop_empty and not text:
            dropped_empty += 1
            continue
        if _is_background_dense_text(block):
            dropped_empty += 1
            continue

        box = getattr(block, "xyxy", None)
        if box is None or _box_area(box) <= 0:
            dropped_empty += 1
            continue

        replacement_index = None
        current_norm = _norm_block_text(text)
        current_bubble = getattr(block, "bubble_xyxy", None)
        for idx, other in enumerate(kept):
            other_box = getattr(other, "xyxy", None)
            if other_box is None:
                continue
            overlap = _overlap_on_smaller(box, other_box)
            bubble_overlap = 0.0
            other_bubble = getattr(other, "bubble_xyxy", None)
            if current_bubble is not None and other_bubble is not None:
                bubble_overlap = _overlap_on_smaller(current_bubble, other_bubble)

            other_text = (getattr(other, "text", "") or "").strip()
            same_text = current_norm and current_norm == _norm_block_text(other_text)
            same_region = overlap >= duplicate_overlap or bubble_overlap >= duplicate_overlap
            if same_region and (same_text or not other_text or not text):
                replacement_index = idx
                break

        if replacement_index is None:
            kept.append(block)
            continue

        other = kept[replacement_index]
        other_text = (getattr(other, "text", "") or "").strip()
        keep_current = False
        if text and not other_text:
            keep_current = True
        elif bool(text) == bool(other_text):
            keep_current = _box_area(box) > _box_area(getattr(other, "xyxy", box))
        if keep_current:
            kept[replacement_index] = block
        dropped_duplicate += 1

    if dropped_empty or dropped_duplicate:
        print(
            f"  prune OCR blocks: -{dropped_empty} empty, -{dropped_duplicate} duplicate",
            flush=True,
        )
    return kept


def looks_like_noisy_ocr(text: str, source_lang: str) -> bool:
    if os.environ.get("SKIP_NOISY_OCR", "true").lower() != "true":
        return False
    text = (text or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 2:
        return False
    letters = [char for char in compact if char.isalpha()]
    bad_symbols = [char for char in compact if not char.isalnum() and char not in ".,!?…'\"-~"]
    if len(bad_symbols) / max(1, len(compact)) >= 0.18:
        return True
    if re.search(r"[=$\\{}<>_|]{1,}", compact):
        return True
    if re.search(r"\bYLAB\b", compact, re.I):
        return True
    if re.fullmatch(r"[A-Z0-9\s.\-:;!?]{2,14}", text) and any(char.isdigit() for char in text):
        return True
    if (source_lang or "").lower() in {"english", "en"}:
        latin_letters = re.findall(r"[A-Za-z]", compact)
        vowels = re.findall(r"[AEIOUYaeiouy]", compact)
        if len(latin_letters) >= 5 and len(vowels) == 0:
            return True
        non_latin_letters = [char for char in letters if not re.match(r"[A-Za-zÀ-ÿ]", char)]
        if len(non_latin_letters) >= 2 and len(latin_letters) < 3:
            return True
    return False


def suppress_noisy_ocr_blocks(blocks, source_lang: str) -> None:
    for block in blocks:
        if looks_like_noisy_ocr(getattr(block, "text", "") or "", source_lang) or looks_like_noisy_ocr(
            getattr(block, "translation", "") or "",
            source_lang,
        ):
            block.translation = ""


def strip_translation_garbage_letters(blocks) -> None:
    """Remove isolated OCR garbage that otherwise renders as stray T/L/Y marks."""
    if os.environ.get("STRIP_TRANSLATION_GARBAGE_LETTERS", "true").lower() != "true":
        return

    garbage = os.environ.get("TRANSLATION_GARBAGE_LETTERS", "T L Y H I E LL").split()
    if not garbage:
        return
    garbage_set = {token.upper() for token in garbage}
    pattern = re.compile(
        r"(?:(?<=^)|(?<=[\s\n.,;:!?]))("
        + "|".join(re.escape(token) for token in sorted(garbage, key=len, reverse=True))
        + r")(?=(?:[\s\n.,;:!?]|$))",
        re.IGNORECASE,
    )
    for block in blocks:
        for attr in ("text", "translation"):
            text = (getattr(block, attr, "") or "").strip()
            if not text:
                continue
            compact = re.sub(r"[^A-Za-z]+", "", text).upper()
            if compact in garbage_set:
                setattr(block, attr, "")
                continue
            cleaned = pattern.sub("", text)
            cleaned = re.sub(r"(?m)^\s*\b[HTLIYE]\b[\s,;:.!?-]+(?=\S)", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?m)(?<=\S)[\s,;:!?-]+\b[HTLIYE]\b\s*$", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?<=\S)\s+\b[HTLIYE]\s*\.$", ".", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip(" \t\n-")
            setattr(block, attr, cleaned)


def debug_dump_blocks(stage: str, blocks) -> None:
    if os.environ.get("DEBUG_TRANSLATION_BLOCKS", "false").lower() != "true":
        return
    print(f"\n--- DEBUG BLOCKS {stage} ---", flush=True)
    for idx, block in enumerate(blocks):
        box = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        cls = getattr(block, "text_class", "")
        bubble = getattr(block, "bubble_xyxy", None) is not None
        text = (getattr(block, "text", "") or "").replace("\n", "\\n")
        trans = (getattr(block, "translation", "") or "").replace("\n", "\\n")
        print(f"{idx:02d} cls={cls} bubble={bubble} box={box} text={text!r} trans={trans!r}", flush=True)


def bubble_outline_protection_mask(image: np.ndarray, blocks) -> np.ndarray:
    if os.environ.get("PROTECT_BUBBLE_OUTLINES", "true").lower() != "true":
        return np.zeros(image.shape[:2], dtype=bool)

    height, width = image.shape[:2]
    protect = np.zeros((height, width), dtype=np.uint8)
    edge_margin = int(os.environ.get("BUBBLE_OUTLINE_EDGE_MARGIN", "34"))
    max_luma = int(os.environ.get("BUBBLE_OUTLINE_MAX_LUMA", "135"))
    max_chroma = int(os.environ.get("BUBBLE_OUTLINE_MAX_CHROMA", "120"))
    min_area = int(os.environ.get("BUBBLE_OUTLINE_MIN_AREA", "8"))
    text_guard = int(os.environ.get("BUBBLE_OUTLINE_TEXT_GUARD", "10"))
    text_overlap_limit = float(os.environ.get("BUBBLE_OUTLINE_TEXT_OVERLAP_LIMIT", "0.35"))
    protect_long_curves = os.environ.get("BUBBLE_OUTLINE_PROTECT_LONG_CURVES", "false").lower() == "true"
    edge_min_extent = int(os.environ.get("BUBBLE_OUTLINE_EDGE_MIN_EXTENT", "28"))
    edge_min_area = int(os.environ.get("BUBBLE_OUTLINE_EDGE_MIN_AREA", "22"))
    max_fill_ratio = float(os.environ.get("BUBBLE_OUTLINE_MAX_FILL_RATIO", "0.42"))

    for block in blocks:
        if getattr(block, "text_class", None) != "text_bubble" or getattr(block, "bubble_xyxy", None) is None:
            continue

        bx1, by1, bx2, by2 = [int(v) for v in block.bubble_xyxy]
        bx1, by1, bx2, by2 = clamp_box(bx1, by1, bx2, by2, width, height)
        if bx2 <= bx1 or by2 <= by1:
            continue
        tx1, ty1, tx2, ty2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        tx1, ty1, tx2, ty2 = clamp_box(tx1 - text_guard, ty1 - text_guard, tx2 + text_guard, ty2 + text_guard, width, height)
        text_box_local = (
            max(0, tx1 - bx1),
            max(0, ty1 - by1),
            max(0, tx2 - bx1),
            max(0, ty2 - by1),
        )

        crop = image[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        luma = crop.mean(axis=2)
        chroma = crop.max(axis=2) - crop.min(axis=2)
        dark = ((luma <= max_luma) & (chroma <= max_chroma)).astype(np.uint8)
        if dark.max() == 0:
            continue

        labels_count, labels, stats, _centroids = imk.connected_components_with_stats(dark, connectivity=8)
        crop_h, crop_w = dark.shape
        component_mask = np.zeros_like(dark, dtype=np.uint8)
        for label in range(1, labels_count):
            x, y, comp_w, comp_h, area = [int(v) for v in stats[label]]
            if area < min_area:
                continue
            touches_edge = (
                x <= edge_margin
                or y <= edge_margin
                or x + comp_w >= crop_w - edge_margin
                or y + comp_h >= crop_h - edge_margin
            )
            long_curve = protect_long_curves and area >= 80 and (comp_h >= crop_h * 0.18 or comp_w >= crop_w * 0.18)
            lx1, ly1, lx2, ly2 = text_box_local
            component_pixels = labels == label
            overlap_area = 0
            if lx2 > lx1 and ly2 > ly1:
                overlap_area = int(component_pixels[ly1:ly2, lx1:lx2].sum())
            text_overlap = overlap_area / max(1, area)
            if text_overlap >= text_overlap_limit:
                continue
            fill_ratio = area / max(1, comp_w * comp_h)
            line_like_edge = touches_edge and area >= edge_min_area and max(comp_w, comp_h) >= edge_min_extent and fill_ratio <= max_fill_ratio
            if line_like_edge or long_curve:
                component_mask[labels == label] = 255

        if component_mask.max() == 0:
            continue
        kernel = np.ones((5, 5), np.uint8)
        component_mask = imk.dilate(component_mask, kernel, iterations=1)
        protect[by1:by2, bx1:bx2] = np.maximum(protect[by1:by2, bx1:bx2], component_mask)

    return protect > 0


def restore_protected_pixels(original: np.ndarray, edited: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
    if protect_mask is None or not protect_mask.any():
        return edited
    result = edited.copy()
    result[protect_mask] = original[protect_mask]
    return result


def remove_text_regions_from_protection(protect_mask: np.ndarray, blocks, image_shape) -> np.ndarray:
    """Do not restore old glyph fragments inside expanded text boxes."""
    if protect_mask is None or not protect_mask.any():
        return protect_mask
    result = protect_mask.copy()
    height, width = image_shape[:2]
    pad_x = int(os.environ.get("PROTECT_TEXT_CLEAR_PAD_X", "48"))
    pad_y = int(os.environ.get("PROTECT_TEXT_CLEAR_PAD_Y", "24"))
    for block in blocks:
        bubble = getattr(block, "bubble_xyxy", None)
        if bubble is not None:
            # Bubble outlines are often close to large text boxes. Clearing protection
            # around those boxes makes the cleanup paint white over the border arcs.
            continue
        x1, y1, x2, y2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = False
    return result


def crop_text_backing_mask(crop: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    seed = (seed_mask > 0).astype(np.uint8) * 255
    if seed.max() == 0:
        return seed

    kernel_7 = np.ones((7, 7), np.uint8)
    kernel_13 = np.ones((13, 13), np.uint8)
    kernel_25 = np.ones((25, 25), np.uint8)
    near_seed = imk.dilate(seed, kernel_25, iterations=1)
    dark = ((gray < int(os.environ.get("CROP_CLEAN_DARK_MAX", "125"))) & (near_seed > 0)).astype(np.uint8) * 255
    white = (
        (gray > int(os.environ.get("CROP_CLEAN_WHITE_MIN", "178")))
        & (hsv[:, :, 1] < int(os.environ.get("CROP_CLEAN_WHITE_SAT_MAX", "105")))
        & (near_seed > 0)
    ).astype(np.uint8) * 255
    backing = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel_7)
    combined = np.maximum(seed, np.maximum(dark, backing))
    combined = cv2.dilate(combined, kernel_7, iterations=int(os.environ.get("CROP_CLEAN_DILATE", "1")))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_13)
    return combined


def looks_like_plain_text_panel(crop: np.ndarray, seed_mask: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    white = (gray > 188) & (hsv[:, :, 1] < 75)
    if white.mean() < float(os.environ.get("PANEL_SKIP_WHITE_RATIO", "0.42")):
        return False

    ys, xs = np.where(seed_mask > 0)
    if len(xs) == 0:
        return False
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = int(os.environ.get("PANEL_SKIP_SAMPLE_PAD", "28"))
    x1, y1, x2, y2 = clamp_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, crop.shape[1], crop.shape[0])
    sample = gray[y1:y2, x1:x2]
    if sample.size == 0:
        return False
    texture = float(sample.std())
    return texture < float(os.environ.get("PANEL_SKIP_MAX_STD", "34"))


def feather_composite(base: np.ndarray, painted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return base
    blur = int(os.environ.get("CROP_CLEAN_FEATHER", "9"))
    if blur % 2 == 0:
        blur += 1
    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    alpha = np.clip(alpha[:, :, None], 0.0, 1.0)
    return np.clip(painted.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def fill_easy_light_text_regions(image: np.ndarray, blocks, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill text on white/light regions directly instead of inpainting gray halos."""
    if os.environ.get("FILL_LIGHT_TEXT_REGIONS", "true").lower() != "true":
        return image, mask

    result = image.copy()
    remaining = mask.copy()
    height, width = image.shape[:2]
    pad = int(os.environ.get("LIGHT_FILL_PAD", "10"))
    white_ratio_min = float(os.environ.get("LIGHT_FILL_WHITE_RATIO", "0.45"))
    full_rect_white_ratio = float(os.environ.get("LIGHT_FILL_FULL_RECT_WHITE_RATIO", "0.62"))
    free_text_extra_pad = int(os.environ.get("LIGHT_FILL_FREE_TEXT_EXTRA_PAD", "28"))
    gray_min = int(os.environ.get("LIGHT_FILL_GRAY_MIN", "180"))
    sat_max = int(os.environ.get("LIGHT_FILL_SAT_MAX", "90"))

    for block in blocks:
        x1, y1, x2, y2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        block_pad = pad
        if getattr(block, "bubble_xyxy", None) is None:
            block_pad += free_text_extra_pad
        x1, y1, x2, y2 = clamp_box(x1 - block_pad, y1 - block_pad, x2 + block_pad, y2 + block_pad, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        light_pixels = (gray >= gray_min) & (hsv[:, :, 1] <= sat_max)
        if float(light_pixels.mean()) < white_ratio_min:
            continue

        fill = np.array([255, 255, 255], dtype=np.uint8)
        if light_pixels.any():
            fill = np.median(crop[light_pixels], axis=0).astype(np.uint8)
        if int(fill.mean()) < 210:
            fill = np.array([255, 255, 255], dtype=np.uint8)

        local_mask = remaining[y1:y2, x1:x2]
        use_full_rect = (
            getattr(block, "bubble_xyxy", None) is None
            and float(light_pixels.mean()) >= full_rect_white_ratio
        )
        if use_full_rect:
            local_mask = np.full(local_mask.shape, 255, dtype=np.uint8)
        elif local_mask.max() > 0:
            kernel_size = int(os.environ.get("LIGHT_FILL_MASK_DILATE", "7"))
            if kernel_size > 1:
                if kernel_size % 2 == 0:
                    kernel_size += 1
                local_mask = imk.dilate(local_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

        if local_mask.max() == 0:
            continue
        painted = result[y1:y2, x1:x2].copy()
        painted[local_mask > 0] = fill
        result[y1:y2, x1:x2] = feather_composite(result[y1:y2, x1:x2], painted, local_mask)
        remaining[y1:y2, x1:x2][local_mask > 0] = 0

    return result, remaining


def clear_light_free_text_render_regions(image: np.ndarray, blocks) -> np.ndarray:
    """Final CV2-free white wipe for narration text on plain light panels."""
    if os.environ.get("CLEAR_LIGHT_RENDER_REGIONS", "true").lower() != "true":
        return image

    result = image.copy()
    height, width = image.shape[:2]
    pad_x = int(os.environ.get("LIGHT_RENDER_CLEAR_PAD_X", "220"))
    pad_y = int(os.environ.get("LIGHT_RENDER_CLEAR_PAD_Y", "58"))
    white_ratio_min = float(os.environ.get("LIGHT_RENDER_CLEAR_WHITE_RATIO", "0.52"))
    band_white_ratio_min = float(os.environ.get("LIGHT_RENDER_CLEAR_BAND_WHITE_RATIO", "0.70"))
    gray_min = int(os.environ.get("LIGHT_RENDER_CLEAR_GRAY_MIN", "180"))
    sat_max = int(os.environ.get("LIGHT_RENDER_CLEAR_SAT_MAX", "100"))

    for block in blocks:
        if getattr(block, "bubble_xyxy", None) is not None:
            continue
        if not (getattr(block, "translation", "") or "").strip():
            continue
        x1, y1, x2, y2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = result[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        light_pixels = (gray >= gray_min) & (hsv[:, :, 1] <= sat_max)
        fill_target = (x1, y1, x2, y2)
        band = result[y1:y2, 0:width]
        band_light = None
        if band.size:
            band_gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
            band_hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
            band_light = (band_gray >= gray_min) & (band_hsv[:, :, 1] <= sat_max)
            if float(band_light.mean()) >= band_white_ratio_min:
                fill_target = (0, y1, width, y2)
                light_pixels = band_light
                crop = band

        if fill_target == (x1, y1, x2, y2) and float(light_pixels.mean()) < white_ratio_min:
            if band_light is None or float(band_light.mean()) < band_white_ratio_min:
                continue
            fill_target = (0, y1, width, y2)
            light_pixels = band_light
            crop = band
        fill = np.median(crop[light_pixels], axis=0).astype(np.uint8) if light_pixels.any() else np.array([255, 255, 255], dtype=np.uint8)
        if int(fill.mean()) < 210:
            fill = np.array([255, 255, 255], dtype=np.uint8)
        fx1, fy1, fx2, fy2 = fill_target
        result[fy1:fy2, fx1:fx2] = fill

    return result


def clear_light_bubble_text_render_regions(image: np.ndarray, blocks, original: np.ndarray | None = None) -> np.ndarray:
    """Wipe light speech-bubble interiors around rendered text without touching bubble borders."""
    if os.environ.get("CLEAR_LIGHT_BUBBLE_RENDER_REGIONS", "false").lower() != "true":
        return image

    source = original if original is not None else image
    result = image.copy()
    height, width = image.shape[:2]
    pad_x = int(os.environ.get("LIGHT_BUBBLE_RENDER_CLEAR_PAD_X", "42"))
    pad_y = int(os.environ.get("LIGHT_BUBBLE_RENDER_CLEAR_PAD_Y", "24"))
    white_ratio_min = float(os.environ.get("LIGHT_BUBBLE_RENDER_CLEAR_WHITE_RATIO", "0.55"))
    gray_min = int(os.environ.get("LIGHT_BUBBLE_RENDER_CLEAR_GRAY_MIN", "176"))
    sat_max = int(os.environ.get("LIGHT_BUBBLE_RENDER_CLEAR_SAT_MAX", "110"))
    edge_guard = int(os.environ.get("LIGHT_BUBBLE_RENDER_EDGE_GUARD", "18"))
    dark_max = int(os.environ.get("LIGHT_BUBBLE_RENDER_DARK_MAX", "150"))
    stroke_gray_min = int(os.environ.get("LIGHT_BUBBLE_RENDER_STROKE_GRAY_MIN", "205"))
    stroke_sat_max = int(os.environ.get("LIGHT_BUBBLE_RENDER_STROKE_SAT_MAX", "85"))
    mask_dilate = int(os.environ.get("LIGHT_BUBBLE_RENDER_MASK_DILATE", "5"))
    full_rect_white_ratio = float(os.environ.get("LIGHT_BUBBLE_RENDER_FULL_RECT_WHITE_RATIO", "1.01"))
    outline_protect = bubble_outline_protection_mask(source, blocks)

    for block in blocks:
        if getattr(block, "bubble_xyxy", None) is None:
            continue
        if not (getattr(block, "translation", "") or "").strip():
            continue

        bx1, by1, bx2, by2 = [int(v) for v in block.bubble_xyxy]
        bx1, by1, bx2, by2 = clamp_box(bx1 + edge_guard, by1 + edge_guard, bx2 - edge_guard, by2 - edge_guard, width, height)
        if bx2 <= bx1 or by2 <= by1:
            continue

        x1, y1, x2, y2 = [int(v) for v in getattr(block, "xyxy", (0, 0, 0, 0))]
        x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
        if os.environ.get("LIGHT_BUBBLE_RENDER_CLAMP_TO_BUBBLE", "false").lower() == "true":
            x1, y1 = max(x1, bx1), max(y1, by1)
            x2, y2 = min(x2, bx2), min(y2, by2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = result[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        light_pixels = (gray >= gray_min) & (hsv[:, :, 1] <= sat_max)
        if float(light_pixels.mean()) < white_ratio_min:
            continue
        fill = np.median(crop[light_pixels], axis=0).astype(np.uint8) if light_pixels.any() else np.array([255, 255, 255], dtype=np.uint8)
        if int(fill.mean()) < 210:
            fill = np.array([255, 255, 255], dtype=np.uint8)

        dark = gray <= dark_max
        nearby = imk.dilate(dark.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=1) > 0
        light_stroke = (gray >= stroke_gray_min) & (hsv[:, :, 1] <= stroke_sat_max) & nearby
        if float(light_pixels.mean()) >= full_rect_white_ratio:
            erase_mask = np.full(gray.shape, 255, dtype=np.uint8)
        else:
            erase_mask = (dark | light_stroke).astype(np.uint8) * 255
        if mask_dilate > 1 and float(light_pixels.mean()) < full_rect_white_ratio:
            if mask_dilate % 2 == 0:
                mask_dilate += 1
            erase_mask = imk.dilate(erase_mask, np.ones((mask_dilate, mask_dilate), np.uint8), iterations=1)
            erase_mask = imk.morphology_ex(erase_mask, imk.MORPH_CLOSE, np.ones((mask_dilate, mask_dilate), np.uint8))

        # Old glyph fragments are also dark outlines. Clear protection only for
        # compact glyph-like components; keep long/touching bubble border curves.
        local_protect = outline_protect[y1:y2, x1:x2].copy()
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            (erase_mask > 0).astype(np.uint8),
            8,
        )
        unprotect = np.zeros_like(local_protect, dtype=bool)
        crop_h, crop_w = erase_mask.shape[:2]
        for comp_idx in range(1, component_count):
            cx, cy, cw, ch, area = component_stats[comp_idx]
            if area < 4:
                continue
            touches_crop_edge = cx <= 1 or cy <= 1 or (cx + cw) >= (crop_w - 1) or (cy + ch) >= (crop_h - 1)
            too_curve_like = (
                cw > crop_w * 0.55
                or ch > crop_h * 0.65
                or (touches_crop_edge and (cw > 35 or ch > crop_h * 0.45))
            )
            if too_curve_like:
                continue
            unprotect[component_labels == comp_idx] = True
        if unprotect.any():
            outline_protect[y1:y2, x1:x2][unprotect] = False
            local_protect[unprotect] = False
        erase_mask[local_protect] = 0
        if erase_mask.max() == 0:
            continue
        painted = result[y1:y2, x1:x2].copy()
        painted[erase_mask > 0] = fill
        result[y1:y2, x1:x2] = feather_composite(result[y1:y2, x1:x2], painted, erase_mask)

    return restore_protected_pixels(source, result, outline_protect)


def process_image(path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    settings = LocalSettings(args)
    main = LocalMain(settings, args.source_lang, args.target_lang)
    device = "cuda" if args.gpu else "cpu"

    image = imk.read_image(str(path))
    print(f"[1/6] detect: {path.name}", flush=True)
    blocks = TextBlockDetector(settings).detect(image)
    if not blocks:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ext = path.suffix if args.output_format == "source" else f".{args.output_format.lstrip('.')}"
        out_path = output_dir / f"{path.stem}{args.suffix}{output_ext}"
        print(f"[2/6] no text blocks, passthrough save: {out_path}", flush=True)
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            if output_ext.lower() in {".jpg", ".jpeg", ".webp"} and source.mode not in {"RGB", "RGBA"}:
                source = source.convert("RGBA" if "A" in source.getbands() else "RGB")
            source.save(out_path)
        apply_page_stamp(out_path, path, args)
        return out_path

    print(f"[2/6] ocr: {len(blocks)} blocks", flush=True)
    ocr = OCRProcessor()
    ocr.initialize(main, args.source_lang)
    ocr.process(image, blocks)
    recover_empty_bubble_text_with_gpt(image, blocks, args)
    blocks = drop_non_english_ocr_blocks(blocks)
    blocks = prune_ocr_blocks(blocks, image)
    strip_translation_garbage_letters(blocks)
    blocks = sort_blk_list(blocks, right_to_left=args.source_lang == "Japanese")

    print("[3/6] clean: segmenter + block-aware LaMa", flush=True)
    config = Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
    mask = build_segmenter_mask(image, blocks)
    if mask.sum() == 0:
        cleaned = image.copy()
    else:
        cleaned = block_inpaint(image, mask, config)

    print(f"[4/6] translate: {args.translator}", flush=True)
    os.environ["TRANSLATION_SOURCE_FILE"] = path.name
    translator = Translator(main, args.source_lang, args.target_lang)
    translator.translate(blocks, image, args.extra_context)
    suppress_noisy_ocr_blocks(blocks, args.source_lang)
    strip_translation_garbage_letters(blocks)
    debug_dump_blocks("after-translate", blocks)
    cleaned = restore_untranslated_regions(image, cleaned, blocks)

    print("[5/6] render", flush=True)
    target_code = get_language_code(args.target_lang)
    render_settings = main.render_settings()
    format_translations(blocks, target_code, upper_case=render_settings.upper_case)
    get_best_render_area(blocks, image, cleaned)
    cleaned = clear_light_free_text_render_regions(cleaned, blocks)
    cleaned = clear_light_bubble_text_render_regions(cleaned, blocks, image)

    text_items_state = []
    for block in blocks:
        translation = block.translation
        if not translation or len(translation) == 1:
            continue

        x1, y1, block_width, block_height = block.xywh
        vertical = is_vertical_block(block, target_code)
        alignment = main.button_to_alignment[render_settings.alignment_id]
        outline_width = float(render_settings.outline_width)

        wrapped, font_size, rendered_width, rendered_height = pyside_word_wrap(
            translation,
            render_settings.font_family,
            int(block_width),
            int(block_height),
            float(render_settings.line_spacing),
            outline_width,
            render_settings.bold,
            render_settings.italic,
            render_settings.underline,
            alignment,
            render_settings.direction,
            render_settings.max_font_size,
            render_settings.min_font_size,
            vertical,
            return_metrics=True,
        )
        if is_no_space_lang(target_code):
            wrapped = wrapped.replace(" ", "")

        font_color = get_smart_text_color(block.font_color, QColor(render_settings.color))
        outline_color = contrast_outline_color(font_color) if render_settings.outline else None
        pos_x = int(x1 + max(0, (block_width - rendered_width) / 2))
        pos_y = int(y1 + max(0, (block_height - rendered_height) / 2))
        props = TextItemProperties(
            text=wrapped,
            font_family=render_settings.font_family,
            font_size=font_size,
            text_color=font_color,
            alignment=alignment,
            line_spacing=float(render_settings.line_spacing),
            outline_color=outline_color,
            outline_width=outline_width,
            bold=render_settings.bold,
            italic=render_settings.italic,
            underline=render_settings.underline,
            position=(pos_x, pos_y),
            rotation=block.angle,
            scale=1.0,
            transform_origin=block.tr_origin_point,
            width=rendered_width,
            height=rendered_height,
            direction=render_settings.direction,
            vertical=vertical,
            selection_outlines=[
                OutlineInfo(0, len(wrapped), outline_color, outline_width, OutlineType.Full_Document)
            ] if render_settings.outline else [],
        )
        text_items_state.append(props.to_dict())

    renderer = ImageSaveRenderer(cleaned)
    renderer.add_state_to_image({"text_items_state": text_items_state})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ext = path.suffix if args.output_format == "source" else f".{args.output_format.lstrip('.')}"
    out_path = output_dir / f"{path.stem}{args.suffix}{output_ext}"
    print(f"[6/6] save: {out_path}", flush=True)
    saved_paths = save_rendered_output(renderer, out_path)
    apply_page_stamp_to_outputs(saved_paths, path, args)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Batch pipeline  (BATCH_PIPELINE=true)
# ─────────────────────────────────────────────────────────────────────────────

def _stage_detect_ocr(task: tuple) -> tuple:
    """Phase-1 worker: detect + OCR for one page (thread-safe, fresh instances per call)."""
    idx, path, args = task
    settings = LocalSettings(args)
    main = LocalMain(settings, args.source_lang, args.target_lang)
    image = imk.read_image(str(path))
    blocks = TextBlockDetector(settings).detect(image)
    if blocks:
        ocr = OCRProcessor()
        ocr.initialize(main, args.source_lang)
        ocr.process(image, blocks)
    recover_empty_bubble_text_with_gpt(image, blocks, args)
    blocks = drop_non_english_ocr_blocks(blocks)
    blocks = prune_ocr_blocks(blocks, image)
    strip_translation_garbage_letters(blocks)
    blocks = sort_blk_list(blocks, right_to_left=args.source_lang == "Japanese")
    print(f"  detect+ocr [{idx+1}] {path.name}: {len(blocks)} blok", flush=True)
    return idx, path, image, blocks


def _batch_translate_all(page_results: list, args: argparse.Namespace) -> None:
    """Phase-2: translate all pages together in as few GPT requests as possible."""
    combined: dict[str, str] = {}
    for pi, (_path, _img, blocks) in enumerate(page_results):
        for bi, blk in enumerate(blocks):
            combined[f"p{pi}_b{bi}"] = blk.text or ""
    if not combined:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = MODEL_MAP.get(args.translator, "gpt-5.4-mini")
    system_prompt = (
        f"You are an expert translator who translates {args.source_lang} to {args.target_lang}. "
        f"You pay attention to style, formality, idioms, slang etc and try to convey it in the way "
        f"a {args.target_lang} speaker would understand.\n"
        f"BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.\n"
        f"You will translate text OCR'd from a comic. The OCR is not always perfect.\n"
        f"Return only the JSON with translated values. Do NOT translate the keys. "
        f"If a block is already in {args.target_lang} or looks like gibberish, output it as-is."
    )

    batch_size = int(os.environ.get("BATCH_TRANSLATE_SIZE", "400"))
    keys = list(combined.keys())
    batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]
    results: dict[str, str] = {}

    for bi, batch_keys in enumerate(batches):
        batch_dict = {k: combined[k] for k in batch_keys}
        user_prompt = (
            f"{args.extra_context}\n"
            f"Make the translation sound as natural as possible.\n"
            f"Translate this:\n"
            f"{json.dumps(batch_dict, ensure_ascii=False, indent=2)}"
        )
        print(f"  GPT call [{bi+1}/{len(batches)}]: {len(batch_keys)} blok", flush=True)
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                    ],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_completion_tokens": max(16000, len(batch_keys) * 300),
                }),
                timeout=180,
            )
            if not resp.ok:
                print(f"  batch {bi+1} translate FAILED: {resp.status_code} {resp.text}", flush=True)
                continue
            resp.raise_for_status()
            resp_data = resp.json()
            from modules.translation.llm.gpt import _log_openai_usage
            _log_openai_usage(model, resp_data.get("usage") or {})
            usage_dbg_path = os.environ.get("BATCH_TRANSLATE_USAGE_PATH")
            if usage_dbg_path:
                with open(usage_dbg_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(resp_data.get("usage") or {}) + "\n")
            content = resp_data["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                results.update(json.loads(m.group(0)))
            else:
                finish_reason = resp_data["choices"][0].get("finish_reason")
                print(
                    f"  batch {bi+1}: yanıtta JSON bulunamadı "
                    f"(finish_reason={finish_reason}, content_len={len(content)})",
                    flush=True,
                )
        except Exception as exc:
            print(f"  batch {bi+1} translate FAILED: {exc}", flush=True)

    for pi, (_path, _img, blocks) in enumerate(page_results):
        for bi, blk in enumerate(blocks):
            val = results.get(f"p{pi}_b{bi}")
            if val is not None:
                blk.translation = val


def _stage_clean(task: tuple) -> tuple:
    """Phase-3a worker: YOLOv8 segmentasyon maskesi + block-aware LaMa ile temizlik."""
    idx, path, image, blocks, args = task
    config = Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)

    mask = build_segmenter_mask(image, blocks)
    if mask.sum() == 0:
        cleaned = image.copy()
    else:
        cleaned = block_inpaint(image, mask, config)

    suppress_noisy_ocr_blocks(blocks, args.source_lang)
    strip_translation_garbage_letters(blocks)
    debug_dump_blocks("after-clean-translate", blocks)
    cleaned = restore_untranslated_regions(image, cleaned, blocks)
    print(f"  clean [{idx+1}] {path.name}: OK", flush=True)
    return idx, path, image, blocks, cleaned


def _do_render_and_save(
    idx: int,
    n_total: int,
    path: Path,
    image: np.ndarray,
    blocks: list,
    cleaned: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    main: LocalMain,
) -> Path:
    """Render translated text onto cleaned image and save. Must run on Qt thread."""
    output_ext = path.suffix if args.output_format == "source" else f".{args.output_format.lstrip('.')}"
    out_path = output_dir / f"{path.stem}{args.suffix}{output_ext}"

    if not blocks:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as src:
            src = ImageOps.exif_transpose(src)
            if output_ext.lower() in {".jpg", ".jpeg", ".webp"} and src.mode not in {"RGB", "RGBA"}:
                src = src.convert("RGBA" if "A" in src.getbands() else "RGB")
            src.save(out_path)
        apply_page_stamp(out_path, path, args, idx + 1)
        print(f"  [{idx+1}/{n_total}] passthrough: {out_path.name}", flush=True)
        return out_path

    target_code = get_language_code(args.target_lang)
    render_settings = main.render_settings()
    format_translations(blocks, target_code, upper_case=render_settings.upper_case)
    get_best_render_area(blocks, image, cleaned)
    cleaned = clear_light_free_text_render_regions(cleaned, blocks)
    cleaned = clear_light_bubble_text_render_regions(cleaned, blocks, image)

    text_items_state = []
    for blk in blocks:
        translation = blk.translation
        if not translation or len(translation) == 1:
            continue
        x1, y1, block_width, block_height = blk.xywh
        vertical = is_vertical_block(blk, target_code)
        alignment = main.button_to_alignment[render_settings.alignment_id]
        outline_width = float(render_settings.outline_width)
        wrapped, font_size, rendered_width, rendered_height = pyside_word_wrap(
            translation,
            render_settings.font_family,
            int(block_width),
            int(block_height),
            float(render_settings.line_spacing),
            outline_width,
            render_settings.bold,
            render_settings.italic,
            render_settings.underline,
            alignment,
            render_settings.direction,
            render_settings.max_font_size,
            render_settings.min_font_size,
            vertical,
            return_metrics=True,
        )
        if is_no_space_lang(target_code):
            wrapped = wrapped.replace(" ", "")
        font_color = get_smart_text_color(blk.font_color, QColor(render_settings.color))
        outline_color = contrast_outline_color(font_color) if render_settings.outline else None
        pos_x = int(x1 + max(0, (block_width - rendered_width) / 2))
        pos_y = int(y1 + max(0, (block_height - rendered_height) / 2))
        props = TextItemProperties(
            text=wrapped,
            font_family=render_settings.font_family,
            font_size=font_size,
            text_color=font_color,
            alignment=alignment,
            line_spacing=float(render_settings.line_spacing),
            outline_color=outline_color,
            outline_width=outline_width,
            bold=render_settings.bold,
            italic=render_settings.italic,
            underline=render_settings.underline,
            position=(pos_x, pos_y),
            rotation=blk.angle,
            scale=1.0,
            transform_origin=blk.tr_origin_point,
            width=rendered_width,
            height=rendered_height,
            direction=render_settings.direction,
            vertical=vertical,
            selection_outlines=[
                OutlineInfo(0, len(wrapped), outline_color, outline_width, OutlineType.Full_Document)
            ] if render_settings.outline else [],
        )
        text_items_state.append(props.to_dict())

    renderer = ImageSaveRenderer(cleaned)
    renderer.add_state_to_image({"text_items_state": text_items_state})
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{idx+1}/{n_total}] save: {out_path.name}", flush=True)
    saved_paths = save_rendered_output(renderer, out_path)
    apply_page_stamp_to_outputs(saved_paths, path, args, idx + 1)
    return out_path


def _already_done(path: Path, output_dir: Path, args: argparse.Namespace) -> bool:
    output_ext = path.suffix if args.output_format == "source" else f".{args.output_format.lstrip('.')}"
    return (output_dir / f"{path.stem}{args.suffix}{output_ext}").exists()


def _save_passthrough(path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    """Copy source image to output without processing (error fallback)."""
    output_ext = path.suffix if args.output_format == "source" else f".{args.output_format.lstrip('.')}"
    out_path = output_dir / f"{path.stem}{args.suffix}{output_ext}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as src:
        src = ImageOps.exif_transpose(src)
        if output_ext.lower() in {".jpg", ".jpeg", ".webp"} and src.mode not in {"RGB", "RGBA"}:
            src = src.convert("RGBA" if "A" in src.getbands() else "RGB")
        src.save(out_path)
    return out_path


def _dynamic_workers(base: int) -> int:
    """Scale worker count up/down based on live CPU load."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.3)
        if cpu > 80:
            adjusted = max(1, base - 1)
        elif cpu < 35:
            adjusted = min(base + 1, os.cpu_count() or base)
        else:
            return base
        if adjusted != base:
            print(f"  [CPU {cpu:.0f}%] worker: {base} -> {adjusted}", flush=True)
        return adjusted
    except ImportError:
        return base


def _rate_translation_quality(page_results: list, args: argparse.Namespace) -> float:
    """
    Sample up to QUALITY_SCORE_SAMPLE block pairs, send to GPT for 1-5 rating.
    Returns average score (0.0 = disabled/failed).
    """
    if os.environ.get("TRANSLATION_QUALITY_SCORE", "false").lower() != "true":
        return 0.0
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 0.0

    import random
    pairs: list[tuple[str, str]] = []
    for _path, _img, blocks in page_results:
        for blk in blocks:
            src = (getattr(blk, "text", "") or "").strip()
            tr = (getattr(blk, "translation", "") or "").strip()
            if src and tr and src != tr:
                pairs.append((src, tr))
    if not pairs:
        return 0.0

    sample = random.sample(pairs, min(int(os.environ.get("QUALITY_SCORE_SAMPLE", "20")), len(pairs)))
    sample_json = json.dumps([{"source": s, "translation": t} for s, t in sample], ensure_ascii=False)
    model = MODEL_MAP.get(args.translator, "gpt-5.4-mini")
    prompt = (
        f"Rate the quality of each {args.source_lang}→{args.target_lang} translation on a scale "
        f"of 1 (bad) to 5 (perfect). Consider accuracy, naturalness, and style. "
        f"Return only compact JSON: {{\"scores\": [1, 4, ...]}}.\n{sample_json}"
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_completion_tokens": 500,
            }),
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            scores = [float(s) for s in json.loads(m.group(0)).get("scores", []) if s]
            if scores:
                avg = sum(scores) / len(scores)
                flag = " [DUSUK KALITE!]" if avg < 3 else ""
                print(f"  [KALITE] Ortalama skor: {avg:.1f}/5{flag}", flush=True)
                return avg
    except Exception as exc:
        print(f"  [KALITE] skor hesaplanamadi: {exc}", flush=True)
    return 0.0


def _stage_clean_safe(task: tuple) -> tuple:
    """_stage_clean with exception fallback: returns original image as cleaned on failure."""
    try:
        return _stage_clean(task)
    except Exception as exc:
        idx, path, image, blocks, args = task
        print(f"  clean FAILED [{idx+1}] {path.name}: {exc}, passthrough", file=sys.stderr, flush=True)
        return idx, path, image, [], image.copy()


def process_all_batched(images: list[Path], output_dir: Path, args: argparse.Namespace) -> int:
    """
    3-phase batch pipeline:
      Faz 1 — Parallel detect+OCR  (BATCH_WORKERS thread)
      Faz 2 — Tüm sayfa metinleri tek GPT call ile translate  +  kalite skoru
      Faz 3 — Parallel clean  +  sequential render+save (Qt main-thread zorunlu)

    Aktifleştir         : BATCH_PIPELINE=true
    Paralelllik         : BATCH_WORKERS=N  (default 3)
    Kalite skoru        : TRANSLATION_QUALITY_SCORE=true
    Retry               : PAGE_MAX_RETRIES=N  (default 3)
    Resume              : zaten çevrilmiş sayfalar otomatik atlanır
    """
    base_workers = int(os.environ.get("BATCH_WORKERS", "2"))
    max_retries = int(os.environ.get("PAGE_MAX_RETRIES", "3"))
    settings = LocalSettings(args)
    main = LocalMain(settings, args.source_lang, args.target_lang)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── Resume: zaten işlenmiş sayfaları atla ───────────────────────────────
    pending = [p for p in images if not _already_done(p, output_dir, args)]
    skipped = len(images) - len(pending)
    if skipped:
        print(f"[BATCH] Resume: {skipped} sayfa zaten mevcut, atlanıyor", flush=True)
    if not pending:
        print("[BATCH] Tüm sayfalar zaten çevrilmiş.", flush=True)
        return 0

    # ── Model warm-up: thread'ler başlamadan önce modeli main thread'de indir ──
    print("[BATCH] Model warm-up...", flush=True)
    try:
        _warmup_settings = LocalSettings(args)
        TextBlockDetector(_warmup_settings).detect(imk.read_image(str(pending[0])))
        print("[BATCH] Model hazır.", flush=True)
    except Exception as _warmup_exc:
        print(f"[BATCH] Warm-up uyarı: {_warmup_exc}", flush=True)

    # ── Faz 1: Parallel detect + OCR ────────────────────────────────────────
    w1 = _dynamic_workers(base_workers)
    print(f"\n[BATCH] Faz 1/3  detect+ocr  {len(pending)} sayfa  {w1} worker", flush=True)
    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=w1) as pool:
        phase1 = list(pool.map(_stage_detect_ocr, [(i, p, args) for i, p in enumerate(pending)]))
    page_results = [(r[1], r[2], r[3]) for r in phase1]
    n_blk = sum(len(bl) for _, _, bl in page_results)
    print(f"[BATCH] Faz 1 bitti: {time.perf_counter()-t1:.1f}s  toplam {n_blk} blok", flush=True)

    # ── Faz 2: Batch translate ───────────────────────────────────────────────
    n_text = sum(1 for _, _, bl in page_results if any(b.text for b in bl))
    if n_text:
        print(f"\n[BATCH] Faz 2/3  batch translate  {n_text} sayfa metinli  1 GPT call", flush=True)
        t2 = time.perf_counter()
        _batch_translate_all(page_results, args)
        print(f"[BATCH] Faz 2 bitti: {time.perf_counter()-t2:.1f}s", flush=True)
        _rate_translation_quality(page_results, args)
    else:
        print(f"\n[BATCH] Faz 2/3  translate atlandi (metin bulunamadi)", flush=True)

    # ── Faz 3a: Parallel clean (with safe fallback) ──────────────────────────
    w3 = _dynamic_workers(base_workers)
    print(f"\n[BATCH] Faz 3/3  clean+render+save  {w3} worker", flush=True)
    t3 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=w3) as pool:
        clean_results = list(pool.map(
            _stage_clean_safe,
            [(i, p, img, bl, args) for i, (p, img, bl) in enumerate(page_results)],
        ))
    print(f"[BATCH] clean bitti: {time.perf_counter()-t3:.1f}s", flush=True)

    # ── Faz 3b: Sequential render + save (Qt main thread) ───────────────────
    failed = 0
    for idx, path, image, blocks, cleaned in clean_results:
        os.environ["TRANSLATION_SOURCE_FILE"] = path.name
        for attempt in range(1, max_retries + 1):
            try:
                _do_render_and_save(idx, len(pending), path, image, blocks, cleaned, output_dir, args, main)
                break
            except Exception as exc:
                if attempt < max_retries:
                    print(f"  render retry {attempt}/{max_retries} {path.name}: {exc}", flush=True)
                else:
                    print(f"  render FAILED {path.name}, passthrough kaydediliyor", file=sys.stderr, flush=True)
                    try:
                        _save_passthrough(path, output_dir, args)
                    except Exception:
                        failed += 1

    print(f"\n[BATCH] Toplam sure: {time.perf_counter()-t_total:.1f}s  hata: {failed}", flush=True)
    return failed


def save_rendered_output(renderer: ImageSaveRenderer, out_path: Path) -> list[Path]:
    final_rgb = renderer.render_to_image()
    max_dimension = int(os.environ.get("WEBP_MAX_DIMENSION", "16000"))
    if out_path.suffix.lower() == ".webp" and final_rgb.shape[0] > max_dimension:
        saved: list[Path] = []
        stem = out_path.stem
        top = 0
        part = 1
        while top < final_rgb.shape[0]:
            bottom = min(final_rgb.shape[0], top + max_dimension)
            target = out_path.with_name(f"{stem}_part{part:03d}{out_path.suffix}")
            imk.write_image(str(target), final_rgb[top:bottom, :, :])
            saved.append(target)
            top = bottom
            part += 1
        return saved
    imk.write_image(str(out_path), final_rgb)
    return [out_path]


def _image_sort_key(p: Path) -> tuple:
    m = re.search(r"(\d+)", p.stem)
    return (0, int(m.group(1)), p.name) if m else (1, 0, p.name)


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=_image_sort_key,
    )


def r2_work_name(prefix: str) -> str:
    return normalize_key(prefix).replace("/", "__").replace(":", "_") or "chapter"


def translated_prefix(raw_prefix: str) -> str:
    raw_prefix = normalize_key(raw_prefix)
    raw_root = normalize_key(os.environ.get("R2_RAW_PREFIX", "raw"))
    translated_root = normalize_key(os.environ.get("R2_TRANSLATED_PREFIX", "translated"))
    if raw_prefix == raw_root:
        return translated_root
    if raw_prefix.startswith(raw_root + "/"):
        return normalize_key(translated_root, raw_prefix[len(raw_root):].lstrip("/"))
    return normalize_key(translated_root, raw_prefix)


def seo_upload_name(prefix: str, page_number: int, extension: str) -> str:
    """Build the uploaded image filename. Fully driven by env vars
    (SEO_IMAGE_BRAND, SEO_FILENAME_LANGUAGE_SLUG, SEO_FILENAME_TEMPLATE) so
    deployments can rebrand without touching code."""
    parts = normalize_key(prefix).split("/")
    series_slug = parts[-2] if len(parts) >= 2 else "manga"
    chapter_slug = parts[-1] if parts else "chapter"
    brand = os.environ.get("SEO_IMAGE_BRAND", "trendmanga").strip().lower() or "trendmanga"
    language_slug = os.environ.get("SEO_FILENAME_LANGUAGE_SLUG", "turkce-manga-oku").strip().lower()
    template = os.environ.get(
        "SEO_FILENAME_TEMPLATE",
        "{brand}-{language_slug}-{series}-{chapter}-sayfa-{page}{ext}",
    )
    return template.format(
        brand=brand,
        language_slug=language_slug,
        series=series_slug,
        chapter=chapter_slug,
        page=page_number,
        ext=extension.lower(),
    )


def prepare_seo_upload_dir(output_dir: Path, out_prefix: str, work_dir: Path) -> Path:
    seo_dir = work_dir / "seo-upload"
    if seo_dir.exists():
        shutil.rmtree(seo_dir)
    seo_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(output_dir)
    for index, image in enumerate(images, start=1):
        target = seo_dir / seo_upload_name(out_prefix, index, image.suffix)
        shutil.copy2(image, target)
    return seo_dir


def parse_args() -> argparse.Namespace:
    load_env(ROOT / ".env")
    detector_default = os.environ.get("DETECTOR", "RT-DETR-v4-s-int8")
    if detector_default in {"Qwen3-VL-Flash", "Qwen3-VL-Plus"}:
        detector_default = "RT-DETR-v4-s-int8"
    parser = argparse.ArgumentParser(description="Headless local manga translation batch runner.")
    parser.add_argument("--input", default="input", help="Input image folder")
    parser.add_argument("--output", default="output", help="Output image folder")
    parser.add_argument("--env", default=".env", help="Env file containing OPENAI_API_KEY")
    parser.add_argument("--source-lang", default=os.environ.get("SOURCE_LANG", "English"))
    parser.add_argument("--target-lang", default=os.environ.get("TARGET_LANG", "Turkish"))
    parser.add_argument("--translator", default=os.environ.get("TRANSLATOR", "GPT-5.4-mini"))
    parser.add_argument("--detector", default=detector_default, choices=["RT-DETR-v2", "RT-DETR-v4-s-int8"])
    parser.add_argument("--ocr", default=os.environ.get("OCR", os.environ.get("OCR_ENGINE", "Default")))
    parser.add_argument("--inpainter", default="SmartCV2", choices=["SmartCV2"])
    parser.add_argument("--mask-refiner", default="none", choices=["none"])
    parser.add_argument("--hd-strategy", default=os.environ.get("HD_STRATEGY", "Resize"), choices=["Resize", "Original", "Crop"])
    parser.add_argument("--resize-limit", type=int, default=int(os.environ.get("RESIZE_LIMIT", "960")))
    parser.add_argument("--extra-context", default=os.environ.get("EXTRA_CONTEXT", ""))
    parser.add_argument("--font", default=os.environ.get("FONT_FAMILY", "Arial"))
    parser.add_argument("--font-file", default=os.environ.get("FONT_FILE", ""))
    parser.add_argument("--min-font-size", type=int, default=int(os.environ.get("MIN_FONT_SIZE", "14")))
    parser.add_argument("--max-font-size", type=int, default=int(os.environ.get("MAX_FONT_SIZE", "58")))
    parser.add_argument("--suffix", default=os.environ.get("OUTPUT_SUFFIX", "_translated"))
    parser.add_argument("--output-format", default=os.environ.get("OUTPUT_FORMAT", "webp"), choices=["source", "webp", "png", "jpg", "jpeg"])
    parser.add_argument("--stamp-enabled", default=os.environ.get("STAMP_ENABLED", "true"), choices=["true", "false"])
    parser.add_argument("--stamp-path", default=os.environ.get("STAMP_PATH", str(DEFAULT_STAMP_PATH)))
    parser.add_argument("--stamp-every-pages", type=int, default=int(os.environ.get("STAMP_EVERY_PAGES", "5")))
    parser.add_argument("--stamp-opacity", type=float, default=float(os.environ.get("STAMP_OPACITY", "1.0")))
    parser.add_argument("--stamp-width-ratio", type=float, default=float(os.environ.get("STAMP_WIDTH_RATIO", "0.2")))
    parser.add_argument("--stamp-black-background", default=os.environ.get("STAMP_BLACK_BACKGROUND", "false"), choices=["true", "false"])
    parser.add_argument("--gpu", action="store_true", default=os.environ.get("USE_GPU", "false").lower() == "true")
    parser.add_argument("--uppercase", action=argparse.BooleanOptionalAction, default=os.environ.get("UPPERCASE", "true").lower() == "true")
    parser.add_argument("--no-outline", action="store_true", default=os.environ.get("NO_OUTLINE", "false").lower() == "true")
    parser.add_argument("--r2-input-prefix", default=os.environ.get("R2_INPUT_PREFIX", ""))
    parser.add_argument("--r2-output-prefix", default=os.environ.get("R2_OUTPUT_PREFIX", ""))
    parser.add_argument("--r2-delete-input", action=argparse.BooleanOptionalAction, default=os.environ.get("R2_DELETE_RAW_AFTER_TRANSLATE", "true").lower() == "true")
    parser.add_argument("--r2-workdir", default=os.environ.get("R2_WORKDIR", "work/r2"))
    parser.add_argument("--r2-keep-workdir", action="store_true", default=os.environ.get("R2_KEEP_WORKDIR", "false").lower() == "true")
    parser.add_argument("--zip-output", action=argparse.BooleanOptionalAction, default=os.environ.get("ZIP_OUTPUT", "false").lower() == "true",
                         help="Zip the output folder into <output>.zip after a local (non-R2) run finishes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["MASK_REFINER"] = args.mask_refiner
    load_env((ROOT / args.env).resolve())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args.resolved_font = resolve_font(args)

    using_r2 = bool(args.r2_input_prefix)
    if using_r2:
        chapter_work = ROOT / args.r2_workdir / r2_work_name(args.r2_input_prefix)
        input_dir = (chapter_work / "input").resolve()
        output_dir = (chapter_work / "output").resolve()
        if chapter_work.exists() and not args.r2_keep_workdir:
            shutil.rmtree(chapter_work)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"R2 download: {args.r2_input_prefix} -> {input_dir}", flush=True)
        downloaded = download_prefix(args.r2_input_prefix, input_dir)
        print(f"R2 downloaded: {len(downloaded)} images", flush=True)
    else:
        input_dir = (ROOT / args.input).resolve()
        output_dir = (ROOT / args.output).resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing. Put it in .env or set it as an environment variable.", file=sys.stderr)
        return 2

    images = iter_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    if os.environ.get("BATCH_PIPELINE", "false").lower() == "true":
        failed = process_all_batched(images, output_dir, args)
    else:
        max_retries = int(os.environ.get("PAGE_MAX_RETRIES", "3"))
        # Resume: skip already-done pages
        pending = [p for p in images if not _already_done(p, output_dir, args)]
        skipped = len(images) - len(pending)
        if skipped:
            print(f"[RESUME] {skipped} sayfa zaten mevcut, atlanıyor", flush=True)
        failed = 0
        for index, path in enumerate(pending, start=1):
            print(f"\n=== {index}/{len(pending)} {path.name} ===", flush=True)
            for attempt in range(1, max_retries + 1):
                try:
                    process_image(path, output_dir, args)
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        print(f"  retry {attempt}/{max_retries}: {exc}", flush=True)
                    else:
                        print(f"FAILED {path.name}, passthrough kaydediliyor", file=sys.stderr, flush=True)
                        try:
                            _save_passthrough(path, output_dir, args)
                        except Exception:
                            failed += 1

    if using_r2 and failed == 0:
        out_prefix = args.r2_output_prefix or translated_prefix(args.r2_input_prefix)
        upload_dir = prepare_seo_upload_dir(output_dir, out_prefix, chapter_work)
        print(f"R2 upload: {upload_dir} -> {out_prefix}", flush=True)
        deleted_output = delete_prefix(out_prefix)
        if deleted_output:
            print(f"R2 old translated deleted: {deleted_output} objects", flush=True)
        manifest = upload_directory(upload_dir, out_prefix)
        print(f"R2 uploaded: {manifest['count']} images", flush=True)
        if args.r2_delete_input:
            deleted = delete_prefix(args.r2_input_prefix)
            print(f"R2 raw deleted: {deleted} objects", flush=True)
        if not args.r2_keep_workdir:
            shutil.rmtree(chapter_work, ignore_errors=True)
    elif using_r2 and failed:
        print("R2 upload/delete skipped because translation had failures.", file=sys.stderr, flush=True)

    if not using_r2 and args.zip_output and not failed:
        zip_base = str(output_dir)
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=output_dir)
        print(f"Zipped output -> {zip_path}", flush=True)

    app.quit()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
