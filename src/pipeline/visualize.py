"""
src/pipeline/visualize.py — draws the final pipeline result onto the image.

IMPORTANT: cv2.putText CANNOT render Arabic text — OpenCV's built-in font
only has Latin glyphs, so Arabic plate text would silently render as empty
boxes. This module uses Pillow + arabic_reshaper + python-bidi instead,
which correctly reshapes Arabic letters into their connected forms and
reorders them right-to-left before drawing.

Setup (once, in your Colab/local environment):
    pip install arabic-reshaper python-bidi
    apt-get install -y fonts-noto-core     # provides an Arabic-capable font

If those aren't installed, this module falls back to drawing boxes without
text labels (with a one-time console warning) rather than crashing or
silently drawing garbled/missing Arabic — the returned plate_text data
itself is never affected, only the visual overlay.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _SHAPING_AVAILABLE = True
except ImportError:
    _SHAPING_AVAILABLE = False

# Common locations for an Arabic-capable font across Colab/Debian/Ubuntu/macOS.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/kacst/KacstOne.ttf",
    "/System/Library/Fonts/Supplemental/GeezaPro.ttc",  # macOS
    "/Library/Fonts/Arial Unicode.ttf",  # macOS
]

_font_cache = {}
_warned_no_font = False
_warned_no_shaping = False


def _find_arabic_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _get_font(size: int = 20):
    global _warned_no_font
    if size in _font_cache:
        return _font_cache[size]

    font_path = _find_arabic_font()
    if font_path is None:
        if not _warned_no_font:
            print("[visualize.py] WARNING: no Arabic-capable font found on this "
                  "system. Plate text labels will be skipped in the overlay "
                  "(boxes still drawn). Fix with: "
                  "apt-get install -y fonts-noto-core")
            _warned_no_font = True
        _font_cache[size] = None
        return None

    font = ImageFont.truetype(font_path, size)
    _font_cache[size] = font
    return font


def _shape_text(text: str) -> str:
    """Reshape + reorder Arabic text for correct display. Falls back to the
    raw string (which will render backwards/disconnected for Arabic, but at
    least renders) if the shaping libraries aren't installed."""
    global _warned_no_shaping
    if not _SHAPING_AVAILABLE:
        if not _warned_no_shaping and any("\u0600" <= c <= "\u06FF" for c in text):
            print("[visualize.py] WARNING: arabic_reshaper/python-bidi not "
                  "installed — Arabic text will render disconnected/reversed. "
                  "Fix with: pip install arabic-reshaper python-bidi")
            _warned_no_shaping = True
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def _draw_label(pil_img, text: str, position: tuple, color: tuple):
    """Draw one text label using PIL, with Arabic shaping if available and a
    font exists. Silently skips drawing (not crashing) if no font is found."""
    font = _get_font(20)
    if font is None:
        return  # already warned once in _get_font
    draw = ImageDraw.Draw(pil_img)
    shaped = _shape_text(text)
    # PIL expects RGB tuples; color passed in is BGR (OpenCV convention) — flip it
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, shaped, font=font, fill=rgb_color)


def draw_results(image: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Draw vehicle boxes (green, labeled with vehicle_type) and plate boxes
    (orange, labeled with plate_text) onto a copy of the image.

    Args:
        image: BGR numpy array (the original image, not a crop).
        results: output of process_image().

    Returns:
        Annotated BGR numpy array (copy — original is not modified).
    """
    out = image.copy()

    # Boxes drawn with OpenCV (fine — no text involved, just rectangles).
    drawn_vehicle_boxes = set()
    for r in results:
        if r["vehicle_bbox"] is not None:
            vbox_key = tuple(r["vehicle_bbox"])
            if vbox_key not in drawn_vehicle_boxes:
                x1, y1, x2, y2 = r["vehicle_bbox"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
                drawn_vehicle_boxes.add(vbox_key)

        px1, py1, px2, py2 = r["plate_bbox"]
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 140, 255), 2)

    # Text drawn with PIL (handles Arabic correctly; vehicle labels are
    # Latin-only but we use the same path for consistency).
    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    drawn_vehicle_labels = set()
    for r in results:
        if r["vehicle_bbox"] is not None:
            vbox_key = tuple(r["vehicle_bbox"])
            if vbox_key not in drawn_vehicle_labels:
                x1, y1, x2, y2 = r["vehicle_bbox"]
                label = r["vehicle_type"]
                if r["vehicle_confidence"] is not None:
                    label += f" {r['vehicle_confidence']:.2f}"
                _draw_label(pil_img, label, (x1, max(0, y1 - 24)), (0, 200, 0))
                drawn_vehicle_labels.add(vbox_key)

        px1, py1, px2, py2 = r["plate_bbox"]
        plate_label = r["plate_text"] if r["plate_text"] else "(no text)"
        if r["confidence"]:
            plate_label += f" {r['confidence']:.2f}"
        _draw_label(pil_img, plate_label, (px1, min(out.shape[0] - 24, py2 + 4)), (0, 140, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
