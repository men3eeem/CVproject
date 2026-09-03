"""
Egyptian License Plate OCR - PaddleOCR version

Pipeline:
    Original image
      -> YOLO plate detection
      -> plate crop + padding
      -> remove the blue/header part of the Egyptian plate
      -> perspective/contrast/scale preprocessing
      -> split the white plate into NUMBERS and ARABIC LETTERS
      -> PaddleOCR Arabic recognition on each side
      -> several preprocessing attempts + voting

IMPORTANT:
No OCR system can guarantee 100% accuracy for every image/condition.
For real deployment, accuracy should be measured on your own Egyptian-plate
validation set and the recognition model should be fine-tuned if necessary.
"""

import argparse
import re
from pathlib import Path
from typing import Union

import cv2
import numpy as np

DEFAULT_PLATE_WEIGHTS = Path("best_p.pt")
OCR_MODEL = "arabic_PP-OCRv3_mobile_rec"
_reader = None

# Arabic letters commonly used on Egyptian plates + Arabic/Western digits.
ARABIC_ALLOWED = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةى")
WESTERN_DIGITS = set("0123456789")
ARABIC_DIGITS = set("٠١٢٣٤٥٦٧٨٩")


def _get_reader(device="auto"):
    """Create PaddleOCR Arabic recognition model once."""
    global _reader
    if _reader is not None:
        return _reader

    from paddleocr import TextRecognition

    if device == "auto":
        try:
            import paddle
            device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        except Exception:
            device = "cpu"

    print(f"Loading PaddleOCR model: {OCR_MODEL}")
    print(f"PaddleOCR device: {device}")

    _reader = TextRecognition(
        model_name=OCR_MODEL,
        device=device,
    )
    return _reader


def _safe_crop(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return image[y1:y2, x1:x2]


def _crop_with_padding(image: np.ndarray, bbox: list, pad_ratio: float = 0.06):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return _safe_crop(image, x1 - px, y1 - py, x2 + px, y2 + py)


def _rectify_plate(plate: np.ndarray) -> np.ndarray:
    """Straighten a tilted Egyptian plate using its blue header as the orientation cue."""
    if plate is None or plate.size == 0:
        return plate

    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([80, 45, 35]), np.array([140, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < max(20.0, plate.shape[0] * plate.shape[1] * 0.02):
        return plate

    rect = cv2.minAreaRect(cnt)
    angle = float(rect[2])
    rw, rh = rect[1]
    if rw < rh:
        angle += 90.0

    # Only correct meaningful tilt; avoid introducing interpolation damage.
    if abs(angle) < 2.0 or abs(angle) > 35.0:
        return plate

    h, w = plate.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - center[0]
    M[1, 2] += nh / 2.0 - center[1]
    return cv2.warpAffine(plate, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _remove_blue_header(plate: np.ndarray) -> np.ndarray:
    """Remove the blue Egyptian plate header while keeping the white plate body."""
    if plate.size == 0:
        return plate

    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    # Blue/cyan range. Kept deliberately broad for different lighting.
    mask = cv2.inRange(hsv, np.array([80, 50, 40]), np.array([140, 255, 255]))

    row_score = mask.mean(axis=1)
    h = plate.shape[0]

    # Only consider the upper ~45%, where the blue header normally lives.
    limit = max(1, int(h * 0.45))
    rows = np.where(row_score[:limit] > 0.08)[0]

    if len(rows) > 0:
        bottom_blue = int(rows.max())
        # Keep a small margin below the detected blue area.
        start = min(h - 1, bottom_blue + max(2, int(h * 0.04)))
        body = plate[start:, :]
        if body.shape[0] >= max(10, int(h * 0.35)):
            return body

    # Fallback: the white character area is normally the lower ~65-75%.
    return plate[int(h * 0.30):, :]


def _resize_for_ocr(img: np.ndarray) -> np.ndarray:
    """Upscale small character regions to a useful OCR height."""
    h, w = img.shape[:2]
    target_h = 180
    scale = max(2.0, target_h / max(1, h))
    new_w = max(32, int(w * scale))
    new_h = max(target_h, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _make_variants(img: np.ndarray) -> list[np.ndarray]:
    """Create robust OCR variants without destroying the original information."""
    if img.size == 0:
        return []

    up = _resize_for_ocr(img)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Mild denoise, then local contrast.
    den = cv2.bilateralFilter(gray, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(den)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )

    variants = [gray, enhanced, sharp, otsu, adaptive]
    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in variants]


def _extract_result(result):
    """Handle PaddleOCR 3.x result object/dict formats."""
    data = result
    try:
        data = result["res"]
    except Exception:
        try:
            data = result.res
        except Exception:
            pass

    if isinstance(data, dict):
        text = data.get("rec_text", "")
        score = data.get("rec_score", 0.0)
    else:
        text = getattr(data, "rec_text", "")
        score = getattr(data, "rec_score", 0.0)

    try:
        score = float(score)
    except Exception:
        score = 0.0
    return str(text or "").strip(), score


def _normalize_text(text: str) -> str:
    """Remove OCR punctuation/spaces while preserving Arabic letters and digits."""
    text = text.replace(" ", "").replace("\u200f", "").replace("\u200e", "")
    text = text.replace("-", "").replace("_", "")
    # Keep only relevant Arabic letters and both digit systems.
    return "".join(c for c in text if c in ARABIC_ALLOWED or c in WESTERN_DIGITS or c in ARABIC_DIGITS)


def _is_letter(c):
    return c in ARABIC_ALLOWED


def _is_digit(c):
    return c in WESTERN_DIGITS or c in ARABIC_DIGITS


def _clean_side(text: str, side: str) -> str:
    """Keep the expected character class for each side."""
    text = _normalize_text(text)
    if side == "letters":
        return "".join(c for c in text if _is_letter(c))
    return "".join(c for c in text if _is_digit(c))


def _recognize_side(reader, side_img: np.ndarray, side: str):
    """Recognize one side several times and vote among valid candidates."""
    variants = _make_variants(side_img)
    candidates = []

    for variant in variants:
        try:
            outputs = reader.predict(input=variant)
            for result in outputs:
                text, score = _extract_result(result)
                cleaned = _clean_side(text, side)
                if cleaned:
                    candidates.append((cleaned, score))
        except Exception as exc:
            print(f"PaddleOCR warning ({side}): {exc}")

    if not candidates:
        return "", 0.0

    # Prefer candidates with the most plausible length, then confidence.
    expected_max = 4 if side == "letters" else 5
    filtered = [(t, s) for t, s in candidates if len(t) <= expected_max]
    if filtered:
        candidates = filtered

    # Weighted voting: confidence contributes, repeated agreement contributes too.
    scores = {}
    best_conf = {}
    for text, conf in candidates:
        scores[text] = scores.get(text, 0.0) + max(0.01, conf)
        best_conf[text] = max(best_conf.get(text, 0.0), conf)

    winner = max(scores, key=lambda t: (scores[t], best_conf[t], len(t)))
    return winner, best_conf[winner]


def _find_split(body: np.ndarray) -> int:
    """Estimate the vertical separator between Arabic letters and numbers."""
    h, w = body.shape[:2]
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Look for a strong vertical line near the middle of the white plate.
    vertical_score = edges.mean(axis=0)
    lo = int(w * 0.35)
    hi = int(w * 0.65)

    if hi > lo:
        local = vertical_score[lo:hi]
        if local.size:
            idx = int(np.argmax(local)) + lo
            # Only trust it if it is meaningfully stronger than the local median.
            med = float(np.median(local))
            if float(vertical_score[idx]) > med * 1.8 and vertical_score[idx] > 3.0:
                return idx

    return int(w * 0.50)


def _split_plate(body: np.ndarray):
    """Return (numbers_crop, letters_crop) with overlap around the divider."""
    h, w = body.shape[:2]
    split = _find_split(body)
    overlap = max(3, int(w * 0.08))

    left = body[:, :min(w, split + overlap)]
    right = body[:, max(0, split - overlap):]

    # In the normal Egyptian plate layout, digits are on the left and Arabic
    # letters are on the right. We keep this explicit and also test both sides
    # independently with character-class filtering.
    return left, right, split


def _save_debug_crops(plate, body, numbers, letters, output_dir, plate_id):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"plate_{plate_id}_crop.jpg"), plate)
    cv2.imwrite(str(out / f"plate_{plate_id}_body.jpg"), body)
    cv2.imwrite(str(out / f"plate_{plate_id}_numbers.jpg"), numbers)
    cv2.imwrite(str(out / f"plate_{plate_id}_letters.jpg"), letters)


def _recognize_plate(reader, plate: np.ndarray, plate_id: int, debug_dir=None):
    """Recognize numbers and Arabic letters separately."""
    # IMPORTANT: straighten first. The uploaded example is strongly tilted,
    # and splitting an unrectified plate makes the separator and OCR crops wrong.
    rectified = _rectify_plate(plate)
    body = _remove_blue_header(rectified)

    if body.size == 0:
        return "", "", 0.0, 0.0

    numbers_img, letters_img, split = _split_plate(body)

    # First assume standard Egyptian layout: numbers LEFT, letters RIGHT.
    numbers, n_conf = _recognize_side(reader, numbers_img, "digits")
    letters, l_conf = _recognize_side(reader, letters_img, "letters")

    # If one side fails, try the opposite side. This makes the code less
    # sensitive to mirrored/rotated crops and unusual plate layouts.
    if not numbers:
        alt_numbers, alt_conf = _recognize_side(reader, letters_img, "digits")
        if alt_conf > n_conf:
            numbers, n_conf = alt_numbers, alt_conf

    if not letters:
        alt_letters, alt_conf = _recognize_side(reader, numbers_img, "letters")
        if alt_conf > l_conf:
            letters, l_conf = alt_letters, alt_conf

    if debug_dir:
        # Save the rectified plate as the main crop so it is easy to inspect
        # whether YOLO cropping/rotation is the source of an OCR error.
        _save_debug_crops(rectified, body, numbers_img, letters_img, debug_dir, plate_id)

    return numbers, letters, n_conf, l_conf


def _format_plate(numbers: str, letters: str) -> str:
    """Readable output; also keep separate fields in the returned dictionary."""
    if letters and numbers:
        return f"{letters} {numbers}"
    return letters or numbers


def read_plates(
    image: Union[str, Path, np.ndarray],
    plate_detections: list[dict],
    conf_threshold: float = 0.0,
    device: str = "auto",
    debug_dir: str | None = None,
) -> list[dict]:
    """Run PaddleOCR on each YOLO plate detection."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be a file path or BGR numpy.ndarray")

    reader = _get_reader(device)
    results = []

    for i, det in enumerate(plate_detections):
        bbox = det["bbox"]
        plate_id = det.get("plate_id", i)
        plate = _crop_with_padding(img, bbox)

        numbers, letters, n_conf, l_conf = _recognize_plate(
            reader, plate, plate_id, debug_dir
        )

        # Average only over sides that produced a result.
        confs = [c for t, c in ((numbers, n_conf), (letters, l_conf)) if t]
        confidence = float(np.mean(confs)) if confs else 0.0

        if confidence < conf_threshold:
            plate_text = ""
        else:
            plate_text = _format_plate(numbers, letters)

        results.append({
            "plate_id": plate_id,
            "plate_text": plate_text,
            "letters": letters,
            "numbers": numbers,
            "letters_confidence": round(float(l_conf), 4),
            "numbers_confidence": round(float(n_conf), 4),
            "confidence": round(confidence, 4),
            "bbox": bbox,
        })

    return results


def _draw_debug_image(image_path, results, out_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        label = (
            f"L:{r['letters']}  N:{r['numbers']} "
            f"({r['confidence']:.2f})"
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(
            img, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2
        )

    cv2.imwrite(out_path, img)
    print(f"Saved annotated debug image to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="YOLO Egyptian plate detection + PaddleOCR Arabic/digit recognition"
    )
    ap.add_argument("--image", required=True, help="Original input image")
    ap.add_argument("--plate_weights", default=str(DEFAULT_PLATE_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.10,
                    help="YOLO plate detection confidence")
    ap.add_argument("--ocr_conf", type=float, default=0.0,
                    help="Minimum combined OCR confidence")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu:0"])
    ap.add_argument("--save_debug", default=None,
                    help="Save annotated original image")
    ap.add_argument("--save_crops", action="store_true",
                    help="Save plate/body/numbers/letters crops")
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"Loading YOLO weights: {args.plate_weights}")
    model = YOLO(args.plate_weights)
    pred = model.predict(source=args.image, conf=args.conf, verbose=False)[0]

    detections = []
    for i, box in enumerate(pred.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "plate_id": i,
            "confidence": round(float(box.conf.item()), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    print(f"Detected {len(detections)} plate(s)")

    debug_dir = "ocr_debug" if args.save_crops else None
    results = read_plates(
        args.image,
        detections,
        conf_threshold=args.ocr_conf,
        device=args.device,
        debug_dir=debug_dir,
    )

    print("\nOCR results:")
    for r in results:
        print(r)

    if args.save_debug:
        _draw_debug_image(args.image, results, args.save_debug)
