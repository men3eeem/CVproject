"""
infer.py — Person 2 (Plate Detection)

Exposes two functions the rest of the team calls directly:

    detect_plates(image) -> list[dict]
        [{"plate_id": 0, "confidence": 0.91, "bbox": [x1,y1,x2,y2]}, ...]

    crop_plate(image, bbox) -> np.ndarray
        Returns the cropped plate region, padded ~10% on each side per the
        project's failure-case fallback (Section 9) so a slightly tight box
        doesn't clip a character. Person 3 uses this directly — note it does
        NOT require a trained model to use; Person 3 can call it on
        ground-truth boxes from the dataset itself while your model is still
        training, per the plan's "don't block on each other" design.

Usage as a library:
    from infer import detect_plates, crop_plate
    plates = detect_plates("image.jpg")
    crop = crop_plate("image.jpg", plates[0]["bbox"])

Usage as a CLI (quick manual test + saves an annotated + a cropped image):
    python infer.py --image test.jpg --weights ../../models/plate_best.pt --save_debug out.jpg
"""

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "plate_best.pt"
CONFIDENCE_THRESHOLD = 0.35
CROP_PADDING_FRACTION = 0.10  # 10% margin, per Section 9 fallback for tight boxes

_model = None


def _get_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS) -> YOLO:
    global _model
    if _model is None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Plate weights not found at {weights_path}. "
                "Run train.py first, or pass weights_path explicitly."
            )
        _model = YOLO(str(weights_path))
    return _model


def detect_plates(
    image: Union[str, Path, np.ndarray],
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Run the fine-tuned YOLOv8n plate detector on a single image."""
    model = _get_model(weights_path)
    results = model.predict(source=image, conf=conf_threshold, verbose=False)[0]

    plates = []
    for i, box in enumerate(results.boxes):
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        plates.append({
            "plate_id": i,
            "confidence": round(confidence, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return plates


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


def crop_plate(
    image: Union[str, Path, np.ndarray],
    bbox: list[int],
    padding_fraction: float = CROP_PADDING_FRACTION,
) -> np.ndarray:
    """
    Crop the plate region from an image given a bbox, with a small padding
    margin so a slightly tight detection box doesn't clip a character.

    Args:
        image: file path, or an already-loaded BGR numpy array.
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        padding_fraction: fraction of the box's own width/height to pad on
            each side (default 10%, per the project's fallback plan).

    Returns:
        Cropped BGR image as a numpy array. Does NOT require a trained
        model — works on any bbox, including ground-truth boxes straight
        from a dataset, which is how Person 3 can start before your model
        is trained.
    """
    img = _load_image(image)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding_fraction)
    pad_y = int(box_h * padding_fraction)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2]


def _draw_debug_image(image_path: str, plates: list[dict], out_path: str):
    img = cv2.imread(image_path)
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        label = f"plate {p['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.imwrite(out_path, img)
    print(f"Saved annotated debug image to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--save_debug", default=None)
    ap.add_argument("--save_crop_prefix", default=None,
                     help="If set, saves each detected plate crop to <prefix>_0.jpg, <prefix>_1.jpg, ...")
    args = ap.parse_args()

    plates = detect_plates(args.image, weights_path=args.weights, conf_threshold=args.conf)

    print(f"\nDetected {len(plates)} plate(s):")
    for p in plates:
        print(f"  {p}")

    if args.save_debug:
        _draw_debug_image(args.image, plates, args.save_debug)

    if args.save_crop_prefix:
        for p in plates:
            crop = crop_plate(args.image, p["bbox"])
            out_path = f"{args.save_crop_prefix}_{p['plate_id']}.jpg"
            cv2.imwrite(out_path, crop)
            print(f"Saved crop to {out_path}")
