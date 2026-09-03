"""
infer.py — Person 1 (Vehicle Detection)

Exposes detect_vehicles(image), the exact function the rest of the team
(Person 4's matching code, Person 5's pipeline) calls. This is the contract —
don't change the return shape without telling everyone.

Return shape (matches Section 6 of the project plan):
    [
        {
            "vehicle_id": 0,
            "vehicle_type": "car",       # one of: car, truck, bus, motorcycle
            "confidence": 0.94,
            "bbox": [x1, y1, x2, y2]     # ints, pixel coords in the ORIGINAL image
        },
        ...
    ]

Usage as a library:
    from infer import detect_vehicles
    results = detect_vehicles("some_image.jpg")

Usage as a CLI (quick manual test + saves an annotated image):
    python infer.py --image test.jpg --weights ../../models/vehicle_best.pt
"""

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "vehicle_best.pt"
CONFIDENCE_THRESHOLD = 0.35  # lower this if the detector is missing vehicles (see Section 9 fallback)

_model = None  # lazy-loaded singleton so we don't reload weights on every call


def _get_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS) -> YOLO:
    global _model
    if _model is None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Vehicle weights not found at {weights_path}. "
                "Run train.py first, or pass weights_path explicitly."
            )
        _model = YOLO(str(weights_path))
    return _model


def detect_vehicles(
    image: Union[str, Path, np.ndarray],
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Run the fine-tuned YOLOv8n vehicle detector on a single image.

    Args:
        image: file path, or an already-loaded BGR numpy array (cv2 style).
        weights_path: override the default trained weights location.
        conf_threshold: minimum confidence to keep a detection.

    Returns:
        List of dicts, one per detected vehicle, per the contract above.
        Empty list if nothing is detected above threshold — callers should
        handle that case (it's a valid, if unlucky, outcome, not an error).
    """
    model = _get_model(weights_path)
    results = model.predict(source=image, conf=conf_threshold, verbose=False)[0]

    vehicles = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        class_name = model.names[cls_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        vehicles.append({
            "vehicle_id": i,
            "vehicle_type": class_name,
            "confidence": round(confidence, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return vehicles


def _draw_debug_image(image_path: str, vehicles: list[dict], out_path: str):
    img = cv2.imread(image_path)
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f"{v['vehicle_type']} {v['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.imwrite(out_path, img)
    print(f"Saved annotated debug image to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--save_debug", default=None, help="Optional path to save an annotated image")
    args = ap.parse_args()

    vehicles = detect_vehicles(args.image, weights_path=args.weights, conf_threshold=args.conf)

    print(f"\nDetected {len(vehicles)} vehicle(s):")
    for v in vehicles:
        print(f"  {v}")

    if args.save_debug:
        _draw_debug_image(args.image, vehicles, args.save_debug)
