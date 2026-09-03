"""
src/pipeline/process_image.py — Final integration.

Combines:
    Person 1: detect_vehicles()          — src/vehicle_detection/infer.py
    Person 2: detect_plates()            — src/plate_detection/infer.py
    Person 3: read_plates()              — src/ocr/infer.py (PaddleOCR, Arabic)
    Person 4: match_plate_to_vehicle()   — src/pipeline/match.py

Note: Person 3's read_plates() does its own cropping/rectification/header
removal internally (it takes the full image + plate bboxes directly), so
this pipeline does NOT call Person 2's crop_plate() or the earlier
preprocess_plate() placeholder in the main path — those remain available
for debugging/visualization but are superseded here by Person 3's more
Egyptian-plate-specific preprocessing.
"""

import sys
from pathlib import Path
from typing import Union
import importlib.util

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent

# vehicle_detection/infer.py and plate_detection/infer.py are both named
# infer.py, which would collide on `import infer` via sys.path. Load each
# explicitly by file path instead of relying on import order.


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vehicle_mod = _load_module("vehicle_infer", _SRC_DIR / "vehicle_detection" / "infer.py")
_plate_mod = _load_module("plate_infer", _SRC_DIR / "plate_detection" / "infer.py")
_ocr_mod = _load_module("ocr_infer", _SRC_DIR / "ocr" / "infer.py")
_match_mod = _load_module("match_mod", _SRC_DIR / "pipeline" / "match.py")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vehicle_mod = _load_module("vehicle_infer", _SRC_DIR / "vehicle_detection" / "infer.py")
_plate_mod = _load_module("plate_infer", _SRC_DIR / "plate_detection" / "infer.py")
_ocr_mod = _load_module("ocr_infer", _SRC_DIR / "ocr" / "infer.py")
_match_mod = _load_module("match_mod", _SRC_DIR / "pipeline" / "match.py")

DEFAULT_VEHICLE_WEIGHTS = _SRC_DIR.parent / "models" / "vehicle_best.pt"
DEFAULT_PLATE_WEIGHTS = _SRC_DIR.parent / "models" / "plate_best.pt"


def process_image(
    image_path: Union[str, Path, np.ndarray],
    vehicle_weights: Union[str, Path] = DEFAULT_VEHICLE_WEIGHTS,
    plate_weights: Union[str, Path] = DEFAULT_PLATE_WEIGHTS,
    vehicle_conf: float = 0.35,
    plate_conf: float = 0.10,  # lower default — plate model showed low-confidence detections in testing
    ocr_conf_threshold: float = 0.0,
    ocr_device: str = "auto",
) -> list[dict]:
    """
    Run the full pipeline on one image.

    Returns a list, one entry per detected plate:
        [
            {
                "vehicle_type": "car" | "truck" | "bus" | "motorcycle" | "unknown",
                "plate_text": "ب3٤5 1234",
                "confidence": 0.62,                # OCR confidence — how much to trust plate_text
                "vehicle_bbox": [x1,y1,x2,y2] | None,
                "plate_bbox": [x1,y1,x2,y2],
                "vehicle_confidence": 0.94 | None,
                "plate_detection_confidence": 0.41,
                "letters": "ب", "numbers": "3٤5",   # raw split, useful for debugging
            },
            ...
        ]
    Empty list if no plates were detected at all — a valid, if unlucky, outcome.
    """
    image = image_path if isinstance(image_path, np.ndarray) else str(image_path)

    vehicles = _vehicle_mod.detect_vehicles(
        image, weights_path=vehicle_weights, conf_threshold=vehicle_conf
    )
    plate_detections = _plate_mod.detect_plates(
        image, weights_path=plate_weights, conf_threshold=plate_conf
    )

    if not plate_detections:
        return []

    ocr_results = _ocr_mod.read_plates(
        image,
        plate_detections,
        conf_threshold=ocr_conf_threshold,
        device=ocr_device,
    )

    matches = _match_mod.match_plate_to_vehicle(vehicles, ocr_results)

    final_results = []
    for m in matches:
        v = m["vehicle"]
        p = m["plate"]
        final_results.append({
            "vehicle_type": v["vehicle_type"] if v else "unknown",
            "plate_text": p["plate_text"],
            "confidence": p["confidence"],
            "vehicle_bbox": v["bbox"] if v else None,
            "plate_bbox": p["bbox"],
            "vehicle_confidence": v["confidence"] if v else None,
            "plate_detection_confidence": None,  # filled below if available
            "letters": p.get("letters", ""),
            "numbers": p.get("numbers", ""),
        })

    # Attach the original YOLO plate-detection confidence (distinct from OCR
    # confidence) by matching on plate_id, for anyone who wants both signals.
    det_conf_by_id = {d["plate_id"]: d["confidence"] for d in plate_detections}
    for r, m in zip(final_results, matches):
        pid = m["plate"].get("plate_id")
        r["plate_detection_confidence"] = det_conf_by_id.get(pid)

    return final_results


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--vehicle_weights", default=str(DEFAULT_VEHICLE_WEIGHTS))
    ap.add_argument("--plate_weights", default=str(DEFAULT_PLATE_WEIGHTS))
    args = ap.parse_args()

    results = process_image(args.image, vehicle_weights=args.vehicle_weights,
                             plate_weights=args.plate_weights)
    print(json.dumps(results, indent=2, ensure_ascii=False))