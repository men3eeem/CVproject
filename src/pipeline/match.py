"""
src/pipeline/match.py — Person 4's role, geometric matching.

Simplest reliable method per Section 9 of the plan: bbox containment /
centroid check, not full IoU (a plate is expected to sit fully inside its
vehicle's box, not partially overlap it).
"""


def match_plate_to_vehicle(vehicles: list[dict], plates: list[dict]) -> list[dict]:
    """
    Pair each detected plate with the vehicle whose box contains the
    plate's centroid.

    Args:
        vehicles: list of {"vehicle_id", "vehicle_type", "confidence", "bbox"}
        plates: list of {"plate_id", "confidence", "bbox", ...} (may already
            have "plate_text"/"ocr_confidence" attached by the caller)

    Returns:
        One entry per plate:
        [{"vehicle": vehicle_dict_or_None, "plate": plate_dict}, ...]
        "vehicle" is None if no vehicle box contains the plate's centroid —
        callers should treat that as "unknown vehicle type", not an error.
    """
    matches = []
    for plate in plates:
        px1, py1, px2, py2 = plate["bbox"]
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2

        candidates = []
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                area = (vx2 - vx1) * (vy2 - vy1)
                candidates.append((area, v))

        if candidates:
            # Tie-break on smallest containing box (most specific match),
            # per Section 9 — handles rare overlapping-vehicle cases.
            candidates.sort(key=lambda c: c[0])
            best_vehicle = candidates[0][1]
        else:
            best_vehicle = None

        matches.append({"vehicle": best_vehicle, "plate": plate})

    return matches
