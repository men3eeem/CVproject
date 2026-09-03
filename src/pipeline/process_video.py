

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from process_image import (
    process_image,
    DEFAULT_VEHICLE_WEIGHTS,
    DEFAULT_PLATE_WEIGHTS,
)

from visualize import draw_results


# -------------------------------------------------------------------------
# Video codec candidates
# -------------------------------------------------------------------------
# Codec availability depends on the OpenCV installation and operating
# system. Try H.264 first, then MP4V, then XVID.

_FOURCC_CANDIDATES = [
    "avc1",
    "mp4v",
    "XVID",
]


# -------------------------------------------------------------------------
# Video writer
# -------------------------------------------------------------------------

def _open_writer(
    out_path: str,
    fps: float,
    width: int,
    height: int
) -> cv2.VideoWriter:
    """
    Open a VideoWriter using the first available codec.
    """

    for fourcc_str in _FOURCC_CANDIDATES:

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            fps,
            (width, height)
        )

        if writer.isOpened():
            print(
                f"Video writer opened successfully using codec: "
                f"{fourcc_str}"
            )
            return writer

        writer.release()

        print(
            f"Codec {fourcc_str} failed, trying next codec..."
        )

    raise RuntimeError(
        f"Could not open a video writer with any of "
        f"{_FOURCC_CANDIDATES}. "
        "Your OpenCV build may be missing video codec support. "
        "Try installing regular opencv-python instead of "
        "opencv-python-headless, or install FFmpeg."
    )


# -------------------------------------------------------------------------
# Process video
# -------------------------------------------------------------------------

def process_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    target_fps: float = 1.0,
    max_sampled_frames: int = 60,

    # IMPORTANT:
    # Use the same default model paths as process_image.py.
    vehicle_weights: Union[str, Path] = DEFAULT_VEHICLE_WEIGHTS,
    plate_weights: Union[str, Path] = DEFAULT_PLATE_WEIGHTS,

    vehicle_conf: float = 0.35,
    plate_conf: float = 0.10,
):
    """
    Process a video using the existing image pipeline.

    The video is NOT processed frame-by-frame.

    Instead:
        Video
          ↓
        Sample frames
          ↓
        process_image()
          ↓
        Vehicle YOLO
          ↓
        Plate YOLO
          ↓
        PaddleOCR
          ↓
        Annotated frame
          ↓
        Output video

    Args:
        video_path:
            Input video path.

        output_path:
            Output annotated video path.

        target_fps:
            Number of frames per second of VIDEO TIME that will actually
            be processed.

            Example:
                target_fps=1.0
                → approximately one frame every second.

                target_fps=2.0
                → approximately two frames every second.

        max_sampled_frames:
            Maximum number of frames that will actually run through the
            YOLO + OCR pipeline.

            This prevents very long videos from taking too long.

        vehicle_weights:
            Path to vehicle YOLO weights.

            Defaults to:
                models/vehicle_best.pt

        plate_weights:
            Path to plate YOLO weights.

            Defaults to:
                models/plate_best.pt

        vehicle_conf:
            Vehicle YOLO confidence threshold.

        plate_conf:
            Plate YOLO confidence threshold.

    Yields:
        Dictionary after every processed frame:

        {
            "fraction": 0.0-1.0,
            "frame_index": int,
            "sampled_count": int,
            "detections": list
        }

    """

    # ---------------------------------------------------------------------
    # Validate target FPS
    # ---------------------------------------------------------------------

    if target_fps <= 0:
        raise ValueError(
            "target_fps must be greater than 0."
        )

    # ---------------------------------------------------------------------
    # Convert paths to strings
    # ---------------------------------------------------------------------

    video_path = str(video_path)
    output_path = str(output_path)

    vehicle_weights = Path(vehicle_weights)
    plate_weights = Path(plate_weights)

    # ---------------------------------------------------------------------
    # Check model files
    # ---------------------------------------------------------------------

    if not vehicle_weights.exists():
        raise FileNotFoundError(
            f"Vehicle model weights not found:\n"
            f"{vehicle_weights}"
        )

    if not plate_weights.exists():
        raise FileNotFoundError(
            f"Plate model weights not found:\n"
            f"{plate_weights}"
        )

    print(
        f"Vehicle weights: {vehicle_weights}"
    )

    print(
        f"Plate weights: {plate_weights}"
    )

    # ---------------------------------------------------------------------
    # Open input video
    # ---------------------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video:\n{video_path}"
        )

    # ---------------------------------------------------------------------
    # Read video properties
    # ---------------------------------------------------------------------

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0:
        source_fps = 25.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # ---------------------------------------------------------------------
    # Calculate sampling interval
    # ---------------------------------------------------------------------

    # Example:
    #
    # source FPS = 30
    # target FPS = 1
    #
    # frame_interval = 30
    #
    # So:
    #
    # frame 0  -> process
    # frame 30 -> process
    # frame 60 -> process
    # ...

    frame_interval = max(
        1,
        int(round(source_fps / target_fps))
    )

    estimated_sampled = min(
        max_sampled_frames,
        (total_frames // frame_interval) + 1
    )

    print(
        f"Source FPS: {source_fps:.2f}"
    )

    print(
        f"Target processing FPS: {target_fps:.2f}"
    )

    print(
        f"Frame interval: {frame_interval}"
    )

    print(
        f"Estimated sampled frames: {estimated_sampled}"
    )

    # ---------------------------------------------------------------------
    # Open output video writer
    # ---------------------------------------------------------------------

    writer = _open_writer(
        output_path,
        source_fps,
        width,
        height
    )

    # ---------------------------------------------------------------------
    # Processing state
    # ---------------------------------------------------------------------

    frame_index = 0
    sampled_count = 0

    last_detections: list[dict] = []

    last_annotated_frame = None

    # ---------------------------------------------------------------------
    # Process video
    # ---------------------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # -------------------------------------------------------------
            # Decide whether this frame should be processed
            # -------------------------------------------------------------

            should_process = (
                frame_index % frame_interval == 0
                and sampled_count < max_sampled_frames
            )

            # -------------------------------------------------------------
            # Process sampled frame
            # -------------------------------------------------------------

            if should_process:

                print(
                    f"Processing video frame "
                    f"{frame_index} "
                    f"({sampled_count + 1}/{estimated_sampled})"
                )

                # IMPORTANT:
                #
                # frame is a numpy array.
                #
                # process_image() was updated to support numpy arrays,
                # so we can pass the frame directly.

                detections = process_image(
                    frame,

                    vehicle_weights=vehicle_weights,
                    plate_weights=plate_weights,

                    vehicle_conf=vehicle_conf,
                    plate_conf=plate_conf,
                )

                # ---------------------------------------------------------
                # Draw detections
                # ---------------------------------------------------------

                if detections:

                    annotated = draw_results(
                        frame,
                        detections
                    )

                else:

                    annotated = frame.copy()

                # ---------------------------------------------------------
                # Save latest result
                # ---------------------------------------------------------

                last_detections = detections

                last_annotated_frame = annotated.copy()

                sampled_count += 1

                # ---------------------------------------------------------
                # Write annotated frame
                # ---------------------------------------------------------

                writer.write(
                    annotated
                )

                # ---------------------------------------------------------
                # Progress
                # ---------------------------------------------------------

                yield {
                    "fraction": min(
                        1.0,
                        sampled_count / max(
                            1,
                            estimated_sampled
                        )
                    ),

                    "frame_index": frame_index,

                    "sampled_count": sampled_count,

                    "detections": detections,
                }

            # -------------------------------------------------------------
            # Frames that are NOT processed
            # -------------------------------------------------------------

            else:

                # Instead of running YOLO + OCR again,
                # reuse the last annotated frame.
                #
                # This keeps the video visually stable and prevents
                # flickering.

                if last_annotated_frame is not None:

                    writer.write(
                        last_annotated_frame
                    )

                else:

                    # No processed frame yet.
                    writer.write(
                        frame
                    )

            # -------------------------------------------------------------
            # Next frame
            # -------------------------------------------------------------

            frame_index += 1

    finally:

        # Always release resources, even if an error happens.

        cap.release()

        writer.release()

        print(
            f"Video processing finished."
        )

        print(
            f"Total frames in input: {frame_index}"
        )

        print(
            f"Processed frames: {sampled_count}"
        )

        print(
            f"Output video: {output_path}"
        )


# -------------------------------------------------------------------------
# Summarize unique plates
# -------------------------------------------------------------------------

def summarize_unique_plates(
    all_detections: list[list[dict]]
) -> list[dict]:
    """
    Collapse detections from multiple video frames into one result
    per unique plate.

    If the same plate appears in many frames:

        Frame 1 → ABC 123
        Frame 2 → ABC 123
        Frame 3 → ABC 123

    only one result is returned.

    The detection with the highest OCR confidence is kept.
    """

    best_by_text: dict[str, dict] = {}

    for frame_detections in all_detections:

        for det in frame_detections:

            text = det.get(
                "plate_text",
                ""
            )

            if not text:
                continue

            confidence = float(
                det.get(
                    "confidence",
                    0.0
                )
            )

            if (
                text not in best_by_text
                or confidence
                > best_by_text[text]["confidence"]
            ):

                best_by_text[text] = det

    # Highest confidence first.

    return sorted(
        best_by_text.values(),
        key=lambda d: -float(
            d.get("confidence", 0.0)
        )
    )

