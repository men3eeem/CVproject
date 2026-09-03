"""
app.py — Streamlit UI for the Egyptian License Plate Recognition pipeline.

Run with:
    streamlit run app.py

Expects models/vehicle_best.pt and models/plate_best.pt to already exist
(train.py in src/vehicle_detection/ and src/plate_detection/ produces
these). Also needs PaddleOCR installed for Person 3's OCR module:
    pip install paddlepaddle paddleocr
    pip install arabic-reshaper python-bidi   # for correct Arabic text in the overlay
    apt-get install -y fonts-noto-core        # Arabic-capable font for the overlay
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC_DIR / "pipeline"))

from process_image import process_image, DEFAULT_VEHICLE_WEIGHTS, DEFAULT_PLATE_WEIGHTS  # noqa: E402
from visualize import draw_results  # noqa: E402
from process_video import process_video, summarize_unique_plates  # noqa: E402

st.set_page_config(page_title="Egyptian License Plate Recognition", layout="wide")
st.title("🚗 Egyptian License Plate Recognition")
st.caption("Vehicle YOLO + Plate YOLO (independent detectors) → PaddleOCR (Arabic) → geometric matching")

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input type", ["Image", "Video"])
    vehicle_conf = st.slider("Vehicle detection confidence", 0.05, 0.95, 0.35, 0.05)
    plate_conf = st.slider("Plate detection confidence", 0.05, 0.95, 0.10, 0.05,
                            help="Kept low by default — early testing showed the plate model needs conf=0.1 to detect reliably")
    ocr_conf_threshold = st.slider("Minimum OCR confidence to show text", 0.0, 1.0, 0.0, 0.05)

    if input_mode == "Video":
        st.divider()
        st.caption("Video is sampled, not processed frame-by-frame — running "
                   "two YOLO models + OCR on every frame would be very slow.")
        target_fps = st.slider("Frames processed per second of footage", 0.2, 3.0, 1.0, 0.2)
        max_sampled_frames = st.slider("Max frames to process (safety cap)", 5, 120, 30, 5)

    st.divider()
    vehicle_weights_exists = Path(DEFAULT_VEHICLE_WEIGHTS).exists()
    plate_weights_exists = Path(DEFAULT_PLATE_WEIGHTS).exists()
    st.write("Vehicle weights:", "✅ found" if vehicle_weights_exists else "❌ missing")
    st.write("Plate weights:", "✅ found" if plate_weights_exists else "❌ missing")
    if not (vehicle_weights_exists and plate_weights_exists):
        st.warning(f"Expected weights at:\n- {DEFAULT_VEHICLE_WEIGHTS}\n- {DEFAULT_PLATE_WEIGHTS}")

if input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save to a temp file — process_image()/read_plates() expect a file path
        # or numpy array; a temp file keeps this simple and matches how the
        # scripts are used from the command line too.
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        original_image = cv2.imread(tmp_path)
        if original_image is None:
            st.error("Could not read the uploaded image. Try a different file.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(Image.open(uploaded_file), use_container_width=True)

            with st.spinner("Running vehicle detection, plate detection, and OCR..."):
                try:
                    results = process_image(
                        tmp_path,
                        vehicle_conf=vehicle_conf,
                        plate_conf=plate_conf,
                        ocr_conf_threshold=ocr_conf_threshold,
                    )
                except FileNotFoundError as e:
                    st.error(f"Model weights not found: {e}")
                    results = None
                except ImportError as e:
                    st.error(
                        f"Missing dependency: {e}. This pipeline needs ultralytics, "
                        "paddleocr, and paddlepaddle installed."
                    )
                    results = None

            if results is not None:
                with col2:
                    st.subheader("Detected")
                    if results:
                        annotated = draw_results(original_image, results)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)
                    else:
                        st.image(Image.open(uploaded_file), use_container_width=True)
                        st.info("No plates detected. Try lowering the plate detection "
                                "confidence in the sidebar.")

                if results:
                    st.subheader("Results")
                    for i, r in enumerate(results):
                        with st.container(border=True):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Vehicle type", r["vehicle_type"])
                            c2.metric("Plate text", r["plate_text"] or "—")
                            c3.metric("OCR confidence", f"{r['confidence']:.2f}" if r["confidence"] else "—")

                            with st.expander("Details"):
                                st.json({
                                    "vehicle_confidence": r["vehicle_confidence"],
                                    "plate_detection_confidence": r["plate_detection_confidence"],
                                    "letters": r["letters"],
                                    "numbers": r["numbers"],
                                    "vehicle_bbox": r["vehicle_bbox"],
                                    "plate_bbox": r["plate_bbox"],
                                })

        Path(tmp_path).unlink(missing_ok=True)
    else:
        st.info("Upload an image with a vehicle and Egyptian plate to get started.")

else:  # Video mode
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_in:
            tmp_in.write(uploaded_video.getvalue())
            tmp_in_path = tmp_in.name
        tmp_out_path = str(Path(tempfile.gettempdir()) / f"annotated_{Path(tmp_in_path).stem}.mp4")

        st.video(uploaded_video)

        if st.button("Run pipeline on this video"):
            progress_bar = st.progress(0.0, text="Starting...")
            all_detections = []

            try:
                for update in process_video(
                    tmp_in_path,
                    tmp_out_path,
                    target_fps=target_fps,
                    max_sampled_frames=max_sampled_frames,
                    vehicle_conf=vehicle_conf,
                    plate_conf=plate_conf,
                ):
                    all_detections.append(update["detections"])
                    progress_bar.progress(
                        update["fraction"],
                        text=f"Processed frame {update['sampled_count']} "
                             f"(video frame #{update['frame_index']})..."
                    )
                progress_bar.progress(1.0, text="Done.")
            except FileNotFoundError as e:
                st.error(f"Model weights not found: {e}")
                all_detections = []
            except ImportError as e:
                st.error(f"Missing dependency: {e}")
                all_detections = []
            except RuntimeError as e:
                st.error(f"Video writer failed: {e}")
                all_detections = []

            if all_detections and Path(tmp_out_path).exists():
                st.subheader("Annotated video")
                st.video(tmp_out_path)
                with open(tmp_out_path, "rb") as f:
                    st.download_button("Download annotated video", f, file_name="annotated_output.mp4")

                summary = summarize_unique_plates(all_detections)
                st.subheader(f"Unique plates detected ({len(summary)})")
                if summary:
                    for r in summary:
                        with st.container(border=True):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Vehicle type", r["vehicle_type"])
                            c2.metric("Plate text", r["plate_text"] or "—")
                            c3.metric("Best OCR confidence", f"{r['confidence']:.2f}")
                else:
                    st.info("No plates detected across the sampled frames. Try lowering "
                            "the plate detection confidence, or increasing frames processed per second.")

        Path(tmp_in_path).unlink(missing_ok=True)
    else:
        st.info("Upload a video with vehicles and Egyptian plates to get started.")