from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import DEFAULT_THRESHOLD
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "data/mvtec_anomaly_detection"
CATEGORY = "bottle"

OUTPUT_DIR = Path("outputs/realtime")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Production threshold loaded from centralized configuration
ANOMALY_THRESHOLD = 17.364517  # Default value; can be overridden by config

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = get_test_transform()


def preprocess_frame(frame):
    """
    Convert OpenCV BGR image into the tensor format
    expected by the feature extractor.
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        rgb
    )

    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# VISUALIZATION
# ============================================================

def create_display_image(
    frame,
    anomaly_score,
    prediction,
    inference_time_ms,
):
    """
    Create the real-time visualization.
    """

    output = frame.copy()

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if prediction == "DEFECT":
        status = "DEFECT DETECTED"
        text_color = (0, 0, 255)
    else:
        status = "NORMAL"
        text_color = (0, 255, 0)

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        output,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Anomaly Score: {anomaly_score:.2f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Latency: {inference_time_ms:.2f} ms",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def run_image_inference(
    image_path,
    patchcore,
    feature_extractor,
):
    """
    Run PatchCore inference on one image.
    """

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    start_time = time.perf_counter()

    tensor = preprocess_frame(
        frame
    )

    with torch.no_grad():

        features = (
            feature_extractor(
                tensor
            )
        )

        anomaly_score, patch_distances = (
            patchcore.predict(
                features
            )
        )

    end_time = time.perf_counter()

    inference_time_ms = (
        end_time - start_time
    ) * 1000.0

    if (
        anomaly_score
        >= ANOMALY_THRESHOLD
    ):
        prediction = "DEFECT"
    else:
        prediction = "NORMAL"

    display = create_display_image(
        frame=frame,
        anomaly_score=anomaly_score,
        prediction=prediction,
        inference_time_ms=inference_time_ms,
    )

    return {
        "image": display,
        "anomaly_score": anomaly_score,
        "prediction": prediction,
        "inference_time_ms": inference_time_ms,
        "patch_distances": patch_distances,
    }


# ============================================================
# REAL-TIME CAMERA / VIDEO PIPELINE
# ============================================================

def run_camera(
    camera_index=0,
    patchcore=None,
    feature_extractor=None,
):
    """
    Run real-time inference using an OpenCV camera.
    """

    cap = cv2.VideoCapture(
        camera_index
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open camera."
        )

    print(
        "\nReal-time camera started."
    )

    print(
        "Press 'q' to quit."
    )

    while True:

        success, frame = (
            cap.read()
        )

        if not success:

            print(
                "Failed to read camera frame."
            )

            break

        start_time = time.perf_counter()

        tensor = preprocess_frame(
            frame
        )

        with torch.no_grad():

            features = (
                feature_extractor(
                    tensor
                )

            )

            anomaly_score, _ = (
                patchcore.predict(
                    features
                )
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        latency_ms = (
            elapsed * 1000.0
        )

        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        if (
            anomaly_score
            >= ANOMALY_THRESHOLD
        ):
            prediction = "DEFECT"
        else:
            prediction = "NORMAL"

        display = create_display_image(
            frame=frame,
            anomaly_score=anomaly_score,
            prediction=prediction,
            inference_time_ms=latency_ms,
        )

        cv2.putText(
            display,
            f"FPS: {fps:.2f}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Edge Defect Detection",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# TEST IMAGE
# ============================================================

def test_sample_image(
    patchcore,
    feature_extractor,
):
    """
    Run the real-time inference path on one
    known MVTec test image.
    """

    image_path = (
        Path(DATASET_ROOT)
        / CATEGORY
        / "test"
        / "good"
        / "000.png"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    result = run_image_inference(
        image_path=image_path,
        patchcore=patchcore,
        feature_extractor=feature_extractor,
    )

    output_path = (
        OUTPUT_DIR
        / "realtime_test.png"
    )

    cv2.imwrite(
        str(output_path),
        result["image"],
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "REAL-TIME INFERENCE TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"Image          : {image_path}"
    )

    print(
        f"Prediction     : "
        f"{result['prediction']}"
    )

    print(
        f"Anomaly score  : "
        f"{result['anomaly_score']:.4f}"
    )

    print(
        f"Threshold      : "
        f"{ANOMALY_THRESHOLD:.4f}"
    )

    print(
        f"Latency        : "
        f"{result['inference_time_ms']:.2f} ms"
    )

    print(
        f"Output         : "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "EDGE REAL-TIME DEFECT DETECTION"
    )
    print("=" * 60)

    print(
        f"Device   : {DEVICE}"
    )

    print(
        f"Category : {CATEGORY}"
    )

    print(
        f"Threshold: {ANOMALY_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Build PatchCore memory bank
    # --------------------------------------------------------

    patchcore = build_memory_bank(
        category=CATEGORY,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(DEVICE)
    )

    feature_extractor.eval()

    # --------------------------------------------------------
    # Test image
    # --------------------------------------------------------

    test_sample_image(
        patchcore=patchcore,
        feature_extractor=feature_extractor,
    )

    print(
        "\nReal-time pipeline test complete."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    #
    # Uncomment when you have a camera available:
    #
    # run_camera(
    #     camera_index=0,
    #     patchcore=patchcore,
    #     feature_extractor=feature_extractor,
    # )


if __name__ == "__main__":
    main()