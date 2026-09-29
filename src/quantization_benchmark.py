import os
import time
import statistics

import numpy as np
import onnxruntime as ort
from PIL import Image


from src.preprocessing import get_test_transform


FP32_MODEL = (
    "models/resnet18_feature_extractor.onnx"
)

INT8_MODEL = (
    "models/resnet18_feature_extractor_int8.onnx"
)

IMAGE_PATH = (
    "data/mvtec_anomaly_detection/"
    "bottle/test/broken_large/000.png"
)

NUM_WARMUP = 5
NUM_RUNS = 50

COSINE_THRESHOLD = 0.999


def load_image():

    image = (
        Image
        .open(IMAGE_PATH)
        .convert("RGB")
    )

    transform = get_test_transform()

    tensor = transform(
        image
    ).unsqueeze(0)

    return tensor.numpy().astype(
        np.float32
    )


def benchmark(
    session,
    input_name,
    image,
):

    for _ in range(NUM_WARMUP):

        session.run(
            None,
            {
                input_name: image
            }
        )

    latencies = []

    outputs = None

    for _ in range(NUM_RUNS):

        start = (
            time.perf_counter()
        )

        outputs = session.run(
            None,
            {
                input_name: image
            }
        )

        end = (
            time.perf_counter()
        )

        latencies.append(
            (end - start) * 1000
        )

    average = (
        statistics.mean(
            latencies
        )
    )

    p50 = float(
        np.percentile(
            latencies,
            50
        )
    )

    p95 = float(
        np.percentile(
            latencies,
            95
        )
    )

    minimum = min(
        latencies
    )

    maximum = max(
        latencies
    )

    fps = (
        1000.0
        / average
    )

    return {
        "average": average,
        "p50": p50,
        "p95": p95,
        "min": minimum,
        "max": maximum,
        "fps": fps,
        "outputs": outputs,
    }


def calculate_consistency(
    fp32_output,
    int8_output,
):

    fp32_output = np.asarray(
        fp32_output,
        dtype=np.float32
    )

    int8_output = np.asarray(
        int8_output,
        dtype=np.float32
    )

    difference = np.abs(
        fp32_output
        - int8_output
    )

    max_abs_error = float(
        np.max(
            difference
        )
    )

    mean_abs_error = float(
        np.mean(
            difference
        )
    )

    fp32_flat = (
        fp32_output
        .reshape(-1)
    )

    int8_flat = (
        int8_output
        .reshape(-1)
    )

    fp32_norm = np.linalg.norm(
        fp32_flat
    )

    int8_norm = np.linalg.norm(
        int8_flat
    )

    if (
        fp32_norm == 0
        or int8_norm == 0
    ):

        cosine_similarity = 0.0

    else:

        cosine_similarity = float(
            np.dot(
                fp32_flat,
                int8_flat
            )
            / (
                fp32_norm
                * int8_norm
            )
        )

    return {
        "max_abs_error":
            max_abs_error,

        "mean_abs_error":
            mean_abs_error,

        "cosine_similarity":
            cosine_similarity,
    }


def main():

    print("=" * 60)
    print("FP32 vs INT8 ONNX BENCHMARK")
    print("=" * 60)

    if not os.path.exists(
        FP32_MODEL
    ):

        raise FileNotFoundError(
            f"FP32 model not found: "
            f"{FP32_MODEL}"
        )

    if not os.path.exists(
        INT8_MODEL
    ):

        raise FileNotFoundError(
            f"INT8 model not found: "
            f"{INT8_MODEL}"
        )

    image = load_image()

    print(
        "\nLoading ONNX Runtime sessions..."
    )

    fp32_session = (
        ort.InferenceSession(
            FP32_MODEL,
            providers=[
                "CPUExecutionProvider"
            ]
        )
    )

    int8_session = (
        ort.InferenceSession(
            INT8_MODEL,
            providers=[
                "CPUExecutionProvider"
            ]
        )
    )

    fp32_input = (
        fp32_session
        .get_inputs()[0]
        .name
    )

    int8_input = (
        int8_session
        .get_inputs()[0]
        .name
    )

    print(
        "\nBenchmarking FP32..."
    )

    fp32 = benchmark(
        fp32_session,
        fp32_input,
        image
    )

    print(
        "Benchmarking INT8..."
    )

    int8 = benchmark(
        int8_session,
        int8_input,
        image
    )

    print(
        "\n"
        + "=" * 60
    )

    print("RESULTS")

    print("=" * 60)

    print(
        f"FP32 average : "
        f"{fp32['average']:.2f} ms"
    )

    print(
        f"INT8 average : "
        f"{int8['average']:.2f} ms"
    )

    print()

    print(
        f"FP32 P50     : "
        f"{fp32['p50']:.2f} ms"
    )

    print(
        f"INT8 P50     : "
        f"{int8['p50']:.2f} ms"
    )

    print()

    print(
        f"FP32 P95     : "
        f"{fp32['p95']:.2f} ms"
    )

    print(
        f"INT8 P95     : "
        f"{int8['p95']:.2f} ms"
    )

    print()

    print(
        f"FP32 min     : "
        f"{fp32['min']:.2f} ms"
    )

    print(
        f"INT8 min     : "
        f"{int8['min']:.2f} ms"
    )

    print()

    print(
        f"FP32 max     : "
        f"{fp32['max']:.2f} ms"
    )

    print(
        f"INT8 max     : "
        f"{int8['max']:.2f} ms"
    )

    print()

    print(
        f"FP32 FPS     : "
        f"{fp32['fps']:.2f}"
    )

    print(
        f"INT8 FPS     : "
        f"{int8['fps']:.2f}"
    )

    speedup = (
        fp32["average"]
        / int8["average"]
    )

    print()

    print(
        f"INT8 speedup : "
        f"{speedup:.2f}x"
    )

    # --------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------

    fp32_size = (
        os.path.getsize(
            FP32_MODEL
        )
        / (1024 * 1024)
    )

    int8_size = (
        os.path.getsize(
            INT8_MODEL
        )
        / (1024 * 1024)
    )

    size_reduction = (
        1
        - (
            int8_size
            / fp32_size
        )
    ) * 100

    size_ratio = (
        fp32_size
        / int8_size
    )

    print(
        "\n"
        + "=" * 60
    )

    print("MODEL SIZE")

    print("=" * 60)

    print(
        f"FP32 ONNX size : "
        f"{fp32_size:.2f} MB"
    )

    print(
        f"INT8 ONNX size : "
        f"{int8_size:.2f} MB"
    )

    print(
        f"Size reduction : "
        f"{size_reduction:.2f}%"
    )

    print(
        f"Compression    : "
        f"{size_ratio:.2f}x"
    )

    # --------------------------------------------------
    # FEATURE CONSISTENCY
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FP32 vs INT8 FEATURE CONSISTENCY"
    )

    print("=" * 60)

    consistency_passed = True

    if len(fp32["outputs"]) != len(
        int8["outputs"]
    ):

        raise RuntimeError(
            "FP32 and INT8 output "
            "counts do not match."
        )

    for i, (
        fp32_output,
        int8_output
    ) in enumerate(
        zip(
            fp32["outputs"],
            int8["outputs"]
        )
    ):

        fp32_array = np.asarray(
            fp32_output
        )

        int8_array = np.asarray(
            int8_output
        )

        print()

        print(
            f"Output {i}"
        )

        print(
            f"  FP32 shape : "
            f"{fp32_array.shape}"
        )

        print(
            f"  INT8 shape: "
            f"{int8_array.shape}"
        )

        if (
            fp32_array.shape
            != int8_array.shape
        ):

            print(
                "  Shape consistency: FAILED"
            )

            consistency_passed = False

            continue

        metrics = calculate_consistency(
            fp32_array,
            int8_array
        )

        print(
            f"  Max absolute error : "
            f"{metrics['max_abs_error']:.8f}"
        )

        print(
            f"  Mean absolute error: "
            f"{metrics['mean_abs_error']:.8f}"
        )

        print(
            f"  Cosine similarity  : "
            f"{metrics['cosine_similarity']:.8f}"
        )

        if (
            metrics["cosine_similarity"]
            < COSINE_THRESHOLD
        ):

            consistency_passed = False

    print()

    if consistency_passed:

        print(
            "Feature consistency: PASSED"
        )

    else:

        print(
            "Feature consistency: "
            "REVIEW REQUIRED"
        )

    # --------------------------------------------------
    # DEPLOYMENT ASSESSMENT
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print("INT8 DEPLOYMENT ASSESSMENT")

    print("=" * 60)

    if (
        consistency_passed
        and speedup >= 1.0
    ):

        print(
            "INT8 deployment assessment: "
            "SUITABLE FOR FURTHER TESTING"
        )

    elif (
        consistency_passed
        and speedup < 1.0
    ):

        print(
            "INT8 deployment assessment: "
            "NOT SELECTED FOR PERFORMANCE"
        )

        print(
            "Reason: INT8 is slower than "
            "FP32 in this runtime configuration."
        )

    else:

        print(
            "INT8 deployment assessment: "
            "REQUIRES REVIEW"
        )

    print(
        "\n"
        + "=" * 60
    )

    print("BENCHMARK COMPLETE")

    print("=" * 60)


if __name__ == "__main__":

    main()
