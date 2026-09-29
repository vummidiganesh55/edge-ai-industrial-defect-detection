import time
import statistics

import numpy as np
import openvino as ov
from PIL import Image

from src.preprocessing import get_test_transform


MODEL_PATH = (
    "models/openvino/"
    "resnet18_feature_extractor.xml"
)

IMAGE_PATH = (
    "data/mvtec_anomaly_detection/"
    "bottle/test/broken_large/000.png"
)

NUM_WARMUP = 5
NUM_RUNS = 50


def load_image():

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    transform = get_test_transform()

    tensor = transform(image).unsqueeze(0)

    return (
        tensor.numpy()
        .astype(np.float32)
    )


def benchmark():

    print("=" * 60)
    print("OPENVINO CPU BENCHMARK")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Image : {IMAGE_PATH}")
    print(f"Warmup runs : {NUM_WARMUP}")
    print(f"Benchmark runs : {NUM_RUNS}")

    core = ov.Core()

    model = core.read_model(
        MODEL_PATH
    )

    compiled_model = core.compile_model(
        model,
        "CPU"
    )

    input_layer = (
        compiled_model.input(0)
    )

    infer_request = (
        compiled_model.create_infer_request()
    )

    image = load_image()

    print("\nWARM-UP")
    print("-" * 60)

    for i in range(NUM_WARMUP):

        infer_request.infer({
            input_layer: image
        })

        print(
            f"Warm-up {i + 1}/{NUM_WARMUP}"
        )

    print("\nBENCHMARK")
    print("-" * 60)

    latencies = []

    for i in range(NUM_RUNS):

        start = time.perf_counter()

        outputs = infer_request.infer({
            input_layer: image
        })

        end = time.perf_counter()

        latency = (
            end - start
        ) * 1000

        latencies.append(
            latency
        )

        print(
            f"Run {i + 1:02d}/{NUM_RUNS} | "
            f"{latency:.2f} ms"
        )

    average = statistics.mean(
        latencies
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

    minimum = min(latencies)
    maximum = max(latencies)

    fps = (
        1000.0 / average
    )

    print("\n")
    print("=" * 60)
    print("OPENVINO CPU RESULTS")
    print("=" * 60)

    print(
        f"Average latency : "
        f"{average:.2f} ms"
    )

    print(
        f"P50 latency     : "
        f"{p50:.2f} ms"
    )

    print(
        f"P95 latency     : "
        f"{p95:.2f} ms"
    )

    print(
        f"Min latency     : "
        f"{minimum:.2f} ms"
    )

    print(
        f"Max latency     : "
        f"{maximum:.2f} ms"
    )

    print(
        f"FPS             : "
        f"{fps:.2f}"
    )

    print("=" * 60)

    return outputs


if __name__ == "__main__":
    benchmark()