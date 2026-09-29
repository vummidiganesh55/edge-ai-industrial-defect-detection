import time
import statistics
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

from src.feature_extractor import ResNet18FeatureExtractor
from src.preprocessing import get_test_transform


ONNX_PATH = "models/resnet18_feature_extractor.onnx"

IMAGE_PATH = (
    "data/mvtec_anomaly_detection/"
    "bottle/test/broken_large/000.png"
)

NUM_WARMUP = 5
NUM_RUNS = 50

DEVICE = torch.device("cpu")


def load_image():

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    transform = get_test_transform()

    tensor = transform(image).unsqueeze(0)

    return tensor


def benchmark_pytorch(image_tensor):

    model = (
        ResNet18FeatureExtractor()
        .to(DEVICE)
    )

    model.eval()

    with torch.no_grad():

        for _ in range(NUM_WARMUP):
            _ = model(image_tensor)

    latencies = []

    for _ in range(NUM_RUNS):

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(image_tensor)

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    return latencies, outputs


def benchmark_onnx(image_tensor):

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    image_numpy = (
        image_tensor
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    for _ in range(NUM_WARMUP):

        session.run(
            None,
            {input_name: image_numpy}
        )

    latencies = []

    outputs = None

    for _ in range(NUM_RUNS):

        start = time.perf_counter()

        outputs = session.run(
            None,
            {input_name: image_numpy}
        )

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    return latencies, outputs


def summarize(name, latencies):

    average = statistics.mean(latencies)

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
        if average > 0
        else 0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Average latency : {average:.2f} ms"
    )

    print(
        f"P50 latency     : {p50:.2f} ms"
    )

    print(
        f"P95 latency     : {p95:.2f} ms"
    )

    print(
        f"Min latency     : {minimum:.2f} ms"
    )

    print(
        f"Max latency     : {maximum:.2f} ms"
    )

    print(
        f"FPS             : {fps:.2f}"
    )

    return {
        "average": average,
        "p50": p50,
        "p95": p95,
        "fps": fps,
    }


def compare_outputs(
    pytorch_outputs,
    onnx_outputs
):

    pytorch_layer2 = (
        pytorch_outputs["layer2"]
        .detach()
        .cpu()
        .numpy()
    )

    pytorch_layer3 = (
        pytorch_outputs["layer3"]
        .detach()
        .cpu()
        .numpy()
    )

    onnx_layer2 = onnx_outputs[0]

    onnx_layer3 = onnx_outputs[1]

    layer2_difference = np.max(
        np.abs(
            pytorch_layer2 -
            onnx_layer2
        )
    )

    layer3_difference = np.max(
        np.abs(
            pytorch_layer3 -
            onnx_layer3
        )
    )

    print("\n" + "=" * 60)
    print("FEATURE CONSISTENCY")
    print("=" * 60)

    print(
        f"Layer2 max absolute difference : "
        f"{layer2_difference:.8f}"
    )

    print(
        f"Layer3 max absolute difference : "
        f"{layer3_difference:.8f}"
    )


def main():

    print("=" * 60)
    print("ONNX RUNTIME BENCHMARK")
    print("=" * 60)

    print(
        f"ONNX model : {ONNX_PATH}"
    )

    print(
        f"Test image : {IMAGE_PATH}"
    )

    print(
        f"Warmup runs: {NUM_WARMUP}"
    )

    print(
        f"Benchmark runs: {NUM_RUNS}"
    )

    image_tensor = load_image()

    print("\nRunning PyTorch benchmark...")

    pytorch_latencies, pytorch_outputs = (
        benchmark_pytorch(
            image_tensor
        )
    )

    pytorch_results = summarize(
        "PYTORCH CPU",
        pytorch_latencies
    )

    print("\nRunning ONNX Runtime benchmark...")

    onnx_latencies, onnx_outputs = (
        benchmark_onnx(
            image_tensor
        )
    )

    onnx_results = summarize(
        "ONNX RUNTIME CPU",
        onnx_latencies
    )

    compare_outputs(
        pytorch_outputs,
        onnx_outputs
    )

    speedup = (
        pytorch_results["average"] /
        onnx_results["average"]
    )

    print("\n" + "=" * 60)
    print("PYTORCH vs ONNX RUNTIME")
    print("=" * 60)

    print(
        f"PyTorch average : "
        f"{pytorch_results['average']:.2f} ms"
    )

    print(
        f"ONNX average    : "
        f"{onnx_results['average']:.2f} ms"
    )

    print(
        f"Speedup         : "
        f"{speedup:.2f}x"
    )

    print(
        f"PyTorch FPS     : "
        f"{pytorch_results['fps']:.2f}"
    )

    print(
        f"ONNX FPS        : "
        f"{onnx_results['fps']:.2f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()