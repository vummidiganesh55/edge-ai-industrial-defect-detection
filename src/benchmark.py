import os
import time
import statistics
from pathlib import Path

import numpy as np
import torch
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore import PatchCore
from src.patchcore_pipeline import build_memory_bank
from src.preprocessing import get_test_transform
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "data/mvtec_anomaly_detection"
CATEGORY = "bottle"

BATCH_SIZE = 8
SAMPLING_RATIO = 0.1

NUM_WARMUP = 5
NUM_RUNS = 50

IMAGE_PATH = (
    "data/mvtec_anomaly_detection/"
    "bottle/test/broken_large/000.png"
)

OUTPUT_DIR = Path("outputs/benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# MEMORY USAGE
# ============================================================

def get_process_memory_mb():
    """
    Returns process RSS memory in MB.
    Works on Windows/Linux/macOS.
    """

    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    except ImportError:
        return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = get_test_transform()

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor.to(DEVICE)


# ============================================================
# SINGLE INFERENCE
# ============================================================

def run_inference(image_tensor, feature_extractor, patchcore):
    with torch.no_grad():
        features = feature_extractor(image_tensor)

        patch_distances = patchcore.predict(features)

        # PatchCore returns multiple distance arrays.
        # Combine them before calculating the final anomaly score.
        if isinstance(patch_distances, (tuple, list)):
            distances = []

            for distance in patch_distances:
                distance = np.asarray(distance, dtype=np.float32)
                distances.append(distance.reshape(-1))

            patch_distances = np.concatenate(distances)

        else:
            patch_distances = np.asarray(
                patch_distances,
                dtype=np.float32
            ).reshape(-1)

        anomaly_score = float(np.max(patch_distances))

    return anomaly_score


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 60)
    print("CPU BASELINE BENCHMARK")
    print("=" * 60)

    print(f"Device       : {DEVICE}")
    print("Benchmark:CPU baseline")
    print(f"Category     : {CATEGORY}")
    print(f"Warmup runs  : {NUM_WARMUP}")
    print(f"Benchmark runs: {NUM_RUNS}")
    print("=" * 60)

    # --------------------------------------------------------
    # Build memory bank
    # --------------------------------------------------------

    print("\nBUILDING PATCHCORE MEMORY BANK")
    print("-" * 60)

    patchcore = build_memory_bank(
        category=CATEGORY,
        batch_size=BATCH_SIZE,
        sampling_ratio=SAMPLING_RATIO,
    )

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    feature_extractor = ResNet18FeatureExtractor().to(DEVICE)

    feature_extractor.eval()

    # --------------------------------------------------------
    # Load test image
    # --------------------------------------------------------

    image_tensor = load_image(IMAGE_PATH)

    print("\nTest image:")
    print(IMAGE_PATH)

    # --------------------------------------------------------
    # Initial memory measurement
    # --------------------------------------------------------

    memory_before = get_process_memory_mb()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\nWARM-UP")
    print("-" * 60)

    for i in range(NUM_WARMUP):

        _ = run_inference(
            image_tensor,
            feature_extractor,
            patchcore,
        )

        print(
            f"Warm-up {i + 1}/{NUM_WARMUP}"
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\nBENCHMARK")
    print("-" * 60)

    latencies = []
    scores = []

    for i in range(NUM_RUNS):

        start = time.perf_counter()

        score = run_inference(
            image_tensor,
            feature_extractor,
            patchcore,
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(latency_ms)
        scores.append(score)

        print(
            f"Run {i + 1:02d}/{NUM_RUNS} | "
            f"{latency_ms:.2f} ms | "
            f"score={score:.4f}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_latency = statistics.mean(
        latencies
    )

    p50_latency = float(
        np.percentile(latencies, 50)
    )

    p95_latency = float(
        np.percentile(latencies, 95)
    )

    minimum_latency = min(latencies)
    maximum_latency = max(latencies)

    fps = (
        1000.0 / average_latency
        if average_latency > 0
        else 0
    )

    throughput = fps

    memory_after = get_process_memory_mb()

    if (
        memory_before is not None
        and memory_after is not None
    ):
        memory_used = (
            memory_after - memory_before
        )
    else:
        memory_used = None

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CPU BASELINE RESULTS")
    print("=" * 60)

    print(
        f"Average latency : {average_latency:.2f} ms"
    )

    print(
        f"P50 latency     : {p50_latency:.2f} ms"
    )

    print(
        f"P95 latency     : {p95_latency:.2f} ms"
    )

    print(
        f"Min latency     : {minimum_latency:.2f} ms"
    )

    print(
        f"Max latency     : {maximum_latency:.2f} ms"
    )

    print(
        f"FPS             : {fps:.2f}"
    )

    print(
        f"Throughput      : {throughput:.2f} images/sec"
    )

    if memory_used is not None:

        print(
            f"Memory change   : {memory_used:.2f} MB"
        )

    else:

        print(
            "Memory change   : unavailable "
            "(install psutil)"
        )

    print(
        f"Mean score      : {statistics.mean(scores):.4f}"
    )

    print(
        f"Score std       : {statistics.stdev(scores):.6f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        "cpu_baseline_results.csv"
    )

    with open(csv_path, "w") as f:

        f.write(
            "metric,value\n"
        )

        f.write(
            f"device,{DEVICE}\n"
        )

        f.write(
            f"category,{CATEGORY}\n"
        )

        f.write(
            f"warmup_runs,{NUM_WARMUP}\n"
        )

        f.write(
            f"benchmark_runs,{NUM_RUNS}\n"
        )

        f.write(
            f"average_latency_ms,{average_latency:.6f}\n"
        )

        f.write(
            f"p50_latency_ms,{p50_latency:.6f}\n"
        )

        f.write(
            f"p95_latency_ms,{p95_latency:.6f}\n"
        )

        f.write(
            f"min_latency_ms,{minimum_latency:.6f}\n"
        )

        f.write(
            f"max_latency_ms,{maximum_latency:.6f}\n"
        )

        f.write(
            f"fps,{fps:.6f}\n"
        )

        f.write(
            f"throughput_images_per_sec,{throughput:.6f}\n"
        )

        if memory_used is not None:

            f.write(
                f"memory_change_mb,{memory_used:.6f}\n"
            )

        f.write(
            f"mean_anomaly_score,"
            f"{statistics.mean(scores):.6f}\n"
        )

        f.write(
            f"score_std,"
            f"{statistics.stdev(scores):.6f}\n"
        )

    print(
        f"\nSaved: {csv_path}"
    )

    print(
        "\nCPU baseline benchmark complete."
    )


if __name__ == "__main__":
    main()