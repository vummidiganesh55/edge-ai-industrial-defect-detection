import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank


# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "data/mvtec_anomaly_detection"
CATEGORY = "bottle"
BATCH_SIZE = 8
SAMPLING_RATIO = 0.1

IMAGE_INDEX = 0

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50

OUTPUT_DIR = Path("outputs/resource_profiling")
OUTPUT_CSV = OUTPUT_DIR / "resource_profile.csv"
OUTPUT_REPORT = OUTPUT_DIR / "resource_profile.txt"


# ============================================================
# GPU INFORMATION
# ============================================================

def get_gpu_info():
    """
    Get GPU utilization and VRAM information using nvidia-smi.
    """

    if not torch.cuda.is_available():
        return {
            "gpu_utilization_percent": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
        }

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,"
                "memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        values = result.stdout.strip().split(",")

        if len(values) >= 3:

            return {
                "gpu_utilization_percent": float(
                    values[0].strip()
                ),
                "vram_used_mb": float(
                    values[1].strip()
                ),
                "vram_total_mb": float(
                    values[2].strip()
                ),
            }

    except Exception:
        pass

    return {
        "gpu_utilization_percent": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
    }


# ============================================================
# CPU / RAM INFORMATION
# ============================================================

def get_system_info():
    """
    Get CPU utilization and RAM usage.
    """

    memory = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(
            interval=0.1
        ),
        "ram_used_mb": memory.used / (1024 ** 2),
        "ram_total_mb": memory.total / (1024 ** 2),
        "ram_percent": memory.percent,
    }


# ============================================================
# SINGLE INFERENCE
# ============================================================

def run_inference(
    model,
    patchcore,
    image_tensor,
    device,
):
    """
    Run complete feature extraction + PatchCore inference.
    """

    image_tensor = image_tensor.to(device)

    if device.type == "cuda":

        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        features = model(
            image_tensor
        )

        anomaly_score, patch_distances = (
            patchcore.predict(
                features
            )
        )

    if device.type == "cuda":

        torch.cuda.synchronize()

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000.0

    return (
        anomaly_score,
        patch_distances,
        latency_ms,
    )


# ============================================================
# RESOURCE PROFILING
# ============================================================

def profile_resources():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("EDGE AI RESOURCE PROFILING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(
            0
        )

    else:

        device = torch.device("cpu")

        gpu_name = "None"

    print(f"Category          : {CATEGORY}")
    print(f"Device            : {device}")
    print(f"GPU               : {gpu_name}")
    print(f"Warmup iterations : {WARMUP_ITERATIONS}")
    print(f"Benchmark         : {BENCHMARK_ITERATIONS}")

    # --------------------------------------------------------
    # Build PatchCore memory bank
    # --------------------------------------------------------

    print("\nBuilding PatchCore memory bank...")

    patchcore = build_memory_bank(
        category=CATEGORY,
        batch_size=BATCH_SIZE,
        sampling_ratio=SAMPLING_RATIO,
    )

    print(
        f"Memory bank shape : "
        f"{patchcore.memory_bank.shape}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(),
    )

    sample = dataset[IMAGE_INDEX]

    image_tensor = (
        sample["image"]
        .unsqueeze(0)
    )

    print(
        f"Test image       : "
        f"{sample['path']}"
    )

    print(
        f"Defect type      : "
        f"{sample['defect_type']}"
    )

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    model = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    model.eval()

    # --------------------------------------------------------
    # Initial resource state
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("\nRunning warmup...")

    for _ in range(
        WARMUP_ITERATIONS
    ):

        run_inference(
            model,
            patchcore,
            image_tensor,
            device,
        )

    print("Warmup complete.")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        "\nRunning resource benchmark..."
    )

    latency_values = []

    cpu_values = []
    ram_values = []
    gpu_values = []
    vram_values = []

    scores = []

    for iteration in range(
        BENCHMARK_ITERATIONS
    ):

        # --------------------------------------------
        # Resource measurement before inference
        # --------------------------------------------

        system_before = get_system_info()
        gpu_before = get_gpu_info()

        # --------------------------------------------
        # Inference
        # --------------------------------------------

        (
            anomaly_score,
            patch_distances,
            latency_ms,
        ) = run_inference(
            model,
            patchcore,
            image_tensor,
            device,
        )

        # --------------------------------------------
        # Resource measurement after inference
        # --------------------------------------------

        system_after = get_system_info()
        gpu_after = get_gpu_info()

        latency_values.append(
            latency_ms
        )

        cpu_values.append(
            system_after["cpu_percent"]
        )

        ram_values.append(
            system_after["ram_used_mb"]
        )

        gpu_values.append(
            gpu_after[
                "gpu_utilization_percent"
            ]
        )

        vram_values.append(
            gpu_after[
                "vram_used_mb"
            ]
        )

        scores.append(
            float(anomaly_score)
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    latency_values = np.asarray(
        latency_values
    )

    cpu_values = np.asarray(
        cpu_values
    )

    ram_values = np.asarray(
        ram_values
    )

    gpu_values = np.asarray(
        gpu_values
    )

    vram_values = np.asarray(
        vram_values
    )

    scores = np.asarray(
        scores
    )

    avg_latency = (
        latency_values.mean()
    )

    p50_latency = np.percentile(
        latency_values,
        50,
    )

    p95_latency = np.percentile(
        latency_values,
        95,
    )

    min_latency = (
        latency_values.min()
    )

    max_latency = (
        latency_values.max()
    )

    fps = (
        1000.0 /
        avg_latency
    )

    avg_cpu = (
        cpu_values.mean()
    )

    avg_ram = (
        ram_values.mean()
    )

    avg_gpu = (
        gpu_values.mean()
    )

    avg_vram = (
        vram_values.mean()
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESOURCE PROFILING RESULTS")
    print("=" * 70)

    print(
        f"Device              : {device}"
    )

    print(
        f"GPU                 : {gpu_name}"
    )

    print(
        f"Average latency     : "
        f"{avg_latency:.2f} ms"
    )

    print(
        f"P50 latency         : "
        f"{p50_latency:.2f} ms"
    )

    print(
        f"P95 latency         : "
        f"{p95_latency:.2f} ms"
    )

    print(
        f"Minimum latency     : "
        f"{min_latency:.2f} ms"
    )

    print(
        f"Maximum latency     : "
        f"{max_latency:.2f} ms"
    )

    print(
        f"Throughput          : "
        f"{fps:.2f} FPS"
    )

    print(
        f"Average CPU         : "
        f"{avg_cpu:.2f} %"
    )

    print(
        f"Average RAM         : "
        f"{avg_ram:.2f} MB"
    )

    print(
        f"Average GPU         : "
        f"{avg_gpu:.2f} %"
    )

    print(
        f"Average VRAM        : "
        f"{avg_vram:.2f} MB"
    )

    print(
        f"Average score       : "
        f"{scores.mean():.4f}"
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "iteration",
                "latency_ms",
                "fps",
                "cpu_percent",
                "ram_used_mb",
                "gpu_utilization_percent",
                "vram_used_mb",
                "anomaly_score",
            ]
        )

        for i in range(
            BENCHMARK_ITERATIONS
        ):

            writer.writerow(
                [
                    i + 1,
                    latency_values[i],
                    1000.0
                    / latency_values[i],
                    cpu_values[i],
                    ram_values[i],
                    gpu_values[i],
                    vram_values[i],
                    scores[i],
                ]
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = f"""
EDGE AI RESOURCE PROFILING REPORT
=================================

Category:
{CATEGORY}

Device:
{device}

GPU:
{gpu_name}

Memory Bank:
{patchcore.memory_bank.shape}

Iterations:
{BENCHMARK_ITERATIONS}

Latency
-------
Average : {avg_latency:.2f} ms
P50     : {p50_latency:.2f} ms
P95     : {p95_latency:.2f} ms
Minimum : {min_latency:.2f} ms
Maximum : {max_latency:.2f} ms

Throughput
----------
FPS     : {fps:.2f}

CPU
---
Average utilization : {avg_cpu:.2f} %

RAM
---
Average used : {avg_ram:.2f} MB

GPU
---
Average utilization : {avg_gpu:.2f} %

VRAM
----
Average used : {avg_vram:.2f} MB

Anomaly Score
-------------
Average : {scores.mean():.4f}
Std     : {scores.std():.4f}
"""

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report.strip()
        )

    print(
        f"\nCSV saved    : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Report saved : "
        f"{OUTPUT_REPORT}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESOURCE PROFILING COMPLETE"
    )

    print(
        "=" * 70


    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    profile_resources()