from pathlib import Path
import json

import cv2
import numpy as np

from src.config import DEFAULT_THRESHOLD


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/monitoring")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = OUTPUT_DIR / "monitoring_report.json"

DRIFT_WARNING_THRESHOLD = 0.10
DRIFT_SIGNIFICANT_THRESHOLD = 0.25


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


# ============================================================
# PREDICTION MONITORING
# ============================================================

def monitor_predictions(
    anomaly_scores,
    threshold=DEFAULT_THRESHOLD,
):
    anomaly_scores = np.asarray(
        anomaly_scores,
        dtype=np.float32,
    )

    if len(anomaly_scores) == 0:
        raise ValueError(
            "No anomaly scores provided."
        )

    threshold = float(threshold)

    predictions = anomaly_scores >= threshold

    defect_count = int(
        np.sum(predictions)
    )

    normal_count = int(
        len(predictions) - defect_count
    )

    defect_rate = (
        defect_count / len(predictions)
    )

    return {
        "threshold": threshold,
        "normal_count": normal_count,
        "defect_count": defect_count,
        "defect_rate": float(defect_rate),
    }


# ============================================================
# LATENCY MONITORING
# ============================================================

def monitor_latency(latencies):
    latencies = np.asarray(
        latencies,
        dtype=np.float32,
    )

    if len(latencies) == 0:
        raise ValueError(
            "No latency values provided."
        )

    mean_latency = float(
        np.mean(latencies)
    )

    return {
        "count": int(len(latencies)),
        "mean_ms": mean_latency,
        "p50_ms": float(
            np.percentile(latencies, 50)
        ),
        "p95_ms": float(
            np.percentile(latencies, 95)
        ),
        "p99_ms": float(
            np.percentile(latencies, 99)
        ),
        "min_ms": float(
            np.min(latencies)
        ),
        "max_ms": float(
            np.max(latencies)
        ),
        "fps": float(
            1000.0 / mean_latency
        ),
    }


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(image_path):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return {
        "brightness": float(
            np.mean(gray)
        ),
        "contrast": float(
            np.std(gray)
        ),
        "width": int(
            image.shape[1]
        ),
        "height": int(
            image.shape[0]
        ),
    }


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    reference_values,
    current_values,
    bins=10,
):
    """
    Calculate Population Stability Index.

    IMPORTANT:
    Histogram bins are created ONLY from the reference
    distribution and then reused for current data.

    This prevents the current distribution from changing
    the monitoring reference.
    """

    reference_values = np.asarray(
        reference_values,
        dtype=np.float64,
    )

    current_values = np.asarray(
        current_values,
        dtype=np.float64,
    )

    if (
        len(reference_values) == 0
        or len(current_values) == 0
    ):
        return 0.0

    reference_min = np.min(
        reference_values
    )

    reference_max = np.max(
        reference_values
    )

    # Constant reference distribution
    if (
        reference_max - reference_min
        < 1e-12
    ):
        return 0.0

    # Reference-defined bins
    edges = np.linspace(
        reference_min,
        reference_max,
        bins + 1,
    )

    # Extend outer edges so current values
    # outside the reference range are still captured.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_hist, _ = np.histogram(
        reference_values,
        bins=edges,
    )

    current_hist, _ = np.histogram(
        current_values,
        bins=edges,
    )

    # Convert counts to probabilities.
    reference_hist = (
        reference_hist.astype(np.float64)
    )

    current_hist = (
        current_hist.astype(np.float64)
    )

    reference_hist += 1e-6
    current_hist += 1e-6

    reference_hist /= np.sum(
        reference_hist
    )

    current_hist /= np.sum(
        current_hist
    )

    psi = np.sum(
        (
            current_hist
            - reference_hist
        )
        * np.log(
            current_hist
            / reference_hist
        )
    )

    return float(psi)


# ============================================================
# DRIFT CLASSIFICATION
# ============================================================

def classify_drift(psi):
    psi = float(psi)

    if psi < DRIFT_WARNING_THRESHOLD:
        return "NO_DRIFT"

    if psi < DRIFT_SIGNIFICANT_THRESHOLD:
        return "MODERATE_DRIFT"

    return "SIGNIFICANT_DRIFT"


# ============================================================
# OVERALL MONITORING STATUS
# ============================================================

def calculate_overall_status(report):
    """
    Combine drift indicators into one operational status.

    HEALTHY:
        No significant drift.

    WARNING:
        Moderate drift detected.

    DRIFT_DETECTED:
        Significant drift detected.
    """

    drift_statuses = []

    anomaly_monitoring = report.get(
        "anomaly_score_monitoring"
    )

    if anomaly_monitoring:
        drift_statuses.append(
            anomaly_monitoring["drift_status"]
        )

    brightness = report.get(
        "brightness"
    )

    if brightness:
        drift_statuses.append(
            brightness["drift_status"]
        )

    contrast = report.get(
        "contrast"
    )

    if contrast:
        drift_statuses.append(
            contrast["drift_status"]
        )

    if "SIGNIFICANT_DRIFT" in drift_statuses:
        return "DRIFT_DETECTED"

    if "MODERATE_DRIFT" in drift_statuses:
        return "WARNING"

    return "HEALTHY"


# ============================================================
# MONITORING REPORT
# ============================================================

def generate_monitoring_report(
    baseline_scores,
    current_scores,
    baseline_brightness=None,
    current_brightness=None,
    baseline_contrast=None,
    current_contrast=None,
    latencies=None,
    threshold=DEFAULT_THRESHOLD,
):
    baseline_scores = np.asarray(
        baseline_scores,
        dtype=np.float32,
    )

    current_scores = np.asarray(
        current_scores,
        dtype=np.float32,
    )

    if len(baseline_scores) == 0:
        raise ValueError(
            "Baseline scores cannot be empty."
        )

    if len(current_scores) == 0:
        raise ValueError(
            "Current scores cannot be empty."
        )

    # --------------------------------------------------------
    # Anomaly score drift
    # --------------------------------------------------------

    score_psi = calculate_psi(
        baseline_scores,
        current_scores,
    )

    report = {
        "anomaly_score_monitoring": {
            "baseline": calculate_statistics(
                baseline_scores
            ),
            "current": calculate_statistics(
                current_scores
            ),
            "psi": score_psi,
            "drift_status": classify_drift(
                score_psi
            ),
        }
    }

    # --------------------------------------------------------
    # Prediction monitoring
    # --------------------------------------------------------

    report["predictions"] = monitor_predictions(
        current_scores,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Brightness drift
    # --------------------------------------------------------

    if (
        baseline_brightness is not None
        and current_brightness is not None
    ):
        brightness_psi = calculate_psi(
            baseline_brightness,
            current_brightness,
        )

        report["brightness"] = {
            "baseline": calculate_statistics(
                baseline_brightness
            ),
            "current": calculate_statistics(
                current_brightness
            ),
            "psi": brightness_psi,
            "drift_status": classify_drift(
                brightness_psi
            ),
        }

    # --------------------------------------------------------
    # Contrast drift
    # --------------------------------------------------------

    if (
        baseline_contrast is not None
        and current_contrast is not None
    ):
        contrast_psi = calculate_psi(
            baseline_contrast,
            current_contrast,
        )

        report["contrast"] = {
            "baseline": calculate_statistics(
                baseline_contrast
            ),
            "current": calculate_statistics(
                current_contrast
            ),
            "psi": contrast_psi,
            "drift_status": classify_drift(
                contrast_psi
            ),
        }

    # --------------------------------------------------------
    # Latency monitoring
    # --------------------------------------------------------

    if latencies is not None:
        report["latency"] = monitor_latency(
            latencies
        )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    report["overall_status"] = calculate_overall_status(
        report
    )

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_monitoring_report(
    report,
    output_path=REPORT_PATH,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    return output_path


# ============================================================
# PRINT REPORT
# ============================================================

def print_monitoring_report(report):
    print("=" * 70)
    print("EDGE AI MONITORING REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    statistics = report[
        "anomaly_score_monitoring"
    ]

    print("\nANOMALY SCORE")
    print("-" * 70)

    print(
        f"Baseline mean : "
        f"{statistics['baseline']['mean']:.4f}"
    )

    print(
        f"Current mean  : "
        f"{statistics['current']['mean']:.4f}"
    )

    print(
        f"PSI           : "
        f"{statistics['psi']:.4f}"
    )

    print(
        f"Drift status  : "
        f"{statistics['drift_status']}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction = report[
        "predictions"
    ]

    print("\nPREDICTIONS")
    print("-" * 70)

    print(
        f"Threshold     : "
        f"{prediction['threshold']:.6f}"
    )

    print(
        f"Normal        : "
        f"{prediction['normal_count']}"
    )

    print(
        f"Defect        : "
        f"{prediction['defect_count']}"
    )

    print(
        f"Defect rate   : "
        f"{prediction['defect_rate']:.4f}"
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    if "brightness" in report:
        print("\nBRIGHTNESS")
        print("-" * 70)

        print(
            f"Baseline mean : "
            f"{report['brightness']['baseline']['mean']:.4f}"
        )

        print(
            f"Current mean  : "
            f"{report['brightness']['current']['mean']:.4f}"
        )

        print(
            f"PSI           : "
            f"{report['brightness']['psi']:.4f}"
        )

        print(
            f"Drift status  : "
            f"{report['brightness']['drift_status']}"
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    if "contrast" in report:
        print("\nCONTRAST")
        print("-" * 70)

        print(
            f"Baseline mean : "
            f"{report['contrast']['baseline']['mean']:.4f}"
        )

        print(
            f"Current mean  : "
            f"{report['contrast']['current']['mean']:.4f}"
        )

        print(
            f"PSI           : "
            f"{report['contrast']['psi']:.4f}"
        )

        print(
            f"Drift status  : "
            f"{report['contrast']['drift_status']}"
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    if "latency" in report:
        latency = report["latency"]

        print("\nLATENCY")
        print("-" * 70)

        print(
            f"Mean          : "
            f"{latency['mean_ms']:.2f} ms"
        )

        print(
            f"P50           : "
            f"{latency['p50_ms']:.2f} ms"
        )

        print(
            f"P95           : "
            f"{latency['p95_ms']:.2f} ms"
        )

        print(
            f"P99           : "
            f"{latency['p99_ms']:.2f} ms"
        )

        print(
            f"FPS           : "
            f"{latency['fps']:.2f}"
        )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    print("\nOVERALL STATUS")
    print("-" * 70)

    print(
        f"Status        : "
        f"{report['overall_status']}"
    )

    print("\n" + "=" * 70)


# ============================================================
# DEMO
# ============================================================

def run_demo():

    print("\n" + "=" * 70)
    print("MONITORING + DRIFT DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Baseline anomaly scores
    # --------------------------------------------------------

    baseline_scores = np.array([
        6.3,
        7.1,
        8.2,
        9.5,
        10.1,
        10.8,
        11.2,
        12.0,
        13.1,
        14.2,
        15.0,
        16.2,
        17.0,
        18.1,
        19.0,
        20.2,
        22.1,
        25.0,
        30.0,
        35.0,
    ])

    # --------------------------------------------------------
    # Current anomaly scores
    # --------------------------------------------------------

    current_scores = np.array([
        6.8,
        7.5,
        8.4,
        9.1,
        10.4,
        11.0,
        11.5,
        12.3,
        13.0,
        14.5,
        15.3,
        16.0,
        17.4,
        18.5,
        19.4,
        21.0,
        23.5,
        26.0,
        31.0,
        36.0,
    ])

    # --------------------------------------------------------
    # Example image-quality monitoring
    # --------------------------------------------------------

    baseline_brightness = np.array([
        110,
        112,
        114,
        116,
        118,
        120,
        121,
        123,
        125,
        126,
        128,
        130,
        131,
        133,
        135,
        137,
        139,
        141,
        143,
        145,
    ])

    current_brightness = np.array([
        111,
        113,
        115,
        117,
        119,
        121,
        122,
        124,
        126,
        127,
        129,
        131,
        132,
        134,
        136,
        138,
        140,
        142,
        144,
        146,
    ])

    baseline_contrast = np.array([
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
    ])

    current_contrast = np.array([
        35.5,
        36.5,
        37.5,
        38.5,
        39.5,
        40.5,
        41.5,
        42.5,
        43.5,
        44.5,
        45.5,
        46.5,
        47.5,
        48.5,
        49.5,
        50.5,
        51.5,
        52.5,
        53.5,
        54.5,
    ])

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latencies = np.array([
        85.2,
        88.1,
        90.3,
        91.4,
        92.0,
        89.8,
        94.2,
        91.7,
        90.1,
        93.5,
    ])

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    report = generate_monitoring_report(
        baseline_scores=baseline_scores,
        current_scores=current_scores,
        baseline_brightness=baseline_brightness,
        current_brightness=current_brightness,
        baseline_contrast=baseline_contrast,
        current_contrast=current_contrast,
        latencies=latencies,
        threshold=DEFAULT_THRESHOLD,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_monitoring_report(
        report
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    saved_path = save_monitoring_report(
        report
    )

    print(
        f"\nJSON report saved: "
        f"{saved_path}"
    )

    print("\nMonitoring demo complete.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_demo()