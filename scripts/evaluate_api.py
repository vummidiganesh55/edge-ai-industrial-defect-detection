import csv
import json
import time
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

API_URL = "http://localhost:8000/predict"

DATASET_DIR = Path(
    "data/mvtec_anomaly_detection/bottle/test"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def expected_label(image_path):
    """
    MVTec:
      test/good/*       -> NORMAL
      test/<defect>/*   -> DEFECT
    """
    if image_path.parent.name == "good":
        return "NORMAL"

    return "DEFECT"


def predict(image_path):
    start = time.perf_counter()

    with open(image_path, "rb") as f:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    image_path.name,
                    f,
                    "image/png"
                )
            },
            timeout=120,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()

    result = response.json()

    return result, elapsed_ms


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PATCHCORE API EVALUATION")
    print("=" * 70)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    images = sorted(DATASET_DIR.rglob("*.png"))

    if not images:
        raise RuntimeError(
            f"No PNG images found in {DATASET_DIR}"
        )

    print(f"Dataset : {DATASET_DIR}")
    print(f"Images  : {len(images)}")
    print()

    y_true = []
    y_pred = []

    latencies = []
    results = []

    for i, image_path in enumerate(images, start=1):

        expected = expected_label(image_path)

        try:
            result, client_latency = predict(image_path)

            prediction = result["prediction"]

            y_true.append(expected)
            y_pred.append(prediction)

            latency = result.get(
                "latency_ms",
                client_latency
            )

            latencies.append(float(latency))

            results.append({
                "filename": image_path.name,
                "defect_type": image_path.parent.name,
                "expected": expected,
                "prediction": prediction,
                "anomaly_score": result.get("anomaly_score"),
                "threshold": result.get("threshold"),
                "risk_level": result.get("risk_level"),
                "action": result.get("action"),
                "latency_ms": latency,
                "device": result.get("device"),
            })

            status = "OK" if expected == prediction else "MISS"

            print(
                f"[{i:03d}/{len(images)}] "
                f"{image_path.parent.name:20s} "
                f"{image_path.name:10s} "
                f"{expected:7s} -> "
                f"{prediction:7s} "
                f"[{status}]"
            )

        except Exception as e:

            print(
                f"[{i:03d}/{len(images)}] "
                f"{image_path.name} ERROR: {e}"
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=["NORMAL", "DEFECT"],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    sorted_latencies = sorted(latencies)

    if sorted_latencies:
        p95_index = int(
            0.95 * len(sorted_latencies)
        ) - 1

        p95_index = max(
            0,
            min(p95_index, len(sorted_latencies) - 1)
        )

        p95_latency = sorted_latencies[p95_index]

    else:
        p95_latency = 0

    throughput = (
        1000 / avg_latency
        if avg_latency > 0
        else 0
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PATCHCORE INDUSTRIAL DEFECT EVALUATION")
    print("=" * 70)

    print(f"Total images       : {len(y_true)}")
    print(f"Correct predictions: {sum(a == b for a, b in zip(y_true, y_pred))}")
    print()

    print("CONFUSION MATRIX")
    print("-" * 40)
    print(f"True Negative  (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive  (TP): {tp}")
    print()

    print("CLASSIFICATION METRICS")
    print("-" * 40)
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Precision          : {precision * 100:.2f}%")
    print(f"Recall             : {recall * 100:.2f}%")
    print(f"F1 Score           : {f1 * 100:.2f}%")
    print(f"False Positive Rate: {false_positive_rate * 100:.2f}%")
    print(f"False Negative Rate: {false_negative_rate * 100:.2f}%")
    print()

    print("PERFORMANCE")
    print("-" * 40)
    print(f"Average latency    : {avg_latency:.2f} ms")
    print(f"P95 latency        : {p95_latency:.2f} ms")
    print(f"Throughput         : {throughput:.2f} images/sec")
    print()

    # ========================================================
    # SAVE JSON
    # ========================================================

    evaluation = {
        "category": "bottle",
        "dataset": "MVTec AD",
        "model": "ResNet18 + PatchCore",
        "device": results[0]["device"] if results else "unknown",

        "total_images": len(y_true),

        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },

        "classification": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },

        "performance": {
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "throughput_images_per_second": throughput,
        },

        "images": results,
    }

    json_path = OUTPUT_DIR / "patchcore_evaluation.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            evaluation,
            f,
            indent=2
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = OUTPUT_DIR / "patchcore_predictions.csv"

    if results:

        fieldnames = results[0].keys()

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()