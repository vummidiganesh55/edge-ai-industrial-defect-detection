import json
import time
from pathlib import Path

import requests


API_URL = "http://localhost:8000/predict"

CALIBRATION_FILE = Path(
    "outputs/threshold_calibration.json"
)

TEST_ROOT = Path(
    "data/mvtec_anomaly_detection/bottle/test"
)

OUTPUT_JSON = Path(
    "outputs/calibrated_threshold_evaluation.json"
)

OUTPUT_CSV = Path(
    "outputs/calibrated_threshold_predictions.csv"
)


def load_threshold():
    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return float(data["threshold"])


def collect_test_images():
    images = []

    for defect_dir in sorted(TEST_ROOT.iterdir()):

        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name

        label = (
            0
            if defect_type == "good"
            else 1
        )

        for image_path in sorted(
            defect_dir.glob("*.png")
        ):

            images.append(
                {
                    "path": image_path,
                    "filename": image_path.name,
                    "defect_type": defect_type,
                    "label": label,
                }
            )

    return images


def main():

    print("=" * 70)
    print("PATCHCORE CALIBRATED THRESHOLD EVALUATION")
    print("=" * 70)

    threshold = load_threshold()

    print(
        f"Locked threshold : {threshold:.6f}"
    )

    print(
        f"Test directory   : {TEST_ROOT}"
    )

    images = collect_test_images()

    print(
        f"Test images      : {len(images)}"
    )

    results = []

    for counter, item in enumerate(
        images,
        start=1,
    ):

        image_path = item["path"]

        start = time.perf_counter()

        with open(
            image_path,
            "rb",
        ) as file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        image_path.name,
                        file,
                        "image/png",
                    )
                },
                timeout=120,
            )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        response.raise_for_status()

        prediction = response.json()

        score = float(
            prediction["anomaly_score"]
        )

        predicted_label = (
            1
            if score >= threshold
            else 0
        )

        results.append(
            {
                "filename": item["filename"],
                "defect_type": item["defect_type"],
                "true_label": item["label"],
                "predicted_label": predicted_label,
                "prediction": (
                    "DEFECT"
                    if predicted_label == 1
                    else "NORMAL"
                ),
                "anomaly_score": score,
                "threshold": threshold,
                "latency_ms": latency_ms,
            }
        )

        print(
            f"{counter:02d}/{len(images)} "
            f"{item['filename']:<12} "
            f"{item['defect_type']:<16} "
            f"score={score:>9.4f} "
            f"prediction="
            f"{'DEFECT' if predicted_label else 'NORMAL':<6} "
            f"latency={latency_ms:>7.2f} ms"
        )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    tp = sum(
        r["true_label"] == 1
        and r["predicted_label"] == 1
        for r in results
    )

    tn = sum(
        r["true_label"] == 0
        and r["predicted_label"] == 0
        for r in results
    )

    fp = sum(
        r["true_label"] == 0
        and r["predicted_label"] == 1
        for r in results
    )

    fn = sum(
        r["true_label"] == 1
        and r["predicted_label"] == 0
        for r in results
    )

    total = len(results)

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    latencies = [
        r["latency_ms"]
        for r in results
    ]

    avg_latency = (
        sum(latencies) / len(latencies)
    )

    sorted_latencies = sorted(latencies)

    p95_index = int(
        0.95 * len(sorted_latencies)
    ) - 1

    p95_index = max(
        0,
        min(
            p95_index,
            len(sorted_latencies) - 1,
        ),
    )

    p95_latency = sorted_latencies[
        p95_index
    ]

    throughput = (
        1000.0 / avg_latency
        if avg_latency > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Per-defect results
    # ---------------------------------------------------------

    defect_summary = {}

    for defect_type in sorted(
        set(
            r["defect_type"]
            for r in results
        )
    ):

        subset = [
            r
            for r in results
            if r["defect_type"] == defect_type
        ]

        correct = sum(
            r["true_label"]
            == r["predicted_label"]
            for r in subset
        )

        defect_summary[defect_type] = {
            "total": len(subset),
            "correct": correct,
            "accuracy": (
                correct / len(subset)
            ),
            "false_negatives": sum(
                r["true_label"] == 1
                and r["predicted_label"] == 0
                for r in subset
            ),
            "false_positives": sum(
                r["true_label"] == 0
                and r["predicted_label"] == 1
                for r in subset
            ),
        }

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CALIBRATED THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"Threshold : {threshold:.6f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1        : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"FPR       : {fpr:.4f} "
        f"({fpr * 100:.2f}%)"
    )

    print(
        f"FNR       : {fnr:.4f} "
        f"({fnr * 100:.2f}%)"
    )

    print("\nConfusion Matrix")
    print("----------------")
    print(
        f"TN = {tn}"
    )
    print(
        f"FP = {fp}"
    )
    print(
        f"FN = {fn}"
    )
    print(
        f"TP = {tp}"
    )

    print("\nLatency")
    print("-------")
    print(
        f"Average : {avg_latency:.2f} ms"
    )
    print(
        f"P95     : {p95_latency:.2f} ms"
    )
    print(
        f"Throughput : {throughput:.2f} images/sec"
    )

    print("\nPer-defect results")
    print("------------------")

    for defect_type, summary in defect_summary.items():

        print(
            f"{defect_type:<18} "
            f"{summary['correct']}/"
            f"{summary['total']} correct "
            f"({summary['accuracy'] * 100:.2f}%) "
            f"FN={summary['false_negatives']} "
            f"FP={summary['false_positives']}"
        )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    output = {
        "threshold": threshold,
        "test_images": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "latency_ms": {
            "average": avg_latency,
            "p95": p95_latency,
        },
        "throughput_images_per_second": throughput,
        "per_defect": defect_summary,
        "predictions": results,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    import csv

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()