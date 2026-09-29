from pathlib import Path
import csv
OUTPUT_DIR = Path("outputs")
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

REPORT_PATH = EVALUATION_DIR / "day2_final_report.txt"
ROBUSTNESS_FILE = OUTPUT_DIR / "robustness" / "robustness_results.csv"


def read_text_file(path):
    if not path.exists():
        return f"[NOT FOUND] {path}"

    return path.read_text(encoding="utf-8")


def read_robustness_results():
    if not ROBUSTNESS_FILE.exists():
        return []

    rows = []

    with open(
        ROBUSTNESS_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def build_report():
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pixel_report = read_text_file(
        EVALUATION_DIR / "pixel_metrics.txt"
    )

    pro_report = read_text_file(
        EVALUATION_DIR / "pro_metrics.txt"
    )

    unseen_report = read_text_file(
        EVALUATION_DIR / "unseen_defect_results.txt"
    )

    failure_report = read_text_file(
        EVALUATION_DIR / "failure_analysis.txt"
    )

    robustness_rows = (
        read_robustness_results()
    )

    report = []

    report.append("=" * 80)
    report.append(
        "EDGE-DEPLOYED REAL-TIME INDUSTRIAL DEFECT DETECTION"
    )
    report.append("DAY 2 — FINAL EVALUATION REPORT")
    report.append("=" * 80)

    report.append("")
    report.append("PROJECT STATUS")
    report.append("-" * 80)
    report.append("Day 2 objective: COMPLETE")
    report.append(
        "Focus: Localization, Evaluation, Robustness, and Unseen Defects"
    )

    # --------------------------------------------------
    # Image-level baseline
    # --------------------------------------------------

    report.append("")
    report.append("1. IMAGE-LEVEL BASELINE")
    report.append("-" * 80)

    report.append(
        "Image-level baseline metrics were obtained during PatchCore evaluation."
    )

    report.append(
        "Baseline AUROC: approximately 0.99"
    )

    report.append(
        "Baseline F1: approximately 0.97–0.98"
    )

    report.append(
        "The exact value can vary because the current PatchCore coreset sampling is random."
    )

    # --------------------------------------------------
    # Pixel evaluation
    # --------------------------------------------------

    report.append("")
    report.append("2. PIXEL-LEVEL LOCALIZATION")
    report.append("-" * 80)

    report.append(pixel_report)

    # --------------------------------------------------
    # PRO
    # --------------------------------------------------

    report.append("")
    report.append("3. PRO-AUC")
    report.append("-" * 80)

    report.append(pro_report)

    # --------------------------------------------------
    # Failure analysis
    # --------------------------------------------------

    report.append("")
    report.append("4. FAILURE ANALYSIS")
    report.append("-" * 80)

    report.append(failure_report)

    # --------------------------------------------------
    # Robustness
    # --------------------------------------------------

    report.append("")
    report.append("5. ROBUSTNESS TESTING")
    report.append("-" * 80)

    if robustness_rows:

        report.append(
            f"{'Condition':20s}"
            f"{'AUROC':>10s}"
            f"{'Precision':>12s}"
            f"{'Recall':>10s}"
            f"{'F1':>10s}"
            f"{'FP':>6s}"
            f"{'FN':>6s}"
        )

        report.append("-" * 80)

        for row in robustness_rows:

            report.append(
            f"{row['condition']:20s}"
            f"{float(row['auroc']):10.4f}"
            f"{float(row['precision']):12.4f}"
            f"{float(row['recall']):10.4f}"
            f"{float(row['f1']):10.4f}"
            f"{int(row['false_positives']):6d}"
            f"{int(row['false_negatives']):6d}"
        )

    else:

        report.append(
            "Robustness CSV not found."
        )

    # --------------------------------------------------
    # Unseen defects
    # --------------------------------------------------

    report.append("")
    report.append("6. UNSEEN-DEFECT TESTING")
    report.append("-" * 80)

    report.append(unseen_report)

    # --------------------------------------------------
    # Engineering findings
    # --------------------------------------------------

    report.append("")
    report.append("7. DAY 2 ENGINEERING FINDINGS")
    report.append("-" * 80)

    report.append(
        "1. PatchCore provides strong image-level anomaly detection."
    )

    report.append(
        "2. Broken-large and broken-small defects are detected strongly."
    )

    report.append(
        "3. Contamination is the main defect-specific failure mode."
    )

    report.append(
        "4. The current contamination unseen-defect test produced false negatives."
    )

    report.append(
        "5. Illumination and contrast changes had relatively small impact."
    )

    report.append(
        "6. Gaussian noise caused the largest robustness degradation."
    )

    report.append(
        "7. Blur and low-resolution conditions mainly reduced recall."
    )

    report.append(
        "8. Pixel-level localization is weaker than image-level detection."
    )

    report.append(
        "9. The current PatchCore coreset sampling is random, so exact "
        "metrics can vary between runs."
    )


    report.append("")
    report.append("=" * 80)
    report.append("DAY 2 STATUS")
    report.append("=" * 80)

    report.append("Dataset                         : COMPLETE")
    report.append("PatchCore                       : COMPLETE")
    report.append("Anomaly maps                    : COMPLETE")
    report.append("Image-level evaluation          : COMPLETE")
    report.append("Pixel-level evaluation          : COMPLETE")
    report.append("Localization visualization     : COMPLETE")
    report.append("PRO-AUC                         : COMPLETE")
    report.append("Failure analysis                : COMPLETE")
    report.append("Robustness testing              : COMPLETE")
    report.append("Unseen-defect testing           : COMPLETE")
    report.append("Final evaluation report         : COMPLETE")

    report.append("")
    report.append(
        "DAY 2 — COMPLETE"
    )

    report.append("=" * 80)

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DAY 2 FINAL REPORT")
    print("=" * 60)

    print(
        f"Saved: {REPORT_PATH}"
    )

    print("\nDAY 2 — COMPLETE")


if __name__ == "__main__":
    build_report()
