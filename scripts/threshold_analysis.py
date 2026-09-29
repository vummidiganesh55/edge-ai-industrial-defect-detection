import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CSV_PATH = "outputs/patchcore_predictions.csv"

df = pd.read_csv(CSV_PATH)

y_true = df["expected"]

scores = df["anomaly_score"]

# Test thresholds from 10 to 25
thresholds = np.arange(10.0, 25.01, 0.25)

results = []

for threshold in thresholds:

    y_pred = np.where(
        scores > threshold,
        "DEFECT",
        "NORMAL"
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="DEFECT",
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=["NORMAL", "DEFECT"]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) else 0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) else 0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })


results_df = pd.DataFrame(results)

# Best F1
best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

# Best recall while maintaining zero FPs
zero_fp = results_df[
    results_df["FP"] == 0
]

best_zero_fp_recall = zero_fp.loc[
    zero_fp["recall"].idxmax()
]

print("=" * 75)
print("PATCHCORE THRESHOLD ANALYSIS")
print("=" * 75)

print()
print("CURRENT BASELINE")
print("-" * 75)

baseline = results_df[
    np.isclose(
        results_df["threshold"],
        19.75
    )
]

print(baseline.to_string(index=False))

print()
print("BEST F1 THRESHOLD")
print("-" * 75)
print(best_f1.to_string())

print()
print("BEST RECALL WITH ZERO FALSE POSITIVES")
print("-" * 75)
print(best_zero_fp_recall.to_string())

print()
print("THRESHOLD COMPARISON")
print("-" * 75)

interesting = results_df[
    results_df["threshold"].between(12, 20)
]

print(
    interesting[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "FP",
            "FN",
        ]
    ].to_string(index=False)
)

# Save complete analysis
output_path = "outputs/threshold_analysis.csv"

results_df.to_csv(
    output_path,
    index=False
)

print()
print("=" * 75)
print(f"Saved: {output_path}")
print("=" * 75)