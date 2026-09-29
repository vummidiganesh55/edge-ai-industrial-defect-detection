import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate_image_level(results, threshold=None):
    y_true = np.array(
        [item["label"] for item in results],
        dtype=int,
    )

    scores = np.array(
        [item["anomaly_score"] for item in results],
        dtype=float,
    )

    # --------------------------------------------------
    # 1. Threshold-independent AUROC
    # --------------------------------------------------
    auroc = roc_auc_score(y_true, scores)

    # --------------------------------------------------
    # 2. Calculate optimal threshold for analysis
    # --------------------------------------------------
    fpr_curve, tpr_curve, thresholds = roc_curve(
        y_true,
        scores,
    )

    j_scores = tpr_curve - fpr_curve
    best_index = np.argmax(j_scores)
    best_threshold = thresholds[best_index]

    # --------------------------------------------------
    # 3. Select evaluation threshold
    # --------------------------------------------------
    if threshold is None:
        evaluation_threshold = best_threshold
        threshold_source = "Youden J"
    else:
        evaluation_threshold = float(threshold)
        threshold_source = "Production"

    # --------------------------------------------------
    # 4. Classification using selected threshold
    # --------------------------------------------------
    y_pred = (
        scores >= evaluation_threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    # --------------------------------------------------
    # 5. Confusion matrix values
    # --------------------------------------------------
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # --------------------------------------------------
    # 6. Print evaluation
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PATCHCORE IMAGE-LEVEL EVALUATION")
    print("=" * 60)

    print(f"Image AUROC       : {auroc:.4f}")
    print(f"Youden threshold  : {best_threshold:.4f}")
    print(f"Evaluation thresh.: {evaluation_threshold:.6f}")
    print(f"Threshold source  : {threshold_source}")

    print("\nClassification Metrics")
    print("-" * 60)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"FPR               : {fpr:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nConfusion Matrix Details")
    print("-" * 60)
    print(f"TN                : {tn}")
    print(f"FP                : {fp}")
    print(f"FN                : {fn}")
    print(f"TP                : {tp}")

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Defect",
            ],
            zero_division=0,
        )
    )

    return {
        "auroc": float(auroc),
        "best_threshold": float(best_threshold),
        "threshold": float(evaluation_threshold),
        "threshold_source": threshold_source,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }