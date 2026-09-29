from dataclasses import dataclass

from src.config import DEFAULT_THRESHOLD


# ============================================================
# DECISION RESULT
# ============================================================

@dataclass
class ProductionDecision:
    anomaly_score: float
    threshold: float
    status: str
    risk_level: str
    action: str


# ============================================================
# PRODUCTION DECISION
# ============================================================

def make_production_decision(
    anomaly_score,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Convert PatchCore anomaly score into a production decision.

    Decision:
        score < threshold  -> NORMAL
        score >= threshold -> DEFECT
    """

    anomaly_score = float(
        anomaly_score
    )

    threshold = float(
        threshold
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if anomaly_score >= threshold:

        status = "DEFECT"

        # Higher score means stronger anomaly signal.
        if anomaly_score >= threshold * 2:

            risk_level = "HIGH"
            action = "REJECT_AND_INSPECT"

        else:

            risk_level = "MEDIUM"
            action = "HOLD_FOR_INSPECTION"

    else:

        status = "NORMAL"
        risk_level = "LOW"
        action = "ACCEPT"

    return ProductionDecision(
        anomaly_score=anomaly_score,
        threshold=threshold,
        status=status,
        risk_level=risk_level,
        action=action,
    )


# ============================================================
# TEST
# ============================================================

def run_demo():

    test_scores = [
        5.0,
        15.0,
        19.0,
        DEFAULT_THRESHOLD,
        25.0,
        40.0,
    ]

    print("=" * 70)
    print("PRODUCTION DECISION LAYER")
    print("=" * 70)

    print(
        f"Threshold: {DEFAULT_THRESHOLD:.4f}"
    )

    print()

    for score in test_scores:

        decision = make_production_decision(
            score
        )

        print(
            f"Score={decision.anomaly_score:8.3f} | "
            f"Status={decision.status:7s} | "
            f"Risk={decision.risk_level:6s} | "
            f"Action={decision.action}"
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_demo()