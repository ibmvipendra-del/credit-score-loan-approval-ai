
import numpy as np


def calculate_auc(y_true, probabilities):

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    positive = probabilities[y_true == 1]
    negative = probabilities[y_true == 0]

    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    # Pairwise ROC-AUC with tie handling.
    comparisons = (
        positive[:, None] > negative[None, :]
    )

    ties = (
        positive[:, None] == negative[None, :]
    )

    return float(
        (comparisons.sum() + 0.5 * ties.sum())
        / (len(positive) * len(negative))
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must match.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    specificity = (
        tn / (tn + fp) if tn + fp else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    auc = calculate_auc(y_true, probabilities)

    brier_score = float(
        np.mean((probabilities - y_true) ** 2)
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "brier_score": brier_score,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }
