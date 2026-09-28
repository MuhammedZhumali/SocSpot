"""Metrics are computed from explicit ground truth, never search query labels."""
import numpy as np
from .video import LABELS


def classification_metrics(targets, predictions):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, pred in zip(targets, predictions, strict=True):
        matrix[int(truth), int(pred)] += 1
    classes = {}
    for i, label in enumerate(LABELS):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        classes[label] = {"precision": precision, "recall": recall,
                          "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                          "support": support}
    return {"accuracy": float(np.trace(matrix)/matrix.sum()) if matrix.sum() else 0.0,
            "macro_f1": float(np.mean([r["f1"] for r in classes.values()])),
            "balanced_accuracy": float(np.mean([r["recall"] for r in classes.values()])),
            "classes": classes, "confusion_matrix": matrix.tolist(), "label_order": LABELS,
            "confusion_matrix_axes": "rows=true, columns=predicted"}


def select_thresholds(targets, probabilities):
    """Validation only. Prefer precision using F0.5; scores are not calibrated."""
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    thresholds = {}
    for i, label in enumerate(LABELS[:2]):
        choices = []
        for threshold in np.arange(.50, .951, .05):
            selected = (probabilities.argmax(1) == i) & (probabilities[:, i] >= threshold)
            tp = int(((targets == i) & selected).sum())
            fp = int(((targets != i) & selected).sum())
            fn = int(((targets == i) & ~selected).sum())
            f05 = 1.25*tp/(1.25*tp+.25*fn+fp) if tp else 0.0
            choices.append((f05, float(threshold)))
        thresholds[label] = round(max(choices)[1], 2)
    return thresholds
