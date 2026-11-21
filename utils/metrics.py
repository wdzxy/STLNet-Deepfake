import numpy as np


def binary_auc(labels, scores):
    """Mann-Whitney ROC AUC with average ranks for tied scores; None for one class."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUC requires equal one-dimensional labels and finite scores")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("AUC labels must be binary")
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels == 1].sum() - positive*(positive+1)/2) / (positive*negative))


def binary_metrics(labels, probabilities, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    labels = np.asarray(labels)
    scores = np.asarray(probabilities)
    if not len(labels):
        raise ValueError("Cannot evaluate an empty dataset")
    predictions = scores >= threshold
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    false_negative = int(np.sum(~predictions & (labels == 1)))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"accuracy": float(np.mean(predictions == labels)), "auc": binary_auc(labels, scores),
            "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
