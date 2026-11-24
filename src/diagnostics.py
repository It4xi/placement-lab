"""Diagnostics for the fixed model on the reserved synthetic test cohort."""

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix


def probability_diagnostics(y_train, y_test, probability):
    truth = np.asarray(y_test)
    prior = float(np.mean(y_train))
    baseline = np.full(len(truth), prior)
    groups = []
    # Put each observation into exactly one fifth of the probability range.
    bins = np.minimum((probability * 5).astype(int), 4)
    for index in sorted(np.unique(bins)):
        mask = bins == index
        groups.append({
            "band": f"{20*index}-{20*(index+1)}%",
            "students": int(mask.sum()),
            "mean_score": float(probability[mask].mean()),
            "observed_rate": float(truth[mask].mean()),
        })
    thresholds = []
    for cutoff in range(10, 100, 5):
        tn, fp, fn, tp = confusion_matrix(
            truth, probability >= cutoff / 100, labels=[0, 1]
        ).ravel().tolist()
        thresholds.append(dict(cutoff=cutoff, tn=tn, fp=fp, fn=fn, tp=tp))
    return {
        "training_prior": prior,
        "baseline_accuracy": float((truth == int(prior >= 0.5)).mean()),
        "baseline_brier": float(brier_score_loss(truth, baseline)),
        "brier": float(brier_score_loss(truth, probability)),
        "calibration": groups,
        "thresholds": thresholds,
    }


def training_size_comparison(y_true, earlier, expanded, seed, repeats=1000):
    """Compare two fixed models using paired resamples of the same test rows."""
    truth = np.asarray(y_true)
    old_correct = ((earlier >= 0.5) == truth).astype(float)
    new_correct = ((expanded >= 0.5) == truth).astype(float)
    old_error = (earlier - truth) ** 2
    new_error = (expanded - truth) ** 2
    changes = np.column_stack([new_correct - old_correct, old_error - new_error])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((repeats, changes.shape[1]))
    for index in range(repeats):
        rows = rng.integers(0, len(truth), size=len(truth))
        bootstrap[index] = changes[rows].mean(axis=0)
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        "earlier_accuracy": float(old_correct.mean()),
        "expanded_accuracy": float(new_correct.mean()),
        "accuracy_gain_ci95": intervals[:, 0].tolist(),
        "earlier_brier": float(old_error.mean()),
        "expanded_brier": float(new_error.mean()),
        "brier_reduction_ci95": intervals[:, 1].tolist(),
        "bootstrap_repeats": repeats,
    }
