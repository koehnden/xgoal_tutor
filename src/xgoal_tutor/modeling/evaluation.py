"""Evaluation helpers for xGoal models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def compute_binary_classification_metrics(y_true: Sequence, proba: Sequence) -> dict:
    """Return AUC, log-loss and Brier score for a binary classifier."""

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if y_true.size == 0 or proba.size == 0 or np.unique(y_true).size < 2:
        return {"auc": np.nan, "logloss": np.nan, "brier": np.nan}

    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "logloss": float(log_loss(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def plot_roc_curve(
    y_true: Sequence,
    proba: Sequence,
    ax: plt.Axes | None = None,
    *,
    label: str | None = None,
    chance_curve: bool = True,
    **plot_kwargs,
) -> plt.Axes:
    """Plot the ROC curve for a binary classifier and return the Matplotlib axis.

    Parameters
    ----------
    y_true
        True binary labels.
    proba
        Predicted positive class probabilities.
    ax
        Optional Matplotlib axis to draw the curve on. A new figure and axis are
        created when *ax* is ``None``.
    label
        Optional legend label. Defaults to ``"ROC curve (AUC = <value>)"``.
    chance_curve
        When ``True`` (default), draw the diagonal chance line.
    **plot_kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the ROC curve plot.

    Raises
    ------
    ValueError
        If fewer than two classes are present in *y_true*.
    """

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if y_true.size == 0 or proba.size == 0 or np.unique(y_true).size < 2:
        raise ValueError("ROC curve requires predictions for at least two classes.")

    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_score = roc_auc_score(y_true, proba)

    if ax is None:
        _, ax = plt.subplots()

    curve_label = label or f"ROC curve (AUC = {auc_score:.3f})"
    ax.plot(fpr, tpr, label=curve_label, **plot_kwargs)

    if chance_curve:
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.6", label="Chance")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic",
    )
    ax.legend(loc="lower right")

    return ax
