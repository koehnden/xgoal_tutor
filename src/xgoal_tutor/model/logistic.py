"""Logistic regression training helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LogisticRegressionResult:
    """Bundle the outputs of a training run."""

    metrics: dict[str, dict[str, float]]
    coefficients: list[dict[str, float]]


def _ensure_dataframe(data: pd.DataFrame | Iterable[dict[str, float]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def train_logistic_regression(
    data: pd.DataFrame | Iterable[dict[str, float]],
    *,
    target: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
    max_iter: int = 1000,
) -> LogisticRegressionResult:
    """Train a logistic regression model and compute evaluation metrics.

    Parameters
    ----------
    data:
        Feature matrix with the training target. Can be a DataFrame or iterable
        of dictionaries that will be converted into a DataFrame.
    target:
        The column containing the binary target variable.
    test_size:
        Proportion of the dataset to include in the test split.
    random_state:
        Seed to make the train/test split reproducible.
    max_iter:
        Number of gradient-descent iterations to perform during training.
    """

    frame = _ensure_dataframe(data)
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found in training data")

    y = frame[target].astype(int)
    X = frame.drop(columns=[target])

    feature_names = list(X.columns)
    if not feature_names:
        raise ValueError("No feature columns remain after dropping the target column")
    X_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    train_idx, test_idx = _stratified_split(y_values, test_size=test_size, random_state=random_state)

    X_train = X_values[train_idx]
    X_test = X_values[test_idx]
    y_train = y_values[train_idx]
    y_test = y_values[test_idx]

    X_train_scaled, means, scales = _standardize(X_train)
    X_test_scaled = _apply_standardization(X_test, means, scales)

    weights_scaled, bias_scaled = _fit_logistic(
        X_train_scaled, y_train, learning_rate=0.05, epochs=max_iter
    )

    prob_train = _sigmoid(X_train_scaled @ weights_scaled + bias_scaled)
    prob_test = _sigmoid(X_test_scaled @ weights_scaled + bias_scaled)

    metrics = {
        "train": _compute_metrics(y_train, prob_train),
        "test": _compute_metrics(y_test, prob_test),
    }

    weights, bias = _to_original_scale(weights_scaled, bias_scaled, means, scales)
    coefficients = _format_coefficients(weights, bias, feature_names)

    return LogisticRegressionResult(metrics=metrics, coefficients=coefficients)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    auc = float(_auc_score(y_true, y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = float(_log_loss(y_true, y_prob))
    return {"auc": auc, "brier": brier, "logloss": logloss}


def _format_coefficients(
    weights: np.ndarray, bias: float, feature_names: list[str]
) -> list[dict[str, float]]:
    coefficients = [
        {"feature": name, "coefficient": float(weight)}
        for name, weight in zip(feature_names, weights, strict=True)
    ]
    coefficients.append({"feature": "intercept", "coefficient": float(bias)})
    return coefficients


def save_metrics(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    """Persist metrics to disk as JSON."""

    output_path = Path(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_coefficients(coefficients: list[dict[str, float]], output_path: Path) -> None:
    """Persist coefficient table to disk as JSON."""

    output_path = Path(output_path)
    output_path.write_text(json.dumps(coefficients, indent=2), encoding="utf-8")


def _stratified_split(
    y: np.ndarray, *, test_size: float, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if not {0.0, 1.0}.issuperset(set(np.unique(y))):
        raise ValueError("Target array must contain only 0 and 1 values")

    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError("Need at least two samples of each class to create a split")

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_test_pos = max(1, int(round(len(pos_idx) * test_size)))
    n_test_neg = max(1, int(round(len(neg_idx) * test_size)))

    if n_test_pos >= len(pos_idx):
        n_test_pos = len(pos_idx) - 1
    if n_test_neg >= len(neg_idx):
        n_test_neg = len(neg_idx) - 1

    test_idx = np.concatenate([pos_idx[:n_test_pos], neg_idx[:n_test_neg]])
    train_idx = np.concatenate([pos_idx[n_test_pos:], neg_idx[n_test_neg:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return train_idx, test_idx


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales == 0.0] = 1.0
    X_scaled = (X - means) / scales
    return X_scaled, means, scales


def _apply_standardization(
    X: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    return (X - means) / scales


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2_penalty: float = 1e-4,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0

    for _ in range(epochs):
        logits = X @ weights + bias
        predictions = _sigmoid(logits)
        errors = predictions - y

        grad_w = (X.T @ errors) / X.shape[0] + l2_penalty * weights
        grad_b = errors.mean()

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _to_original_scale(
    weights: np.ndarray,
    bias: float,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, float]:
    weights_original = weights / scales
    bias_original = bias - np.dot(weights_original, means)
    return weights_original, bias_original


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def _auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = _average_ranks(y_prob[order])

    pos_mask = y_true == 1.0
    neg_mask = y_true == 0.0

    n_pos = np.count_nonzero(pos_mask)
    n_neg = np.count_nonzero(neg_mask)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must be present to compute AUC")

    rank_sum = ranks[pos_mask].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_ranks(sorted_values: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(sorted_values, dtype=float)
    i = 0
    length = len(sorted_values)
    while i < length:
        j = i
        while j + 1 < length and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    return ranks
