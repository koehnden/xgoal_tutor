from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgoal_tutor.model.logistic import (
    LogisticRegressionResult,
    save_coefficients,
    save_metrics,
    train_logistic_regression,
)
from scripts.train_logreg import _load_features


def _make_dataset(n_samples: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    weights = np.array([0.5, -0.8, 0.3, 0.1, -0.2])
    bias = -0.1
    X = rng.normal(size=(n_samples, weights.size))
    logits = X @ weights + bias
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probabilities)
    columns = [f"feature_{idx}" for idx in range(weights.size)]
    frame = pd.DataFrame(X, columns=columns)
    frame["is_goal"] = y
    return frame


def test_train_logistic_regression_returns_metrics_and_coefficients() -> None:
    frame = _make_dataset()
    result = train_logistic_regression(frame, target="is_goal", random_state=0)

    assert isinstance(result, LogisticRegressionResult)
    assert set(result.metrics.keys()) == {"train", "test"}
    for split in ("train", "test"):
        metrics = result.metrics[split]
        assert set(metrics.keys()) == {"auc", "brier", "logloss"}
        assert 0.0 <= metrics["auc"] <= 1.0
        assert metrics["logloss"] >= 0.0
        assert 0.0 <= metrics["brier"] <= 1.0

    # There should be one coefficient per feature plus an intercept term.
    assert len(result.coefficients) == 6
    names = {entry["feature"] for entry in result.coefficients}
    assert "intercept" in names
    assert {f"feature_{idx}" for idx in range(5)}.issubset(names)


def test_save_functions_emit_json(tmp_path: Path) -> None:
    frame = _make_dataset()
    result = train_logistic_regression(frame, target="is_goal", random_state=0)

    metrics_path = tmp_path / "metrics.json"
    coeffs_path = tmp_path / "coefficients.json"

    save_metrics(result.metrics, metrics_path)
    save_coefficients(result.coefficients, coeffs_path)

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_coeffs = json.loads(coeffs_path.read_text(encoding="utf-8"))

    assert saved_metrics["train"]["auc"] == result.metrics["train"]["auc"]
    assert saved_coeffs[-1]["feature"] == "intercept"


def test_cli_loader_supports_csv_and_parquet(tmp_path: Path) -> None:
    frame = _make_dataset()
    csv_path = tmp_path / "features.csv"
    parquet_path = tmp_path / "features.parquet"

    frame.to_csv(csv_path, index=False)
    try:
        import pyarrow  # type: ignore  # noqa: F401

        frame.to_parquet(parquet_path, index=False)
        parquet_available = True
    except ImportError:
        parquet_available = False

    loaded_csv = _load_features(csv_path)
    pd.testing.assert_frame_equal(loaded_csv, frame)

    if parquet_available:
        loaded_parquet = _load_features(parquet_path)
        pd.testing.assert_frame_equal(loaded_parquet, frame)
