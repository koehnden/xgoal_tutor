from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from xgoal_tutor.api.models import LogisticRegressionModel
from xgoal_tutor.api.services import calculate_probabilities, format_reason_codes


def _make_model(coeffs: Dict[str, float], intercept: float = 0.0) -> LogisticRegressionModel:
    return LogisticRegressionModel(intercept=intercept, coefficients=coeffs)


def test_calculate_probabilities_alignment_and_values() -> None:
    # Feature frame has an extra column and is missing one; alignment should zero-fill.
    features = pd.DataFrame(
        [
            {"f1": 2.0, "f2": 1.0, "extra": 5.0},
            {"f1": -1.0, "f2": 0.0, "extra": 0.0},
        ]
    )
    model = _make_model({"f1": 0.5, "f2": -1.0, "f_missing": 2.0}, intercept=0.0)

    probs, contrib = calculate_probabilities(features, model)

    # Linear: row0 = 0.5*2 + (-1)*1 + 2*0 = 1 - 1 + 0 = 0 -> sigmoid(0)=0.5
    # Linear: row1 = 0.5*(-1) + (-1)*0 + 2*0 = -0.5 -> sigmoid(-0.5)
    assert math.isclose(float(probs.iloc[0]), 0.5, rel_tol=1e-9, abs_tol=1e-9)
    expected_row1 = 1.0 / (1.0 + math.exp(0.5))
    assert math.isclose(float(probs.iloc[1]), expected_row1, rel_tol=1e-9, abs_tol=1e-9)

    # Contributions should multiply aligned values by coefficients, missing feature zero
    c0 = contrib.iloc[0]
    assert math.isclose(float(c0["f1"]), 1.0, abs_tol=1e-9)
    assert math.isclose(float(c0["f2"]), -1.0, abs_tol=1e-9)
    assert math.isclose(float(c0["f_missing"]), 0.0, abs_tol=1e-9)


def test_format_reason_codes_order_and_values() -> None:
    # Prepare a row and contributions where magnitudes order is clear
    row = pd.Series({"a": 1.0, "b": 2.0, "c": -3.0})
    contributions = pd.Series({"a": 0.1, "b": -0.5, "c": 0.3})
    model = _make_model({"a": 0.1, "b": -0.25, "c": 0.1})

    reasons = format_reason_codes(row, contributions, model, max_reasons=2)

    assert [r.feature for r in reasons] == ["b", "c"]
    assert math.isclose(reasons[0].value, 2.0, abs_tol=1e-9)
    assert math.isclose(reasons[0].coefficient, -0.25, abs_tol=1e-9)
    assert math.isclose(reasons[0].contribution, -0.5, abs_tol=1e-9)
