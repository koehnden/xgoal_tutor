from __future__ import annotations

import math
import numpy as np
import pytest

from xgoal_tutor.llm.move_simulation import (
    goal_biased_headings,
    assign_marking,
    defender_response_options,
    gk_response_options,
    simulate_one_direction,
)


class DummyModel:
    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        return row


def _plateau_features(start_x: float):
    def _features_fn(point, defenders, gk):
        gain = max(0.0, min(0.02, (point[0] - start_x) * 0.02))
        return np.array([[0.3 + gain]])

    return _features_fn


def test_goal_biased_headings_returns_expected_angles():
    headings = goal_biased_headings((90.0, 40.0))

    assert len(headings) == 8
    norms = [math.hypot(h[0], h[1]) for h in headings]
    assert all(abs(n - 1.0) < 1e-6 for n in norms)

    angles = [math.atan2(h[1], h[0]) for h in headings[:5]]
    assert angles[0] == pytest.approx(0.0, abs=1e-3)
    assert angles[1] == pytest.approx(-0.143, abs=1e-3)
    assert angles[2] == pytest.approx(0.143, abs=1e-3)
    assert angles[3] == pytest.approx(0.524, abs=1e-3)
    assert angles[4] == pytest.approx(-0.524, abs=1e-3)


def test_assign_marking_tags_closest_defenders():
    defenders = [(52.5, 40.0), (70.0, 69.0), (40.0, 40.0)]
    teammates = [(52.0, 40.0), (70.5, 70.0)]
    pass_threats = [0.2, 0.2]

    busy = assign_marking(defenders, teammates, pass_threats)

    assert busy == {0: 0, 1: 1}


def test_response_options_produce_clamped_moves():
    defender = (100.0, 40.0)
    shooter = (105.0, 40.0)

    options = defender_response_options(defender, shooter, 1.2)
    assert len(options) == 5
    assert options[0] == pytest.approx((101.2, 40.0))
    assert len({tuple(opt) for opt in options}) == len(options)

    gk = (112.0, 48.0)
    gk_opts = gk_response_options(gk, shooter, 0.72)
    assert gk_opts[0] == pytest.approx((113.7143, 46.5), rel=1e-3, abs=1e-3)
    assert all(113.7 <= opt[0] <= 120.0 for opt in gk_opts)
    assert all(33.5 <= opt[1] <= 46.5 for opt in gk_opts)


def test_simulation_stops_without_gain():
    start = (100.0, 40.0)
    features_fn = _plateau_features(start[0])
    model = DummyModel()

    best_xg, trace, final_point = simulate_one_direction(
        start,
        np.array([1.0, 0.0]),
        [(80.0, 40.0)],
        [],
        [],
        None,
        features_fn,
        model,
        eps_gain=0.015,
        k_max=3,
    )

    assert best_xg == pytest.approx(0.32)
    assert trace == [pytest.approx(0.3), pytest.approx(0.32)]
    assert final_point == pytest.approx((101.0, 40.0))
