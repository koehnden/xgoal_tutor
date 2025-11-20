from __future__ import annotations

import math

import numpy as np

from xgoal_tutor.modeling.constants import (
    GOAL_HALF_WIDTH_SB,
    GOAL_Y_CENTER_SB,
)
from xgoal_tutor.modeling.feature_engineering import (
    _distance_sb,
    _opening_angle_deg_sb,
)


def test_distance_basic_cases() -> None:
    x = np.array([120.0, 110.0, 120.0])
    y = np.array([GOAL_Y_CENTER_SB, GOAL_Y_CENTER_SB, GOAL_Y_CENTER_SB + GOAL_HALF_WIDTH_SB])

    d = _distance_sb(x, y)

    assert math.isclose(d[0], 0.0, abs_tol=1e-9)
    assert math.isclose(d[1], 10.0, abs_tol=1e-9)
    assert math.isclose(d[2], GOAL_HALF_WIDTH_SB, rel_tol=1e-9, abs_tol=1e-9)


def test_opening_angle_symmetry_and_monotonicity() -> None:
    # Symmetry around goal centre line
    x = np.array([110.0, 110.0])
    y = np.array([GOAL_Y_CENTER_SB - 10.0, GOAL_Y_CENTER_SB + 10.0])
    angles = _opening_angle_deg_sb(x, y)
    assert math.isclose(angles[0], angles[1], rel_tol=1e-9, abs_tol=1e-9)

    # Angle increases as shooter moves closer to goal on centreline
    x_far = np.array([100.0])
    y_far = np.array([GOAL_Y_CENTER_SB])
    x_near = np.array([110.0])
    y_near = np.array([GOAL_Y_CENTER_SB])
    angle_far = float(_opening_angle_deg_sb(x_far, y_far)[0])
    angle_near = float(_opening_angle_deg_sb(x_near, y_near)[0])
    assert angle_near > angle_far


def test_opening_angle_extreme_case_goal_line_center() -> None:
    angle = float(_opening_angle_deg_sb(np.array([120.0]), np.array([GOAL_Y_CENTER_SB]))[0])
    assert math.isclose(angle, 180.0, abs_tol=1e-9)
