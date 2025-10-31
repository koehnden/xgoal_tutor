"""Default logistic regression parameters for asynchronous jobs."""

from __future__ import annotations

from xgoal_tutor.api.models import LogisticRegressionModel


_DEFAULT_COEFFICIENTS = {
    "dist_sb": -0.09,
    "angle_deg_sb": 0.045,
    "is_set_piece": 0.25,
    "is_corner": -0.35,
    "is_free_kick": 0.28,
    "first_time": 0.05,
    "under_pressure": -0.12,
    "is_header": -0.4,
    "gk_depth_sb": -0.015,
    "gk_offset_sb": -0.02,
    "ff_opponents": -0.03,
}

_DEFAULT_INTERCEPT = -1.8


def load_default_model() -> LogisticRegressionModel:
    """Return a logistic regression model usable for offline summaries."""

    return LogisticRegressionModel(intercept=_DEFAULT_INTERCEPT, coefficients=_DEFAULT_COEFFICIENTS)
