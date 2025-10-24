"""Reusable helpers for xGoal modeling notebooks."""

from xgoal_tutor.modeling.constants import (
    GOAL_HALF_WIDTH_SB,
    GOAL_Y_CENTER_SB,
    PITCH_LENGTH_SB,
    PITCH_WIDTH_SB,
)
from xgoal_tutor.modeling.data import load_wide_df
from xgoal_tutor.modeling.evaluation import (
    compute_binary_classification_metrics,
    plot_roc_curve,
)
from xgoal_tutor.modeling.feature_engineering import build_feature_matrix
from xgoal_tutor.modeling.preprocessing import prepare_shot_dataframe
from xgoal_tutor.modeling.splitting import grouped_train_test_split

__all__ = [
    "PITCH_LENGTH_SB",
    "PITCH_WIDTH_SB",
    "GOAL_Y_CENTER_SB",
    "GOAL_HALF_WIDTH_SB",
    "load_wide_df",
    "prepare_shot_dataframe",
    "build_feature_matrix",
    "grouped_train_test_split",
    "compute_binary_classification_metrics",
    "plot_roc_curve",
]
