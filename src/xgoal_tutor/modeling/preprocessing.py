"""Data preparation utilities for xGoal modeling."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def prepare_shot_dataframe(
    df: pd.DataFrame,
    *,
    drop_penalties: bool = True,
    drop_own_goals: bool = True,
    outcome_column: str = "outcome",
    goal_value: str = "Goal",
    drop_columns: Iterable[str] = ("end_x", "end_y", "end_z"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter the raw shots dataframe and build the binary target column."""

    df0 = df.copy()

    mask = pd.Series(True, index=df0.index)
    if drop_penalties and "is_penalty" in df0.columns:
        mask &= ~df0["is_penalty"].astype(bool)
    if drop_own_goals and "is_own_goal" in df0.columns:
        mask &= ~df0["is_own_goal"].astype(bool)

    filtered = df0.loc[mask].copy()
    y = (filtered[outcome_column] == goal_value).astype(int)

    if drop_columns:
        filtered = filtered.drop(columns=list(drop_columns), errors="ignore")

    return filtered, y
