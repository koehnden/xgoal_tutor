"""Train/test split helpers for xGoal models."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


logger = logging.getLogger(__name__)


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence | None,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
    max_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/test indices honouring match (group) boundaries."""

    idx = np.arange(len(X))
    if len(idx) == 0:
        return idx, idx

    if groups is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(train_fraction * len(idx))
        return idx[:cut], idx[cut:]

    groups_series = pd.Series(groups).reset_index(drop=True)
    unknown_mask = groups_series.isna() | (groups_series == 0)
    known_idx = idx[~unknown_mask.to_numpy()]
    unknown_idx = idx[unknown_mask.to_numpy()]

    if len(known_idx) == 0:
        logger.warning("All match_ids are missing/zero; reserving them for test only.")
        return np.array([], dtype=int), unknown_idx

    known_groups = groups_series.loc[~unknown_mask]
    if known_groups.nunique() < 2 or len(known_idx) < 5:
        rng = np.random.default_rng(seed)
        rng.shuffle(known_idx)
        cut = int(train_fraction * len(known_idx))
        tr_known = known_idx[:cut]
        te_known = known_idx[cut:]
    else:
        n_splits = min(max_splits, int(known_groups.nunique()))
        gkf = GroupKFold(n_splits=n_splits)
        group_values = known_groups.to_numpy()
        tr_loc, te_loc = next(gkf.split(X.iloc[known_idx], y.iloc[known_idx], group_values))
        tr_known = known_idx[tr_loc]
        te_known = known_idx[te_loc]

    test_idx = np.sort(np.concatenate([te_known, unknown_idx]))
    train_mask = np.ones(len(idx), dtype=bool)
    train_mask[test_idx] = False
    train_idx = idx[train_mask]
    return train_idx.astype(int), test_idx.astype(int)
