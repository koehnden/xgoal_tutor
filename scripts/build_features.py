#!/usr/bin/env python3
"""Generate train/test feature CSVs for xGoal modeling experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from xgoal_tutor.modeling import (
    build_feature_matrix,
    grouped_train_test_split,
    load_wide_df,
    prepare_shot_dataframe,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/test feature CSVs for the baseline xGoal dataset.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/xgoal-db.sqlite"),
        help="Path to the StatsBomb SQLite database export.",
    )
    parser.add_argument(
        "--no-materialized",
        dest="use_materialized",
        action="store_false",
        help="Disable use of the materialized shots_wide table if present.",
    )
    parser.set_defaults(use_materialized=True)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of samples to allocate to the training split (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling the grouped split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where train.csv and test.csv will be written.",
    )
    return parser.parse_args()


def _attach_metadata(
    features: pd.DataFrame,
    source: pd.DataFrame,
    indices: Sequence[int],
    target: pd.Series,
) -> pd.DataFrame:
    """Combine feature matrix with metadata and target column."""

    frame = features.iloc[indices].copy()

    for column in ["shot_id", "match_id"]:
        if column in source.columns:
            frame.insert(0, column, source.iloc[indices][column].to_numpy())

    frame["target"] = target.iloc[indices].to_numpy()
    return frame


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    np.random.seed(args.seed)

    df = load_wide_df(args.db_path, use_materialized=args.use_materialized)
    if df is None or df.empty:
        logger.error("No data available at %s.", args.db_path)
        return 1

    data, y = prepare_shot_dataframe(df)
    if len(data) == 0:
        logger.error("No shots remain after filtering.")
        return 1

    X = build_feature_matrix(data)
    groups = data["match_id"] if "match_id" in data.columns else None
    train_idx, test_idx = grouped_train_test_split(
        X,
        y,
        groups,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    if len(train_idx) == 0:
        logger.error("Training split is empty; aborting feature export.")
        return 1

    train_frame = _attach_metadata(X, data, train_idx, y)
    test_frame = _attach_metadata(X, data, test_idx, y) if len(test_idx) else pd.DataFrame(columns=train_frame.columns)

    ensure_dir(args.output_dir)
    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test.csv"

    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    logger.info(
        "Saved %d training rows and %d test rows to %s",
        len(train_frame),
        len(test_frame),
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
