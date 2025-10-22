#!/usr/bin/env python3
"""Train a logistic-regression xG model using the shared modeling utilities."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgoal_tutor.modeling import (
    build_feature_matrix,
    compute_binary_classification_metrics,
    grouped_train_test_split,
    load_wide_df,
    plot_roc_curve,
    prepare_shot_dataframe,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline logistic regression xG model using StatsBomb data."
    )
    parser.add_argument(
        "--database-path",
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
        help="Random seed controlling the grouped split and model initialization.",
    )
    parser.add_argument(
        "--no-calibration",
        dest="calibrate",
        action="store_false",
        help="Skip probability calibration with Platt scaling.",
    )
    parser.set_defaults(calibrate=True)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Optional path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--roc-path",
        type=Path,
        help="Optional path to save a ROC curve plot for the selected probabilities.",
    )
    parser.add_argument(
        "--load-data-from-csv",
        action="store_true",
        help="Load precomputed train/test feature CSVs instead of rebuilding from the raw database.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to the precomputed training features CSV (used with --load-data-from-csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to the precomputed test features CSV (used with --load-data-from-csv).",
    )
    return parser.parse_args()


def build_model(seed: int) -> Any:
    """Construct the sklearn pipeline for the baseline logistic regression."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=seed),
    )


def ensure_parent_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_splits_from_csv(
    train_path: Path, test_path: Path
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Load precomputed train/test splits from CSV files."""

    if not train_path.exists():
        logger.error("Training CSV not found at %s", train_path)
        return None
    if not test_path.exists():
        logger.error("Test CSV not found at %s", test_path)
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "target" not in train_df.columns:
        logger.error("Training CSV %s is missing the 'target' column", train_path)
        return None
    if "target" not in test_df.columns:
        logger.error("Test CSV %s is missing the 'target' column", test_path)
        return None

    metadata_cols = [col for col in ["shot_id", "match_id", "target"] if col in train_df.columns]
    train_y = train_df["target"].astype(int)
    train_X = train_df.drop(columns=metadata_cols, errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )

    metadata_cols_test = [col for col in ["shot_id", "match_id", "target"] if col in test_df.columns]
    test_y = test_df["target"].astype(int)
    test_X = test_df.drop(columns=metadata_cols_test, errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )

    test_X = test_X.reindex(columns=train_X.columns, fill_value=np.nan)

    return train_X, test_X, train_y, test_y


def build_splits_from_raw(
    db_path: Path,
    *,
    use_materialized: bool,
    train_fraction: float,
    seed: int,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Load raw data, engineer features, and perform the grouped split."""

    df = load_wide_df(db_path, use_materialized=use_materialized)
    if df is None or df.empty:
        logger.error("No data available at %s.", db_path)
        return None

    data, y = prepare_shot_dataframe(df)
    if len(data) == 0:
        logger.error("No shots remain after filtering.")
        return None

    X = build_feature_matrix(data)
    groups = data["match_id"] if "match_id" in data.columns else None
    train_idx, test_idx = grouped_train_test_split(
        X,
        y,
        groups,
        train_fraction=train_fraction,
        seed=seed,
    )

    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    return X_tr, X_te, y_tr, y_te


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    args = parse_args()

    np.random.seed(args.seed)

    if args.load_data_from_csv:
        splits = load_splits_from_csv(args.train_csv, args.test_csv)
        if splits is None:
            return 1
        X_tr, X_te, y_tr, y_te = splits
        logger.info(
            "Loaded precomputed features from %s and %s", args.train_csv, args.test_csv
        )
    else:
        splits = build_splits_from_raw(
            args.database_path,
            use_materialized=args.use_materialized,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        if splits is None:
            return 1
        X_tr, X_te, y_tr, y_te = splits
        logger.info("Constructed features from raw data at %s", args.database_path)

    test_pos = float(y_te.mean()) if len(y_te) else float("nan")
    logger.info(
        "train=%d (pos=%.3f), test=%d (pos=%.3f)",
        len(X_tr),
        float(y_tr.mean()),
        len(X_te),
        test_pos,
    )

    if len(X_tr) == 0 or np.unique(y_tr).size < 2:
        logger.error("Not enough training data to fit the model.")
        return 1

    base_model = build_model(args.seed)
    base_model.fit(X_tr, y_tr)

    metrics: Dict[str, Dict[str, float]] = {}

    if len(X_te) > 0:
        proba_raw = base_model.predict_proba(X_te)[:, 1]
        metrics["uncalibrated"] = compute_binary_classification_metrics(y_te, proba_raw)
    else:
        proba_raw = np.array([])
        metrics["uncalibrated"] = {"auc": float("nan"), "logloss": float("nan"), "brier": float("nan")}

    selected_proba = proba_raw

    if args.calibrate:
        calibrator = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
        calibrator.fit(X_tr, y_tr)
        if len(X_te) > 0:
            proba_cal = calibrator.predict_proba(X_te)[:, 1]
            metrics["calibrated"] = compute_binary_classification_metrics(y_te, proba_cal)
            selected_proba = proba_cal
        else:
            metrics["calibrated"] = {
                "auc": float("nan"),
                "logloss": float("nan"),
                "brier": float("nan"),
            }
            selected_proba = proba_cal = np.array([])

    metrics_text = json.dumps(metrics, indent=2, sort_keys=True)
    logger.info("Evaluation metrics:\n%s", metrics_text)

    if args.metrics_path is not None:
        ensure_parent_dir(args.metrics_path)
        with args.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, sort_keys=True)
        logger.info("Saved metrics to %s", args.metrics_path)

    if args.roc_path is not None and len(selected_proba) > 0 and len(np.unique(y_te)) >= 2:
        ax = plot_roc_curve(y_te, selected_proba)
        ensure_parent_dir(args.roc_path)
        ax.figure.savefig(args.roc_path, bbox_inches="tight")
        plt.close(ax.figure)
        logger.info("Saved ROC curve to %s", args.roc_path)
    elif args.roc_path is not None:
        logger.warning("Skipping ROC curve plot due to insufficient test data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
