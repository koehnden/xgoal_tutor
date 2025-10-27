#!/usr/bin/env python3
"""Train a logistic-regression xG model using the shared modeling utilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    grouped_train_test_split,
    load_wide_df,
    plot_roc_curve,
    prepare_shot_dataframe,
)
from xgoal_tutor.modeling.evaluation import compute_binary_classification_metrics


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = (SCRIPT_DIR / "../data/xgoal-db.sqlite").resolve()
DEFAULT_TRAIN_CSV = (SCRIPT_DIR / "../data/processed/train.csv").resolve()
DEFAULT_TEST_CSV = (SCRIPT_DIR / "../data/processed/test.csv").resolve()
DEFAULT_METRICS_PATH = (SCRIPT_DIR / "../results/logreg_metrics.json").resolve()
DEFAULT_COEFFICIENTS_PATH = (SCRIPT_DIR / "../results/logreg_coefficients.json").resolve()


@dataclass
class DatasetSplits:
    """Container for feature matrices, targets and optional metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_metadata: Optional[pd.DataFrame]
    test_metadata: Optional[pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline logistic regression xG model using StatsBomb data."
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
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
        default=DEFAULT_METRICS_PATH,
        help=(
            "Path to save evaluation metrics as JSON (default: results/logreg_metrics.json)."
        ),
    )
    parser.add_argument(
        "--coefficients-path",
        type=Path,
        default=DEFAULT_COEFFICIENTS_PATH,
        help=(
            "Path to save model coefficients as JSON "
            "(default: results/logreg_coefficients.json)."
        ),
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
        default=DEFAULT_TRAIN_CSV,
        help="Path to the precomputed training features CSV (used with --load-data-from-csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
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


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["shot_id", "match_id"] if col in frame.columns]


def load_splits_from_csv(train_path: Path, test_path: Path) -> Optional[DatasetSplits]:
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

    metadata_cols = _metadata_columns(train_df)
    train_metadata = train_df[metadata_cols].reset_index(drop=True) if metadata_cols else None
    test_metadata = test_df[metadata_cols].reset_index(drop=True) if metadata_cols else None

    drop_cols = metadata_cols + ["target"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")
    X_test = test_df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan).reset_index(drop=True)

    y_train = train_df["target"].astype(int).reset_index(drop=True)
    y_test = test_df["target"].astype(int).reset_index(drop=True)

    return DatasetSplits(X_train, X_test, y_train, y_test, train_metadata, test_metadata)


def build_splits_from_raw(
    db_path: Path,
    *,
    use_materialized: bool,
    train_fraction: float,
    seed: int,
) -> Optional[DatasetSplits]:
    """Load raw data, engineer features, and perform the grouped split."""

    df = load_wide_df(db_path, use_materialized=use_materialized)
    if df is None or df.empty:
        logger.error("No data available at %s.", db_path)
        return None

    data, y = prepare_shot_dataframe(df)
    if len(data) == 0:
        logger.error("No shots remain after filtering.")
        return None

    feature_matrix = build_feature_matrix(data)
    groups = data["match_id"] if "match_id" in data.columns else None
    train_idx, test_idx = grouped_train_test_split(
        feature_matrix,
        y,
        groups,
        train_fraction=train_fraction,
        seed=seed,
    )

    X_train = feature_matrix.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    if len(test_idx) > 0:
        X_test = feature_matrix.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
    else:
        X_test = pd.DataFrame(columns=feature_matrix.columns)
        y_test = pd.Series(dtype=y.dtype, name=y.name)

    metadata_cols = _metadata_columns(data)
    train_metadata = (
        data.iloc[train_idx][metadata_cols].reset_index(drop=True)
        if metadata_cols
        else None
    )
    test_metadata = (
        data.iloc[test_idx][metadata_cols].reset_index(drop=True)
        if metadata_cols and len(test_idx) > 0
        else (pd.DataFrame(columns=metadata_cols) if metadata_cols else None)
    )

    return DatasetSplits(X_train, X_test, y_train, y_test, train_metadata, test_metadata)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    args = parse_args()

    np.random.seed(args.seed)

    if args.load_data_from_csv:
        splits = load_splits_from_csv(args.train_csv, args.test_csv)
        if splits is None:
            return 1
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
        logger.info("Constructed features from raw data at %s", args.database_path)

    X_train, X_test, y_train, y_test = (
        splits.X_train,
        splits.X_test,
        splits.y_train,
        splits.y_test,
    )

    logger.info("Training samples: %d (pos=%.3f)", len(X_train), float(y_train.mean()))
    logger.info(
        "Test samples: %d (pos=%.3f)",
        len(X_test),
        float(y_test.mean()) if len(y_test) else float("nan"),
    )

    if len(X_train) == 0 or np.unique(y_train).size < 2:
        logger.error("Not enough training data to fit the model.")
        return 1

    model = build_model(args.seed)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    if len(X_test) > 0:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics_uncalibrated = compute_binary_classification_metrics(y_test, test_proba)
    else:
        test_proba = np.array([])
        test_metrics_uncalibrated = {
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
        }

    metrics_uncalibrated = {
        "train": compute_binary_classification_metrics(y_train, train_proba),
        "test": test_metrics_uncalibrated,
    }
    logger.info(
        "Uncalibrated evaluation metrics:\n%s",
        json.dumps(metrics_uncalibrated, indent=2, sort_keys=True),
    )

    selected_train_proba = train_proba
    selected_test_proba = test_proba

    metrics = metrics_uncalibrated

    if args.calibrate:
        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        calibrator.fit(X_train, y_train)
        selected_train_proba = calibrator.predict_proba(X_train)[:, 1]
        if len(X_test) > 0:
            selected_test_proba = calibrator.predict_proba(X_test)[:, 1]
        metrics = {
            "train": compute_binary_classification_metrics(y_train, selected_train_proba),
            "test": compute_binary_classification_metrics(y_test, selected_test_proba),
        }
        logger.info(
            "Calibrated evaluation metrics:\n%s",
            json.dumps(metrics, indent=2, sort_keys=True),
        )

    metrics_text = json.dumps(metrics, indent=2, sort_keys=True)
    logger.info("Evaluation metrics:\n%s", metrics_text)

    if args.metrics_path is not None:
        ensure_parent_dir(args.metrics_path)
        with args.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, sort_keys=True)
        logger.info("Saved metrics to %s", args.metrics_path)

    logistic_model = model.named_steps.get("logisticregression")
    if logistic_model is not None:
        coefficients = [
            {"feature": feature, "coefficient": float(coef)}
            for feature, coef in zip(X_train.columns, logistic_model.coef_.ravel())
        ]
        if hasattr(logistic_model, "intercept_"):
            intercept_values = np.atleast_1d(logistic_model.intercept_)
            for idx, intercept_value in enumerate(intercept_values):
                feature_name = "__intercept__" if intercept_values.size == 1 else f"__intercept__{idx}"
                coefficients.append(
                    {
                        "feature": feature_name,
                        "coefficient": float(intercept_value),
                    }
                )
        if args.coefficients_path is not None:
            ensure_parent_dir(args.coefficients_path)
            with args.coefficients_path.open("w", encoding="utf-8") as fp:
                json.dump(coefficients, fp, indent=2)
            logger.info("Saved coefficients to %s", args.coefficients_path)

    if (
        args.roc_path is not None
        and len(selected_test_proba) > 0
        and len(np.unique(y_test)) >= 2
    ):
        ax = plot_roc_curve(y_test, selected_test_proba)
        ensure_parent_dir(args.roc_path)
        ax.figure.savefig(args.roc_path, bbox_inches="tight")
        plt.close(ax.figure)
        logger.info("Saved ROC curve to %s", args.roc_path)
    elif args.roc_path is not None:
        logger.warning("Skipping ROC curve plot due to insufficient test data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
