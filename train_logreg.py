#!/usr/bin/env python3
"""Train a logistic-regression xG model using the shared modeling utilities."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    args = parse_args()

    np.random.seed(args.seed)

    db_path = args.db_path
    df = load_wide_df(db_path, use_materialized=args.use_materialized)
    if df is None or df.empty:
        logger.error("No data available at %s.", db_path)
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

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

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
