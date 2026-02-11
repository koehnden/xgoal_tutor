"""Train the logistic regression baseline and export metrics/coefs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xgoal_tutor.model.logistic import (
    LogisticRegressionResult,
    save_coefficients,
    save_metrics,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="Path to the feature table (CSV or Parquet)")
    parser.add_argument(
        "--target",
        default="is_goal",
        help="Name of the target column in the feature table (default: is_goal)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where outputs (JSON) will be stored",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to allocate to the test split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Number of gradient-descent iterations to run",
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="logreg_metrics.json",
        help="Filename for the exported metrics JSON",
    )
    parser.add_argument(
        "--coefficients-filename",
        type=str,
        default="logreg_coefficients.json",
        help="Filename for the exported coefficients JSON",
    )
    return parser.parse_args()


def _load_features(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{suffix}' for features table")


def main() -> None:
    args = parse_args()
    features = _load_features(args.features)

    result: LogisticRegressionResult = train_logistic_regression(
        features,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / args.metrics_filename
    coeffs_path = output_dir / args.coefficients_filename

    save_metrics(result.metrics, metrics_path)
    save_coefficients(result.coefficients, coeffs_path)


if __name__ == "__main__":
    main()
