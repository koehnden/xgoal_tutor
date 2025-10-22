#!/usr/bin/env python3
"""Train a LightGBM-based xG model using the shared modeling utilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from tqdm.auto import tqdm

from xgoal_tutor.modeling import (
    build_feature_matrix,
    grouped_train_test_split,
    load_wide_df,
    prepare_shot_dataframe,
)
from xgoal_tutor.modeling.evaluation import compute_binary_classification_metrics

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = (SCRIPT_DIR / "../data/xgoal-db.sqlite").resolve()
DEFAULT_TRAIN_CSV = (SCRIPT_DIR / "../data/processed/train.csv").resolve()
DEFAULT_TEST_CSV = (SCRIPT_DIR / "../data/processed/test.csv").resolve()
DEFAULT_METRICS_PATH = (SCRIPT_DIR / "../results/lightgbm_metrics.json").resolve()
DEFAULT_IMPORTANCE_PATH = (SCRIPT_DIR / "../results/lightgbm_feature_importances.json").resolve()
DEFAULT_SHAP_PATH = (SCRIPT_DIR / "../results/lightgbm_shap_values.json").resolve()


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
        description="Train a LightGBM classifier to predict goal probability.",
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
        help="Fraction of data allocated to the training split when building from raw data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling the grouped split and model initialization.",
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
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of boosting rounds for LightGBM.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM boosting.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="Central value used to construct the LightGBM num_leaves tuning grid.",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=40,
        help="Central value used to construct the LightGBM min_data_in_leaf tuning grid.",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=1.0,
        help="L2 regularization strength (lambda_l2) applied to the LightGBM model.",
    )
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=DEFAULT_IMPORTANCE_PATH,
        help="Where to write the feature importances JSON output.",
    )
    parser.add_argument(
        "--shap-path",
        type=Path,
        default=DEFAULT_SHAP_PATH,
        help="Where to write the SHAP values JSON output.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to write the evaluation metrics JSON output.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["shot_id", "match_id"] if col in frame.columns]


def load_splits_from_csv(train_path: Path, test_path: Path) -> Optional[DatasetSplits]:
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
    train_X = train_df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")
    test_X = test_df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reindex(columns=train_X.columns, fill_value=np.nan).reset_index(drop=True)

    train_y = train_df["target"].astype(int).reset_index(drop=True)
    test_y = test_df["target"].astype(int).reset_index(drop=True)

    return DatasetSplits(train_X, test_X, train_y, test_y, train_metadata, test_metadata)


def build_splits_from_raw(
    db_path: Path,
    *,
    use_materialized: bool,
    train_fraction: float,
    seed: int,
) -> Optional[DatasetSplits]:
    df = load_wide_df(db_path, use_materialized=use_materialized)
    if df is None or df.empty:
        logger.error("No data available at %s", db_path)
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

    train_X = X.iloc[train_idx].reset_index(drop=True)
    train_y = y.iloc[train_idx].reset_index(drop=True)

    if len(test_idx) > 0:
        test_X = X.iloc[test_idx].reset_index(drop=True)
        test_y = y.iloc[test_idx].reset_index(drop=True)
    else:
        test_X = pd.DataFrame(columns=X.columns)
        test_y = pd.Series(dtype=train_y.dtype, name=y.name)

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

    return DatasetSplits(train_X, test_X, train_y, test_y, train_metadata, test_metadata)


def build_model(
    args: argparse.Namespace,
    *,
    num_leaves: Optional[int] = None,
    min_data_in_leaf: Optional[int] = None,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=num_leaves if num_leaves is not None else args.num_leaves,
        min_data_in_leaf=(
            min_data_in_leaf if min_data_in_leaf is not None else args.min_data_in_leaf
        ),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        objective="binary",
        n_jobs=-1,
        is_unbalanced=True,
        lambda_l2=args.lambda_l2,
    )


def _safe_nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _binary_log_loss(y_true: pd.Series, proba: np.ndarray) -> float:
    eps = 1e-15
    clipped = np.clip(proba, eps, 1 - eps)
    y_array = np.asarray(y_true, dtype=float)
    return float(-np.mean(y_array * np.log(clipped) + (1 - y_array) * np.log(1 - clipped)))


def _construct_tuning_grid(center: int, *, lower_factor: float, upper_factor: float) -> list[int]:
    candidates = {
        max(2, int(round(center * lower_factor))),
        max(2, int(round(center))),
        max(2, int(round(center * upper_factor))),
    }
    return sorted(candidates)


def _make_parameter_grid(num_leaves_center: int, min_data_center: int) -> list[tuple[int, int]]:
    leaves_grid = _construct_tuning_grid(num_leaves_center, lower_factor=0.5, upper_factor=2.0)
    min_data_grid = _construct_tuning_grid(min_data_center, lower_factor=0.5, upper_factor=2.0)
    return [(num_leaves, min_data) for num_leaves, min_data in product(leaves_grid, min_data_grid)]


def _build_group_folds(groups: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups.isna().any():
        raise ValueError("Group assignments contain missing values, cannot build folds.")

    unique_groups = np.unique(groups.to_numpy())
    if unique_groups.size < 2:
        raise ValueError("At least two distinct groups are required for cross-validation.")

    n_splits = min(n_splits, unique_groups.size)
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_groups, copy=True)
    rng.shuffle(shuffled)
    group_folds = np.array_split(shuffled, n_splits)

    group_array = groups.to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_groups in group_folds:
        mask = np.isin(group_array, fold_groups)
        val_idx = np.flatnonzero(mask)
        train_idx = np.flatnonzero(~mask)
        if val_idx.size == 0 or train_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))

    if len(folds) < 2:
        raise ValueError("Unable to construct at least two non-empty group folds.")

    return folds


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    match_ids: pd.Series,
    args: argparse.Namespace,
) -> dict[str, int]:
    parameter_grid = _make_parameter_grid(args.num_leaves, args.min_data_in_leaf)
    logger.info(
        "Tuning LightGBM over %d combinations for num_leaves=%s and min_data_in_leaf=%s",
        len(parameter_grid),
        sorted({combo[0] for combo in parameter_grid}),
        sorted({combo[1] for combo in parameter_grid}),
    )

    folds = _build_group_folds(match_ids.reset_index(drop=True), n_splits=5, seed=args.seed)

    best_params: dict[str, int] | None = None
    best_score = float("inf")
    best_auc = float("nan")
    results_summary: list[dict[str, float]] = []

    for num_leaves, min_data in tqdm(parameter_grid, desc="Tuning LightGBM", unit="combo"):
        fold_loglosses: list[float] = []
        fold_aucs: list[float] = []
        fold_briers: list[float] = []

        for train_idx, val_idx in folds:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            if np.unique(y_tr).size < 2:
                logger.debug(
                    "Skipping fold during tuning because training data lacks class diversity (num_leaves=%d, min_data_in_leaf=%d).",
                    num_leaves,
                    min_data,
                )
                continue

            candidate_model = build_model(
                args,
                num_leaves=num_leaves,
                min_data_in_leaf=min_data,
            )
            candidate_model.fit(X_tr, y_tr)
            val_proba = candidate_model.predict_proba(X_val)[:, 1]

            fold_loglosses.append(_binary_log_loss(y_val, val_proba))
            metrics = compute_binary_classification_metrics(y_val, val_proba)
            fold_aucs.append(metrics["auc"])
            fold_briers.append(metrics["brier"])

        if not fold_loglosses:
            logger.warning(
                "No valid folds evaluated for num_leaves=%d, min_data_in_leaf=%d; skipping combination.",
                num_leaves,
                min_data,
            )
            continue

        mean_logloss = float(np.mean(fold_loglosses))
        mean_auc = _safe_nanmean(fold_aucs)
        mean_brier = _safe_nanmean(fold_briers)

        results_summary.append(
            {
                "num_leaves": float(num_leaves),
                "min_data_in_leaf": float(min_data),
                "mean_logloss": mean_logloss,
                "mean_auc": mean_auc,
                "mean_brier": mean_brier,
            }
        )

        if mean_logloss < best_score or (
            np.isclose(mean_logloss, best_score) and (np.isnan(best_auc) or mean_auc > best_auc)
        ):
            best_score = mean_logloss
            best_auc = mean_auc
            best_params = {"num_leaves": num_leaves, "min_data_in_leaf": min_data}

    if best_params is None:
        raise RuntimeError("Hyperparameter tuning failed to evaluate any parameter combinations.")

    top_results = sorted(results_summary, key=lambda row: row["mean_logloss"])[:3]
    for rank, result in enumerate(top_results, start=1):
        logger.info(
            "Tuning rank %d: num_leaves=%d, min_data_in_leaf=%d, mean_logloss=%.5f, mean_auc=%s",
            rank,
            int(result["num_leaves"]),
            int(result["min_data_in_leaf"]),
            result["mean_logloss"],
            "nan" if np.isnan(result["mean_auc"]) else f"{result['mean_auc']:.5f}",
        )

    logger.info(
        "Selected best parameters: num_leaves=%d, min_data_in_leaf=%d (mean_logloss=%.5f, mean_auc=%s)",
        best_params["num_leaves"],
        best_params["min_data_in_leaf"],
        best_score,
        "nan" if np.isnan(best_auc) else f"{best_auc:.5f}",
    )

    return best_params


def _select_positive_class(value: np.ndarray | list[float] | float) -> float:
    if isinstance(value, list):
        return float(value[1] if len(value) > 1 else value[0])
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        if value.size == 1:
            return float(value.item())
        return float(value.flat[1])
    return float(value)


def save_feature_importances(model: lgb.LGBMClassifier, path: Path) -> None:
    booster = model.booster_
    feature_names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    records = []
    for name, gain_value, split_value in zip(feature_names, gain, split):
        records.append(
            {
                "feature": name,
                "importance_gain": float(gain_value),
                "importance_split": float(split_value),
            }
        )

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
    logger.info("Saved feature importances to %s", path)


def save_shap_values(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    proba: np.ndarray,
    metadata: Optional[pd.DataFrame],
    path: Path,
) -> None:
    if X.empty:
        logger.warning("Skipping SHAP computation because the evaluation set is empty.")
        return

    explainer = shap.TreeExplainer(model.booster_)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1]) if len(shap_values) > 1 else np.asarray(shap_values[0])
        expected_value = _select_positive_class(explainer.expected_value)
    else:
        shap_array = np.asarray(shap_values)
        expected_value = float(explainer.expected_value)

    payload: dict[str, object] = {
        "feature_names": list(X.columns),
        "expected_value": expected_value,
        "shap_values": shap_array.tolist(),
        "predictions": proba.tolist(),
        "targets": y_true.tolist(),
    }
    if metadata is not None and not metadata.empty:
        payload["metadata"] = metadata.to_dict(orient="list")

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    logger.info("Saved SHAP values to %s", path)


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

    X_tr, X_te, y_tr, y_te, train_meta, test_meta = (
        splits.X_train,
        splits.X_test,
        splits.y_train,
        splits.y_test,
        splits.train_metadata,
        splits.test_metadata,
    )

    if len(X_tr) == 0 or np.unique(y_tr).size < 2:
        logger.error("Not enough training data to fit the model.")
        return 1

    logger.info("Training samples: %d (pos=%.3f)", len(X_tr), float(y_tr.mean()))
    logger.info("Test samples: %d (pos=%.3f)", len(X_te), float(y_te.mean()) if len(y_te) else float("nan"))

    if train_meta is None or "match_id" not in train_meta.columns:
        logger.error(
            "Training metadata must include match_id for grouped cross-validation tuning."
        )
        return 1

    match_ids = train_meta["match_id"].reset_index(drop=True)

    try:
        best_params = tune_hyperparameters(X_tr, y_tr, match_ids, args)
    except (ValueError, RuntimeError) as exc:
        logger.error("Hyperparameter tuning failed: %s", exc)
        return 1

    logger.info(
        "Retraining final model on full training set with tuned parameters: %s",
        best_params,
    )

    model = build_model(
        args,
        num_leaves=best_params["num_leaves"],
        min_data_in_leaf=best_params["min_data_in_leaf"],
    )
    model.fit(X_tr, y_tr)
    logger.info("Trained LightGBM model with %d features", X_tr.shape[1])

    metrics = {}

    train_proba = model.predict_proba(X_tr)[:, 1]
    metrics["train"] = compute_binary_classification_metrics(y_tr, train_proba)

    if len(X_te) > 0 and np.unique(y_te).size >= 2:
        test_proba = model.predict_proba(X_te)[:, 1]
        metrics["test"] = compute_binary_classification_metrics(y_te, test_proba)
    else:
        test_proba = np.array([])
        metrics["test"] = {"auc": float("nan"), "logloss": float("nan"), "brier": float("nan")}

    metrics_text = json.dumps(metrics, indent=2, sort_keys=True)
    logger.info("Evaluation metrics:\n%s", metrics_text)

    ensure_parent_dir(args.metrics_path)
    with args.metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
    logger.info("Saved metrics to %s", args.metrics_path)

    save_feature_importances(model, args.feature_importance_path)

    if len(test_proba) > 0:
        save_shap_values(model, X_te, y_te, test_proba, test_meta, args.shap_path)
    else:
        logger.warning("Skipping SHAP export because no valid test predictions were produced.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
