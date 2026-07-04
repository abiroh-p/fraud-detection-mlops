"""
Model definition and wrapper for the fraud detection pipeline.

Why a wrapper around XGBoost?
- Isolates the rest of the codebase from XGBoost's API
- If we switch to LightGBM or a neural network later,
  only this file changes — nothing else does
- Adds pipeline steps (scaling, encoding) cleanly
- Enforces consistent feature column ordering

This follows the Adapter design pattern — wrapping a third-party
library behind our own interface.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config_loader import load_config
from src.utils.exceptions import ModelLoadError, ModelPredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("model_config.yaml")
_model_cfg = _cfg["model"]
_feature_cfg = _cfg["features"]

# The canonical ordered list of features the model expects
# This order must never change between training and serving
FEATURE_COLUMNS: list[str] = _feature_cfg["numeric_features"]


def build_pipeline() -> Pipeline:
    """
    Build the full sklearn Pipeline: scaler → XGBoost classifier.

    Why a Pipeline instead of just XGBoost directly?

    A Pipeline chains preprocessing and model into one object.
    When you call pipeline.predict(X), it automatically:
      1. Scales X with StandardScaler
      2. Passes scaled X to XGBoost

    This guarantees that the SAME scaling applied during training
    is applied during inference. Without a pipeline, a common
    production bug is training with scaled features but serving
    with unscaled features — silent performance degradation.

    Why StandardScaler with XGBoost?
    XGBoost is tree-based and technically does not need scaling.
    However, scaling helps with:
    - Faster convergence
    - Better numerical stability
    - Easier addition of linear models later

    Returns:
        Unfitted sklearn Pipeline.
    """
    params = _model_cfg["hyperparameters"]

    classifier = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric=params["eval_metric"],
        random_state=params["random_state"],
        # Use histogram-based tree method — much faster on large datasets
        tree_method="hist",
        # Suppress XGBoost's own verbose output — we use our logger
        verbosity=0,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    logger.info(
        "Built pipeline: StandardScaler → XGBClassifier(n_estimators=%d, "
        "max_depth=%d, scale_pos_weight=%d)",
        params["n_estimators"],
        params["max_depth"],
        params["scale_pos_weight"],
    )

    return pipeline


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and order the feature matrix from a raw DataFrame.

    This function enforces the canonical column order defined in
    model_config.yaml. Even if the DataFrame has extra columns
    (like transaction_id, user_id), only the model features are
    selected and always in the same order.

    Column order consistency between training and serving is one
    of the most common sources of silent production bugs in ML.

    Args:
        df: DataFrame containing at minimum all FEATURE_COLUMNS.

    Returns:
        DataFrame with exactly FEATURE_COLUMNS in canonical order.

    Raises:
        ModelPredictionError: If any required feature column is missing.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ModelPredictionError(
            f"Feature matrix is missing required columns: {missing}. "
            f"Check that the feature engineering pipeline is up to date."
        )

    return df[FEATURE_COLUMNS]


def get_target_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract the target column from a DataFrame.

    Args:
        df: DataFrame containing the 'is_fraud' column.

    Returns:
        Binary Series of fraud labels.

    Raises:
        ModelPredictionError: If target column is missing.
    """
    target_col = _feature_cfg["target"]

    if target_col not in df.columns:
        raise ModelPredictionError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    return df[target_col].astype(int)
