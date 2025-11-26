"""Model building, training, and optimization.

This module contains all logic for creating sklearn pipelines, optimizing
hyperparameters with Optuna, training models, and evaluating performance.
"""

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from wine_quality_mlops.config import config


class ClippedMinMaxScaler(BaseEstimator, TransformerMixin):
    """MinMaxScaler that clips values during inverse_transform to guarantee bounds.

    This transformer wraps MinMaxScaler and ensures predictions stay within
    the specified range by clipping during inverse_transform.
    """

    def __init__(self, min_val: float, max_val: float):
        """Initialize clipper with min and max bounds.

        Args:
            min_val: Minimum value for clipping
            max_val: Maximum value for clipping
        """
        self.min_val = min_val
        self.max_val = max_val
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        """Fit the underlying MinMaxScaler."""
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """Transform using underlying MinMaxScaler."""
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        """Inverse transform and clip to guarantee bounds."""
        transformed = self.scaler.inverse_transform(X)
        return np.clip(transformed, self.min_val, self.max_val)


def build_pipeline(regressor: Any) -> Pipeline:
    """Create sklearn pipeline with preprocessing and target transformation.

    Includes imputation, scaling, and target transformation to constrain predictions.
    """
    # Create and fit target transformer to valid quality score range
    target_transformer = ClippedMinMaxScaler(
        min_val=config.model.quality_score_min,
        max_val=config.model.quality_score_max
    )
    target_transformer.fit([[config.model.quality_score_min], [config.model.quality_score_max]])

    # Build pipeline
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    transformer=target_transformer,
                ),
            ),
        ]
    )

    return pipeline


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> dict[str, Any]:
    """Optimize hyperparameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: evaluate alpha parameter via cross-validation."""
        alpha = trial.suggest_float(
            "alpha",
            config.optimization.alpha_min,
            config.optimization.alpha_max,
            log=True,
        )
        regressor = Ridge(alpha=alpha, random_state=config.data.random_state)
        pipeline = build_pipeline(regressor)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=config.optimization.cv_folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        return -cv_scores.mean()

    study = optuna.create_study(direction="minimize", study_name="wine_quality_regression")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best CV MAE: {study.best_value:.4f}")

    return study.best_params


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
) -> Pipeline:
    """Train final model with best hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets
        best_params: Dictionary of hyperparameters from optimization

    Returns:
        Trained sklearn pipeline
    """
    regressor = Ridge(
        alpha=best_params["alpha"],
        random_state=config.data.random_state,
    )

    pipeline = build_pipeline(regressor)
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate model performance on a dataset.

    Args:
        pipeline: Trained model pipeline
        X: Features
        y: True targets

    Returns:
        Dictionary containing MAE, MSE, RMSE, and R² metrics
    """
    y_pred = pipeline.predict(X)

    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred),
    }

    return metrics
