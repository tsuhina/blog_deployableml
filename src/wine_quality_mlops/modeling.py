"""Model building, training, and optimization.

This module contains all logic for creating sklearn pipelines, optimizing
hyperparameters with Optuna, training models, and evaluating performance.
"""

import logging
from typing import Any, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from optuna.trial import TrialState
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

logger = logging.getLogger(__name__)


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
    """Optimize hyperparameters using Optuna with pruning and configurable sampler.

    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials

    Returns:
        Dictionary containing best hyperparameters found during optimization.

    Raises:
        ValueError: If pruner_type or sampler_type configuration is invalid.
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: evaluate alpha parameter via cross-validation.

        Implements pruning support by reporting intermediate values and checking
        if trial should be pruned.

        Args:
            trial: Optuna trial object

        Returns:
            Mean absolute error (negative of CV score mean)

        Raises:
            optuna.TrialPruned: If trial should be pruned early
        """
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
        mean_score = -cv_scores.mean()

        # Report intermediate value for pruning
        trial.report(mean_score, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_score

    # Create pruner based on configuration
    pruner: Optional[Any] = None
    if config.optimization.enable_pruning:
        pruner_type = config.optimization.pruner_type.lower()
        if pruner_type == "median":
            pruner = MedianPruner(
                n_startup_trials=config.optimization.n_startup_trials,
                n_warmup_steps=config.optimization.n_warmup_steps,
            )
            logger.info(
                f"Using MedianPruner (startup_trials={config.optimization.n_startup_trials}, "
                f"warmup_steps={config.optimization.n_warmup_steps})"
            )
        elif pruner_type == "successive_halving":
            pruner = SuccessiveHalvingPruner()
            logger.info("Using SuccessiveHalvingPruner")
        elif pruner_type == "none":
            pruner = None
            logger.info("Pruning disabled")
        else:
            raise ValueError(
                f"Invalid pruner_type: {config.optimization.pruner_type}. "
                f"Must be 'median', 'successive_halving', or 'none'"
            )
    else:
        logger.info("Pruning disabled via enable_pruning=False")

    # Create sampler based on configuration
    sampler_seed = (
        config.optimization.sampler_seed
        if config.optimization.sampler_seed is not None
        else config.data.random_state
    )
    sampler_type = config.optimization.sampler_type.lower()

    if sampler_type == "tpe":
        sampler = TPESampler(seed=sampler_seed)
        logger.info(f"Using TPESampler (seed={sampler_seed})")
    elif sampler_type == "random":
        sampler = RandomSampler(seed=sampler_seed)
        logger.info(f"Using RandomSampler (seed={sampler_seed})")
    elif sampler_type == "cmaes":
        sampler = CmaEsSampler(seed=sampler_seed)
        logger.info(f"Using CmaEsSampler (seed={sampler_seed})")
    else:
        raise ValueError(
            f"Invalid sampler_type: {config.optimization.sampler_type}. "
            f"Must be 'tpe', 'random', or 'cmaes'"
        )

    # Create study with configured sampler and pruner
    study = optuna.create_study(
        direction="minimize",
        study_name="wine_quality_regression",
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Log results
    finished_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info(f"Optimization completed:")
    logger.info(f"  Best hyperparameters: {study.best_params}")
    logger.info(f"  Best CV MAE: {study.best_value:.4f}")
    logger.info(f"  Finished trials: {len(finished_trials)}")
    logger.info(f"  Pruned trials: {len(pruned_trials)}")
    logger.info(f"  Total trials: {len(study.trials)}")

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
