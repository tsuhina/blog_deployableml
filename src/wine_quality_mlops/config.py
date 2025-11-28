"""Configuration management with Pydantic and YAML.

This module provides a type-safe configuration system that loads settings from
config.yaml and exposes them as nested attributes with IDE autocomplete support.

Usage:
    from wine_quality_mlops.config import config

    # Access nested configuration
    data_path = config.data.path
    n_trials = config.optimization.n_trials
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data loading and splitting configuration."""

    path: str = Field(description="Path to the wine quality CSV file")
    inference_path: str = Field(description="Path to inference data CSV")
    target_column: str = Field(description="Name of the target column")
    test_size: float = Field(ge=0.0, le=1.0, description="Test set proportion")
    random_state: int = Field(description="Random seed for reproducibility")


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    type: str = Field(description="Model type (e.g., Ridge, Lasso)")
    quality_score_min: int = Field(description="Minimum quality score for target transformation")
    quality_score_max: int = Field(description="Maximum quality score for target transformation")


class OptimizationConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    n_trials: int = Field(gt=0, description="Number of Optuna trials")
    cv_folds: int = Field(gt=1, description="Number of cross-validation folds")
    alpha_min: float = Field(gt=0.0, description="Minimum alpha value for Ridge")
    alpha_max: float = Field(gt=0.0, description="Maximum alpha value for Ridge")
    enable_pruning: bool = Field(
        default=True, description="Enable Optuna pruning to stop unpromising trials early"
    )
    pruner_type: str = Field(
        default="median",
        description="Pruner type: 'median', 'successive_halving', or 'none'",
    )
    n_startup_trials: int = Field(
        default=5,
        gt=0,
        description="Number of startup trials before pruning starts",
    )
    n_warmup_steps: int = Field(
        default=1,
        ge=0,
        description="Number of warmup steps before pruning evaluation",
    )
    sampler_type: str = Field(
        default="tpe",
        description="Sampler type: 'tpe', 'random', or 'cmaes'",
    )
    sampler_seed: Optional[int] = Field(
        default=None,
        description="Seed for sampler. Uses data.random_state if None",
    )


class DataQualityConfig(BaseModel):
    """Data quality test threshold configuration."""

    max_allowed_failures: int = Field(
        ge=0,
        description="Maximum number of test failures allowed before stopping pipeline",
    )
    min_pass_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum pass rate required (0.0-1.0), optional alternative to max_allowed_failures",
    )


class DefaultTagsConfig(BaseModel):
    """Tags used to mark 'active' runs for automatic fallback."""

    prepare_data: str = Field(description="Tag for prepare-data runs")
    training: str = Field(description="Tag for training runs")
    inference_dq: str = Field(description="Tag for inference-dq runs")


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""

    experiment_name: str = Field(description="MLflow experiment name")
    tracking_uri: str = Field(description="MLflow tracking URI")
    default_tags: DefaultTagsConfig = Field(description="Default tags for automatic run discovery")
    model_name: str = Field(
        default="wine_quality_regressor",
        min_length=1,
        description="Name for registered model in Model Registry"
    )


class Config(BaseModel):
    """Root configuration model with nested settings."""

    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    data_quality: DataQualityConfig
    mlflow: MLflowConfig


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Validated Config instance with all settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config structure is invalid.
    """
    if config_path is None:
        # Default to config.yaml in project root
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.yaml in the project root."
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


# Singleton instance - loaded once at import time
config = load_config()
