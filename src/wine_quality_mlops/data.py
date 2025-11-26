"""Data loading and preprocessing utilities.

This module handles loading wine quality datasets and performing train/test splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality_mlops.config import config


def load_data(data_path: str, require_target: bool = True) -> pd.DataFrame:
    """Load wine quality dataset.

    Assumes semicolon-delimited CSV. Optionally validates target column exists.

    Args:
        data_path: Path to the CSV file
        require_target: If True, validates target column exists.
                       If False, target column is optional (for inference data).

    Returns:
        DataFrame with all columns (including target if present)

    Raises:
        FileNotFoundError: If data file doesn't exist
        KeyError: If require_target=True and target column is missing
    """
    # Load data (semicolon-delimited as per UCI format)
    df = pd.read_csv(data_path, sep=";")

    # Validate target column exists if required
    if require_target and config.data.target_column not in df.columns:
        raise KeyError(
            f"'{config.data.target_column}' column not found in dataset. Available columns: {df.columns.tolist()}"
        )

    return df


def normalize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data types to ensure consistency across training and inference.

    This function applies consistent type casting that matches what was used
    during the prepare-data step. Ensures training and inference data have
    identical data types for fair comparison and drift detection.

    All numeric features are cast to float64 to ensure consistent types across
    the pipeline. This is important because pandas may infer different types
    (int64, float64) when loading CSV files, which can cause issues with
    drift detection and model consistency.

    Args:
        df: Raw dataframe (may or may not have target column)

    Returns:
        DataFrame with normalized types (all numeric columns as float64)
    """
    df = df.copy()
    
    # Cast all feature columns (all columns except target) to float64
    # All features in wine quality dataset are numeric
    feature_columns = df.columns.difference([config.data.target_column])
    for col in feature_columns:
        df[col] = df[col].astype("float64")
    
    # Cast target column to float64 if present (handles inference data without target)
    if config.data.target_column in df.columns:
        df[config.data.target_column] = df[config.data.target_column].astype("float64")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.

    Args:
        df: Full dataframe with target column

    Returns:
        X: Features (all columns except target)
        y: Target (target column)

    Raises:
        KeyError: If target column is not found
    """
    if config.data.target_column not in df.columns:
        raise KeyError(
            f"Cannot split - '{config.data.target_column}' column not found. Available columns: {df.columns.tolist()}"
        )

    X = df.drop(config.data.target_column, axis=1)
    y = df[config.data.target_column]

    return X, y
