"""Visualization utilities for model evaluation.

This module provides plotting functions for analyzing regression model performance.
All functions return matplotlib Figure objects ready for display or logging.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Create scatter plot comparing predictions to actual values.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Matplotlib figure with scatter plot and perfect prediction line

    Note:
        Caller is responsible for closing the figure to free memory.
        Example: plt.close(fig) when done.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Quality", fontsize=12)
    ax.set_ylabel("Predicted Quality", fontsize=12)
    ax.set_title("Predictions vs Actual Values", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Create residual plot for regression analysis.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Matplotlib figure with residual scatter plot

    Note:
        Caller is responsible for closing the figure to free memory.
        Example: plt.close(fig) when done.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)

    ax.set_xlabel("Predicted Quality", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_coefficients(
    pipeline: Pipeline,
    feature_names: pd.Index,
) -> plt.Figure:
    """Create bar chart of feature coefficients from linear models.

    Works with linear models (Ridge, Lasso, LinearRegression) that have .coef_ attribute.

    Args:
        pipeline: Trained sklearn pipeline containing the model
        feature_names: Names of features (column names from dataframe)

    Returns:
        Matplotlib figure with horizontal bar chart of coefficients

    Note:
        Caller is responsible for closing the figure to free memory.
        Example: plt.close(fig) when done.
    """
    # Extract coefficients from the linear regressor
    # Navigate through pipeline: 'regressor' step -> regressor_ attribute
    regressor = pipeline.named_steps["regressor"].regressor_
    coefficients = regressor.coef_

    # Sort by absolute value
    indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_coefficients = coefficients[indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["green" if c > 0 else "red" for c in sorted_coefficients]
    ax.barh(sorted_features, sorted_coefficients, color=colors, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Feature Coefficients (Linear Model)", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_distribution_comparison(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Create histogram comparing actual and predicted distributions.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Matplotlib figure with overlaid histograms

    Note:
        Caller is responsible for closing the figure to free memory.
        Example: plt.close(fig) when done.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(y_true, bins=20, alpha=0.5, label="Actual", color="blue", edgecolor="black")
    ax.hist(y_pred, bins=20, alpha=0.5, label="Predicted", color="orange", edgecolor="black")

    ax.set_xlabel("Quality Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
