"""Wine Quality MLOps - Production-ready ML pipeline with MLflow tracking.

This package provides a complete ML pipeline for wine quality regression,
demonstrating best practices for MLOps with scikit-learn and MLflow.
"""

__version__ = "0.1.0"

# Expose main components for easier imports
from .cli import app
from .config import config

__all__ = [
    "app",
    "config",
]
