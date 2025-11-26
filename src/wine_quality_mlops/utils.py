"""Utility functions for the wine quality MLOps pipeline."""

import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tomllib
import typer

from wine_quality_mlops.config import config
from wine_quality_mlops.data import split_features_target

logger = logging.getLogger(__name__)


def save_and_log_evidently_report(
    snapshot,
    artifact_path: str = "validation_reports",
    report_prefix: str = "report_",
) -> None:
    """Save Evidently report as HTML and log to MLflow.

    Args:
        snapshot: Evidently Snapshot object from report.run()
        artifact_path: MLflow artifact path to log report to
        report_prefix: Prefix for temporary HTML file name
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix=report_prefix
    ) as tmp_file:
        tmp_path = tmp_file.name
        snapshot.save_html(tmp_path)
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    # File automatically deleted when context exits


def get_project_dependencies() -> List[str]:
    """Extract dependencies from pyproject.toml and pin exact versions from uv.lock.

    This function reads the project's pyproject.toml file and extracts
    the list of dependencies, filtering out conditional dependencies that
    don't apply to the current Python version. Then it looks up exact versions
    from uv.lock to pin dependencies for reproducibility.

    Returns:
        List of dependency strings in pip format with exact versions
        (e.g., ["pandas==2.3.3", "scikit-learn==1.7.2"])
        Conditional dependencies that don't match current Python version are excluded.

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        KeyError: If dependencies are not found in pyproject.toml
    """
    # Navigate from src/wine_quality_mlops/utils.py to root/
    root_dir = Path(__file__).parent.parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    uv_lock_path = root_dir / "uv.lock"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    dependencies = data.get("project", {}).get("dependencies", [])

    if not dependencies:
        raise KeyError("No dependencies found in pyproject.toml [project.dependencies]")

    # Parse uv.lock to create package name -> version lookup dict
    versions = {}
    if uv_lock_path.exists():
        try:
            with open(uv_lock_path, "rb") as f:
                lock_data = tomllib.load(f)
            # Normalize package names: lowercase, replace hyphens with underscores
            versions = {
                pkg["name"].lower().replace("-", "_"): pkg["version"]
                for pkg in lock_data.get("package", [])
            }
        except Exception:
            # If uv.lock parsing fails, fall back to pyproject.toml specifiers
            pass

    # Filter out conditional dependencies and replace versions with exact ones
    filtered_deps = []
    dep_pattern = re.compile(r"^([a-zA-Z0-9_-]+(\[[^\]]+\])?)(.*)$")

    for dep in dependencies:
        # Check for environment markers (e.g., python_version < '3.11')
        if "python_version" in dep:
            # For Python 3.12+, skip dependencies with python_version < '3.11'
            if "python_version < '3.11'" in dep and sys.version_info >= (3, 11):
                continue
            # Remove the marker from the dependency string
            dep = dep.split(";")[0].strip()

        # Extract package name (with extras) and version specifier
        match = dep_pattern.match(dep)
        if match:
            full_name = match.group(1)  # e.g., "scikit-learn[extra]" or "pandas"

            # Extract base name for matching (remove extras)
            base_name = re.sub(r"\[.*\]", "", full_name)
            normalized_name = base_name.lower().replace("-", "_")

            # Look up exact version in uv.lock
            if versions and normalized_name in versions:
                # Preserve extras notation from pyproject.toml
                filtered_deps.append(f"{full_name}=={versions[normalized_name]}")
            else:
                # Fallback to original specifier from pyproject.toml
                filtered_deps.append(dep)
        else:
            # If regex doesn't match, keep original dependency as-is
            filtered_deps.append(dep)

    # Add pip explicitly to avoid "Failed to resolve installed pip version" warning
    # This is especially important when using uv as the package manager
    pip_name = "pip"
    pip_normalized = pip_name.lower().replace("-", "_")
    if versions and pip_normalized in versions:
        filtered_deps.append(f"{pip_name}=={versions[pip_normalized]}")
    else:
        filtered_deps.append("pip>=24.0")

    return filtered_deps


def load_raw_dataset_from_run(run_id: str) -> pd.DataFrame:
    """Load raw dataset from MLflow run artifact.

    This function loads the raw dataset artifact that was logged using mlflow.log_artifact().
    Uses MLflow's artifact download API to retrieve the parquet file and loads it with pandas.

    Best practice: For dataset reconstruction, prefer using load_datasets_from_run() which
    reconstructs from original source + parameters for better reproducibility.

    Args:
        run_id: MLflow run ID containing the raw dataset artifact

    Returns:
        DataFrame containing the raw dataset

    Raises:
        ValueError: If artifact is not found in the run
        FileNotFoundError: If artifact file cannot be loaded
    """
    try:
        # Use MLflow's artifact URI to download the parquet artifact
        artifact_uri = f"runs:/{run_id}/raw_data.parquet"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        
        # Load the parquet file
        return pd.read_parquet(local_path)
    except Exception as e:
        raise ValueError(
            f"Could not load raw dataset from run {run_id}: {e}"
        ) from e


def load_model_from_run(run_id: str) -> Pipeline:
    """Load a trained model from an MLflow run.

    Args:
        run_id: MLflow run ID containing the model

    Returns:
        Trained sklearn pipeline

    Raises:
        MlflowException: If model is not found in the run
    """
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def load_model_from_registry(model_name: str, alias: str) -> tuple[Pipeline, dict]:
    """Load model from Model Registry by alias.

    Args:
        model_name: Registered model name from config
        alias: Model alias (e.g., "champion", "staging")

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - version: model version number
        - run_id: original training run ID
        - alias: alias used

    Raises:
        mlflow.exceptions.MlflowException: If model or alias not found

    Example:
        pipeline, metadata = load_model_from_registry("wine_quality_regressor", "champion")
        print(f"Loaded version {metadata['version']}")
    """
    client = mlflow.MlflowClient()

    try:
        # Get model version by alias
        model_version = client.get_model_version_by_alias(model_name, alias)

        # CRITICAL: Use @ syntax for aliases (not /)
        model_uri = f"models:/{model_name}@{alias}"
        pipeline = mlflow.sklearn.load_model(model_uri)

        return pipeline, {
            "version": model_version.version,
            "run_id": model_version.run_id,
            "alias": alias,
        }
    except mlflow.exceptions.MlflowException as e:
        error_msg = str(e)
        if "No model version" in error_msg or "No registered model" in error_msg:
            raise mlflow.exceptions.MlflowException(
                f"Model '{model_name}' with alias '{alias}' not found.\n"
                f"Check MLflow UI (mlflow ui) for available models and aliases."
            ) from e
        raise


def get_latest_run_by_tag(
    experiment_name: str,
    tag_key: str,
    tag_value: str,
) -> str:
    """Find the most recent MLflow run with a specific tag.

    Searches for runs in the specified experiment that have the given tag,
    and returns the run ID of the most recent one (by start time).

    Args:
        experiment_name: Name of the MLflow experiment to search
        tag_key: Tag key to filter by
        tag_value: Tag value to match

    Returns:
        Run ID of the most recent matching run

    Raises:
        ValueError: If no runs found with the specified tag
    """
    client = mlflow.MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search for runs with the specified tag
    filter_string = f"tags.{tag_key} = '{tag_value}'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No runs found with tag {tag_key}='{tag_value}' "
            f"in experiment '{experiment_name}'"
        )

    return runs[0].info.run_id


def load_datasets_from_run(run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train/test datasets from MLflow training run.

    Best practice: Reconstructs datasets from original source + split parameters
    rather than loading duplicated artifacts. This ensures reproducibility by
    using the single source of truth (original data) + logged parameters.

    Args:
        run_id: MLflow run ID from training step

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) DataFrames/Series

    Raises:
        ValueError: If run not found or datasets cannot be reconstructed
        KeyError: If required parameters or dataset inputs are missing
    """
    # Get run metadata
    run = mlflow.get_run(run_id)
    
    # Extract split parameters
    params = run.data.params
    if "test_size" not in params or "random_state" not in params:
        raise KeyError(
            f"Required split parameters not found in run {run_id}. "
            f"Expected: test_size, random_state"
        )
    
    test_size = float(params["test_size"])
    random_state = int(params["random_state"])
    
    # Get prepare_run_id to find original data source
    if "prepare_run_id" not in params:
        raise KeyError(
            f"prepare_run_id not found in run {run_id}. Cannot reconstruct datasets."
        )
    
    prepare_run_id = params["prepare_run_id"]
    
    # Load original raw dataset from prepare-data run artifact
    # This is the single source of truth
    raw_df = load_raw_dataset_from_run(prepare_run_id)
    
    # Reconstruct exact split using logged parameters
    X, y = split_features_target(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def extract_test_summary(report_dict: Dict) -> Dict:
    """Extract test summary from Evidently report dictionary.

    Parses Evidently snapshot dictionary to extract test statistics and details.
    Handles edge cases like missing tests, empty test arrays, and various test statuses.

    Evidently test statuses can be: "SUCCESS", "FAIL", "WARNING", "ERROR", "SKIPPED"

    Args:
        report_dict: Dictionary from Evidently snapshot.dict() containing test results

    Returns:
        Dictionary with keys:
            - n_passed: Number of tests with status "SUCCESS"
            - n_failed: Number of tests with status "FAIL" or "ERROR"
            - n_warning: Number of tests with status "WARNING"
            - n_total: Total number of tests
            - pass_rate: Pass rate as float (0.0-1.0)
            - failed_tests: List of failed test dictionaries (for debugging)
            - warning_tests: List of warning test dictionaries (optional)
    """
    # Handle missing or empty tests
    if "tests" not in report_dict or not report_dict["tests"]:
        return {
            "n_passed": 0,
            "n_failed": 0,
            "n_warning": 0,
            "n_total": 0,
            "pass_rate": 0.0,
            "failed_tests": [],
            "warning_tests": [],
        }

    tests = report_dict["tests"]
    
    # Count tests by status
    n_passed = sum(1 for test in tests if test.get("status") == "SUCCESS")
    n_failed = sum(1 for test in tests if test.get("status") in ("FAIL", "ERROR"))
    n_warning = sum(1 for test in tests if test.get("status") == "WARNING")
    n_total = len(tests)
    
    # Calculate pass rate
    pass_rate = n_passed / n_total if n_total > 0 else 0.0
    
    # Extract failed and warning tests for debugging
    failed_tests = [
        test for test in tests 
        if test.get("status") in ("FAIL", "ERROR")
    ]
    warning_tests = [
        test for test in tests 
        if test.get("status") == "WARNING"
    ]
    
    return {
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_warning": n_warning,
        "n_total": n_total,
        "pass_rate": pass_rate,
        "failed_tests": failed_tests,
        "warning_tests": warning_tests,
    }


def check_quality_threshold(
    test_summary: Dict,
    data_quality_config,
) -> Dict:
    """Check if test results exceed quality thresholds.

    Validates test results against configured thresholds:
    - max_allowed_failures: Maximum number of failed tests allowed
    - min_pass_rate: Minimum pass rate required (if configured)

    Args:
        test_summary: Dictionary from extract_test_summary()
        data_quality_config: DataQualityConfig instance from config

    Returns:
        Dictionary with keys:
            - exceeded: Boolean indicating if threshold was exceeded
            - reason: String describing why threshold was exceeded (if applicable)
            - failed_count: Number of failed tests
            - threshold: Threshold value that was checked
    """
    n_failed = test_summary["n_failed"]
    n_passed = test_summary["n_passed"]
    n_total = test_summary["n_total"]
    
    # Check max_allowed_failures threshold
    if n_failed > data_quality_config.max_allowed_failures:
        return {
            "exceeded": True,
            "reason": f"Failed tests ({n_failed}) exceeded threshold ({data_quality_config.max_allowed_failures})",
            "failed_count": n_failed,
            "threshold": data_quality_config.max_allowed_failures,
        }
    
    # Check min_pass_rate threshold (if configured)
    if data_quality_config.min_pass_rate is not None:
        pass_rate = test_summary["pass_rate"]
        if pass_rate < data_quality_config.min_pass_rate:
            return {
                "exceeded": True,
                "reason": f"Pass rate ({pass_rate:.2%}) below minimum ({data_quality_config.min_pass_rate:.2%})",
                "failed_count": n_failed,
                "threshold": data_quality_config.min_pass_rate,
            }
    
    return {
        "exceeded": False,
        "reason": None,
        "failed_count": n_failed,
        "threshold": None,
    }


def raise_on_quality_failure(
    test_summary: Dict,
    threshold_check: Dict,
    context: str = "Data quality checks",
) -> None:
    """Raise typer.Exit if quality threshold is exceeded.

    Logs detailed error messages including failed test information and raises
    typer.Exit(code=1) to stop pipeline execution.

    Args:
        test_summary: Dictionary from extract_test_summary()
        threshold_check: Dictionary from check_quality_threshold()
        context: Context string for error messages (e.g., "Data quality checks", "Drift detection")

    Raises:
        typer.Exit: Always raises with code=1 if threshold exceeded
    """
    if not threshold_check["exceeded"]:
        return
    
    n_failed = test_summary["n_failed"]
    n_total = test_summary["n_total"]
    
    # Log main error message
    logger.error(
        f"{context} failed: {n_failed}/{n_total} tests failed"
    )
    logger.error(f"Reason: {threshold_check['reason']}")
    
    # Log failed test details for debugging
    if test_summary["failed_tests"]:
        failed_test_names = [
            test.get("name", "Unknown") 
            for test in test_summary["failed_tests"]
        ]
        logger.error(f"Failed tests: {', '.join(failed_test_names)}")
    
    logger.error("Pipeline stopped to prevent proceeding with problematic data")
    raise typer.Exit(code=1)


def get_active_run_id(step_type: str) -> str:
    """Get the active run ID for a specific pipeline step using config defaults.

    This function uses the default tags from config.yaml to find the most
    recent "active" run for a given step type (prepare_data, training, etc.).

    Args:
        step_type: Type of pipeline step ("prepare_data", "training", "inference_dq")

    Returns:
        Run ID of the active run for this step type

    Raises:
        ValueError: If no active run found or step_type not in config
        KeyError: If step_type not found in config.mlflow.default_tags

    Example:
        # Get active prepare-data run
        prepare_run_id = get_active_run_id("prepare_data")

        # Get active training run
        training_run_id = get_active_run_id("training")
    """

    # Get tag value from config
    if not hasattr(config.mlflow, "default_tags"):
        raise ValueError(
            "No default_tags configured in config.yaml under mlflow.default_tags"
        )

    tags_dict = config.mlflow.default_tags
    if not hasattr(tags_dict, step_type):
        raise KeyError(
            f"Step type '{step_type}' not found in config.mlflow.default_tags. "
            f"Available: {list(vars(tags_dict).keys())}"
        )

    tag_value = getattr(tags_dict, step_type)

    # Search for latest run with this tag in main experiment
    experiment_name = config.mlflow.experiment_name

    # Search for latest run with this tag
    return get_latest_run_by_tag(
        experiment_name=experiment_name,
        tag_key="status",
        tag_value=tag_value,
    )
