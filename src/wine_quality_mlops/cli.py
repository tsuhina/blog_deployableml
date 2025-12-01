"""Command-line interface for wine quality MLOps pipeline.

This module provides Typer-based CLI commands for the MLOps pipeline with
separate entrypoints for data quality, training, inference, and inference data quality.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
import typer
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from wine_quality_mlops.config import config
from wine_quality_mlops.data import (
    load_data,
    normalize_dataframe_types,
    split_features_target,
)
from wine_quality_mlops.modeling import (
    evaluate_model,
    optimize_hyperparameters,
    train_model,
)
from wine_quality_mlops.utils import (
    check_quality_threshold,
    extract_test_summary,
    get_active_run_id,
    get_project_dependencies,
    load_model_from_registry,
    load_raw_dataset_from_run,
    raise_on_quality_failure,
    save_and_log_evidently_report,
)
from wine_quality_mlops.visualization import (
    plot_distribution_comparison,
    plot_feature_coefficients,
    plot_predictions_vs_actual,
    plot_residuals,
)

# Initialize Typer app
app = typer.Typer(help="Wine Quality MLOps Pipeline")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.command(name="prepare-data")
def prepare_data(
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for this MLflow run",
    ),
    tag_as_active: bool = typer.Option(
        True,
        "--tag-as-active/--no-tag",
        help="Tag this run as active for automatic fallback (default: True)",
    ),
) -> None:
    """Load raw data, persist dataset as artifact, and run Evidently quality checks.

    This command:
    1. Loads wine quality data from config.data.path
    2. Runs Evidently data quality validation (DataSummaryPreset) on raw data
    3. Logs raw dataset as persistent artifact first (immutable snapshot)
    4. Creates dataset source pointing to artifact for lineage tracking (log_input)
    5. Logs original source path as parameter for provenance (metadata only)
    6. Logs validation report to MLflow data quality experiment

    Best practices:
    - Logs artifact first to ensure immutable lineage (artifacts cannot be overwritten)
    - Dataset lineage points to MLflow artifacts, not source files
    - Original source path logged as parameter for reference
    - Maintains single source of truth for reproducibility

    This is the first step in the MLOps pipeline.
    """
    logger.info("Starting data quality pipeline")
    
    # 1. Load raw data
    logger.info("Loading raw data...")
    raw_df = load_data(config.data.path, require_target=True)
    logger.info(f"Dataset shape: {raw_df.shape}")
    logger.info(f"Columns: {list(raw_df.columns)}")
    if config.data.target_column in raw_df.columns:
        logger.info(f"Target range: [{raw_df[config.data.target_column].min():.1f}, {raw_df[config.data.target_column].max():.1f}]")

    # 2. Normalize data types for consistency (before quality checks)
    # Best practice: Normalize types first, then validate what we'll actually store/use
    # This ensures we catch type casting errors and validate the exact data that will be logged
    raw_df = normalize_dataframe_types(raw_df)

    # 3. Run Evidently data quality checks on normalized data
    # Best practice: Validate the data as it will be used downstream (after normalization)
    # This ensures consistency between what we validate and what we store/log
    logger.info("Running Evidently data quality validation on normalized data...")

    # Define schema (all features except target)
    numerical_features = raw_df.columns.difference([config.data.target_column]).tolist()
    schema = DataDefinition(numerical_columns=numerical_features)

    current_dataset = Dataset.from_pandas(raw_df, data_definition=schema)

    # Create quality report (no drift, first run)
    report = Report([DataSummaryPreset()], include_tests="True")
    snapshot = report.run(current_dataset)

    # 4. Log normalized dataset as persistent artifact to MLflow
    logger.info("Logging normalized dataset to MLflow...")

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Create run name with step type appended
    final_run_name = f"{run_name}-prepare-data" if run_name else "prepare_data"

    with mlflow.start_run(run_name=final_run_name) as run:
        # Log original source path as parameter for provenance (metadata only)
        # This preserves the original source location without relying on it for lineage
        original_source_path = str(Path(config.data.path).resolve())
        mlflow.log_param("original_data_source", original_source_path)

        # Log raw dataset as persistent artifact first (immutable snapshot)
        # Best practice: Log artifact first to ensure immutable lineage
        # This ensures dataset lineage always points to MLflow artifacts, not overwritable files
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            temp_path = tmp_file.name
            raw_df.to_parquet(temp_path)
            mlflow.log_artifact(temp_path, "raw_data.parquet")
            logger.info(f"Logged raw dataset as artifact: {len(raw_df)} rows")
            # Clean up temp file
            Path(temp_path).unlink()

        # Create dataset source pointing to artifact (immutable lineage)
        # Best practice: Point dataset lineage to artifacts, not source files
        artifact_uri = mlflow.get_artifact_uri("raw_data.parquet")
        artifact_source = LocalArtifactDatasetSource(uri=artifact_uri)
        raw_dataset = from_pandas(
            raw_df,
            source=artifact_source,
            name="wine_quality_raw",
            targets=config.data.target_column,
        )
        mlflow.log_input(raw_dataset, context="raw")

        prepare_run_id = run.info.run_id
        logger.info(f"Registered dataset metadata to run: {prepare_run_id}")

        # 5. Log validation report to same MLflow run
        logger.info("Logging validation report to MLflow...")

        # Log data metadata
        mlflow.log_param("data_rows", len(raw_df))
        mlflow.log_param("data_columns", len(raw_df.columns))

        # Extract and log test metrics from Evidently
        # Best practice: Use utility functions for consistent test result handling
        report_dict = snapshot.dict()
        test_summary = extract_test_summary(report_dict)

        # Log test metrics to MLflow
        mlflow.log_metric("tests_passed", test_summary["n_passed"])
        mlflow.log_metric("tests_failed", test_summary["n_failed"])
        mlflow.log_metric("tests_total", test_summary["n_total"])
        mlflow.log_metric("test_pass_rate", test_summary["pass_rate"])
        if test_summary["n_warning"] > 0:
            mlflow.log_metric("tests_warning", test_summary["n_warning"])
        
        # Check if quality thresholds are exceeded
        threshold_check = check_quality_threshold(test_summary, config.data_quality)
        
        # Raise error and stop pipeline if threshold exceeded
        raise_on_quality_failure(
            test_summary,
            threshold_check,
            context="Data quality checks",
        )

        # Save and log HTML report
        save_and_log_evidently_report(snapshot, report_prefix="dq_report_")

    # Tag run as active if requested
    if tag_as_active:
        tag_value = config.mlflow.default_tags.prepare_data
        client = MlflowClient()
        client.set_tag(prepare_run_id, "status", tag_value)
        logger.info(f"Tagged run with status={tag_value}")

    logger.info(f"Prepare data run ID: {prepare_run_id}")
    logger.info("Prepare data pipeline completed")


@app.command(name="training")
def training(
    prepare_run_id: Optional[str] = typer.Option(
        None,
        "--prepare-run-id",
        help="MLflow run ID from prepare-data step (defaults to active tagged run)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for this MLflow run",
    ),
    tag_as_active: bool = typer.Option(
        True,
        "--tag-as-active/--no-tag",
        help="Tag this run as active for automatic fallback (default: True)",
    ),
) -> None:
    """Load raw dataset from prepare-data run, split, optimize, train, and evaluate model.

    This command:
    1. Loads raw dataset artifact from previous prepare-data run
    2. Performs train/test split
    3. Logs train and test datasets separately (log_input with different contexts)
    4. Optimizes hyperparameters with Optuna
    5. Trains final model with best parameters
    6. Evaluates on train and test sets
    7. Logs model, metrics, and visualizations to MLflow

    Best practices:
    - Logs train and test datasets separately (once per context)
    - Both datasets point to same source (reconstruction via split parameters)
    - Logs split parameters for reproducibility

    This is the second step in the MLOps pipeline.
    """
    logger.info("Starting training pipeline")

    # Set MLflow tracking URI first
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    # 1. Resolve prepare_run_id (explicit or fallback to active)
    if prepare_run_id is None:
        logger.info("Finding active prepare-data run...")
        try:
            prepare_run_id = get_active_run_id("prepare_data")
            logger.info(f"Using active prepare-data run: {prepare_run_id}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error: {e}")
            logger.error("Please run prepare-data first or provide --prepare-run-id")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Using provided prepare-data run {prepare_run_id}...")

    # 2. Load raw dataset from MLflow artifact
    logger.info("Loading raw dataset from artifact...")
    raw_df = load_raw_dataset_from_run(prepare_run_id)
    logger.info(f"Loaded raw dataset: {len(raw_df)} rows")

    # 3. Split into features/target and train/test
    logger.info("Splitting dataset into train/test sets...")
    X, y = split_features_target(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Create train and test dataframes for dataset logging
    # Best practice: Log separate datasets for train/test splits
    train_df = X_train.copy()
    train_df[config.data.target_column] = y_train
    test_df = X_test.copy()
    test_df[config.data.target_column] = y_test

    # Create dataset source pointing to prepare-data artifact for lineage
    # Both train and test datasets point to same source (reconstruction via split parameters)
    artifact_uri = f"runs:/{prepare_run_id}/raw_data.parquet"
    dataset_source = LocalArtifactDatasetSource(uri=artifact_uri)

    # 4. Start MLflow run early to enable Optuna integration
    logger.info("Logging to MLflow...")
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Create run name with step type appended
    final_run_name = f"{run_name}-training" if run_name else "training"

    with mlflow.start_run(run_name=final_run_name) as run:
        # Log lineage and split parameters early
        mlflow.log_param("prepare_run_id", prepare_run_id)
        mlflow.log_param("test_size", config.data.test_size)
        mlflow.log_param("random_state", config.data.random_state)

        # Log optimization configuration
        mlflow.log_param("optimization_enable_pruning", config.optimization.enable_pruning)
        mlflow.log_param("optimization_pruner_type", config.optimization.pruner_type)
        mlflow.log_param("optimization_sampler_type", config.optimization.sampler_type)
        mlflow.log_param("optimization_n_startup_trials", config.optimization.n_startup_trials)
        mlflow.log_param("optimization_n_warmup_steps", config.optimization.n_warmup_steps)

        # 5. Hyperparameter optimization
        logger.info("Optimizing hyperparameters with Optuna...")
        best_params = optimize_hyperparameters(
            X_train,
            y_train,
            n_trials=config.optimization.n_trials,
        )

        # Log Optuna optimization results
        mlflow.log_param("optimization_n_trials", config.optimization.n_trials)
        mlflow.log_param("optimization_cv_folds", config.optimization.cv_folds)

        # 6. Train final model
        logger.info("Training final model...")
        pipeline = train_model(X_train, y_train, best_params)

        # 7. Evaluate
        logger.info("Evaluating model...")
        train_metrics = evaluate_model(pipeline, X_train, y_train)
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        logger.info("Train Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        # 8. Create visualizations
        logger.info("Creating visualizations...")
        y_test_pred = pipeline.predict(X_test)

        predictions_vs_actual_fig = plot_predictions_vs_actual(y_test, y_test_pred)
        residuals_fig = plot_residuals(y_test, y_test_pred)
        feature_coefficients_fig = plot_feature_coefficients(pipeline, X_train.columns)
        distribution_comparison_fig = plot_distribution_comparison(y_test, y_test_pred)
        # Log train and test datasets separately
        # Best practice: Log each dataset once per context/transformation
        # Both point to same source - reconstruction via split parameters ensures reproducibility
        train_dataset = from_pandas(
            train_df,
            source=dataset_source,
            name="wine_quality_train",
            targets=config.data.target_column,
        )
        mlflow.log_input(train_dataset, context="training")

        test_dataset = from_pandas(
            test_df,
            source=dataset_source,
            name="wine_quality_test",
            targets=config.data.target_column,
        )
        mlflow.log_input(test_dataset, context="testing")

        # Log model parameters
        mlflow.log_param("model_type", config.model.type)
        mlflow.log_param("alpha", best_params["alpha"])
        mlflow.log_param("cv_folds", config.optimization.cv_folds)
        mlflow.log_param("n_trials", config.optimization.n_trials)

        # Log metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Log model with signature and input example
        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=get_project_dependencies(),
        )

        # Log figures
        mlflow.log_figure(predictions_vs_actual_fig, "predictions_vs_actual.png")
        mlflow.log_figure(residuals_fig, "residuals.png")
        mlflow.log_figure(feature_coefficients_fig, "feature_coefficients.png")
        mlflow.log_figure(distribution_comparison_fig, "distribution.png")

        # Clean up matplotlib figures to prevent memory accumulation
        # Best practice: Close figures after logging to free resources
        for fig in [predictions_vs_actual_fig, residuals_fig, feature_coefficients_fig, distribution_comparison_fig]:
            plt.close(fig)

        # Model Registry integration - auto-register all models
        training_run_id = run.info.run_id
        logger.info("Registering model to Model Registry...")
        model_uri = f"runs:/{training_run_id}/model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=config.mlflow.model_name
        )
        logger.info(f"Registered as version {model_version.version}")

        # Set "staging" alias for new models
        client = MlflowClient()
        client.set_registered_model_alias(
            name=config.mlflow.model_name,
            alias="staging",
            version=model_version.version
        )
        logger.info(f"Set alias 'staging' for version {model_version.version}")
        logger.info("Promote to 'champion' manually when ready for production")

        logger.info(f"Training run ID: {training_run_id}")

    # Tag run as active if requested
    if tag_as_active:
        tag_value = config.mlflow.default_tags.training
        client = MlflowClient()
        client.set_tag(training_run_id, "status", tag_value)
        logger.info(f"Tagged run with status={tag_value}")

    logger.info("Training pipeline completed")


@app.command(name="inference")
def inference(
    model_alias: str = typer.Option(
        "champion",
        "--model-alias",
        help="Model Registry alias to load (e.g., 'champion', 'staging'). Defaults to 'champion'.",
    ),
    inference_dq_run_id: Optional[str] = typer.Option(
        None,
        "--inference-dq-run-id",
        help="MLflow run ID from inference-dq step (defaults to active tagged run)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for this MLflow run",
    ),
) -> None:
    """Load trained model from Model Registry, make predictions on validated inference data, and log results.

    Model is loaded from Model Registry using the specified alias (defaults to 'champion').
    Inference data is loaded from the inference-dq run artifact (after drift checks pass).

    This is the final step in the MLOps pipeline. Drift detection is handled by the
    inference-dq command before running inference.
    """
    logger.info("Starting inference pipeline")

    # Set MLflow tracking URI first
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    # Load model from Model Registry using alias
    logger.info(f"Loading model from Model Registry with alias '{model_alias}'...")
    try:
        pipeline, model_metadata = load_model_from_registry(
            config.mlflow.model_name,
            model_alias
        )
        resolved_model_run_id = model_metadata["run_id"]
        logger.info(f"Loaded model version {model_metadata['version']} from registry")
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Failed to load model with alias '{model_alias}': {e}")
        logger.error("Run 'mlflow ui' to check available models and aliases")
        logger.error("Or use --model-alias <alias> to specify a different alias")
        raise typer.Exit(code=1)

    # Get prepare_run_id for dataset lineage
    model_run = mlflow.get_run(resolved_model_run_id)
    if "prepare_run_id" not in model_run.data.params:
        raise KeyError(f"Parameter 'prepare_run_id' not found in run {resolved_model_run_id}")
    prepare_run_id = model_run.data.params["prepare_run_id"]

    # 2. Resolve inference_dq_run_id (explicit or fallback to active)
    if inference_dq_run_id is None:
        logger.info("Finding active inference-dq run...")
        try:
            inference_dq_run_id = get_active_run_id("inference_dq")
            logger.info(f"Using active inference-dq run: {inference_dq_run_id}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error: {e}")
            logger.error("Please run inference-dq first or provide --inference-dq-run-id")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Using provided inference-dq run {inference_dq_run_id}...")

    # 3. Load validated inference data from inference-dq artifact
    # Best practice: Load from the validated artifact logged by inference-dq after quality checks pass
    # This ensures we use the exact same data that passed drift detection
    logger.info("Loading validated inference data from inference-dq artifact...")
    
    # Load inference_features.parquet artifact from inference-dq run
    try:
        artifact_uri = f"runs:/{inference_dq_run_id}/inference_features.parquet"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        X_inference = pd.read_parquet(local_path)
        logger.info(f"Loaded validated inference features: {X_inference.shape}")
        # Note: Data is already normalized (was normalized before logging in inference-dq)
    except (mlflow.exceptions.MlflowException, OSError, FileNotFoundError) as e:
        logger.error(f"Could not load inference features from run {inference_dq_run_id}: {e}")
        logger.error("Make sure inference-dq has completed successfully and logged inference_features.parquet")
        raise typer.Exit(code=1)

    # 4. Make predictions
    logger.info("Making predictions...")
    predictions = pipeline.predict(X_inference)
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    # 5. Log predictions dataframe to MLflow
    logger.info("Logging predictions to MLflow...")

    mlflow.set_experiment(config.mlflow.experiment_name)

    # Create run name with step type appended
    final_run_name = f"{run_name}-inference" if run_name else "inference"

    with mlflow.start_run(run_name=final_run_name) as run:
        # Log lineage
        mlflow.log_param("model_run_id", resolved_model_run_id)
        mlflow.log_param("prepare_run_id", prepare_run_id)
        mlflow.log_param("inference_dq_run_id", inference_dq_run_id)

        # Log model metadata if loaded from registry
        if model_metadata:
            mlflow.log_param("model_alias", model_metadata["alias"])
            mlflow.log_param("model_version", model_metadata["version"])
            mlflow.log_param("model_source", "registry")
        else:
            mlflow.log_param("model_source", "run_id")

        # Create predictions-only dataframe (same index as inference-dq dataset)
        # Best practice: Log predictions separately, indexed to match inference-dq dataset for alignment
        predictions_df = pd.DataFrame({'predictions': predictions}, index=X_inference.index)
        
        # Log predictions as persistent artifact
        # Best practice: Use log_artifact() for file persistence, not log_table()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            temp_path = tmp_file.name
            predictions_df.to_parquet(temp_path)
            mlflow.log_artifact(temp_path, "predictions.parquet")
            logger.info(f"Logged predictions as artifact: {len(predictions_df)} rows")
            # Clean up temp file
            Path(temp_path).unlink()

        # Log predictions dataset (output of this step)
        # Best practice: Log dataset once per context - this is the OUTPUT
        artifact_uri = mlflow.get_artifact_uri("predictions.parquet")
        dataset_source = LocalArtifactDatasetSource(uri=artifact_uri)
        predictions_dataset = from_pandas(
            predictions_df,
            source=dataset_source,
            name="wine_quality_predictions",
            targets="predictions",
        )
        mlflow.log_input(predictions_dataset, context="inference")
        
        # Log input dataset pointing to inference-dq artifact for complete lineage
        # Best practice: Log dataset once per context - this is the INPUT (different context)
        input_artifact_uri = f"runs:/{inference_dq_run_id}/inference_features.parquet"
        input_dataset_source = LocalArtifactDatasetSource(uri=input_artifact_uri)
        input_dataset = from_pandas(
            X_inference,
            source=input_dataset_source,
            name="wine_quality_inference_input",
        )
        mlflow.log_input(input_dataset, context="inference_input")

        # Log basic prediction stats
        mlflow.log_metric("num_predictions", len(predictions))
        mlflow.log_metric("prediction_mean", predictions.mean())
        mlflow.log_metric("prediction_std", predictions.std())

        inference_run_id = run.info.run_id
        logger.info(f"Inference run ID: {inference_run_id}")

    logger.info("Inference pipeline completed")


@app.command(name="inference-dq")
def inference_dq(
    model_run_id: Optional[str] = typer.Option(
        None,
        "--model-run-id",
        help="MLflow run ID from training step (defaults to active tagged run)",
    ),
    inference_data_path: Optional[str] = typer.Option(
        None,
        "--inference-data-path",
        help="Path to inference data file (defaults to config.data.inference_path)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for this MLflow run",
    ),
    tag_as_active: bool = typer.Option(
        True,
        "--tag-as-active/--no-tag",
        help="Tag this run as active for automatic fallback (default: True)",
    ),
) -> None:
    """Compare inference vs training data for drift detection BEFORE running inference.

    This command:
    1. Loads training dataset from logged artifact (prepare-data run)
    2. Loads inference data from raw file (features only, before predictions)
    3. Runs Evidently drift detection comparing feature distributions
    4. Logs validated inference data as artifact (after quality checks pass)
    5. Logs drift report (HTML) and test metrics to MLflow
    6. Blocks pipeline if quality thresholds exceeded

    This is the THIRD step in the MLOps pipeline (before inference).

    Best practices:
    - Compares against exact training data artifact (not source file)
    - Logs validated inference_features.parquet artifact after checks pass
    - This artifact is used by inference command to make predictions
    - HTML drift report is logged as artifact for inspection

    Note: This command focuses on data drift detection only. Model performance
    evaluation happens in the training step on the test set.
    """
    logger.info("Starting inference data quality pipeline")

    # Set MLflow tracking URI first
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    # 1. Resolve model_run_id (explicit or fallback to active)
    if model_run_id is None:
        logger.info("Finding active training run...")
        try:
            model_run_id = get_active_run_id("training")
            logger.info(f"Using active training run: {model_run_id}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error: {e}")
            logger.error("Please run training first or provide --model-run-id")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Using provided training run {model_run_id}...")

    # Get prepare_run_id for dataset lineage
    model_run = mlflow.get_run(model_run_id)
    if "prepare_run_id" not in model_run.data.params:
        raise KeyError(f"Parameter 'prepare_run_id' not found in run {model_run_id}")
    prepare_run_id = model_run.data.params["prepare_run_id"]

    # 2. Resolve inference data path
    if inference_data_path is None:
        inference_data_path = config.data.inference_path
        logger.info("Using default inference data path from config...")
    else:
        logger.info("Using provided inference data path...")

    logger.info(f"Path: {inference_data_path}")

    # 3. Load raw training dataset from logged artifact for drift comparison
    # Best practice: Load from the exact dataset artifact used during training
    # This ensures we compare against the same data the model was trained on, not just the same source file
    logger.info("Loading raw training dataset from logged artifact for drift comparison...")

    # Load from prepare-data artifact (same dataset used during training)
    reference_raw_df = load_raw_dataset_from_run(prepare_run_id)
    # Note: Dataset is already normalized (was normalized before logging in prepare-data step)
    
    # Remove target column to get features only (same format as inference data)
    reference_features = reference_raw_df.drop(columns=[config.data.target_column])
    logger.info(f"Loaded raw training dataset: {reference_features.shape}")

    # Create reference dataset pointing to prepare-data artifact for lineage
    reference_artifact_uri = f"runs:/{prepare_run_id}/raw_data.parquet"
    reference_dataset_source = LocalArtifactDatasetSource(uri=reference_artifact_uri)
    reference_dataset_mlflow = from_pandas(
        reference_features,
        source=reference_dataset_source,
        name="wine_quality_reference",
    )

    # 4. Load inference data from file and normalize
    logger.info("Loading inference data...")

    # Load inference data and apply same normalization as training data
    inference_raw_df = load_data(inference_data_path, require_target=False)
    inference_raw_df = normalize_dataframe_types(inference_raw_df)
    
    # Remove target column if present (features only for drift comparison)
    inference_features = inference_raw_df.drop(columns=[config.data.target_column]) if config.data.target_column in inference_raw_df.columns else inference_raw_df
    logger.info(f"Loaded inference data: {inference_features.shape}")

    # Create inference dataset pointing to original file path for lineage
    inference_dataset_source = LocalArtifactDatasetSource(uri=str(Path(inference_data_path).resolve()))
    inference_dataset_mlflow = from_pandas(
        inference_features,
        source=inference_dataset_source,
        name="wine_quality_inference_dq",
    )

    # 5. Run Evidently drift detection
    logger.info("Running Evidently drift detection...")

    # Define schema (features only, no target column)
    numerical_features = reference_features.columns.tolist()
    schema = DataDefinition(numerical_columns=numerical_features)

    # Create datasets for drift detection
    reference_dataset = Dataset.from_pandas(reference_features, data_definition=schema)
    current_dataset = Dataset.from_pandas(inference_features, data_definition=schema)

    # Create drift report (DataSummaryPreset + DataDriftPreset)
    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests="True",
    )
    snapshot = report.run(current_dataset, reference_dataset)

    # 6. Log to MLflow
    logger.info("Logging results to MLflow...")

    mlflow.set_experiment(config.mlflow.experiment_name)

    # Create run name with step type appended
    final_run_name = f"{run_name}-inference-dq" if run_name else "inference_dq"

    with mlflow.start_run(run_name=final_run_name) as run:
        # Log lineage
        mlflow.log_param("model_run_id", model_run_id)
        mlflow.log_param("prepare_run_id", prepare_run_id)
        
        # Log datasets for complete lineage
        mlflow.log_input(reference_dataset_mlflow, context="reference")
        mlflow.log_input(inference_dataset_mlflow, context="inference")

        # Log data info
        mlflow.log_param("inference_data_rows", len(inference_features))
        mlflow.log_param("training_data_rows", len(reference_features))

        # Extract and log test results from Evidently
        # Best practice: Use utility functions for consistent test result handling
        report_dict = snapshot.dict()
        test_summary = extract_test_summary(report_dict)

        # Log test metrics to MLflow
        mlflow.log_metric("tests_passed", test_summary["n_passed"])
        mlflow.log_metric("tests_failed", test_summary["n_failed"])
        mlflow.log_metric("tests_total", test_summary["n_total"])
        mlflow.log_metric("test_pass_rate", test_summary["pass_rate"])
        if test_summary["n_warning"] > 0:
            mlflow.log_metric("tests_warning", test_summary["n_warning"])
        
        # Display drift detection results
        logger.info("Drift Detection Results:")
        logger.info(f"  Tests Passed: {test_summary['n_passed']}/{test_summary['n_total']}")
        logger.info(f"  Tests Failed: {test_summary['n_failed']}/{test_summary['n_total']}")
        logger.info(f"  Pass Rate: {test_summary['pass_rate'] * 100:.1f}%")

        # Check if quality thresholds are exceeded
        threshold_check = check_quality_threshold(test_summary, config.data_quality)
        
        # Raise error and stop pipeline if threshold exceeded
        raise_on_quality_failure(
            test_summary,
            threshold_check,
            context="Data quality checks",
        )
        
        # If we get here, thresholds were not exceeded
        logger.info("No significant drift detected")

        # Log validated inference data as artifact (after quality checks pass)
        # This artifact will be used by the inference command to make predictions
        logger.info("Logging validated inference data as artifact...")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            temp_path = tmp_file.name
            inference_features.to_parquet(temp_path)
            mlflow.log_artifact(temp_path, "inference_features.parquet")
            logger.info(f"Logged validated inference features as artifact: {len(inference_features)} rows")
            # Clean up temp file
            Path(temp_path).unlink()

        # Save and log HTML report
        save_and_log_evidently_report(snapshot, report_prefix="inference_dq_report_")

        inference_dq_run_id = run.info.run_id
        logger.info(f"Inference DQ run ID: {inference_dq_run_id}")

    # Tag run as active if requested
    if tag_as_active:
        tag_value = config.mlflow.default_tags.inference_dq
        client = MlflowClient()
        client.set_tag(inference_dq_run_id, "status", tag_value)
        logger.info(f"Tagged run with status={tag_value}")

    logger.info("Inference data quality pipeline completed")


if __name__ == "__main__":
    app()
