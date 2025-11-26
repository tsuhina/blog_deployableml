# Wine Quality MLOps Pipeline

A demonstration ML project that serves as the basis for a blog series on deployable machine learning workflows. This repository demonstrates a 4-step workflow (data-quality → training → inference → inference-dq) using industry-standard tools like MLflow, Evidently, Optuna, and Pydantic.

> **Note:** This repository is created for demonstration and educational purposes as part of a blog series on deployable machine learning. The code and patterns shown here are intended to illustrate concepts and architectural decisions, not to serve as production-ready templates.

## About This Project

This repository serves as the **codebase foundation** for a blog series on deployable machine learning workflows. It demonstrates **one approach** to building a production-oriented ML project structure using industry-standard tools (MLflow, Evidently, Optuna, Pydantic). It serves as an **illustrative example** and learning resource, not as production-ready code to copy verbatim.

**Key Points:**
- Created for demonstration and educational purposes as part of a blog series
- Shows patterns for code organization, experiment tracking, and data quality monitoring
- Design decisions are opinionated and simplified for clarity
- Real-world projects require adaptation based on specific requirements
- Not exhaustive or claiming to represent "best practices" - just one reasonable approach
- Focus is on demonstrating architectural patterns, not production perfection

Think of this as a reference implementation to learn from and adapt, not a template to use as-is.

## Features

- 4-step MLOps workflow with MLflow lineage tracking
- Experiment tracking and model management with MLflow
- Data quality validation and drift detection with Evidently AI
- Model signature logging for deployment readiness
- Configuration management with Pydantic and YAML
- Reproducible preprocessing and feature engineering using scikit-learn pipelines
- Command-line interface built with Typer
- Modular, testable, and maintainable code structure (7 focused modules)
- Hyperparameter optimization with Optuna (50 trials, 5-fold CV)
- Regression metrics and visualizations (MAE, RMSE, R², residuals, coefficients)

## Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install the project:

```bash
git clone <repository-url>
cd wine-quality-mlops
uv sync
```

### Run the Training Pipeline

```bash
# Run the complete pipeline (data loading, optimization, training, evaluation, MLflow logging)
uv run wine-quality train

# With custom run name
uv run wine-quality train --run-name experiment-v1
```

This single command:
1. Loads wine quality data from config.yaml path
2. Splits into train/test sets (80/20)
3. Optimizes Ridge alpha parameter with Optuna (50 trials, 5-fold CV)
4. Trains final model with best hyperparameters
5. Evaluates on train and test sets (MAE, MSE, RMSE, R²)
6. Logs everything to MLflow with model signature and 4 visualizations

### View Results in MLflow UI

```bash
uv run wine-quality mlflow-ui
```

Then open your browser to [http://localhost:5000](http://localhost:5000)

## Project Structure

```
wine-quality-mlops/
├── config.yaml                # Configuration file (Pydantic + YAML)
├── main_legacy.py             # Original single-script (reference only)
├── pyproject.toml             # Dependencies and configuration
├── README.md                  # This file
├── CLAUDE.md                  # Developer guide for Claude Code
├── data/
│   └── raw/                   # Wine quality CSV files
├── mlruns/                    # MLflow tracking directory (gitignored)
└── src/
    └── wine_quality_mlops/
        ├── __init__.py        # Package initialization
        ├── config.py          # Pydantic configuration manager
        ├── data.py            # Data loading and splitting
        ├── modeling.py        # Pipeline, training, optimization, evaluation
        ├── visualization.py   # Plot generation (4 plots)
        └── cli.py             # Typer CLI (train, mlflow-ui)
```

## CLI Commands

### 1. Train Model

```bash
uv run wine-quality train
```

Runs the complete pipeline:
- Loads data from config.yaml path
- Splits into train/test
- Optimizes hyperparameters with Optuna (50 trials, 5-fold CV)
- Trains Ridge regression
- Evaluates with MAE, MSE, RMSE, R²
- Logs to MLflow with signature and visualizations

**Options:**
- `--run-name NAME` - Custom MLflow run name

**Example:**
```bash
uv run wine-quality train --run-name baseline-v1
```

### 2. Launch MLflow UI

```bash
uv run wine-quality mlflow-ui
```

Opens MLflow experiment tracking interface.

**Options:**
- `--host HOST` - Host address (default: 127.0.0.1)
- `--port PORT` - Port number (default: 5000)

## Example Workflow

```bash
# Run training pipeline
uv run wine-quality train --run-name experiment-v1

# Output shows:
# - Optuna optimization progress (50 trials)
# - Train/test metrics (MAE, MSE, RMSE, R²)
# - MLflow run ID

# View results in MLflow UI
uv run wine-quality mlflow-ui
# Open http://localhost:5000

# Load model in Python
import mlflow
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
predictions = model.predict(X_new)  # Accepts raw features
```

## Technical Details

### Dataset

The [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from UCI ML Repository contains:
- Red wine: 1,599 samples
- White wine: 4,898 samples
- Features: 11 physicochemical properties (acidity, sugar, pH, alcohol, etc.)
- Target: Quality score (continuous, range 3-8 for red wine)

### Model

- **Algorithm**: Ridge Regression (L2 regularization)
- **Target Transformation**: MinMaxScaler fitted to [0, 10] range to constrain predictions
- **Preprocessing**: StandardScaler for all 11 numerical features
- **Hyperparameter Optimization**: Optuna with 50 trials, 5-fold CV
- **Search Space**: alpha from 0.001 to 100.0 (log scale)
- **Scoring Metric**: Negative MAE (minimization)
- **Architecture**: Complete pipeline (StandardScaler → TransformedTargetRegressor(Ridge))

### MLflow Tracking

**Datasets Logged:**
- **Training dataset** (context="training")
  - Name: wine_quality_train
  - Rows: 1,279 samples
  - Schema: 12 columns (11 features + quality target)
  - Source: data/raw/winequality-red.csv
  - Digest: SHA256 hash for versioning
- **Test dataset** (context="testing")
  - Name: wine_quality_test
  - Rows: 320 samples
  - Schema: 12 columns (11 features + quality target)
  - Source: data/raw/winequality-red.csv
  - Digest: SHA256 hash for versioning

**Parameters Logged:**
- Model: type, alpha (best from Optuna)
- Data: test_size, random_state
- Optimization: cv_folds, n_trials

**Metrics Logged:**
- Training: MAE, MSE, RMSE, R²
- Test: MAE, MSE, RMSE, R²

**Model Artifacts:**
- **model**: Complete sklearn pipeline with signature and input example
- **Signature**: Input schema (11 features with types) and output type
- **Input Example**: 5 sample rows for deployment testing

**Visualizations (PNG):**
1. predictions_vs_actual.png - Scatter plot with perfect prediction line
2. residuals.png - Residual analysis
3. feature_coefficients.png - Ridge coefficients (feature importance)
4. distribution.png - Actual vs predicted distributions

### Evaluation Metrics

**Regression Metrics:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

**Visualizations:**
- Predictions vs Actual Values (scatter plot)
- Residual Plot (error analysis)
- Feature Coefficients (Ridge weights)
- Distribution Comparison (actual vs predicted histograms)

## Configuration

All configuration is centralized in `config.yaml` at the project root, parsed by Pydantic models in `config.py`.

### config.yaml Structure

```yaml
data:
  path: "data/raw/winequality-red.csv"
  test_size: 0.2
  random_state: 42

model:
  type: "Ridge"
  quality_score_min: 0
  quality_score_max: 10

optimization:
  n_trials: 50
  cv_folds: 5
  alpha_min: 0.001
  alpha_max: 100.0

mlflow:
  experiment_name: "wine_quality_regression"
  tracking_uri: "mlruns"
```

### Usage in Code

```python
from wine_quality_mlops.config import config

data_path = config.data.path  # Full IDE autocomplete
n_trials = config.optimization.n_trials
```

### Benefits

- **Type-safe**: Pydantic validates all values
- **IDE-friendly**: Nested attribute access with autocomplete
- **No code changes**: Modify config.yaml to change behavior
- **Validation**: Automatic range checks (e.g., test_size ∈ [0, 1])

## Architecture

### 5-Module Design

The codebase is organized into 5 focused modules:

**1. config.py** - Pydantic configuration manager
- Loads and validates config.yaml
- Provides type-safe nested access

**2. data.py** - Data operations
- `load_wine_quality_data()` - CSV loading
- `split_train_test()` - Train/test splitting

**3. modeling.py** - Complete ML workflow
- `build_pipeline()` - Create sklearn pipeline with target transformation
- `optimize_hyperparameters()` - Optuna optimization
- `train_model()` - Train with best params
- `evaluate_model()` - Calculate metrics

**4. visualization.py** - Plot generation
- 4 plotting functions returning matplotlib figures
- Ready for MLflow logging

**5. cli.py** - Command-line interface
- `train` command - Orchestrates full pipeline
- `mlflow-ui` command - Launch experiment viewer

### Design Philosophy

- **Minimal complexity**: 5 modules, not 10+
- **Configuration-driven**: YAML over hardcoded values
- **Type-safe**: Pydantic + type hints throughout
- **Single responsibility**: Each module has one clear purpose
- **Testable**: Independent, focused modules

## Reproducibility

The pipeline ensures reproducibility through:
- Fixed random seeds (config.data.random_state)
- Versioned dependencies managed by uv and pyproject.toml
- MLflow tracking of all parameters, metrics, and artifacts
- Consistent preprocessing with scikit-learn pipelines
- Model signatures for input/output validation

## Requirements

- Python: 3.10 or higher
- uv: Latest version (for dependency management)
- System dependencies: None (all Python packages)

## Development

### Install in Development Mode

```bash
uv sync
```

### Code Quality

The codebase follows:
- Type hints on all functions
- Google-style docstrings
- Pydantic for configuration validation
- Configuration-driven design
- Error handling with informative messages
- PEP 8 style guide

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit a Pull Request if you find obvious improvement. The goal here is to share!

## Acknowledgments

- Dataset: UCI Machine Learning Repository - Wine Quality Dataset
- Framework: scikit-learn, MLflow, Typer, Pydantic, Evidently, Optuna
- Inspiration: Production ML best practices
