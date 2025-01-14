# FeatureFlex: An AutoML Project with Advanced Feature Selection

![PyPI - Version](https://img.shields.io/pypi/v/FeatureFlex)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/FeatureFlex)
![GitHub - License](https://img.shields.io/github/license/SaintAngeLs/CS-MINI-2024Z-AutoML_project_2)
![GitHub - Issues](https://img.shields.io/github/issues/SaintAngeLs/CS-MINI-2024Z-AutoML_project_2)
![GitHub - Forks](https://img.shields.io/github/forks/SaintAngeLs/CS-MINI-2024Z-AutoML_project_2?style=social)

## Overview
FeatureFlex is an AutoML project that provides a comprehensive suite of machine learning capabilities. It includes advanced preprocessing, feature selection, model optimization, and evaluation functionalities, making it a robust choice for tackling classification tasks with large and complex datasets.

This package is particularly suited for tasks requiring feature selection and comparison across multiple methods, including SHAP, Boruta, SelectKBest, and more.

---

## Features

- **Advanced Feature Selection**:
  - Boruta
  - SelectKBest
  - SHAP-based feature selection
  - ReliefF (via scikit-rebate)

- **Dynamic Model Optimization**:
  - Grid search
  - Random search
  - Bayesian optimization

- **Evaluation Metrics**:
  - AUC, Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC and Precision-Recall Curves

- **Comparison Tool**:
  - Compare feature selection methods based on model performance.

---

## Installation

Install FeatureFlex via PyPI:

```bash
pip install FeatureFlex
```

---

## Usage

### Preprocessing Data

FeatureFlex includes a `DataPreprocessor` for preprocessing data with missing values, scaling, and encoding.

```python
from preprocessing import DataPreprocessor

data = ...  # Load your dataset
preprocessor = DataPreprocessor()
X, y, _ = preprocessor.preprocess(data, target_column="click")
```

### Feature Selection

FeatureFlex allows you to use various feature selection techniques:

```python
from feature_selector import EnhancedFeatureSelector

# Using SHAP-based feature selection
selector = EnhancedFeatureSelector(input_dim=X.shape[1])
top_features = selector.select_via_shap(X, y, n_features=10)
```

### Model Optimization

Optimize models using dynamic, grid, random, or Bayesian search:

```python
from model_optimizer import ModelOptimizer

optimizer = ModelOptimizer()
best_model, best_score = optimizer.optimize_model(X, y, method="bayesian")
```

### Comparison of Feature Selection Methods

Compare different feature selection methods using the provided comparison script:

```python
from comparison import compare_feature_selectors

results = compare_feature_selectors(data, target_column="click", n_features=10)
```

### Evaluation

Evaluate your model with various metrics and generate reports:

```python
from evaluation import ModelEvaluator

metrics = ModelEvaluator.evaluate(model, X_test, y_test, output_format="html")
```

---

## Example Dataset

The package includes utilities tested with datasets such as:

- [Avazu CTR Prediction Dataset (50k random rows)](https://www.kaggle.com/datasets/gauravduttakiit/avazu-ctr-prediction-with-random-50k-rows)

For more information on the dataset's context, see:
- [AutoML Research](https://arxiv.org/pdf/2204.09078)

---

## Comparison with Existing Packages

FeatureFlex distinguishes itself from existing feature selection packages:

| **Method**       | **FeatureFlex** | **Boruta** | **SelectKBest** | **SHAP** | **ReliefF** |
|------------------|-----------------|------------|-----------------|----------|-------------|
| Multi-method support | ✔          | ✘        | ✘             | ✘      | ✘         |
| Dynamic optimization | ✔      | ✘        | ✘             | ✘      | ✘         |
| Scalable to large datasets | ✔  | ✔        | ✔             | ✔      | ✔         |
| Integrated evaluation  | ✔    | ✘        | ✘             | ✘      | ✘         |

---

## Full Comparison Script

Here is an example script to compare various feature selection techniques:

```python
from comparison import compare_feature_selectors, save_results_and_plots
import pandas as pd

# Load dataset
data = pd.read_csv("path/to/dataset.csv")
results = compare_feature_selectors(data, target_column="click", n_features=10)

# Save results to CSV and generate plots
save_results_and_plots(results)
```

---

## Dependencies

FeatureFlex depends on the following libraries:

- **Core**:
  - `numpy`, `pandas`, `scikit-learn`
  - `matplotlib`, `shap`, `optuna`

- **Feature Selection**:
  - `BorutaPy`
  - `scikit-rebate`

- **Optimization**:
  - `optuna`

Refer to the `requirements.txt` file for the full list of dependencies.

---

## Project Links

- **GitHub**: [FeatureFlex Repository](https://github.com/SaintAngeLs/CS-MINI-2024Z-AutoML_project_2)
- **PyPI**: [FeatureFlex on PyPI](https://pypi.org/project/FeatureFlex/)

---

## License

This project is licensed under the MIT License.

---

## Contact

For queries or suggestions, contact the author at **info@itsharppro.com**.

---

## Changelog

FeatureFlex uses a dynamic versioning system. The current version is updated automatically at build time.

---

## Contribution

Contributions are welcome! Feel free to fork the repository and submit a pull request.

