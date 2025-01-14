from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import optuna


class ModelOptimizer:
    """
    Automates the selection and optimization of machine learning models for classification problems,
    supporting grid search, random search, and Bayesian optimization.
    """
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type

        self.models = {
            'classification': [
                ('RandomForest', RandomForestClassifier(random_state=42)),
                ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
                ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
                ('SVM', SVC(probability=True, random_state=42)),
                ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=42)),
                ('KNN', KNeighborsClassifier())
            ],
        }

    def optimize_model(self, X, y, param_grids=None, method="grid", n_iter_random=20, n_trials_bayes=50):
        """
        Optimize models using the specified optimization method: grid search, random search, or Bayesian optimization.

        :param X: Feature matrix (training).
        :param y: Target vector (training).
        :param param_grids: Optional dict of parameter grids for each model.
        :param method: Optimization method: "grid", "random", or "bayesian".
        :param n_iter_random: Number of iterations for random search (default: 20).
        :param n_trials_bayes: Number of trials for Bayesian optimization (default: 50).
        :return: (best_model, best_score) => Best model and its cross-validated score.
        """
        scoring_metric = "roc_auc"
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Convert to NumPy arrays for consistency
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if param_grids is None:
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'class_weight': ['balanced', None]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [3, 5, 7]
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10],
                    'class_weight': ['balanced', None]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [3, 5, 7],
                    'scale_pos_weight': [1, 2]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }

        best_model = None
        best_score = 0.0

        for name, model in tqdm(self.models.get(self.problem_type, []), desc="Model Optimization"):
            try:
                if method == "grid":
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids.get(name, {}),
                        scoring=scoring_metric,
                        cv=skf,
                    )
                    grid_search.fit(X, y)
                    score = grid_search.best_score_
                    if score > best_score:
                        best_model = grid_search.best_estimator_
                        best_score = score

                elif method == "random":
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grids.get(name, {}),
                        n_iter=n_iter_random,
                        scoring=scoring_metric,
                        cv=skf,
                        random_state=42
                    )
                    random_search.fit(X, y)
                    score = random_search.best_score_
                    if score > best_score:
                        best_model = random_search.best_estimator_
                        best_score = score

                elif method == "bayesian":
                    def objective(trial):
                        params = {
                            param: trial.suggest_categorical(param, values)
                            for param, values in param_grids.get(name, {}).items()
                        }
                        model.set_params(**params)
                        scores = []
                        for train_idx, val_idx in skf.split(X, y):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            model.fit(X_train, y_train)
                            score = model.score(X_val, y_val)
                            scores.append(score)
                        return np.mean(scores)

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_trials_bayes)

                    trial_params = study.best_params
                    model.set_params(**trial_params)
                    model.fit(X, y)
                    score = study.best_value
                    if score > best_score:
                        best_model = model
                        best_score = score
                
                elif method == "dynamic":
                    # Dynamically select method based on problem size
                    n_samples, n_features = X.shape
                    if n_samples * n_features <= 10000:
                        chosen_method = "grid"
                    elif n_samples * n_features <= 100000:
                        chosen_method = "random"
                    else:
                        chosen_method = "bayesian"

                    best_model, best_score = self.optimize_model(
                        X, y, param_grids=param_grids, method=chosen_method,
                        n_iter_random=n_iter_random, n_trials_bayes=n_trials_bayes
                    )

            except Exception as e:
                print(f"Error optimizing model {name}: {e}")
                continue

        if best_model is None:
            raise ValueError("No model could be successfully trained. Please check the data and parameter grids.")

        return best_model, best_score

