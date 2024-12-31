from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class ModelOptimizer:
    """
    Automates the selection and optimization of machine learning models for classification problems.
    """
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        # Initialize your base models with random states for consistency
        self.models = {
            'classification': [
                ('RandomForest', RandomForestClassifier(random_state=42)),
                ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
                ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
                ('SVM', SVC(probability=True, random_state=42))
            ],
        }

    def optimize_model(self, X, y, param_grids=None):
        """
        Optimize models using GridSearchCV with parameter grids.

        :param X: Feature matrix (training).
        :param y: Target vector (training).
        :param param_grids: A dictionary containing parameter grids for each model.
        :return: (best_model, best_score) where best_score is the best CV score (AUC).
        """
        # We'll optimize based on AUC
        scoring_metric = "roc_auc"  # <-- Use the built-in 'roc_auc'

        if param_grids is None:
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'class_weight': ['balanced']
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.01]
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10],
                    'class_weight': ['balanced']
                },
                'SVM': {
                    'C': [0.1, 1],
                    'kernel': ['linear', 'rbf']
                }
            }

        best_model = None
        best_score = 0.0

        # Iterate over each model type
        for name, model in tqdm(self.models.get(self.problem_type, []), desc="Model Optimization"):
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids.get(name, {}),
                scoring=scoring_metric,  # <--- using 'roc_auc'
                cv=3
            )
            grid_search.fit(X, y)

            # GridSearchCV.best_score_ will be the mean cross-validated ROC AUC
            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

        return best_model, best_score
