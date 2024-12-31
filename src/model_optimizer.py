#model_optimizer.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class ModelOptimizer:
    """
    Automates the selection and optimization of machine learning models for classification problems.
    """
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {
            'classification': [
                ('RandomForest', RandomForestClassifier()),
                ('GradientBoosting', GradientBoostingClassifier()),
                ('LogisticRegression', LogisticRegression(max_iter=1000)),
                ('SVM', SVC(probability=True))
            ],
        }

    def optimize_model(self, X, y, param_grids=None):
        """
        Optimize models using GridSearchCV with parameter grids.

        :param X: Feature matrix.
        :param y: Target vector.
        :param param_grids: A dictionary containing parameter grids for each model.
        :return: Best model and its corresponding score.
        """
        if param_grids is None:
            param_grids = {
                'RandomForest': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'max_depth': [None, 10], 'class_weight': ['balanced']},
                'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
                'LogisticRegression': {'C': [0.1, 1, 10], 'class_weight': ['balanced']},
                'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            }

        best_model = None
        best_score = 0

        for name, model in tqdm(self.models.get(self.problem_type, []), desc="Model Optimization"):
            grid_search = GridSearchCV(model, param_grids.get(name, {}), scoring='accuracy', cv=3)
            grid_search.fit(X, y)
            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

        return best_model, best_score