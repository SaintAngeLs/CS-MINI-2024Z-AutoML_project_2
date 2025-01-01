from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

class ModelOptimizer:
    """
    Automates the selection and optimization of machine learning models for classification problems.
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

    def optimize_model(self, X, y, param_grids=None):
        """
        Optimize models using GridSearchCV with parameter grids, scoring by ROC AUC, 
        and a custom StratifiedKFold cross-validator.

        :param X: Feature matrix (training).
        :param y: Target vector (training).
        :param param_grids: Optional dict of parameter grids for each model.
        :return: (best_model, best_score) => best_score is the best CV AUC.
        """
        scoring_metric = "roc_auc" 

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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
                },
                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.01],
                    'scale_pos_weight': [1, 2],  
                    'max_depth': [3, 5]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }

        best_model = None
        best_score = 0.0

        for name, model in tqdm(self.models.get(self.problem_type, []), desc="Model Optimization"):
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids.get(name, {}),
                scoring=scoring_metric,
                cv=skf,  
            )
            grid_search.fit(X, y)

            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

        return best_model, best_score
