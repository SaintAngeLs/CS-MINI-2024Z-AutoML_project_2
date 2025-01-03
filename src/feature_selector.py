# feature_selector.py

import torch
import torch.nn as nn
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from model_optimizer import ModelOptimizer

class EnhancedFeatureSelector(nn.Module):
    """
    Dynamically selects features using a trainable controller mechanism 
    and supports multiple ways to pick top features.
    """
    def __init__(self, input_dim):
        super(EnhancedFeatureSelector, self).__init__()
        # 'alpha' is a trainable parameter if you want to do 
        # gradient-based selection. (Optional usage.)
        self.alpha = nn.Parameter(torch.rand(input_dim))

    def forward(self, x):
        """
        Forward pass to calculate feature selection probabilities and apply them.
        """
        probabilities = torch.sigmoid(self.alpha)
        # Expand to match x's batch dimension
        probabilities = probabilities.unsqueeze(0).expand(x.shape[0], -1)
        selected_features = probabilities * x
        return selected_features, probabilities

    # ----------------------------------------------------------------
    # 1) Original SHAP-based Method
    # ----------------------------------------------------------------
    @staticmethod
    def select_via_shap(X, y, n_features=10):
        """
        Use SHAP to select top n_features based on feature importance 
        from a RandomForestClassifier.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Train a quick RandomForest
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # SHAP Explainer
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
        shap_values = np.array(shap_values)

        # If binary => shap_values has shape (2, n_samples, n_features)
        # Average across classes => (n_samples, n_features)
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=0)

        feature_importances = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(feature_importances)[-n_features:]
        return top_features

    # ----------------------------------------------------------------
    # 2) Model-Optimizer-based Method
    # ----------------------------------------------------------------
    @staticmethod
    def select_via_model_optimizer(X, y, n_features=10, param_grids=None):
        """
        1) Use your ModelOptimizer to find the BEST classifier 
           across multiple models/hyperparams using the FULL feature set.
        2) Extract feature importance or coefficients from that best model.
        3) Return the top N feature indices.

        NOTE: Some models (KNN, SVM) do not expose 'feature_importances_' or 'coef_'.
              If such a model is best, raise an error or handle gracefully.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        # 1) Optimize across multiple classifiers
        optimizer = ModelOptimizer()
        best_model, best_score = optimizer.optimize_model(X, y, param_grids=param_grids)

        # 2) Extract feature importances or coefficients
        feature_importances = None
        model_name = type(best_model).__name__  # e.g. 'RandomForestClassifier'

        if hasattr(best_model, "feature_importances_"):
            # e.g. RandomForestClassifier, GradientBoostingClassifier, XGBClassifier
            feature_importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            # e.g. LogisticRegression
            # 'coef_' shape => (1, n_features) or (n_classes, n_features)
            coefs = best_model.coef_
            # If multiclass => average across classes, or pick 1
            if coefs.ndim == 2:
                # We'll just do mean across classes
                coefs = np.mean(coefs, axis=0)
            feature_importances = np.abs(coefs)
        else:
            # e.g. SVC(kernel='rbf'), KNN, etc. => no direct feature importance
            msg = (f"Best model is {model_name} which does not expose feature importances. "
                   "Cannot proceed with feature-based selection.")
            raise ValueError(msg)

        if feature_importances is None:
            raise ValueError(f"Could not obtain feature importances from model {model_name}")

        # 3) Sort and pick top N
        top_features = np.argsort(feature_importances)[-n_features:]
        return top_features
