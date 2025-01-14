import torch
import torch.nn as nn
import numpy as np
import shap
from .model_optimizer import ModelOptimizer
from sklearn.ensemble import RandomForestClassifier


class EnhancedFeatureSelector(nn.Module):
    """
    Dynamically selects features using a trainable controller mechanism 
    and supports multiple optimization methods for feature selection.
    """
    def __init__(self, input_dim):
        super(EnhancedFeatureSelector, self).__init__()
        # Trainable parameter for gradient-based feature selection
        self.alpha = nn.Parameter(torch.rand(input_dim))

    def forward(self, x):
        """
        Forward pass to calculate feature selection probabilities and apply them.
        """
        probabilities = torch.sigmoid(self.alpha)
        probabilities = probabilities.unsqueeze(0).expand(x.shape[0], -1)
        selected_features = probabilities * x
        return selected_features, probabilities

    @staticmethod
    def select_via_shap(X, y, n_features=10):
        """
        Use SHAP to select top n_features based on feature importance from a RandomForestClassifier.
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

        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=0)

        feature_importances = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(feature_importances)[-n_features:]
        return top_features

    @staticmethod
    def select_via_model_optimizer(X, y, n_features=10, param_grids=None, method="dynamic"):
        """
        Select top features using ModelOptimizer by dynamically choosing the optimization method.

        :param X: Feature matrix.
        :param y: Target labels.
        :param n_features: Number of top features to select.
        :param param_grids: Hyperparameter grids for optimization.
        :param method: Optimization method: "grid", "random", "bayesian", or "dynamic".
        :return: Indices of top N features.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Determine optimization method dynamically if set to "dynamic"
        if method == "dynamic":
            n_samples, n_features_total = X.shape
            if n_features_total <= 50:
                method = "grid"  # Small feature set, exhaustive grid search
            elif 50 < n_features_total <= 200:
                method = "random"  # Medium feature set, random search
            else:
                method = "bayesian"  # Large feature set, Bayesian optimization

        optimizer = ModelOptimizer()
        best_model, best_score = optimizer.optimize_model(X, y, param_grids=param_grids, method=method)

        # Extract feature importances or coefficients
        feature_importances = None
        model_name = type(best_model).__name__

        if hasattr(best_model, "feature_importances_"):
            feature_importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            coefs = best_model.coef_
            if coefs.ndim == 2:
                coefs = np.mean(coefs, axis=0)
            feature_importances = np.abs(coefs)
        else:
            raise ValueError(f"Model {model_name} does not support feature importances.")

        if feature_importances is None:
            raise ValueError(f"Could not extract feature importances from {model_name}.")

        # Select top features
        top_features = np.argsort(feature_importances)[-n_features:]
        return top_features
