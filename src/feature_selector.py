import torch
import torch.nn as nn
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

class EnhancedFeatureSelector(nn.Module):
    """
    Dynamically selects features using a trainable controller mechanism and SHAP importance.
    """
    def __init__(self, input_dim):
        super(EnhancedFeatureSelector, self).__init__()
        # 'alpha' will learn weights (probabilities) for each input feature
        self.alpha = nn.Parameter(torch.rand(input_dim))

    def forward(self, x):
        """
        Forward pass to calculate feature selection probabilities and apply them.
        """
        # Convert alpha -> probabilities in [0,1] via sigmoid
        probabilities = torch.sigmoid(self.alpha)              # shape: (input_dim,)
        probabilities = probabilities.unsqueeze(0).expand(
            x.shape[0], -1
        )                                                      # shape: (batch_size, input_dim)
        selected_features = probabilities * x                  # element-wise multiplication
        return selected_features, probabilities

    @staticmethod
    def select_via_shap(X, y, n_features=10):
        """
        Use SHAP to select top n_features based on feature importance.
        - For binary classification, shap_values often has shape (2, n_samples, n_features).
        - We'll average across classes so that we end up with (n_samples, n_features).
        """
        # If X is sparse, convert to dense for the RandomForest and SHAP
        if hasattr(X, "toarray"):
            X = X.toarray()

        model = RandomForestClassifier()
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
        shap_values = np.array(shap_values)  # Ensure it's a NumPy array

        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=0)

        feature_importances = np.abs(shap_values).mean(axis=0)  # shape: (n_features,)
        top_features = np.argsort(feature_importances)[-n_features:]  # shape: (n_features,)
        return top_features
