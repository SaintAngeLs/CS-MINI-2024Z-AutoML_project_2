# feature_selector.py
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
        self.alpha = nn.Parameter(torch.rand(input_dim))

    def forward(self, x):
        """
        Forward pass to calculate feature selection probabilities and apply them.
        """
        probabilities = torch.sigmoid(self.alpha) 
        probabilities = probabilities.unsqueeze(0).expand(
            x.shape[0], -1
        )                                            
        selected_features = probabilities * x       
        return selected_features, probabilities

    @staticmethod
    def select_via_shap(X, y, n_features=10):
        """
        Use SHAP to select top n_features based on feature importance.
        For binary classification, shap_values often has shape (2, n_samples, n_features).
        We'll average across classes => (n_samples, n_features).
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=0)  

        feature_importances = np.abs(shap_values).mean(axis=0) 
        top_features = np.argsort(feature_importances)[-n_features:]
        return top_features
