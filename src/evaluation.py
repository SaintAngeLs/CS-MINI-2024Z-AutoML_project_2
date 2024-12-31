# evaluation.py
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class ModelEvaluator:
    """
    Class to handle evaluation of machine learning models using multiple metrics.
    """
    @staticmethod
    def evaluate(model, X, y):
        """
        Evaluate the model using ROC AUC, Accuracy, Precision, Recall, and F1-Score.
        
        :param model: Trained model.
        :param X: Feature matrix for evaluation.
        :param y: True target values.
        :return: Dictionary of evaluation metrics.
        """
        print("Evaluating model predictions...")
        predictions = model.predict(X)
        probabilities = (
            model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else predictions
        )

        # Avoid division by zero in Precision, Recall, and F1-Score
        unique_preds = set(predictions)
        if len(unique_preds) < 2:
            print("Warning: Model predicts only one class. Adjusting metrics to avoid zero-division.")
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(y, predictions, zero_division=0)
            recall = recall_score(y, predictions, zero_division=0)
            f1 = f1_score(y, predictions, zero_division=0)

        return {
            "AUC": float(roc_auc_score(y, probabilities)),
            "Accuracy": float(accuracy_score(y, predictions)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
        }
