import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

REPORT_DIR = "report"  

class ModelEvaluator:
    """
    A flexible evaluation class that computes multiple metrics
    and can output results to console or an HTML file with additional plots.
    """
    @staticmethod
    def evaluate(
        model, 
        X, 
        y,
        output_format="console",
        output_filename="evaluation_report.html",
        output_path=None
    ):
        """
        Evaluate the model with metrics and optionally generate an HTML report 
        (plus plots) in a dedicated 'report/' directory or a user-specified path.

        :param model: Trained model.
        :param X: Feature matrix for evaluation.
        :param y: True target labels.
        :param output_format: How to output results. One of {"console", "html"}.
        :param output_filename: Name of the HTML file to create if output_format == "html".
        :param output_path: Directory path to store the report. Defaults to 'report/' if None.
        :return: Dictionary of evaluation metrics (always returned in Python).
        """
        print("Evaluating model predictions...")

        # Determine the output directory
        if output_path is None:
            output_path = REPORT_DIR

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # 1) Predictions & Probabilities
        predictions = model.predict(X)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            probabilities = predictions  # fallback for older models

        # 2) Check if model predicts only one class
        unique_preds = set(predictions)
        single_class = (len(unique_preds) < 2)

        # 3) Compute core metrics
        if single_class:
            print("Warning: Model predicts only one class. Adjusting metrics to avoid zero-division.")
            auc_val = 0.5
            acc_val = accuracy_score(y, predictions)
            prec_val = 0.0
            rec_val = 0.0
            f1_val = 0.0
        else:
            auc_val = roc_auc_score(y, probabilities)
            acc_val = accuracy_score(y, predictions)
            prec_val = precision_score(y, predictions, zero_division=0)
            rec_val = recall_score(y, predictions, zero_division=0)
            f1_val = f1_score(y, predictions, zero_division=0)

        metrics_dict = {
            "AUC": float(auc_val),
            "Accuracy": float(acc_val),
            "Precision": float(prec_val),
            "Recall": float(rec_val),
            "F1-Score": float(f1_val),
        }

        # 4) Confusion Matrix & Classification Report
        if not single_class:
            cm = confusion_matrix(y, predictions)
            cls_report = classification_report(y, predictions, zero_division=0, digits=4)
        else:
            cm = None
            cls_report = "Single-class prediction; no classification report."

        # 5) Decide output
        if output_format == "console":
            # Print to console
            ModelEvaluator._print_to_console(metrics_dict, cm, cls_report)
        else:
            # Build file paths
            html_output = os.path.join(output_path, output_filename)

            # 5a) Save confusion matrix plot if possible
            cm_img_path = None
            roc_img_path = None
            pr_img_path  = None

            if cm is not None:
                cm_img_path = os.path.join(output_path, "confusion_matrix.png")
                ModelEvaluator._plot_confusion_matrix(cm, cm_img_path)

            # 5b) Additional plots (ROC & Precision-Recall) if not single class
            if not single_class:
                roc_img_path = os.path.join(output_path, "roc_curve.png")
                pr_img_path  = os.path.join(output_path, "precision_recall_curve.png")
                ModelEvaluator._plot_roc_curve(y, probabilities, roc_img_path)
                ModelEvaluator._plot_precision_recall_curve(y, probabilities, pr_img_path)

            # 5c) Build HTML content
            html_content = ModelEvaluator._build_html_report(
                metrics_dict, cm, cls_report,
                cm_img_path=os.path.basename(cm_img_path) if cm_img_path else None,
                roc_img_path=os.path.basename(roc_img_path) if roc_img_path else None,
                pr_img_path=os.path.basename(pr_img_path) if pr_img_path else None,
                title="Evaluation Report"
            )

            # 5d) Write HTML to the specified folder
            with open(html_output, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"HTML report saved to: {html_output}")

        return metrics_dict

    @staticmethod
    def _print_to_console(metrics_dict, cm, cls_report):
        """
        Prints evaluation results to the console.
        """
        print("\n=== Evaluation Metrics ===")
        for k, v in metrics_dict.items():
            print(f"{k}: {v:.4f}")

        if cm is not None:
            print("\n=== Confusion Matrix ===")
            print(cm)
            print("\n=== Classification Report ===")
            print(cls_report)

    # ------------------------------------------------------------------
    # PLOT FUNCTIONS
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_confusion_matrix(cm, output_path):
        """
        Plots and saves the confusion matrix as a PNG file.
        """
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _plot_roc_curve(y_true, y_prob, output_path):
        """
        Plots the ROC curve given the true labels and predicted probabilities.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _plot_precision_recall_curve(y_true, y_prob, output_path):
        """
        Plots the Precision-Recall curve.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = ModelEvaluator._auc_of_pr(precision, recall)

        plt.figure(figsize=(4, 4))
        plt.plot(recall, precision, color='green', lw=2,
                 label=f"Precision-Recall (AUC={pr_auc:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _auc_of_pr(precision, recall):
        """
        Computes area under the Precision-Recall curve using the trapezoidal rule.
        """
        pr_auc = 0.0
        for i in range(1, len(precision)):
            pr_auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0
        return pr_auc

    # ------------------------------------------------------------------
    # HTML REPORT
    # ------------------------------------------------------------------
    @staticmethod
    def _build_html_report(
        metrics_dict, 
        cm, 
        cls_report,
        cm_img_path=None,
        roc_img_path=None,
        pr_img_path=None,
        title="Evaluation Report"
    ):
        """
        Constructs an HTML string with:
          - Key metrics
          - Confusion matrix image (if available)
          - ROC curve image (if available)
          - Precision-Recall curve image (if available)
          - Classification report
          
        The images are assumed to be in the same folder as the HTML file. 
        We'll reference them by their filenames (not absolute paths).
        """
        metrics_html = "<ul>"
        for k, v in metrics_dict.items():
            metrics_html += f"<li><strong>{k}:</strong> {v:.4f}</li>"
        metrics_html += "</ul>"

        if cm is not None and cm_img_path:
            cm_html = f'<h2>Confusion Matrix</h2><img src="{cm_img_path}" width="300">'
        else:
            cm_html = "<p>No confusion matrix (single-class prediction?).</p>"

        cls_html = f"<pre>{cls_report}</pre>" if cls_report else "<p>No classification report available.</p>"

        roc_html = f'<h2>ROC Curve</h2><img src="{roc_img_path}" width="300">' if roc_img_path else ""

        pr_html = f'<h2>Precision-Recall Curve</h2><img src="{pr_img_path}" width="300">' if pr_img_path else ""

        html_template = f"""
        <html>
          <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin: 5px 0; font-size: 16px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
                img {{ margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .key-metrics {{ padding: 10px; background: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 5px; }}
            </style>
          </head>
          <body>
            <h1>{title}</h1>
            <div class="key-metrics">
              <h2>Key Metrics</h2>
              {metrics_html}
            </div>
            {cm_html}
            {roc_html}
            {pr_html}
            <h2>Classification Report</h2>
            {cls_html}
          </body>
        </html>
        """
        return html_template
