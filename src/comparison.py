# comparison.py

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from boruta import BorutaPy
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

import shap
from skrebate import ReliefF

from preprocessing import DataPreprocessor
from feature_selector import EnhancedFeatureSelector

def compare_feature_selectors(data, target_column, n_features=10):
    """
    Compares various feature selection methods on the given dataset:
      1) Baseline (no feature selection)
      2) FeatureFlex-based (EnhancedFeatureSelector)
      3) Boruta
      4) SelectKBest
      5) ReliefF (scikit-rebate)

    Returns a dictionary of results: e.g.
      {
        "Baseline": {"AUC": 0.68, "Accuracy": 0.85},
        "SHAP":     {"AUC": 0.47, "Accuracy": 0.73},
        "Boruta":   {"AUC": 0.66, "Accuracy": 0.83},
        ...
      }
    """
    results = {}

    # 1) Preprocess data
    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    X, y, _ = preprocessor.preprocess(data, target_column)

    y = y.values

    if hasattr(X, "toarray"):
        X = X.toarray()

    print("Removing constant or zero-variance features...")
    vt = VarianceThreshold(threshold=0.0)
    X = vt.fit_transform(X)

    # Impute leftover NaNs if any
    if np.isnan(X).any():
        print("Imputing leftover NaNs with column means...")
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Baseline (no selection)
    # ---------------------------
    print("Training baseline model without feature selection...")
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)

    baseline_preds = baseline_model.predict(X_test)
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
    baseline_acc = accuracy_score(y_test, baseline_preds)
    results["Baseline"] = {"AUC": baseline_auc, "Accuracy": baseline_acc}

    # ---------------------------
    # FeatureFlex-based (EnhancedFeatureSelector)
    # ---------------------------
    print("Testing FeatureFlex-based (EnhancedFeatureSelector) feature selection...")
    model_opt_selector = EnhancedFeatureSelector(input_dim=X_train.shape[1])
    # top_features = model_opt_selector.select_via_shap(X_train, y_train, n_features=n_features)
    top_features = model_opt_selector.select_via_model_optimizer(X_train, y_train, n_features=n_features)

    X_train_shap = X_train[:, top_features]
    X_test_shap  = X_test[:, top_features]

    shap_model = RandomForestClassifier(random_state=42)
    shap_model.fit(X_train_shap, y_train)

    shap_preds = shap_model.predict(X_test_shap)
    shap_auc   = roc_auc_score(y_test, shap_model.predict_proba(X_test_shap)[:, 1])
    shap_acc   = accuracy_score(y_test, shap_preds)
    results["SHAP"] = {"AUC": shap_auc, "Accuracy": shap_acc}

    # ---------------------------
    # Boruta
    # ---------------------------
    print("Testing Boruta feature selection...")
    boruta_selector = BorutaPy(
        estimator=RandomForestClassifier(random_state=42),
        n_estimators='auto',
        random_state=42
    )
    boruta_selector.fit(X_train, y_train)

    X_train_boruta = X_train[:, boruta_selector.support_]
    X_test_boruta  = X_test[:, boruta_selector.support_]

    boruta_model = RandomForestClassifier(random_state=42)
    boruta_model.fit(X_train_boruta, y_train)

    boruta_preds = boruta_model.predict(X_test_boruta)
    boruta_auc   = roc_auc_score(y_test, boruta_model.predict_proba(X_test_boruta)[:, 1])
    boruta_acc   = accuracy_score(y_test, boruta_preds)
    results["Boruta"] = {"AUC": boruta_auc, "Accuracy": boruta_acc}

    # ---------------------------
    # SelectKBest
    # ---------------------------
    print("Testing SelectKBest feature selection...")
    try:
        skb_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_skb = skb_selector.fit_transform(X_train, y_train)
        X_test_skb  = skb_selector.transform(X_test)

        skb_model = RandomForestClassifier(random_state=42)
        skb_model.fit(X_train_skb, y_train)

        skb_preds = skb_model.predict(X_test_skb)
        skb_auc   = roc_auc_score(y_test, skb_model.predict_proba(X_test_skb)[:, 1])
        skb_acc   = accuracy_score(y_test, skb_preds)
        results["SelectKBest"] = {"AUC": skb_auc, "Accuracy": skb_acc}
    except ValueError as e:
        print(f"SelectKBest encountered an issue: {e}")
        results["SelectKBest"] = {"AUC": None, "Accuracy": None}

    # ---------------------------
    # ReliefF
    # ---------------------------
    print("Testing ReliefF feature selection...")
    try:
        relief_selector = ReliefF(n_features_to_select=n_features)
        relief_selector.fit(X_train, y_train)

        X_train_relief = relief_selector.transform(X_train)
        X_test_relief  = relief_selector.transform(X_test)

        relief_model = RandomForestClassifier(random_state=42)
        relief_model.fit(X_train_relief, y_train)

        relief_preds = relief_model.predict(X_test_relief)
        relief_auc   = roc_auc_score(y_test, relief_model.predict_proba(X_test_relief)[:, 1])
        relief_acc   = accuracy_score(y_test, relief_preds)
        results["ReliefF"] = {"AUC": relief_auc, "Accuracy": relief_acc}
    except ValueError as e:
        print(f"ReliefF encountered an issue: {e}")
        results["ReliefF"] = {"AUC": None, "Accuracy": None}

    # Print final results to console
    print("\nComparison Results:")
    for method, metrics in results.items():
        auc = metrics.get("AUC", "N/A")
        acc = metrics.get("Accuracy", "N/A")
        print(f"{method}: AUC={auc}, Accuracy={acc}")

    return results


def save_results_and_plots(results, csv_filename="comparison_results.csv", json_filename="comparison_results.json"):
    """
    Saves the 'results' dictionary to a CSV and JSON file, 
    then creates bar plots for AUC and Accuracy, saving them as PNG images.
    """
    df = pd.DataFrame.from_dict(results, orient="index")  # index=method, columns=AUC/Accuracy
    df.to_csv(csv_filename)
    print(f"Saved comparison results to {csv_filename}.")

    with open(json_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved comparison results to {json_filename}.")

    methods = df.index.tolist()
    auc_values = df["AUC"].tolist()
    acc_values = df["Accuracy"].tolist()

    plt.figure(figsize=(6, 4))
    plt.bar(methods, auc_values, color='skyblue')
    plt.title("AUC Comparison")
    plt.xlabel("Method")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig("comparison_auc.png")
    plt.close()
    print("Saved AUC bar chart as comparison_auc.png.")

    plt.figure(figsize=(6, 4))
    plt.bar(methods, acc_values, color='lightgreen')
    plt.title("Accuracy Comparison")
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("comparison_accuracy.png")
    plt.close()
    print("Saved Accuracy bar chart as comparison_accuracy.png.")


if __name__ == "__main__":
    dataset_path = "../data/50krecords.csv"
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)

    columns = [
        'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
        'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
        'device_ip', 'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    data = data[columns]
    target_column = "click"

    results = compare_feature_selectors(data, target_column, n_features=10)
    
    # Save results & generate plots
    save_results_and_plots(results, 
                           csv_filename="comparison_results.csv", 
                           json_filename="comparison_results.json")
