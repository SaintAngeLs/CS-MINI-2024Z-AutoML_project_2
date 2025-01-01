# comparison.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from boruta import BorutaPy
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import shap
from autofeat import AutoFeatClassifier

from preprocessing import DataPreprocessor
from feature_selector import EnhancedFeatureSelector


def compare_feature_selectors(data, target_column, n_features=10):
    """
    Compares different feature selection methods on the given dataset.
    Methods included:
      - Custom SHAP-based solution (EnhancedFeatureSelector)
      - Boruta
      - SelectKBest
      - AutoFeat

    Returns a dictionary of the results, each key being a method, 
    and each value a dict with "AUC" and "Accuracy".
    """
    results = {}

    # ----------------------------------------------------------------
    # 1) Preprocessing
    # ----------------------------------------------------------------
    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    X, y, _ = preprocessor.preprocess(data, target_column)

    if hasattr(X, "toarray"):
        X = X.toarray()

    print("Removing constant or zero-variance features...")
    vt = VarianceThreshold(threshold=0.0)
    X = vt.fit_transform(X)

    if np.isnan(X).any():
        print("Imputing any remaining NaNs with column means...")
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]

    # ----------------------------------------------------------------
    # 2) Train-test split
    # ----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------------------------------------------
    # Baseline Model (No Feature Selection)
    # ----------------------------------------------------------------
    print("Training baseline model without feature selection...")
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
    results['Baseline'] = {
        "AUC": baseline_auc,
        "Accuracy": accuracy_score(y_test, baseline_preds)
    }

    # ----------------------------------------------------------------
    # 3) SHAP-based Feature Selection
    # ----------------------------------------------------------------
    print("Testing SHAP-based feature selection...")
    selector = EnhancedFeatureSelector(input_dim=X_train.shape[1])
    top_features = selector.select_via_shap(X_train, y_train, n_features=n_features)

    X_train_shap = X_train[:, top_features]
    X_test_shap  = X_test[:,  top_features]

    shap_model = RandomForestClassifier(random_state=42)
    shap_model.fit(X_train_shap, y_train)
    shap_preds = shap_model.predict(X_test_shap)
    shap_auc = roc_auc_score(y_test, shap_model.predict_proba(X_test_shap)[:, 1])
    results['SHAP'] = {
        "AUC": shap_auc,
        "Accuracy": accuracy_score(y_test, shap_preds)
    }

    # ----------------------------------------------------------------
    # 4) Boruta
    # ----------------------------------------------------------------
    print("Testing Boruta feature selection...")
    boruta_selector = BorutaPy(
        estimator=RandomForestClassifier(random_state=42),
        n_estimators='auto',
        random_state=42
    )
    boruta_selector.fit(X_train, y_train)
    X_train_boruta = X_train[:, boruta_selector.support_]
    X_test_boruta  = X_test[:,  boruta_selector.support_]

    boruta_model = RandomForestClassifier(random_state=42)
    boruta_model.fit(X_train_boruta, y_train)
    boruta_preds = boruta_model.predict(X_test_boruta)
    boruta_auc = roc_auc_score(y_test, boruta_model.predict_proba(X_test_boruta)[:, 1])
    results['Boruta'] = {
        "AUC": boruta_auc,
        "Accuracy": accuracy_score(y_test, boruta_preds)
    }

    # ----------------------------------------------------------------
    # 5) SelectKBest
    # ----------------------------------------------------------------
    print("Testing SelectKBest feature selection...")
    try:
        skb_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_skb = skb_selector.fit_transform(X_train, y_train)
        X_test_skb  = skb_selector.transform(X_test)

        skb_model = RandomForestClassifier(random_state=42)
        skb_model.fit(X_train_skb, y_train)
        skb_preds = skb_model.predict(X_test_skb)
        skb_auc = roc_auc_score(y_test, skb_model.predict_proba(X_test_skb)[:, 1])
        results['SelectKBest'] = {
            "AUC": skb_auc,
            "Accuracy": accuracy_score(y_test, skb_preds)
        }
    except ValueError as e:
        print(f"SelectKBest encountered an issue: {e}")
        results['SelectKBest'] = {"AUC": None, "Accuracy": None}

    # ----------------------------------------------------------------
    # 6) AutoFeat
    # ----------------------------------------------------------------
    # print("Testing AutoFeat...")
    # autofeat_model = AutoFeatClassifier(verbose=0)

    # # Fit can raise errors if data has leftover NaNs
    # autofeat_model.fit(X_train, y_train)

    # X_train_autofeat = autofeat_model.transform(X_train)
    # X_test_autofeat  = autofeat_model.transform(X_test)

    # # Then train a random forest on the transformed features
    # autofeat_rf_model = RandomForestClassifier(random_state=42)
    # autofeat_rf_model.fit(X_train_autofeat, y_train)
    # autofeat_preds = autofeat_rf_model.predict(X_test_autofeat)
    # autofeat_auc = roc_auc_score(y_test, autofeat_rf_model.predict_proba(X_test_autofeat)[:, 1])
    # results['AutoFeat'] = {
    #     "AUC": autofeat_auc,
    #     "Accuracy": accuracy_score(y_test, autofeat_preds)
    # }

    # ----------------------------------------------------------------
    # Print Results
    # ----------------------------------------------------------------
    print("\nComparison Results:")
    for method, metrics in results.items():
        auc = metrics.get('AUC', 'N/A')
        acc = metrics.get('Accuracy', 'N/A')
        print(f"{method}: AUC={auc}, Accuracy={acc}")

    return results


# Example Usage
if __name__ == "__main__":
    # Example dataset
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
