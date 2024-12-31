# main.py
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from preprocessing import DataPreprocessor
from feature_selector import EnhancedFeatureSelector
from model_optimizer import ModelOptimizer
from evaluation import ModelEvaluator

def main():
    # Path to your dataset
    dataset_path = "../data/50krecords.csv"

    # Load dataset in chunks to avoid memory issues (if needed)
    chunksize = 10000
    data_chunks = []
    with pd.read_csv(dataset_path, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc="Loading Dataset Chunks"):
            data_chunks.append(chunk)

    # Combine chunks into a single dataframe
    data = pd.concat(data_chunks, ignore_index=True)

    # Columns in the dataset (subset you care about)
    columns = [
        'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
        'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
        'device_ip', 'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    data = data[columns]

    # Identify the target
    target_column = "click"

    # -----------------------------
    # 1) Preprocess
    # -----------------------------
    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    X, y, _ = preprocessor.preprocess(data, target_column)

    # -----------------------------
    # 2) Train-Val Split
    # -----------------------------
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3) SHAP Feature Selection
    # -----------------------------
    print("Performing feature selection...")
    # a) Create a feature-selector with the original dimensionality
    selector = EnhancedFeatureSelector(input_dim=X_train.shape[1])
    # b) Select top N features using SHAP
    top_features = selector.select_via_shap(X_train, y_train, n_features=10)

    # c) Re-initialize with the reduced input dimension
    reduced_selector = EnhancedFeatureSelector(input_dim=len(top_features))

    # If X is sparse, convert to dense and slice columns
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()[:, top_features]  # shape => (train_size, 10)
        X_val_dense   = X_val.toarray()[:, top_features]    # shape => (val_size, 10)
    else:
        X_train_dense = X_train[:, top_features]
        X_val_dense   = X_val[:, top_features]

    # -----------------------------
    # 4) Handle Imbalance via SMOTE
    # -----------------------------
    print("Applying SMOTE on the training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_dense, y_train)

    # Optional: pass the oversampled data to reduced_selector 
    # (if you want a trainable approach for alpha)
    X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
    selected_features, probabilities = reduced_selector(X_train_tensor)

    # -----------------------------
    # 5) Model Optimization
    # -----------------------------
    print("Optimizing models...")
    optimizer = ModelOptimizer()
    best_model, best_score = optimizer.optimize_model(X_train_res, y_train_res)
    print(f"Best Model Score (CV AUC): {best_score}")

    # -----------------------------
    # 6) Final Evaluation
    # -----------------------------
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate(best_model, X_val_dense, y_val)
    print("Evaluation Results:", evaluation_results)


if __name__ == "__main__":
    main()
