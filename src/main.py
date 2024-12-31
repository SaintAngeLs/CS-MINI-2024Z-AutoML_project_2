import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessor
from feature_selector import EnhancedFeatureSelector
from model_optimizer import ModelOptimizer
from evaluation import ModelEvaluator
from tqdm import tqdm

def main():
    # Load dataset from local path
    dataset_path = "../data/50krecords.csv"

    chunksize = 10000
    data_chunks = []
    with pd.read_csv(dataset_path, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc="Loading Dataset Chunks"):
            data_chunks.append(chunk)

    data = pd.concat(data_chunks, ignore_index=True)

    columns = [
        'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model',
        'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    data = data[columns]

    target_column = "click"  # The target column for prediction

    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    X, y, _ = preprocessor.preprocess(data, target_column)

    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Performing feature selection...")
    selector = EnhancedFeatureSelector(input_dim=X_train.shape[1])

    top_features = selector.select_via_shap(X_train, y_train, n_features=10)

    reduced_selector = EnhancedFeatureSelector(input_dim=len(top_features))

    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()[:, top_features]  # shape: (batch_size, 10)
        X_val_dense   = X_val.toarray()[:, top_features]    # shape: (batch_size, 10)
    else:
        X_train_dense = X_train[:, top_features]
        X_val_dense   = X_val[:, top_features]

    X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)

    selected_features, probabilities = reduced_selector(X_train_tensor)


    print("Optimizing models...")
    optimizer = ModelOptimizer()
    best_model, best_score = optimizer.optimize_model(X_train_dense, y_train)
    print(f"Best Model Score: {best_score}")

    print("Evaluating model...")
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate(best_model, X_val_dense, y_val)
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    main()
