import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from preprocessing import DataPreprocessor
from feature_selector import EnhancedFeatureSelector
from model_optimizer import ModelOptimizer
from evaluation import ModelEvaluator

def preprocess_and_train(data, target_column, columns_to_use=None, n_features=10, output_filename="evaluation_report.html", output_path=None):
    # Preprocessing
    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    X, y, _ = preprocessor.preprocess(data, target_column)

    # Train-Test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Selection
    print("Performing feature selection...")
    selector = EnhancedFeatureSelector(input_dim=X_train.shape[1])
    top_features = selector.select_via_model_optimizer(
        X_train, y_train, n_features=n_features, method="dynamic"
    )

    reduced_selector = EnhancedFeatureSelector(input_dim=len(top_features))

    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()[:, top_features]
        X_test_dense = X_test.toarray()[:, top_features]
    else:
        X_train_dense = X_train[:, top_features]
        X_test_dense = X_test[:, top_features]

    # SMOTE-Tomek
    print("Applying SMOTE-Tomek on the training set...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train_dense, y_train)

    X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
    selected_features, probabilities = reduced_selector(X_train_tensor)

    # Model Optimization
    print("Optimizing models...")
    optimizer = ModelOptimizer()
    best_model, best_score = optimizer.optimize_model(
        X_train_res, y_train_res, method="dynamic"
    )
    print(f"Best Model Score (CV AUC): {best_score}")

    # Evaluation
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate(
        best_model,
        X_test_dense,
        y_test,
        output_format="html",
        output_filename=output_filename,
        output_path=output_path
    )
    print("Evaluation Results:", evaluation_results)

def main():
    # Dataset 1
    dataset_path1 = "../data/50krecords.csv"
    print("Processing dataset 1: 50krecords.csv")
    chunksize = 10000
    data_chunks = []
    with pd.read_csv(dataset_path1, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc="Loading Dataset Chunks"):
            data_chunks.append(chunk)

    data1 = pd.concat(data_chunks, ignore_index=True)
    columns1 = [
        'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
        'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
        'device_ip', 'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    data1 = data1[columns1]
    preprocess_and_train(data1, target_column="click", output_filename="evaluation_report_50krecords.html", output_path="../reports/50krecords")

    # Dataset 2
    dataset_path2 = "../data/world-happiness-report-2021.csv"
    print("Processing dataset 2: world-happiness-report-2021.csv")
    data2 = pd.read_csv(dataset_path2)
    columns2 = [
        "Country name", "Regional indicator", "Ladder score", "Logged GDP per capita", 
        "Social support", "Healthy life expectancy", "Freedom to make life choices", 
        "Generosity", "Perceptions of corruption"
    ]
    data2 = data2[columns2]

    target_column2 = "Ladder score"
    data2[target_column2] = (data2[target_column2] > data2[target_column2].mean()).astype(int)
    preprocess_and_train(data2, target_column=target_column2, output_filename="evaluation_report_happiness.html", output_path="../reports/happiness")

if __name__ == "__main__":
    main()
