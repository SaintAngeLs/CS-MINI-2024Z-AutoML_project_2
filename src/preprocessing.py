#preporcessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Handles preprocessing of data including missing values, scaling, and encoding.
    """
    def preprocess(self, data, target_column, selected_features=None):
        """
        Preprocess the dataset and optionally apply feature selection.
        
        :param data: Input dataset.
        :param target_column: Name of the target column.
        :param selected_features: List of selected feature indices.
        :return: Preprocessed feature matrix, target vector, and preprocessor object.
        """
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(include=['number']).columns
        if target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ]
        )

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_preprocessed = preprocessor.fit_transform(X)
        if selected_features is not None:
            X_preprocessed = X_preprocessed[:, selected_features]

        return X_preprocessed, y, preprocessor