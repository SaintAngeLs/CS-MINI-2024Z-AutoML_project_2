import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from datetime columns.
    """
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            X[col + '_year'] = X[col].dt.year
            X[col + '_month'] = X[col].dt.month
            X[col + '_day'] = X[col].dt.day
            X[col + '_weekday'] = X[col].dt.weekday
            X.drop(columns=[col], inplace=True)
        return X

class DataPreprocessor:
    """
    Handles preprocessing of data including missing values, scaling, encoding, and feature extraction.
    """
    def __init__(
        self,
        scale_method="standard",
        handle_outliers=False,
        include_interactions=False,
        date_columns=None
    ):
        """
        :param scale_method: Scaling method ('standard', 'minmax', 'robust').
        :param handle_outliers: Whether to handle outliers by clipping.
        :param include_interactions: Whether to generate polynomial features or interactions.
        :param date_columns: List of datetime columns to extract features from.
        """
        self.scale_method = scale_method
        self.handle_outliers = handle_outliers
        self.include_interactions = include_interactions
        self.date_columns = date_columns

    def preprocess(self, data, target_column, selected_features=None):
        """
        Preprocess the dataset and optionally apply feature selection.

        :param data: Input dataset (pandas DataFrame).
        :param target_column: Name of the target column in the DataFrame.
        :param selected_features: List of selected feature indices (optional).
        :return: (X_preprocessed, y, preprocessor)
        """
        # Identify column types
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(include=['number']).columns

        if target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)

        # Date Feature Extraction
        if self.date_columns:
            date_extractor = DateFeatureExtractor(self.date_columns)
            data = date_extractor.transform(data)

        # Scaling
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }.get(self.scale_method, StandardScaler())

        # Outlier Handling
        if self.handle_outliers:
            data[numerical_cols] = data[numerical_cols].clip(
                lower=data[numerical_cols].quantile(0.01),
                upper=data[numerical_cols].quantile(0.99),
                axis=1
            )

        # Pipelines
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', scaler)
        ])

        # Polynomial Features
        if self.include_interactions:
            num_pipeline.steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True)))

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
