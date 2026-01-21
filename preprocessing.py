"""
preprocessing.py
Data preprocessing module for the Data Mining project
Handles missing values, encoding, normalization, and data splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Class for preprocessing data before training ML models"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer_numeric = SimpleImputer(strategy='mean')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        - Numeric columns: fill with mean
        - Categorical columns: fill with mode (most frequent)
        """
        df_copy = df.copy()

        # Define missing value symbols
        missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!", " "]

        # Replace missing symbols with NaN across all columns
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(missing_symbols, np.nan)

        # Convert numeric columns to numeric type (handles string numbers like "38.5")
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        # Try to convert object columns that might be numeric
        object_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
            except:
                pass

        # Re-identify columns after conversion
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle numeric columns - fill with mean
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if df_copy[col].isnull().sum() > 0:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)

        # Handle categorical columns - fill with mode
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df_copy[col].isnull().sum() > 0:
                    mode_val = df_copy[col].mode()
                    if len(mode_val) > 0:
                        df_copy[col].fillna(mode_val[0], inplace=True)

        return df_copy

    def encode_categorical(self, df, target_column=None):
        """
        Encode categorical variables
        - Use Label Encoding for binary categories
        - Use One-Hot Encoding for multi-class categories
        """
        df_copy = df.copy()
        label_encoders = {}

        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]

        for col in categorical_cols:
            # If binary or low cardinality, use label encoding
            if df_copy[col].nunique() <= 2:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                label_encoders[col] = le
            else:
                # Use one-hot encoding for multi-class
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy.drop(col, axis=1, inplace=True)

        return df_copy, label_encoders

    def normalize_features(self, X_train, X_test):
        """
        Normalize features using StandardScaler
        Fit on training data, transform both train and test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def remove_outliers_iqr(self, X, y):
        """
        Remove outliers using IQR method
        Only applied on numeric features
        """
        X_df = pd.DataFrame(X)

        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((X_df < (Q1 - 1.5 * IQR)) | (X_df > (Q3 + 1.5 * IQR))).any(axis=1)

        X_clean = X_df[mask].values
        y_clean = y[mask]

        return X_clean, y_clean

    def prepare_data(self, df, target_column):
        """
        Complete preprocessing pipeline
        1. Handle missing values
        2. Encode categorical variables
        3. Separate features and target
        4. Encode target if classification

        Returns: X (features), y (target), feature_names, label_encoder for target
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df)

        # Separate target
        y = df_clean[target_column].copy()
        X = df_clean.drop(target_column, axis=1)

        # Encode target if categorical (classification)
        le_target = None
        if y.dtype in ['object', 'category']:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Encode categorical features
        X, _ = self.encode_categorical(X)

        # Remove outliers (IQR)
        X, y = self.remove_outliers_iqr(X, y)

        # Get feature names (before converting to numpy array)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = None

        # Convert to numpy arrays
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if hasattr(y, 'values') else y

        return X, y, feature_names, le_target

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) < 20 else None  # Stratify for classification
        )

        return X_train, X_test, y_train, y_test