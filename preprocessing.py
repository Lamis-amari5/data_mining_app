"""
preprocessing.py
Advanced data preprocessing module for Data Mining project
Safe against data leakage and compatible with ML pipelines
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    Professional preprocessing pipeline using sklearn best practices
    """

    def __init__(self):
        self.target_encoder = None
        self.preprocessor = None
        self.feature_names = None

    # ---------------------------------------------------
    # Target processing
    # ---------------------------------------------------
    def encode_target(self, y, binary_mapping=None):
        """
        Encode target variable
        - binary_mapping: dict for manual binary conversion
          Example: {'CL0': 0, 'CL1': 0, 'CL2': 1, 'CL3': 1, ...}
        """
        if binary_mapping is not None:
            y = y.map(binary_mapping)
            return y.values, None

        if y.dtype in ['object', 'category']:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))

        return y.values, self.target_encoder

    # ---------------------------------------------------
    # Feature preprocessing
    # ---------------------------------------------------
    def build_preprocessor(self, X):
        """
        Build ColumnTransformer for numeric and categorical features
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())  # robust to outliers
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        return self.preprocessor

    # ---------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------
    def prepare_data(
        self,
        df,
        target_column,
        test_size=0.2,
        random_state=42,
        binary_target_mapping=None
    ):
        """
        Full preprocessing pipeline (SAFE)
        """

        # --------------------
        # 1. Clean missing symbols
        # --------------------
        missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!", " "]
        df = df.replace(missing_symbols, np.nan)

        # --------------------
        # 2. Split features / target
        # --------------------
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # --------------------
        # 3. Encode target
        # --------------------
        y, _ = self.encode_target(y, binary_target_mapping)

        # --------------------
        # 4. Train / Test split (BEFORE preprocessing)
        # --------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) <= 10 else None
        )

        # --------------------
        # 5. Build & fit preprocessing pipeline
        # --------------------
        self.build_preprocessor(X_train)

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # --------------------
        # 6. Feature names (for analysis)
        # --------------------
        try:
            num_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
            cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            self.feature_names = list(num_features) + list(cat_features)
        except:
            self.feature_names = None

        return X_train_processed, X_test_processed, y_train, y_test, self.feature_names
