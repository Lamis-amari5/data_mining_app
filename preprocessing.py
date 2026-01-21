"""
preprocessing.py
Unified and safe preprocessing module for Data Mining project
Compatible with Drug Consumption and UCI Adult Income datasets
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    Universal preprocessing pipeline:
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical features
    - Prevents data leakage
    """

    def __init__(self):
        self.preprocessor = None
        self.target_encoder = None
        self.feature_names = None

    # --------------------------------------------------
    # Target encoding
    # --------------------------------------------------
    def encode_target(self, y, binary_mapping=None):
     """
     Encode target variable safely (pandas or numpy)
     """
     # Ensure pandas Series
     if isinstance(y, np.ndarray):
        y = pd.Series(y)

     y = y.astype(str)

     if binary_mapping is not None:
        y = y.map(binary_mapping)
        if y.isnull().any():
            raise ValueError("Binary target mapping is incomplete.")
        return y.to_numpy(), None

     self.target_encoder = LabelEncoder()
     y_encoded = self.target_encoder.fit_transform(y)

     return y_encoded, self.target_encoder

    # --------------------------------------------------
    # Feature preprocessing
    # --------------------------------------------------
    def build_preprocessor(self, X):
        """
        Build ColumnTransformer for numeric and categorical features
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ],
            remainder="drop"
        )

    # --------------------------------------------------
    # Main preprocessing pipeline
    # --------------------------------------------------
    def prepare_data(
        self,
        df,
        target_column,
        test_size=0.2,
        random_state=42,
        binary_target_mapping=None
    ):
        """
        Full preprocessing pipeline (safe & reusable)
        """

        # ----------------------------
        # 1. Replace common missing symbols
        # ----------------------------
        missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!", " "]
        df = df.replace(missing_symbols, np.nan)

        # ----------------------------
        # 2. Separate features / target
        # ----------------------------
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ----------------------------
        # 3. Encode target
        # ----------------------------
        y, _ = self.encode_target(y, binary_target_mapping)

        # ----------------------------
        # 4. Train / Test split (NO leakage)
        # ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) <= 10 else None
        )

        # ----------------------------
        # 5. Build & fit preprocessing pipeline
        # ----------------------------
        self.build_preprocessor(X_train)

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # ----------------------------
        # 6. Feature names (for analysis/report)
        # ----------------------------
        try:
            num_features = self.preprocessor.named_transformers_["num"] \
                .named_steps["imputer"].get_feature_names_out(
                    self.preprocessor.transformers_[0][2]
                )

            cat_features = self.preprocessor.named_transformers_["cat"] \
                .named_steps["encoder"].get_feature_names_out(
                    self.preprocessor.transformers_[1][2]
                )

            self.feature_names = list(num_features) + list(cat_features)
        except Exception:
            self.feature_names = None

        return (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            self.feature_names
        )
