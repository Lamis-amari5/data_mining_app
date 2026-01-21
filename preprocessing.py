import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.target_encoder = None
        self.preprocessor = None
        self.scaler = None

    # -------------------------------------------------
    # Handle custom missing values
    # -------------------------------------------------
    def handle_missing_symbols(self, df):
        df_copy = df.copy()
        missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!", " "]
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(missing_symbols, np.nan)
        return df_copy

    # -------------------------------------------------
    # Encode target
    # -------------------------------------------------
    def encode_target(self, y, mapping=None):
        y_array = y.to_numpy() if not isinstance(y, np.ndarray) else y

        if mapping is not None:
            y_encoded = np.array([mapping[val] for val in y_array])
            return y_encoded

        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y_array)
        return y_encoded

    # -------------------------------------------------
    # Remove outliers using IQR
    # -------------------------------------------------
    def remove_outliers_iqr(self, X, y):
        X_df = pd.DataFrame(X)
        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((X_df < (Q1 - 1.5 * IQR)) | (X_df > (Q3 + 1.5 * IQR))).any(axis=1)

        X_clean = X_df[mask].values
        y_clean = y[mask]

        return X_clean, y_clean

    # -------------------------------------------------
    # Main preprocessing pipeline
    # -------------------------------------------------
    def prepare_data(self, df, target_column, binary_target_mapping=None):
        # 1️⃣ Handle custom missing values
        df_clean = self.handle_missing_symbols(df)

        # 2️⃣ Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        # 3️⃣ Detect numeric and categorical columns
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 4️⃣ Pipelines
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # 5️⃣ ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        # 6️⃣ Fit + transform features
        X_processed = self.preprocessor.fit_transform(X)

        # 7️⃣ Encode target
        y_encoded = self.encode_target(y, binary_target_mapping)

        # 8️⃣ Remove outliers
        X_processed, y_encoded = self.remove_outliers_iqr(X_processed, y_encoded)

        # 9️⃣ Feature names
        feature_names = []
        if numeric_features:
            feature_names.extend(numeric_features)
        if categorical_features:
            ohe = self.preprocessor.named_transformers_["cat"]["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features)
            feature_names.extend(cat_names.tolist())

        return X_processed, y_encoded, feature_names, self.target_encoder

    # -------------------------------------------------
    # Train/test split
    # -------------------------------------------------
    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) < 20 else None
        )
