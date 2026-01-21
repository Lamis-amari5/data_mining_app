import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class DataPreprocessor:

    def prepare_data(self, df, target, remove_outliers=False):
        """
        Prepares features and target for ML models.
        Args:
            df (pd.DataFrame): input dataset
            target (str): target column name
            remove_outliers (bool): if True, remove rows with outliers using IQR
        Returns:
            X (pd.DataFrame): features
            y (pd.Series): target
            feature_names (list): feature column names
            le (LabelEncoder or None): label encoder if target is categorical
        """

        df = df.copy()

        # Drop ID column if exists
        id_cols = [col for col in df.columns if col.lower() == 'id']
        if id_cols:
            df.drop(columns=id_cols, inplace=True)
            print(f"⚠️ Dropped ID column(s): {id_cols}")

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical target
        le = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Handle categorical features (one-hot encoding)
        X = pd.get_dummies(X, drop_first=True)

        # Remove outliers using IQR if requested
        if remove_outliers:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            filter_mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
            if filter_mask.sum() > 0:
                X = X.loc[filter_mask]
                y = y.loc[filter_mask]
            # Ensure at least one row remains
            if X.shape[0] == 0:
                X = df.drop(columns=[target])
                y = df[target]
                if le:
                    y = le.transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y, X.columns.tolist(), le

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split dataset into train and test sets
        """
        if X.shape[0] == 0:
            raise ValueError("Dataset has zero rows. Cannot split data.")
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) < 20 else None
        )