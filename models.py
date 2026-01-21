"""
models.py
Machine Learning models module for the Data Mining project
Implements: KNN, Naive Bayes, Decision Trees, Linear Regression, Neural Networks
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

class MLModels:
    """Class containing all machine learning models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def train_knn(self, X_train, y_train, is_classification=True, n_neighbors=5, weights='uniform'):
        """
        Train K-Nearest Neighbors model
        
        Parameters:
        - n_neighbors: number of neighbors to use
        - weights: 'uniform' or 'distance'
        """
        if is_classification:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        
        # Scale features for KNN (distance-based algorithm)
        X_train_scaled = self.scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # Wrap model to include scaling in prediction
        class ScaledKNN:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def predict_proba(self, X):
                if hasattr(self.model, 'predict_proba'):
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict_proba(X_scaled)
        
        return ScaledKNN(model, self.scaler)
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Naive Bayes classifier
        Uses Gaussian Naive Bayes (suitable for continuous features)
        """
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model
    
    def train_decision_tree(self, X_train, y_train, is_classification=True, 
                           max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                           criterion='entropy'):
        """
        Train Decision Tree model (C4.5 variant)
        
        Parameters:
        - max_depth: maximum depth of the tree
        - min_samples_split: minimum samples required to split a node
        - min_samples_leaf: minimum samples required at a leaf node
        - criterion: 'gini' or 'entropy' (entropy = Information Gain like C4.5)
        """
        if is_classification:
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        return model
    
    def train_chaid(self, X_train, y_train, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        """
        Train CHAID-like Decision Tree
        CHAID uses chi-square test for categorical variables
        This is a simplified implementation using sklearn's DecisionTree
        
        Note: For a full CHAID implementation, consider using the 'CHAID' library
        """
        # For simplicity, we use a decision tree with specific parameters
        # that mimic CHAID's behavior (multi-way splits would require custom implementation)
        model = DecisionTreeClassifier(
            criterion='gini',  # CHAID uses chi-square, but gini is similar for splits
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        return model
    
    def train_linear_regression(self, X_train, y_train, fit_intercept=True):
        """
        Train Linear Regression model (works for both simple and multiple regression)
        
        Parameters:
        - fit_intercept: whether to calculate the intercept
        """
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        return model
    
    def train_mlp(self, X_train, y_train, is_classification=True,
                  hidden_layer_sizes=(100, 50), max_iter=200, learning_rate_init=0.001):
        """
        Train Multi-Layer Perceptron (Neural Network)
        
        Parameters:
        - hidden_layer_sizes: tuple of hidden layer sizes
        - max_iter: maximum number of iterations
        - learning_rate_init: initial learning rate
        """
        if is_classification:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        # Scale features for neural networks
        X_train_scaled = self.scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # Wrap model to include scaling
        class ScaledMLP:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def predict_proba(self, X):
                if hasattr(self.model, 'predict_proba'):
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict_proba(X_scaled)
        
        return ScaledMLP(model, self.scaler)
    
    def train_cnn(self, X_train, y_train, input_shape, num_classes, epochs=50, batch_size=32):
        """
        Train Convolutional Neural Network
        This requires TensorFlow/Keras and image data
        
        Note: This is a placeholder. Full implementation will be added with your CNN notebook.
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Simple CNN architecture
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            
            return model
        except ImportError:
            raise ImportError("TensorFlow is required for CNN. Please install: pip install tensorflow")