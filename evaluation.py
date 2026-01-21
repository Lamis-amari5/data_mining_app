"""
evaluation.py
Model evaluation module for the Data Mining project
Calculates metrics for classification and regression tasks
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

class ModelEvaluator:
    """Class for evaluating machine learning models"""
    
    def evaluate_classification(self, y_true, y_pred):
        """
        Evaluate classification model performance
        
        Returns:
        - accuracy: overall accuracy
        - precision: precision score
        - recall: recall score
        - f1_score: F1 score
        - confusion_matrix: confusion matrix
        - classification_report: detailed report
        """
        # Handle multi-class vs binary
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression model performance
        
        Returns:
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R-squared score
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5, is_classification=True):
        """
        Perform cross-validation on a model
        
        Parameters:
        - model: trained model
        - X: features
        - y: target
        - cv: number of folds
        - is_classification: whether it's a classification task
        
        Returns:
        - cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        if is_classification:
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def calculate_feature_importance(self, model, feature_names):
        """
        Calculate feature importance for tree-based models
        
        Returns:
        - DataFrame with features and their importance scores
        """
        import pandas as pd
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            return None