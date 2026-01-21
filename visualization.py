"""
visualization.py
Visualization module for the Data Mining project
Creates plots and charts for data exploration and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_missing_values(self, df):
        """
        Create a bar plot showing missing values in the dataset
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        missing.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Number of Missing Values', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Create a heatmap of the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels if labels else 'auto',
                   yticklabels=labels if labels else 'auto',
                   cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_regression_results(self, y_true, y_pred):
        """
        Create scatter plot comparing actual vs predicted values for regression
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_title('Actual vs Predicted Values', fontsize=16, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=10):
        """
        Create bar plot showing feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            # Handle wrapped models
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
            else:
                return None
        else:
            importances = model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='steelblue')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_target_distribution(self, y, title='Target Distribution'):
        """
        Plot distribution of target variable
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(np.unique(y)) < 20:  # Categorical
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts, color='teal', edgecolor='black')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
        else:  # Continuous
            ax.hist(y, bins=30, color='teal', edgecolor='black', alpha=0.7)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_matrix(self, df, figsize=(12, 10)):
        """
        Create correlation matrix heatmap for numeric features
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_learning_curve(self, train_scores, val_scores, title='Learning Curve'):
        """
        Plot learning curve showing training and validation scores over epochs
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_scores) + 1)
        ax.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        ax.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig