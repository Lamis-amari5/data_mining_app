"""
Data Mining Final Project - Interactive ML Application
Author: [Your Name]
Date: January 2026
Description: Interactive Streamlit application for exploring various machine learning algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import arff
import io
import tempfile

# Import custom modules
from preprocessing import DataPreprocessor
from models import MLModels
from evaluation import ModelEvaluator
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Data Mining Project",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def load_dataset(uploaded_file):
    """Load dataset from CSV or ARFF file"""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".arff"):
            # Save uploaded ARFF to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".arff") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load ARFF from temp file
            data, meta = arff.loadarff(tmp_path)
            df = pd.DataFrame(data)

            # Decode bytes only (object columns)
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

        else:
            st.error("Unsupported file format. Please upload CSV or ARFF file.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Title and description
    st.title("🎓 Data Mining Final Project")
    st.markdown("""
    ### Interactive Machine Learning Application
    This application allows you to explore various machine learning algorithms on your datasets.
    Upload a dataset, select a target variable, choose an algorithm, and evaluate the results!
    """)
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select a step:",
        ["1. Upload Dataset", "2. Data Exploration", "3. Select Algorithm", "4. Results & Evaluation"]
    )
    
    # Step 1: Upload Dataset
    if page == "1. Upload Dataset":
        st.header("📁 Step 1: Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or ARFF file",
            type=['csv', 'arff'],
            help="Upload your dataset in CSV or ARFF format"
        )
        
        if uploaded_file is not None:
            df = load_dataset(uploaded_file)
            
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                # Display first few rows
                st.subheader("Preview of Dataset")
                st.dataframe(df.head(10))

                # Calculate missing values
                missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!" , " "]
    
                missing_count = df.isnull().sum().sum()
                missing_symbols_count = df.astype(str).isin(missing_symbols).sum().sum()
                total_missing = missing_count + missing_symbols_count

                # Basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Rows", df.shape[0])
                with col2:
                    st.metric("Number of Columns", df.shape[1])
                with col3:
                    st.metric("Total Missing Values", total_missing)
                    st.caption(f"NaN: {missing_count} | Symbols: {missing_symbols_count}")
                
    
    # Step 2: Data Exploration
    elif page == "2. Data Exploration":
        st.header("🔍 Step 2: Explore Your Data")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        # Remove ID column(s)
        id_cols = [col for col in df.columns if col.lower() == 'id']
        df = df.drop(columns=id_cols, errors='ignore')
        
        # Define missing value symbols
        missing_symbols = ["?", "NA", "N/A", "na", "null", "None", "unknown", "Unknown", "!" , " "]
       
       # Calculate missing symbols per column
        def count_missing_symbols(column):
         return column.astype(str).isin(missing_symbols).sum()
       
       
        # Dataset information
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Names and Types:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Missing Symbols': [count_missing_symbols(df[col]) for col in df.columns]
            })
            st.dataframe(info_df)
        
        with col2:
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe())
        
        # Missing values visualization
        st.subheader("Missing Values Analysis")
        missing_nan = df.isnull().sum()
        missing_sym = pd.Series([count_missing_symbols(df[col]) for col in df.columns], index=df.columns)
        total_missing = missing_nan + missing_sym

        if total_missing.sum() > 0:
            st.write("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': total_missing[total_missing > 0].index,
                'Missing Count': missing_nan[total_missing > 0].values,
                'Missing Symbols': missing_sym[total_missing > 0].values,
                'Total Missing': total_missing[total_missing > 0].values,
                'Percentage': (total_missing[total_missing > 0].values / len(df) * 100).round(2)
            })
           
            st.dataframe(missing_df)
            
            # Visualize missing values
            viz = Visualizer()
            fig = viz.plot_missing_values(df)
            st.pyplot(fig)
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Feature Visualization
        st.subheader("Data Visualization")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if numeric_cols:
            st.write("**Numeric Features Distribution**")
            for col in numeric_cols:
                st.write(f"Histogram of `{col}`")
                st.bar_chart(df[col].value_counts().sort_index())
            
            st.write("**Boxplots for Numeric Features**")
            for col in numeric_cols:
                st.write(f"Boxplot of `{col}`")
                fig = viz.plot_boxplot(df, col)
                st.pyplot(fig)

            # Scatter plots for numeric features
            st.write("**Scatter Plots (Feature vs Feature)**")  
            if len(numeric_cols) > 1:
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col_x = numeric_cols[i]
                        col_y = numeric_cols[j]
                        st.write(f"Scatter plot: `{col_x}` vs `{col_y}`")
                        fig = viz.plot_scatter(df, col_x, col_y)
                        st.pyplot(fig) 
        # Categorical features
        if categorical_cols:
            st.write("**Categorical Features Distribution**")
            for col in categorical_cols:
              st.write(f"Bar chart of `{col}`")
              fig = viz.plot_categorical_distribution(df, col)
              st.pyplot(fig)
        

        # Select target column
        st.subheader("Select Target Column")
        target_col = st.selectbox(
            "Choose the target (class) column for prediction:",
            options=df.columns.tolist(),
            help="This is the variable you want to predict"
        )
        
        if st.button("Confirm Target Column"):
            st.session_state.target_column = target_col
            st.success(f"✅ Target column set to: {target_col}")
            
            # Display target distribution
            st.write("**Target Distribution:**")
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
                target_counts = df[target_col].value_counts()
                st.bar_chart(target_counts)
            else:
                st.line_chart(df[target_col])
    
    # Step 3: Select Algorithm
    elif page == "3. Select Algorithm":
        st.header("🤖 Step 3: Select and Configure Algorithm")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        if st.session_state.target_column is None:
            st.warning("⚠️ Please select a target column first!")
            return
        
        df = st.session_state.data
        target = st.session_state.target_column
        
        # Determine problem type
        is_classification = df[target].dtype in ['object', 'category'] or df[target].nunique() < 20
        problem_type = "Classification" if is_classification else "Regression"
        
        st.info(f"📊 Detected problem type: **{problem_type}**")
        
        # Algorithm selection
        st.subheader("Select Algorithm")
        
        if is_classification:
            algorithms = [
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes",
                "Decision Tree (C4.5)",
                "Decision Tree (CHAID)",
                "Neural Network (MLP)",
                "Convolutional Neural Network (CNN)"
            ]
        else:
            algorithms = [
                "K-Nearest Neighbors (KNN)",
                "Linear Regression",
                "Multiple Linear Regression",
                "Decision Tree",
                "Neural Network (MLP)"
            ]
        
        selected_algo = st.selectbox("Choose an algorithm:", algorithms)
        
        # Hyperparameter configuration
        st.subheader("Configure Hyperparameters")
        
        hyperparams = {}
        
        if "KNN" in selected_algo:
            col1, col2 = st.columns(2)
            with col1:
                hyperparams['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, 20, 5)
            with col2:
                hyperparams['weights'] = st.selectbox("Weight Function", ['uniform', 'distance'])
        
        elif "Naive Bayes" in selected_algo:
            st.info("Naive Bayes uses default parameters (Gaussian for continuous, Multinomial for discrete)")
        
        elif "Decision Tree" in selected_algo:
            col1, col2, col3 = st.columns(3)
            with col1:
                hyperparams['max_depth'] = st.slider("Max Depth", 1, 20, 5)
            with col2:
                hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
            with col3:
                hyperparams['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 10, 1)
            
            if "C4.5" in selected_algo:
                hyperparams['criterion'] = 'entropy'  # Information Gain
            elif "CHAID" in selected_algo:
                st.info("CHAID implementation uses chi-square for splitting")
        
        elif "Linear Regression" in selected_algo or "Multiple Linear Regression" in selected_algo:
            hyperparams['fit_intercept'] = st.checkbox("Fit Intercept", value=True)
        
        elif "Neural Network" in selected_algo and "CNN" not in selected_algo:
            col1, col2, col3 = st.columns(3)
            with col1:
                hidden_layers = st.text_input("Hidden Layer Sizes (comma-separated)", "100,50")
                hyperparams['hidden_layer_sizes'] = tuple(map(int, hidden_layers.split(',')))
            with col2:
                hyperparams['max_iter'] = st.slider("Max Iterations", 100, 1000, 200)
            with col3:
                hyperparams['learning_rate_init'] = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.001, 0.01, 0.1],
                    value=0.001
                )
        
        elif "CNN" in selected_algo:
            st.info("CNN requires image data. Configuration will be available when image dataset is uploaded.")
            col1, col2 = st.columns(2)
            with col1:
                hyperparams['epochs'] = st.slider("Epochs", 10, 100, 50)
            with col2:
                hyperparams['batch_size'] = st.slider("Batch Size", 16, 128, 32)
        
        # Data split configuration
        st.subheader("Train-Test Split")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        random_state = 42
        
        # Train model button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... Please wait."):
                try:
                    # Preprocessing
                    preprocessor = DataPreprocessor()
                    X, y, feature_names, le = preprocessor.prepare_data(df, target)
                    X_train, X_test, y_train, y_test = preprocessor.split_data(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    ml_models = MLModels()
                    
                    if "KNN" in selected_algo:
                        model = ml_models.train_knn(X_train, y_train, is_classification, **hyperparams)
                    elif "Naive Bayes" in selected_algo:
                        model = ml_models.train_naive_bayes(X_train, y_train)
                    elif "Decision Tree" in selected_algo:
                        if "CHAID" in selected_algo:
                            model = ml_models.train_chaid(X_train, y_train, **hyperparams)
                        else:
                            model = ml_models.train_decision_tree(X_train, y_train, is_classification, **hyperparams)
                    elif "Linear Regression" in selected_algo or "Multiple Linear Regression" in selected_algo:
                        model = ml_models.train_linear_regression(X_train, y_train, **hyperparams)
                    elif "Neural Network" in selected_algo and "CNN" not in selected_algo:
                        model = ml_models.train_mlp(X_train, y_train, is_classification, **hyperparams)
                    elif "CNN" in selected_algo:
                        st.warning("CNN implementation requires image data preprocessing. This will be added with your notebook.")
                        return
                    
                    # Store results in session state
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.is_classification = is_classification
                    st.session_state.feature_names = feature_names
                    st.session_state.label_encoder = le
                    st.session_state.algorithm = selected_algo
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model trained successfully! Go to 'Results & Evaluation' to see the results.")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.exception(e)
    
    # Step 4: Results & Evaluation
    elif page == "4. Results & Evaluation":
        st.header("📊 Step 4: Results & Evaluation")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first!")
            return
        
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        is_classification = st.session_state.is_classification
        algorithm = st.session_state.algorithm
        
        st.subheader(f"Results for: {algorithm}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        viz = Visualizer()
        
        if is_classification:
            st.write("### Classification Metrics")
            
            # Metrics
            metrics = evaluator.evaluate_classification(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            # Confusion Matrix
            st.write("### Confusion Matrix")
            fig = viz.plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig)
            
            # Classification Report
            st.write("### Detailed Classification Report")
            st.text(metrics['classification_report'])
            
        else:
            st.write("### Regression Metrics")
            
            # Metrics
            metrics = evaluator.evaluate_regression(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{metrics['mse']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            
            # Prediction vs Actual plot
            st.write("### Predictions vs Actual Values")
            fig = viz.plot_regression_results(y_test, y_pred)
            st.pyplot(fig)
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.write("### Feature Importance")
            fig = viz.plot_feature_importance(model, st.session_state.feature_names)
            st.pyplot(fig)
        
        # Download predictions
        st.write("### Download Predictions")
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()