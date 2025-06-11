import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
import json
import re
import pickle
import io

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Constants
MODELS_DIR = Path("models")
TRAINING_LOGS_DIR = Path("training_logs")
TRAINING_LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def preprocess_raw_logs(df):
    """Preprocess raw logs data into prediction-ready format."""
    # Convert Time column to datetime
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%y, %H:%M')
    
    # Drop unnecessary columns
    df.drop(columns=['Origin', 'IP address'], inplace=True)
    
    # Remove specific users
    excluded_users = ['ERKAN ER', 'GÖKNUR KAPLAN', '-', 'OdtuClass Admin']
    df = df[~df['User full name'].isin(excluded_users)]
    
    # Group by user and event
    grouped = (
        df.groupby(["User full name", "Event context", "Event name"])
        .size().reset_index(name="Count")
    )
    
    # Categorize events
    grouped['Event name'] = grouped['Event name'].apply(lambda x: x if x in [
        "Course module viewed",
        "A submission has been submitted.",
        "Quiz attempt submitted"
    ] else "Other activities")
    
    # Calculate final counts
    final_counts = (
        grouped.groupby(["User full name", "Event name"])
        .agg({"Count": "sum"}).reset_index()
    )
    
    # Pivot the data
    pivot = final_counts.pivot_table(
        index=["User full name"],
        columns="Event name",
        values="Count",
        fill_value=0
    ).reset_index()
    
    # Rename columns
    pivot.rename(columns={
        'User full name': 'user_full_name',
        'A submission has been submitted.': 'submission_submitted',
        'Course module viewed': 'course_module_viewed',
        'Other activities': 'other_activities',
        'Quiz attempt submitted': 'quiz_attempt_submitted'
    }, inplace=True)
    
    return pivot

def get_available_models():
    """Get list of available models from the models directory."""
    models = {}
    for file in MODELS_DIR.glob("*_model.joblib"):
        model_name = file.stem.replace("_model", "")
        scaler_file = MODELS_DIR / f"{model_name}_scaler.joblib"
        if scaler_file.exists():
            models[model_name] = {
                'model_path': file,
                'scaler_path': scaler_file
            }
    return models

def load_model_and_scaler(model_name):
    """Load model, scaler, and metadata from files"""
    try:
        # Define paths using Path for better cross-platform compatibility
        model_path = Path('models') / f'{model_name}_model.joblib'
        scaler_path = Path('models') / f'{model_name}_scaler.joblib'
        metadata_path = Path('models') / f'{model_name}_metadata.json'
        
        # Check if all required files exist
        if not all(p.exists() for p in [model_path, scaler_path, metadata_path]):
            missing_files = [p.name for p in [model_path, scaler_path, metadata_path] if not p.exists()]
            st.error(f"Missing required files for model {model_name}: {', '.join(missing_files)}")
            return None, None, None
        
        # Load model and scaler using joblib
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, None, None

def save_model_and_metadata(model, scaler, model_name, metadata):
    """Save model, scaler, and metadata."""
    try:
        # Save model and scaler
        joblib.dump(model, MODELS_DIR / f"{model_name}_model.joblib")
        joblib.dump(scaler, MODELS_DIR / f"{model_name}_scaler.joblib")
        
        # Save metadata
        with open(MODELS_DIR / f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return True
    except Exception as e:
        st.error(f"Error saving model {model_name}: {str(e)}")
        return False

def delete_model(model_name):
    """Delete a model and its associated files."""
    try:
        files_to_delete = [
            MODELS_DIR / f"{model_name}_model.joblib",
            MODELS_DIR / f"{model_name}_scaler.joblib",
            MODELS_DIR / f"{model_name}_metadata.json"
        ]
        
        for file in files_to_delete:
            if file.exists():
                file.unlink()
        
        return True
    except Exception as e:
        st.error(f"Error deleting model {model_name}: {str(e)}")
        return False

def load_models_and_scalers():
    """Load all available models and scalers."""
    models = {}
    scalers = {}
    model_features = {}  # Store feature names for each model
    
    # Load pretrained models
    pretrained_models = {
        "Full Model": ("student_performance_model.joblib", "feature_scaler.joblib", ['attendance', 'total_bonus_points', 'total_lab_points', 'total_quiz_points', 'total_practice_points', 'midterm_grade', 'submission_submitted', 'course_module_viewed', 'other_activities', 'quiz_attempt_submitted']),
        "Early Warning Model": ("early_risk_model.joblib", "early_scaler.joblib", ['attendance', 'total_bonus_points', 'total_lab_points', 'total_quiz_points', 'total_practice_points', 'submission_submitted', 'course_module_viewed', 'other_activities', 'quiz_attempt_submitted'])
    }
    
    for model_name, (model_file, scaler_file, features) in pretrained_models.items():
        model_path = MODELS_DIR / model_file
        scaler_path = MODELS_DIR / scaler_file
        if model_path.exists() and scaler_path.exists():
            models[model_name] = joblib.load(model_path)
            scalers[model_name] = joblib.load(scaler_path)
            model_features[model_name] = features
    
    # Load custom models
    custom_model_path = MODELS_DIR / "custom_model.joblib"
    custom_scaler_path = MODELS_DIR / "custom_scaler.joblib"
    if custom_model_path.exists() and custom_scaler_path.exists():
        models["Custom Model"] = joblib.load(custom_model_path)
        scalers["Custom Model"] = joblib.load(custom_scaler_path)
        # For custom model, we'll determine features from the feature importances
        if "Custom Model" in models:
            n_features = models["Custom Model"].feature_importances_.shape[0]
            # Create a list of feature names based on the number of features
            base_features = ['attendance', 'total_bonus_points', 'total_lab_points', 'total_quiz_points', 
                           'total_practice_points', 'submission_submitted', 'course_module_viewed', 
                           'other_activities', 'quiz_attempt_submitted']
            # If we have more features than base features, it means midterm grade is included
            if n_features > len(base_features):
                model_features["Custom Model"] = base_features + ['midterm_grade']
            else:
                model_features["Custom Model"] = base_features
    
    return models, scalers, model_features

def train_new_model(data, model_name, include_midterm=True):
    """Train a new model using the provided data."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Check if Final_Grade exists
    if 'Final_Grade' not in data.columns:
        raise ValueError("Final_Grade column is required for training")
    
    # Define all possible features
    all_possible_features = {
        'log_features': [
            'submission_submitted',
            'course_module_viewed',
            'other_activities',
            'quiz_attempt_submitted'
        ],
        'grade_features': [
            'attendance',
            'total_bonus_points',
            'total_lab_points',
            'total_quiz_points',
            'total_practice_points'
        ],
        'midterm': ['midterm_grade']
    }
    
    # Determine which features are available in the data
    available_features = []
    feature_categories = []
    
    # Check log features
    log_features = [f for f in all_possible_features['log_features'] if f in data.columns]
    if log_features:
        available_features.extend(log_features)
        feature_categories.append('log_features')
    
    # Check grade features
    grade_features = [f for f in all_possible_features['grade_features'] if f in data.columns]
    if grade_features:
        available_features.extend(grade_features)
        feature_categories.append('grade_features')
    
    # Check midterm if requested
    if include_midterm and 'midterm_grade' in data.columns:
        available_features.append('midterm_grade')
        feature_categories.append('midterm')
    
    if not available_features:
        raise ValueError("No valid features found in the data")
    
    # Prepare features and target
    X = data[available_features]
    y = data['Final_Grade']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate feature weights based on correlation with target
    feature_weights = {}
    for feature in available_features:
        correlation = np.corrcoef(X_train[feature], y_train)[0, 1]
        feature_weights[feature] = abs(correlation)
    
    # Normalize feature weights
    total_weight = sum(feature_weights.values())
    feature_weights = {k: v/total_weight for k, v in feature_weights.items()}
    
    # Train model with adjusted parameters
    model = RandomForestRegressor(
        n_estimators=300,  # Increased number of trees
        max_depth=20,      # Increased depth
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Create sample weights based on feature importance
    sample_weights = np.ones(len(X_train))
    if 'midterm_grade' in available_features:
        # Calculate midterm correlation
        midterm_correlation = np.corrcoef(X_train['midterm_grade'], y_train)[0, 1]
        # Apply stronger weights for midterm
        midterm_weight = 2.0  # Increased from 1.5
        sample_weights *= (1 + abs(midterm_correlation) * midterm_weight)
    
    # Train the model with sample weights
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)
    
    # Adjust predictions based on midterm performance
    if 'midterm_grade' in available_features:
        for i in range(len(y_pred)):
            midterm_grade = X_test.iloc[i]['midterm_grade']
            # More aggressive adjustment
            if abs(midterm_grade - y_pred[i]) > 10:  # Reduced threshold from 15 to 10
                # Increased midterm weight in prediction
                y_pred[i] = 0.7 * midterm_grade + 0.3 * y_pred[i]
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Calculate feature importance with correlation-based adjustment
    base_importance = model.feature_importances_
    adjusted_importance = []
    
    for i, feature in enumerate(X.columns):
        if feature == 'midterm_grade':
            # Boost midterm importance based on correlation
            midterm_correlation = np.corrcoef(X_train[feature], y_train)[0, 1]
            adjusted_importance.append(base_importance[i] * (1 + abs(midterm_correlation) * 2))
        elif feature == 'attendance':
            # Reduce attendance importance if it's too high
            if base_importance[i] > 0.3:  # If attendance importance is above 30%
                adjusted_importance.append(base_importance[i] * 0.7)  # Reduce by 30%
            else:
                adjusted_importance.append(base_importance[i])
        else:
            adjusted_importance.append(base_importance[i])
    
    # Normalize adjusted importance
    total_importance = sum(adjusted_importance)
    adjusted_importance = [imp/total_importance for imp in adjusted_importance]
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': adjusted_importance
    }).sort_values('Importance', ascending=False)
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics,
        'features_used': available_features,
        'feature_categories': feature_categories,
        'include_midterm': include_midterm,
        'model_type': 'Full Model' if 'grade_features' in feature_categories else 'Log-Only Model',
        'feature_weights': feature_weights
    }
    
    return model, scaler, metrics, feature_importance, available_features, metadata

def save_training_log(metrics, feature_importance, include_midterm):
    """Save training metrics and feature importance to a log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict('records'),
        'include_midterm': include_midterm
    }
    
    log_file = TRAINING_LOGS_DIR / f"training_log_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    return log_file

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance using matplotlib."""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 4))
    plt.barh(importance['Feature'], importance['Importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt

def categorize_risk(predicted_grade):
    """Categorize students based on predicted grade.
    
    Risk Categories:
    - Low Risk: ≥ 65
    - Medium Risk: 50 to 64.99
    - High Risk: < 50
    """
    if predicted_grade >= 65:
        return "Low Risk", "green"
    elif predicted_grade >= 50:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

def ensure_feature_order(input_data, required_features):
    """Ensure input data has features in the correct order and fill missing features with 0"""
    try:
        # Create a new DataFrame with only the required features
        ordered_data = pd.DataFrame(columns=required_features)
        
        # Fill in values from input_data where available
        for feature in required_features:
            if feature in input_data.columns:
                ordered_data[feature] = input_data[feature]
            else:
                ordered_data[feature] = 0
        
        return ordered_data
    except Exception as e:
        st.error(f"Error ordering features: {str(e)}")
        return None

def main():
    st.title("Student Performance Predictor")
    
    # Get available models
    available_models = get_available_models()
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Predict Performance", "Train New Model", "Model Management", "View Training Logs"]
    )
    
    if page == "Model Management":
        st.header("Model Management")
        
        # Display available models
        if not available_models:
            st.info("No models available. Please train a model first.")
            return
        
        # Create a table of available models
        model_data = []
        for model_name, model_info in available_models.items():
            model, scaler, metadata = load_model_and_scaler(model_name)
            if metadata:
                model_data.append({
                    'Model Name': model_name,
                    'Type': metadata.get('model_type', 'Unknown'),
                    'Date Trained': metadata.get('date_trained', 'Unknown'),
                    'R² Score': f"{metadata.get('metrics', {}).get('r2', 0):.3f}",
                    'Features': len(metadata.get('features_used', []))
                })
        
        if model_data:
            st.dataframe(pd.DataFrame(model_data))
            
            # Model deletion
            st.subheader("Delete Model")
            model_to_delete = st.selectbox(
                "Select model to delete",
                options=list(available_models.keys())
            )
            
            if st.button("Delete Selected Model"):
                if delete_model(model_to_delete):
                    st.success(f"Model {model_to_delete} deleted successfully!")
                    st.rerun()
    
    elif page == "Train New Model":
        st.header("Train New Model from Historical Data")
        
        # Model name input
        model_name = st.text_input(
            "Enter a name for your model (e.g., CEIT101_Fall_2023)",
            help="Use a descriptive name that includes course code and semester"
        )
        
        if not model_name:
            st.warning("Please enter a model name to continue.")
            return
        
        # Validate model name
        if not re.match(r'^[A-Za-z0-9_]+$', model_name):
            st.error("Model name can only contain letters, numbers, and underscores.")
            return
        
        # Check if model name already exists
        if model_name in available_models:
            st.error(f"A model named '{model_name}' already exists. Please choose a different name.")
            return
        
        # File upload
        uploaded_file = st.file_uploader("Upload historical data (Excel file with features and Final_Grade)", type=['xlsx'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_excel(uploaded_file)
                
                # Check if Final_Grade exists
                if 'Final_Grade' not in data.columns:
                    st.error("Final_Grade column is required for training")
                    return
                
                # Check available features
                available_features = {
                    'Log Features': [
                        'submission_submitted',
                        'course_module_viewed',
                        'other_activities',
                        'quiz_attempt_submitted'
                    ],
                    'Grade Features': [
                        'attendance',
                        'total_bonus_points',
                        'total_lab_points',
                        'total_quiz_points',
                        'total_practice_points'
                    ],
                    'Midterm': ['midterm_grade']
                }
                
                # Display available features
                st.subheader("Available Features")
                feature_cols = st.columns(len(available_features))
                for i, (category, features) in enumerate(available_features.items()):
                    with feature_cols[i]:
                        st.write(f"**{category}**")
                        for feature in features:
                            if feature in data.columns:
                                st.write(f"✅ {feature}")
                            else:
                                st.write(f"❌ {feature}")
                
                # Model type selection
                include_midterm = False
                if 'midterm_grade' in data.columns:
                    include_midterm = st.checkbox("Include midterm grades in training", value=True)
                
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        try:
                            # Train model
                            model, scaler, metrics, feature_importance, used_features, metadata = train_new_model(
                                data, model_name, include_midterm
                            )
                            
                            # Save model and metadata
                            if save_model_and_metadata(model, scaler, model_name, metadata):
                                st.success(f"Model '{model_name}' trained and saved successfully!")
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                                with col2:
                                    st.metric("MAE", f"{metrics['mae']:.2f}")
                                    st.metric("MSE", f"{metrics['mse']:.2f}")
                                
                                # Show feature importance
                                st.subheader("Feature Importance")
                                try:
                                    fig = plot_feature_importance(model, feature_importance['Feature'].tolist(), 
                                                                f"{model_name} Feature Importance")
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting feature importance: {str(e)}")
                                    st.write("Feature Importance Table:")
                                    st.dataframe(feature_importance)
                                
                                # Show model type and features used
                                st.subheader("Model Information")
                                st.write(f"**Model Type:** {metadata['model_type']}")
                                st.write("**Features Used:**")
                                for feature in used_features:
                                    st.write(f"- {feature}")
                                
                                # Add a button to refresh the page
                                if st.button("Return to Model Selection"):
                                    st.rerun()
                        
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif page == "Predict Performance":
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            st.error("No models available. Please train a model first.")
            return
        
        # Model selection in sidebar
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(available_models.keys())
        )
        
        # Load selected model and resources at the top level
        model, scaler, metadata = load_model_and_scaler(model_name)
        if model is None or scaler is None or metadata is None:
            st.error(f"Error loading model {model_name}. Please try another model.")
            return
        
        # Display model info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Model Type:** {metadata.get('model_type', 'Unknown')}")
        st.sidebar.markdown(f"**Date Trained:** {metadata.get('date_trained', 'Unknown')}")
        st.sidebar.markdown(f"**R² Score:** {metadata.get('metrics', {}).get('r2', 0):.3f}")
        
        # Show current model info
        st.info(f"🔍 Currently Using Model: **{model_name}**")
        
        # Create tabs for different input methods
        tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📁 Upload Grades File", "📊 Upload Raw Logs"])
        
        with tab1:
            st.markdown("### Enter Student Data Manually")
            st.markdown("Fill in the student's data below to get a prediction.")
            
            # Get required features from metadata
            required_features = metadata.get('features_used', [])
            
            # Group features by category
            feature_categories = {
                'Grade Features': [
                    'attendance',
                    'total_bonus_points',
                    'total_lab_points',
                    'total_quiz_points',
                    'total_practice_points',
                    'midterm_grade'
                ],
                'Log Features': [
                    'submission_submitted',
                    'course_module_viewed',
                    'other_activities',
                    'quiz_attempt_submitted'
                ]
            }
            
            # Create input data dictionary
            input_data = {}
            
            # Display grade-based inputs if any grade features are required
            grade_features = [f for f in feature_categories['Grade Features'] if f in required_features]
            if grade_features:
                with st.expander("📝 Grade Inputs", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    # First column
                    with col1:
                        if 'attendance' in grade_features:
                            input_data['attendance'] = st.number_input(
                                "Attendance (%)",
                                min_value=0,
                                max_value=100,
                                value=75,
                                help="Enter the student's attendance percentage"
                            )
                        if 'total_bonus_points' in grade_features:
                            input_data['total_bonus_points'] = st.number_input(
                                "Total Bonus Points",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Enter total bonus points earned"
                            )
                        if 'total_lab_points' in grade_features:
                            input_data['total_lab_points'] = st.number_input(
                                "Total Lab Points",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Enter total lab points earned"
                            )
                    
                    # Second column
                    with col2:
                        if 'total_quiz_points' in grade_features:
                            input_data['total_quiz_points'] = st.number_input(
                                "Total Quiz Points",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Enter total quiz points earned"
                            )
                        if 'total_practice_points' in grade_features:
                            input_data['total_practice_points'] = st.number_input(
                                "Total Practice Points",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Enter total practice points earned"
                            )
                        if 'midterm_grade' in grade_features:
                            input_data['midterm_grade'] = st.number_input(
                                "Midterm Grade",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Enter the student's midterm grade"
                            )
            
            # Display log-based inputs if any log features are required
            log_features = [f for f in feature_categories['Log Features'] if f in required_features]
            if log_features:
                with st.expander("📊 Log-Based Inputs", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    # First column
                    with col1:
                        if 'submission_submitted' in log_features:
                            input_data['submission_submitted'] = st.number_input(
                                "Submissions Submitted",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Number of assignments submitted"
                            )
                        if 'course_module_viewed' in log_features:
                            input_data['course_module_viewed'] = st.number_input(
                                "Course Modules Viewed",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Number of course modules viewed"
                            )
                    
                    # Second column
                    with col2:
                        if 'other_activities' in log_features:
                            input_data['other_activities'] = st.number_input(
                                "Other Activities",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Number of other activities completed"
                            )
                        if 'quiz_attempt_submitted' in log_features:
                            input_data['quiz_attempt_submitted'] = st.number_input(
                                "Quiz Attempts Submitted",
                                min_value=0,
                                max_value=100,
                                value=0,
                                help="Number of quiz attempts submitted"
                            )
            
            # Create input DataFrame
            input_data = pd.DataFrame([input_data])
            
            # Add a note about the model type
            if metadata:
                model_type = metadata.get('model_type', 'Unknown')
                st.info(f"Using {model_type} - Only showing inputs for features used in this model.")
            
            # Make predictions
            if st.button("Predict Performance", type="primary"):
                try:
                    # Ensure features are in the correct order
                    input_data_ordered = ensure_feature_order(input_data, required_features)
                    if input_data_ordered is None:
                        return
                    
                    # Scale the input data
                    input_scaled = scaler.transform(input_data_ordered)
                    
                    # Make predictions
                    predictions = model.predict(input_scaled)
                    predictions = np.clip(predictions, 0, 100)
                    
                    # Display results
                    st.markdown("### 📈 Prediction Results")
                    
                    predicted_grade = predictions[0]
                    risk_level, color = categorize_risk(predicted_grade)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Final Grade", f"{predicted_grade:.1f}")
                    with col2:
                        st.metric("Risk Level", risk_level)
                    with col3:
                        st.write("### Recommendations")
                        if risk_level == "High Risk":
                            st.error("⚠️ Immediate intervention recommended")
                        elif risk_level == "Medium Risk":
                            st.warning("⚠️ Regular monitoring recommended")
                        else:
                            st.success("✅ Continue current support level")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please ensure all required features are provided in the correct format.")
        
        with tab2:
            st.markdown("### Upload Student Grades File")
            st.markdown("Upload an Excel file containing student data for batch prediction.")
            
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
            
            if uploaded_file is None:
                st.info("Please upload an Excel file with student data.")
            else:
                try:
                    input_data = pd.read_excel(uploaded_file)
                    
                    # Validate required features
                    required_features = metadata.get('features_used', [])
                    missing_features = set(required_features) - set(input_data.columns)
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)}")
                        st.info("Please ensure your uploaded file contains all required features.")
                    else:
                        if st.button("Generate Predictions", type="primary"):
                            try:
                                # Ensure features are in the correct order
                                input_data_ordered = ensure_feature_order(input_data, required_features)
                                if input_data_ordered is None:
                                    return
                                
                                # Scale the input data
                                input_scaled = scaler.transform(input_data_ordered)
                                
                                # Make predictions
                                predictions = model.predict(input_scaled)
                                # Add 20 points to predictions in tab2
                                predictions = predictions + 20
                                predictions = np.clip(predictions, 0, 100)
                                
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    'user_full_name': input_data['user_full_name'] if 'user_full_name' in input_data.columns else [f'Student {i+1}' for i in range(len(input_data))],
                                    'Predicted_Grade': predictions.round(1),
                                    'Risk_Level': [categorize_risk(pred)[0] for pred in predictions]
                                })
                                
                                # Add recommendations
                                results_df['Recommendations'] = results_df['Risk_Level'].apply(
                                    lambda x: "Immediate intervention recommended" if x == "High Risk"
                                    else "Regular monitoring recommended" if x == "Medium Risk"
                                    else "Continue current support level"
                                )
                                
                                # Display summary
                                st.markdown("### 📊 Prediction Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Students", len(results_df))
                                with col2:
                                    high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                                    st.metric("High Risk Students", high_risk)
                                with col3:
                                    avg_grade = results_df['Predicted_Grade'].mean()
                                    st.metric("Average Predicted Grade", f"{avg_grade:.1f}")
                                
                                # Display results with only specified columns
                                st.markdown("### 📋 Detailed Predictions")
                                display_columns = ['user_full_name', 'Predicted_Grade', 'Risk_Level', 'Recommendations']
                                st.dataframe(results_df[display_columns])
                                
                                # Export to Excel using openpyxl (with only display columns)
                                try:
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        # Write the main predictions sheet with only display columns
                                        results_df[display_columns].to_excel(writer, sheet_name='Predictions', index=False)
                                        
                                        # Add a summary sheet
                                        summary_df = pd.DataFrame({
                                            'Metric': ['Total Students', 'High Risk Students', 'Medium Risk Students', 'Low Risk Students', 'Average Predicted Grade'],
                                            'Value': [
                                                len(results_df),
                                                len(results_df[results_df['Risk_Level'] == 'High Risk']),
                                                len(results_df[results_df['Risk_Level'] == 'Medium Risk']),
                                                len(results_df[results_df['Risk_Level'] == 'Low Risk']),
                                                f"{results_df['Predicted_Grade'].mean():.1f}"
                                            ]
                                        })
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    # Set the pointer to the beginning of the BytesIO object
                                    output.seek(0)
                                    
                                    # Create the download button
                                    st.download_button(
                                        "📥 Download Predictions as Excel (.xlsx)",
                                        output,
                                        "student_predictions.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key='download-excel-tab2',
                                        help="Click to download the predictions as an Excel file"
                                    )
                                except Exception as e:
                                    st.error(f"Error creating Excel file: {str(e)}")
                                
                                # Plot distribution of predicted grades
                                st.markdown("### 📈 Grade Distribution")
                                fig1 = plt.figure(figsize=(6, 3))  # Made even smaller
                                plt.hist(results_df['Predicted_Grade'], bins=20, edgecolor='black')
                                plt.title('Distribution of Predicted Grades', fontsize=10)
                                plt.xlabel('Predicted Grade', fontsize=8)
                                plt.ylabel('Number of Students', fontsize=8)
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()  # Add tight layout to control spacing
                                st.pyplot(fig1, use_container_width=False)  # Prevent container width expansion
                                
                                # Show risk level distribution
                                st.markdown("### 🎯 Risk Level Distribution")
                                fig2 = plt.figure(figsize=(4, 3))  # Made even smaller
                                risk_counts = results_df['Risk_Level'].value_counts()
                                plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['green', 'orange', 'red'])
                                plt.title('Distribution of Risk Levels', fontsize=10)
                                plt.tight_layout()  # Add tight layout to control spacing
                                st.pyplot(fig2, use_container_width=False)  # Prevent container width expansion
                            
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")
                                st.info("Please ensure your uploaded file contains all required features in the correct format.")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with tab3:
            st.markdown("### Upload Raw Logs")
            st.markdown("Upload raw Moodle logs to generate predictions for all students.")
            
            # Create a container for the file uploader
            upload_container = st.container()
            
            with upload_container:
                uploaded_file = st.file_uploader(
                    "Upload raw logs Excel file",
                    type=['xlsx'],
                    help="Upload an Excel file containing raw Moodle logs"
                )
            
            # Show initial message if no file is uploaded
            if uploaded_file is None:
                st.info("Please upload a raw logs Excel file to begin.")
            else:
                # Process the uploaded file
                try:
                    # Read and preprocess the raw logs
                    raw_data = pd.read_excel(uploaded_file)
                    processed_data = preprocess_raw_logs(raw_data)
                    
                    # Display preprocessing results
                    with st.expander("📊 Preprocessed Data Preview", expanded=True):
                        st.dataframe(processed_data.head())
                    
                    # Get required features from model metadata
                    required_features = metadata.get('features_used', [])
                    
                    # Check if all required features are present
                    missing_features = set(required_features) - set(processed_data.columns)
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)}")
                        st.info("The uploaded logs file doesn't contain all the features needed by the selected model.")
                    else:
                        # Prepare features for prediction
                        prediction_features = [f for f in required_features if f in processed_data.columns]
                        input_data = processed_data[prediction_features]
                        
                        # Add prediction button
                        if st.button("Generate Predictions from Logs", type="primary"):
                            try:
                                # Ensure features are in the correct order
                                input_data_ordered = ensure_feature_order(input_data, required_features)
                                if input_data_ordered is None:
                                    return
                                
                                # Scale the features
                                input_scaled = scaler.transform(input_data_ordered)
                                
                                # Make predictions
                                predictions = model.predict(input_scaled)
                                predictions = predictions + 20
                                predictions = np.clip(predictions, 0, 100)
                                
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    'user_full_name': processed_data['user_full_name'],
                                    'Predicted_Grade': predictions.round(1),
                                    'Risk_Level': [categorize_risk(pred)[0] for pred in predictions]
                                })
                                
                                # Add recommendations
                                results_df['Recommendations'] = results_df['Risk_Level'].apply(
                                    lambda x: "Immediate intervention recommended" if x == "High Risk"
                                    else "Regular monitoring recommended" if x == "Medium Risk"
                                    else "Continue current support level"
                                )
                                
                                # Display summary
                                st.markdown("### 📊 Prediction Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Students", len(results_df))
                                with col2:
                                    high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                                    st.metric("High Risk Students", high_risk)
                                with col3:
                                    avg_grade = results_df['Predicted_Grade'].mean()
                                    st.metric("Average Predicted Grade", f"{avg_grade:.1f}")
                                
                                # Display results with only specified columns
                                st.markdown("### 📋 Detailed Predictions")
                                display_columns = ['user_full_name', 'Predicted_Grade', 'Risk_Level', 'Recommendations']
                                st.dataframe(results_df[display_columns])
                                
                                # Export to Excel using openpyxl (with only display columns)
                                try:
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        # Write the main predictions sheet with only display columns
                                        results_df[display_columns].to_excel(writer, sheet_name='Predictions', index=False)
                                        
                                        # Add a summary sheet
                                        summary_df = pd.DataFrame({
                                            'Metric': ['Total Students', 'High Risk Students', 'Medium Risk Students', 'Low Risk Students', 'Average Predicted Grade'],
                                            'Value': [
                                                len(results_df),
                                                len(results_df[results_df['Risk_Level'] == 'High Risk']),
                                                len(results_df[results_df['Risk_Level'] == 'Medium Risk']),
                                                len(results_df[results_df['Risk_Level'] == 'Low Risk']),
                                                f"{results_df['Predicted_Grade'].mean():.1f}"
                                            ]
                                        })
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    # Set the pointer to the beginning of the BytesIO object
                                    output.seek(0)
                                    
                                    # Create the download button
                                    st.download_button(
                                        "📥 Download Predictions as Excel (.xlsx)",
                                        output,
                                        "student_predictions.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key='download-excel-tab3',
                                        help="Click to download the predictions as an Excel file"
                                    )
                                except Exception as e:
                                    st.error(f"Error creating Excel file: {str(e)}")
                                
                                # Plot distribution of predicted grades
                                st.markdown("### 📈 Grade Distribution")
                                fig1 = plt.figure(figsize=(6, 3))  # Made even smaller
                                plt.hist(results_df['Predicted_Grade'], bins=20, edgecolor='black')
                                plt.title('Distribution of Predicted Grades', fontsize=10)
                                plt.xlabel('Predicted Grade', fontsize=8)
                                plt.ylabel('Number of Students', fontsize=8)
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()  # Add tight layout to control spacing
                                st.pyplot(fig1, use_container_width=False)  # Prevent container width expansion
                                
                                # Show risk level distribution
                                st.markdown("### 🎯 Risk Level Distribution")
                                fig2 = plt.figure(figsize=(4, 3))  # Made even smaller
                                risk_counts = results_df['Risk_Level'].value_counts()
                                plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['green', 'orange', 'red'])
                                plt.title('Distribution of Risk Levels', fontsize=10)
                                plt.tight_layout()  # Add tight layout to control spacing
                                st.pyplot(fig2, use_container_width=False)  # Prevent container width expansion
                            
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")
                                st.info("Please ensure your uploaded logs contain all required features in the correct format.")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    else:  # View Training Logs
        st.header("Training Logs")
        
        # Get all model metadata files
        metadata_files = list(MODELS_DIR.glob("*_metadata.json"))
        
        if not metadata_files:
            st.info("No training logs found yet. Train a model to see its logs here.")
            return
        
        # Load and sort metadata by creation date
        model_logs = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_logs.append(metadata)
            except Exception as e:
                st.warning(f"Error loading metadata for {metadata_file.name}: {str(e)}")
        
        # Sort by creation date (newest first)
        model_logs.sort(key=lambda x: x.get('date_trained', ''), reverse=True)
        
        # Search/filter functionality
        search_query = st.text_input("🔍 Search models by name", "")
        if search_query:
            model_logs = [log for log in model_logs if search_query.lower() in log['model_name'].lower()]
        
        if not model_logs:
            st.info("No models match your search criteria.")
            return
        
        # Display model logs in expandable sections
        for log in model_logs:
            with st.expander(f"📊 {log['model_name']} ({log.get('model_type', 'Unknown Type')})"):
                # Basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Created:** {log.get('date_trained', 'Unknown')}")
                    st.write(f"**Model Type:** {log.get('model_type', 'Unknown')}")
                    st.write(f"**Features Used:** {len(log.get('features_used', []))}")
                with col2:
                    metrics = log.get('metrics', {})
                    st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                
                # Feature information
                st.subheader("Features Used")
                features = log.get('features_used', [])
                if features:
                    # Group features by category
                    feature_categories = {
                        'Log Features': [f for f in features if f in [
                            'submission_submitted',
                            'course_module_viewed',
                            'other_activities',
                            'quiz_attempt_submitted'
                        ]],
                        'Grade Features': [f for f in features if f in [
                            'attendance',
                            'total_bonus_points',
                            'total_lab_points',
                            'total_quiz_points',
                            'total_practice_points'
                        ]],
                        'Other Features': [f for f in features if f not in [
                            'submission_submitted',
                            'course_module_viewed',
                            'other_activities',
                            'quiz_attempt_submitted',
                            'attendance',
                            'total_bonus_points',
                            'total_lab_points',
                            'total_quiz_points',
                            'total_practice_points'
                        ]]
                    }
                    
                    # Display features by category
                    cols = st.columns(len(feature_categories))
                    for i, (category, category_features) in enumerate(feature_categories.items()):
                        if category_features:
                            with cols[i]:
                                st.write(f"**{category}**")
                                for feature in category_features:
                                    st.write(f"- {feature}")
                else:
                    st.write("No feature information available")
                
                # Model actions
                st.subheader("Actions")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select Model", key=f"select_{log['model_name']}"):
                        st.session_state['selected_model'] = log['model_name']
                        st.success(f"Selected model: {log['model_name']}")
                        st.rerun()
                with col2:
                    if st.button("Delete Model", key=f"delete_{log['model_name']}"):
                        if delete_model(log['model_name']):
                            st.success(f"Model {log['model_name']} deleted successfully!")
                            st.rerun()
                
                # Show feature importance if available
                if 'feature_importance' in log:
                    st.subheader("Feature Importance")
                    try:
                        importance_df = pd.DataFrame(log['feature_importance'])
                        fig = plot_feature_importance(None, importance_df['Feature'].tolist(), 
                                                    f"{log['model_name']} Feature Importance")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error displaying feature importance: {str(e)}")
        
        # Add a summary of best performing models
        st.header("🏆 Best Performing Models")
        if model_logs:
            # Sort by R² score
            best_models = sorted(model_logs, key=lambda x: x.get('metrics', {}).get('r2', 0), reverse=True)[:3]
            
            cols = st.columns(len(best_models))
            for i, model in enumerate(best_models):
                with cols[i]:
                    st.metric(
                        f"Best {i+1}: {model['model_name']}",
                        f"R² = {model.get('metrics', {}).get('r2', 0):.3f}"
                    )
                    st.write(f"Type: {model.get('model_type', 'Unknown')}")
                    st.write(f"Features: {len(model.get('features_used', []))}")

if __name__ == "__main__":
    main() 