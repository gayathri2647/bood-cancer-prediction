from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and features if they exist
model = None
expected_feature_names = None
feature_names_map = None
feature_metadata = None

def load_model_and_features():
    global model, expected_feature_names, feature_names_map, feature_metadata
    try:
        # Load model using joblib instead of pickle
        model = joblib.load('model.pkl')
        
        # Extract expected feature names directly from the model
        # For scikit-learn pipelines, the feature names are stored in the preprocessor
        if hasattr(model, 'feature_names_in_'):
            expected_feature_names = model.feature_names_in_.tolist()
        elif hasattr(model, 'steps'):
            # For scikit-learn pipelines
            if hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'feature_names_in_'):
                expected_feature_names = model.named_steps['model'].feature_names_in_.tolist()
            elif hasattr(model[-1], 'feature_names_in_'):
                expected_feature_names = model[-1].feature_names_in_.tolist()
            elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                # Try to get feature names from the preprocessor
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    expected_feature_names = preprocessor.get_feature_names_out().tolist()
                else:
                    # Fallback to loading from features.pkl
                    expected_feature_names = joblib.load('features.pkl')
            else:
                # Fallback to loading from features.pkl
                expected_feature_names = joblib.load('features.pkl')
        else:
            # Fallback to loading from features.pkl if model doesn't have feature_names_in_
            expected_feature_names = joblib.load('features.pkl')
        
        print(f"Expected feature names: {expected_feature_names}")
        
        # Create a mapping between lowercase and actual feature names
        # This helps handle case-insensitive feature matching
        feature_names_map = {name.lower(): name for name in expected_feature_names}
        
        # Try to load the feature metadata if available
        try:
            feature_metadata = joblib.load('feature_metadata.pkl')
            print("Loaded feature metadata successfully")
        except FileNotFoundError:
            print("Feature metadata not found, using basic feature list only")
            feature_metadata = {
                'feature_list': expected_feature_names,
                'numerical_features': ['Age', 'WBC', 'RBC', 'Hemoglobin', 'Platelets'],
                'categorical_features': ['Gender'],
                'symptom_features': [f for f in expected_feature_names if f.startswith('Symptom_')]
            }
        
        return True
    except FileNotFoundError:
        return False

# Define the symptoms list for the form
symptoms = [
    'fatigue', 'fever', 'night_sweats', 'weight_loss', 
    'bleeding_gums', 'bone_pain', 'frequent_infections'
]

def validate_features(input_df, expected_features):
    """
    Utility function to validate input features against expected features.
    
    Args:
        input_df: DataFrame containing input features
        expected_features: List of expected feature names
        
    Returns:
        tuple: (is_valid, message)
            is_valid: Boolean indicating if validation passed
            message: String with validation details
    """
    validation_messages = []
    
    # Check for missing features
    missing_features = set(expected_features) - set(input_df.columns)
    if missing_features:
        validation_messages.append(f"Missing features: {missing_features}")
    
    # Check for extra features
    extra_features = set(input_df.columns) - set(expected_features)
    if extra_features:
        validation_messages.append(f"Extra features: {extra_features}")
    
    # Check column order
    if list(input_df.columns) != expected_features:
        validation_messages.append("Column order doesn't match expected order")
    
    # Check for NaN values
    nan_columns = input_df.columns[input_df.isna().any()].tolist()
    if nan_columns:
        validation_messages.append(f"NaN values found in columns: {nan_columns}")
    
    is_valid = len(validation_messages) == 0
    message = "Feature validation passed" if is_valid else "; ".join(validation_messages)
    
    return is_valid, message

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if model is loaded
        if model is None or expected_feature_names is None or feature_names_map is None:
            return render_template('index.html', 
                                  symptoms=symptoms, 
                                  error="Model not loaded. Please try again later.")
        
        try:
            # Get form data
            age = float(request.form['age'])
            gender = request.form['gender']
            wbc = float(request.form['wbc'])
            rbc = float(request.form['rbc'])
            hemoglobin = float(request.form['hemoglobin'])
            platelets = float(request.form['platelets'])
            
            # Convert gender to numerical value
            gender_mapping = {'male': 0, 'female': 1, 'other': 2}
            gender_value = gender_mapping[gender]
            
            # Create a dictionary to hold all features with form field names
            form_data = {
                'age': age,
                'gender': gender_value,
                'wbc': wbc,
                'rbc': rbc,
                'hemoglobin': hemoglobin,
                'platelets': platelets
            }
            
            # Get selected symptoms from form
            selected_symptoms = request.form.getlist('symptoms')
            for symptom in selected_symptoms:
                # Check if the symptom is "others"
                if symptom == "others":
                    # Get the custom symptom text
                    other_symptom = request.form.get('other_symptom', '').strip()
                    if other_symptom:
                        # Convert to snake_case format for consistency
                        other_symptom = other_symptom.lower().replace(' ', '_')
                        # Add as a symptom feature
                        form_data[f'Symptom_{other_symptom}'] = 1
                else:
                    # Use the correct format for symptom features (Symptom_symptom instead of symptom_symptom)
                    form_data[f'Symptom_{symptom}'] = 1
            
            # Create a DataFrame with form data
            input_df = pd.DataFrame([form_data])
            
            # Map form field names to expected feature names
            field_to_feature = {
                'age': 'Age',
                'gender': 'Gender',
                'wbc': 'WBC',
                'rbc': 'RBC',
                'hemoglobin': 'Hemoglobin',
                'platelets': 'Platelets'
            }
            
            # Rename columns to match expected feature names
            input_df = input_df.rename(columns=field_to_feature)
            
            # Create a DataFrame with all expected features
            final_df = pd.DataFrame(columns=expected_feature_names)
            
            # Copy values from input_df to final_df where column names match
            for col in input_df.columns:
                if col in expected_feature_names:
                    final_df[col] = input_df[col]
            
            # Handle symptom features
            for symptom in selected_symptoms:
                if symptom == "others":
                    # Skip "others" as it's already handled when processing the form data
                    continue
                symptom_feature = f'Symptom_{symptom}'
                if symptom_feature in expected_feature_names:
                    final_df[symptom_feature] = 1
                    
            # Handle custom symptom if "others" was selected
            if "others" in selected_symptoms:
                other_symptom = request.form.get('other_symptom', '').strip()
                if other_symptom:
                    other_symptom = other_symptom.lower().replace(' ', '_')
                    symptom_feature = f'Symptom_{other_symptom}'
                    # If this symptom is in the expected features, set it
                    if symptom_feature in expected_feature_names:
                        final_df[symptom_feature] = 1
                    # Otherwise, we'll note it but can't use it for prediction without retraining
                    else:
                        print(f"Custom symptom '{other_symptom}' not in model's expected features. Consider retraining the model.")
                        # We could store this for future model improvements
            
            # Fill missing values with 0
            final_df = final_df.fillna(0)
            
            # Ensure all expected features are present
            for feature in expected_feature_names:
                if feature not in final_df.columns:
                    final_df[feature] = 0
                    print(f"Added missing feature: {feature}")
            
            # Reindex the DataFrame to match the exact order of expected features
            final_df = final_df.reindex(columns=expected_feature_names)
            
            # Validate features before prediction
            is_valid, validation_message = validate_features(final_df, expected_feature_names)
            if not is_valid:
                print(f"Feature validation warning: {validation_message}")
            
            # Apply appropriate data types if available in metadata
            if feature_metadata and 'feature_types' in feature_metadata:
                for col, dtype_str in feature_metadata['feature_types'].items():
                    if col in final_df.columns:
                        try:
                            # Convert string representation of dtype back to actual dtype
                            if 'int' in dtype_str:
                                final_df[col] = final_df[col].astype(int)
                            elif 'float' in dtype_str:
                                final_df[col] = final_df[col].astype(float)
                        except Exception as e:
                            print(f"Warning: Could not convert {col} to {dtype_str}: {e}")
            
            # Make prediction
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]  # Probability of positive class
            
            # Determine result
            result = "Positive" if prediction == 1 else "Negative"
            risk_level = "High" if probability > 0.75 else "Medium" if probability > 0.5 else "Low"
            
            return render_template('result.html', 
                                  prediction=result, 
                                  probability=round(probability * 100, 2),
                                  risk_level=risk_level)
                                  
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            print(error_message)
            return render_template('index.html', 
                                  symptoms=symptoms, 
                                  error=error_message)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Check if model exists, if not, train it
    if not load_model_and_features():
        print("Model or features file not found. Running train_model.py...")
        import subprocess
        try:
            subprocess.run(['python', 'train_model.py'], check=True)
            if not load_model_and_features():
                print("Failed to generate model files. Please check train_model.py for errors.")
                exit(1)
        except subprocess.CalledProcessError:
            print("Error running train_model.py. Please check the script for errors.")
            exit(1)
    
    app.run(debug=True)