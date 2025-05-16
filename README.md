# Feature Name Handling in Blood Cancer Prediction App

This project has been updated to handle feature name mismatches between the trained model and prediction input data. The key improvements include:

1. Using joblib instead of pickle for model serialization
2. Extracting expected feature names directly from the model
3. Properly handling feature name mismatches during prediction
4. Adding comprehensive test cases for different input formats

## Key Changes

### 1. Model Loading with joblib

The application now uses joblib instead of pickle for loading and saving the model:

```python
# Load model using joblib instead of pickle
model = joblib.load('model.pkl')
```

### 2. Feature Name Extraction

The application now extracts expected feature names directly from the model:

```python
# Extract expected feature names directly from the model
if hasattr(model, 'feature_names_in_'):
    expected_feature_names = model.feature_names_in_.tolist()
elif hasattr(model, 'steps') and hasattr(model[-1], 'feature_names_in_'):
    # For pipelines, the last step might have the feature names
    expected_feature_names = model[-1].feature_names_in_.tolist()
else:
    # Fallback to loading from features.pkl
    expected_feature_names = joblib.load('features.pkl')
```

### 3. Feature Name Mismatch Handling

The prediction route now properly handles feature name mismatches:

```python
# Create a DataFrame with all expected features
final_df = pd.DataFrame(columns=expected_feature_names)

# Copy values from input_df to final_df where column names match
for col in input_df.columns:
    if col in expected_feature_names:
        final_df[col] = input_df[col]

# Fill missing values with 0
final_df = final_df.fillna(0)

# Reindex the DataFrame to match the exact order of expected features
final_df = final_df.reindex(columns=expected_feature_names)
```

### 4. Comprehensive Testing

The test_prediction.py script now includes a new test case for form-like input:

```python
# Test case 4: Form-like input (simulating Flask form data)
form_data = {
    'age': 50,
    'gender': 'male',
    'wbc': 7000,
    'rbc': 5.0,
    'hemoglobin': 14.0,
    'platelets': 250000,
    'symptoms': ['fatigue', 'fever']
}
```

## Usage

1. On the home page, fill in the patient information form with:
   - Age and gender
   - Blood test results (WBC, RBC, Hemoglobin, Platelets)
   - Select any symptoms the patient is experiencing

2. Click the "Predict Risk" button to submit the form

3. View the prediction results, which include:
   - Blood cancer risk assessment (Positive/Negative)
   - Risk level (High/Medium/Low)
   - Probability percentage
   - Recommended next steps

## Feature Handling and Compatibility

The application includes robust feature handling to ensure compatibility between training and prediction:

### Feature Name and Order Consistency

- Feature names are extracted directly from the model using `model.feature_names_in_`
- During prediction, input data is automatically aligned to match the expected features
- Case-insensitive feature matching helps prevent errors from capitalization differences

### Feature Metadata

- Additional feature metadata is stored in `feature_metadata.pkl`
- Includes information about numerical features, categorical features, and symptom features
- Data types for each feature are preserved to ensure type consistency

### Graceful Error Handling

- Missing features are automatically added with default values
- Extra features are ignored
- Validation is performed before prediction to ensure data quality
- Detailed error messages help diagnose any issues

### Testing

The `test_prediction.py` script can be used to verify that the model works correctly with different input formats:
```
python test_prediction.py
```

## Disclaimer

This application is for educational and demonstration purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
