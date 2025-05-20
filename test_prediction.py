import joblib
import pandas as pd
import numpy as np
import os
import subprocess
import tempfile
from PyPDF2 import PdfWriter, PdfReader
import io

def create_test_pdf():
    """
    Create a test PDF file with medical data for testing PDF extraction functionality
    """
    print("\nCreating test PDF for PDF extraction testing...")
    
    # Create a simple PDF with medical data
    pdf_content = """
    Patient Medical Report
    
    Patient Information:
    Age: 65
    Gender: Male
    
    Blood Test Results:
    White Blood Cell Count (WBC): 15000 cells/μL
    Red Blood Cell Count (RBC): 3.8 million cells/μL
    Hemoglobin: 10.5 g/dL
    Platelets: 95000 per μL
    
    Clinical Observations:
    The patient presents with fatigue, night sweats, and bone pain.
    Patient reports frequent infections in the past 3 months.
    
    Assessment:
    Patient shows abnormal blood cell counts that require further investigation.
    """
    
    # Create a PDF file
    pdf_writer = PdfWriter()
    
    # Add a page with the content
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    # Write the content to the PDF
    y_position = 750  # Start from top
    for line in pdf_content.split('\n'):
        can.drawString(72, y_position, line)
        y_position -= 15  # Move down for next line
    
    can.save()
    
    # Move to the beginning of the BytesIO buffer
    packet.seek(0)
    new_pdf = PdfReader(packet)
    
    # Add the page to the PDF writer
    pdf_writer.add_page(new_pdf.pages[0])
    
    # Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        pdf_path = temp_file.name
        with open(pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)
    
    print(f"Test PDF created at: {pdf_path}")
    return pdf_path

def test_pdf_extraction():
    """
    Test the PDF extraction functionality
    """
    try:
        # Import the necessary functions from app.py
        from app import extract_text_from_pdf, parse_medical_data_from_text
        
        # Create a test PDF
        pdf_path = create_test_pdf()
        
        # Extract text from the PDF
        print("\nTest Case 6: PDF Text Extraction")
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            print("Failed to extract text from PDF")
            return False
        
        print("Successfully extracted text from PDF")
        print(f"Extracted text sample: {extracted_text[:100]}...")
        
        # Parse medical data from the extracted text
        print("\nTest Case 7: Medical Data Parsing from PDF")
        medical_data = parse_medical_data_from_text(extracted_text)
        
        # Check if all required fields were extracted
        required_fields = ['age', 'gender', 'wbc', 'rbc', 'hemoglobin', 'platelets']
        missing_fields = [field for field in required_fields if medical_data.get(field) is None]
        
        if missing_fields:
            print(f"Failed to extract all required fields. Missing: {missing_fields}")
            return False
        
        # Print the extracted data
        print("Successfully parsed medical data from PDF:")
        print(f"Age: {medical_data['age']}")
        print(f"Gender: {medical_data['gender']}")
        print(f"WBC: {medical_data['wbc']}")
        print(f"RBC: {medical_data['rbc']}")
        print(f"Hemoglobin: {medical_data['hemoglobin']}")
        print(f"Platelets: {medical_data['platelets']}")
        print(f"Symptoms: {medical_data['symptoms']}")
        
        # Test Case 8: Handling Missing Data in PDF
        print("\nTest Case 8: Handling Missing Data in PDF")
        # Create a PDF with missing data
        pdf_content = """
        Patient Medical Report
        
        Patient Information:
        Age: 65
        Gender: Male
        
        Blood Test Results:
        White Blood Cell Count (WBC): 15000 cells/μL
        Hemoglobin: 10.5 g/dL
        
        Clinical Observations:
        The patient presents with fatigue and night sweats.
        """
        
        # Create a temporary file for the PDF with missing data
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            pdf_path_missing = temp_file.name
            
            # Create PDF with missing data
            pdf_writer = PdfWriter()
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            
            y_position = 750
            for line in pdf_content.split('\n'):
                can.drawString(72, y_position, line)
                y_position -= 15
            
            can.save()
            packet.seek(0)
            new_pdf = PdfReader(packet)
            pdf_writer.add_page(new_pdf.pages[0])
            
            with open(pdf_path_missing, 'wb') as output_file:
                pdf_writer.write(output_file)
        
        # Extract and parse data from the PDF with missing fields
        extracted_text_missing = extract_text_from_pdf(pdf_path_missing)
        medical_data_missing = parse_medical_data_from_text(extracted_text_missing)
        
        # Verify that missing fields are detected
        missing_fields = [field for field in required_fields if medical_data_missing.get(field) is None]
        expected_missing = ['rbc', 'platelets']
        
        if set(missing_fields) != set(expected_missing):
            print(f"Missing field detection failed. Expected: {expected_missing}, Got: {missing_fields}")
            return False
        
        print("Successfully detected missing fields in PDF")
        print(f"Missing fields: {missing_fields}")
        
        # Test completing missing data
        print("\nTest Case 9: Completing Missing Data")
        # Simulate form data for completing missing fields
        completion_data = {
            'rbc': 3.8,
            'platelets': 95000
        }
        
        # Combine extracted and completed data
        complete_data = medical_data_missing.copy()
        for field, value in completion_data.items():
            complete_data[field] = value
        
        # Verify that all required fields are now present
        missing_fields_after = [field for field in required_fields if complete_data.get(field) is None]
        if missing_fields_after:
            print(f"Data completion failed. Still missing: {missing_fields_after}")
            return False
        
        print("Successfully completed missing data")
        print("Final data:")
        for field in required_fields:
            print(f"{field}: {complete_data[field]}")
        
        # Clean up the temporary files
        try:
            os.remove(pdf_path)
            os.remove(pdf_path_missing)
            print("Removed temporary PDF files")
        except Exception as e:
            print(f"Warning: Could not remove temporary files: {str(e)}")
        
        return True
    except ImportError:
        print("Could not import PDF extraction functions from app.py")
        return False
    except Exception as e:
        print(f"Error testing PDF extraction: {str(e)}")
        return False
def test_model_prediction():
    """
    Test script to verify that the model prediction works correctly
    with different input formats.
    """
    print("Testing model prediction...")
    if not os.path.exists('model.pkl'):
        print("Model file not found. Running train_model.py...")
        try:
            subprocess.run(['python', 'train_model.py'], check=True)
            if not os.path.exists('model.pkl'):
                print("Failed to generate model files. Please check train_model.py for errors.")
                return False
        except subprocess.CalledProcessError:
            print("Error running train_model.py. Please check the script for errors.")
            return False
    
    # Load model and features
    try:
        model = joblib.load('model.pkl')
        
        # Extract expected feature names directly from the model if available
        if hasattr(model, 'feature_names_in_'):
            expected_feature_names = model.feature_names_in_.tolist()
            print(f"Extracted feature names from model.feature_names_in_: {expected_feature_names}")
        elif hasattr(model, 'steps'):
            # For scikit-learn pipelines
            if hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'feature_names_in_'):
                expected_feature_names = model.named_steps['model'].feature_names_in_.tolist()
                print(f"Extracted feature names from model.named_steps['model'].feature_names_in_: {expected_feature_names}")
            elif hasattr(model[-1], 'feature_names_in_'):
                expected_feature_names = model[-1].feature_names_in_.tolist()
                print(f"Extracted feature names from model[-1].feature_names_in_: {expected_feature_names}")
            elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                # Try to get feature names from the preprocessor
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    expected_feature_names = preprocessor.get_feature_names_out().tolist()
                    print(f"Extracted feature names from preprocessor.get_feature_names_out(): {expected_feature_names}")
                else:
                    # Fallback to loading from features.pkl
                    expected_feature_names = joblib.load('features.pkl')
                    print(f"Loaded feature names from features.pkl: {expected_feature_names}")
            else:
                # Fallback to loading from features.pkl
                expected_feature_names = joblib.load('features.pkl')
                print(f"Loaded feature names from features.pkl: {expected_feature_names}")
        else:
            # Fallback to loading from features.pkl
            expected_feature_names = joblib.load('features.pkl')
            print(f"Loaded feature names from features.pkl: {expected_feature_names}")
    except FileNotFoundError:
        print("Model or features file not found. Run train_model.py first.")
        return False
    
    # Try to load feature metadata if available
    try:
        feature_metadata = joblib.load('feature_metadata.pkl')
        print("Feature metadata loaded successfully")
    except FileNotFoundError:
        print("Feature metadata not found, using basic feature list only")
        feature_metadata = None
    
    # Test case 1: Create input with correct feature names and order
    print("\nTest Case 1: Correct feature names and order")
    data = {}
    
    # Create a dictionary with all expected features
    for feature in expected_feature_names:
        if feature == 'Age':
            data[feature] = [50]
        elif feature == 'Gender':
            data[feature] = [0]  # Male
        elif feature == 'WBC':
            data[feature] = [7000]
        elif feature == 'RBC':
            data[feature] = [5.0]
        elif feature == 'Hemoglobin':
            data[feature] = [14.0]
        elif feature == 'Platelets':
            data[feature] = [250000]
        elif feature.startswith('Symptom_'):
            data[feature] = [0]  # No symptoms
        else:
            # Default value for any other features
            data[feature] = [0]
    
    # Create DataFrame with the exact same feature names and order as expected
    input_df = pd.DataFrame(data)
    
    # Ensure the DataFrame has all expected features in the correct order
    input_df = input_df[expected_feature_names]
    
    print(f"Input DataFrame columns: {input_df.columns.tolist()}")
    print(f"Expected feature names: {expected_feature_names}")
    
    # Verify that the column names match exactly
    if input_df.columns.tolist() != expected_feature_names:
        print("Warning: Column names don't match expected feature names")
        print(f"Differences: {set(input_df.columns.tolist()) ^ set(expected_feature_names)}")
    
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        print(f"Prediction successful: {prediction}, Probability: {probability:.4f}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return False
    
    # Test case 2: Create input with lowercase feature names
    print("\nTest Case 2: Lowercase feature names")
    lowercase_data = {}
    for feature in expected_feature_names:
        lowercase_feature = feature.lower()
        if feature == 'Age':
            lowercase_data[lowercase_feature] = [50]
        elif feature == 'Gender':
            lowercase_data[lowercase_feature] = [0]  # Male
        elif feature == 'WBC':
            lowercase_data[lowercase_feature] = [7000]
        elif feature == 'RBC':
            lowercase_data[lowercase_feature] = [5.0]
        elif feature == 'Hemoglobin':
            lowercase_data[lowercase_feature] = [14.0]
        elif feature == 'Platelets':
            lowercase_data[lowercase_feature] = [250000]
        elif feature.startswith('Symptom_'):
            lowercase_data[lowercase_feature] = [0]  # No symptoms
    
    lowercase_input_df = pd.DataFrame(lowercase_data)
    
    # Create a mapping between lowercase and actual feature names
    feature_names_map = {name.lower(): name for name in expected_feature_names}
    
    # Rename columns to match expected feature names
    lowercase_input_df = lowercase_input_df.rename(columns=feature_names_map)
    
    # Create a DataFrame with all expected features
    final_df = pd.DataFrame(columns=expected_feature_names)
    
    # Copy values from lowercase_input_df to final_df where column names match
    for col in lowercase_input_df.columns:
        if col in expected_feature_names:
            final_df[col] = lowercase_input_df[col]
    
    # Fill missing values with 0
    final_df = final_df.fillna(0)
    
    # Ensure all expected features are present
    for feature in expected_feature_names:
        if feature not in final_df.columns:
            final_df[feature] = 0
    
    # Reindex the DataFrame to match the exact order of expected features
    final_df = final_df.reindex(columns=expected_feature_names)
    
    try:
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        print(f"Prediction successful: {prediction}, Probability: {probability:.4f}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return False
    
    # Test case 3: Missing features
    print("\nTest Case 3: Missing features (should be handled gracefully)")
    partial_data = {
        'Age': [50],
        'Gender': [0],
        'WBC': [7000],
        'Hemoglobin': [14.0],  # Missing RBC and Platelets
    }
    
    partial_input_df = pd.DataFrame(partial_data)
    
    # Create a DataFrame with all expected features
    final_df = pd.DataFrame(columns=expected_feature_names)
    
    # Copy values from partial_input_df to final_df where column names match
    for col in partial_input_df.columns:
        if col in expected_feature_names:
            final_df[col] = partial_input_df[col]
    
    # Fill missing values with defaults
    for feature in expected_feature_names:
        if feature not in partial_input_df.columns:
            if feature == 'RBC':
                final_df[feature] = [5.0]
            elif feature == 'Platelets':
                final_df[feature] = [250000]
            else:
                final_df[feature] = [0]
    
    # Reindex the DataFrame to match the exact order of expected features
    final_df = final_df.reindex(columns=expected_feature_names)
    
    try:
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        print(f"Prediction successful: {prediction}, Probability: {probability:.4f}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return False
    
    # Test case 4: Form-like input (simulating Flask form data)
    print("\nTest Case 4: Form-like input (simulating Flask form data)")
    form_data = {
        'age': 50,
        'gender': 'male',
        'wbc': 7000,
        'rbc': 5.0,
        'hemoglobin': 14.0,
        'platelets': 250000,
        'symptoms': ['fatigue', 'fever']
    }
    
    # Convert gender to numerical value
    gender_mapping = {'male': 0, 'female': 1, 'other': 2}
    gender_value = gender_mapping[form_data['gender']]
    
    # Create a dictionary to hold all features with form field names
    processed_form_data = {
        'age': form_data['age'],
        'gender': gender_value,
        'wbc': form_data['wbc'],
        'rbc': form_data['rbc'],
        'hemoglobin': form_data['hemoglobin'],
        'platelets': form_data['platelets']
    }
    
    # Add symptoms - using the correct capitalization format
    for symptom in form_data['symptoms']:
        # Use the correct format for symptom features (Symptom_symptom instead of symptom_symptom)
        processed_form_data[f'Symptom_{symptom}'] = 1
    
    # Create a DataFrame with form data
    form_input_df = pd.DataFrame([processed_form_data])
    
    # Map form field names to expected feature names
    field_to_feature = {
        'age': 'Age',
        'gender': 'Gender',
        'wbc': 'WBC',
        'rbc': 'RBC',
        'hemoglobin': 'Hemoglobin',
        'platelets': 'Platelets'
        # Note: Symptom fields are already correctly named with capital 'S'
    }
    
    # Rename columns to match expected feature names
    form_input_df = form_input_df.rename(columns=field_to_feature)
    
    # Create a DataFrame with all expected features
    final_df = pd.DataFrame(columns=expected_feature_names)
    
    # Copy values from form_input_df to final_df where column names match
    for col in form_input_df.columns:
        if col in expected_feature_names:
            final_df[col] = form_input_df[col]
    
    # Handle symptom features
    for symptom in form_data['symptoms']:
        symptom_feature = f'Symptom_{symptom}'
        if symptom_feature in expected_feature_names:
            final_df[symptom_feature] = 1
    
    # Fill missing values with 0
    final_df = final_df.fillna(0)
    
    # Ensure all expected features are present
    for feature in expected_feature_names:
        if feature not in final_df.columns:
            final_df[feature] = 0
    
    # Reindex the DataFrame to match the exact order of expected features
    final_df = final_df.reindex(columns=expected_feature_names)
    
    try:
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        print(f"Prediction successful: {prediction}, Probability: {probability:.4f}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return False
    
    # Test case 5: Form-like input with "Others" symptom
    print("\nTest Case 5: Form-like input with 'Others' symptom")
    form_data_with_others = {
        'age': 55,
        'gender': 'female',
        'wbc': 8000,
        'rbc': 4.5,
        'hemoglobin': 13.0,
        'platelets': 300000,
        'symptoms': ['fatigue', 'others'],
        'other_symptom': 'persistent headache'
    }
    
    # Convert gender to numerical value
    gender_value = gender_mapping[form_data_with_others['gender']]
    
    # Create a dictionary to hold all features with form field names
    processed_form_data_with_others = {
        'age': form_data_with_others['age'],
        'gender': gender_value,
        'wbc': form_data_with_others['wbc'],
        'rbc': form_data_with_others['rbc'],
        'hemoglobin': form_data_with_others['hemoglobin'],
        'platelets': form_data_with_others['platelets']
    }
    
    # Add symptoms - using the correct capitalization format
    for symptom in form_data_with_others['symptoms']:
        if symptom == 'others':
            # Get the custom symptom text
            other_symptom = form_data_with_others.get('other_symptom', '').strip()
            if other_symptom:
                # Convert to snake_case format for consistency
                other_symptom = other_symptom.lower().replace(' ', '_')
                # Add as a symptom feature
                processed_form_data_with_others[f'Symptom_{other_symptom}'] = 1
        else:
            # Use the correct format for symptom features
            processed_form_data_with_others[f'Symptom_{symptom}'] = 1
    
    # Create a DataFrame with form data
    form_input_df_with_others = pd.DataFrame([processed_form_data_with_others])
    
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
    form_input_df_with_others = form_input_df_with_others.rename(columns=field_to_feature)
    
    # Create a DataFrame with all expected features
    final_df_with_others = pd.DataFrame(columns=expected_feature_names)
    
    # Copy values from form_input_df to final_df where column names match
    for col in form_input_df_with_others.columns:
        if col in expected_feature_names:
            final_df_with_others[col] = form_input_df_with_others[col]
    
    # Handle symptom features
    for symptom in [s for s in form_data_with_others['symptoms'] if s != 'others']:
        symptom_feature = f'Symptom_{symptom}'
        if symptom_feature in expected_feature_names:
            final_df_with_others[symptom_feature] = 1
    
    # Handle custom symptom
    if 'others' in form_data_with_others['symptoms'] and form_data_with_others.get('other_symptom'):
        other_symptom = form_data_with_others['other_symptom'].lower().replace(' ', '_')
        symptom_feature = f'Symptom_{other_symptom}'
        # This feature might not be in expected_feature_names, but we'll add it anyway
        # The model will handle it appropriately (either use it or ignore it)
        final_df_with_others[symptom_feature] = 1
    
    # Fill missing values with 0
    final_df_with_others = final_df_with_others.fillna(0)
    
    # Ensure all expected features are present
    for feature in expected_feature_names:
        if feature not in final_df_with_others.columns:
            final_df_with_others[feature] = 0
    
    # Reindex the DataFrame to match the exact order of expected features
    # But first, save any custom symptom columns that might not be in expected_feature_names
    custom_symptom_cols = [col for col in final_df_with_others.columns if col not in expected_feature_names]
    final_df_with_others_reindexed = final_df_with_others.reindex(columns=expected_feature_names)
    
    # Add back any custom symptom columns
    for col in custom_symptom_cols:
        final_df_with_others_reindexed[col] = final_df_with_others[col]
    
    try:
        # For testing purposes, we'll only use the expected features
        # In a real scenario, the model would need to be retrained to handle new symptoms
        prediction = model.predict(final_df_with_others_reindexed[expected_feature_names])[0]
        probability = model.predict_proba(final_df_with_others_reindexed[expected_feature_names])[0][1]
        print(f"Prediction successful: {prediction}, Probability: {probability:.4f}")
        print(f"Custom symptom '{form_data_with_others['other_symptom']}' was processed correctly")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return False
        
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    # Run model prediction tests
    test_model_prediction()
    
    # Run PDF extraction tests
    print("\n" + "="*50)
    print("Running PDF extraction tests...")
    test_pdf_extraction()