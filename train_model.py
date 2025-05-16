import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('blood_cancer_dataset.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Data Preprocessing
print("\nPreprocessing data...")

# Drop unnecessary columns
columns_to_drop = ['Patient_ID', 'Date_of_Test', 'Cancer_Type', 'Stage']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Encode Gender
gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
df['Gender'] = df['Gender'].map(gender_mapping)
print("Encoded Gender column")

# One-hot encode the Symptoms column
# First, create a list of all possible symptoms
all_symptoms = set()
for symptoms in df['Symptoms'].str.split(','):
    if isinstance(symptoms, list):
        all_symptoms.update(symptoms)

# Remove 'none' from symptoms if it exists
if 'none' in all_symptoms:
    all_symptoms.remove('none')

# Create binary columns for each symptom
for symptom in all_symptoms:
    df[f'Symptom_{symptom}'] = df['Symptoms'].str.contains(symptom).astype(int)

# Drop the original Symptoms column
df = df.drop(columns=['Symptoms'])
print(f"One-hot encoded Symptoms into {len(all_symptoms)} binary columns")

# Encode Diagnosis_Result
diagnosis_mapping = {'Negative': 0, 'Positive': 1}
df['Diagnosis_Result'] = df['Diagnosis_Result'].map(diagnosis_mapping)
print("Encoded Diagnosis_Result column")

# Split features and target
X = df.drop(columns=['Diagnosis_Result'])
y = df['Diagnosis_Result']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Split data into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets")

# Identify numerical columns for normalization
numerical_cols = ['Age', 'WBC', 'RBC', 'Hemoglobin', 'Platelets']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='passthrough'
)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

results = {}
best_model = None
best_accuracy = 0

print("\nTraining and evaluating models...")
for name, model in models.items():
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Update best model if current model has higher accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

# Print the best model
print(f"\nBest Model: {best_model} with accuracy {best_accuracy:.4f}")

# Save the best model
with open('model.pkl', 'wb') as file:
    joblib.dump(results[best_model]['pipeline'], file)
print(f"Saved {best_model} as model.pkl")

# Get the pipeline
best_pipeline = results[best_model]['pipeline']

# Create a feature list for the Flask application
feature_list = X.columns.tolist()  # Use the original feature list from the training data
with open('features.pkl', 'wb') as file:
    joblib.dump(feature_list, file)
print("Saved feature list as features.pkl")

# Save additional feature metadata for better compatibility
feature_metadata = {
    'feature_list': feature_list,
    'numerical_features': numerical_cols,
    'categorical_features': [col for col in X.columns if col not in numerical_cols and not col.startswith('Symptom_')],
    'symptom_features': [col for col in X.columns if col.startswith('Symptom_')],
    'feature_types': {col: str(X[col].dtype) for col in X.columns}
}
with open('feature_metadata.pkl', 'wb') as file:
    joblib.dump(feature_metadata, file)
print("Saved feature metadata as feature_metadata.pkl")

# Plot confusion matrices
plt.figure(figsize=(15, 5))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
import os
os.makedirs('static', exist_ok=True)
plt.savefig('static/confusion_matrices.png')
print("Saved confusion matrices visualization")

# Plot model comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
f1_scores = [results[name]['f1_score'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy')
plt.bar(x + width/2, f1_scores, width, label='F1 Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.tight_layout()
os.makedirs('static', exist_ok=True)
plt.savefig('static/model_comparison.png')
print("Saved model comparison visualization")

print("\nModel training and evaluation complete!")