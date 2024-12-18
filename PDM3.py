import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Set Streamlit page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Bank Marketing Campaign - Predict Subscription", layout="wide")

# Now, proceed with the rest of your code
# Load dataset
def load_data():
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    return df

# EDA Function
def perform_eda(df):
    st.write("## Exploratory Data Analysis")
    
    # Data preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Summary statistics
    st.write("### Summary Statistics")
    st.dataframe(df.describe())

    # Missing values
    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    # Class distribution
    st.write("### Target Variable Distribution")
    st.bar_chart(df['y'].value_counts())

      # Correlation Heatmap for Numerical Columns
    st.write("### Correlation Heatmap (Numerical Features)")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot()

# Data Preprocessing Function
# Data Preprocessing Function
def preprocess_data(df):
    # Identify available categorical and numerical columns
    cat_cols = [col for col in df.select_dtypes(include=['object']).columns if col in df.columns]
    num_cols = [col for col in df.select_dtypes(include=['number']).columns if col in df.columns]

    # Safely remove the target column 'y' if present
    if 'y' in num_cols:
        num_cols.remove('y')

    # Define preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)],
        remainder='drop')  # Ensure only specified columns are processed

    # Check if 'y' exists before splitting features and target
    if 'y' not in df.columns:
        raise ValueError("Target column 'y' not found in the dataset.")

    # Split features and target
    X = df.drop(columns=['y'])
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Apply preprocessing to X
    X_preprocessed = preprocessor.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

    return X_resampled, y_resampled, preprocessor

# Model Training Function
def train_model(X, y, model_name='Logistic Regression'):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X, y)
    return model

# Model Evaluation Function
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    st.write(f"### Model Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
    st.write(f"ROC-AUC: {roc_auc:.4f}")
    
    return accuracy, precision, recall, f1, roc_auc

# Save model function
def save_model(model, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    st.success(f"Model saved as {filename}")

# Load model function
def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Main App Structure
def main():
    st.title("Bank Marketing Campaign: Predict Subscription to Term Deposit")

    # Load data
    df = load_data()

    # Sidebar navigation
    pages = {
        "Home": home_page,
        "Exploratory Data Analysis (EDA)": lambda: perform_eda(df),
        "Model Training": model_training_page,
        "Model Evaluation": model_evaluation_page,
    }

    page = st.sidebar.selectbox("Choose a page", list(pages.keys()))
    pages[page]()

def home_page():
    st.write("""
    ## Welcome to the Bank Marketing Campaign Prediction App!
    This app helps you predict whether a client will subscribe to a term deposit based on features such as their job, age, and previous campaign outcomes.
    You can explore the data, train machine learning models, and evaluate model performance here.
    """)

# Model Training Page
def model_training_page():
    st.write("## Model Training")
    
    # Load dataset and preprocess
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    
    # Choose model
    model_name = st.selectbox("Choose a model", ['Logistic Regression', 'Random Forest', 'Gradient Boosting'])
    
    # Train model
    model = train_model(X, y, model_name)
    
    # Save model
    if st.button("Save Model"):
        save_model(model)

    st.write(f"### Trained {model_name} Model")
    st.write("You can now evaluate the model or save it for future use.")

# Model Evaluation Page
def model_evaluation_page():
    st.write("## Model Evaluation")
    
    # Load model
    try:
        model = load_model()
        st.write(f"### Evaluating the loaded model")
    except FileNotFoundError:
        st.error("No model found! Please train and save a model first.")
        return
    
    # Load dataset and preprocess
    df = load_data()
    X, y, _ = preprocess_data(df)
    
    # Evaluate the model
    evaluate_model(model, X, y)

if __name__ == "__main__":
    main()

