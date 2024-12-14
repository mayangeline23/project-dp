import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import ssl
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load datasets
def load_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    # Read the CSV file
    data = pd.read_csv(url, header=None, names=columns)
    
    # Replace '?' with NaN
    data.replace('?', pd.NA, inplace=True)
    
    # Convert to numeric and handle missing data
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values or handle them as needed (e.g., fill with mean or median)
    data = data.dropna().astype(float)
    
    return data
    
@st.cache_data
def load_diabetes_data():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    data = diabetes.data
    data["target"] = (diabetes.target > 140).astype(int)  # Binarize target
    return data

@st.cache_data
def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer(as_frame=True)
    data = cancer.data
    data["target"] = cancer.target
    return data

@st.cache_data
def load_liver_disorders_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
    columns = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
    data = pd.read_csv(url, header=None, names=columns)
    return data

# Unified function for training and evaluation
def train_and_evaluate_model(X, y, model_type="RandomForest"):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=2000, random_state=42)  # Increased max_iter for convergence
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy_score(y_test, y_pred), pd.DataFrame(report).transpose()

# Custom CSS to change the font to Times New Roman for all titles and markdown content
st.markdown("""
    <style>
        /* Set Times New Roman for all titles and markdown */
        body {
            font-family: 'Times New Roman', serif;
        }

        /* Title Styling */
        .title {
            font-family: 'Times New Roman', serif;
            font-size: 36px;
            color: #2C3E50;
            text-align: center;
            padding-bottom: 20px;
        }

        /* Dataset Description Section */
        .dataset-description {
            font-family: 'Times New Roman', serif;
            font-size: 18px;
            color: #34495E;
            line-height: 1.5;
        }

        /* Model and Evaluation Section */
        .model-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Visualization Section */
        .visualization-section {
            margin-top: 30px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Subheader Font */
        .streamlit-expanderHeader {
            font-family: 'Times New Roman', serif;
            font-size: 24px;
            font-weight: bold;
            color: #1E90FF;
        }

    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("Health Disease Prediction🩺")

# Dataset selection below the title
dataset_choice = st.selectbox(
    "Choose a dataset to explore:",
    ["Select", "Heart Disease", "Chronic Kidney Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

# Dataset descriptions
dataset_info = {
    "Heart Disease": {
        "description": "This dataset contains 303 records and 14 attributes related to diagnosing heart disease. The target variable indicates the presence of heart disease (1: disease, 0: no disease).",
        "source": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "attributes": [
            "age", "sex", "cp (chest pain type)", "trestbps (resting blood pressure)",
            "chol (serum cholesterol)", "fbs (fasting blood sugar)", "restecg (resting ECG)",
            "thalach (max heart rate achieved)", "exang (exercise-induced angina)",
            "oldpeak (ST depression)", "slope (slope of peak exercise ST segment)",
            "ca (number of vessels colored by fluoroscopy)", "thal (thalassemia)", "target"
        ]
    },
    "Diabetes": {
        "description": "The diabetes dataset from sklearn consists of 442 records with 10 attributes. It is used for regression but has been modified here for classification by binarizing the target variable.",
        "source": "https://archive.ics.uci.edu/dataset/34/diabetes",
        "attributes": list(load_diabetes_data().columns)
    },
    "Breast Cancer": {
        "description": "This dataset contains 569 records and 30 attributes related to diagnosing breast cancer. The target variable indicates whether the cancer is malignant or benign.",
        "source": "https://archive.ics.uci.edu/dataset/14/breast+cancer",
        "attributes": list(load_breast_cancer_data().columns)
    },
    "Liver Disorders": {
        "description": "The liver disorders dataset contains 345 records and 7 attributes related to the diagnosis of liver disorders.",
        "source": "https://archive.ics.uci.edu/ml/datasets/Liver+Disorders",
        "attributes": ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
    }
}
# Dataset selection logic
if dataset_choice != "Select":
    # Custom CSS for increasing the font size of the dataset title
    st.markdown(f"<h2 style='font-size: 32px;'>{dataset_choice}</h2>", unsafe_allow_html=True)
    # Load the dataset and process it
    if dataset_choice == "Heart Disease":
        data = load_heart_disease_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Diabetes":
        data = load_diabetes_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Liver Disorders":
        data = load_liver_disorders_data()
        X = data.drop("selector", axis=1)
        y = data["selector"]

    # Inject custom CSS for subheader font style
    st.markdown("""<style>
    .streamlit-expanderHeader {
        font-family: 'Courier New', Courier, monospace;
        font-size: 18px;
        font-weight: bold;
        color: #1E90FF;
    }
    </style>""", unsafe_allow_html=True)

    # Display dataset information
    st.subheader("Dataset Information")
    st.write(dataset_info[dataset_choice]["description"])
    st.markdown(f"**Source:** [Dataset Link]({dataset_info[dataset_choice]['source']})")
    st.markdown("**Attributes:**")
    st.write(dataset_info[dataset_choice]["attributes"])


    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write("""This section provides a detailed look into the structure and contents of the dataset.
    It includes an overview of the dataset\'s attributes, size, and any notable characteristics or potential data quality issues.""")
    st.write(f"Shape: {data.shape}")
    st.write(data.head())

    # Model Selection and Evaluation
    model_choice = st.selectbox(
        "Choose a model:",
        ["RandomForest", "LogisticRegression", "GradientBoosting"]
    )

    if st.button("Train and Evaluate Model"):
        st.subheader("Model Performance")
        accuracy, report_df = train_and_evaluate_model(X, y, model_type=model_choice)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(report_df)

        # Feature Importance for Tree-based models
        if model_choice in ["RandomForest", "GradientBoosting"]:
            model = RandomForestClassifier(random_state=42) if model_choice == "RandomForest" else GradientBoostingClassifier(random_state=42)
            model.fit(X, y)
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("Feature Importance")
            st.bar_chart(feature_importances)

    st.markdown("---")

    # Visualization Section (only shown after dataset is selected)
    st.subheader("Visualization Options")
    st.write("""
    Visualizations provide insights into the dataset's structure and relationships 
    between features. Use the tools below to explore pairwise feature relationships 
    and correlation patterns.
    """)

    # Pairplot Visualization
    if st.checkbox("Show Pairplot"):
        st.subheader("Pairplot")
        st.write("A pairplot displays pairwise relationships in the dataset.")
        fig = sns.pairplot(data).fig  # Get the figure object
        st.pyplot(fig)

    # Correlation Heatmap Visualization
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        st.write("This heatmap shows the correlation coefficients between features.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

else:
    st.write("Please select a dataset from the dropdown to begin.")