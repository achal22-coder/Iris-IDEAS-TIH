import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier # New Import
from sklearn.svm import SVC # New Import
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Good practice for SVM

# --- Page Configuration ---
st.set_page_config(
    page_title="Classification Comparison",
    page_icon="🤖",
)
st.title("🤖 Iris Flower Classification Comparison")
st.markdown("Use the sidebar to input measurements and select a model for prediction.")

# 1. Load Data, Scale, and Train Models (Cached)
@st.cache_resource
def train_models():
    """Loads data, scales features, trains RF, KNN, and SVM models, and returns them."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data (Essential for SVM and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine (SVM)': SVC(gamma='auto', random_state=42)
    }
    
    accuracies = {}
    
    for name, model in models.items():
        # Use scaled data for KNN and SVM
        if name in ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else: # Use unscaled data for Random Forest (it doesn't require scaling)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracies[name] = accuracy_score(y_test, y_pred)
        
    return models, iris.feature_names, iris.target_names, scaler, accuracies

# Load trained models, names, scaler, and accuracies
models, feature_names, target_names, scaler, accuracies = train_models()

# 2. Sidebar Configuration and Input Features
st.sidebar.header("Input Features & Model Selection")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Select Classification Model:",
    list(models.keys())
)
selected_model = models[selected_model_name]

st.sidebar.markdown(f"**{selected_model_name} Accuracy:** **{accuracies[selected_model_name]:.2f}**")
st.sidebar.markdown("---")

# Input Features using Streamlit Sliders
input_data = {}
for i, feature in enumerate(feature_names):
    min_val = round(load_iris().data[:, i].min(), 1)
    max_val = round(load_iris().data[:, i].max(), 1)
    default_val = round(load_iris().data[:, i].mean(), 1)
    
    input_data[feature] = st.sidebar.slider(
        feature.replace(" (cm)", ""),
        min_val,
        max_val,
        default_val,
        step=0.1
    )

# Convert user inputs to a DataFrame
input_df = pd.DataFrame([input_data])

# 3. Prediction and Output
st.header(f"Prediction using: {selected_model_name}")
st.subheader("Your Input:")
st.dataframe(input_df)

if st.button(f'Predict Species with {selected_model_name}'):
    
    # Scaling input data only if the selected model requires it
    if selected_model_name in ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)']:
        input_data_scaled = scaler.transform(input_df)
        prediction_input = input_data_scaled
    else:
        prediction_input = input_df

    # Make prediction
    prediction_id = selected_model.predict(prediction_input)[0]
    prediction_species = target_names[prediction_id].capitalize()
    
    # Display result
    st.success(f"The predicted Iris species using **{selected_model_name}** is: **{prediction_species}**")
    
    # Display all model accuracies for comparison
    st.markdown("---")
    st.subheader("Model Comparison (Test Accuracy)")
    accuracy_data = pd.DataFrame(
        list(accuracies.items()), 
        columns=['Model', 'Accuracy']
    ).set_index('Model')
    
    st.dataframe(accuracy_data.style.highlight_max(axis=0, color='lightgreen'))