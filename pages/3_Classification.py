import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

# --- Page Configuration ---
st.set_page_config(
    page_title="Classification",
    page_icon="🤖",
)
st.title("🤖 Iris Flower Classification")
st.markdown("Use the sidebar sliders to input the measurements and predict the species.")

# 1. Load Data and Train Model (Cached)
@st.cache_resource
def train_model():
    """Loads data, trains a Random Forest model, and returns the model and feature names."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data for a basic test (though we primarily use it for prediction here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy (for display only)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")
    
    return model, iris.feature_names, iris.target_names

# Load the trained model and names
model, feature_names, target_names = train_model()

# 2. Input Features using Streamlit Sliders
st.sidebar.header("Input Features")

# Create a dictionary to hold user inputs
input_data = {}
for i, feature in enumerate(feature_names):
    # Determine min, max, and default values based on the dataset
    # (Simplified min/max/default logic for display)
    min_val = round(load_iris().data[:, i].min(), 1)
    max_val = round(load_iris().data[:, i].max(), 1)
    default_val = round(load_iris().data[:, i].mean(), 1)
    
    # Create the slider
    input_data[feature] = st.sidebar.slider(
        feature.replace(" (cm)", ""), # Remove (cm) for cleaner display
        min_val,
        max_val,
        default_val,
        step=0.1
    )

# Convert user inputs to a DataFrame suitable for prediction
input_df = pd.DataFrame([input_data])

# 3. Prediction Button and Output
st.header("Prediction")

if st.button('Predict Species'):
    # Make prediction
    prediction_id = model.predict(input_df)[0]
    prediction_species = target_names[prediction_id].capitalize()
    
    # Display result
    st.success(f"The predicted Iris species is: **{prediction_species}**")
    
    # Optional: Show feature importance (if desired)
    st.subheader("Model Feature Importance:")
    
    # Create a DataFrame for importance visualization
    importance_df = pd.DataFrame({
        'Feature': [f.replace(" (cm)", "") for f in feature_names],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))