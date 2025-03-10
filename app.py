import os
import pandas as pd
import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration - Update these paths as needed
MODEL_PATH = "extra_tree_model.pkl"  # Path to your saved top model
SCALER_PATH = "scaler.pkl"     # Path to saved scaler
RFECV_PATH = "rfe_selector.pkl" # Path to saved RFECV object
TB_PEAKS = [10660, 10100, 9768, 9813, 7931, 7974]  # Same as training peaks

def baseline_correction(df):
    """Apply baseline correction (dummy function, customize as needed)"""
    # Example: Subtract minimum intensity from all intensities for baseline correction
    df["Intensity"] = df["Intensity"] - df["Intensity"].min()
    return df

def get_peak_features(df, peak):
    """Extract peak features such as Intensity, FWHM, and Area around the given peak"""
    # Example: This is a simplified version for demonstration purposes.
    # A more complex method could be used for real analysis.
    peak_window = df[(df["m/z"] > peak - 20) & (df["m/z"] < peak + 20)]
    intensity = peak_window["Intensity"].max()
    fwhm = 10  # Dummy value, replace with actual FWHM calculation
    area = peak_window["Intensity"].sum()
    
    return {"Intensity": intensity, "FWHM": fwhm, "Area": area}

def process_single_file(file_path):
    """Process a single .txt file for prediction"""
    # Load the raw data
    df = pd.read_csv(file_path, sep='\s+', header=None, 
                    names=["m/z", "Intensity"])
    
    # Apply preprocessing pipeline
    processed_df = baseline_correction(df.copy())
    
    # Extract features (same as training)
    features = []
    for peak in TB_PEAKS:
        peak_features = get_peak_features(processed_df, peak)
        features.extend([peak_features['Intensity'],
                       peak_features['FWHM'],
                       peak_features['Area']])
    
    return np.array(features).reshape(1, -1)

# Load required artifacts
scaler = joblib.load(SCALER_PATH)
rfe_selector = joblib.load(RFECV_PATH)
model = joblib.load(MODEL_PATH)

# Streamlit App
st.title('Tuberculosis Prediction Using MALDI-TOF MS Data')

# Upload file section
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

if uploaded_file is not None:
    # Process the uploaded file
    with open("uploaded_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    raw_features = process_single_file("uploaded_file.txt")
    
    # Apply transformations
    scaled_features = scaler.transform(raw_features)
    selected_features = rfe_selector.transform(scaled_features)

    # Make prediction
    prediction = model.predict(selected_features)
    prediction_prob = model.predict_proba(selected_features)[:, 1]

    # Display results
    st.subheader(f"Prediction Results for {uploaded_file.name}:")
    st.write(f"**Class:** {'TB' if prediction[0] == 1 else 'Non-TB'}")
    st.write(f"**Probability:** {prediction_prob[0]:.4f}")
    st.write(f"**Confidence:** {'High' if prediction_prob[0] > 0.9 else 'Medium' if prediction_prob[0] > 0.7 else 'Low'}")

    # Optional: Display feature values
    feature_names = [f'{metric}_{peak}' for peak in TB_PEAKS for metric in ['Intensity', 'FWHM', 'Area']]
    selected_feature_names = [feature_names[i] for i in rfe_selector.support_]
    
    st.subheader("Selected Feature Values:")
    feature_values = {name: value for name, value in zip(selected_feature_names, selected_features[0])}
    
    for name, value in feature_values.items():
        st.write(f"{name}: {value:.2f}")
