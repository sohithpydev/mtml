import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import joblib

# Configuration
TB_PEAKS = [10660, 10100, 9768, 9813, 7931, 7974]  # Same as training peaks
output_file = "processed_dataset_with_dynamic_tolerance_2.csv"

# Baseline correction function
def baseline_correction(data, poly_order=3):
    x = data['m/z']
    y = data['Intensity']
    coeffs = np.polyfit(x, y, poly_order)
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    corrected[corrected < 0] = 0  # Clip negative values
    data['Corrected'] = corrected
    return data

# Dynamic tolerance for peak region extraction
def dynamic_tolerance(data, peak_mz, window=50, min_tol=0.5, max_tol=20):
    region = data[(data['m/z'] > peak_mz - window) & (data['m/z'] < peak_mz + window)]
    if region.empty:
        return min_tol
    tol = 2 * region['m/z'].std()
    return np.clip(tol, min_tol, max_tol)

# Gaussian fitting for peak analysis
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

# Feature extraction for each peak
def get_peak_features(data, peak_mz):
    tol = dynamic_tolerance(data, peak_mz)
    peak_data = data[(data['m/z'] > peak_mz - tol) & (data['m/z'] < peak_mz + tol)]

    if len(peak_data) < 3:
        return {'Intensity': 0, 'FWHM': 0, 'Area': 0}

    x = peak_data['m/z'].values
    y = peak_data['Corrected'].values
    y_smooth = gaussian_filter1d(y, sigma=1)

    try:
        popt, _ = curve_fit(gaussian, x, y,
                          p0=[y_smooth.max(), x[y_smooth.argmax()], 1],
                          bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
        amp, cen, wid = popt
        fwhm = 2.355 * wid
        area = np.trapz(y, x)
    except:
        amp = y.max()
        fwhm = 0
        area = 0

    return {'Intensity': amp, 'FWHM': fwhm, 'Area': area}

# Process uploaded file
def process_single_file(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None, names=["m/z", "Intensity"])
    df = baseline_correction(df)

    features = []
    for peak in TB_PEAKS:
        peak_features = get_peak_features(df, peak)
        features.extend([peak_features['Intensity'], peak_features['FWHM'], peak_features['Area']])

    return df, np.array(features).reshape(1, -1)

# Streamlit App
st.title('Tuberculosis Prediction Using MALDI-TOF MS Data')

# Upload file section
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

if uploaded_file is not None:
    # Save uploaded file
    with open("uploaded_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the uploaded file
    df, raw_features = process_single_file("uploaded_file.txt")
    
    # Plot the raw spectrum (Intensity vs. m/z)
    st.subheader('Raw Spectrum (Intensity vs. m/z)')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['m/z'], df['Intensity'], label="Raw Spectrum")
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title("MALDI-TOF Mass Spectrum")
    ax.legend()
    st.pyplot(fig)
    
    # Apply transformations and make predictions (using pre-trained model)
    # Assuming pre-trained scaler, RFE selector, and model are loaded
    scaler = joblib.load("scaler.pkl")
    rfe_selector = joblib.load("rfe_selector.pkl")
    model = joblib.load("extra_tree_model.pkl")

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

    # Feature values (Optional: show selected features)
    feature_names = [f'Intensity_{peak}' for peak in TB_PEAKS]
    selected_feature_names = [feature_names[i] for i in rfe_selector.support_]
    st.subheader("Selected Feature Values:")
    for name, value in zip(selected_feature_names, selected_features[0]):
        st.write(f"{name}: {value:.2f}")
    
    # Display prediction probability chart
    st.subheader("Prediction Probability Chart:")
    st.bar_chart([prediction_prob[0], 1 - prediction_prob[0]], x=["TB", "Non-TB"], use_container_width=True)

