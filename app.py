import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import joblib

# Configuration
TB_PEAKS = [10660, 10100, 9768, 9813, 7931, 7974]  # Peaks used in training
output_file = "processed_dataset_with_dynamic_tolerance.csv"

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

# Gaussian fitting function
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

    return {'Intensity': amp, 'FWHM': fwhm, 'Area': area, 'Center': cen, 'Width': wid}

# Peak Fitting Plot
st.subheader("Gaussian Fit on Selected Peak")
peak_mz = st.selectbox("Select m/z value for peak fitting", TB_PEAKS)
tol = dynamic_tolerance(df, peak_mz)
peak_data = df[(df['m/z'] > peak_mz - tol) & (df['m/z'] < peak_mz + tol)]
x = peak_data['m/z'].values
y = peak_data['Corrected'].values
y_smooth = gaussian_filter1d(y, sigma=1)

try:
    popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x[np.argmax(y)], 1])
    amp, cen, wid = popt
    fwhm = 2.355 * wid
    area = np.trapz(gaussian(x, *popt), x)

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = gaussian(x_fit, *popt)

    fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
    ax_fit.scatter(x, y, color='blue', label="Raw Data", s=10)
    ax_fit.plot(x, y_smooth, 'g-', label="Smoothed Data")
    ax_fit.plot(x_fit, y_fit, 'r-', label="Gaussian Fit")
    ax_fit.fill_between(x_fit, y_fit, alpha=0.3, color='red', label="Shaded Area (Integral)")
    ax_fit.axvline(cen - fwhm / 2, color='purple', linestyle='--', label="FWHM Bounds")
    ax_fit.axvline(cen + fwhm / 2, color='purple', linestyle='--')
    ax_fit.legend()
    st.pyplot(fig_fit)

    st.subheader("Extracted Feature Values")
    st.write(f"**Intensity:** {amp:.2f}")
    st.write(f"**FWHM:** {fwhm:.2f}")
    st.write(f"**Area:** {area:.2f}")
    st.write(f"**Center:** {cen:.2f}")
    st.write(f"**Width:** {wid:.2f}")
except:
    st.warning(f"Gaussian fit failed for peak at m/z = {peak_mz:.1f}")

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
    
    # Load pre-trained model components
    scaler = joblib.load("scaler.pkl")
    rfe_selector = joblib.load("rfe_selector.pkl")
    model = joblib.load("extra_tree_model.pkl")

    # Define feature names
    feature_names = []
    for peak in TB_PEAKS:
        feature_names.extend([f'Intensity_{peak}', f'FWHM_{peak}', f'Area_{peak}'])

    # Apply transformations
    scaled_features = scaler.transform(raw_features)
    selected_feature_indices = np.where(rfe_selector.support_)[0]
    selected_features = rfe_selector.transform(scaled_features)
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    # Make prediction
    prediction = model.predict(selected_features)
    prediction_prob = model.predict_proba(selected_features)[:, 1]

    # Display results
    st.subheader(f"Prediction Results for {uploaded_file.name}:")
    st.write(f"**Class:** {'TB' if prediction[0] == 1 else 'Non-TB'}")
    st.write(f"**Probability:** {prediction_prob[0]:.4f}")
    st.write(f"**Confidence:** {'High' if prediction_prob[0] > 0.9 else 'Medium' if prediction_prob[0] > 0.7 else 'Low'}")

    # Display Selected Feature Values
    st.subheader("Selected Feature Values:")
    for name, value in zip(selected_feature_names, selected_features[0]):
        st.write(f"**{name}:** {value:.2f}")

    # Display prediction probability chart
    st.subheader("Prediction Probability Chart:")
    prediction_data = pd.DataFrame({
        'Class': ['TB', 'Non-TB'],
        'Probability': [prediction_prob[0], 1 - prediction_prob[0]]
    })
    st.bar_chart(prediction_data.set_index('Class')['Probability'])

    # Zoomed-In Peak View
    st.subheader("Zoomed-In View of the Peak")
    mz_min, mz_max = st.slider("Select m/z range", float(df['m/z'].min()), float(df['m/z'].max()), 
                               (df['m/z'].min(), df['m/z'].max()), step=0.1)
    zoomed_df = df[(df['m/z'] >= mz_min) & (df['m/z'] <= mz_max)]
    fig_zoom, ax_zoom = plt.subplots(figsize=(10, 6))
    ax_zoom.plot(zoomed_df['m/z'], zoomed_df['Corrected'], label="Zoomed Spectrum")
    ax_zoom.set_xlabel("m/z")
    ax_zoom.set_ylabel("Intensity")
    ax_zoom.set_title("Zoomed-In MALDI-TOF Spectrum")
    ax_zoom.legend()
    st.pyplot(fig_zoom)

    # Peak Fitting Plot
    st.subheader("Gaussian Fit on Selected Peak")
    peak_mz = st.selectbox("Select m/z value for peak fitting", TB_PEAKS)
    tol = dynamic_tolerance(df, peak_mz)
    peak_data = df[(df['m/z'] > peak_mz - tol) & (df['m/z'] < peak_mz + tol)]
    x = peak_data['m/z'].values
    y = peak_data['Corrected'].values

    try:
        popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x[np.argmax(y)], 1])
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = gaussian(x_fit, *popt)
        fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
        ax_fit.plot(x, y, 'bo', markersize=3, label="Raw Data")
        ax_fit.plot(x_fit, y_fit, 'r-', label="Gaussian Fit")
        ax_fit.legend()
        st.pyplot(fig_fit)
    except:
        st.warning(f"Gaussian fit failed for peak at m/z = {peak_mz:.1f}")
