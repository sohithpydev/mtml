import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

# Define processing functions
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid ** 2))

def baseline_correction(data, poly_order=3):
    x = data['m/z']
    y = data['Intensity']
    coeffs = np.polyfit(x, y, poly_order)
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    corrected[corrected < 0] = 0
    data['Corrected'] = corrected
    return data

def dynamic_tolerance(data, peak_mz, window=50, min_tol=0.5, max_tol=20):
    region = data[(data['m/z'] > peak_mz - window) & (data['m/z'] < peak_mz + window)]
    if region.empty:
        return min_tol
    tol = 2 * region['m/z'].std()
    return np.clip(tol, min_tol, max_tol)

def get_peak_features(data, peak_mz):
    tol = dynamic_tolerance(data, peak_mz)
    peak_data = data[(data['m/z'] > peak_mz - tol) & (data['m/z'] < peak_mz + tol)]
    if len(peak_data) < 3:
        return {'Intensity': 0, 'FWHM': 0, 'Area': 0}
    x = peak_data['m/z'].values
    y = peak_data['Corrected'].values
    y_smooth = gaussian_filter1d(y, sigma=1)
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=[y_smooth.max(), x[y_smooth.argmax()], 1], bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
        amp, cen, wid = popt
        fwhm = 2.355 * wid
        area = np.trapz(y, x)
    except Exception as e:
        st.error(f"Error in curve fitting: {e}")
        amp = y.max() if len(y) > 0 else 0
        fwhm = 0
        area = 0
    return {'Intensity': amp, 'FWHM': fwhm, 'Area': area}

# Streamlit app
st.title('Mass Spectrometry Data Processing and Classification')
st.write('This app processes and classifies mass spectrometry data.')

# Model uploader
uploaded_model = st.file_uploader("Upload the ExtraTrees RFE Model (extratrees_rfe.pkl)", type="pkl")
if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

# Data uploader
uploaded_file = st.file_uploader("Choose a TXT file", type="txt")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep='\s+', header=None, names=["m/z", "Intensity"])
    st.write("Data Preview:")
    st.write(data.head())

    if st.button('Process and Classify Data'):
        st.write("Processing data...")
        processed_data = baseline_correction(data)
        st.write("Processed Data:")
        st.write(processed_data.head())

        # Feature extraction
        tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]
        features = []
        for peak in tb_peaks:
            peak_features = get_peak_features(processed_data, peak)
            features.extend([peak_features['Intensity'], peak_features['FWHM'], peak_features['Area']])
        st.write("Extracted Features:")
        st.write(features)

        # Feature scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform([features])
        joblib.dump(scaler, 'scaler.pkl')
        st.write("Scaled Features:")
        st.write(scaled_features)

        # Classification
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        st.write("Prediction:")
        st.write("TB" if prediction[0] == 1 else "Non-TB")
        st.write("Prediction Probability:")
        st.write(prediction_proba)

        st.success("Data processing, feature extraction, and classification complete!")
