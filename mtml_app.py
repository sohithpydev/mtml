import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os

# Define processing functions
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

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
        popt, _ = curve_fit(gaussian, x, y,
                          p0=[y_smooth.max(), x[y_smooth.argmax()], 1],
                          bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
        amp, cen, wid = popt
        fwhm = 2.355 * wid
        area = np.trapz(y, x)
    except:
        amp, fwhm, area = y.max() if len(y) > 0 else 0, 0, 0

    return {'Intensity': amp, 'FWHM': fwhm, 'Area': area}

def visualize_real_tolerance(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    main_peak = df.loc[df['Corrected'].idxmax(), 'm/z']
    tol = dynamic_tolerance(df, main_peak)
    ax.plot(df['m/z'], df['Corrected'], 'k-', label='Full Spectrum')
    ax.axvspan(main_peak - tol, main_peak + tol, color='orange', alpha=0.3, label=f'Tolerance: ±{tol:.1f}')
    ax.set_title(f"Example\nm/z: {main_peak:.1f}")
    ax.set_xlabel('m/z')
    ax.set_ylabel('Corrected Intensity')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# Load pre-trained model
model_path = 'extratrees_rfe.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = joblib.load(model_path)

tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]

st.title("Tuberculosis Detection from Mass Spectrometry Data")
st.write("Upload your mass spectrometry data (TXT with 'intensity m/z' format)")

uploaded_file = st.file_uploader("Choose a TXT file", type="txt")

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, names=['Intensity', 'm/z'])
        processed_data = baseline_correction(raw_data)
        features = []
        for peak in tb_peaks:
            peak_features = get_peak_features(processed_data, peak)
            features.extend([peak_features['Intensity'], peak_features['FWHM'], peak_features['Area']])
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        result = "TB Positive" if prediction[0] == 1 else "TB Negative"
        st.subheader("Prediction Result")
        st.write(f"**Result**: {result}")
        st.subheader("Processed Mass Spectrometry Data")
        visualize_real_tolerance(processed_data)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
