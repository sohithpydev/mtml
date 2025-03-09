import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os

# Define processing functions from the original notebook
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
    except Exception as e:
        st.error(f"Error in curve fitting: {e}")
        amp = y.max() if len(y) > 0 else 0
        fwhm = 0
        area = 0

    return {'Intensity': amp, 'FWHM': fwhm, 'Area': area}

# Load pre-trained model from the backend
model_path = 'extratrees_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = joblib.load(model_path)

tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]

st.title("Tuberculosis Detection from Mass Spectrometry Data")
st.write("Upload your mass spectrometry data (TXT with 'm/z intensity' format)")

uploaded_file = st.file_uploader("Choose a TXT file", type="txt")

if uploaded_file is not None:
    try:
        # Process data
        raw_data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, names=['m/z', 'Intensity'])
        processed_data = baseline_correction(raw_data)

        # Extract features
        features = []
        for peak in tb_peaks:
            peak_features = get_peak_features(processed_data, peak)
            features.extend([peak_features['Intensity'], peak_features['FWHM'], peak_features['Area']])

        # Create feature names
        feature_names = [f'{ft}_{peak}' for peak in tb_peaks for ft in ['Intensity', 'FWHM', 'Area']]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Predict
        prediction = model.predict(features_df)
        proba = model.predict_proba(features_df)

        # Display results
        st.subheader("Prediction Result")
        result = "TB Positive" if prediction[0] == 1 else "TB Negative"
        st.write(f"**Result**: {result} (confidence: {proba[0][1]*100:.1f}%)")

        # Plot
        st.subheader("Processed Mass Spectrometry Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(processed_data['m/z'], processed_data['Corrected'], label='Processed Spectrum')

        # Mark TB peaks
        for peak in tb_peaks:
            ax.axvline(x=peak, color='r', linestyle='--', alpha=0.5)
            ax.text(peak, processed_data['Corrected'].max()*0.9, str(peak),
                    rotation=90, ha='center', va='top')

        ax.set_xlabel("m/z")
        ax.set_ylabel("Corrected Intensity")
        ax.set_title("Mass Spectrum with TB Diagnostic Peaks Marked")
        st.pyplot(fig)

        # Interpretation
        st.subheader("Interpretation")
        st.write("The model analyzed these diagnostic peaks for TB detection:")

        peak_features = []
        for i, peak in enumerate(tb_peaks):
            intensity = features[i*3]
            fwhm = features[i*3+1]
            area = features[i*3+2]
            peak_features.append({
                'Peak': peak,
                'Intensity': intensity,
                'FWHM': fwhm,
                'Area': area
            })

        # Show top contributing peaks
        feature_importances = model.feature_importances_
        top_idx = np.argsort(feature_importances)[::-1][:3]  # Top 3 features
        st.write("**Most influential features**:")

        for idx in top_idx:
            feat_name = feature_names[idx]
            peak = feat_name.split('_')[1]
            ft_type = feat_name.split('_')[0]
            importance = feature_importances[idx]

            st.write(f"- Peak {peak} ({ft_type}): importance {importance:.3f}")

        # Explanation
        st.write("""
        **Interpretation Guide**:
        - Peaks at specific m/z values are associated with TB biomarkers
        - Higher intensity, wider FWHM, and larger area values contribute to TB positive diagnosis
        - The model combines these features using learned patterns from clinical data
        """)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

st.sidebar.markdown("""
**Sample Data Format**:
- TXT with 2 columns: 'm/z' and 'intensity'
- m/z values should include the diagnostic peaks (7931, 7974, 9768, 9813, 10100, 10660)
- Intensity values should be raw instrument measurements
""")
