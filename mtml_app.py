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

def visualize_real_tolerance(df):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find dominant peak near center
    main_peak = df.loc[df['Corrected'].idxmax(), 'm/z']
    tol = dynamic_tolerance(df, main_peak)

    # Full spectrum plot
    ax.plot(df['m/z'], df['Corrected'], 'k-', label='Full Spectrum')
    ax.axvspan(main_peak - tol, main_peak + tol,
               color='orange', alpha=0.3,
               label=f'Tolerance: ±{tol:.1f}')

    # Peak region analysis
    peak_df = df[(df['m/z'] > main_peak - tol) &
                 (df['m/z'] < main_peak + tol)].copy()
    x = peak_df['m/z'].values
    y = peak_df['Corrected'].values

    # Gaussian fit
    try:
        y_smooth = gaussian_filter1d(y, sigma=1)
        popt, _ = curve_fit(gaussian, x, y,
                            p0=[y_smooth.max(), x.mean(), 1],
                            bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
        fwhm = 2.355 * popt[2]
        area = np.trapz(y, x)
        fit_success = True
    except:
        fwhm, area = 0, 0
        fit_success = False

    # Add text annotations on left side
    text_x = 0.05  # 5% from left edge
    text_y = 0.85  # 85% from bottom
    text = f"FWHM: {fwhm:.1f}\nArea: {area:.1f}"
    ax.text(text_x, text_y, text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))

    # Add inset for peak zoom on right side
    ax_inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    ax_inset.plot(x, y, 'bo', markersize=3)
    if fit_success:
        ax_inset.plot(x, gaussian(x, *popt), 'r-')
    ax_inset.set_title("Peak Region Zoom")
    ax_inset.grid(alpha=0.3)

    # Main plot decorations
    ax.set_title(f"Example\nm/z: {main_peak:.1f}")
    ax.set_xlabel('m/z')
    ax.set_ylabel('Corrected Intensity')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

# Load pre-trained models
model_path = 'extratrees_tb_model.pkl'
rfe_model_path = 'rfe_model.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(rfe_model_path) or not os.path.exists(scaler_path):
    st.error(f"Model file not found: {model_path} or {rfe_model_path} or {scaler_path}")
else:
    model = joblib.load(model_path)
    rfe = joblib.load(rfe_model_path)
    scaler = joblib.load(scaler_path)

tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]

st.title("Tuberculosis Detection from Mass Spectrometry Data")
st.write("Upload your mass spectrometry data (TXT with 'intensity m/z' format)")

uploaded_file = st.file_uploader("Choose a TXT file", type="txt")

if uploaded_file is not None:
    try:
        # Process data
        raw_data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, names=['Intensity', 'm/z'])
        processed_data = baseline_correction(raw_data)
        
        # Log processed data for debugging
        st.write("Processed Data:")
        st.dataframe(processed_data)

        # Extract features
        features = []
        for peak in tb_peaks:
            peak_features = get_peak_features(processed_data, peak)
            features.extend([peak_features['Intensity'], peak_features['FWHM'], peak_features['Area']])

        # Log features for debugging
        st.write("Extracted Features:")
        st.write(features)

        # Create feature names
        feature_names = [f'{ft}_{peak}' for peak in tb_peaks for ft in ['Intensity', 'FWHM', 'Area']]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Standardize the features
        features_scaled = scaler.transform(features_df)
        
        # Log scaled features for debugging
        st.write("Scaled Features:")
        st.write(features_scaled)

        # Reduce features using RFE
        features_reduced = rfe.transform(features_scaled)
        
        # Log reduced features for debugging
        st.write("Reduced Features:")
        st.write(features_reduced)

        # Predict
        prediction = model.predict(features_reduced)

        # Display results
        st.subheader("Prediction Result")
        result = "TB Positive" if prediction[0] == 1 else "TB Negative"
        st.write(f"**Result**: {result}")

        # Plot
        st.subheader("Processed Mass Spectrometry Data")
        visualize_real_tolerance(processed_data)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

st.sidebar.markdown("""
**Sample Data Format**:
- TXT with 2 columns: 'intensity' and 'm/z'
- m/z values should include the diagnostic peaks (7931, 7974, 9768, 9813, 10100, 10660)
- Intensity values should be raw instrument measurements
""")
