import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Define processing functions
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid ** 2)

def baseline_correction(data, poly_order=3):
    try:
        x = data['m/z']
        y = data['Intensity']
        coeffs = np.polyfit(x, y, poly_order)
        baseline = np.polyval(coeffs, x)
        corrected = y - baseline
        corrected[corrected < 0] = 0
        data['Corrected'] = corrected
        return data
    except Exception as e:
        st.error(f"Baseline correction failed: {str(e)}")
        return None

def dynamic_tolerance(data, peak_mz, window=50, min_tol=0.5, max_tol=20):
    try:
        region = data[(data['m/z'] > peak_mz - window) & (data['m/z'] < peak_mz + window)]
        if region.empty:
            return min_tol
        tol = 2 * region['m/z'].std()
        return np.clip(tol, min_tol, max_tol)
    except:
        return min_tol

def get_peak_features(data, peak_mz):
    try:
        tol = dynamic_tolerance(data, peak_mz)
        peak_data = data[(data['m/z'] > peak_mz - tol) & (data['m/z'] < peak_mz + tol)].copy()
        
        if len(peak_data) < 3:
            return {'Intensity': 0, 'FWHM': 0, 'Area': 0}
        
        x = peak_data['m/z'].values
        y = peak_data['Corrected'].values
        
        y_smooth = gaussian_filter1d(y, sigma=1)
        if len(y_smooth) == 0:
            return {'Intensity': 0, 'FWHM': 0, 'Area': 0}

        try:
            popt, _ = curve_fit(gaussian, x, y, 
                                p0=[y_smooth.max(), x[y_smooth.argmax()], 1],
                                bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
            amp, cen, wid = popt
            fwhm = 2.355 * wid
            area = np.trapz(y, x)
        except RuntimeError:
            amp = y.max() if len(y) > 0 else 0
            fwhm = 0
            area = np.trapz(y, x) if len(y) > 0 else 0
        
        return {'Intensity': amp, 'FWHM': fwhm, 'Area': area}
    
    except Exception as e:
        st.error(f"Error processing peak {peak_mz}: {str(e)}")
        return {'Intensity': 0, 'FWHM': 0, 'Area': 0}

# Streamlit app interface
st.title('TB Detection from Mass Spectrometry Data')
st.write('Upload trained model and mass spec data for analysis')

# File upload section
uploaded_model = st.file_uploader("Upload trained model (pkl)", type="pkl")
uploaded_file = st.file_uploader("Upload mass spec data (txt)", type="txt")

if uploaded_model and uploaded_file:
    try:
        # Load model and scaler from tuple
        model, scaler = joblib.load(uploaded_model)
        tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]
        
        if st.button('Analyze Data'):
            with st.spinner('Processing...'):
                # Load and process data
                raw_data = pd.read_csv(uploaded_file, sep='\s+', 
                                     header=None, names=["m/z", "Intensity"])
                
                # Data validation
                if raw_data.empty or (raw_data['Intensity'] < 0).any():
                    st.error("Invalid data format or negative intensities")
                    st.stop()

                processed_data = baseline_correction(raw_data)
                if processed_data is None:
                    st.stop()

                # Feature extraction
                features = []
                for peak in tb_peaks:
                    peak_features = get_peak_features(processed_data, peak)
                    features.extend([peak_features['Intensity'],
                                    peak_features['FWHM'],
                                    peak_features['Area'])

                # Validate feature count
                expected_features = len(tb_peaks) * 3
                if len(features) != expected_features:
                    st.error(f"Feature mismatch: Expected {expected_features}, got {len(features)}")
                    st.stop()

                # Transform features
                scaled_features = scaler.transform([features])
                
                # Make prediction
                prediction = model.predict(scaled_features)
                proba = model.predict_proba(scaled_features)

                # Display results
                st.success("Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", 
                            "TB Positive" if prediction[0] == 1 else "TB Negative",
                            f"{proba[0][1]*100:.1f}% confidence" if prediction[0] == 1 else f"{proba[0][0]*100:.1f}% confidence")
                
                with col2:
                    st.write("Probability Distribution:")
                    st.dataframe(pd.DataFrame({
                        'Class': ['Non-TB
