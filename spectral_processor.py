import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import io

class SpectralProcessor:
    """
    Spectral data processing class that implements the exact pipeline from the notebook
    for TB vs NTM classification using mass spectrometry data.
    """
    
    def __init__(self):
        self.tb_peaks = [10660, 10100, 9768, 9813, 7931, 7974]
    
    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares baseline correction
        
        Parameters:
        y: intensity values
        lam: smoothness parameter
        p: asymmetry parameter
        niter: number of iterations
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = spsolve(W + lam * D.dot(D.T), w * y)
            w = p * (y > Z) + (1 - p) * (y < Z)
        return y - Z
    
    def normalize_intensity(self, intensity):
        """Normalize intensity values between 0 and 1"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(intensity.reshape(-1, 1)).flatten()
    
    def baseline_correction(self, data, method='als'):
        """
        Baseline correction with polynomial or ALS method
        
        Parameters:
        data: DataFrame with 'm/z' and 'Intensity' columns
        method: 'als' or 'polynomial'
        """
        x = data['m/z'].values
        y = data['Intensity'].values
        
        if method == 'als':
            corrected = self.baseline_als(y)
        else:  # polynomial method
            coeffs = np.polyfit(x, y, 3)
            corrected = y - np.polyval(coeffs, x)
        
        corrected[corrected < 0] = 0
        data = data.copy()
        data['Corrected'] = corrected
        return data
    
    def dynamic_tolerance(self, data, peak_mz, window=50, min_tol=0.5, max_tol=20):
        """Calculate dynamic tolerance based on local m/z variation"""
        region = data[(data['m/z'] > peak_mz - window) & (data['m/z'] < peak_mz + window)]
        if region.empty:
            return min_tol
        tol = 2 * region['m/z'].std()
        return np.clip(tol, min_tol, max_tol)
    
    def gaussian(self, x, amp, cen, wid):
        """Gaussian function for peak fitting"""
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    
    def get_peak_features(self, data, peak_mz):
        """Extract detailed features for a specific peak"""
        tol = self.dynamic_tolerance(data, peak_mz)
        peak_data = data[(data['m/z'] > peak_mz - tol) & (data['m/z'] < peak_mz + tol)]
        
        if len(peak_data) < 3:
            return {'Intensity': 0, 'FWHM': 0, 'Area': 0, 'Present': 0}
        
        x = peak_data['m/z'].values
        y = peak_data['Corrected'].values
        y_smooth = gaussian_filter1d(y, sigma=1)
        
        try:
            popt, _ = curve_fit(self.gaussian, x, y_smooth,
                              p0=[y_smooth.max(), x[y_smooth.argmax()], 1],
                              bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
            amp, cen, wid = popt
            fwhm = 2.355 * wid
            area = trapezoid(y_smooth, x)
            present = 1
        except:
            amp = y_smooth.max()
            fwhm = 0
            area = 0
            present = 1 if amp > 0.1 else 0  # Threshold for presence
        
        return {'Intensity': amp, 'FWHM': fwhm, 'Area': area, 'Present': present}
    
    def extract_global_features(self, data):
        """Extract global spectrum features"""
        y = data['Corrected'].values
        peaks, _ = find_peaks(y, height=0.1)
        intensity_peaks = y[peaks]
        
        # Handle edge cases
        if len(intensity_peaks) == 0:
            intensity_peaks = np.array([0])
        
        features = {
            'TIC': np.sum(y),
            'mean_intensity': np.mean(intensity_peaks),
            'std_intensity': np.std(intensity_peaks),
            'skew_intensity': pd.Series(intensity_peaks).skew() if len(intensity_peaks) > 1 else 0,
            'kurt_intensity': pd.Series(intensity_peaks).kurtosis() if len(intensity_peaks) > 1 else 0,
            'peak_count': len(peaks)
        }
        
        # Handle NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0
                
        return features
    
    def process_single_file(self, file_content, filename, peaks, baseline_method='als', 
                           apply_normalization=True, apply_smoothing=True):
        """
        Process a single spectral data file
        
        Parameters:
        file_content: string content of the file
        filename: name of the file
        peaks: list of peak positions to analyze
        baseline_method: 'als' or 'polynomial'
        apply_normalization: whether to normalize intensities
        apply_smoothing: whether to apply Gaussian smoothing
        """
        try:
            # Parse the file content
            lines = file_content.strip().split('\n')
            data_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            data_lines.append([mz, intensity])
                        except ValueError:
                            continue
            
            if not data_lines:
                raise ValueError("No valid data found in file")
            
            # Create DataFrame
            df = pd.DataFrame(data_lines, columns=['m/z', 'Intensity'])
            
            # Store raw data
            raw_data = df.copy()
            
            # Baseline correction
            df = self.baseline_correction(df, method=baseline_method)
            
            if 'Corrected' not in df.columns:
                raise ValueError("Baseline correction failed")
            
            # Normalization
            if apply_normalization:
                df['Corrected'] = self.normalize_intensity(df['Corrected'].values)
            
            # Smoothing
            if apply_smoothing:
                df['Corrected'] = gaussian_filter1d(df['Corrected'].values, sigma=1)
            
            # Extract global features
            global_features = self.extract_global_features(df)
            
            # Extract peak-specific features
            peak_features = {}
            for peak in peaks:
                features = self.get_peak_features(df, peak)
                peak_features[peak] = features
            
            return {
                'filename': filename,
                'raw_data': raw_data,
                'processed_data': df,
                'global_features': global_features,
                'peak_features': peak_features
            }
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return None
    
    def extract_feature_vector(self, processed_data, peaks):
        """
        Extract feature vector for machine learning prediction
        
        Parameters:
        processed_data: dictionary from process_single_file
        peaks: list of peak positions
        
        Returns:
        numpy array of features
        """
        # Global features
        global_features = processed_data['global_features']
        feature_vector = [
            global_features['TIC'],
            global_features['mean_intensity'],
            global_features['std_intensity'],
            global_features['skew_intensity'],
            global_features['kurt_intensity'],
            global_features['peak_count']
        ]
        
        # Peak-specific features
        for peak in peaks:
            if peak in processed_data['peak_features']:
                features = processed_data['peak_features'][peak]
                feature_vector.extend([
                    features['Intensity'],
                    features['FWHM'],
                    features['Area'],
                    features['Present']
                ])
            else:
                # If peak not found, add zeros
                feature_vector.extend([0, 0, 0, 0])
        
        return np.array(feature_vector)
    
    def make_predictions(self, processed_data_list, model, scaler):
        """
        Make predictions on processed data using trained model
        
        Parameters:
        processed_data_list: list of processed data dictionaries
        model: trained LightGBM model
        scaler: fitted StandardScaler or MinMaxScaler
        
        Returns:
        list of prediction dictionaries
        """
        try:
            predictions = []
            
            for processed_data in processed_data_list:
                # Extract feature vector
                feature_vector = self.extract_feature_vector(
                    processed_data, 
                    list(processed_data['peak_features'].keys())
                )
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
                
                # Make prediction
                prediction = model.predict(feature_vector_scaled)[0]
                prediction_proba = model.predict_proba(feature_vector_scaled)[0]
                
                # Calculate confidence as the maximum probability
                confidence = max(prediction_proba)
                
                predictions.append({
                    'filename': processed_data['filename'],
                    'prediction': int(prediction),
                    'probability': confidence,
                    'tb_probability': prediction_proba[1] if len(prediction_proba) > 1 else (confidence if prediction == 1 else 1-confidence),
                    'ntm_probability': prediction_proba[0] if len(prediction_proba) > 1 else (confidence if prediction == 0 else 1-confidence)
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def create_feature_dataframe(self, processed_data_list, peaks, labels=None):
        """
        Create a feature DataFrame for machine learning
        
        Parameters:
        processed_data_list: list of processed data dictionaries
        peaks: list of peak positions
        labels: optional list of labels (0 for NTM, 1 for TB)
        
        Returns:
        pandas DataFrame with features
        """
        data = []
        
        for i, processed_data in enumerate(processed_data_list):
            # Extract feature vector
            feature_vector = self.extract_feature_vector(processed_data, peaks)
            
            # Create row
            row_data = feature_vector.tolist()
            
            # Add label if provided
            if labels is not None:
                row_data.append(labels[i])
            
            # Add filename
            row_data.append(processed_data['filename'])
            
            data.append(row_data)
        
        # Create feature names
        global_feature_names = ['TIC', 'mean_intensity', 'std_intensity', 
                               'skew_intensity', 'kurt_intensity', 'peak_count']
        
        peak_feature_names = []
        for peak in peaks:
            peak_feature_names.extend([
                f'Intensity_{peak}',
                f'FWHM_{peak}',
                f'Area_{peak}',
                f'Present_{peak}'
            ])
        
        feature_names = global_feature_names + peak_feature_names
        
        if labels is not None:
            feature_names.append('Label')
        
        feature_names.append('Filename')
        
        return pd.DataFrame(data, columns=feature_names)
