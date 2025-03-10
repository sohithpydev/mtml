import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import streamlit as st

# Gaussian function for peak fitting
def gaussian(x, amp, cen, wid):
    """Gaussian function."""
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

# Baseline correction function
def baseline_correction(data, poly_order=3):
    """Perform baseline correction."""
    x = data['m/z']
    y = data['Intensity']
    coeffs = np.polyfit(x, y, poly_order)
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    corrected[corrected < 0] = 0  # Clip negative values
    data['Corrected'] = corrected
    return data

# Dynamic tolerance calculation for peak region extraction
def dynamic_tolerance(data, peak_mz, window=50, min_tol=0.5, max_tol=20):
    """Determine peak-specific tolerance dynamically."""
    region = data[(data['m/z'] > peak_mz - window) & (data['m/z'] < peak_mz + window)]
    if region.empty:
        return min_tol
    tol = 2 * region['m/z'].std()
    return np.clip(tol, min_tol, max_tol)

# Visualization of peak fitting and tolerance
def visualize_tolerance(file_path):
    """Visualize peak fitting, FWHM, and tolerance for the given file."""
    df = pd.read_csv(file_path, sep='\s+', header=None, names=["m/z", "Intensity"])
    df = baseline_correction(df)

    # Find the main peak (highest intensity after baseline correction)
    main_peak = df.loc[df['Corrected'].idxmax(), 'm/z']
    tol = dynamic_tolerance(df, main_peak)

    # Plot the full spectrum with tolerance region highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['m/z'], df['Corrected'], 'k-', label='Full Spectrum')
    ax.axvspan(main_peak - tol, main_peak + tol, color='orange', alpha=0.3, label=f'Tolerance: ±{tol:.1f}')
    
    # Zoom in on the peak region
    peak_df = df[(df['m/z'] > main_peak - tol) & (df['m/z'] < main_peak + tol)].copy()
    x = peak_df['m/z'].values
    y = peak_df['Corrected'].values

    try:
        y_smooth = gaussian_filter1d(y, sigma=1)
        popt, _ = curve_fit(gaussian, x, y, p0=[y_smooth.max(), x.mean(), 1], bounds=([0, x.min(), 0], [np.inf, x.max(), np.inf]))
        amp, cen, wid = popt
        fwhm = 2.355 * wid
        area = np.trapz(y, x)
        fit_success = True
    except:
        fwhm, area = 0, 0
        fit_success = False

    # Plot zoom-in with fitted Gaussian curve
    ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])  # inset position [x, y, width, height]
    ax_inset.plot(x, y, 'bo', markersize=3, label="Raw Data")

    if fit_success:
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = gaussian(x_fit, *popt)
        ax_inset.plot(x_fit, y_fit, 'r-', label="Gaussian Fit")
        ax_inset.fill_between(x_fit, y_fit, alpha=0.3, color="red", label="Area Under Curve")

        # Mark FWHM
        half_max = amp / 2
        fwhm_range = [cen - (fwhm / 2), cen + (fwhm / 2)]
        ax_inset.hlines(half_max, fwhm_range[0], fwhm_range[1], colors="blue", linestyles="dashed", label="FWHM")

    ax_inset.set_title("Peak Zoom-In")
    ax_inset.grid(alpha=0.3)
    ax_inset.legend()

    # Main plot decorations
    ax.set_title(f"m/z: {main_peak:.1f}")
    ax.set_xlabel('m/z')
    ax.set_ylabel('Corrected Intensity')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Streamlit App
st.title('MALDI-TOF MS Data Peak Visualization')

# File upload section
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("uploaded_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Visualize tolerance and peak fitting
    visualize_tolerance("uploaded_file.txt")
