import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import io
import zipfile
from spectral_processor import SpectralProcessor

# Page configuration
st.set_page_config(
    page_title="TB vs NTM Spectral Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize spectral processor
@st.cache_resource
def load_processor():
    return SpectralProcessor()

processor = load_processor()

# Load models
@st.cache_resource
def load_models():
    try:
        # Try to load the models if they exist
        if os.path.exists('models/fed_nsmote_lgbm.pkl'):
            with open('models/fed_nsmote_lgbm.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            model = None
            
        if os.path.exists('models/fed_nsmote_scaler.pkl'):
            with open('models/fed_nsmote_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, scaler = load_models()

# Main title and description
st.title("🔬 TB vs NTM Spectral Analysis Tool")
st.markdown("""
This application performs tuberculosis (TB) vs non-tuberculous mycobacteria (NTM) classification 
using advanced spectral analysis and machine learning. Upload your spectral data files to get 
predictions with detailed analysis and visualization.
""")

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_files = st.file_uploader(
        "Upload spectral data files (.txt)",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload one or more .txt files containing spectral data with m/z and intensity columns"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        st.session_state.uploaded_files = uploaded_files
    
    st.header("⚙️ Processing Options")
    
    baseline_method = st.selectbox(
        "Baseline Correction Method",
        ["als", "polynomial"],
        index=0,
        help="ALS (Asymmetric Least Squares) is recommended for spectral data"
    )
    
    normalization = st.checkbox(
        "Apply Normalization",
        value=True,
        help="Normalize intensity values between 0 and 1"
    )
    
    smoothing = st.checkbox(
        "Apply Gaussian Smoothing",
        value=True,
        help="Apply Gaussian filter to reduce noise"
    )
    
    # TB peaks configuration
    st.header("🎯 Peak Analysis")
    tb_peaks_default = [10660, 10100, 9768, 9813, 7931, 7974]
    
    use_default_peaks = st.checkbox(
        "Use Default TB Peaks",
        value=True,
        help="Use the standard TB peak positions for analysis"
    )
    
    if use_default_peaks:
        tb_peaks = tb_peaks_default
        st.write("Default TB peaks:", tb_peaks)
    else:
        peak_input = st.text_area(
            "Custom Peak Positions",
            value=", ".join(map(str, tb_peaks_default)),
            help="Enter peak positions separated by commas"
        )
        try:
            tb_peaks = [float(x.strip()) for x in peak_input.split(',') if x.strip()]
        except:
            st.error("Invalid peak format. Using default peaks.")
            tb_peaks = tb_peaks_default
    
    # Process button
    process_button = st.button(
        "🚀 Process Data",
        disabled=not uploaded_files,
        use_container_width=True
    )

# Main content area
if not uploaded_files:
    st.info("👆 Please upload spectral data files using the sidebar to begin analysis.")
    
    # Show example of expected data format
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your .txt files should contain spectral data with two columns:
    - **Column 1**: m/z values (mass-to-charge ratio)
    - **Column 2**: Intensity values
    
    The columns can be separated by spaces, tabs, or other whitespace.
    """)
    
    # Example data structure
    example_data = pd.DataFrame({
        'm/z': [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        'Intensity': [1500, 1520, 1480, 1600, 1550]
    })
    st.dataframe(example_data, use_container_width=True)

elif process_button or st.session_state.processed_data is not None:
    
    if process_button:
        # Process the uploaded files
        with st.spinner("Processing spectral data..."):
            try:
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Read file content
                    content = uploaded_file.read().decode('utf-8')
                    
                    # Process the spectral data
                    result = processor.process_single_file(
                        content, 
                        uploaded_file.name,
                        tb_peaks,
                        baseline_method=baseline_method,
                        apply_normalization=normalization,
                        apply_smoothing=smoothing
                    )
                    
                    if result is not None:
                        results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if results:
                    st.session_state.processed_data = results
                    
                    # Make predictions if models are available
                    if model is not None and scaler is not None:
                        predictions = processor.make_predictions(results, model, scaler)
                        st.session_state.predictions = predictions
                    else:
                        st.warning("⚠️ Prediction models not found. Only spectral analysis will be performed.")
                        st.session_state.predictions = None
                    
                    st.success(f"✅ Successfully processed {len(results)} files!")
                else:
                    st.error("❌ No files could be processed. Please check your data format.")
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
    
    # Display results if available
    if st.session_state.processed_data is not None:
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Spectral Visualization", 
            "🎯 Peak Analysis", 
            "🤖 Predictions",
            "📥 Download Results"
        ])
        
        with tab1:
            st.subheader("📊 Processing Overview")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Files Processed", len(st.session_state.processed_data))
            
            with col2:
                total_peaks = sum(len(data['peak_features']) for data in st.session_state.processed_data)
                st.metric("Total Peaks Analyzed", total_peaks)
            
            with col3:
                if st.session_state.predictions is not None:
                    tb_count = sum(1 for p in st.session_state.predictions if p['prediction'] == 1)
                    st.metric("TB Predictions", tb_count)
                else:
                    st.metric("TB Predictions", "N/A")
            
            with col4:
                if st.session_state.predictions is not None:
                    ntm_count = sum(1 for p in st.session_state.predictions if p['prediction'] == 0)
                    st.metric("NTM Predictions", ntm_count)
                else:
                    st.metric("NTM Predictions", "N/A")
            
            # File list with basic info
            st.subheader("📋 Processed Files")
            
            overview_data = []
            for i, data in enumerate(st.session_state.processed_data):
                row = {
                    'Filename': data['filename'],
                    'Data Points': len(data['raw_data']),
                    'TIC': f"{data['global_features']['TIC']:.2e}",
                    'Peak Count': data['global_features']['peak_count']
                }
                
                if st.session_state.predictions is not None:
                    pred = st.session_state.predictions[i]
                    row['Prediction'] = "TB" if pred['prediction'] == 1 else "NTM"
                    row['Confidence'] = f"{pred['probability']:.1%}"
                
                overview_data.append(row)
            
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Interactive Spectral Visualization")
            
            # File selector
            file_options = [data['filename'] for data in st.session_state.processed_data]
            selected_file = st.selectbox("Select file to visualize:", file_options)
            
            # Find selected data
            selected_data = None
            for data in st.session_state.processed_data:
                if data['filename'] == selected_file:
                    selected_data = data
                    break
            
            if selected_data is not None:
                # Visualization options
                col1, col2 = st.columns(2)
                with col1:
                    show_raw = st.checkbox("Show Raw Spectrum", value=True)
                with col2:
                    show_processed = st.checkbox("Show Processed Spectrum", value=True)
                
                # Create subplot
                fig = make_subplots(
                    rows=2 if (show_raw and show_processed) else 1,
                    cols=1,
                    subplot_titles=['Raw Spectrum', 'Processed Spectrum'] if (show_raw and show_processed) else None,
                    vertical_spacing=0.1
                )
                
                row_idx = 1
                
                if show_raw:
                    # Raw spectrum
                    raw_data = selected_data['raw_data']
                    fig.add_trace(
                        go.Scatter(
                            x=raw_data['m/z'],
                            y=raw_data['Intensity'],
                            mode='lines',
                            name='Raw Intensity',
                            line=dict(color='blue', width=1)
                        ),
                        row=row_idx, col=1
                    )
                    
                    if show_processed:
                        row_idx = 2
                
                if show_processed:
                    # Processed spectrum
                    processed_data = selected_data['processed_data']
                    fig.add_trace(
                        go.Scatter(
                            x=processed_data['m/z'],
                            y=processed_data['Corrected'],
                            mode='lines',
                            name='Processed Intensity',
                            line=dict(color='red', width=1)
                        ),
                        row=row_idx, col=1
                    )
                    
                    # Add peak markers
                    peak_features = selected_data['peak_features']
                    for i, (peak_mz, features) in enumerate(peak_features.items()):
                        if features['Present'] == 1:
                            # Find closest m/z point
                            mz_array = processed_data['m/z'].values
                            closest_idx = np.argmin(np.abs(mz_array - peak_mz))
                            closest_mz = mz_array[closest_idx]
                            intensity = processed_data['Corrected'].iloc[closest_idx]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[closest_mz],
                                    y=[intensity],
                                    mode='markers+text',
                                    name=f'Peak {i+1}',
                                    marker=dict(size=10, color='green'),
                                    text=[f'{i+1}'],
                                    textposition='top center',
                                    hovertemplate=f'Peak {i+1}<br>m/z: {peak_mz}<br>Intensity: {features["Intensity"]:.3f}<br>FWHM: {features["FWHM"]:.3f}<extra></extra>'
                                ),
                                row=row_idx, col=1
                            )
                
                # Update layout
                fig.update_layout(
                    title=f"Spectral Analysis: {selected_file}",
                    height=600 if (show_raw and show_processed) else 400,
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="m/z")
                fig.update_yaxes(title_text="Intensity")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Zoom controls
                st.subheader("🔍 Region Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    mz_min = st.number_input(
                        "m/z Min",
                        value=float(selected_data['raw_data']['m/z'].min()),
                        min_value=float(selected_data['raw_data']['m/z'].min()),
                        max_value=float(selected_data['raw_data']['m/z'].max())
                    )
                
                with col2:
                    mz_max = st.number_input(
                        "m/z Max",
                        value=float(selected_data['raw_data']['m/z'].max()),
                        min_value=float(selected_data['raw_data']['m/z'].min()),
                        max_value=float(selected_data['raw_data']['m/z'].max())
                    )
                
                if mz_min < mz_max:
                    # Create zoomed view
                    if show_processed:
                        zoom_data = selected_data['processed_data']
                        zoom_mask = (zoom_data['m/z'] >= mz_min) & (zoom_data['m/z'] <= mz_max)
                        
                        if zoom_mask.any():
                            zoom_fig = go.Figure()
                            zoom_fig.add_trace(
                                go.Scatter(
                                    x=zoom_data.loc[zoom_mask, 'm/z'],
                                    y=zoom_data.loc[zoom_mask, 'Corrected'],
                                    mode='lines',
                                    name='Zoomed Region',
                                    line=dict(color='purple', width=2)
                                )
                            )
                            
                            zoom_fig.update_layout(
                                title=f"Zoomed Region: {mz_min:.1f} - {mz_max:.1f} m/z",
                                xaxis_title="m/z",
                                yaxis_title="Intensity",
                                height=300
                            )
                            
                            st.plotly_chart(zoom_fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Peak Analysis Details")
            
            # File selector
            file_options = [data['filename'] for data in st.session_state.processed_data]
            selected_file_peaks = st.selectbox("Select file for peak analysis:", file_options, key="peak_file")
            
            # Find selected data
            selected_data_peaks = None
            for data in st.session_state.processed_data:
                if data['filename'] == selected_file_peaks:
                    selected_data_peaks = data
                    break
            
            if selected_data_peaks is not None:
                # Peak summary table
                peak_data = []
                for i, (peak_mz, features) in enumerate(selected_data_peaks['peak_features'].items()):
                    peak_data.append({
                        'Peak #': i + 1,
                        'Target m/z': peak_mz,
                        'Present': '✅' if features['Present'] else '❌',
                        'Intensity': f"{features['Intensity']:.4f}",
                        'FWHM': f"{features['FWHM']:.4f}",
                        'Area': f"{features['Area']:.4f}"
                    })
                
                peak_df = pd.DataFrame(peak_data)
                st.dataframe(peak_df, use_container_width=True)
                
                # Global features
                st.subheader("📊 Global Spectral Features")
                global_features = selected_data_peaks['global_features']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Ion Current (TIC)", f"{global_features['TIC']:.2e}")
                    st.metric("Mean Intensity", f"{global_features['mean_intensity']:.4f}")
                
                with col2:
                    st.metric("Std Intensity", f"{global_features['std_intensity']:.4f}")
                    st.metric("Peak Count", global_features['peak_count'])
                
                with col3:
                    st.metric("Skewness", f"{global_features['skew_intensity']:.4f}")
                    st.metric("Kurtosis", f"{global_features['kurt_intensity']:.4f}")
        
        with tab4:
            st.subheader("🤖 Machine Learning Predictions")
            
            if st.session_state.predictions is not None:
                # Prediction summary
                st.subheader("📊 Prediction Summary")
                
                prediction_data = []
                for i, (data, pred) in enumerate(zip(st.session_state.processed_data, st.session_state.predictions)):
                    prediction_data.append({
                        'Filename': data['filename'],
                        'Prediction': "TB" if pred['prediction'] == 1 else "NTM",
                        'Confidence': f"{pred['probability']:.1%}",
                        'TB Probability': f"{pred['tb_probability']:.1%}",
                        'NTM Probability': f"{pred['ntm_probability']:.1%}"
                    })
                
                pred_df = pd.DataFrame(prediction_data)
                st.dataframe(pred_df, use_container_width=True)
                
                # Confidence distribution
                st.subheader("📈 Confidence Distribution")
                
                confidences = [pred['probability'] for pred in st.session_state.predictions]
                predictions = ["TB" if pred['prediction'] == 1 else "NTM" for pred in st.session_state.predictions]
                
                fig = px.histogram(
                    x=confidences,
                    color=predictions,
                    title="Prediction Confidence Distribution",
                    labels={'x': 'Confidence', 'count': 'Count'},
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual prediction details
                st.subheader("🔍 Individual Prediction Details")
                
                selected_pred_file = st.selectbox(
                    "Select file for detailed prediction:", 
                    [data['filename'] for data in st.session_state.processed_data],
                    key="pred_detail_file"
                )
                
                # Find selected prediction
                selected_pred_idx = None
                for i, data in enumerate(st.session_state.processed_data):
                    if data['filename'] == selected_pred_file:
                        selected_pred_idx = i
                        break
                
                if selected_pred_idx is not None:
                    pred = st.session_state.predictions[selected_pred_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Prediction",
                            "TB" if pred['prediction'] == 1 else "NTM",
                            delta=f"{pred['probability']:.1%} confidence"
                        )
                    
                    with col2:
                        # Confidence gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = pred['probability'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Prediction Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Feature importance (if available)
                    st.subheader("🎯 Feature Contributions")
                    st.info("Feature importance analysis requires additional model information.")
            
            else:
                st.warning("⚠️ Prediction models are not available. Please ensure the model files are present in the 'models' directory.")
                st.markdown("""
                **Required model files:**
                - `models/fed_nsmote_lgbm.pkl` - Trained LightGBM model
                - `models/fed_nsmote_scaler.pkl` - Feature scaler
                """)
        
        with tab5:
            st.subheader("📥 Download Results")
            
            # Create download packages
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Processed Data")
                
                if st.button("Generate Data Package", use_container_width=True):
                    with st.spinner("Preparing download package..."):
                        # Create ZIP file in memory
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            
                            # Add individual processed files
                            for data in st.session_state.processed_data:
                                # Raw data
                                raw_csv = data['raw_data'].to_csv(index=False)
                                zip_file.writestr(f"raw_{data['filename']}.csv", raw_csv)
                                
                                # Processed data
                                processed_csv = data['processed_data'].to_csv(index=False)
                                zip_file.writestr(f"processed_{data['filename']}.csv", processed_csv)
                            
                            # Summary report
                            summary_data = []
                            for i, data in enumerate(st.session_state.processed_data):
                                row = {
                                    'filename': data['filename'],
                                    **data['global_features']
                                }
                                
                                # Add peak features
                                for peak_mz, features in data['peak_features'].items():
                                    for feature_name, value in features.items():
                                        row[f'peak_{peak_mz}_{feature_name}'] = value
                                
                                # Add predictions if available
                                if st.session_state.predictions is not None:
                                    pred = st.session_state.predictions[i]
                                    row.update({
                                        'prediction': pred['prediction'],
                                        'prediction_label': "TB" if pred['prediction'] == 1 else "NTM",
                                        'confidence': pred['probability'],
                                        'tb_probability': pred['tb_probability'],
                                        'ntm_probability': pred['ntm_probability']
                                    })
                                
                                summary_data.append(row)
                            
                            summary_df = pd.DataFrame(summary_data)
                            summary_csv = summary_df.to_csv(index=False)
                            zip_file.writestr("summary_report.csv", summary_csv)
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Data Package",
                            data=zip_buffer.getvalue(),
                            file_name="spectral_analysis_results.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            
            with col2:
                st.subheader("📋 Analysis Report")
                
                if st.button("Generate Report", use_container_width=True):
                    with st.spinner("Generating analysis report..."):
                        # Create comprehensive report
                        report_lines = []
                        report_lines.append("TB vs NTM Spectral Analysis Report")
                        report_lines.append("=" * 50)
                        report_lines.append("")
                        
                        # Processing summary
                        report_lines.append("PROCESSING SUMMARY")
                        report_lines.append("-" * 20)
                        report_lines.append(f"Files processed: {len(st.session_state.processed_data)}")
                        report_lines.append(f"Baseline correction: {baseline_method}")
                        report_lines.append(f"Normalization: {'Yes' if normalization else 'No'}")
                        report_lines.append(f"Smoothing: {'Yes' if smoothing else 'No'}")
                        report_lines.append(f"Peak positions: {tb_peaks}")
                        report_lines.append("")
                        
                        # Individual file results
                        for i, data in enumerate(st.session_state.processed_data):
                            report_lines.append(f"FILE: {data['filename']}")
                            report_lines.append("-" * 30)
                            
                            # Global features
                            gf = data['global_features']
                            report_lines.append(f"Total Ion Current: {gf['TIC']:.2e}")
                            report_lines.append(f"Mean Intensity: {gf['mean_intensity']:.4f}")
                            report_lines.append(f"Peak Count: {gf['peak_count']}")
                            
                            # Peak analysis
                            report_lines.append("\nPeak Analysis:")
                            for j, (peak_mz, features) in enumerate(data['peak_features'].items()):
                                status = "Present" if features['Present'] else "Absent"
                                report_lines.append(f"  Peak {j+1} ({peak_mz} m/z): {status}")
                                if features['Present']:
                                    report_lines.append(f"    Intensity: {features['Intensity']:.4f}")
                                    report_lines.append(f"    FWHM: {features['FWHM']:.4f}")
                            
                            # Prediction
                            if st.session_state.predictions is not None:
                                pred = st.session_state.predictions[i]
                                pred_label = "TB" if pred['prediction'] == 1 else "NTM"
                                report_lines.append(f"\nPrediction: {pred_label} ({pred['probability']:.1%} confidence)")
                            
                            report_lines.append("")
                        
                        report_text = "\n".join(report_lines)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=report_text,
                            file_name="spectral_analysis_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>TB vs NTM Spectral Analysis Tool | Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)
