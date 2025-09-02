#!/usr/bin/env python3
"""
Stratified Glucose Response Streamlit App

A web application for visualizing glucose response patterns stratified by diabetic status.
This app shows the clear glucose spikes that were previously hidden in averaged data.

Run with: streamlit run stratified_glucose_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="Stratified Glucose Response Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def classify_diabetic_status(a1c):
    """Classify diabetic status based on A1c levels."""
    if a1c < 5.7:
        return "Normal"
    elif a1c <= 6.4:
        return "Pre-diabetic" 
    else:
        return "Type2Diabetic"

@st.cache_data
def load_and_process_data():
    """Load and process CGMacros data."""
    
    with st.spinner("Loading CGMacros data..."):
        data_all_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
        
        if not os.path.exists("CGMacros"):
            st.error("CGMacros directory not found. Please ensure the data is in the correct location.")
            return None
        
        # Process each subject directory
        processed_subjects = 0
        for sub_dir in sorted(os.listdir("CGMacros")):
            if sub_dir[:8] != "CGMacros":
                continue
                
            csv_path = os.path.join("CGMacros", sub_dir, sub_dir + '.csv')
            if not os.path.exists(csv_path):
                continue
                
            try:
                data = pd.read_csv(csv_path)
                data_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
                
                # Extract breakfast meals
                breakfast_mask = (data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast")
                
                for index in data[breakfast_mask].index:
                    data_meal = {}
                    data_meal["sub"] = sub_dir[-3:]
                    
                    # Extract glucose readings (every 15 minutes for 2 hours)
                    glucose_readings = data["Libre GL"][index:index+135:15].to_list()
                    if len(glucose_readings) < 9:
                        continue
                        
                    data_meal["Libre GL"] = glucose_readings
                    data_meal["Carb"] = data["Carbs"][index]
                    data_meal["Protein"] = data["Protein"][index] 
                    data_meal["Fat"] = data["Fat"][index]
                    data_meal["Fiber"] = data["Fiber"][index]
                    data_meal["Calories"] = data["Calories"][index] if "Calories" in data.columns else 0
                    
                    data_sub = pd.concat([data_sub, pd.DataFrame([data_meal])], ignore_index=True)
                
                if len(data_sub) > 0:
                    data_all_sub = pd.concat([data_all_sub, data_sub], ignore_index=True)
                    processed_subjects += 1
                    
            except Exception as e:
                st.warning(f"Error processing {sub_dir}: {e}")
                continue
        
        # Load bio data for A1c classification
        if not os.path.exists("CGMacros/bio.csv"):
            st.error("Bio data (CGMacros/bio.csv) not found.")
            return None
            
        bio_df = pd.read_csv("CGMacros/bio.csv")
        bio_df.columns = bio_df.columns.str.strip()
        bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
        
        # Add baseline glucose (first reading) 
        baseline_glucose = []
        for i in range(len(data_all_sub)):
            baseline_glucose.append(data_all_sub["Libre GL"].iloc[i][0])
        data_all_sub["Baseline_Libre"] = baseline_glucose
        
        # Merge with bio data
        diabetic_status_list = []
        a1c_list = []
        
        for i, sub_id in enumerate(data_all_sub["sub"]):
            # Convert subject ID to match bio data
            bio_subject_id = int(sub_id)
            bio_row = bio_df[bio_df['subject'] == bio_subject_id]
            
            if len(bio_row) > 0:
                diabetic_status_list.append(bio_row.iloc[0]['diabetic_status'])
                a1c_list.append(bio_row.iloc[0]['A1c PDL (Lab)'])
            else:
                diabetic_status_list.append('Unknown')
                a1c_list.append(np.nan)
        
        data_all_sub['diabetic_status'] = diabetic_status_list
        data_all_sub['A1c'] = a1c_list
        
        # Remove unknown status entries
        data_all_sub = data_all_sub[data_all_sub['diabetic_status'] != 'Unknown']
    
    return data_all_sub

def create_interactive_glucose_curves(data_all_sub):
    """Create interactive glucose response curves using Plotly."""
    
    # Time points (0, 15, 30, 45, 60, 75, 90, 105, 120 minutes)
    time_points = [i * 15 for i in range(9)]
    
    # Colors for each status
    colors = {
        'Normal': '#2E86AB',          # Blue
        'Pre-diabetic': '#A23B72',   # Purple
        'Type2Diabetic': '#F18F01'   # Orange
    }
    
    status_order = ['Normal', 'Pre-diabetic', 'Type2Diabetic']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Mean Glucose Response with Confidence Intervals',
            'Glucose Excursion from Baseline', 
            'Individual Response Curves (Sample)',
            'Peak Glucose Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Mean curves with confidence intervals
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        # Calculate mean and std for each time point
        glucose_arrays = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                glucose_arrays.append(glucose_curve)
        
        if glucose_arrays:
            glucose_matrix = np.array(glucose_arrays)
            mean_glucose = np.mean(glucose_matrix, axis=0)
            std_glucose = np.std(glucose_matrix, axis=0)
            
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=time_points, y=mean_glucose,
                    mode='lines+markers',
                    name=f'{status} (n={len(glucose_arrays)})',
                    line=dict(color=colors[status], width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=time_points + time_points[::-1],
                    y=list(mean_glucose + std_glucose) + list((mean_glucose - std_glucose)[::-1]),
                    fill='toself',
                    fillcolor=colors[status],
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{status} ±1 SD'
                ),
                row=1, col=1
            )
    
    # Plot 2: Glucose excursion from baseline
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        excursion_arrays = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                baseline = glucose_curve[0]
                excursion = [g - baseline for g in glucose_curve]
                excursion_arrays.append(excursion)
        
        if excursion_arrays:
            excursion_matrix = np.array(excursion_arrays)
            mean_excursion = np.mean(excursion_matrix, axis=0)
            std_excursion = np.std(excursion_matrix, axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=time_points, y=mean_excursion,
                    mode='lines+markers',
                    name=f'{status} Excursion',
                    line=dict(color=colors[status], width=3),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Confidence interval for excursion
            fig.add_trace(
                go.Scatter(
                    x=time_points + time_points[::-1],
                    y=list(mean_excursion + std_excursion) + list((mean_excursion - std_excursion)[::-1]),
                    fill='toself',
                    fillcolor=colors[status],
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Plot 3: Sample individual curves
    n_samples = 10  # Show 10 random curves per status
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        # Sample random curves
        sample_data = status_data.sample(min(n_samples, len(status_data)))
        
        for i, (_, row) in enumerate(sample_data.iterrows()):
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                fig.add_trace(
                    go.Scatter(
                        x=time_points, y=glucose_curve,
                        mode='lines',
                        name=f'{status}' if i == 0 else None,
                        line=dict(color=colors[status], width=1),
                        opacity=0.6,
                        showlegend=i == 0,
                        legendgroup=f'{status}_individual'
                    ),
                    row=2, col=1
                )
    
    # Plot 4: Peak glucose box plot
    peak_data = {}
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        peaks = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                peaks.append(max(glucose_curve))
        
        peak_data[status] = peaks
    
    for status, peaks in peak_data.items():
        fig.add_trace(
            go.Box(
                y=peaks,
                name=f'{status}',
                marker_color=colors[status],
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Stratified Glucose Response Analysis",
        title_x=0.5,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (minutes)", row=1, col=1)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Glucose Excursion (mg/dL)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=2, col=1)
    
    fig.update_xaxes(title_text="Diabetic Status", row=2, col=2)
    fig.update_yaxes(title_text="Peak Glucose (mg/dL)", row=2, col=2)
    
    # Add horizontal line at zero for excursion plot
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
    
    return fig

def display_summary_statistics(data_all_sub):
    """Display summary statistics in organized columns."""
    
    st.subheader("📊 Summary Statistics by Diabetic Status")
    
    time_points = [i * 15 for i in range(9)]
    
    # Create columns for each status
    col1, col2, col3 = st.columns(3)
    
    for i, (col, status) in enumerate(zip([col1, col2, col3], ['Normal', 'Pre-diabetic', 'Type2Diabetic'])):
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        with col:
            st.markdown(f"### {status}")
            
            status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
            
            # Basic info
            st.metric("Number of meals", len(status_data))
            st.metric("Number of subjects", len(status_data['sub'].unique()))
            st.metric("A1c range", f"{status_data['A1c'].min():.1f} - {status_data['A1c'].max():.1f}%")
            
            # Calculate glucose statistics
            glucose_arrays = []
            baselines = []
            peaks = []
            
            for _, row in status_data.iterrows():
                glucose_curve = row['Libre GL'][:len(time_points)]
                if len(glucose_curve) == len(time_points):
                    glucose_arrays.append(glucose_curve)
                    baselines.append(glucose_curve[0])
                    peaks.append(max(glucose_curve))
            
            if glucose_arrays:
                st.metric("Baseline glucose", f"{np.mean(baselines):.1f} ± {np.std(baselines):.1f} mg/dL")
                st.metric("Peak glucose", f"{np.mean(peaks):.1f} ± {np.std(peaks):.1f} mg/dL")
                
                excursion_mean = np.mean(peaks) - np.mean(baselines)
                st.metric("Peak excursion", f"{excursion_mean:.1f} mg/dL")
                
                # Find time to peak
                time_to_peaks = []
                for glucose_curve in glucose_arrays:
                    peak_idx = np.argmax(glucose_curve)
                    time_to_peaks.append(time_points[peak_idx])
                
                st.metric("Time to peak", f"{np.mean(time_to_peaks):.0f} ± {np.std(time_to_peaks):.0f} min")

def main():
    """Main Streamlit app."""
    
    st.title("📊 Stratified Glucose Response Analysis")
    st.markdown("**Revealing the glucose spikes hidden in averaged data**")
    
    st.markdown("""
    This analysis solves the "flat glucose curve" problem by stratifying participants based on their diabetic status.
    Instead of averaging all participants together (which hides the glucose spikes), we analyze each group separately.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Overview", "📈 Interactive Analysis", "📊 Static Visualization", "ℹ️ About"]
    )
    
    if page == "🏠 Overview":
        st.header("Problem & Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ The Problem")
            st.markdown("""
            - Previous glucose response curves were flat
            - Averaging all participants together obscured glucose spikes
            - Type 2 diabetic responses were diluted by normal responses
            - Class imbalance was suspected but wasn't the real issue
            """)
        
        with col2:
            st.subheader("✅ The Solution")
            st.markdown("""
            - Stratify analysis by diabetic status (Normal, Pre-diabetic, Type2Diabetic)
            - Show separate glucose response curves for each group
            - Reveal the dramatic glucose spikes in diabetic participants
            - Better than upsampling - shows real physiological differences
            """)
        
        st.subheader("🔍 Key Findings Preview")
        st.markdown("""
        - **Normal participants**: Stable glucose responses, minimal spikes
        - **Pre-diabetic participants**: Moderate glucose excursions (~48 mg/dL peak excursion)
        - **Type2Diabetic participants**: Dramatic glucose spikes (~93 mg/dL peak excursion)
        - **Time to peak**: Diabetic participants take longer to reach peak glucose
        """)
    
    elif page == "📈 Interactive Analysis":
        # Load data
        data_all_sub = load_and_process_data()
        
        if data_all_sub is None:
            st.error("Could not load data. Please check the data directory structure.")
            return
        
        if len(data_all_sub) == 0:
            st.error("No valid data found after processing.")
            return
        
        # Display data info
        st.success(f"✅ Loaded {len(data_all_sub)} meal records from {len(data_all_sub['sub'].unique())} subjects")
        
        # Show distribution
        distribution = data_all_sub['diabetic_status'].value_counts()
        st.write("**Distribution by diabetic status:**")
        st.write(distribution)
        
        # Create interactive visualization
        fig = create_interactive_glucose_curves(data_all_sub)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        display_summary_statistics(data_all_sub)
    
    elif page == "📊 Static Visualization":
        st.header("Static Visualization")
        
        # Check if static plot exists
        if os.path.exists("stratified_glucose_responses.png"):
            st.image("stratified_glucose_responses.png", 
                    caption="Stratified Glucose Response Patterns by Diabetic Status",
                    use_container_width=True)
            
            st.markdown("""
            This static visualization was generated by the `stratified_glucose_visualization.py` script.
            It shows four key perspectives on the glucose response data:
            
            1. **Individual Curves**: All glucose response curves colored by diabetic status
            2. **Mean Curves with Confidence Intervals**: Average responses with standard deviation bands  
            3. **Glucose Excursion from Baseline**: How much glucose rises from starting level
            4. **Peak Glucose Distribution**: Box plots showing the range of peak glucose values
            """)
        else:
            st.warning("Static visualization not found. Please run `python stratified_glucose_visualization.py` first.")
            
            if st.button("Generate Static Visualization"):
                with st.spinner("Generating static visualization..."):
                    os.system("python stratified_glucose_visualization.py")
                    st.rerun()
    
    elif page == "ℹ️ About":
        st.header("About This Analysis")
        
        st.markdown("""
        ### Background
        This analysis was developed to address the issue of "flat glucose curves" in the CGMacros dataset.
        The original analysis averaged glucose responses across all participants, which obscured the 
        dramatic glucose spikes present in diabetic participants.
        
        ### Methodology
        - **Data Source**: CGMacros dataset with continuous glucose monitoring data
        - **Approach**: Stratified analysis by diabetic status (based on HbA1c levels)
        - **Classification**: 
          - Normal: HbA1c < 5.7%
          - Pre-diabetic: HbA1c 5.7-6.4%
          - Type2Diabetic: HbA1c > 6.4%
        
        ### Key Insights
        1. **Class imbalance was not the issue** - groups were well balanced
        2. **Averaging was masking the signal** - diabetic glucose spikes were being diluted
        3. **Stratification reveals clear patterns** - each group has distinct response profiles
        4. **No need for upsampling** - real physiological differences are dramatic enough
        
        ### Technical Details
        - Time series: 120 minutes post-meal (15-minute intervals)
        - Focus on breakfast meals for consistency
        - Analysis includes individual curves, means, confidence intervals, and distributions
        
        ### Files
        - `stratified_glucose_visualization.py`: Main analysis script
        - `stratified_glucose_app.py`: This Streamlit app
        - `parse_data.ipynb`: Original analysis notebook (reference)
        """)

if __name__ == "__main__":
    main()