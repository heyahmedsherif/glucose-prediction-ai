#!/usr/bin/env python3
"""
Interactive Glucose Spike Analyzer

A comprehensive Streamlit app that allows users to:
- Select different spike emphasis methods
- Compare multiple methods side by side
- Interactive controls for customizing analysis
- Real-time visualization updates

Run with: streamlit run interactive_spike_analyzer.py
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
    page_title="Interactive Glucose Spike Analyzer",
    page_icon="🎛️",
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
    
    with st.spinner("🔄 Loading CGMacros data..."):
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

def calculate_spike_curve(glucose_matrix, method, custom_multiplier=1.0):
    """Calculate spike curve based on selected method."""
    
    mean_glucose = np.mean(glucose_matrix, axis=0)
    std_glucose = np.std(glucose_matrix, axis=0)
    
    if method == "mean":
        return mean_glucose, "Mean Response"
    elif method == "upper_ci":
        return mean_glucose + std_glucose, "Mean + 1 SD"
    elif method == "upper_ci_15":
        return mean_glucose + 1.5 * std_glucose, "Mean + 1.5 SD"
    elif method == "upper_ci_2":
        return mean_glucose + 2 * std_glucose, "Mean + 2 SD"
    elif method == "custom_multiplier":
        return mean_glucose + custom_multiplier * std_glucose, f"Mean + {custom_multiplier} SD"
    elif method == "95th_percentile":
        return np.percentile(glucose_matrix, 95, axis=0), "95th Percentile"
    elif method == "90th_percentile":
        return np.percentile(glucose_matrix, 90, axis=0), "90th Percentile"
    elif method == "75th_percentile":
        return np.percentile(glucose_matrix, 75, axis=0), "75th Percentile"
    elif method == "max":
        return np.max(glucose_matrix, axis=0), "Maximum Response"
    else:
        return mean_glucose, "Mean Response"

def create_interactive_spike_visualization(data_all_sub, spike_methods, custom_multiplier=1.0, 
                                         show_individual_curves=False, selected_statuses=None):
    """Create interactive glucose spike visualization with multiple methods."""
    
    # Time points (0, 15, 30, 45, 60, 75, 90, 105, 120 minutes)
    time_points = [i * 15 for i in range(9)]
    
    # Enhanced colors for each status
    colors = {
        'Normal': '#2E86AB',
        'Pre-diabetic': '#A23B72', 
        'Type2Diabetic': '#F18F01'
    }
    
    if selected_statuses is None:
        selected_statuses = ['Normal', 'Pre-diabetic', 'Type2Diabetic']
    
    # Create subplots based on number of methods
    n_methods = len(spike_methods)
    if n_methods == 1:
        rows, cols = 1, 1
    elif n_methods == 2:
        rows, cols = 1, 2
    elif n_methods <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 3, 2
    
    subplot_titles = [f"Method: {method.replace('_', ' ').title()}" for method in spike_methods]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Store spike data for each method
    all_spike_data = {}
    
    for method_idx, method in enumerate(spike_methods):
        row = method_idx // cols + 1
        col = method_idx % cols + 1
        
        spike_data = {}
        
        for status in selected_statuses:
            if status not in data_all_sub['diabetic_status'].values:
                continue
                
            status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
            
            # Calculate glucose arrays
            glucose_arrays = []
            for _, row_data in status_data.iterrows():
                glucose_curve = row_data['Libre GL'][:len(time_points)]
                if len(glucose_curve) == len(time_points):
                    # Check for and handle NaN values
                    if not any(pd.isna(x) for x in glucose_curve):
                        glucose_arrays.append(glucose_curve)
                    else:
                        # Skip curves with NaN values for now - could interpolate in future
                        continue
            
            if glucose_arrays:
                glucose_matrix = np.array(glucose_arrays)
                spike_curve, method_label = calculate_spike_curve(glucose_matrix, method, custom_multiplier)
                
                spike_data[status] = {
                    'curve': spike_curve,
                    'n_samples': len(glucose_arrays),
                    'method_label': method_label
                }
                
                # Plot spike curve
                fig.add_trace(
                    go.Scatter(
                        x=time_points, y=spike_curve,
                        mode='lines+markers',
                        name=f'{status} ({method_label})' if n_methods > 1 else f'{status}',
                        line=dict(color=colors[status], width=3),
                        marker=dict(size=6),
                        showlegend=(method_idx == 0),  # Only show legend for first method
                        legendgroup=status
                    ),
                    row=row, col=col
                )
                
                # Optionally show individual curves
                if show_individual_curves:
                    sample_size = min(10, len(glucose_arrays))
                    sample_indices = np.random.choice(len(glucose_arrays), sample_size, replace=False)
                    
                    for i in sample_indices:
                        fig.add_trace(
                            go.Scatter(
                                x=time_points, y=glucose_arrays[i],
                                mode='lines',
                                line=dict(color=colors[status], width=1),
                                opacity=0.3,
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
        
        all_spike_data[method] = spike_data
    
    # Update layout
    fig.update_layout(
        height=300 * rows + 100,
        title_text="🎛️ Interactive Glucose Spike Analysis",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        font=dict(size=11)
    )
    
    # Update axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text="Time (minutes)", row=i, col=j)
            fig.update_yaxes(title_text="Glucose (mg/dL)", row=i, col=j)
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)', row=i, col=j)
    
    return fig, all_spike_data

def create_method_comparison_chart(all_spike_data, selected_statuses):
    """Create a comparison chart showing peak values across methods."""
    
    comparison_data = []
    
    for method, spike_data in all_spike_data.items():
        for status in selected_statuses:
            if status in spike_data:
                peak_glucose = max(spike_data[status]['curve'])
                comparison_data.append({
                    'Method': method.replace('_', ' ').title(),
                    'Status': status,
                    'Peak_Glucose': peak_glucose,
                    'Method_Label': spike_data[status]['method_label']
                })
    
    if not comparison_data:
        return None
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    fig = px.bar(
        df_comparison, 
        x='Method', 
        y='Peak_Glucose', 
        color='Status',
        barmode='group',
        title="Peak Glucose Comparison Across Methods",
        labels={'Peak_Glucose': 'Peak Glucose (mg/dL)', 'Method': 'Spike Emphasis Method'},
        color_discrete_map={
            'Normal': '#2E86AB',
            'Pre-diabetic': '#A23B72',
            'Type2Diabetic': '#F18F01'
        }
    )
    
    fig.update_layout(height=400, font=dict(size=12))
    
    return fig

def display_spike_metrics(all_spike_data, selected_statuses):
    """Display detailed spike metrics for each method."""
    
    st.subheader("📊 Spike Metrics Comparison")
    
    # Create tabs for each method
    if len(all_spike_data) > 1:
        tabs = st.tabs([method.replace('_', ' ').title() for method in all_spike_data.keys()])
        
        for tab_idx, (method, spike_data) in enumerate(all_spike_data.items()):
            with tabs[tab_idx]:
                display_method_metrics(spike_data, selected_statuses, method)
    else:
        method, spike_data = next(iter(all_spike_data.items()))
        display_method_metrics(spike_data, selected_statuses, method)

def display_method_metrics(spike_data, selected_statuses, method):
    """Display metrics for a single method."""
    
    time_points = [i * 15 for i in range(9)]
    cols = st.columns(len(selected_statuses))
    
    for col_idx, status in enumerate(selected_statuses):
        if status not in spike_data:
            continue
            
        with cols[col_idx]:
            st.markdown(f"### {status}")
            
            curve = spike_data[status]['curve']
            n_samples = spike_data[status]['n_samples']
            method_label = spike_data[status]['method_label']
            
            # Basic metrics
            baseline = curve[0]
            peak_glucose = max(curve)
            peak_excursion = peak_glucose - baseline
            peak_time = time_points[np.argmax(curve)]
            
            st.metric("Sample Size", n_samples)
            st.metric("Method", method_label)
            st.metric("Peak Glucose", f"{peak_glucose:.1f} mg/dL")
            st.metric("Peak Excursion", f"{peak_excursion:.1f} mg/dL")
            st.metric("Time to Peak", f"{peak_time} min")
            
            # Calculate area under curve (spike intensity)
            spike_excursion_curve = [max(0, g - baseline) for g in curve]
            spike_intensity = np.trapz(spike_excursion_curve, time_points)
            st.metric("Spike Intensity", f"{spike_intensity:.0f} AUC")

def main():
    """Main Streamlit app."""
    
    st.title("🎛️ Interactive Glucose Spike Analyzer")
    st.markdown("**Complete control over glucose spike visualization methods**")
    
    # Load data
    data_all_sub = load_and_process_data()
    
    if data_all_sub is None or len(data_all_sub) == 0:
        st.error("Could not load data. Please check the data directory structure.")
        return
    
    st.success(f"✅ Loaded {len(data_all_sub)} meal records from {len(data_all_sub['sub'].unique())} subjects")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Spike Analysis Controls")
    
    # Method selection
    st.sidebar.subheader("📈 Spike Emphasis Methods")
    
    method_options = {
        "mean": "Mean Response (Original)",
        "upper_ci": "Mean + 1 SD (Upper CI)",
        "upper_ci_15": "Mean + 1.5 SD",
        "upper_ci_2": "Mean + 2 SD",
        "custom_multiplier": "Custom Multiplier",
        "75th_percentile": "75th Percentile",
        "90th_percentile": "90th Percentile", 
        "95th_percentile": "95th Percentile",
        "max": "Maximum Response"
    }
    
    selected_methods = []
    for method_key, method_name in method_options.items():
        if st.sidebar.checkbox(method_name, 
                              value=(method_key in ["mean", "upper_ci"]),  # Default to mean and upper_ci
                              key=f"method_{method_key}"):
            selected_methods.append(method_key)
    
    if not selected_methods:
        st.sidebar.warning("Please select at least one spike method.")
        return
    
    # Custom multiplier setting
    custom_multiplier = 1.0
    if "custom_multiplier" in selected_methods:
        custom_multiplier = st.sidebar.slider(
            "Custom SD Multiplier", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.1,
            help="Multiply standard deviation by this value"
        )
    
    # Status selection
    st.sidebar.subheader("👥 Diabetic Status Groups")
    all_statuses = ['Normal', 'Pre-diabetic', 'Type2Diabetic']
    selected_statuses = []
    for status in all_statuses:
        if st.sidebar.checkbox(status, value=True, key=f"status_{status}"):
            selected_statuses.append(status)
    
    if not selected_statuses:
        st.sidebar.warning("Please select at least one diabetic status group.")
        return
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization Options")
    show_individual_curves = st.sidebar.checkbox("Show Individual Curves (Sample)", 
                                                value=False,
                                                help="Shows sample individual glucose curves for context")
    
    show_comparison_chart = st.sidebar.checkbox("Show Method Comparison", 
                                               value=len(selected_methods) > 1,
                                               help="Shows peak glucose comparison across methods")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("📊 Data Overview")
        distribution = data_all_sub['diabetic_status'].value_counts()
        for status in selected_statuses:
            if status in distribution.index:
                count = distribution[status]
                pct = (count / len(data_all_sub)) * 100
                st.metric(status, f"{count} meals", delta=f"{pct:.1f}%")
    
    with col1:
        # Generate visualization
        with st.spinner("🎨 Creating interactive visualization..."):
            fig, all_spike_data = create_interactive_spike_visualization(
                data_all_sub, 
                selected_methods, 
                custom_multiplier,
                show_individual_curves, 
                selected_statuses
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Method comparison chart
    if show_comparison_chart and len(selected_methods) > 1:
        st.subheader("⚖️ Method Comparison")
        comparison_fig = create_method_comparison_chart(all_spike_data, selected_statuses)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Detailed metrics
    display_spike_metrics(all_spike_data, selected_statuses)
    
    # Method descriptions
    with st.expander("ℹ️ Method Descriptions", expanded=False):
        st.markdown("""
        ### Spike Emphasis Methods
        
        **📊 Statistical Methods:**
        - **Mean Response**: Traditional average (may appear flat) ← Compare this with enhanced methods!
        - **Mean + 1 SD**: Upper confidence interval, clinically relevant
        - **Mean + 1.5 SD**: Enhanced spike emphasis
        - **Mean + 2 SD**: High spike emphasis (captures extreme responses)
        - **Custom Multiplier**: User-defined standard deviation multiplier
        
        **📈 Percentile Methods:**
        - **75th Percentile**: Response level that 75% of participants stay below
        - **90th Percentile**: Response level that 90% of participants stay below
        - **95th Percentile**: Response level that 95% of participants stay below
        - **Maximum Response**: Highest recorded response at each time point
        
        ### Why Use Spike Emphasis?
        - **Clinical Relevance**: Peak glucose levels are more predictive of complications
        - **Individual Variation**: Shows what patients actually experience, not just averages
        - **Visual Impact**: Makes glucose spikes clearly visible for interpretation
        - **Risk Assessment**: Better represents metabolic stress during meals
        """)
    
    # Export options
    st.subheader("💾 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Current Data"):
            # Create export DataFrame
            export_data = []
            time_points = [i * 15 for i in range(9)]
            
            for method, spike_data in all_spike_data.items():
                for status in selected_statuses:
                    if status in spike_data:
                        curve = spike_data[status]['curve']
                        for t, glucose in zip(time_points, curve):
                            export_data.append({
                                'Method': method,
                                'Status': status,
                                'Time_Minutes': t,
                                'Glucose_mgdL': glucose
                            })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"glucose_spike_analysis_{'-'.join(selected_methods)}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Report"):
            st.markdown("### Analysis Report")
            
            for method, spike_data in all_spike_data.items():
                st.markdown(f"**{method.replace('_', ' ').title()} Method:**")
                for status in selected_statuses:
                    if status in spike_data:
                        curve = spike_data[status]['curve']
                        peak = max(curve)
                        st.write(f"- {status}: Peak {peak:.1f} mg/dL")

if __name__ == "__main__":
    main()