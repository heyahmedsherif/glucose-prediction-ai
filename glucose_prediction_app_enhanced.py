#!/usr/bin/env python3
"""
Enhanced Glucose Prediction Streamlit App

A web application for predicting blood glucose responses after meals using
the enhanced CGMacros glucose prediction models with diabetic status integration.

Features:
- Diabetic status selection (Normal, Pre-diabetic, Type2Diabetic)
- Automatic baseline glucose prediction based on patient profile
- Enhanced glucose response predictions
- Interactive visualizations and insights

Run with: streamlit run glucose_prediction_app_enhanced.py
"""

import streamlit as st
import json
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple
import os
import pandas as pd
import pickle
import sys

# Add the current directory to Python path so we can import our custom classes
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the enhanced pipeline class
try:
    from train_enhanced_glucose_models import EnhancedGlucosePredictionPipeline
except ImportError:
    st.error("Could not import EnhancedGlucosePredictionPipeline. Please ensure train_enhanced_glucose_models.py is in the same directory.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Enhanced Glucose Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_enhanced_models():
    """Load the enhanced trained models and metadata."""
    
    models_dir = "glucose_prediction_models"
    
    if not os.path.exists(models_dir):
        st.error(f"❌ Models directory '{models_dir}' not found. Please ensure the trained models are available.")
        st.stop()
    
    try:
        # First try to load the complete pipeline
        pipeline_path = os.path.join(models_dir, "glucose_prediction_pipeline.pkl")
        if os.path.exists(pipeline_path):
            try:
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                
                # Load metadata
                metadata_path = os.path.join(models_dir, "model_metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check model version
                if metadata.get('pipeline_version') == '2.0.0':
                    st.success("✅ Using Enhanced Models v2.0.0 with Diabetic Status Integration")
                else:
                    st.warning("⚠️ Using older model version")
                
                return pipeline, metadata
                
            except Exception as pickle_error:
                st.warning(f"⚠️ Could not load pickled pipeline: {pickle_error}")
                st.info("🔄 Loading models individually...")
        
        # Fallback: Load models individually and create pipeline
        pipeline = EnhancedGlucosePredictionPipeline()
        
        # Load metadata
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load individual model components
        target_variables = ['glucose_30min', 'glucose_60min', 'glucose_90min', 
                           'glucose_120min', 'glucose_180min']
        
        for target_var in target_variables:
            # Load model
            model_file = os.path.join(models_dir, f"{target_var}_model.joblib")
            if os.path.exists(model_file):
                pipeline.models[target_var] = joblib.load(model_file)
            
            # Load scaler
            scaler_file = os.path.join(models_dir, f"{target_var}_scaler.joblib")
            if os.path.exists(scaler_file):
                pipeline.scalers[target_var] = joblib.load(scaler_file)
        
        # Load baseline predictor if available
        baseline_model_file = os.path.join(models_dir, "baseline_model.joblib")
        baseline_scaler_file = os.path.join(models_dir, "baseline_scaler.joblib")
        
        if os.path.exists(baseline_model_file) and os.path.exists(baseline_scaler_file):
            pipeline.baseline_predictors = {
                'model': joblib.load(baseline_model_file),
                'scaler': joblib.load(baseline_scaler_file),
                'features': ['diabetic_status_encoded', 'age', 'bmi', 'a1c', 'fasting_glucose'],
                'performance': metadata.get('baseline_predictor_info', {})
            }
        
        # Set feature sets and model info from metadata
        pipeline.feature_sets = metadata.get('feature_sets', {})
        pipeline.model_info = metadata.get('model_info', {})
        
        # Check model version
        if metadata.get('pipeline_version') == '2.0.0':
            st.success("✅ Using Enhanced Models v2.0.0 with Diabetic Status Integration")
        else:
            st.warning("⚠️ Using older model version")
        
        st.info("🔧 Models loaded individually and pipeline reconstructed")
        
        return pipeline, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error("Please ensure the enhanced models have been trained by running: python train_enhanced_glucose_models.py")
        st.stop()

@st.cache_data
def load_baseline_lookup():
    """Load the diabetic status baseline lookup table."""
    try:
        lookup_df = pd.read_csv('diabetic_status_baseline_lookup.csv')
        return lookup_df
    except:
        # Fallback data if file not found
        return pd.DataFrame({
            'diabetic_status': ['Normal', 'Pre-diabetic', 'Type2Diabetic'],
            'mean_baseline': [78.3, 95.8, 130.1],
            'std_baseline': [6.1, 15.2, 28.4]
        })

def get_diabetic_status_info(status: str) -> Dict[str, Any]:
    """Get information about diabetic status."""
    info = {
        'Normal': {
            'description': 'Normal glucose metabolism (HbA1c < 5.7%)',
            'risk_level': 'Low',
            'color': '#2E8B57',
            'icon': '✅',
            'baseline_range': '70-95 mg/dL',
            'recommendations': [
                'Maintain healthy diet and exercise',
                'Regular monitoring recommended',
                'Focus on preventing future complications'
            ]
        },
        'Pre-diabetic': {
            'description': 'Impaired glucose tolerance (HbA1c 5.7-6.4%)',
            'risk_level': 'Medium',
            'color': '#FF8C00',
            'icon': '⚠️',
            'baseline_range': '85-125 mg/dL',
            'recommendations': [
                'Lifestyle modifications strongly recommended',
                'Regular glucose monitoring important',
                'Consider dietary consultation'
            ]
        },
        'Type2Diabetic': {
            'description': 'Diabetes mellitus type 2 (HbA1c ≥ 6.5%)',
            'risk_level': 'High',
            'color': '#DC143C',
            'icon': '🔴',
            'baseline_range': '95-200 mg/dL',
            'recommendations': [
                'Continuous glucose monitoring recommended',
                'Medical supervision required',
                'Medication management may be necessary'
            ]
        }
    }
    return info.get(status, info['Normal'])

def create_glucose_curve_plot(predictions: Dict[str, float], diabetic_status: str) -> go.Figure:
    """Create an interactive glucose response curve."""
    
    # Extract time points and values
    time_points = [0]  # Start with baseline
    glucose_values = [predictions['baseline']]
    
    for time_var in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
        minutes = int(time_var.replace('glucose_', '').replace('min', ''))
        time_points.append(minutes)
        glucose_values.append(predictions[time_var])
    
    # Get status info for styling
    status_info = get_diabetic_status_info(diabetic_status)
    
    # Create the plot
    fig = go.Figure()
    
    # Add the glucose curve
    fig.add_trace(go.Scatter(
        x=time_points,
        y=glucose_values,
        mode='lines+markers',
        name='Glucose Response',
        line=dict(color=status_info['color'], width=3),
        marker=dict(size=8, color=status_info['color']),
        hovertemplate='<b>Time:</b> %{x} min<br><b>Glucose:</b> %{y:.1f} mg/dL<extra></extra>'
    ))
    
    # Add reference ranges
    fig.add_hline(y=70, line_dash="dash", line_color="blue", 
                  annotation_text="Hypoglycemic threshold", annotation_position="bottom right")
    fig.add_hline(y=140, line_dash="dash", line_color="orange",
                  annotation_text="Post-meal target", annotation_position="top right")
    fig.add_hline(y=180, line_dash="dash", line_color="red",
                  annotation_text="High glucose alert", annotation_position="top right")
    
    # Styling
    fig.update_layout(
        title=f'Predicted Glucose Response - {diabetic_status} Patient',
        xaxis_title='Time (minutes)',
        yaxis_title='Glucose Level (mg/dL)',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    # Set y-axis range
    fig.update_layout(yaxis=dict(range=[50, max(200, max(glucose_values) + 20)]))
    
    return fig

def create_comparison_plot(predictions: Dict[str, float]) -> go.Figure:
    """Create a comparison plot showing glucose at different time points."""
    
    time_points = []
    glucose_values = []
    colors = []
    
    for time_var in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
        minutes = int(time_var.replace('glucose_', '').replace('min', ''))
        glucose = predictions[time_var]
        
        time_points.append(f'{minutes} min')
        glucose_values.append(glucose)
        
        # Color coding based on glucose level
        if glucose < 70:
            colors.append('#1f77b4')  # Blue for low
        elif glucose < 140:
            colors.append('#2ca02c')  # Green for normal
        elif glucose < 180:
            colors.append('#ff7f0e')  # Orange for elevated
        else:
            colors.append('#d62728')  # Red for high
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_points,
        y=glucose_values,
        marker_color=colors,
        text=[f'{v:.1f}' for v in glucose_values],
        textposition='outside',
        hovertemplate='<b>Time:</b> %{x}<br><b>Glucose:</b> %{y:.1f} mg/dL<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=140, line_dash="dash", line_color="orange",
                  annotation_text="Target: <140 mg/dL", annotation_position="top right")
    
    fig.update_layout(
        title='Glucose Levels at Different Time Points',
        xaxis_title='Time After Meal',
        yaxis_title='Glucose Level (mg/dL)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Main application function."""
    
    # Load models and data
    pipeline, metadata = load_enhanced_models()
    baseline_lookup = load_baseline_lookup()
    
    # App header
    st.title("🩺 Enhanced Glucose Prediction App")
    st.markdown("### Personalized Blood Glucose Response Predictions with Diabetic Status Integration")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Patient Profile & Meal Information")
    
    # Diabetic Status Selection
    st.sidebar.subheader("🔍 Diabetic Status")
    diabetic_status = st.sidebar.selectbox(
        "Select diabetic status:",
        options=['Normal', 'Pre-diabetic', 'Type2Diabetic'],
        help="Based on HbA1c levels: Normal (<5.7%), Pre-diabetic (5.7-6.4%), Type2Diabetic (≥6.5%)"
    )
    
    # Show status information
    status_info = get_diabetic_status_info(diabetic_status)
    st.sidebar.markdown(f"""
    **{status_info['icon']} {diabetic_status}**
    
    {status_info['description']}
    
    **Risk Level:** {status_info['risk_level']}
    
    **Typical Baseline:** {status_info['baseline_range']}
    """)
    
    # Demographics
    st.sidebar.subheader("👤 Demographics")
    age = st.sidebar.slider("Age (years)", 18, 80, 40, help="Patient age in years")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 0 if gender == "Female" else 1
    
    height_ft = st.sidebar.slider("Height (feet)", 4, 7, 5)
    height_in = st.sidebar.slider("Height (inches)", 0, 11, 6)
    height_total_in = height_ft * 12 + height_in
    
    weight_lbs = st.sidebar.slider("Weight (lbs)", 80, 400, 150)
    
    # Calculate BMI
    bmi = (weight_lbs / (height_total_in ** 2)) * 703
    st.sidebar.metric("Calculated BMI", f"{bmi:.1f}")
    
    # Optional clinical parameters
    with st.sidebar.expander("🧪 Clinical Parameters (Optional)"):
        a1c = st.number_input("HbA1c (%)", 4.0, 15.0, 
                             value=5.4 if diabetic_status == 'Normal' else 
                                   6.0 if diabetic_status == 'Pre-diabetic' else 7.5,
                             step=0.1)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 60, 300, 
                                        value=90 if diabetic_status == 'Normal' else
                                              110 if diabetic_status == 'Pre-diabetic' else 150,
                                        step=5)
        fasting_insulin = st.number_input("Fasting Insulin (μU/mL)", 1.0, 50.0, 8.0, step=0.5)
    
    # Meal composition
    st.sidebar.subheader("🍽️ Meal Composition")
    
    # Predefined meal options
    meal_presets = {
        "Custom": {"carbs": 45, "protein": 25, "fat": 12, "fiber": 3, "calories": 350},
        
        # Breakfast Options
        "Light Breakfast": {"carbs": 30, "protein": 15, "fat": 8, "fiber": 4, "calories": 240},
        "Standard Breakfast": {"carbs": 45, "protein": 20, "fat": 12, "fiber": 5, "calories": 350},
        "Heavy Breakfast": {"carbs": 65, "protein": 25, "fat": 18, "fiber": 6, "calories": 490},
        
        # Lunch Options
        "Light Lunch": {"carbs": 35, "protein": 25, "fat": 10, "fiber": 5, "calories": 310},
        "Standard Lunch": {"carbs": 55, "protein": 30, "fat": 15, "fiber": 6, "calories": 450},
        "Heavy Lunch": {"carbs": 75, "protein": 35, "fat": 22, "fiber": 8, "calories": 590},
        
        # Dinner Options
        "Light Dinner": {"carbs": 40, "protein": 30, "fat": 12, "fiber": 6, "calories": 370},
        "Standard Dinner": {"carbs": 65, "protein": 35, "fat": 18, "fiber": 7, "calories": 530},
        "Heavy Dinner": {"carbs": 85, "protein": 45, "fat": 25, "fiber": 9, "calories": 700},
        
        # Snack Options
        "Light Snack": {"carbs": 15, "protein": 5, "fat": 3, "fiber": 2, "calories": 105},
        "Standard Snack": {"carbs": 25, "protein": 8, "fat": 6, "fiber": 3, "calories": 180},
        "Heavy Snack": {"carbs": 40, "protein": 12, "fat": 10, "fiber": 4, "calories": 290},
        
        # Specific Food Categories
        "Small Sandwich": {"carbs": 30, "protein": 15, "fat": 8, "fiber": 3, "calories": 240},
        "Large Sandwich": {"carbs": 55, "protein": 25, "fat": 15, "fiber": 5, "calories": 430},
        "Small Pasta": {"carbs": 45, "protein": 12, "fat": 5, "fiber": 3, "calories": 270},
        "Large Pasta": {"carbs": 80, "protein": 20, "fat": 10, "fiber": 5, "calories": 480},
        "Small Salad": {"carbs": 15, "protein": 20, "fat": 12, "fiber": 8, "calories": 230},
        "Large Salad": {"carbs": 25, "protein": 35, "fat": 18, "fiber": 12, "calories": 370},
        "Small Rice Bowl": {"carbs": 50, "protein": 18, "fat": 8, "fiber": 2, "calories": 340},
        "Large Rice Bowl": {"carbs": 85, "protein": 30, "fat": 15, "fiber": 4, "calories": 580},
        "Pizza Slice": {"carbs": 35, "protein": 12, "fat": 10, "fiber": 2, "calories": 270},
        "Whole Pizza": {"carbs": 140, "protein": 48, "fat": 40, "fiber": 8, "calories": 1080},
        "Burger Small": {"carbs": 35, "protein": 20, "fat": 15, "fiber": 3, "calories": 340},
        "Burger Large": {"carbs": 50, "protein": 30, "fat": 25, "fiber": 4, "calories": 530}
    }
    
    meal_preset = st.sidebar.selectbox("Meal Preset", list(meal_presets.keys()))
    preset = meal_presets[meal_preset]
    
    carbohydrates = st.sidebar.slider("Carbohydrates (g)", 0, 150, preset["carbs"])
    protein = st.sidebar.slider("Protein (g)", 0, 80, preset["protein"])  
    fat = st.sidebar.slider("Fat (g)", 0, 50, preset["fat"])
    fiber = st.sidebar.slider("Fiber (g)", 0, 20, preset["fiber"])
    calories = st.sidebar.slider("Total Calories", 0, 1000, preset["calories"])
    
    # Activity level (optional)
    with st.sidebar.expander("🏃 Activity Level (Optional)"):
        activity_level = st.selectbox("Activity Level", 
                                    ["Sedentary", "Light", "Moderate", "Active"])
        
        activity_defaults = {
            "Sedentary": {"steps": 50, "steps_mean": 1.5, "steps_max": 10, "active_min": 2, "hr": 70},
            "Light": {"steps": 150, "steps_mean": 5, "steps_max": 25, "active_min": 8, "hr": 75},
            "Moderate": {"steps": 300, "steps_mean": 10, "steps_max": 40, "active_min": 15, "hr": 85},
            "Active": {"steps": 500, "steps_mean": 16, "steps_max": 60, "active_min": 25, "hr": 95}
        }
        
        activity_data = activity_defaults[activity_level]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔬 Predict Glucose Response", type="primary", use_container_width=True):
            # Prepare input features
            input_features = {
                'diabetic_status': diabetic_status,
                'carbohydrates': float(carbohydrates),
                'protein': float(protein),
                'fat': float(fat),
                'fiber': float(fiber),
                'calories': float(calories),
                'age': float(age),
                'gender': float(gender_encoded),
                'bmi': float(bmi),
                'a1c': float(a1c),
                'fasting_glucose': float(fasting_glucose),
                'fasting_insulin': float(fasting_insulin),
                'steps_total': activity_data['steps'],
                'steps_mean_per_minute': activity_data['steps_mean'],
                'steps_max_per_minute': activity_data['steps_max'],
                'active_minutes': activity_data['active_min'],
                'hr_mean': activity_data['hr']
            }
            
            # Make prediction
            with st.spinner("Calculating glucose response..."):
                predictions = pipeline.predict_enhanced(input_features)
            
            # Display results
            st.success("✅ Prediction completed!")
            
            # Show predicted baseline
            st.info(f"**Predicted Baseline Glucose:** {predictions['baseline']:.1f} mg/dL")
            
            # Create and display plots
            st.subheader("📈 Glucose Response Curve")
            curve_fig = create_glucose_curve_plot(predictions, diabetic_status)
            st.plotly_chart(curve_fig, use_container_width=True)
            
            st.subheader("📊 Time Point Comparison")
            bar_fig = create_comparison_plot(predictions)
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            results_data = []
            results_data.append(["Baseline", f"{predictions['baseline']:.1f} mg/dL", "Starting glucose level"])
            
            for time_var in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
                minutes = int(time_var.replace('glucose_', '').replace('min', ''))
                glucose = predictions[time_var]
                
                # Interpretation
                if glucose < 70:
                    interp = "⚠️ Low (Hypoglycemic)"
                elif glucose < 140:
                    interp = "✅ Normal"
                elif glucose < 180:
                    interp = "⚠️ Elevated"
                else:
                    interp = "🔴 High"
                
                results_data.append([f"{minutes} minutes", f"{glucose:.1f} mg/dL", interp])
            
            results_df = pd.DataFrame(results_data, columns=["Time Point", "Glucose Level", "Interpretation"])
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Clinical insights
            st.subheader("🔍 Clinical Insights")
            
            # Peak glucose and time to peak
            glucose_values = [predictions[f'glucose_{t}min'] for t in [30, 60, 90, 120, 180]]
            peak_glucose = max(glucose_values)
            peak_time_idx = glucose_values.index(peak_glucose)
            peak_times = [30, 60, 90, 120, 180]
            peak_time = peak_times[peak_time_idx]
            
            # Glucose excursion
            excursion = peak_glucose - predictions['baseline']
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Peak Glucose", f"{peak_glucose:.1f} mg/dL", f"+{excursion:.1f} from baseline")
            
            with col4:
                st.metric("Time to Peak", f"{peak_time} min")
            
            with col5:
                return_to_baseline = predictions['glucose_180min'] <= predictions['baseline'] + 10
                st.metric("Return to Baseline", "✅ Yes" if return_to_baseline else "❌ No")
            
            # Recommendations based on diabetic status and results
            st.subheader("💡 Recommendations")
            
            recommendations = status_info['recommendations'].copy()
            
            if peak_glucose > 180:
                recommendations.append("🔴 High glucose spike detected - consider medical consultation")
            elif peak_glucose > 140:
                recommendations.append("⚠️ Elevated post-meal glucose - monitor closely")
            
            if excursion > 50:
                recommendations.append("📈 Large glucose excursion - consider meal timing/composition adjustments")
            
            for rec in recommendations:
                st.write(f"• {rec}")
    
    with col2:
        # Model information panel
        st.subheader("ℹ️ Model Information")
        
        st.markdown(f"""
        **Model Version:** {metadata.get('pipeline_version', 'Unknown')}
        
        **Enhancements:**
        """)
        
        for enhancement in metadata.get('enhancements', []):
            st.write(f"• {enhancement}")
        
        st.markdown("**Model Performance:**")
        for target_var, info in metadata.get('model_info', {}).items():
            minutes = target_var.replace('glucose_', '').replace('min', '')
            mae = info.get('mae', 0)
            r2 = info.get('r2_score', 0)
            st.write(f"• {minutes} min: MAE {mae:.1f} mg/dL, R² {r2:.3f}")
        
        # Baseline predictor performance
        baseline_perf = metadata.get('baseline_predictor_info', {})
        if baseline_perf:
            st.markdown("**Baseline Predictor:**")
            st.write(f"• MAE: {baseline_perf.get('mae', 0):.2f} mg/dL")
            st.write(f"• R²: {baseline_perf.get('r2', 0):.3f}")
        
        # Diabetic status distribution in training
        st.subheader("📊 Training Data")
        baseline_stats = baseline_lookup.set_index('diabetic_status')['mean_baseline']
        
        st.markdown("**Baseline Glucose by Status:**")
        for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
            if status in baseline_stats.index:
                avg = baseline_stats[status]
                st.write(f"• {status}: {avg:.1f} mg/dL")

if __name__ == "__main__":
    main()