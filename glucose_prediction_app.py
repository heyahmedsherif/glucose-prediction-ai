#!/usr/bin/env python3
"""
Glucose Prediction Streamlit App

A web application for predicting blood glucose responses after meals using
the trained CGMacros glucose prediction models.

Run with: streamlit run glucose_prediction_app.py
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

# Set page config
st.set_page_config(
    page_title="Glucose Prediction App",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load the trained models and metadata."""
    
    models_dir = "glucose_prediction_models"
    
    if not os.path.exists(models_dir):
        st.error(f"❌ Models directory '{models_dir}' not found. Please ensure the trained models are available.")
        st.stop()
    
    # Load metadata
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load models and scalers
    models = {}
    scalers = {}
    
    for target_var in metadata['feature_sets'].keys():
        model_path = os.path.join(models_dir, f"{target_var}_model.joblib")
        scaler_path = os.path.join(models_dir, f"{target_var}_scaler.joblib")
        
        models[target_var] = joblib.load(model_path)
        scalers[target_var] = joblib.load(scaler_path)
    
    return models, scalers, metadata

def predict_glucose(input_features: Dict[str, Any], models, scalers, metadata) -> Dict[str, float]:
    """Make glucose predictions using the loaded models."""
    
    feature_sets = metadata['feature_sets']
    predictions = {}
    
    for target_var in feature_sets.keys():
        # Get required features
        required_features = feature_sets[target_var]
        
        # Extract feature values
        feature_values = []
        for feature in required_features:
            if feature in input_features:
                feature_values.append(input_features[feature])
            else:
                # Handle missing features with defaults
                if feature in ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']:
                    feature_values.append(0)  # Default for activity features
                elif feature == 'hr_mean':
                    feature_values.append(75)  # Default resting heart rate
                elif feature in ['a1c', 'fasting_glucose', 'fasting_insulin']:
                    # Use median values for missing biomarkers
                    if feature == 'a1c':
                        feature_values.append(5.7)  # Normal A1C
                    elif feature == 'fasting_glucose':
                        feature_values.append(95)  # Normal fasting glucose
                    else:  # fasting_insulin
                        feature_values.append(8.0)  # Normal fasting insulin
                else:
                    st.error(f"Required feature '{feature}' missing for {target_var} prediction")
                    return {}
        
        # Make prediction
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = scalers[target_var].transform(X)
        pred = models[target_var].predict(X_scaled)[0]
        predictions[target_var] = max(0, float(pred))  # Ensure non-negative glucose
    
    return predictions

def create_glucose_curve_plot(predictions: Dict[str, float], estimated_baseline: float = None):
    """Create an interactive plot of the glucose response curve."""
    
    # Extract time points and glucose values
    time_points = [30, 60, 90, 120, 180]  # minutes (no baseline since we don't predict from current glucose)
    glucose_values = []
    
    # Add predictions in order
    for minutes in [30, 60, 90, 120, 180]:
        target_var = f"glucose_{minutes}min"
        glucose_values.append(predictions.get(target_var, 100))  # Default fallback
    
    # Create the plot
    fig = go.Figure()
    
    # Add glucose curve
    fig.add_trace(go.Scatter(
        x=time_points,
        y=glucose_values,
        mode='lines+markers',
        name='Predicted Glucose',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, color='#ff6b6b'),
        hovertemplate='<b>%{x} minutes</b><br>Glucose: %{y:.1f} mg/dL<extra></extra>'
    ))
    
    # Add normal glucose range
    fig.add_hline(y=70, line_dash="dash", line_color="green", 
                  annotation_text="Normal Low (70 mg/dL)", annotation_position="bottom right")
    fig.add_hline(y=140, line_dash="dash", line_color="orange", 
                  annotation_text="Normal High (140 mg/dL)", annotation_position="top right")
    fig.add_hline(y=180, line_dash="dash", line_color="red", 
                  annotation_text="High Risk (180 mg/dL)", annotation_position="top right")
    
    # Update layout
    fig.update_layout(
        title="Predicted Blood Glucose Response",
        xaxis_title="Time After Meal (minutes)",
        yaxis_title="Blood Glucose (mg/dL)",
        template="plotly_white",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Set y-axis range
    min_glucose = min(glucose_values) - 10
    max_glucose = max(glucose_values) + 20
    fig.update_layout(yaxis_range=[max(50, min_glucose), min(250, max_glucose)])
    
    return fig

def create_summary_metrics(predictions: Dict[str, float]):
    """Create summary metrics for the prediction."""
    
    glucose_values = [predictions[f"glucose_{m}min"] for m in [30, 60, 90, 120, 180]]
    time_points = [30, 60, 90, 120, 180]
    
    peak_glucose = max(glucose_values)
    peak_index = glucose_values.index(peak_glucose)
    peak_time = time_points[peak_index]
    
    # Calculate glucose variability
    glucose_range = max(glucose_values) - min(glucose_values)
    
    # Find when glucose stabilizes (difference between consecutive readings < 10 mg/dL)
    stabilization_time = 180  # Default
    for i in range(1, len(glucose_values)):
        if abs(glucose_values[i] - glucose_values[i-1]) < 10:
            stabilization_time = time_points[i]
            break
    
    return {
        'peak_glucose': peak_glucose,
        'peak_time': peak_time,
        'glucose_range': glucose_range,
        'stabilization_time': stabilization_time,
        'min_glucose': min(glucose_values)
    }

def main():
    """Main Streamlit app."""
    
    # Load models
    models, scalers, metadata = load_models()
    
    # App header
    st.title("🍎 Glucose Prediction App")
    st.markdown("""
    **Predict your blood glucose response after eating a meal**
    
    This app uses machine learning models trained on the CGMacros dataset - continuous glucose monitoring data 
    from 42 individuals with 1,269 meals to predict how your blood glucose will respond to different meals over time.
    
    *Built using the [CGMacros scientific dataset](https://github.com/PSI-TAMU/CGMacros/tree/main) for personalized nutrition research.*
    """)
    
    # Sidebar for inputs
    st.sidebar.header("📝 Input Your Information")
    
    # Personal Information
    st.sidebar.subheader("👤 Personal Information")
    age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=35, step=1)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0)
    height_cm = st.sidebar.number_input("Height (cm)", min_value=140, max_value=220, value=170, step=1)
    weight_kg = st.sidebar.number_input("Weight (kg)", min_value=40, max_value=200, value=70, step=1)
    
    # Calculate BMI
    bmi = weight_kg / (height_cm / 100) ** 2
    st.sidebar.info(f"📊 BMI: {bmi:.1f}")
    
    # A1C level
    st.sidebar.subheader("🩸 A1C Level")
    a1c = st.sidebar.number_input(
        "A1C (%) - 3-month average blood glucose", 
        min_value=4.0, max_value=15.0, value=5.7, step=0.1,
        help="Your A1C level (hemoglobin A1C) - a measure of average blood glucose over the past 3 months"
    )
    st.sidebar.info(f"📊 Estimated avg glucose: {((a1c * 28.7) - 46.7):.0f} mg/dL")
    
    # Meal Information
    st.sidebar.subheader("🍽️ Meal Information")
    
    # Meal presets
    meal_preset = st.sidebar.selectbox(
        "Choose a meal preset (optional)", 
        ["Custom", "Small Breakfast", "Large Breakfast", "Light Lunch", "Heavy Lunch", "Light Dinner", "Heavy Dinner"]
    )
    
    # Define meal presets
    meal_presets = {
        "Small Breakfast": {"carbs": 30, "protein": 15, "fat": 10, "fiber": 3, "calories": 250},
        "Large Breakfast": {"carbs": 60, "protein": 25, "fat": 20, "fiber": 5, "calories": 480},
        "Light Lunch": {"carbs": 40, "protein": 20, "fat": 12, "fiber": 4, "calories": 330},
        "Heavy Lunch": {"carbs": 75, "protein": 30, "fat": 25, "fiber": 8, "calories": 600},
        "Light Dinner": {"carbs": 35, "protein": 25, "fat": 15, "fiber": 5, "calories": 350},
        "Heavy Dinner": {"carbs": 80, "protein": 40, "fat": 30, "fiber": 10, "calories": 720}
    }
    
    # Set default values based on preset
    if meal_preset != "Custom":
        preset = meal_presets[meal_preset]
        default_carbs = preset["carbs"]
        default_protein = preset["protein"]
        default_fat = preset["fat"]
        default_fiber = preset["fiber"]
        default_calories = preset["calories"]
    else:
        default_carbs = 50
        default_protein = 20
        default_fat = 15
        default_fiber = 5
        default_calories = 400
    
    # Meal macronutrients
    carbohydrates = st.sidebar.number_input("Carbohydrates (g)", min_value=0, max_value=200, value=default_carbs, step=1)
    protein = st.sidebar.number_input("Protein (g)", min_value=0, max_value=100, value=default_protein, step=1)
    fat = st.sidebar.number_input("Fat (g)", min_value=0, max_value=100, value=default_fat, step=1)
    fiber = st.sidebar.number_input("Fiber (g)", min_value=0, max_value=50, value=default_fiber, step=1)
    calories = st.sidebar.number_input("Total Calories", min_value=50, max_value=2000, value=default_calories, step=10)
    
    # Activity Information (Optional)
    st.sidebar.subheader("🚶 Activity (Optional)")
    include_activity = st.sidebar.checkbox("Include activity data", value=False)
    
    if include_activity:
        steps_total = st.sidebar.number_input("Steps in 30-min window around meal", min_value=0, max_value=2000, value=200, step=10)
        hr_mean = st.sidebar.number_input("Average heart rate (bpm)", min_value=50, max_value=150, value=75, step=1)
        
        # Calculate derived activity features
        steps_mean_per_minute = steps_total / 30
        steps_max_per_minute = steps_mean_per_minute * 2  # Estimate peak as 2x average
        active_minutes = min(30, steps_total // 5)  # Estimate active minutes
    else:
        steps_total = 0
        steps_mean_per_minute = 0
        steps_max_per_minute = 0
        active_minutes = 0
        hr_mean = 75
    
    # Prediction button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Glucose Response", type="primary")
    
    # Main content area
    if predict_button:
        # Prepare input features
        input_features = {
            'carbohydrates': float(carbohydrates),
            'protein': float(protein),
            'fat': float(fat),
            'fiber': float(fiber),
            'calories': float(calories),
            'age': float(age),
            'gender': 1.0 if gender == "Male" else 0.0,
            'bmi': float(bmi),
            'a1c': float(a1c),
            'steps_total': float(steps_total),
            'steps_mean_per_minute': float(steps_mean_per_minute),
            'steps_max_per_minute': float(steps_max_per_minute),
            'active_minutes': float(active_minutes),
            'hr_mean': float(hr_mean)
        }
        
        # Make predictions
        with st.spinner("🔄 Calculating glucose predictions..."):
            predictions = predict_glucose(input_features, models, scalers, metadata)
        
        if predictions:
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot glucose curve
                fig = create_glucose_curve_plot(predictions)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary metrics
                metrics = create_summary_metrics(predictions)
                
                st.subheader("📊 Summary")
                st.metric("Peak Glucose", f"{metrics['peak_glucose']:.1f} mg/dL")
                st.metric("Time to Peak", f"{metrics['peak_time']} minutes")
                st.metric("Glucose Range", f"{metrics['glucose_range']:.1f} mg/dL", 
                         f"{metrics['min_glucose']:.1f} - {metrics['peak_glucose']:.1f}")
                st.metric("Stabilization Time", f"~{metrics['stabilization_time']} minutes")
                
                # Risk assessment
                if metrics['peak_glucose'] < 140:
                    st.success("✅ Normal glucose response")
                elif metrics['peak_glucose'] < 180:
                    st.warning("⚠️ Elevated glucose response")
                else:
                    st.error("🚨 High glucose response")
            
            # Detailed predictions table
            st.subheader("📋 Detailed Predictions")
            
            pred_data = []
            for minutes in [30, 60, 90, 120, 180]:
                target_var = f"glucose_{minutes}min"
                glucose = predictions.get(target_var, 0)
                pred_data.append({
                    "Time After Meal": f"{minutes} minutes",
                    "Predicted Glucose (mg/dL)": f"{glucose:.1f}",
                    "Risk Level": "Normal" if glucose < 140 else "Elevated" if glucose < 180 else "High"
                })
            
            df = pd.DataFrame(pred_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Model information
            st.subheader("🤖 Model Information")
            model_info = metadata['model_info']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Records", "1,269 meals")
                st.metric("Subjects", "42 individuals")
            
            with col2:
                st.metric("Best 30-min Accuracy", f"{model_info['glucose_30min']['mae']:.1f} mg/dL MAE")
                st.metric("Best 60-min Accuracy", f"{model_info['glucose_60min']['mae']:.1f} mg/dL MAE")
    
    else:
        # Instructions when no prediction is made
        st.info("""
        👈 **Please fill in your information in the sidebar and click "Predict Glucose Response"**
        
        **Required Information:**
        - Personal details (age, gender, height, weight)
        - A1C level (3-month average blood glucose)
        - Meal composition (carbs, protein, fat, fiber, calories)
        
        **Optional:**
        - Activity data (steps, heart rate) for improved accuracy
        """)
        
        # Show example visualization
        st.subheader("📈 Example Glucose Response")
        example_times = [30, 60, 90, 120, 180]
        example_glucose = [120, 135, 125, 110, 100]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=example_times, y=example_glucose,
            mode='lines+markers', name='Example Response',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title="Example: Response to a Mixed Meal",
            xaxis_title="Time After Meal (minutes)",
            yaxis_title="Blood Glucose (mg/dL)",
            template="plotly_white", height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This app is for educational purposes only and should not replace professional medical advice. 
    Always consult with healthcare providers for diabetes management decisions.
    
    **📚 Model:** Trained on CGMacros dataset with RandomForest algorithms achieving 34-42 mg/dL prediction accuracy using A1C-based modeling.
    
    **📊 Data Source:** This application uses the [CGMacros dataset](https://github.com/PSI-TAMU/CGMacros/tree/main) - 
    A scientific dataset for personalized nutrition and diet monitoring developed by the Phenotype Science Initiative (PSI) at Texas A&M University.
    """)

if __name__ == "__main__":
    main()