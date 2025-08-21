#!/usr/bin/env python3
"""
Glucose Prediction with Spike Visualization App

This app combines meal input prediction with spike visualization methods, allowing users to:
- Input meal parameters (carbs, protein, fat, etc.)
- Input patient characteristics (age, BMI, diabetic status)
- Get glucose predictions using trained models
- Visualize predictions using different spike emphasis methods
- Compare predictions across different diabetic status groups

Run with: streamlit run glucose_prediction_spike_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import joblib
import pickle
from typing import Dict, Any, List
import sys
from scipy.interpolate import interp1d

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set page config
st.set_page_config(
    page_title="Glucose Prediction with Spike Visualization",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GlucosePredictionSpike:
    """Enhanced glucose prediction with spike visualization capabilities."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = None
        self.baseline_stats = {
            'Normal': {'mean': 85, 'std': 8},
            'Pre-diabetic': {'mean': 105, 'std': 15}, 
            'Type2Diabetic': {'mean': 140, 'std': 25}
        }
        self.load_models()
    
    def load_models(self):
        """Load glucose prediction models."""
        
        model_dir = "glucose_prediction_models"
        if not os.path.exists(model_dir):
            st.warning("⚠️ Models directory not found. Using simplified prediction.")
            return
        
        # Load model metadata to understand expected features
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            except Exception as e:
                st.warning(f"Could not load model metadata: {e}")
                self.model_metadata = None
        else:
            self.model_metadata = None
        
        # Try to load individual models
        time_points = ['30min', '60min', '90min', '120min', '180min']
        
        for time_point in time_points:
            model_path = os.path.join(model_dir, f"glucose_{time_point}_model.joblib")
            scaler_path = os.path.join(model_dir, f"glucose_{time_point}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[time_point] = joblib.load(model_path)
                    self.scalers[time_point] = joblib.load(scaler_path)
                except Exception as e:
                    st.warning(f"Could not load model for {time_point}: {e}")
    
    def predict_baseline(self, diabetic_status: str, age: float, bmi: float, 
                        a1c: float = None, fasting_glucose: float = None) -> float:
        """Predict baseline glucose based on patient characteristics."""
        
        # Get baseline stats for diabetic status
        stats = self.baseline_stats[diabetic_status]
        baseline = stats['mean']
        
        # Adjust for age (glucose increases with age)
        if age > 40:
            baseline += (age - 40) * 0.3
        
        # Adjust for BMI (higher BMI = higher glucose)
        if bmi > 25:
            baseline += (bmi - 25) * 0.8
        
        # Use A1c if available
        if a1c:
            if diabetic_status == 'Normal' and a1c > 5.5:
                baseline += (a1c - 5.5) * 10
            elif diabetic_status == 'Pre-diabetic':
                baseline += (a1c - 6.0) * 8
            elif diabetic_status == 'Type2Diabetic':
                baseline += (a1c - 7.0) * 12
        
        # Use fasting glucose if available
        if fasting_glucose:
            baseline = 0.7 * fasting_glucose + 0.3 * baseline
        
        # Add some realistic variation
        noise = np.random.normal(0, stats['std'] * 0.2)
        baseline += noise
        
        return max(70, min(200, baseline))
    
    def encode_diabetic_status(self, status: str) -> int:
        """Encode diabetic status as integer for models."""
        encoding = {
            'Normal': 0,
            'Pre-diabetic': 1, 
            'Type2Diabetic': 2
        }
        return encoding.get(status, 0)
    
    def prepare_feature_vector(self, meal_inputs: Dict[str, float], 
                             patient_inputs: Dict[str, Any], baseline: float, 
                             time_point: str) -> np.ndarray:
        """Prepare feature vector for model prediction with proper feature ordering."""
        
        if not self.model_metadata or time_point not in self.model_metadata['feature_sets']:
            # Fallback to simplified features
            return np.array([
                meal_inputs['carbohydrates'],
                meal_inputs['protein'], 
                meal_inputs['fat'],
                meal_inputs['fiber'],
                patient_inputs['age'],
                patient_inputs['bmi']
            ]).reshape(1, -1)
        
        # Get expected features for this time point
        expected_features = self.model_metadata['feature_sets'][time_point]
        feature_vector = []
        
        for feature in expected_features:
            if feature == 'carbohydrates':
                feature_vector.append(meal_inputs['carbohydrates'])
            elif feature == 'protein':
                feature_vector.append(meal_inputs['protein'])
            elif feature == 'fat':
                feature_vector.append(meal_inputs['fat'])
            elif feature == 'fiber':
                feature_vector.append(meal_inputs['fiber'])
            elif feature == 'calories':
                feature_vector.append(meal_inputs.get('calories', 
                    (meal_inputs['carbohydrates'] * 4) + 
                    (meal_inputs['protein'] * 4) + 
                    (meal_inputs['fat'] * 9)))
            elif feature == 'age':
                feature_vector.append(patient_inputs['age'])
            elif feature == 'gender':
                # Encode gender: Male=0, Female=1
                gender_val = patient_inputs.get('gender', 'Male')
                feature_vector.append(1 if gender_val == 'Female' else 0)  
            elif feature == 'bmi':
                feature_vector.append(patient_inputs['bmi'])
            elif feature == 'diabetic_status_encoded':
                feature_vector.append(self.encode_diabetic_status(patient_inputs['diabetic_status']))
            elif feature == 'a1c':
                feature_vector.append(patient_inputs.get('a1c', 6.0 if patient_inputs['diabetic_status'] == 'Normal' else 7.0))
            elif feature == 'fasting_glucose':
                feature_vector.append(patient_inputs.get('fasting_glucose', 100))
            elif feature == 'fasting_insulin':
                # Estimate based on diabetic status if not provided
                status = patient_inputs['diabetic_status']
                if status == 'Normal':
                    feature_vector.append(8.0)
                elif status == 'Pre-diabetic':
                    feature_vector.append(15.0)
                else:
                    feature_vector.append(25.0)
            elif feature == 'baseline':
                feature_vector.append(baseline)
            elif feature in ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes', 'hr_mean']:
                # Default activity values - could be made interactive
                if feature == 'steps_total':
                    feature_vector.append(8000)  # Average daily steps
                elif feature == 'steps_mean_per_minute':
                    feature_vector.append(5.5)
                elif feature == 'steps_max_per_minute':
                    feature_vector.append(120)
                elif feature == 'active_minutes':
                    feature_vector.append(30)
                elif feature == 'hr_mean':
                    feature_vector.append(75)
            else:
                # Unknown feature, use 0
                feature_vector.append(0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_glucose_response(self, meal_inputs: Dict[str, float], 
                               patient_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Predict glucose response over time."""
        
        # Get baseline
        baseline = self.predict_baseline(
            patient_inputs['diabetic_status'],
            patient_inputs['age'],
            patient_inputs['bmi'],
            patient_inputs.get('a1c'),
            patient_inputs.get('fasting_glucose')
        )
        
        # Time points
        time_points = [0, 30, 60, 90, 120, 180]
        predictions = {'baseline': baseline}
        
        # If models are available, use them
        if self.models:
            # Predict for each time point using proper feature vectors
            for time_str in ['30min', '60min', '90min', '120min', '180min']:
                if time_str in self.models:
                    try:
                        # Prepare proper feature vector for this time point
                        X = self.prepare_feature_vector(meal_inputs, patient_inputs, baseline, time_str)
                        X_scaled = self.scalers[time_str].transform(X)
                        pred = self.models[time_str].predict(X_scaled)[0]
                        predictions[f'glucose_{time_str}'] = float(pred)
                    except Exception as e:
                        # If model prediction fails, fall back to simplified
                        predictions[f'glucose_{time_str}'] = self._simplified_prediction(
                            baseline, meal_inputs, patient_inputs, int(time_str.replace('min', ''))
                        )
                else:
                    predictions[f'glucose_{time_str}'] = self._simplified_prediction(
                        baseline, meal_inputs, patient_inputs, int(time_str.replace('min', ''))
                    )
        else:
            # Use simplified prediction model
            for minutes in [30, 60, 90, 120, 180]:
                predictions[f'glucose_{minutes}min'] = self._simplified_prediction(
                    baseline, meal_inputs, patient_inputs, minutes
                )
        
        return predictions
    
    def _simplified_prediction(self, baseline: float, meal_inputs: Dict[str, float], 
                             patient_inputs: Dict[str, Any], minutes: int) -> float:
        """Simplified glucose prediction model."""
        
        # Carb impact (main driver)
        carb_impact = meal_inputs['carbohydrates'] * 1.5
        
        # Protein impact (smaller, delayed)
        protein_impact = meal_inputs['protein'] * 0.3 if minutes >= 60 else 0
        
        # Fat impact (delayed, prolonged)
        fat_impact = meal_inputs['fat'] * 0.2 if minutes >= 90 else 0
        
        # Fiber reduces impact
        fiber_reduction = meal_inputs['fiber'] * 0.5
        
        # Time-based curve (glucose rise and fall)
        if minutes <= 60:
            time_multiplier = minutes / 60.0  # Rise phase
        else:
            time_multiplier = 1.0 - ((minutes - 60) / 120.0)  # Fall phase
        time_multiplier = max(0.1, time_multiplier)
        
        # Diabetic status multiplier
        status_multipliers = {
            'Normal': 0.6,
            'Pre-diabetic': 1.0,
            'Type2Diabetic': 1.8
        }
        
        status_mult = status_multipliers[patient_inputs['diabetic_status']]
        
        # Calculate glucose
        glucose_increase = (carb_impact + protein_impact + fat_impact - fiber_reduction) * time_multiplier * status_mult
        glucose = baseline + glucose_increase
        
        # Age and BMI adjustments
        if patient_inputs['age'] > 50:
            glucose *= 1.1
        if patient_inputs['bmi'] > 28:
            glucose *= 1.05
        
        return max(70, min(400, glucose))
    
    def calculate_spike_curves(self, predictions: Dict[str, float], 
                             spike_methods: List[str], custom_multiplier: float = 1.5) -> Dict[str, Dict[str, List[float]]]:
        """Calculate different spike emphasis curves from predictions."""
        
        time_points = [0, 30, 60, 90, 120, 180]
        base_curve = [
            predictions['baseline'],
            predictions['glucose_30min'],
            predictions['glucose_60min'], 
            predictions['glucose_90min'],
            predictions['glucose_120min'],
            predictions['glucose_180min']
        ]
        
        # Estimate standard deviation based on diabetic status and glucose levels
        mean_glucose = np.mean(base_curve)
        if mean_glucose < 120:
            std_estimate = 15  # Normal range
        elif mean_glucose < 160:
            std_estimate = 25  # Pre-diabetic range
        else:
            std_estimate = 40  # Diabetic range
        
        spike_curves = {}
        
        for method in spike_methods:
            if method == "mean":
                spike_curves[method] = {
                    'curve': base_curve,
                    'label': 'Mean Response (Predicted)',
                    'color': 'blue'
                }
            elif method == "upper_ci":
                spike_curve = [g + std_estimate for g in base_curve]
                spike_curves[method] = {
                    'curve': spike_curve,
                    'label': 'Predicted Spike (Mean + 1 SD)',
                    'color': 'red'
                }
            elif method == "upper_ci_15":
                spike_curve = [g + 1.5 * std_estimate for g in base_curve]
                spike_curves[method] = {
                    'curve': spike_curve,
                    'label': 'Enhanced Spike (Mean + 1.5 SD)',
                    'color': 'orange'
                }
            elif method == "custom_multiplier":
                spike_curve = [g + custom_multiplier * std_estimate for g in base_curve]
                spike_curves[method] = {
                    'curve': spike_curve,
                    'label': f'Custom Spike (Mean + {custom_multiplier} SD)',
                    'color': 'purple'
                }
            elif method == "95th_percentile":
                spike_curve = [g + 1.65 * std_estimate for g in base_curve]  # Approx 95th percentile
                spike_curves[method] = {
                    'curve': spike_curve,
                    'label': 'Predicted 95th Percentile',
                    'color': 'green'
                }
        
        return spike_curves, time_points

def smooth_glucose_curve(time_points: List[int], glucose_values: List[float], 
                        smoothing_points: int = 50) -> tuple:
    """Smooth glucose curve using spline interpolation for more natural appearance."""
    
    # Convert to numpy arrays
    x = np.array(time_points)
    y = np.array(glucose_values)
    
    # Create spline interpolation
    try:
        # Use cubic spline for smooth curves
        spline = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        # Create more points for smoother curve
        x_smooth = np.linspace(x.min(), x.max(), smoothing_points)
        y_smooth = spline(x_smooth)
        
        # Ensure no unrealistic values (glucose shouldn't go negative or too high)
        y_smooth = np.clip(y_smooth, 50, 500)
        
        return x_smooth.tolist(), y_smooth.tolist()
        
    except Exception:
        # Fall back to original points if smoothing fails
        return time_points, glucose_values

@st.cache_resource
def load_predictor():
    """Load the glucose prediction spike analyzer."""
    return GlucosePredictionSpike()

def create_prediction_visualization(spike_curves: Dict, time_points: List[int], 
                                  meal_info: str, patient_info: str, enable_smoothing: bool = True):
    """Create interactive prediction visualization with spike methods."""
    
    fig = go.Figure()
    
    # Add each spike curve
    for method, data in spike_curves.items():
        if enable_smoothing:
            # Get smooth curve
            x_smooth, y_smooth = smooth_glucose_curve(time_points, data['curve'])
            
            # Add smooth line
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=data['label'],
                line=dict(color=data['color'], width=3, shape='spline', smoothing=0.3),
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x} min<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
            ))
            
            # Add original data points as markers for reference
            fig.add_trace(go.Scatter(
                x=time_points,
                y=data['curve'],
                mode='markers',
                name=data['label'] + ' (Data Points)',
                marker=dict(size=6, color=data['color'], symbol='circle'),
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x} min<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
            ))
        else:
            # Use original sharp lines
            fig.add_trace(go.Scatter(
                x=time_points,
                y=data['curve'],
                mode='lines+markers',
                name=data['label'],
                line=dict(color=data['color'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x} min<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
            ))
    
    # Add reference lines
    fig.add_hline(y=140, line_dash="dash", line_color="orange", opacity=0.7,
                 annotation_text="Pre-diabetes threshold (140 mg/dL)")
    fig.add_hline(y=200, line_dash="dash", line_color="red", opacity=0.7,
                 annotation_text="Diabetes threshold (200 mg/dL)")
    
    # Update layout
    fig.update_layout(
        title=f"🍽️ Predicted Glucose Response<br><sub>{meal_info} | {patient_info}</sub>",
        xaxis_title="Time (minutes)",
        yaxis_title="Blood Glucose (mg/dL)",
        height=600,
        hovermode='x unified',
        showlegend=True
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    return fig

def main():
    """Main Streamlit app."""
    
    st.title("🍽️ Glucose Prediction with Spike Visualization")
    st.markdown("**Predict glucose responses and visualize with different spike emphasis methods**")
    
    # Load predictor
    predictor = load_predictor()
    
    # Sidebar inputs
    st.sidebar.title("🎛️ Prediction Inputs")
    
    # Patient characteristics
    st.sidebar.subheader("👤 Patient Characteristics")
    
    diabetic_status = st.sidebar.selectbox(
        "Diabetic Status",
        ["Normal", "Pre-diabetic", "Type2Diabetic"],
        help="Based on HbA1c levels"
    )
    
    age = st.sidebar.slider("Age", 18, 80, 45, help="Patient age in years")
    bmi = st.sidebar.slider("BMI", 18.0, 45.0, 25.0, step=0.1, help="Body Mass Index")
    
    # Optional advanced inputs
    with st.sidebar.expander("🔬 Advanced Patient Data (Optional)"):
        st.markdown("*Providing these values improves prediction accuracy*")
        
        # Set default A1c based on diabetic status for better UX
        default_a1c = {
            'Normal': 5.2,
            'Pre-diabetic': 6.0,
            'Type2Diabetic': 7.5
        }.get(diabetic_status, 6.0)
        
        a1c = st.number_input(
            "HbA1c (%)", 
            4.0, 12.0, 
            default_a1c, 
            step=0.1,
            help=f"Normal: <5.7%, Pre-diabetic: 5.7-6.4%, Diabetic: ≥6.5%",
            key="a1c_input"
        )
        
        # Set default fasting glucose based on diabetic status
        default_fasting = {
            'Normal': 90,
            'Pre-diabetic': 110,
            'Type2Diabetic': 140
        }.get(diabetic_status, 100)
        
        fasting_glucose = st.number_input(
            "Fasting Glucose (mg/dL)", 
            70, 250, 
            default_fasting,
            help="Normal: <100, Pre-diabetic: 100-125, Diabetic: ≥126",
            key="fasting_glucose_input"
        )
        
        # Add gender selection since models use it
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Used in prediction models for accuracy",
            key="gender_input"
        )
        
        # Show what this section does
        st.info("💡 These values help the trained models provide more accurate predictions specific to your metabolic profile.")
    
    # Meal inputs
    st.sidebar.subheader("🍽️ Meal Composition")
    
    carbohydrates = st.sidebar.slider("Carbohydrates (g)", 0, 150, 50,
                                     help="Total carbs in grams")
    protein = st.sidebar.slider("Protein (g)", 0, 100, 20,
                               help="Total protein in grams")
    fat = st.sidebar.slider("Fat (g)", 0, 50, 10,
                           help="Total fat in grams")
    fiber = st.sidebar.slider("Fiber (g)", 0, 30, 5,
                             help="Total fiber in grams")
    
    # Calculate calories
    calories = (carbohydrates * 4) + (protein * 4) + (fat * 9)
    st.sidebar.metric("Estimated Calories", f"{calories:.0f} kcal")
    
    # Spike visualization options
    st.sidebar.subheader("📈 Spike Visualization Options")
    
    # Smoothing option
    enable_smoothing = st.sidebar.checkbox(
        "Smooth Glucose Curves", 
        value=True,
        help="Makes curves more natural and realistic looking"
    )
    
    spike_methods = []
    method_options = {
        "mean": "Mean Response (Predicted)",
        "upper_ci": "Predicted Spike (Mean + 1 SD)",
        "upper_ci_15": "Enhanced Spike (Mean + 1.5 SD)",
        "custom_multiplier": "Custom Spike Multiplier",
        "95th_percentile": "Predicted 95th Percentile"
    }
    
    for method_key, method_name in method_options.items():
        if st.sidebar.checkbox(method_name, 
                              value=(method_key in ["mean", "upper_ci"]),
                              key=f"spike_{method_key}"):
            spike_methods.append(method_key)
    
    custom_multiplier = 1.5
    if "custom_multiplier" in spike_methods:
        custom_multiplier = st.sidebar.slider("Custom SD Multiplier", 0.5, 3.0, 1.5, step=0.1)
    
    # Main content
    if st.button("🚀 Predict Glucose Response", type="primary"):
        
        # Prepare inputs
        meal_inputs = {
            'carbohydrates': carbohydrates,
            'protein': protein,
            'fat': fat,
            'fiber': fiber,
            'calories': calories
        }
        
        patient_inputs = {
            'diabetic_status': diabetic_status,
            'age': age,
            'bmi': bmi,
            'a1c': a1c,  # Always use the input A1c value
            'fasting_glucose': fasting_glucose,  # Always use the input fasting glucose
            'gender': gender
        }
        
        if not spike_methods:
            st.warning("Please select at least one spike visualization method.")
            return
        
        # Make predictions
        with st.spinner("🔮 Generating glucose predictions..."):
            predictions = predictor.predict_glucose_response(meal_inputs, patient_inputs)
            spike_curves, time_points = predictor.calculate_spike_curves(
                predictions, spike_methods, custom_multiplier
            )
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create meal and patient info strings
            meal_info = f"{carbohydrates}g carbs, {protein}g protein, {fat}g fat, {fiber}g fiber"
            patient_info = f"{diabetic_status}, Age {age}, BMI {bmi:.1f}"
            
            # Create visualization
            fig = create_prediction_visualization(spike_curves, time_points, meal_info, patient_info, enable_smoothing)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Prediction Summary")
            
            # Show patient profile being used
            st.markdown("**Patient Profile:**")
            st.write(f"• {diabetic_status}, {gender}, Age {age}")
            st.write(f"• BMI: {bmi:.1f}, A1c: {a1c:.1f}%")
            st.write(f"• Fasting Glucose: {fasting_glucose:.0f} mg/dL")
            st.write("")
            
            # Baseline
            st.metric("Predicted Baseline", f"{predictions['baseline']:.1f} mg/dL")
            
            # Peak analysis
            for method, data in spike_curves.items():
                peak_glucose = max(data['curve'])
                peak_time = time_points[data['curve'].index(peak_glucose)]
                excursion = peak_glucose - predictions['baseline']
                
                st.markdown(f"**{data['label']}:**")
                st.write(f"• Peak: {peak_glucose:.1f} mg/dL at {peak_time} min")
                st.write(f"• Excursion: +{excursion:.1f} mg/dL")
                st.write("")
        
        # Detailed predictions table
        st.subheader("📈 Detailed Time Course Predictions")
        
        # Create comparison table
        table_data = []
        for method, data in spike_curves.items():
            for i, (time, glucose) in enumerate(zip(time_points, data['curve'])):
                table_data.append({
                    'Method': data['label'],
                    'Time (min)': time,
                    'Glucose (mg/dL)': round(glucose, 1),
                    'Excursion': round(glucose - predictions['baseline'], 1)
                })
        
        df_results = pd.DataFrame(table_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Export option
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"glucose_prediction_{diabetic_status}_{carbohydrates}g_carbs.csv",
            mime="text/csv"
        )
        
        # Clinical insights
        st.subheader("💡 Clinical Insights")
        
        baseline = predictions['baseline']
        mean_peak = max(spike_curves['mean']['curve']) if 'mean' in spike_curves else baseline
        
        if diabetic_status == "Normal":
            if mean_peak > 140:
                st.warning("⚠️ Predicted response higher than normal. Consider meal modification.")
            else:
                st.success("✅ Predicted response within normal range.")
        elif diabetic_status == "Pre-diabetic":
            if mean_peak > 180:
                st.error("🔴 High predicted response. Consider reducing carbohydrates.")
            elif mean_peak > 140:
                st.warning("⚠️ Elevated response typical for pre-diabetic status.")
            else:
                st.success("✅ Well-controlled predicted response.")
        else:  # Type2Diabetic
            if mean_peak > 250:
                st.error("🔴 Very high predicted spike. May need medical attention.")
            elif mean_peak > 200:
                st.warning("⚠️ High response - monitor closely and consider medication timing.")
            else:
                st.success("✅ Relatively controlled response for diabetic individual.")
    
    # Information section
    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
        ### Prediction Methods
        This app combines glucose prediction models with spike visualization techniques:
        
        **🔮 Prediction Models:**
        - Uses trained models on CGMacros dataset when available
        - Falls back to physiologically-based equations
        - Considers meal composition, patient characteristics, and diabetic status
        - Advanced patient data (A1c, fasting glucose, gender) improves accuracy
        
        **📈 Spike Visualization:**
        - **Mean Response**: Traditional predicted average response
        - **Predicted Spike**: Shows likely peak responses (Mean + 1 SD)
        - **Enhanced Spike**: Emphasizes higher responses (Mean + 1.5 SD)
        - **95th Percentile**: Shows response level that 95% stay below
        
        ### Clinical Relevance
        - Peak glucose levels are more predictive of complications than averages
        - Spike visualization helps with meal planning and medication timing
        - Different diabetic statuses show dramatically different response patterns
        """)

if __name__ == "__main__":
    main()