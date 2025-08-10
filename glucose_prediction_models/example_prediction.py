#!/usr/bin/env python3
"""
Example: Using Saved Glucose Prediction Models

This script demonstrates how to load and use the trained glucose prediction models
for making real-time predictions.
"""

import json
import numpy as np
import joblib
from typing import Dict, Any, Tuple


def load_and_predict(input_features: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Load models and make predictions."""
    
    # Load metadata
    with open("model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    feature_sets = metadata['feature_sets']
    model_info = metadata['model_info']
    
    predictions = {}
    
    for target_var in feature_sets.keys():
        # Load model and scaler
        model = joblib.load(f"{target_var}_model.joblib")
        scaler = joblib.load(f"{target_var}_scaler.joblib")
        
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
                else:
                    raise ValueError(f"Required feature '{feature}' missing for {target_var} prediction")
        
        # Make prediction
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        predictions[target_var] = float(pred)
    
    return predictions, model_info


def main():
    # Example input: A meal with user characteristics
    example_input = {
        # Meal composition
        'carbohydrates': 50.0,      # grams
        'protein': 25.0,            # grams  
        'fat': 15.0,                # grams
        'fiber': 5.0,               # grams
        'calories': 400.0,          # total calories
        
        # Demographics
        'age': 35.0,                # years
        'gender': 0.0,              # 0=Female, 1=Male
        'bmi': 24.5,                # kg/m²
        'baseline': 95.0,           # mg/dL (current glucose)
        
        # Activity (optional - will use defaults if missing)
        'steps_total': 150,         # steps in 30-min window
        'steps_mean_per_minute': 5.0,   # average steps/min
        'steps_max_per_minute': 25.0,   # peak steps/min
        'active_minutes': 8,            # minutes with activity
        'hr_mean': 78.0             # average heart rate
    }
    
    # Make predictions
    predictions, model_info = load_and_predict(example_input)
    
    # Display results
    print("\n" + "="*50)
    print("GLUCOSE PREDICTION RESULTS")
    print("="*50)
    print(f"Baseline glucose: {example_input['baseline']:.1f} mg/dL")
    print(f"Meal: {example_input['calories']:.0f} cal, {example_input['carbohydrates']:.0f}g carbs")
    print("")
    print("Predicted glucose levels:")
    for time_point, glucose in predictions.items():
        minutes = time_point.replace('glucose_', '').replace('min', '')
        print(f"  {minutes:>3} minutes: {glucose:.1f} mg/dL")
    
    # Show model performance info
    print("\n" + "="*50)
    print("MODEL PERFORMANCE (from training)")
    print("="*50)
    for target_var, info in model_info.items():
        minutes = target_var.replace('glucose_', '').replace('min', '')
        print(f"{minutes:>3} min model: MAE = {info['mae']:.1f} mg/dL, R² = {info['r2_score']:.3f}")


if __name__ == "__main__":
    main()