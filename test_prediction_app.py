#!/usr/bin/env python3
"""
Test Glucose Prediction Spike App Functionality

Simple test to validate the prediction functionality works properly.
"""

import sys
import os
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our prediction class
from glucose_prediction_spike_app import GlucosePredictionSpike

def test_prediction_functionality():
    """Test the core prediction functionality."""
    
    print("🧪 Testing Glucose Prediction Spike App")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = GlucosePredictionSpike()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return False
    
    # Test sample inputs
    test_cases = [
        {
            "name": "Normal Individual - Low Carb Meal",
            "meal_inputs": {
                'carbohydrates': 30,
                'protein': 25,
                'fat': 15,
                'fiber': 8
            },
            "patient_inputs": {
                'diabetic_status': 'Normal',
                'age': 35,
                'bmi': 23.5,
                'a1c': None,
                'fasting_glucose': None
            }
        },
        {
            "name": "Type2Diabetic - High Carb Meal",
            "meal_inputs": {
                'carbohydrates': 80,
                'protein': 20,
                'fat': 10,
                'fiber': 3
            },
            "patient_inputs": {
                'diabetic_status': 'Type2Diabetic',
                'age': 55,
                'bmi': 32.0,
                'a1c': 8.5,
                'fasting_glucose': 160
            }
        },
        {
            "name": "Pre-diabetic - Moderate Meal",
            "meal_inputs": {
                'carbohydrates': 50,
                'protein': 20,
                'fat': 12,
                'fiber': 5
            },
            "patient_inputs": {
                'diabetic_status': 'Pre-diabetic',
                'age': 45,
                'bmi': 28.0,
                'a1c': 6.0,
                'fasting_glucose': None
            }
        }
    ]
    
    print(f"\n🔮 Testing {len(test_cases)} prediction scenarios...\n")
    
    all_tests_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Make prediction
            predictions = predictor.predict_glucose_response(
                test_case['meal_inputs'], 
                test_case['patient_inputs']
            )
            
            # Validate predictions structure
            required_keys = ['baseline', 'glucose_30min', 'glucose_60min', 
                           'glucose_90min', 'glucose_120min', 'glucose_180min']
            
            missing_keys = [key for key in required_keys if key not in predictions]
            if missing_keys:
                print(f"❌ Missing prediction keys: {missing_keys}")
                all_tests_passed = False
                continue
            
            # Check realistic glucose values
            baseline = predictions['baseline']
            if not (50 <= baseline <= 300):
                print(f"❌ Unrealistic baseline glucose: {baseline:.1f} mg/dL")
                all_tests_passed = False
                continue
                
            # Check all glucose predictions are reasonable
            glucose_values = [predictions[key] for key in required_keys[1:]]
            if not all(50 <= val <= 400 for val in glucose_values):
                print(f"❌ Unrealistic glucose predictions: {glucose_values}")
                all_tests_passed = False
                continue
            
            print(f"✅ Baseline: {baseline:.1f} mg/dL")
            print(f"✅ Peak: {max(glucose_values):.1f} mg/dL at {30 + glucose_values.index(max(glucose_values)) * 30} min")
            print(f"✅ Final: {glucose_values[-1]:.1f} mg/dL")
            
            # Test spike curve calculation
            try:
                spike_methods = ['mean', 'upper_ci', 'upper_ci_15']
                spike_curves, time_points = predictor.calculate_spike_curves(
                    predictions, spike_methods, custom_multiplier=1.5
                )
                
                if len(spike_curves) != len(spike_methods):
                    print(f"❌ Expected {len(spike_methods)} spike curves, got {len(spike_curves)}")
                    all_tests_passed = False
                    continue
                
                print(f"✅ Generated {len(spike_curves)} spike curves successfully")
                
                # Validate spike curve structure
                for method, data in spike_curves.items():
                    if not all(key in data for key in ['curve', 'label', 'color']):
                        print(f"❌ Invalid spike curve structure for {method}")
                        all_tests_passed = False
                        continue
                    
                    if len(data['curve']) != len(time_points):
                        print(f"❌ Curve length mismatch for {method}")
                        all_tests_passed = False
                        continue
                
                print(f"✅ All spike curves validated")
                
            except Exception as e:
                print(f"❌ Spike curve calculation failed: {e}")
                all_tests_passed = False
                continue
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            all_tests_passed = False
            continue
        
        print()
    
    # Test model loading status
    print("📊 Model Loading Status:")
    print("-" * 30)
    if predictor.models:
        print(f"✅ {len(predictor.models)} trained models loaded")
        for time_point in predictor.models.keys():
            print(f"  • {time_point} model: Available")
    else:
        print("⚠️  No trained models loaded - using simplified prediction")
    
    print()
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Glucose prediction spike app is ready for deployment")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please check the implementation before deployment")
        return False

def main():
    """Run the tests."""
    success = test_prediction_functionality()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())