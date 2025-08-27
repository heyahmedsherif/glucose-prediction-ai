#!/usr/bin/env python3
"""
Pre-commit validation script
Runs all tests and validates the app is ready for production
"""

import sys
import os
import subprocess
from unittest.mock import MagicMock

def setup_environment():
    """Set up test environment."""
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Mock external dependencies
    mock_st = MagicMock()
    mock_st.set_page_config = MagicMock()
    mock_st.cache_resource = lambda func: func
    sys.modules['streamlit'] = mock_st
    sys.modules['plotly'] = MagicMock()
    sys.modules['plotly.graph_objects'] = MagicMock()
    sys.modules['plotly.express'] = MagicMock()
    sys.modules['plotly.subplots'] = MagicMock()

def test_critical_errors():
    """Test for critical errors that would break the app."""
    
    print("🔍 TESTING FOR CRITICAL ERRORS")
    print("-" * 50)
    
    from fixed_glucose_prediction_logic import CorrectedGlucosePrediction
    
    predictor = CorrectedGlucosePrediction()
    
    # Test scenarios that caused original errors
    critical_scenarios = [
        {
            'name': 'TypeError Test (method signature)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'name': 'KeyError Test (missing UI keys)',
            'meal': {'carbohydrates': 30, 'protein': 15, 'fat': 5, 'fiber': 8},
            'patient': {'diabetic_status': 'Pre-diabetic', 'age': 45, 'bmi': 28},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        }
    ]
    
    ui_expected_keys = [
        'saturation_level', 'effectiveness_percentage', 'timing_component',
        'dawn_component', 'first_meal_component', 'fiber_carb_ratio', 
        'total_mg_dl_reduction', 'timing_adjustment'
    ]
    
    for scenario in critical_scenarios:
        print(f"Testing: {scenario['name']}")
        
        try:
            # Test method call (this would raise TypeError if signature wrong)
            predictions = predictor.predict_glucose_with_corrected_timing(
                scenario['meal'], scenario['patient'], scenario['timing']
            )
            print("  ✅ No TypeError")
            
            # Test all UI keys (this would raise KeyError if missing)
            fiber_effects = predictions['fiber_effects']
            for key in ui_expected_keys:
                value = fiber_effects[key]  # Would raise KeyError if missing
                assert isinstance(value, (int, float)), f"Key {key} should be numeric"
            print("  ✅ No KeyError - all UI keys present")
            
            # Test realistic ranges
            peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
            assert 70 <= peak <= 500, f"Peak {peak} outside realistic range"
            print(f"  ✅ Realistic peak: {peak:.1f} mg/dL")
            
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR: {e}")
            return False
    
    print("✅ All critical error tests passed!")
    return True

def test_realistic_predictions():
    """Test that predictions are realistic."""
    
    print("\n🎯 TESTING REALISTIC PREDICTIONS")
    print("-" * 50)
    
    from fixed_glucose_prediction_logic import CorrectedGlucosePrediction
    
    predictor = CorrectedGlucosePrediction()
    
    # Test against known working scenarios
    test_cases = [
        {
            'name': 'Normal Individual',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True},
            'expected_range': (120, 180)
        },
        {
            'name': 'Type2 Diabetic',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 7, 'is_first_meal': True},
            'expected_range': (250, 350)
        }
    ]
    
    for case in test_cases:
        predictions = predictor.predict_glucose_with_corrected_timing(
            case['meal'], case['patient'], case['timing']
        )
        
        peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
        min_expected, max_expected = case['expected_range']
        
        if min_expected <= peak <= max_expected:
            print(f"✅ {case['name']}: {peak:.1f} mg/dL (expected {min_expected}-{max_expected})")
        else:
            print(f"❌ {case['name']}: {peak:.1f} mg/dL outside range {min_expected}-{max_expected}")
            return False
    
    print("✅ All realistic prediction tests passed!")
    return True

def run_full_test_suite():
    """Run the full test suite."""
    
    print("\n🧪 RUNNING FULL TEST SUITE")
    print("-" * 50)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run([
            sys.executable, os.path.join(test_dir, 'test_runner.py')
        ], capture_output=True, text=True, cwd=os.path.dirname(test_dir))
        
        if result.returncode == 0:
            print("✅ Full test suite passed!")
            return True
        else:
            print("❌ Test suite failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running test suite: {e}")
        return False

def main():
    """Main validation function."""
    
    print("🚀 PRE-COMMIT VALIDATION")
    print("=" * 60)
    
    setup_environment()
    
    # Run validation steps
    tests = [
        ("Critical Error Prevention", test_critical_errors),
        ("Realistic Predictions", test_realistic_predictions),
        ("Full Test Suite", run_full_test_suite)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n▶️  {test_name}")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ App is ready for commit and deployment")
        print("✅ No TypeError or KeyError issues")
        print("✅ Realistic glucose predictions")
        print("✅ Full UI compatibility confirmed")
        print("\n🚀 Safe to commit!")
        return True
    else:
        print("❌ VALIDATION FAILED!")
        print("🛠️  Please fix issues before committing")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)