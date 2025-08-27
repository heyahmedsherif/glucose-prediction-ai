#!/usr/bin/env python3
"""
Comprehensive unit tests for glucose prediction interface
Prevents KeyError and method signature issues
"""

import unittest
import sys
from unittest.mock import MagicMock
from typing import Dict, Any

# Mock streamlit to avoid import issues
mock_st = MagicMock()
mock_st.set_page_config = MagicMock()
mock_st.cache_resource = lambda func: func
sys.modules['streamlit'] = mock_st

# Mock plotly to avoid import issues  
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

from fixed_glucose_prediction_logic import CorrectedGlucosePrediction

class TestGlucosePredictionInterface(unittest.TestCase):
    """Test glucose prediction interface compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = CorrectedGlucosePrediction()
        
        # Standard test inputs
        self.meal_inputs = {
            'carbohydrates': 50, 
            'protein': 20, 
            'fat': 10, 
            'fiber': 5
        }
        
        self.patient_inputs = {
            'diabetic_status': 'Normal', 
            'age': 35, 
            'bmi': 23, 
            'a1c': 5.2, 
            'fasting_glucose': 90
        }
        
        self.timing_inputs = {
            'meal_type': 'breakfast', 
            'meal_hour': 8, 
            'is_first_meal': True
        }
    
    def test_method_signatures(self):
        """Test that all methods have correct signatures."""
        
        # Test main prediction method signature
        try:
            predictions = self.predictor.predict_glucose_with_corrected_timing(
                self.meal_inputs, self.patient_inputs, self.timing_inputs
            )
            self.assertIsInstance(predictions, dict)
        except TypeError as e:
            self.fail(f"Method signature error in predict_glucose_with_corrected_timing: {e}")
        
        # Test baseline method signature
        try:
            baseline = self.predictor.predict_baseline_with_timing(
                self.patient_inputs['diabetic_status'],
                self.patient_inputs['age'],
                self.patient_inputs['bmi'],
                self.timing_inputs['meal_hour'],
                self.timing_inputs['is_first_meal'],
                self.patient_inputs.get('a1c'),
                self.patient_inputs.get('fasting_glucose')
            )
            self.assertIsInstance(baseline, (int, float))
        except TypeError as e:
            self.fail(f"Method signature error in predict_baseline_with_timing: {e}")
            
        # Test fiber effectiveness method signature
        try:
            fiber_effectiveness = self.predictor.calculate_corrected_fiber_effectiveness(
                self.meal_inputs['fiber'],
                self.meal_inputs['carbohydrates'],
                self.timing_inputs['meal_type'],
                self.timing_inputs['meal_hour'],
                self.timing_inputs['is_first_meal'],
                self.patient_inputs['diabetic_status']
            )
            self.assertIsInstance(fiber_effectiveness, (int, float))
        except TypeError as e:
            self.fail(f"Method signature error in calculate_corrected_fiber_effectiveness: {e}")
    
    def test_prediction_output_structure(self):
        """Test that prediction output has all required keys."""
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            self.meal_inputs, self.patient_inputs, self.timing_inputs
        )
        
        # Check required time points
        required_time_keys = ['baseline', 'glucose_30min', 'glucose_60min', 
                             'glucose_90min', 'glucose_120min', 'glucose_180min']
        
        for key in required_time_keys:
            self.assertIn(key, predictions, f"Missing required key: {key}")
            self.assertIsInstance(predictions[key], (int, float), 
                                f"Key {key} should be numeric, got {type(predictions[key])}")
        
        # Check fiber_effects structure
        self.assertIn('fiber_effects', predictions, "Missing fiber_effects key")
        fiber_effects = predictions['fiber_effects']
        
        # Test all keys that UI expects
        expected_fiber_keys = [
            'total_mg_dl_reduction',
            'timing_adjustment', 
            'fiber_carb_ratio',
            'saturation_level',
            'timing_component',
            'dawn_component', 
            'first_meal_component'
        ]
        
        for key in expected_fiber_keys:
            self.assertIn(key, fiber_effects, f"Missing fiber_effects key: {key}")
            self.assertIsInstance(fiber_effects[key], (int, float), 
                                f"fiber_effects[{key}] should be numeric, got {type(fiber_effects[key])}")
    
    def test_realistic_prediction_values(self):
        """Test that predictions are within realistic ranges."""
        
        test_cases = [
            ('Normal', 110, 180),      # Normal range
            ('Pre-diabetic', 140, 220), # Pre-diabetic range 
            ('Type2Diabetic', 180, 350) # Type 2 range
        ]
        
        for diabetic_status, min_peak, max_peak in test_cases:
            with self.subTest(diabetic_status=diabetic_status):
                patient = self.patient_inputs.copy()
                patient['diabetic_status'] = diabetic_status
                
                predictions = self.predictor.predict_glucose_with_corrected_timing(
                    self.meal_inputs, patient, self.timing_inputs
                )
                
                # Check baseline is reasonable
                baseline = predictions['baseline']
                self.assertGreaterEqual(baseline, 70, f"Baseline too low: {baseline}")
                self.assertLessEqual(baseline, 200, f"Baseline too high: {baseline}")
                
                # Check peak is in reasonable range
                peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
                self.assertGreaterEqual(peak, min_peak, f"Peak too low for {diabetic_status}: {peak}")
                self.assertLessEqual(peak, max_peak, f"Peak too high for {diabetic_status}: {peak}")
    
    def test_fiber_effects_ranges(self):
        """Test that fiber effects are within realistic ranges."""
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            self.meal_inputs, self.patient_inputs, self.timing_inputs
        )
        
        fiber_effects = predictions['fiber_effects']
        
        # Test fiber reduction is reasonable (should be positive and capped)
        fiber_reduction = fiber_effects['total_mg_dl_reduction']
        self.assertGreaterEqual(fiber_reduction, 0, "Fiber reduction should be non-negative")
        self.assertLessEqual(fiber_reduction, 50, "Fiber reduction should be capped at reasonable level")
        
        # Test timing adjustment is small additive value
        timing_adjustment = fiber_effects['timing_adjustment']
        self.assertGreaterEqual(timing_adjustment, -30, "Timing adjustment too negative")
        self.assertLessEqual(timing_adjustment, 50, "Timing adjustment too positive")
        
        # Test saturation level is between 0 and 1
        saturation = fiber_effects['saturation_level']
        self.assertGreaterEqual(saturation, 0, "Saturation level should be >= 0")
        self.assertLessEqual(saturation, 1, "Saturation level should be <= 1")
        
        # Test fiber-carb ratio is reasonable
        fiber_carb_ratio = fiber_effects['fiber_carb_ratio']
        self.assertGreaterEqual(fiber_carb_ratio, 0, "Fiber-carb ratio should be non-negative")
        self.assertLessEqual(fiber_carb_ratio, 1, "Fiber-carb ratio should be <= 1")
    
    def test_edge_cases(self):
        """Test edge cases that might cause errors."""
        
        # Test with zero fiber
        zero_fiber_meal = self.meal_inputs.copy()
        zero_fiber_meal['fiber'] = 0
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            zero_fiber_meal, self.patient_inputs, self.timing_inputs
        )
        
        self.assertIn('fiber_effects', predictions)
        self.assertEqual(predictions['fiber_effects']['total_mg_dl_reduction'], 0)
        
        # Test with high fiber
        high_fiber_meal = self.meal_inputs.copy()
        high_fiber_meal['fiber'] = 25
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            high_fiber_meal, self.patient_inputs, self.timing_inputs
        )
        
        self.assertIn('fiber_effects', predictions)
        # Should be capped at reasonable level
        self.assertLessEqual(predictions['fiber_effects']['total_mg_dl_reduction'], 50)
        
        # Test different meal types
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            timing = self.timing_inputs.copy()
            timing['meal_type'] = meal_type
            
            predictions = self.predictor.predict_glucose_with_corrected_timing(
                self.meal_inputs, self.patient_inputs, timing
            )
            
            self.assertIn('fiber_effects', predictions)
            self.assertIn('total_mg_dl_reduction', predictions['fiber_effects'])
    
    def test_consistency_with_original(self):
        """Test that corrected predictions are close to realistic values."""
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            self.meal_inputs, self.patient_inputs, self.timing_inputs
        )
        
        # Peak should be reasonable for normal individual with this meal
        peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
        
        # Should be close to original working predictions (around 135-155 mg/dL)
        expected_range = (120, 170)
        self.assertGreaterEqual(peak, expected_range[0], 
                               f"Peak {peak} below expected range {expected_range}")
        self.assertLessEqual(peak, expected_range[1], 
                           f"Peak {peak} above expected range {expected_range}")

class TestUICompatibility(unittest.TestCase):
    """Test UI compatibility to prevent KeyError issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = CorrectedGlucosePrediction()
        
        self.meal_inputs = {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5}
        self.patient_inputs = {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23}
        self.timing_inputs = {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
    
    def test_fiber_effects_ui_keys(self):
        """Test that fiber_effects has all keys expected by UI."""
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            self.meal_inputs, self.patient_inputs, self.timing_inputs
        )
        
        fiber_effects = predictions['fiber_effects']
        
        # Keys that the UI code tries to access
        ui_expected_keys = [
            'fiber_carb_ratio',      # Line 644
            'saturation_level',      # Line 665
            'total_mg_dl_reduction', # Line 672
            'timing_component',      # Line 673
            'dawn_component',        # Line 674
            'first_meal_component'   # Line 675
        ]
        
        for key in ui_expected_keys:
            self.assertIn(key, fiber_effects, 
                         f"UI expects fiber_effects['{key}'] but it's missing")
    
    def test_prediction_time_keys(self):
        """Test that all time point keys exist."""
        
        predictions = self.predictor.predict_glucose_with_corrected_timing(
            self.meal_inputs, self.patient_inputs, self.timing_inputs
        )
        
        expected_time_keys = ['baseline', 'glucose_30min', 'glucose_60min', 
                             'glucose_90min', 'glucose_120min', 'glucose_180min']
        
        for key in expected_time_keys:
            self.assertIn(key, predictions, f"Missing time key: {key}")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)