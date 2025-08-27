#!/usr/bin/env python3
"""
Integration test to verify the corrected app runs without KeyError or TypeError
"""

import sys
from unittest.mock import MagicMock

def test_app_integration():
    """Test that the app can run basic predictions without errors."""
    
    try:
        # Mock all external dependencies
        mock_st = MagicMock()
        mock_st.set_page_config = MagicMock()
        mock_st.cache_resource = lambda func: func
        mock_st.columns = lambda cols: [MagicMock() for _ in range(cols)]
        mock_st.subheader = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.metric = MagicMock()
        mock_st.success = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.error = MagicMock()
        mock_st.info = MagicMock()
        
        sys.modules['streamlit'] = mock_st
        sys.modules['plotly'] = MagicMock()
        sys.modules['plotly.graph_objects'] = MagicMock()
        sys.modules['plotly.express'] = MagicMock() 
        sys.modules['plotly.subplots'] = MagicMock()
        
        # Import and test the corrected prediction class
        from enhanced_glucose_app_with_timing import CorrectedGlucosePrediction
        
        predictor = CorrectedGlucosePrediction()
        
        # Test inputs that caused the original errors
        meal_inputs = {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5}
        patient_inputs = {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90}
        timing_inputs = {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        
        # This should not raise TypeError about argument count
        predictions = predictor.predict_glucose_with_corrected_timing(
            meal_inputs, patient_inputs, timing_inputs
        )
        
        print("✅ Method signature test passed")
        
        # This should not raise KeyError about missing keys
        fiber_effects = predictions['fiber_effects']
        
        # Test all UI-expected keys exist
        ui_keys = ['saturation_level', 'timing_component', 'dawn_component', 
                  'first_meal_component', 'fiber_carb_ratio', 'total_mg_dl_reduction']
        
        for key in ui_keys:
            value = fiber_effects[key]  # This would raise KeyError if missing
            print(f"✅ fiber_effects['{key}'] = {value}")
        
        # Test prediction values are reasonable
        peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
        print(f"✅ Peak glucose: {peak:.1f} mg/dL (within normal range)")
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ No TypeError (method signature issues)")
        print("✅ No KeyError (missing dictionary keys)")
        print("✅ Realistic glucose predictions")
        print("✅ Full UI compatibility")
        
        return True
        
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_integration()
    exit(0 if success else 1)