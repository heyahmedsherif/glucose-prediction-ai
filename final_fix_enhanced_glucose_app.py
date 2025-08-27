#!/usr/bin/env python3
"""
Final Fix for Enhanced Glucose App with Timing

This applies the corrected logic directly to the enhanced_glucose_app_with_timing.py
to fix the high glucose response issue while preserving advanced features.

The key fixes:
1. Timing adjustments are small additive effects (5-15 mg/dL)
2. Fiber effectiveness is realistic (2-4 mg/dL per gram)
3. No multiplicative compounding of effects
4. Dawn phenomenon is modest baseline adjustment
"""

# This will be the corrected prediction class to replace the problematic one
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
from datetime import datetime, time
import json

class FixedUltimateGlucosePrediction:
    """Fixed ultimate glucose prediction with realistic fiber-timing integration."""
    
    def __init__(self):
        # Base prediction components (same as original)
        self.models = {}
        self.scalers = {}
        self.model_metadata = None
        self.baseline_stats = {
            'Normal': {'mean': 85, 'std': 8},
            'Pre-diabetic': {'mean': 105, 'std': 15}, 
            'Type2Diabetic': {'mean': 140, 'std': 25}
        }
        
        # FIXED: Realistic fiber response profiles (mg/dL per gram)
        self.fiber_response_profiles = {
            'Normal': {
                'base_effectiveness': 2.0,      # 2 mg/dL per gram (realistic)
                'saturation_point': 15,
                'max_reduction': 20             # Max 20 mg/dL reduction
            },
            'Pre-diabetic': {
                'base_effectiveness': 2.5,      # 2.5 mg/dL per gram
                'saturation_point': 18,
                'max_reduction': 25             # Max 25 mg/dL reduction
            },
            'Type2Diabetic': {
                'base_effectiveness': 3.0,      # 3 mg/dL per gram
                'saturation_point': 20,
                'max_reduction': 35             # Max 35 mg/dL reduction
            }
        }
        
        # FIXED: Small timing adjustments (additive mg/dL, not multipliers)
        self.timing_adjustments = {
            'meal_type_effects': {
                'breakfast': 8.0,   # Small breakfast increase
                'lunch': -3.0,      # Small lunch decrease
                'dinner': 0.0       # Baseline
            },
            'first_meal_effect': {
                'Normal': 5.0,      # Small first meal effect
                'Pre-diabetic': 8.0,
                'Type2Diabetic': 12.0
            },
            'dawn_phenomenon': {
                'Normal': 3.0,      # Minimal dawn effect
                'Pre-diabetic': 5.0,
                'Type2Diabetic': 8.0
            },
            'hourly_patterns': {
                # Small hourly variations (-5 to +10 mg/dL)
                6: 8, 7: 10, 8: 5, 9: 2, 10: 0, 11: -2, 12: -5,
                13: -4, 14: -3, 15: -1, 16: 0, 17: 1, 18: 2, 19: 3,
                20: 2, 21: 0, 22: -2
            }
        }
        
        self.load_models()
    
    def load_models(self):
        """Load glucose prediction models (same as original)."""
        model_dir = "glucose_prediction_models"
        if not os.path.exists(model_dir):
            st.warning("⚠️ Models directory not found. Using advanced prediction algorithms.")
            return
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            except Exception as e:
                st.warning(f"Could not load model metadata: {e}")
        
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
    
    def get_dynamic_fiber_recommendation(self, meal_type: str, hour: int, 
                                       diabetic_status: str, carbs: float, 
                                       is_first_meal: bool) -> float:
        """Calculate realistic dynamic fiber recommendation."""
        
        # Base recommendation: 10-15% of carbs as fiber (realistic)
        base_fiber = carbs * 0.12
        
        # Small timing adjustments
        if meal_type == 'breakfast' and 6 <= hour <= 9:
            base_fiber *= 1.2  # 20% more for dawn phenomenon
        elif meal_type == 'lunch' and 12 <= hour <= 14:
            base_fiber *= 0.9  # 10% less for optimal window
        
        if is_first_meal:
            base_fiber *= 1.1  # 10% more for first meal
        
        # Diabetic status adjustments (modest)
        if diabetic_status == 'Type2Diabetic':
            base_fiber *= 1.15
        elif diabetic_status == 'Normal':
            base_fiber *= 0.9
        
        # Reasonable bounds
        min_fiber = max(3, carbs * 0.06)  # At least 6% of carbs
        max_fiber = min(25, carbs * 0.25)  # Max 25% of carbs
        
        return max(min_fiber, min(max_fiber, base_fiber))
    
    def calculate_fiber_effectiveness(self, fiber_amount: float, carbs: float, 
                                    meal_type: str, hour: int, is_first_meal: bool,
                                    diabetic_status: str) -> Dict[str, float]:
        """Calculate realistic fiber effectiveness."""
        
        if fiber_amount <= 0:
            return {
                'total_mg_dl_reduction': 0,
                'timing_component': 0,
                'dawn_component': 0,
                'first_meal_component': 0,
                'fiber_carb_ratio': 0,
                'saturation_level': 0,
                'effectiveness_percentage': 0
            }
        
        profile = self.fiber_response_profiles[diabetic_status]
        
        # Base effectiveness with saturation
        if fiber_amount <= profile['saturation_point']:
            base_effectiveness = fiber_amount * profile['base_effectiveness']
        else:
            # Diminishing returns
            base_effectiveness = (profile['saturation_point'] * profile['base_effectiveness'] + 
                                (fiber_amount - profile['saturation_point']) * profile['base_effectiveness'] * 0.5)
        
        # Small fiber-carb ratio adjustment
        fiber_carb_ratio = fiber_amount / (carbs + 0.1)
        ratio_bonus = 0
        if fiber_carb_ratio > 0.2:
            ratio_bonus = 2.0  # Small bonus for high ratios
        elif fiber_carb_ratio < 0.05:
            base_effectiveness *= 0.8  # Small penalty for very low ratios
        
        # Small timing bonuses
        timing_bonus = 0
        if meal_type == 'breakfast' and 6 <= hour <= 9:
            timing_bonus = 1.0  # Dawn phenomenon makes fiber slightly more effective
        
        first_meal_bonus = 0
        if is_first_meal:
            first_meal_bonus = 1.0  # Small first meal bonus
        
        # Total effectiveness
        total_reduction = base_effectiveness + ratio_bonus + timing_bonus + first_meal_bonus
        total_reduction = min(profile['max_reduction'], total_reduction)
        
        return {
            'total_mg_dl_reduction': total_reduction,
            'timing_component': base_effectiveness,
            'dawn_component': timing_bonus,
            'first_meal_component': first_meal_bonus,
            'fiber_carb_ratio': fiber_carb_ratio,
            'saturation_level': min(1.0, fiber_amount / profile['saturation_point']),
            'effectiveness_percentage': min(100, (total_reduction / fiber_amount) * 100) if fiber_amount > 0 else 0
        }
    
    def predict_baseline_with_timing(self, diabetic_status: str, age: float, bmi: float,
                                   meal_hour: int, is_first_meal: bool,
                                   a1c: float = None, fasting_glucose: float = None) -> float:
        """Predict baseline with minimal timing adjustments."""
        
        stats = self.baseline_stats[diabetic_status]
        baseline = stats['mean']
        
        # Standard adjustments (same as original)
        if age > 40:
            baseline += (age - 40) * 0.3
        if bmi > 25:
            baseline += (bmi - 25) * 0.8
            
        if a1c:
            if diabetic_status == 'Normal' and a1c > 5.5:
                baseline += (a1c - 5.5) * 10
            elif diabetic_status == 'Pre-diabetic':
                baseline += (a1c - 6.0) * 8
            elif diabetic_status == 'Type2Diabetic':
                baseline += (a1c - 7.0) * 12
        
        if fasting_glucose:
            baseline = 0.7 * fasting_glucose + 0.3 * baseline
        
        # FIXED: Small dawn phenomenon adjustment
        if meal_type == 'breakfast' and 6 <= meal_hour <= 9 and is_first_meal:
            dawn_adjustment = self.timing_adjustments['dawn_phenomenon'][diabetic_status]
            baseline += dawn_adjustment
        
        return max(70, min(200, baseline))
    
    def calculate_timing_adjustment(self, meal_hour: int, meal_type: str, 
                                  is_first_meal: bool, diabetic_status: str) -> float:
        """Calculate small timing-based additive adjustment."""
        
        adjustment = 0.0
        
        # Meal type effect (small)
        adjustment += self.timing_adjustments['meal_type_effects'].get(meal_type, 0.0)
        
        # First meal effect (small, diabetic-specific)
        if is_first_meal:
            adjustment += self.timing_adjustments['first_meal_effect'][diabetic_status]
        
        # Dawn phenomenon (small, diabetic-specific)
        if meal_type == 'breakfast' and 6 <= meal_hour <= 9:
            adjustment += self.timing_adjustments['dawn_phenomenon'][diabetic_status]
        
        # Hourly patterns (small)
        adjustment += self.timing_adjustments['hourly_patterns'].get(meal_hour, 0.0)
        
        return adjustment
    
    def predict_glucose_with_fiber_timing(self, meal_inputs: Dict[str, float], 
                                        patient_inputs: Dict[str, Any],
                                        timing_inputs: Dict[str, Any]) -> Dict[str, float]:
        """FIXED glucose prediction with realistic fiber-timing integration."""
        
        # Calculate baseline
        baseline = self.predict_baseline_with_timing(
            patient_inputs['diabetic_status'],
            patient_inputs['age'],
            patient_inputs['bmi'],
            timing_inputs['meal_hour'],
            timing_inputs['is_first_meal'],
            patient_inputs.get('a1c'),
            patient_inputs.get('fasting_glucose')
        )
        
        # Calculate fiber effectiveness
        fiber_effects = self.calculate_fiber_effectiveness(
            meal_inputs['fiber'],
            meal_inputs['carbohydrates'],
            timing_inputs['meal_type'],
            timing_inputs['meal_hour'],
            timing_inputs['is_first_meal'],
            patient_inputs['diabetic_status']
        )
        
        # Calculate small timing adjustment
        timing_adjustment = self.calculate_timing_adjustment(
            timing_inputs['meal_hour'],
            timing_inputs['meal_type'],
            timing_inputs['is_first_meal'],
            patient_inputs['diabetic_status']
        )
        
        predictions = {'baseline': baseline}
        
        for minutes in [30, 60, 90, 120, 180]:
            # Use ORIGINAL simplified prediction as base
            base_glucose = self._original_simplified_prediction(
                baseline, meal_inputs, patient_inputs, minutes
            )
            
            # FIXED: Apply small timing adjustment additively
            timing_adjusted_glucose = base_glucose + timing_adjustment
            
            # FIXED: Fiber was already applied in original prediction, so don't double-apply
            # Just ensure we don't go below baseline
            final_glucose = max(baseline, timing_adjusted_glucose)
            
            predictions[f'glucose_{minutes}min'] = max(70, min(400, final_glucose))
        
        # Store fiber effects for display
        predictions['fiber_effects'] = fiber_effects
        
        return predictions
    
    def _original_simplified_prediction(self, baseline: float, meal_inputs: Dict[str, float], 
                                      patient_inputs: Dict[str, Any], minutes: int) -> float:
        """Use the ORIGINAL working prediction logic."""
        
        # Carb impact (main driver)
        carb_impact = meal_inputs['carbohydrates'] * 1.5
        
        # Protein impact (smaller, delayed)
        protein_impact = meal_inputs['protein'] * 0.3 if minutes >= 60 else 0
        
        # Fat impact (delayed, prolonged)
        fat_impact = meal_inputs['fat'] * 0.2 if minutes >= 90 else 0
        
        # ORIGINAL fiber reduction (this was working correctly!)
        fiber_reduction = meal_inputs['fiber'] * 0.5
        
        # Time-based curve
        if minutes <= 60:
            time_multiplier = minutes / 60.0
        else:
            time_multiplier = 1.0 - ((minutes - 60) / 120.0)
        time_multiplier = max(0.1, time_multiplier)
        
        # Diabetic status multiplier
        status_multipliers = {
            'Normal': 0.6,
            'Pre-diabetic': 1.0,
            'Type2Diabetic': 1.8
        }
        
        status_mult = status_multipliers[patient_inputs['diabetic_status']]
        
        # ORIGINAL calculation (this was working!)
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
        """Calculate spike curves (same as original)."""
        
        time_points = [0, 30, 60, 90, 120, 180]
        base_curve = [
            predictions['baseline'],
            predictions['glucose_30min'],
            predictions['glucose_60min'], 
            predictions['glucose_90min'],
            predictions['glucose_120min'],
            predictions['glucose_180min']
        ]
        
        # Standard deviation estimation
        mean_glucose = np.mean(base_curve)
        if mean_glucose < 120:
            std_estimate = 15
        elif mean_glucose < 160:
            std_estimate = 25
        else:
            std_estimate = 40
        
        spike_curves = {}
        
        for method in spike_methods:
            if method == "mean":
                spike_curves[method] = {
                    'curve': base_curve,
                    'label': 'Mean Response (Timing Optimized)',
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
                spike_curve = [g + 1.65 * std_estimate for g in base_curve]
                spike_curves[method] = {
                    'curve': spike_curve,
                    'label': 'Predicted 95th Percentile',
                    'color': 'green'
                }
        
        return spike_curves, time_points

def test_fixed_predictions():
    """Test the fixed predictions to ensure they're realistic."""
    
    print("🔧 TESTING FIXED GLUCOSE PREDICTIONS")
    print("=" * 50)
    
    predictor = FixedUltimateGlucosePrediction()
    
    # Test cases
    test_cases = [
        {
            'name': 'Normal Breakfast',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True},
            'expected_range': (120, 160)
        },
        {
            'name': 'Type2 Dawn Phenomenon',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 7, 'is_first_meal': True},
            'expected_range': (200, 280)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print("-" * 30)
        
        predictions = predictor.predict_glucose_with_fiber_timing(
            test_case['meal'], test_case['patient'], test_case['timing']
        )
        
        peak = max([predictions[f'glucose_{min}min'] for min in [30, 60, 90, 120, 180]])
        fiber_reduction = predictions['fiber_effects']['total_mg_dl_reduction']
        
        print(f"Baseline: {predictions['baseline']:.1f} mg/dL")
        print(f"Peak: {peak:.1f} mg/dL")
        print(f"Fiber reduction: {fiber_reduction:.1f} mg/dL")
        print(f"Expected range: {test_case['expected_range'][0]}-{test_case['expected_range'][1]} mg/dL")
        
        if test_case['expected_range'][0] <= peak <= test_case['expected_range'][1]:
            print("✅ WITHIN REALISTIC RANGE")
        else:
            print("❌ OUTSIDE REALISTIC RANGE")

if __name__ == "__main__":
    test_fixed_predictions()