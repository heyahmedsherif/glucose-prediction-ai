#!/usr/bin/env python3
"""
Diagnosis Script: Compare Glucose Predictions Between Old and New Apps

This script compares glucose predictions between:
1. Original glucose_prediction_spike_app.py (baseline)
2. New enhanced_glucose_app_with_timing.py (with fiber-timing optimization)

To understand if the new app shows higher glucose responses than expected.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

# Import the original prediction logic
class OriginalGlucosePrediction:
    """Original glucose prediction logic from glucose_prediction_spike_app.py"""
    
    def __init__(self):
        self.baseline_stats = {
            'Normal': {'mean': 85, 'std': 8},
            'Pre-diabetic': {'mean': 105, 'std': 15}, 
            'Type2Diabetic': {'mean': 140, 'std': 25}
        }
    
    def predict_baseline(self, diabetic_status: str, age: float, bmi: float, 
                        a1c: float = None, fasting_glucose: float = None) -> float:
        """Original baseline prediction logic."""
        
        stats = self.baseline_stats[diabetic_status]
        baseline = stats['mean']
        
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
        
        # Skip random noise for comparison
        # noise = np.random.normal(0, stats['std'] * 0.2)
        # baseline += noise
        
        return max(70, min(200, baseline))
    
    def _simplified_prediction(self, baseline: float, meal_inputs: Dict[str, float], 
                             patient_inputs: Dict[str, Any], minutes: int) -> float:
        """Original simplified prediction model."""
        
        # Carb impact (main driver)
        carb_impact = meal_inputs['carbohydrates'] * 1.5
        
        # Protein impact (smaller, delayed)
        protein_impact = meal_inputs['protein'] * 0.3 if minutes >= 60 else 0
        
        # Fat impact (delayed, prolonged)
        fat_impact = meal_inputs['fat'] * 0.2 if minutes >= 90 else 0
        
        # Fiber reduces impact
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
        
        # Calculate glucose
        glucose_increase = (carb_impact + protein_impact + fat_impact - fiber_reduction) * time_multiplier * status_mult
        glucose = baseline + glucose_increase
        
        # Age and BMI adjustments
        if patient_inputs['age'] > 50:
            glucose *= 1.1
        if patient_inputs['bmi'] > 28:
            glucose *= 1.05
        
        return max(70, min(400, glucose))
    
    def predict_glucose_response(self, meal_inputs: Dict[str, float], 
                               patient_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Original glucose response prediction."""
        
        baseline = self.predict_baseline(
            patient_inputs['diabetic_status'],
            patient_inputs['age'],
            patient_inputs['bmi'],
            patient_inputs.get('a1c'),
            patient_inputs.get('fasting_glucose')
        )
        
        predictions = {'baseline': baseline}
        
        for minutes in [30, 60, 90, 120, 180]:
            predictions[f'glucose_{minutes}min'] = self._simplified_prediction(
                baseline, meal_inputs, patient_inputs, minutes
            )
        
        return predictions

# New enhanced prediction logic (simplified for comparison)
class NewEnhancedPrediction:
    """New enhanced prediction logic from enhanced_glucose_app_with_timing.py"""
    
    def __init__(self):
        self.baseline_stats = {
            'Normal': {'mean': 85, 'std': 8},
            'Pre-diabetic': {'mean': 105, 'std': 15}, 
            'Type2Diabetic': {'mean': 140, 'std': 25}
        }
        
        # Fiber response profiles
        self.fiber_response_profiles = {
            'Normal': {
                'base_sensitivity': 0.6,
                'saturation_point': 10,
                'timing_sensitivity': 0.8,
                'ratio_importance': 0.5,
                'dawn_fiber_multiplier': 1.2
            },
            'Pre-diabetic': {
                'base_sensitivity': 1.0,
                'saturation_point': 12,
                'timing_sensitivity': 1.0,
                'ratio_importance': 0.8,
                'dawn_fiber_multiplier': 1.4
            },
            'Type2Diabetic': {
                'base_sensitivity': 1.5,
                'saturation_point': 15,
                'timing_sensitivity': 1.3,
                'ratio_importance': 1.2,
                'dawn_fiber_multiplier': 1.6
            }
        }
        
        # Timing adjustments
        self.timing_adjustments = {
            'meal_type_effects': {
                'breakfast': 1.33,
                'lunch': 0.67,
                'dinner': 1.0
            },
            'first_meal_effect': 1.24,
            'dawn_phenomenon': {
                'early_breakfast': 1.22,
                'other_breakfast': 1.0
            },
            'hourly_patterns': {
                6: 1.4, 7: 1.5, 8: 1.3, 9: 1.2, 10: 1.1, 11: 1.0, 12: 0.9,
                13: 0.8, 14: 0.8, 15: 0.9, 16: 0.9, 17: 1.0, 18: 1.0, 19: 1.1,
                20: 1.1, 21: 1.0, 22: 0.9
            }
        }
        
        self.fiber_timing_effectiveness = {
            'breakfast_early': 1.5,
            'breakfast_late': 1.0,
            'lunch_optimal': 1.2,
            'lunch_other': 1.0,
            'dinner': 0.8,
            'first_meal_bonus': 1.3
        }
    
    def _predict_baseline_with_timing(self, diabetic_status: str, age: float, bmi: float,
                                    meal_hour: int, is_first_meal: bool,
                                    a1c: float = None, fasting_glucose: float = None) -> float:
        """Enhanced baseline prediction with timing."""
        
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
        
        # NEW: Dawn phenomenon baseline adjustment
        if 6 <= meal_hour <= 9 and is_first_meal:
            dawn_adjustment = 5 + (10 if diabetic_status == 'Type2Diabetic' else 0)
            baseline += dawn_adjustment
        
        return max(70, min(200, baseline))
    
    def _calculate_timing_adjustment(self, meal_hour: int, meal_type: str, 
                                   is_first_meal: bool, diabetic_status: str) -> float:
        """Calculate timing-based adjustment factor."""
        
        adjustment = 1.0
        
        # Meal type effect
        adjustment *= self.timing_adjustments['meal_type_effects'].get(meal_type, 1.0)
        
        # First meal effect
        if is_first_meal:
            if diabetic_status == 'Normal':
                adjustment *= 1.15
            elif diabetic_status == 'Pre-diabetic':
                adjustment *= 1.25
            else:  # Type2Diabetic
                adjustment *= 1.45
        
        # Dawn phenomenon
        if meal_type == 'breakfast' and 6 <= meal_hour <= 9:
            dawn_multiplier = 1.22
            if diabetic_status == 'Type2Diabetic':
                dawn_multiplier *= 1.4
            elif diabetic_status == 'Pre-diabetic':
                dawn_multiplier *= 1.2
            else:
                dawn_multiplier *= 1.1
            adjustment *= dawn_multiplier
        
        # Hourly patterns
        adjustment *= self.timing_adjustments['hourly_patterns'].get(meal_hour, 1.0)
        
        return adjustment
    
    def calculate_fiber_effectiveness(self, fiber_amount: float, carbs: float, 
                                    meal_type: str, hour: int, is_first_meal: bool,
                                    diabetic_status: str) -> float:
        """Calculate fiber effectiveness for glucose reduction."""
        
        profile = self.fiber_response_profiles[diabetic_status]
        
        # Base effectiveness with saturation
        if fiber_amount <= profile['saturation_point']:
            base_effectiveness = fiber_amount * profile['base_sensitivity']
        else:
            base_effectiveness = (profile['saturation_point'] * profile['base_sensitivity'] + 
                                (fiber_amount - profile['saturation_point']) * profile['base_sensitivity'] * 0.3)
        
        # Fiber-carb ratio bonus/penalty
        fiber_carb_ratio = fiber_amount / (carbs + 0.1)
        
        if fiber_carb_ratio > 0.2:
            ratio_multiplier = 1.0 + (fiber_carb_ratio - 0.2) * profile['ratio_importance']
        elif fiber_carb_ratio < 0.05:
            ratio_multiplier = 0.5 + (fiber_carb_ratio / 0.05) * 0.5
        else:
            ratio_multiplier = 1.0
        
        base_effectiveness *= ratio_multiplier
        
        # Timing effectiveness
        if meal_type == 'breakfast' and 6 <= hour <= 9:
            timing_key = 'breakfast_early'
        elif meal_type == 'lunch' and 12 <= hour <= 14:
            timing_key = 'lunch_optimal'
        else:
            timing_key = meal_type if meal_type in ['breakfast', 'lunch', 'dinner'] else 'dinner'
        
        timing_multiplier = self.fiber_timing_effectiveness.get(timing_key, 1.0)
        timing_effectiveness = base_effectiveness * timing_multiplier * profile['timing_sensitivity']
        
        # Dawn and first meal bonuses
        dawn_effectiveness = 0
        if meal_type == 'breakfast' and 6 <= hour <= 9:
            dawn_effectiveness = fiber_amount * 0.8 * profile['dawn_fiber_multiplier']
        
        first_meal_effectiveness = 0
        if is_first_meal:
            first_meal_effectiveness = fiber_amount * 0.6 * self.fiber_timing_effectiveness['first_meal_bonus']
        
        total_effectiveness = timing_effectiveness + dawn_effectiveness + first_meal_effectiveness
        return min(50, total_effectiveness)
    
    def _simplified_prediction(self, baseline: float, meal_inputs: Dict[str, float], 
                             patient_inputs: Dict[str, Any], minutes: int) -> float:
        """Base glucose prediction model (same as original)."""
        
        # Carb impact
        carb_impact = meal_inputs['carbohydrates'] * 1.5
        
        # Protein impact (delayed)
        protein_impact = meal_inputs['protein'] * 0.3 if minutes >= 60 else 0
        
        # Fat impact (delayed, prolonged)
        fat_impact = meal_inputs['fat'] * 0.2 if minutes >= 90 else 0
        
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
        
        # Calculate glucose (without fiber reduction - handled separately in new version)
        glucose_increase = (carb_impact + protein_impact + fat_impact) * time_multiplier * status_mult
        glucose = baseline + glucose_increase
        
        # Age and BMI adjustments
        if patient_inputs['age'] > 50:
            glucose *= 1.1
        if patient_inputs['bmi'] > 28:
            glucose *= 1.05
        
        return max(70, min(400, glucose))
    
    def predict_glucose_with_fiber_timing(self, meal_inputs: Dict[str, float], 
                                        patient_inputs: Dict[str, Any],
                                        timing_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced glucose prediction with fiber-timing integration."""
        
        # Enhanced baseline with timing
        baseline = self._predict_baseline_with_timing(
            patient_inputs['diabetic_status'],
            patient_inputs['age'],
            patient_inputs['bmi'],
            timing_inputs['meal_hour'],
            timing_inputs['is_first_meal'],
            patient_inputs.get('a1c'),
            patient_inputs.get('fasting_glucose')
        )
        
        # Calculate fiber effectiveness
        fiber_reduction = self.calculate_fiber_effectiveness(
            meal_inputs['fiber'],
            meal_inputs['carbohydrates'],
            timing_inputs['meal_type'],
            timing_inputs['meal_hour'],
            timing_inputs['is_first_meal'],
            patient_inputs['diabetic_status']
        )
        
        # Calculate timing adjustment
        timing_adjustment = self._calculate_timing_adjustment(
            timing_inputs['meal_hour'],
            timing_inputs['meal_type'],
            timing_inputs['is_first_meal'],
            patient_inputs['diabetic_status']
        )
        
        predictions = {'baseline': baseline}
        
        for minutes in [30, 60, 90, 120, 180]:
            # Base prediction (same as original but without fiber reduction)
            base_glucose = self._simplified_prediction(
                baseline, meal_inputs, patient_inputs, minutes
            )
            
            # Apply timing adjustment
            excursion = base_glucose - baseline
            timing_adjusted_excursion = excursion * timing_adjustment
            
            # Apply fiber reduction (time-dependent)
            if minutes <= 60:
                fiber_reduction_factor = 0.7
            elif minutes <= 120:
                fiber_reduction_factor = 1.0
            else:
                fiber_reduction_factor = 0.8
            
            actual_fiber_reduction = fiber_reduction * fiber_reduction_factor
            
            # Final glucose value
            final_excursion = max(0, timing_adjusted_excursion - actual_fiber_reduction)
            final_glucose = baseline + final_excursion
            
            predictions[f'glucose_{minutes}min'] = max(70, min(400, final_glucose))
        
        return predictions

def run_comparison_tests():
    """Run comprehensive comparison tests between old and new prediction models."""
    
    print("🔍 GLUCOSE PREDICTION DIAGNOSTIC COMPARISON")
    print("=" * 60)
    
    # Initialize both predictors
    original = OriginalGlucosePrediction()
    enhanced = NewEnhancedPrediction()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Normal Breakfast (No Timing Effects)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'name': 'Normal Lunch (Optimal Timing)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        },
        {
            'name': 'Type2 Dawn Phenomenon',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 7, 'is_first_meal': True}
        },
        {
            'name': 'High Fiber Test',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 15},
            'patient': {'diabetic_status': 'Pre-diabetic', 'age': 45, 'bmi': 28, 'a1c': 6.2, 'fasting_glucose': 110},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'name': 'Low Fiber Test',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 1},
            'patient': {'diabetic_status': 'Pre-diabetic', 'age': 45, 'bmi': 28, 'a1c': 6.2, 'fasting_glucose': 110},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print("-" * 40)
        
        # Original prediction
        orig_pred = original.predict_glucose_response(scenario['meal'], scenario['patient'])
        
        # Enhanced prediction
        enhanced_pred = enhanced.predict_glucose_with_fiber_timing(
            scenario['meal'], scenario['patient'], scenario['timing']
        )
        
        # Calculate differences
        print("Time Point | Original | Enhanced | Difference | % Change")
        print("-" * 55)
        
        time_points = ['baseline', 'glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
        
        scenario_results = {
            'scenario': scenario['name'],
            'meal': scenario['meal'],
            'patient': scenario['patient'],
            'timing': scenario['timing']
        }
        
        for time_point in time_points:
            orig_val = orig_pred[time_point]
            enhanced_val = enhanced_pred[time_point]
            diff = enhanced_val - orig_val
            pct_change = (diff / orig_val) * 100 if orig_val != 0 else 0
            
            time_label = time_point.replace('glucose_', '').replace('min', ' min')
            print(f"{time_label:>10} | {orig_val:>8.1f} | {enhanced_val:>8.1f} | {diff:>10.1f} | {pct_change:>7.1f}%")
            
            scenario_results[f'orig_{time_point}'] = orig_val
            scenario_results[f'enhanced_{time_point}'] = enhanced_val
            scenario_results[f'diff_{time_point}'] = diff
            scenario_results[f'pct_{time_point}'] = pct_change
        
        # Calculate peak values
        orig_peak = max([orig_pred[tp] for tp in time_points if tp != 'baseline'])
        enhanced_peak = max([enhanced_pred[tp] for tp in time_points if tp != 'baseline'])
        peak_diff = enhanced_peak - orig_peak
        peak_pct = (peak_diff / orig_peak) * 100 if orig_peak != 0 else 0
        
        print(f"\n📊 Peak Analysis:")
        print(f"Original Peak: {orig_peak:.1f} mg/dL")
        print(f"Enhanced Peak: {enhanced_peak:.1f} mg/dL")
        print(f"Peak Difference: {peak_diff:.1f} mg/dL ({peak_pct:.1f}%)")
        
        scenario_results['orig_peak'] = orig_peak
        scenario_results['enhanced_peak'] = enhanced_peak
        scenario_results['peak_diff'] = peak_diff
        scenario_results['peak_pct'] = peak_pct
        
        results.append(scenario_results)
    
    # Summary analysis
    print(f"\n📈 SUMMARY ANALYSIS")
    print("=" * 40)
    
    avg_peak_diff = np.mean([r['peak_diff'] for r in results])
    avg_peak_pct = np.mean([r['peak_pct'] for r in results])
    
    print(f"Average Peak Difference: {avg_peak_diff:.1f} mg/dL")
    print(f"Average Peak % Change: {avg_peak_pct:.1f}%")
    
    # Identify potential issues
    print(f"\n🔍 DIAGNOSTIC FINDINGS:")
    
    high_increase_scenarios = [r for r in results if r['peak_diff'] > 20]
    if high_increase_scenarios:
        print(f"❌ HIGH INCREASE DETECTED:")
        for r in high_increase_scenarios:
            print(f"  • {r['scenario']}: +{r['peak_diff']:.1f} mg/dL ({r['peak_pct']:.1f}%)")
    
    moderate_increase_scenarios = [r for r in results if 5 < r['peak_diff'] <= 20]
    if moderate_increase_scenarios:
        print(f"⚠️  MODERATE INCREASE:")
        for r in moderate_increase_scenarios:
            print(f"  • {r['scenario']}: +{r['peak_diff']:.1f} mg/dL ({r['peak_pct']:.1f}%)")
    
    decrease_scenarios = [r for r in results if r['peak_diff'] < 0]
    if decrease_scenarios:
        print(f"✅ DECREASE (Good):")
        for r in decrease_scenarios:
            print(f"  • {r['scenario']}: {r['peak_diff']:.1f} mg/dL ({r['peak_pct']:.1f}%)")
    
    return results

if __name__ == "__main__":
    results = run_comparison_tests()
    
    # Save results for further analysis
    df = pd.DataFrame(results)
    df.to_csv("glucose_prediction_comparison.csv", index=False)
    print(f"\n💾 Results saved to: glucose_prediction_comparison.csv")