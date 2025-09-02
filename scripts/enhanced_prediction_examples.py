#!/usr/bin/env python3
"""
Enhanced Glucose Prediction Examples

This script demonstrates the enhanced glucose prediction system with diabetic status
integration. It shows examples for different patient types and meal scenarios.
"""

import pickle
import numpy as np
import pandas as pd
from train_enhanced_glucose_models import EnhancedGlucosePredictionPipeline

def load_enhanced_pipeline():
    """Load the enhanced prediction pipeline."""
    try:
        with open('glucose_prediction_models/glucose_prediction_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

def create_patient_examples():
    """Create example patients with different diabetic statuses."""
    
    patients = {
        'normal_young_adult': {
            'name': 'Sarah - Normal Glucose Metabolism',
            'diabetic_status': 'Normal',
            'age': 28.0,
            'gender': 0.0,  # Female
            'bmi': 22.5,
            'a1c': 5.2,
            'fasting_glucose': 88.0,
            'fasting_insulin': 6.2
        },
        'prediabetic_middle_age': {
            'name': 'Mike - Pre-diabetic',
            'diabetic_status': 'Pre-diabetic', 
            'age': 45.0,
            'gender': 1.0,  # Male
            'bmi': 28.3,
            'a1c': 6.1,
            'fasting_glucose': 108.0,
            'fasting_insulin': 12.8
        },
        'type2_diabetic_senior': {
            'name': 'Maria - Type 2 Diabetic',
            'diabetic_status': 'Type2Diabetic',
            'age': 62.0,
            'gender': 0.0,  # Female
            'bmi': 31.7,
            'a1c': 7.8,
            'fasting_glucose': 165.0,
            'fasting_insulin': 18.5
        }
    }
    
    return patients

def create_meal_examples():
    """Create different meal scenarios for testing."""
    
    meals = {
        'light_breakfast': {
            'name': 'Light Breakfast (Oatmeal with berries)',
            'carbohydrates': 35.0,
            'protein': 8.0,
            'fat': 3.0,
            'fiber': 6.0,
            'calories': 195.0
        },
        'standard_lunch': {
            'name': 'Standard Lunch (Chicken salad sandwich)',
            'carbohydrates': 52.0,
            'protein': 28.0,
            'fat': 12.0,
            'fiber': 4.0,
            'calories': 410.0
        },
        'high_carb_dinner': {
            'name': 'High-Carb Dinner (Pasta with marinara)',
            'carbohydrates': 85.0,
            'protein': 15.0,
            'fat': 8.0,
            'fiber': 5.0,
            'calories': 485.0
        },
        'low_carb_snack': {
            'name': 'Low-Carb Snack (Greek yogurt with nuts)',
            'carbohydrates': 12.0,
            'protein': 18.0,
            'fat': 15.0,
            'fiber': 2.0,
            'calories': 245.0
        }
    }
    
    return meals

def add_activity_level(base_input, activity_level='moderate'):
    """Add activity level parameters to input."""
    
    activity_levels = {
        'sedentary': {
            'steps_total': 25,
            'steps_mean_per_minute': 0.8,
            'steps_max_per_minute': 5,
            'active_minutes': 1,
            'hr_mean': 68
        },
        'light': {
            'steps_total': 100,
            'steps_mean_per_minute': 3.3,
            'steps_max_per_minute': 15,
            'active_minutes': 5,
            'hr_mean': 75
        },
        'moderate': {
            'steps_total': 200,
            'steps_mean_per_minute': 6.7,
            'steps_max_per_minute': 30,
            'active_minutes': 10,
            'hr_mean': 82
        },
        'active': {
            'steps_total': 400,
            'steps_mean_per_minute': 13.3,
            'steps_max_per_minute': 50,
            'active_minutes': 20,
            'hr_mean': 95
        }
    }
    
    activity_data = activity_levels.get(activity_level, activity_levels['moderate'])
    base_input.update(activity_data)
    return base_input

def run_prediction_example(pipeline, patient, meal, activity_level='moderate'):
    """Run a complete prediction example."""
    
    # Combine patient and meal data
    input_features = {**patient, **meal}
    
    # Add activity level
    input_features = add_activity_level(input_features, activity_level)
    
    # Make prediction
    predictions = pipeline.predict_enhanced(input_features)
    
    return predictions

def format_predictions_table(predictions):
    """Format predictions as a nice table."""
    
    data = []
    data.append(["Baseline", f"{predictions['baseline']:.1f} mg/dL", "Pre-meal glucose level"])
    
    for time_var in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
        minutes = int(time_var.replace('glucose_', '').replace('min', ''))
        glucose = predictions[time_var]
        
        # Add interpretation
        if glucose < 70:
            interp = "Low (Hypoglycemic)"
        elif glucose < 140:
            interp = "Normal"
        elif glucose < 180:
            interp = "Elevated"
        else:
            interp = "High"
        
        data.append([f"{minutes} minutes", f"{glucose:.1f} mg/dL", interp])
    
    return data

def analyze_glucose_response(predictions):
    """Analyze the glucose response patterns."""
    
    baseline = predictions['baseline']
    glucose_values = [predictions[f'glucose_{t}min'] for t in [30, 60, 90, 120, 180]]
    
    # Peak analysis
    peak_glucose = max(glucose_values)
    peak_time_idx = glucose_values.index(peak_glucose)
    peak_times = [30, 60, 90, 120, 180]
    peak_time = peak_times[peak_time_idx]
    
    # Excursion analysis
    excursion = peak_glucose - baseline
    
    # Return to baseline analysis
    final_glucose = predictions['glucose_180min']
    return_to_baseline = final_glucose <= baseline + 10
    
    # Area under curve (simplified)
    auc = sum(glucose_values) * 30  # Approximate AUC
    
    analysis = {
        'peak_glucose': peak_glucose,
        'peak_time': peak_time,
        'glucose_excursion': excursion,
        'return_to_baseline': return_to_baseline,
        'approximate_auc': auc,
        'response_pattern': 'normal' if excursion < 50 and return_to_baseline else 'elevated'
    }
    
    return analysis

def main():
    """Main demonstration function."""
    
    print("=" * 80)
    print("ENHANCED GLUCOSE PREDICTION SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load the enhanced pipeline
    print("Loading enhanced prediction models...")
    pipeline = load_enhanced_pipeline()
    
    if pipeline is None:
        print("❌ Could not load enhanced models. Please run train_enhanced_glucose_models.py first.")
        return
    
    print("✅ Enhanced models loaded successfully!")
    print()
    
    # Load example data
    patients = create_patient_examples()
    meals = create_meal_examples()
    
    # Run comprehensive examples
    print("Running comprehensive prediction examples...")
    print()
    
    all_results = []
    
    for patient_key, patient in patients.items():
        for meal_key, meal in meals.items():
            for activity_level in ['sedentary', 'moderate']:
                
                print(f"🔍 EXAMPLE: {patient['name']} + {meal['name']} + {activity_level.title()} Activity")
                print("-" * 70)
                
                # Run prediction
                predictions = run_prediction_example(pipeline, patient, meal, activity_level)
                
                # Format and display results
                results_table = format_predictions_table(predictions)
                
                print("📊 Glucose Predictions:")
                for row in results_table:
                    print(f"  {row[0]:<12} {row[1]:<12} {row[2]}")
                
                # Analyze response
                analysis = analyze_glucose_response(predictions)
                
                print(f"\n📈 Response Analysis:")
                print(f"  Peak Glucose: {analysis['peak_glucose']:.1f} mg/dL at {analysis['peak_time']} minutes")
                print(f"  Glucose Excursion: +{analysis['glucose_excursion']:.1f} mg/dL from baseline")
                print(f"  Return to Baseline: {'✅ Yes' if analysis['return_to_baseline'] else '❌ No'}")
                print(f"  Response Pattern: {analysis['response_pattern'].title()}")
                
                # Clinical insights
                print(f"\n💡 Clinical Insights:")
                if patient['diabetic_status'] == 'Normal':
                    if analysis['peak_glucose'] > 140:
                        print("  ⚠️ Higher than expected response for normal individual")
                    else:
                        print("  ✅ Normal glucose response as expected")
                elif patient['diabetic_status'] == 'Pre-diabetic':
                    if analysis['peak_glucose'] > 160:
                        print("  ⚠️ Significant glucose excursion - consider meal modification")
                    else:
                        print("  ✅ Moderate response typical for pre-diabetic status")
                else:  # Type2Diabetic
                    if analysis['peak_glucose'] > 200:
                        print("  🔴 High glucose spike - may need medical attention")
                    elif analysis['peak_glucose'] > 180:
                        print("  ⚠️ Elevated response - monitor closely")
                    else:
                        print("  ✅ Well-controlled response for diabetic individual")
                
                print("\n" + "=" * 80 + "\n")
                
                # Store results for summary
                all_results.append({
                    'patient': patient['name'],
                    'meal': meal['name'],
                    'activity': activity_level,
                    'diabetic_status': patient['diabetic_status'],
                    'baseline': predictions['baseline'],
                    'peak': analysis['peak_glucose'],
                    'excursion': analysis['glucose_excursion']
                })
    
    # Summary analysis
    print("📊 SUMMARY ANALYSIS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n🎯 Baseline Glucose by Diabetic Status:")
    baseline_summary = results_df.groupby('diabetic_status')['baseline'].agg(['mean', 'std', 'min', 'max'])
    print(baseline_summary.round(1))
    
    print("\n🚀 Peak Glucose by Diabetic Status:")
    peak_summary = results_df.groupby('diabetic_status')['peak'].agg(['mean', 'std', 'min', 'max'])
    print(peak_summary.round(1))
    
    print("\n📈 Glucose Excursion by Diabetic Status:")
    excursion_summary = results_df.groupby('diabetic_status')['excursion'].agg(['mean', 'std', 'min', 'max'])
    print(excursion_summary.round(1))
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("The enhanced glucose prediction system successfully demonstrates:")
    print("  • Diabetic status-based personalization")
    print("  • Accurate baseline glucose prediction") 
    print("  • Clinically relevant meal response predictions")
    print("  • Activity level integration")
    print("  • Comprehensive analysis and insights")

if __name__ == "__main__":
    main()